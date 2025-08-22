import numpy as np
from numba import njit

@njit(inline='always')
def fast_exp(x, lut, lut_min, lut_step):
    """
    exp(x), x ∈ [lut_min, 0]
    lut      : 预先计算好的 exp 表 (float32)
    lut_min  : 最小值 (一般 -12)
    lut_step : (lut_max - lut_min)/(lut_size-1)
    """
    if x <= lut_min:
        return lut[0]  # lower than lut_min
    if x >= 0.0:
        return 1.0     # exp(0)=1
    idx = int((x - lut_min) / lut_step + 0.5)  
    return lut[idx]

@njit(inline='always')
def _sample_step_njit(saliency, y0, x0, dy_flat, dx_flat, log_mask_flat, beta, T, exp_lut, lut_min, lut_step,ys_buf, xs_buf, log_buf, max_retry=10):

    H, W = saliency.shape
    s_cur = saliency[y0, x0]
    n_total = dy_flat.shape[0]
    count = 0

    for i in range(n_total):
        dy_val = dy_flat[i]
        dx_val = dx_flat[i]
        if dy_val == 0 and dx_val == 0:
            continue
        y1 = y0 + dy_val
        x1 = x0 + dx_val
        if 0 <= y1 < H and 0 <= x1 < W:
            log_prob = beta * (saliency[y1, x1] - s_cur) + log_mask_flat[i]
            ys_buf[count] = y1
            xs_buf[count] = x1
            log_buf[count] = log_prob
            count += 1

    if count == 0:
        return y0, x0

    exp_resovoir = np.empty(count, dtype=np.float32)

    max_log = log_buf[0]
    for i in range(1, count):
        if log_buf[i] > max_log:
            max_log = log_buf[i]

    # softmax
    norm_sum = 0.0
    for i in range(count):
        val = fast_exp(log_buf[i] - max_log, exp_lut, lut_min, lut_step)
        exp_resovoir[i] = val
        norm_sum += val

    inv_norm_sum = 1.0 / norm_sum
    for i in range(count):
        exp_resovoir[i] *= inv_norm_sum

    # 采样
    for _ in range(max_retry):
        r = np.random.rand()
        cumsum = 0.0
        for i in range(count):
            cumsum += exp_resovoir[i]
            if r < cumsum:
                y1, x1 = ys_buf[i], xs_buf[i]
                dS = saliency[y1, x1] - s_cur
                acc = fast_exp(dS / T, exp_lut, lut_min, lut_step) if dS <= 0 else 1.0
                if np.random.rand() < acc:
                    return y1, x1
                break

    return ys_buf[0], xs_buf[0]


@njit(cache=True, fastmath=True)
def unravel_argmax(saliency):
    flat_index = np.argmax(saliency)
    H, W = saliency.shape
    y = flat_index // W
    x = flat_index % W
    return y, x

from numba import njit
import numpy as np

@njit(cache=True, fastmath=True)
def walk_njit(k, saliency, dy_flat, dx_flat, log_mask_flat,
              beta, T, exp_lut, lut_min, lut_step,
              ys_buf, xs_buf, log_buf, max_retry=10):
    H, W = saliency.shape
    traj = np.empty((k, 2), dtype=np.int32)

    flat_index = np.argmax(saliency)
    y0, x0 = flat_index // W, flat_index % W
    traj[0, 0], traj[0, 1] = y0, x0

    for i in range(1, k):
        y0, x0 = traj[i-1]

        # 改造后的 _sample_step_njit 需要支持 buf
        y1, x1 = _sample_step_njit(
            saliency, y0, x0,
            dy_flat, dx_flat, log_mask_flat,
            beta, T, exp_lut, lut_min, lut_step,
            ys_buf, xs_buf, log_buf, max_retry
        )
        traj[i, 0], traj[i, 1] = y1, x1

    return traj





class FastLevyFlight:
    def __init__(self, beta=1.0, D=0.8, T=1.5, R_max=None, seed=None,
                 lut_min=-12.0, lut_max=0.0, lut_size=1024):
        self.beta, self.D, self.T = beta, D, T
        self.R_max = R_max
        self._last_shape = None
        self._base_seed = seed if seed is not None else 42

        # ---- 预计算 exp 查表 ----
        self.lut_min = lut_min
        self.lut_max = lut_max
        self.lut_size = lut_size
        xs = np.linspace(lut_min, lut_max, lut_size, dtype=np.float32)
        self.exp_lut = np.exp(xs).astype(np.float32)
        self.lut_step = (lut_max - lut_min) / (lut_size - 1)
        self.ys_buf = None
        self.xs_buf = None
        self.log_buf = None
        
    def set_saliency_shape(self, H: int, W: int):
        """
        call this method before calling `forward`. automatically set the shape of the saliency map, and allocate buffers.
        """
        R_auto = int(np.sqrt(H**2+W**2)+0.5) // 3
        R = self.R_max if self.R_max is not None else R_auto
        R = min(R, max(H, W))

        cache_key = (H, W, R)
        if self._last_shape == cache_key:
            return

        # create grid
        dy, dx = np.mgrid[-R:R+1, -R:R+1]
        self.dy = dy.astype(np.int16)      # 2D
        self.dx = dx.astype(np.int16)      # 2D

        r = np.sqrt(dy**2 + dx**2) + 1e-10
        mask = self.D / (r + self.D**2)
        mask[R, R] = 0.0
        self.log_mask = np.log(mask + 1e-20)  # 2D

        self.dy_flat = dy.ravel().astype(np.int16)
        self.dx_flat = dx.ravel().astype(np.int16)
        self.log_mask_flat = self.log_mask.ravel().astype(np.float32)

        self._last_shape = cache_key

        self.ys_buf = np.empty(H*W, dtype=np.int32)
        self.xs_buf = np.empty(H*W, dtype=np.int32)
        self.log_buf = np.empty(H*W, dtype=np.float32)

        print(f"[Info] set_saliency_shape: H={H}, W={W}, R={R}, flat shapes: {self.dy_flat.shape}")

    def sample_step(self, saliency: np.ndarray, pos: tuple[int, int]):
        y0, x0 = pos
        y1, x1 = _sample_step_njit(
            saliency, y0, x0,
            self.dy_flat, self.dx_flat, self.log_mask_flat,  
            self.beta, self.T, max_retry=10
        )
        return int(y1), int(x1)

    def walk(self, k, saliency):
        """
        Levy Flight on a saliency map
        """
        _max_saliency = np.max(saliency)
        saliency = saliency / (1e-3+_max_saliency) * 4
        return walk_njit(
            k, saliency,
            self.dy_flat, self.dx_flat, self.log_mask_flat,
            self.beta, self.T, self.exp_lut, self.lut_min, self.lut_step,
            self.ys_buf, self.xs_buf, self.log_buf, max_retry=10
        )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def line_density(traj, H=100, W=100):
        """return a line-wise heatmap, use bresenham-like line algorithm"""
        img = np.zeros((H, W), dtype=np.float32)
        for k in range(len(traj) - 1):
            y0, x0 = traj[k]
            y1, x1 = traj[k + 1]
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            steps = max(dx, dy) or 1
            for i in range(steps + 1):
                t = i / steps
                x = int(x0 + t * (x1 - x0) + 0.5)
                y = int(y0 + t * (y1 - y0) + 0.5)
                if 0 <= x < W and 0 <= y < H:
                    img[y, x] += 1.0
        return img
    
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def line_density_binomial(traj, H=100, W=100, n=1):
        """return a line-wise heatmap, use bresenham-like line algorithm with binomial noise of max_derivation n"""
        img = np.zeros((H, W), np.float32)
        for k in range(len(traj) - 1):
            y0, x0 = traj[k]
            y1, x1 = traj[k + 1]
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            steps = max(dx, dy) or 1
            for i in range(steps + 1):
                t = i / steps
                x_exact = x0 + t * (x1 - x0)
                y_exact = y0 + t * (y1 - y0)
                # 二项扰动
                dx_bin = np.random.binomial(2*n, 0.5) - n
                dy_bin = np.random.binomial(2*n, 0.5) - n
                x = int(x_exact + dx_bin + 0.5)
                y = int(y_exact + dy_bin + 0.5)
                if 0 <= x < W and 0 <= y < H:
                    img[y, x] += 1.0
        return img


# ------------------ demo ------------------
def make_test_saliency(shape=(100, 100)):
    y, x = np.ogrid[0:shape[0], 0:shape[1]]
    centers = np.array([[25, 30], [70, 60], [50, 80], [20, 75]])
    sigma = 8.0
    sal = np.sum([np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
                  for cy, cx in centers], axis=0)
    return sal / sal.max() * 3


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    sal = make_test_saliency()
    lf = FastLevyFlight(beta=1.0, D=0.8, T=1.5)
    lf.set_saliency_shape(*sal.shape)
    traj = lf.walk(500, sal)                     # 500 points
    traj = lf.walk(500, sal)
    heat = lf.line_density_binomial(traj, *sal.shape)     # 100×100 Heatmap.
    heat = lf.line_density_binomial(traj, *sal.shape)

    
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 10)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[0, 3:6])
    ax3 = fig.add_subplot(gs[0, 6:10])
    axs = [ax1, ax2, ax3]
    axs[0].imshow(sal)
    axs[0].set_title('Saliency Map')
    axs[1].imshow(sal)
    axs[1].plot(traj[:, 1], traj[:, 0], c = 'red', label='Trajectory',linewidth=0.5)
    axs[1].scatter(traj[0, 1], traj[0, 0], c='purple', label='Start point')
    axs[1].scatter(traj[-1, 1], traj[-1, 0], c='green', label='End point')
    axs[1].set_title('Levy Flight Trajectory')
    axs[1].legend()
    # axs[2].imshow(heat, cmap='hot')
    axs[2].set_title("Line Density Heatmap (500-step Levy Flight)")
    fig.colorbar(axs[2].imshow(heat, cmap='hot'), ax=axs[2])
    plt.tight_layout()
    plt.show()