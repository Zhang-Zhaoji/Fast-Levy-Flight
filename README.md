# Fast-Levy-Flight
using jit compilation, we designed a module which could accelerate Levy Flight on a pre-defined map based on Metropolis Sampling.

the jit compilation works not only on sampling trajectory, but also on creating heatmaps. walk for 500 step on a 100*100 map only consume 10~20 millicesonds, and create line-wise heatmap only cost half millisecond.

Here is one example:

<img width="1923" height="600" alt="fastLevyFilght" src="https://github.com/Zhang-Zhaoji/Fast-Levy-Flight/blob/main/fastLevyFilght.png" />

## Usage

just clone or download this code, and import it in your code like:

```from LevyFlight import FastLevyFlight```

before create flight trajectory, please initialize the module via:

```
lf = FastLevyFlight(beta=1.0, D=0.8, T=1.5)
lf.set_saliency_shape(*sal.shape)
```

the `walk` method generate a l\'evy trajectory with pre-defined step on the map, and `line_density_binomial` or `line_density` generates the line-wise heatmap.

```
    traj = lf.walk(500, sal)                              # 500 points
    heat = lf.line_density_binomial(traj, *sal.shape)     # 100×100 Heatmap.
```

you may cite this repo via:
```
@misc{github_repo,
author = {Zhaoji Zhang},
title = {Fast-Levy-Flight},
year = {2025},
url = {https://github.com/Zhang-Zhaoji/Fast-Levy-Flight},
}
```
