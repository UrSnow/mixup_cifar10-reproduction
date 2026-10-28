# Measured reproduction results

Paired seeds: 3. Positive differences favour mixup.

| Method | Mean test error (%) | Sample SD (percentage points) |
|---|---:|---:|
| erm | 7.870 | 0.773 |
| mixup | 11.477 | 5.625 |

| Seed | ERM test error (%) | mixup test error (%) | ERM minus mixup (pp) |
|---:|---:|---:|---:|
| 1 | 7.71 | 17.77 | -10.06 |
| 2 | 8.71 | 9.72 | -1.01 |
| 3 | 7.19 | 6.94 | +0.25 |

Mean paired improvement: -3.607 percentage points (sample SD of paired differences: 5.624).
Mixup wins: 1/3; ties: 0.

These results apply to the frozen reduced-scale protocol. No significance claim is inferred from three seeds.
