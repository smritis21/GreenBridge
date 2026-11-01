# GreenBridge results

Measured on the held-out local patches.

| Training | Test country | Samples | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| nl_only | NL | 25 | 0.760 | 0.463 |
| nl_only | IN | 3 | 1.000 | 0.333 |
| nl_plus_india | NL | 25 | 0.680 | 0.384 |
| nl_plus_india | IN | 3 | 1.000 | 0.333 |

Coverage warning: nl_only, IN has no test examples for vegetation, water. Treat this as a pipeline check, not a full benchmark.

Coverage warning: nl_plus_india, IN has no test examples for vegetation, water. Treat this as a pipeline check, not a full benchmark.

## Limits

Two city samples do not represent entire countries. Labels, seasons, cloud cover and unequal training sizes can affect results. This does not measure park access or recommend construction sites.
