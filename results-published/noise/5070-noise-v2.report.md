20 baseline records (1 unstable), 5 measured runs each, 190 A/A pairs.

### ttft_ms (baseline median 34.17 ms)

| Rule | False positives (A/A) | Min detectable (simulated, 80%) | Detected at +-0.1% (150 us/step) | Detected at +-0.0% (300 us/step) | Detected at +-0.0% (750 us/step) |
| --- | --- | --- | --- | --- | --- |
| `fixed_5pct` | 41.1% | 7.0% | 17.8% | 3.5% | 19.0% |
| `fixed_7pct` | 32.1% | 10.0% | 10.2% | 3.0% | 15.5% |
| `cv_band` | 6.8% | 10.0% | 0.0% | 0.0% | 0.0% |
| `ci_overlap` | 3.2% | n/a | 1.0% | 0.5% | 0.5% |
| `bootstrap` | 33.7% | 2.0% | 9.8% | 10.0% | 11.5% |
| `rank_test` | 16.3% | 3.0% | 3.5% | 3.5% | 4.8% |

### itl_ms_p50 (baseline median 14.72 ms)

| Rule | False positives (A/A) | Min detectable (simulated, 80%) | Detected at +-1.9% (150 us/step) | Detected at +-0.9% (300 us/step) | Detected at +2.3% (750 us/step) |
| --- | --- | --- | --- | --- | --- |
| `fixed_5pct` | 0.0% | 7.0% | 0.0% | 0.0% | 0.0% |
| `fixed_7pct` | 0.0% | 10.0% | 0.0% | 0.0% | 0.0% |
| `cv_band` | 0.0% | 10.0% | 0.0% | 0.0% | 0.0% |
| `ci_overlap` | 20.0% | 2.0% | 0.0% | 0.0% | 85.0% |
| `bootstrap` | 50.5% | 1.0% | 0.0% | 0.0% | 85.0% |
| `rank_test` | 35.8% | 1.0% | 0.0% | 0.0% | 85.0% |

### itl_ms_p99 (baseline median 15.78 ms)

| Rule | False positives (A/A) | Min detectable (simulated, 80%) | Detected at +166.2% (150 us/step) | Detected at +179.4% (300 us/step) | Detected at +239.8% (750 us/step) |
| --- | --- | --- | --- | --- | --- |
| `fixed_5pct` | 82.1% | n/a | 99.8% | 92.8% | 100.0% |
| `fixed_7pct` | 76.3% | n/a | 99.5% | 92.5% | 100.0% |
| `cv_band` | 24.2% | 20.0% | 100.0% | 94.2% | 100.0% |
| `ci_overlap` | 0.0% | n/a | 20.5% | 37.8% | 64.2% |
| `bootstrap` | 12.1% | n/a | 71.0% | 77.0% | 95.0% |
| `rank_test` | 0.0% | n/a | 36.0% | 57.0% | 86.5% |

### e2e_wall_ms (baseline median 1913.32 ms)

| Rule | False positives (A/A) | Min detectable (simulated, 80%) | Detected at +3.2% (150 us/step) | Detected at +4.3% (300 us/step) | Detected at +10.5% (750 us/step) |
| --- | --- | --- | --- | --- | --- |
| `fixed_5pct` | 0.0% | 7.0% | 14.5% | 38.2% | 98.5% |
| `fixed_7pct` | 0.0% | 10.0% | 8.2% | 12.2% | 88.8% |
| `cv_band` | 0.0% | 10.0% | 4.8% | 4.8% | 28.0% |
| `ci_overlap` | 1.1% | 7.0% | 6.2% | 20.5% | 75.8% |
| `bootstrap` | 12.6% | 5.0% | 30.2% | 62.0% | 98.8% |
| `rank_test` | 4.7% | 5.0% | 11.5% | 39.0% | 94.8% |
results/figures-v2/detection_ttft_ms.png
results/figures-v2/detection_itl_ms_p50.png
results/figures-v2/detection_itl_ms_p99.png
results/figures-v2/detection_e2e_wall_ms.png
