20 baseline records (0 unstable), 5 measured runs each, 190 A/A pairs.

### ttft_ms (baseline median 34.15 ms)

| Rule | False positives (A/A) | Min detectable (simulated, 80%) | Detected at +-0.0% (150 us/step) | Detected at +0.0% (300 us/step) | Detected at +-0.0% (750 us/step) |
| --- | --- | --- | --- | --- | --- |
| `fixed_5pct` | 10.0% | 7.0% | 9.5% | 21.2% | 14.0% |
| `fixed_7pct` | 10.0% | 10.0% | 9.2% | 18.5% | 9.5% |
| `cv_band` | 0.0% | 7.0% | 0.0% | 0.0% | 0.0% |
| `ci_overlap` | 4.2% | 2.0% | 1.2% | 1.5% | 0.5% |
| `bootstrap` | 31.1% | 2.0% | 16.2% | 16.5% | 17.8% |
| `rank_test` | 16.3% | 2.0% | 6.5% | 8.8% | 10.2% |

### itl_ms_p50 (baseline median 14.69 ms)

| Rule | False positives (A/A) | Min detectable (simulated, 80%) | Detected at +-0.1% (150 us/step) | Detected at +0.0% (300 us/step) | Detected at +-0.2% (750 us/step) |
| --- | --- | --- | --- | --- | --- |
| `fixed_5pct` | 0.0% | 7.0% | 0.0% | 0.0% | 0.0% |
| `fixed_7pct` | 0.0% | 10.0% | 0.0% | 0.0% | 0.0% |
| `cv_band` | 0.0% | 7.0% | 0.0% | 0.0% | 0.0% |
| `ci_overlap` | 4.7% | 2.0% | 3.5% | 1.8% | 1.5% |
| `bootstrap` | 34.2% | 1.0% | 14.8% | 15.5% | 10.2% |
| `rank_test` | 14.7% | 1.0% | 7.2% | 6.2% | 5.2% |

### itl_ms_p99 (baseline median 15.69 ms)

| Rule | False positives (A/A) | Min detectable (simulated, 80%) | Detected at +0.4% (150 us/step) | Detected at +0.4% (300 us/step) | Detected at +0.3% (750 us/step) |
| --- | --- | --- | --- | --- | --- |
| `fixed_5pct` | 61.1% | 20.0% | 66.2% | 45.8% | 39.0% |
| `fixed_7pct` | 55.8% | n/a | 63.5% | 44.8% | 37.0% |
| `cv_band` | 0.0% | 7.0% | 15.2% | 5.5% | 15.0% |
| `ci_overlap` | 0.0% | n/a | 0.0% | 1.2% | 3.5% |
| `bootstrap` | 22.1% | n/a | 13.5% | 11.0% | 16.0% |
| `rank_test` | 3.7% | n/a | 5.2% | 3.8% | 5.0% |

### e2e_wall_ms (baseline median 1903.06 ms)

| Rule | False positives (A/A) | Min detectable (simulated, 80%) | Detected at +0.1% (150 us/step) | Detected at +0.1% (300 us/step) | Detected at +0.2% (750 us/step) |
| --- | --- | --- | --- | --- | --- |
| `fixed_5pct` | 0.0% | 7.0% | 0.0% | 0.0% | 0.0% |
| `fixed_7pct` | 0.0% | 10.0% | 0.0% | 0.0% | 0.0% |
| `cv_band` | 0.0% | 7.0% | 0.0% | 0.0% | 0.0% |
| `ci_overlap` | 1.1% | 5.0% | 1.2% | 0.5% | 0.5% |
| `bootstrap` | 25.8% | 3.0% | 20.5% | 11.0% | 20.5% |
| `rank_test` | 8.4% | 5.0% | 3.8% | 4.0% | 5.0% |
results/figures/detection_ttft_ms.png
results/figures/detection_itl_ms_p50.png
results/figures/detection_itl_ms_p99.png
results/figures/detection_e2e_wall_ms.png
