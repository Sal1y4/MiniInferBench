**bs4-generate** vs **bs4-loop**: slower, most likely cause `launch_overhead`

| | bs4-loop | bs4-generate |
| --- | --- | --- |
| Time per token (ms, median) | 15.06 | 15.68 (+4.1%) |
| Timed runs | 20 | 20 |
| Kernels per decode step | 1335 | 1366 |
| GPU busy per step (us) | 4095 | 4147 |
| Blocking host calls per step | 0.02 | 7.12 |

Output tokens identical: yes

Evidence: busy time per kernel 3.07 -> 3.04 us; busy time growth along the sequence 1.01 -> 1.01; allocator calls per step 0.00 -> 0.00; kernels per step 1335 -> 1366; each added kernel is busy 1.68 us.

