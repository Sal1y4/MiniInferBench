**bs1-generate** vs **bs1-loop**: slower, most likely cause `launch_overhead`

| | bs1-loop | bs1-generate |
| --- | --- | --- |
| Time per token (ms, median) | 14.38 | 15.57 (+8.3%) |
| Timed runs | 20 | 20 |
| Kernels per decode step | 1303 | 1331 |
| GPU busy per step (us) | 3640 | 3674 |
| Blocking host calls per step | 0.02 | 10.09 |

Output tokens identical: yes

Evidence: busy time per kernel 2.79 -> 2.76 us; busy time growth along the sequence 1.00 -> 1.00; allocator calls per step 0.00 -> 0.00; kernels per step 1303 -> 1331; each added kernel is busy 1.22 us.

