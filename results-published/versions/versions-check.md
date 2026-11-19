**generate@4.51.3** vs **generate@4.46.3**: slower, most likely cause `unknown`

| | generate@4.46.3 | generate@4.51.3 |
| --- | --- | --- |
| Time per token (ms, median) | 16.31 | 17.16 (+5.3%) |
| Timed runs | 20 | 20 |
| Kernels per decode step | 1435 | 1435 |
| GPU busy per step (us) | 5172 | 5177 |
| Blocking host calls per step | 13.08 | 12.09 |

Output tokens identical: yes

Evidence: busy time per kernel 3.60 -> 3.61 us; busy time growth along the sequence 1.02 -> 1.02; allocator calls per step 0.00 -> 0.00; kernels per step 1435 -> 1435; blocking host calls per step 13.08 -> 12.09; CPU side of a step 24331 -> 21535 us.

**generate@4.57.6** vs **generate@4.51.3**: faster

| | generate@4.51.3 | generate@4.57.6 |
| --- | --- | --- |
| Time per token (ms, median) | 17.16 | 16.27 (-5.2%) |
| Timed runs | 20 | 20 |
| Kernels per decode step | 1435 | 1371 |
| GPU busy per step (us) | 5177 | 4157 |
| Blocking host calls per step | 12.09 | 12.09 |

Output tokens identical: yes

**generate@5.2.0** vs **generate@4.57.6**: no change

| | generate@4.57.6 | generate@5.2.0 |
| --- | --- | --- |
| Time per token (ms, median) | 16.27 | 16.16 (-0.4%) |
| Timed runs | 20 | 20 |
| Kernels per decode step | 1371 | 1367 |
| GPU busy per step (us) | 4157 | 4160 |
| Blocking host calls per step | 12.09 | 13.09 |

Output tokens identical: yes

**generate@5.4.0** vs **generate@5.2.0**: no change

| | generate@5.2.0 | generate@5.4.0 |
| --- | --- | --- |
| Time per token (ms, median) | 16.16 | 15.88 (-1.7%) |
| Timed runs | 20 | 20 |
| Kernels per decode step | 1367 | 1369 |
| GPU busy per step (us) | 4160 | 4154 |
| Blocking host calls per step | 13.09 | 9.09 |

Output tokens identical: yes

**generate@5.12.1** vs **generate@5.4.0**: no change

| | generate@5.4.0 | generate@5.12.1 |
| --- | --- | --- |
| Time per token (ms, median) | 15.88 | 15.93 (+0.3%) |
| Timed runs | 20 | 20 |
| Kernels per decode step | 1369 | 1370 |
| GPU busy per step (us) | 4154 | 4152 |
| Blocking host calls per step | 9.09 | 9.09 |

Output tokens identical: yes

**generate@5.22.1** vs **generate@5.12.1**: no change

| | generate@5.12.1 | generate@5.22.1 |
| --- | --- | --- |
| Time per token (ms, median) | 15.93 | 15.87 (-0.4%) |
| Timed runs | 20 | 20 |
| Kernels per decode step | 1370 | 1366 |
| GPU busy per step (us) | 4152 | 4141 |
| Blocking host calls per step | 9.09 | 7.12 |

Output tokens identical: yes

**generate@5.23.0** vs **generate@5.22.1**: no change

| | generate@5.22.1 | generate@5.23.0 |
| --- | --- | --- |
| Time per token (ms, median) | 15.87 | 15.36 (-3.2%) |
| Timed runs | 20 | 20 |
| Kernels per decode step | 1366 | 1366 |
| GPU busy per step (us) | 4141 | 4146 |
| Blocking host calls per step | 7.12 | 7.12 |

Output tokens identical: yes

**generate@5.23.0** vs **generate@4.46.3**: faster

| | generate@4.46.3 | generate@5.23.0 |
| --- | --- | --- |
| Time per token (ms, median) | 16.31 | 15.36 (-5.8%) |
| Timed runs | 20 | 20 |
| Kernels per decode step | 1435 | 1366 |
| GPU busy per step (us) | 5172 | 4146 |
| Blocking host calls per step | 13.08 | 7.12 |

Output tokens identical: yes

