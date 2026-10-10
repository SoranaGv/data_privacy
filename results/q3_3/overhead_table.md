| Stage | Q3.1 plain 256d (ms) | plain 64d (ms) | Q3.3 encrypted 64d (ms) | Overhead vs 3.1 | Overhead vs 64d |
|---|---|---|---|---|---|
| Encode question | 30.338 | 30.338 | 30.338 | 1.0x | 1.0x |
| Encrypt query | 0.000 | 0.000 | 3.823 | n/a (new) | n/a (new) |
| Server similarity | 0.237 | 0.205 | 6647.661 | 28,013.4x | 32,465.7x |
| Decrypt scores | 0.000 | 0.000 | 742.386 | n/a (new) | n/a (new) |
| Top-k selection | 0.118 | 0.065 | 0.084 | 0.7x | 1.3x |
| Fetch + decrypt docs | 0.011 | 0.012 | 0.141 | 13.3x | 11.3x |
| Total | 30.704 | 30.621 | 7424.433 | 241.8x | 242.5x |
