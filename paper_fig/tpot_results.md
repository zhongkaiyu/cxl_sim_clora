# Decode TPOT (ms/token), H100 — lower is better

Batch 32 (synthetic) / emergent 64 (LMSYS). **CLoRA is lowest in every cell.** CLoRA Qwen3-30B is the real ~2 ms (figure draws it at the Llama3-8B height for visibility). S-LoRA: measured on Llama2-7B/13B; scaled by model size for Llama3-8B (×16/14) and Qwen3-30B (×6/14).

## Llama2-7B (MHA)
| Workload | CLoRA | CLoRA-NoCXL | Grace-Hopper | CPU-LoRA-Offload | S-LoRA |
|---|---|---|---|---|---|
| Uniform | 4.2 | 46.4 | 36.1 | 171.1 | 54.6 |
| Uniform-long | 9.3 | 51.6 | 132.4 | 604.9 | 54.7 |
| Skewed | 4.2 | 41.1 | 26.9 | 127.6 | 53.4 |
| Skewed-long | 9.0 | 45.7 | 121.9 | 554.6 | 50.9 |
| LMSYS | 4.2 | 36.5 | 15.0 | 53.2 | 16.2 |

## Llama2-13B (MHA)
| Workload | CLoRA | CLoRA-NoCXL | Grace-Hopper | CPU-LoRA-Offload | S-LoRA |
|---|---|---|---|---|---|
| Uniform | 7.8 | 60.5 | 58.0 | 272.8 | 56.4 |
| Uniform-long | 15.4 | 68.3 | 208.3 | 949.6 | 55.6 |
| Skewed | 7.8 | 54.0 | 40.8 | 191.0 | 54.9 |
| Skewed-long | 14.7 | 60.6 | 189.6 | 857.9 | 52.5 |
| LMSYS | 7.8 | 48.2 | 24.6 | 84.0 | 26.8 |

## Llama3-8B (GQA)
| Workload | CLoRA | CLoRA-NoCXL | Grace-Hopper | CPU-LoRA-Offload | S-LoRA |
|---|---|---|---|---|---|
| Uniform | 4.8 | 47.1 | 19.9 | 98.7 | 62.4 |
| Uniform-long | 5.6 | 47.1 | 44.5 | 209.3 | 62.5 |
| Skewed | 4.6 | 41.8 | 12.1 | 60.5 | 61.0 |
| Skewed-long | 5.5 | 41.4 | 35.3 | 163.6 | 58.2 |
| LMSYS | 4.8 | 37.1 | 7.6 | 17.9 | 18.5 |

## Qwen3-30B (GQA+MoE)
| Workload | CLoRA | CLoRA-NoCXL | Grace-Hopper | CPU-LoRA-Offload | S-LoRA |
|---|---|---|---|---|---|
| Uniform | 2.2 | 66.0 | 73.5 | 382.3 | 23.4 |
| Uniform-long | 2.9 | 66.4 | 82.8 | 423.8 | 23.4 |
| Skewed | 2.0 | 56.8 | 36.5 | 199.4 | 22.9 |
| Skewed-long | 2.1 | 57.1 | 41.0 | 214.9 | 21.8 |
| LMSYS | 1.8 | 50.4 | 19.2 | 24.1 | 6.9 |
