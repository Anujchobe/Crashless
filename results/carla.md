# Results: CARLA (Town02, CARLA 0.9.16, `-nullrhi`)

100 episodes per trained policy, deterministic actions, eval seeds 1,000,000+, same reward terms and observation contract as the 2D sim (progress and lateral offset are measured differently; see README). Mean ± std across training seeds.

**Not a held-out test.** Eval seeds change the random draws, but routes still start from Town02's spawn points on the map the CARLA agents trained on; only obstacle placement is new. Rows are comparable (same episodes for every policy), but these are not unseen roads.

| Policy | CARLA training | Seeds | Success | Collision | Off-road | Timeout | Speed m/s | Abs steer change |
|---|---|---|---|---|---|---|---|---|
| Lane-follow PD (no obstacle sensing) | none | n/a | 33% | 56% | 11% | 0% | 6.1 | 0.021 |
| SAC trained in 2D only (zero-shot) | none | 3 | 48% ± 16% | 20% ± 6% | 18% ± 12% | 14% ± 13% | 8.3 ± 2.5 | 0.091 ± 0.026 |
| SAC from scratch in CARLA | 30k steps | 3 | 45% ± 31% | 30% ± 5% | 9% ± 5% | 16% ± 23% | 8.8 ± 4.6 | 0.091 ± 0.025 |
| **SAC 2D-pretrained + fine-tuned** | 30k steps | 3 | 75% ± 14% | 21% ± 8% | 4% ± 5% | 1% ± 1% | 11.5 ± 1.4 | 0.081 ± 0.009 |

Success per seed: sac_2d: 69%, 31%, 44%; sac_scratch: 68%, 1%, 65%; sac_finetune: 83%, 55%, 87%

Fine-tuned beat scratch on 3/3 seeds: one-sided sign test p = 0.125 (not significant at 0.05).

Training-time success (stochastic actions, last 50 episodes), mean across seeds:

| CARLA steps | 5k | 10k | 15k | 20k | 25k | 30k |
|---|---|---|---|---|---|---|
| Fine-tuned (3 seeds) | 23% | 32% | 51% | 59% | 65% | 73% |
| From scratch (3 seeds) | 0% | 2% | 15% | 31% | 45% | 47% |
