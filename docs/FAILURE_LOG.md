# Failure log

## 1. Steering chatter: agents succeeded but drove like they were having a seizure

**Symptom.** The first SAC agent finished 90% of held-out roads after 30k steps, which looked like a win. Its mean absolute steering change was 0.395 per step, though, against 0.027 for the hand-written PD lane keeper. Over a 0.1 s step that is roughly a fifth of full lock. PPO showed the same pattern, less severely (0.176). Success rate alone hid the problem entirely.

**Ruled out first: action saturation.** The usual cause of bang-bang control in SAC is the policy pinning the tanh at ±1 (for example, from a missing log-det-Jacobian term in the log-probability). I measured it directly: only 1% of steering actions had |a| > 0.95 for either agent. The SAC log-prob was also already covered by a test against `TransformedDistribution(Normal, TanhTransform)`. Not that.

**Diagnosis: a reward-scale imbalance.** The smoothness penalty was `0.1 · |Δsteer|`, at most 0.2 per step, while progress pays about 1.3 per step at speed. Weaving cost almost nothing. Meanwhile the range sensor is noisy as a function of heading (rays sweep across obstacle edges), so a policy reacting to it every step oscillates unless something tells it not to.

**Hypothesis 1, wrong: limit the actuator.** Real steering racks are rate-limited, so I added `steer_rate` (max change 0.15 per step) to the environment. Chatter dropped (0.568 → 0.135 for PPO), but success fell from 96% to 71% for PPO and to 36% for SAC, and collisions jumped to 29%. The car could no longer turn in time to clear an obstacle 16 m ahead at 14 m/s. The hard limit removed the symptom by removing the capability.

**Fix: make chatter expensive, not impossible.** I raised the smoothness weight from 0.1 to 1.0. A full left-to-right flip (Δ = 2) now costs as much as about 1.5 steps of progress, against 0.15 steps before. 3-seed results ([results/smoothness.md](../results/smoothness.md)):

| | PPO success | PPO mean abs steer change | SAC success | SAC mean abs steer change |
|---|---|---|---|---|
| weight 0.1 | 96% ± 1% | 0.568 ± 0.375 | 68% ± 41% | 0.403 ± 0.167 |
| rate limit 0.15 | 71% ± 2% | 0.135 | 36% ± 26% | 0.129 |
| **weight 1.0** | 93% ± 2% | **0.133** (−77%) | **89% ± 11%** | **0.120** (−70%) |

PPO gives up about 3 points of success for 77% less steering activity. For SAC it is a net improvement: one of the three weight-0.1 seeds (seed 1) had collapsed into crawling at 2.8 m/s and timing out on 68% of roads, and with weight 1.0 all seeds drive at full speed.

**What I took from it.**
1. Report a control-quality metric next to success rate. `mean_abs_steer_change` is now in every evaluation.
2. Measure before assuming the textbook cause (saturation).
3. A constraint that fixes a symptom can quietly destroy the capability you need. The ablation table is what showed it.

**Reproduce:** `python scripts/benchmark.py smoothness && python scripts/report.py smoothness`

## 2. Obstacle spacing (caught before any training)

**Symptom (found during design review).** The first obstacle sampler drew obstacle positions uniformly along the road. Two obstacles on opposite sides of the lane a few metres apart leave no gap a car of radius 1 m can get through at speed. Any agent would "fail" those episodes, inflating collision rates with impossible layouts.

**Fix.** One obstacle per equal road segment, with an 8 m margin from each segment edge, which guarantees at least 16 m spacing along x and the first obstacle at least 33 m from the start. `test_obstacle_layout_always_passable` checks this spacing over 200 seeds. Despite its name, the test does not prove a layout is drivable. It checks neither the lateral gap nor whether a car at 14 m/s can swerve between alternating-side obstacles 16 m apart. Proving that would need a scripted planner driving every layout.

## 3. Time-limit truncation treated as a crash (prevented, with a regression test)

**The trap.** When an episode hits the 400-step limit, the car is not in a terminal state; the clock just ran out. Treating that step as terminal tells the value function "this state is worth 0", which is false and biases learning toward ending episodes early. The opposite mistake, bootstrapping across the boundary, leaks the next episode's rewards into this one's advantage.

**Implementation.** `compute_gae` takes `terminated` (cuts the bootstrap) and `ends` (stops the GAE recursion) separately, and PPO evaluates V on the true next observation *before* the auto-reset. SAC stores only `terminated` in the replay buffer.

**Tests.** `test_gae_truncation_bootstraps_but_does_not_leak` builds a two-step case where the right answer is 5.0, the leaking bug gives 105, and the treat-as-crash bug gives 0. `test_gae_matches_naive_reference` checks the recursion against a brute-force sum on random rollouts.
