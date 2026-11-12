# CARLA port: guide and build log

Goal: run the same task in CARLA, measure how a 2D-trained policy transfers, then fine-tune it in CARLA.

## Setup (Linux, run once) — tested on Ubuntu 26.04, GTX 960M 4 GB, 7 GB RAM

```bash
# 1. CARLA 0.9.16 (8.3 GB). 0.9.16, not 0.9.15: its Python client supports 3.10-3.12.
mkdir -p ~/carla && cd ~/carla
wget -c https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz
tar -xzf CARLA_0.9.16.tar.gz
sudo apt-get install -y libomp5          # the only missing system library

# 2. Python 3.12 via uv (Ubuntu 26.04 ships 3.14, which has no carla wheel)
python3 -m venv ~/.uvenv && ~/.uvenv/bin/pip install uv && ~/.uvenv/bin/uv python install 3.12
cd ~/Crashless && ~/.uvenv/bin/uv venv -p 3.12 .venv
~/.uvenv/bin/uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cpu
~/.uvenv/bin/uv pip install -p .venv -r requirements.txt carla==0.9.16

# 3. Server (terminal 1): small map, NO renderer
~/carla/CarlaUE4.sh /Game/Carla/Maps/Town02 -nullrhi -nosound -carla-rpc-port=2000
```

Why those server flags (each was a crash first):

| Attempt | Result | Cause | Fix |
|---|---|---|---|
| `-RenderOffScreen` | `VK_ERROR_MEMORY_MAP_FAILED` | Laptop has Intel + NVIDIA GPUs; Vulkan used the Intel one | `VK_ICD_FILENAMES=.../nvidia_icd.json -prefernvidia` |
| + NVIDIA forced | `Killed` | Out of memory loading the default map (Town10HD, ~9 GB) | Start in `/Game/Carla/Maps/Town02` |
| + Town02 | `Pure virtual function called` on client connect | Vulkan renderer on an old GPU | `-nullrhi`: no renderer at all |
| `-nullrhi` | Works, ~105 steps/s, ~5 GB RAM | Physics, collision sensor and ray-cast lidar do not need rendering | (cameras would not work) |

## Steps (terminal 2)

| # | Command | What you learn | Good result |
|---|---|---|---|
| 1 | `python scripts/carla_check.py` | Connection, sensors, sim speed, lidar accuracy | Routes ~215 m, 4/4 obstacles, steps/s printed, lidar error < 0.5 m |
| 2 | `python scripts/carla_eval.py lane-follow` | Baseline in CARLA | Mostly crashes (it ignores obstacles), few off-roads |
| 3 | `python scripts/carla_eval.py runs/transfer/sac_2d_s0` | **Zero-shot transfer**: 2D policy, no CARLA training | Whatever it is; this is the gap to explain |
| 4 | `python train.py --algo sac --env carla --init-from runs/transfer/sac_2d_s0/model.pt --learning-starts 1000 --steps 30000 --out runs/transfer/sac_finetune` | Fine-tuning closes the gap | Success above step 3 |
| 5 | `python train.py --algo sac --env carla --steps 30000 --out runs/transfer/sac_scratch` | Same budget from scratch | Lower than step 4 = 2D pretraining helped |

Steps 4 and 5 at 20-100 steps/s take ~5-25 min each. If CARLA hangs, kill both terminals and restart the server: a client that dies in synchronous mode leaves the server waiting for a tick.

## How the port works (read with `crashless/carla_env.py`)

1. **Same contract.** 14 observations, 2 actions, the same reward terms and episode rules as `env.py`. `test_carla_config_matches_2d_contract` checks the shared constants. Two measurements differ: the 2D env measures progress along x with a first-order lateral offset, while CARLA uses arc length and an exact projection.
2. **Route.** Follow CARLA waypoints 1 m apart from a random spawn point; at junctions take the straightest branch. Corridor = whole two-lane road: centre line, half-width = 1 lane (4 m in Town02, same as 2D).
3. **Geometry** (`crashless/route.py`, 9 tests): project the car onto the route polyline → progress `s`, lateral offset, road heading. A search window around the last segment stops it snapping to a nearby part of the route.
4. **Rays.** Road edges come from route geometry. Obstacles come from a 1-channel semantic lidar on the front bumper, keeping only points whose object id is one of our parked cars; `lidar_ray_distances` bins them into the 9 rays. `obstacle_sensing: ground_truth` swaps in exact boxes for debugging.
5. **Coordinates.** CARLA is left-handed (y right, yaw clockwise, degrees). `carla_to_local` flips y and yaw on the way in; steering is negated on the way out (CARLA +steer = right).
6. **Actuation.** Action ±1 → 0.5 rad at the wheels (scaled by the car's max steer angle). Throttle/brake split by sign; throttle cut above 15 m/s.
7. **Synchronous mode**, fixed 0.1 s steps, no rendering: the sim waits for our `tick()`, so physics and sensors line up with each action, and the GPU barely works.
8. **Parked cars** spawn with physics on, settle onto the road for 10 ticks, then freeze.

## Known gaps between the two sims (expect transfer loss here)

- Dynamics: CARLA has real vehicle physics (inertia, tyre slip, engine response) vs a kinematic model.
- Obstacles are cars (~3.8 m × 1.8 m), not 1 m discs. Town02 lanes are 4 m, so the corridor half-width matches the 2D sim (4 m).
- The ego car starts in its own lane (lateral ≈ −0.5 normalised), not on the centre line.
- Town02 junctions turn 90°; 2D roads are gentle curves.

## Build log

| Date | What | Result |
|---|---|---|
| 2026-11-10 | `route.py` + tests, `carla_env.py`, scripts | 28 tests (227 cases) pass on Mac |
| 2026-11-10 | Server setup on the Linux laptop | 3 crashes before it ran; see table above |
| 2026-11-10 | First `carla_check` | Worked first time: routes 214 m, Town02 half-width 4.0 m (same as 2D), 105 steps/s |
| 2026-11-10 | Lidar vs ground truth: 0.93 m mean / 7.5 m p95 error | Cause: each ray took the nearest point in a +-7.5 deg cone, 2D rays are thin. Cone narrowed to +-1 deg: p95 error 0.12 m |
| 2026-11-10 | Zero-shot transfer, 20 episodes | 2D SAC: 65% success (96% in 2D), 30% collision. Lane-follow PD: 35% |
| 2026-11-10 | Fine-tune vs scratch, 30k CARLA steps each (~11 min each, ~45 steps/s with updates) | 20-episode evals said 65% vs 50%: too noisy |
| 2026-11-10 | Re-evaluated all four on 100 episodes | Lane-follow 33%, zero-shot 69%, scratch 68%, **fine-tuned 83%** (seed 0) |
| 2026-11-11 | Seeds 1-2 (`scripts/carla_seeds.sh`, `scripts/carla_report.py`) | Fine-tuned 75% ± 14% vs scratch 45% ± 31% vs zero-shot 48% ± 16%; fine-tuned beat scratch on all 3 seeds (one-sided sign test p = 0.125, not significant). Evaluated on Town02, the training map |
| 2026-11-11 | Scratch seed 1 failed: 1% success | It learned to crawl at 2.3 m/s and time out on 48% of routes: same local optimum as 2D SAC seed 1 in the smoothness ablation (2.8 m/s). Slow driving avoids the -50 crash penalty |
| 2026-11-11 | Zero-shot varies 31-69% across seeds that all scored 96-100% in 2D | 2D success does not predict transfer. Most zero-shot losses are off-road (18% avg) at Town02's 90 deg junctions, which the 2D roads never have |
