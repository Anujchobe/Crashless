import math

import numpy as np
import pytest
import torch

from crashless.networks import SquashedGaussianActor
from crashless.ppo import PPO, PPOConfig, compute_gae
from crashless.sac import SAC, ReplayBuffer, SACConfig


# ------------------------------------------------------------------ GAE
def naive_gae(r, v, nv, term, end, g, lam):
    """Reference: compute every advantage as an explicit discounted sum of TD errors."""
    T, N = r.shape
    adv = np.zeros_like(r)
    for n in range(N):
        for t in range(T):
            total, k = 0.0, t
            while k < T:
                delta = r[k, n] + g * nv[k, n] * (1 - term[k, n]) - v[k, n]
                total += (g * lam) ** (k - t) * delta
                if end[k, n]:
                    break
                k += 1
            adv[t, n] = total
    return adv


def test_gae_matches_naive_reference():
    rng = np.random.default_rng(0)
    T, N = 30, 4
    r, v, nv = rng.normal(size=(3, T, N))
    term = (rng.random((T, N)) < 0.1).astype(float)
    trunc = (rng.random((T, N)) < 0.1).astype(float) * (1 - term)
    end = np.maximum(term, trunc)
    adv, ret = compute_gae(r, v, nv, term, end, 0.99, 0.95)
    np.testing.assert_allclose(adv, naive_gae(r, v, nv, term, end, 0.99, 0.95), atol=1e-10)
    np.testing.assert_allclose(ret, adv + v)


def test_gae_truncation_bootstraps_but_does_not_leak():
    """Regression test for the time-limit bug: a truncated step must use V(s')
    but must not pull in the advantage of the next (unrelated) episode."""
    r = np.array([[0.0], [100.0]])           # step 1 belongs to a NEW episode
    v = np.zeros((2, 1))
    nv = np.array([[5.0], [0.0]])
    term = np.zeros((2, 1))
    end = np.array([[1.0], [0.0]])          # step 0 was truncated
    adv, _ = compute_gae(r, v, nv, term, end, gamma=1.0, lam=1.0)
    assert adv[0, 0] == pytest.approx(5.0)  # bootstrapped value, not 105
    # Treating truncation as termination (the bug) would give 0 instead.
    adv_bug, _ = compute_gae(r, v, nv, end, end, 1.0, 1.0)
    assert adv_bug[0, 0] == pytest.approx(0.0)


# ------------------------------------------------------------------ SAC policy
def test_squashed_gaussian_logprob_matches_density():
    """log pi(a) must equal the change-of-variables density of tanh(u).
    Checked against torch's TransformedDistribution."""
    torch.manual_seed(0)
    actor = SquashedGaussianActor(5, 2, 32)
    obs = torch.randn(1000, 5)
    torch.manual_seed(1)
    a, logp = actor(obs)
    mu, log_std = actor.body(obs).split(2, -1)
    base = torch.distributions.Normal(mu, log_std.clamp(-5, 2).exp())
    ref = torch.distributions.TransformedDistribution(
        base, [torch.distributions.transforms.TanhTransform(cache_size=1)])
    a_c = a.clamp(-1 + 1e-6, 1 - 1e-6)
    torch.testing.assert_close(logp, ref.log_prob(a_c).sum(-1), atol=2e-3, rtol=1e-3)


def test_squashed_actions_in_bounds_even_for_huge_inputs():
    actor = SquashedGaussianActor(5, 2, 32)
    a, logp = actor(torch.randn(256, 5) * 1e4)
    assert torch.all(a.abs() <= 1) and torch.all(torch.isfinite(logp))


def test_replay_buffer_wraps_and_samples():
    b = ReplayBuffer(3, 2, size=10)
    for i in range(25):
        b.add(np.full(3, i), np.zeros(2), float(i), np.zeros(3), 0.0)
    assert len(b) == 10
    o, a, r, o2, t = b.sample(64)
    assert r.min() >= 15 and r.max() <= 24


def test_sac_target_ignores_bootstrap_only_on_terminal():
    agent = SAC({"n_obstacles": 0}, SACConfig(batch_size=2, learning_starts=0))
    od = agent.env.obs_dim
    for term in (0.0, 1.0):
        agent.buf.add(np.zeros(od), np.zeros(2), 1.0, np.zeros(od), term)
    stats = agent.update()
    assert np.isfinite(stats["q_loss"])


# ------------------------------------------------------------------ learning smoke tests
@pytest.mark.slow
def test_ppo_improves_on_empty_road():
    from crashless.evaluate import evaluate
    env_cfg = {"n_obstacles": 0, "road_length": 100}
    agent = PPO(env_cfg, PPOConfig(total_steps=40_960), seed=0)
    before = evaluate(agent, env_cfg, episodes=10)["mean_progress"]
    agent.train()
    after = evaluate(agent, env_cfg, episodes=10)["mean_progress"]
    assert after > before + 0.2, (before, after)


@pytest.mark.slow
def test_sac_learns_to_finish_empty_road():
    from crashless.evaluate import evaluate
    env_cfg = {"n_obstacles": 0, "road_length": 100}
    agent = SAC(env_cfg, SACConfig(total_steps=8_000, learning_starts=1_000), seed=0)
    agent.train()
    assert evaluate(agent, env_cfg, episodes=10)["success_rate"] >= 0.8
