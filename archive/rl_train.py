"""
rl_train.py — Train a PPO Agent for Microgrid Battery Dispatch.

Trains a Proximal Policy Optimisation (PPO) agent inside MicrogridEnv.
Saves the best and final policy to outputs/rl_policy/.
Logs training curves to outputs/rl_logs/ (view with TensorBoard).

Usage
-----
    python rl_train.py                   # default 300 k steps
    python rl_train.py --timesteps 500000
    python rl_train.py --timesteps 50000 --n_days 7   # quick smoke test

TensorBoard
-----------
    tensorboard --logdir outputs/rl_logs/
"""

import os
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from rl_environment import MicrogridEnv


# ── Paths ──────────────────────────────────────────────────────────────────────
LOAD_PATH  = r'datasets\LD2011_2014.txt'
SOLAR_PATH = r'datasets\Plant_1_Generation_Data.csv'

POLICY_DIR = 'outputs/rl_policy'
LOG_DIR    = 'outputs/rl_logs'


# ── Custom callback — prints episode statistics ────────────────────────────────

class EpisodeStatsCallback(BaseCallback):
    """
    Logs mean episode reward and blackout count every N episodes.
    """

    def __init__(self, log_every: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.log_every = log_every
        self._ep_count = 0
        self._rewards:   list = []
        self._blackouts: list = []

    def _on_step(self) -> bool:
        # SB3 Monitor wrapper stores episode info in 'infos'
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._ep_count += 1
                self._rewards.append(info['episode']['r'])
                # collect blackout sum from last ep (stored in info by our env)
                self._blackouts.append(info.get('episode_blackouts', 0))

                if self._ep_count % self.log_every == 0 and self.verbose:
                    mean_r  = np.mean(self._rewards[-self.log_every:])
                    mean_bo = np.mean(self._blackouts[-self.log_every:])
                    print(
                        f"  [Ep {self._ep_count:5d}] "
                        f"mean_reward={mean_r:+8.2f}  "
                        f"mean_blackouts={mean_bo:.1f}"
                    )
        return True


# ── Factory function so make_vec_env can create multiple copies ───────────────

def _make_monitored_env(load_path, solar_path, n_days):
    """Return a Monitor-wrapped MicrogridEnv."""
    def _init():
        env = MicrogridEnv(load_path, solar_path, n_days=n_days)
        env = Monitor(env)
        return env
    return _init


# ── Main training function ────────────────────────────────────────────────────

def train(
    timesteps: int  = 300_000,
    n_days:    int  = 30,
    n_envs:    int  = 2,
    lr:        float = 3e-4,
    seed:      int  = 42,
) -> PPO:
    """
    Train a PPO agent and return the trained model.

    Parameters
    ----------
    timesteps : int
        Total environment steps to collect (more = better policy).
    n_days : int
        Simulation horizon per episode (days).
    n_envs : int
        Number of parallel environments (speeds up data collection).
    lr : float
        PPO learning rate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    PPO
        The trained stable-baselines3 PPO model.
    """
    os.makedirs(POLICY_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,    exist_ok=True)

    print("=" * 60)
    print("  MICROGRID RL TRAINING — PPO")
    print("=" * 60)
    print(f"  Timesteps   : {timesteps:,}")
    print(f"  Days/episode: {n_days}  ({n_days * 24} steps/ep)")
    print(f"  Parallel env: {n_envs}")
    print(f"  Learning rate: {lr}")
    print(f"  Policy dir  : {POLICY_DIR}")
    print(f"  Log dir     : {LOG_DIR}")
    print()

    # ── Training environments (vectorised) ───────────────────────────────
    train_env = make_vec_env(
        _make_monitored_env(LOAD_PATH, SOLAR_PATH, n_days),
        n_envs=n_envs,
        seed=seed,
    )

    # ── Evaluation environment (single, deterministic) ───────────────────
    eval_env = Monitor(MicrogridEnv(LOAD_PATH, SOLAR_PATH, n_days=n_days))

    # ── PPO agent ────────────────────────────────────────────────────────
    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = lr,
        n_steps         = 1024,        # steps per env before update
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,        # discount factor
        gae_lambda      = 0.95,        # GAE smoothing
        ent_coef        = 0.005,       # entropy bonus (exploration)
        clip_range      = 0.2,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        seed            = seed,
        tensorboard_log = LOG_DIR,
        verbose         = 0,           # silence SB3's own prints
        policy_kwargs   = dict(
            net_arch = [dict(pi=[128, 128], vf=[128, 128])],
        ),
    )

    # ── Callbacks ────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = POLICY_DIR,
        log_path             = LOG_DIR,
        eval_freq            = max(n_days * 24, 1000),   # evaluate once per ~episode
        n_eval_episodes      = 3,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    stats_callback = EpisodeStatsCallback(log_every=5, verbose=1)

    callback = CallbackList([eval_callback, stats_callback])

    # ── Train ────────────────────────────────────────────────────────────
    print("Starting training... (this may take several minutes)")
    print("Monitor progress with:  tensorboard --logdir outputs/rl_logs/\n")

    model.learn(
        total_timesteps     = timesteps,
        callback            = callback,
        progress_bar        = False,   # requires tqdm/rich (stable-baselines3[extra])
    )

    # ── Save final model ─────────────────────────────────────────────────
    final_path = os.path.join(POLICY_DIR, 'final_ppo_model')
    model.save(final_path)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Best model  : {POLICY_DIR}/best_model.zip")
    print(f"  Final model : {final_path}.zip")
    print(f"  Logs        : tensorboard --logdir {LOG_DIR}/")
    print()

    train_env.close()
    eval_env.close()

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO agent for microgrid dispatch')
    parser.add_argument('--timesteps', type=int,   default=300_000,
                        help='Total training timesteps (default: 300000)')
    parser.add_argument('--n_days',    type=int,   default=30,
                        help='Simulation days per episode (default: 30)')
    parser.add_argument('--n_envs',    type=int,   default=2,
                        help='Parallel environments (default: 2)')
    parser.add_argument('--lr',        type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--seed',      type=int,   default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    train(
        timesteps = args.timesteps,
        n_days    = args.n_days,
        n_envs    = args.n_envs,
        lr        = args.lr,
        seed      = args.seed,
    )
