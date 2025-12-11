# bsk_ppo_scheduler_full.py
"""
Basilisk + PPO scheduler (complete file)
- Features:
  * Basilisk-backed environment wrapper (single sat LEO) or mock fallback
  * PPO (PyTorch) actor-critic training (GAE, clipping, entropy)
  * Computes reward = seconds target is inside FOV during decision interval
  * Fixes: robust array shapes, reduces SWIG memory leak by cleaning GC when rebuilding sim
- Usage:
  * pip install torch numpy
  * If you have Basilisk and Vizard, set USE_BASILISK=True and VIZ=True (run Vizard first)
  * Run: python bsk_ppo_scheduler_full.py
"""

import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import time

# ----------------------
# Try import Basilisk (optional)
# ----------------------
BASILISK_AVAILABLE = False
try:
    from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, vizSupport, unitTestSupport
    from Basilisk.simulation import spacecraft
    BASILISK_AVAILABLE = True
except Exception:
    # Basilisk not available on this system; use mock mode
    BASILISK_AVAILABLE = False

# ----------------------
# Actor-Critic (PPO)
# ----------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.fc(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

def categorical_sample(logits: torch.Tensor):
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return a, dist.log_prob(a), dist.entropy()

# ----------------------
# Basilisk environment wrapper (or mock fallback)
# ----------------------
class BasiliskScheduleEnv:
    """
    action: 0..n_targets-1 choose target, n_targets = idle
    state: [pos(3)/1e7, vel(3)/1e3, (cosang, range) * n_targets]
    reward: seconds observed in decision interval (float)
    """
    def __init__(self,
                 n_targets: int = 4,
                 decision_dt: float = 10.0,
                 sample_dt: float = 1.0,
                 fov_deg: float = 30.0,
                 use_basilisk: bool = True,
                 viz: bool = False):
        self.n_targets = n_targets
        self.decision_dt = float(decision_dt)
        self.sample_dt = float(sample_dt)
        self.fov = math.radians(fov_deg)
        self.earth_radius = 6371e3
        self.sma = 7000e3
        self.use_basilisk = bool(use_basilisk and BASILISK_AVAILABLE)
        self.viz = bool(viz and self.use_basilisk)

        self._build_targets()
        if self.use_basilisk:
            self._build_basilisk_sim()
        else:
            # mock internal state for fallback
            self._mock_pos = np.array([self.sma, 0.0, 0.0], dtype=np.float64)
            mu = 398600.4418e9
            v_mag = math.sqrt(mu / self.sma)
            self._mock_vel = np.array([0.0, v_mag, 0.0], dtype=np.float64)

    def _build_targets(self):
        rng = np.random.RandomState(42)
        lats = rng.uniform(-60.0, 60.0, size=self.n_targets) * math.pi/180.0
        lons = rng.uniform(-180.0, 180.0, size=self.n_targets) * math.pi/180.0
        self.targets_latlon = list(zip(lats, lons))
        self.target_ecef = []
        for lat, lon in self.targets_latlon:
            x = self.earth_radius * math.cos(lat) * math.cos(lon)
            y = self.earth_radius * math.cos(lat) * math.sin(lon)
            z = self.earth_radius * math.sin(lat)
            self.target_ecef.append(np.array([x, y, z], dtype=np.float64))

    def _cleanup_prev_sim(self):
        # Try to reduce SWIG/Py memory leaks: drop references and GC
        try:
            if hasattr(self, 'vizObj') and self.vizObj is not None:
                self.vizObj = None
            if hasattr(self, 'scSim') and self.scSim is not None:
                self.scSim = None
            if hasattr(self, 'dataRec') and self.dataRec is not None:
                self.dataRec = None
            gc.collect()
        except Exception:
            gc.collect()

    def _build_basilisk_sim(self):
        # Clean previous
        self._cleanup_prev_sim()

        # Build sim
        self.scSim = SimulationBaseClass.SimBaseClass()
        self.scSim.SetProgressBar(False)
        self.processName = "dynProcess"
        self.taskName = "dynTask"

        dynProcess = self.scSim.CreateNewProcess(self.processName)
        self.sim_step_ns = int(macros.sec2nano(self.sample_dt))
        dynProcess.addTask(self.scSim.CreateNewTask(self.taskName, self.sim_step_ns))

        # spacecraft
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "bsk-Sat"
        self.scSim.AddModelToTask(self.taskName, self.scObject)

        # gravity
        gravFactory = simIncludeGravBody.gravBodyFactory()
        planet = gravFactory.createEarth()
        planet.isCentralBody = True
        gravFactory.addBodiesTo(self.scObject)
        self.planet = planet

        # orbit initial conditions (circular-ish)
        oe = orbitalMotion.ClassicElements()
        oe.a = self.sma
        oe.e = 0.001
        oe.i = 28.5 * macros.D2R
        oe.Omega = 0.0
        oe.omega = 0.0
        oe.f = 0.0
        mu = planet.mu
        rN, vN = orbitalMotion.elem2rv(mu, oe)
        # ensure shapes are 1D numpy
        self.scObject.hub.r_CN_NInit = np.array(rN, dtype=np.float64).reshape(3,)
        self.scObject.hub.v_CN_NInit = np.array(vN, dtype=np.float64).reshape(3,)

        # recorder
        self.dataRec = self.scObject.scStateOutMsg.recorder(self.sim_step_ns)
        self.scSim.AddModelToTask(self.taskName, self.dataRec)

        # viz
        self.vizObj = None
        if self.viz:
            try:
                self.vizObj = vizSupport.enableUnityVisualization(self.scSim, self.taskName, self.scObject, liveStream=True)
            except Exception as e:
                print("Warning: could not enable Vizard viz:", e)
                self.vizObj = None

        # init
        self.scSim.InitializeSimulation()
        self._rec_index = 0
        self._sim_time = 0.0

    def reset(self):
        if self.use_basilisk:
            # Rebuild sim once (clean) to reset
            self._build_basilisk_sim()
            # initial state from hub init
            pos = np.array(self.scObject.hub.r_CN_NInit, dtype=np.float64).reshape(3,)
            vel = np.array(self.scObject.hub.v_CN_NInit, dtype=np.float64).reshape(3,)
        else:
            pos = self._mock_pos.copy()
            vel = self._mock_vel.copy()
        state = self._build_state_vector(pos, vel)
        return state

    def _build_state_vector(self, pos: np.ndarray, vel: np.ndarray):
        # robust shaping: accept (N,3) or (3,) arrays
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)

        if pos.ndim > 1:
            if pos.shape[-1] == 3:
                pos = pos[-1]
            else:
                pos = pos.reshape(-1)[-3:]
        if vel.ndim > 1:
            if vel.shape[-1] == 3:
                vel = vel[-1]
            else:
                vel = vel.reshape(-1)[-3:]

        pos = pos.reshape(3,)
        vel = vel.reshape(3,)

        boresight = -pos
        bnorm = np.linalg.norm(boresight) + 1e-12

        state_list = [pos / 1e7, vel / 1e3]
        for tgt in self.target_ecef:
            tgt = np.asarray(tgt, dtype=np.float64).reshape(3,)
            sat2t = tgt - pos
            sat2t_norm = np.linalg.norm(sat2t) + 1e-12
            cosang = np.dot(sat2t, boresight) / (sat2t_norm * bnorm)
            cosang = float(np.clip(cosang, -1.0, 1.0))
            rng = float(sat2t_norm) / 1e6
            state_list.append(np.array([cosang, rng], dtype=np.float32))

        state_vec = np.concatenate([s.reshape(-1) for s in state_list]).astype(np.float32)
        return state_vec

    def _get_new_rec_samples(self):
        try:
            pos_all = np.array(self.dataRec.r_BN_N)
            vel_all = np.array(self.dataRec.v_BN_N)
            times = np.array(self.dataRec.times())
        except Exception:
            return np.empty((0,3)), np.empty((0,3)), np.empty((0,))

        if len(times) <= self._rec_index:
            return np.empty((0,3)), np.empty((0,3)), np.empty((0,))

        new_pos = pos_all[self._rec_index:]
        new_vel = vel_all[self._rec_index:]
        new_times = times[self._rec_index:]
        self._rec_index = len(times)

        if new_pos.ndim == 1 and new_pos.size == 3:
            new_pos = new_pos.reshape(1,3)
        if new_vel.ndim == 1 and new_vel.size == 3:
            new_vel = new_vel.reshape(1,3)

        return new_pos.astype(np.float64), new_vel.astype(np.float64), new_times.astype(np.float64)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if action is None:
            action = self.n_targets
        action = int(action)

        if self.use_basilisk:
            next_stop = self._sim_time + self.decision_dt
            # Basilisk expects absolute ns stop time
            self.scSim.ConfigureStopTime(macros.sec2nano(next_stop))
            self.scSim.ExecuteSimulation()
            self._sim_time = next_stop

            pos_samples, vel_samples, t_samples = self._get_new_rec_samples()
            if pos_samples.shape[0] == 0:
                # no samples: use last init
                last_pos = np.array(self.scObject.hub.r_CN_NInit).reshape(3,)
                last_vel = np.array(self.scObject.hub.v_CN_NInit).reshape(3,)
                reward = 0.0 if action < 0 or action >= self.n_targets else 0.0
            else:
                last_pos = pos_samples[-1].reshape(3,)
                last_vel = vel_samples[-1].reshape(3,)
                if 0 <= action < self.n_targets:
                    tgt = self.target_ecef[action]
                    visible_count = 0
                    # sample-based check
                    for p in pos_samples:
                        p = np.asarray(p, dtype=np.float64).reshape(3,)
                        boresight = -p
                        sat2t = tgt - p
                        cosang = np.dot(sat2t, boresight) / ((np.linalg.norm(sat2t)+1e-12) * (np.linalg.norm(boresight)+1e-12))
                        cosang = float(np.clip(cosang, -1.0, 1.0))
                        ang = math.acos(cosang)
                        if ang <= (self.fov / 2.0):
                            visible_count += 1
                    reward = visible_count * self.sample_dt
                else:
                    reward = -0.1 * self.decision_dt
        else:
            # mock propagation by rotation around z
            mu = 398600.4418e9
            n = math.sqrt(mu / (self.sma**3))
            theta = n * self.decision_dt
            c = math.cos(theta); s = math.sin(theta)
            Rz = np.array([[c, -s, 0],[s, c, 0],[0,0,1]])
            pos_new = Rz.dot(self._mock_pos)
            vel_new = (pos_new - self._mock_pos) / self.decision_dt

            if 0 <= action < self.n_targets:
                tgt = self.target_ecef[action]
                K = max(2, int(max(2, self.decision_dt / self.sample_dt)))
                visseconds = 0.0
                for k in range(K):
                    alpha = k / float(K-1)
                    p = (1-alpha) * self._mock_pos + alpha * pos_new
                    boresight = -p
                    sat2t = tgt - p
                    cosang = np.dot(sat2t, boresight) / ((np.linalg.norm(sat2t)+1e-12) * (np.linalg.norm(boresight)+1e-12))
                    cosang = float(np.clip(cosang, -1.0, 1.0))
                    ang = math.acos(cosang)
                    if ang <= (self.fov / 2.0):
                        visseconds += self.decision_dt / K
                reward = visseconds
            else:
                reward = -0.1 * self.decision_dt

            self._mock_pos = pos_new
            self._mock_vel = vel_new
            last_pos = self._mock_pos
            last_vel = self._mock_vel

        state = self._build_state_vector(last_pos, last_vel)
        done = False
        info = {}
        return state.astype(np.float32), float(reward), done, info

    def action_space(self):
        return self.n_targets + 1

    def state_dim(self):
        return 6 + self.n_targets * 2

# ----------------------
# GAE / PPO utilities
# ----------------------
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        next_value = values[t+1] if t+1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:len(rewards)]
    return advantages, returns

# ----------------------
# PPO training
# ----------------------
def train_ppo(env,
              epochs=200,
              batch_steps=2048,
              minibatch_size=64,
              ppo_epochs=4,
              gamma=0.99,
              lam=0.95,
              clip_eps=0.2,
              vf_coeff=0.5,
              ent_coeff=0.01,
              lr=3e-4,
              device='cpu'):
    state_dim = env.state_dim()
    action_dim = env.action_space()
    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_rewards = []

    for epoch in range(1, epochs+1):
        states_buf, actions_buf, logps_buf, rewards_buf, dones_buf, values_buf = [], [], [], [], [], []
        steps_collected = 0
        ep_rewards = []
        while steps_collected < batch_steps:
            s = env.reset()
            done = False
            ep_r = 0.0
            while not done and steps_collected < batch_steps:
                s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                logits, v = model(s_t)
                a_t, logp_t, ent_t = categorical_sample(logits)
                a = int(a_t.item())
                v_val = float(v.item())

                s2, r, done, _ = env.step(a)
                states_buf.append(s.copy())
                actions_buf.append(a)
                logps_buf.append(float(logp_t.item()))
                rewards_buf.append(float(r))
                dones_buf.append(float(done))
                values_buf.append(v_val)

                s = s2
                steps_collected += 1
                ep_r += r
            ep_rewards.append(ep_r)

        values_buf.append(0.0)

        states_np = np.array(states_buf, dtype=np.float32)
        actions_np = np.array(actions_buf, dtype=np.int64)
        logps_np = np.array(logps_buf, dtype=np.float32)
        rewards_np = np.array(rewards_buf, dtype=np.float32)
        dones_np = np.array(dones_buf, dtype=np.float32)
        values_np = np.array(values_buf, dtype=np.float32)

        advantages, returns = compute_gae(rewards_np, values_np, dones_np, gamma=gamma, lam=lam)
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        dataset_size = len(states_np)
        indices = np.arange(dataset_size)
        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, minibatch_size):
                mb_idx = indices[start:start+minibatch_size]
                mb_states = torch.tensor(states_np[mb_idx], dtype=torch.float32, device=device)
                mb_actions = torch.tensor(actions_np[mb_idx], dtype=torch.int64, device=device)
                mb_old_logps = torch.tensor(logps_np[mb_idx], dtype=torch.float32, device=device)
                mb_returns = torch.tensor(returns[mb_idx], dtype=torch.float32, device=device)
                mb_advantages = torch.tensor(advantages[mb_idx], dtype=torch.float32, device=device)

                logits, values = model(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                mb_logps = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(mb_logps - mb_old_logps)
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1.0-clip_eps, 1.0+clip_eps) * mb_advantages
                policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))

                value_loss = torch.mean((mb_returns - values)**2)
                loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        avg_ep_reward = float(np.mean(ep_rewards)) if len(ep_rewards)>0 else 0.0
        total_rewards.append(avg_ep_reward)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | AvgEpReward {avg_ep_reward:.3f} | Steps {dataset_size}")

    return model, total_rewards

# ----------------------
# Run example
# ----------------------
if __name__ == "__main__":
    # CONFIG
    USE_BASILISK =True   # set True if you have Basilisk installed
    VIZ = False           # set True to attempt Vizard live stream (requires Basilisk vizInterface)
    N_TARGETS = 100
    DECISION_DT = 10.0
    SAMPLE_DT = 1.0

    env = BasiliskScheduleEnv(n_targets=N_TARGETS,
                              decision_dt=DECISION_DT,
                              sample_dt=SAMPLE_DT,
                              fov_deg=30.0,
                              use_basilisk=USE_BASILISK,
                              viz=VIZ)

    device = 'cpu'
    t1 = time.time()
    model, rewards = train_ppo(env, epochs=30, batch_steps=512, minibatch_size=64, device=device)
    print("Training done. Last 5 avg rewards:", rewards[-5:])
    t2 = time.time()

    # playback one episode (if VIZ enabled you'll see live stream)
    s = env.reset()
    total = 0.0
    for step in range(60):
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        logits, v = model(s_t)
        action = int(torch.distributions.Categorical(logits=logits).sample().item())
        s, r, done, _ = env.step(action)
        total += r
        print(f"Step {step:02d} | action {action} | reward {r:.2f}")
    print("Playback total reward:", total)
