import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List

BASILISK_AVAILABLE = False
try:
    from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, vizSupport
    from Basilisk.simulation import spacecraft
    BASILISK_AVAILABLE = True
except Exception:
    BASILISK_AVAILABLE = False

# MAPPO model (shared actor + centralized critic)
class MAPPOModel(nn.Module):
    def __init__(self, local_state_dim: int, action_dim: int, n_agents: int, hidden=128):
        super().__init__()
        self.n_agents = n_agents
        self.actor = nn.Sequential(
            nn.Linear(local_state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(local_state_dim * n_agents, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_agents)
        )

    def forward_actor(self, local_obs: torch.Tensor):
        return self.actor(local_obs)

    def forward_critic(self, global_obs: torch.Tensor):
        return self.critic(global_obs)

# Multi-satellite env with downlink and wheel spin
class MultiSatScheduleEnv:
    def __init__(self,
                 n_agents: int = 3,
                 n_targets: int = 4,
                 decision_dt: float = 10.0,
                 sample_dt: float = 1.0,
                 fov_deg: float = 30.0,
                 use_basilisk: bool = False,
                 viz: bool = False,
                 downlink_rate_mb_s: float = 5.0,
                 wheel_boost_factor: float = 1.5,
                 wheel_boost_duration: float = 20.0,
                 coverage_coeff: float = 2.0):
        self.n_agents = int(n_agents)
        self.n_targets = int(n_targets)
        self.decision_dt = float(decision_dt)
        self.sample_dt = float(sample_dt)
        self.fov_nom = math.radians(fov_deg)
        self.earth_radius = 6371e3
        self.sma = 7000e3
        self.use_basilisk = bool(use_basilisk and BASILISK_AVAILABLE)
        self.viz = bool(viz and self.use_basilisk)

        # downlink & wheel parameters
        self.downlink_rate = float(downlink_rate_mb_s)  # MB/s
        self.wheel_boost_factor = float(wheel_boost_factor)
        self.wheel_boost_duration = float(wheel_boost_duration)
        self.coverage_coeff = float(coverage_coeff)

        # build targets and ground station
        self._build_targets()
        self._build_ground_station()

        # init satellite mock states and data/wheel buffers
        self._init_sat_states()

        # attempt Basilisk multi-sat if requested
        if self.use_basilisk:
            try:
                self._build_basilisk_sim()
            except Exception as e:
                print('Warning: Basilisk multi-sat build failed, falling back to mock:', e)
                self.use_basilisk = False

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

    def _build_ground_station(self):
        gs_lat = 0.0 * math.pi/180.0
        gs_lon = 0.0 * math.pi/180.0
        x = self.earth_radius * math.cos(gs_lat) * math.cos(gs_lon)
        y = self.earth_radius * math.cos(gs_lat) * math.sin(gs_lon)
        z = self.earth_radius * math.sin(gs_lat)
        self.gs_ecef = np.array([x, y, z], dtype=np.float64)

    def _init_sat_states(self):
        # mock initialization
        self._mock_pos = []
        self._mock_vel = []
        mu = 398600.4418e9
        v_mag = math.sqrt(mu / self.sma)
        for i in range(self.n_agents):
            theta = (2.0 * math.pi * i) / self.n_agents
            pos = np.array([self.sma * math.cos(theta), self.sma * math.sin(theta), 0.0], dtype=np.float64)
            vel = np.array([-v_mag * math.sin(theta), v_mag * math.cos(theta), 0.0], dtype=np.float64)
            self._mock_pos.append(pos)
            self._mock_vel.append(vel)
        self._mock_pos = np.stack(self._mock_pos, axis=0)
        self._mock_vel = np.stack(self._mock_vel, axis=0)

        # agent internal buffers
        self.data_buffer = np.zeros(self.n_agents, dtype=np.float32)  # MB of stored data
        self.wheel_boost = np.zeros(self.n_agents, dtype=np.float32)  # seconds remaining

    def _cleanup_prev_sim(self):
        try:
            for name in ['vizObjs', 'scSim', 'sc_objects', 'dataRec', 'vizObj', 'scObject', 'dataRec']:
                if hasattr(self, name):
                    try:
                        delattr(self, name)
                    except Exception:
                        try:
                            delattr(self, name)
                        except Exception:
                            try:
                                delattr(self, name)
                            except Exception:
                                pass
            for n in ['vizObjs', 'scSim', 'sc_objects', 'dataRec', 'vizObj', 'scObject']:
                if n in self.__dict__:
                    try:
                        del self.__dict__[n]
                    except Exception:
                        pass
            gc.collect()
        except Exception:
            gc.collect()

    def _build_basilisk_sim(self):
        # build a Basilisk sim with multiple spacecraft and recorders
        self._cleanup_prev_sim()
        self.scSim = SimulationBaseClass.SimBaseClass()
        self.scSim.SetProgressBar(False)
        self.processName = 'dynProcess'
        self.taskName = 'dynTask'

        dynProcess = self.scSim.CreateNewProcess(self.processName)
        self.sim_step_ns = int(macros.sec2nano(self.sample_dt))
        dynProcess.addTask(self.scSim.CreateNewTask(self.taskName, self.sim_step_ns))

        self.sc_objects = []
        self.dataRec = []
        gravFactory = simIncludeGravBody.gravBodyFactory()
        planet = gravFactory.createEarth()
        planet.isCentralBody = True

        for i in range(self.n_agents):
            sc = spacecraft.Spacecraft()
            sc.ModelTag = f'bsk-Sat-{i}'
            self.scSim.AddModelToTask(self.taskName, sc)
            gravFactory.addBodiesTo(sc)
            oe = orbitalMotion.ClassicElements()
            oe.a = self.sma
            oe.e = 0.001
            oe.i = 28.5 * macros.D2R
            oe.Omega = 0.0
            oe.omega = 0.0
            oe.f = (2.0 * math.pi * i) / self.n_agents
            rN, vN = orbitalMotion.elem2rv(planet.mu, oe)
            sc.hub.r_CN_NInit = np.array(rN, dtype=np.float64).reshape(3,)
            sc.hub.v_CN_NInit = np.array(vN, dtype=np.float64).reshape(3,)
            rec = sc.scStateOutMsg.recorder(self.sim_step_ns)
            self.scSim.AddModelToTask(self.taskName, rec)
            self.sc_objects.append(sc)
            self.dataRec.append(rec)

        self.vizObjs = []
        if self.viz:
            try:
                for sc in self.sc_objects:
                    viz = vizSupport.enableUnityVisualization(self.scSim, self.taskName, sc, liveStream=True)
                    self.vizObjs.append(viz)
            except Exception as e:
                print('Warning: could not enable Vizard viz for multi-sat:', e)
                self.vizObjs = []

        self.scSim.InitializeSimulation()
        self._rec_index = [0] * self.n_agents
        self._sim_time = 0.0

    def reset(self):
        if self.use_basilisk:
            self._build_basilisk_sim()
            states = []
            for i, sc in enumerate(self.sc_objects):
                pos = np.array(sc.hub.r_CN_NInit, dtype=np.float64).reshape(3,)
                vel = np.array(sc.hub.v_CN_NInit, dtype=np.float64).reshape(3,)
                states.append(self._build_local_state_vector(pos, vel, self.wheel_boost[i]))
            self.data_buffer = np.zeros(self.n_agents, dtype=np.float32)
            self.wheel_boost = np.zeros(self.n_agents, dtype=np.float32)
            return np.stack(states, axis=0)
        else:
            self._init_sat_states()
            self.data_buffer = np.zeros(self.n_agents, dtype=np.float32)
            self.wheel_boost = np.zeros(self.n_agents, dtype=np.float32)
            states = []
            for i in range(self.n_agents):
                states.append(self._build_local_state_vector(self._mock_pos[i], self._mock_vel[i], self.wheel_boost[i]))
            return np.stack(states, axis=0)

    def _build_local_state_vector(self, pos: np.ndarray, vel: np.ndarray, wheel_boost_remaining: float):
        pos = np.asarray(pos, dtype=np.float64).reshape(3,)
        vel = np.asarray(vel, dtype=np.float64).reshape(3,)
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

        # wheel boost remaining normalized by wheel_boost_duration (clip 0..1)
        wb_norm = float(np.clip(wheel_boost_remaining / (self.wheel_boost_duration + 1e-12), 0.0, 1.0))
        state_list.append(np.array([wb_norm], dtype=np.float32))

        state_vec = np.concatenate([s.reshape(-1) for s in state_list]).astype(np.float32)
        return state_vec

    def _get_new_rec_samples_multi(self):
        pos_all = []
        vel_all = []
        times_all = []
        for rec in self.dataRec:
            try:
                pos_all.append(np.array(rec.r_BN_N))
                vel_all.append(np.array(rec.v_BN_N))
                times_all.append(np.array(rec.times()))
            except Exception:
                pos_all.append(np.empty((0,3)))
                vel_all.append(np.empty((0,3)))
                times_all.append(np.empty((0,)))
        return pos_all, vel_all, times_all

    def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        if actions is None:
            actions = [self.n_targets] * self.n_agents
        actions = [int(a) for a in actions]

        rewards = np.zeros(self.n_agents, dtype=np.float32)
        next_states = [None] * self.n_agents

        # track which target each agent observed (or -1)
        observed_targets = [-1] * self.n_agents

        if self.use_basilisk:
            next_stop = self._sim_time + self.decision_dt
            self.scSim.ConfigureStopTime(macros.sec2nano(next_stop))
            self.scSim.ExecuteSimulation()
            self._sim_time = next_stop

            pos_all, vel_all, times_all = self._get_new_rec_samples_multi()
            for i in range(self.n_agents):
                pos_i = pos_all[i]
                vel_i = vel_all[i]
                if pos_i.shape[0] == 0:
                    pos = np.array(self.sc_objects[i].hub.r_CN_NInit).reshape(3,)
                    vel = np.array(self.sc_objects[i].hub.v_CN_NInit).reshape(3,)
                else:
                    pos = pos_i[-1].reshape(3,)
                    vel = vel_i[-1].reshape(3,)

                # effective fov
                efov = self.fov_nom * (self.wheel_boost_factor if self.wheel_boost[i] > 0.0 else 1.0)

                a = actions[i]
                if 0 <= a < self.n_targets:
                    tgt = self.target_ecef[a]
                    visible_count = 0
                    for p in pos_i:
                        p = np.asarray(p, dtype=np.float64).reshape(3,)
                        boresight = -p
                        sat2t = tgt - p
                        cosang = np.dot(sat2t, boresight) / ((np.linalg.norm(sat2t)+1e-12)*(np.linalg.norm(boresight)+1e-12))
                        cosang = float(np.clip(cosang, -1.0, 1.0))
                        ang = math.acos(cosang)
                        if ang <= (efov/2.0):
                            visible_count += 1
                    img_seconds = visible_count * self.sample_dt
                    rewards[i] = img_seconds
                    self.data_buffer[i] += img_seconds * 0.5  
                    if img_seconds > 0.0:
                        observed_targets[i] = a
                elif a == self.n_targets + 1:
                    # downlink action
                    visible_count = 0
                    for p in pos_i:
                        p = np.asarray(p, dtype=np.float64).reshape(3,)
                        boresight = -p
                        gs_vec = self.gs_ecef - p
                        cosang = np.dot(gs_vec, boresight) / ((np.linalg.norm(gs_vec)+1e-12)*(np.linalg.norm(boresight)+1e-12))
                        cosang = float(np.clip(cosang, -1.0, 1.0))
                        ang = math.acos(cosang)
                        if ang <= (self.fov_nom/2.0):
                            visible_count += 1
                    dl_seconds = visible_count * self.sample_dt
                    transmit = min(self.data_buffer[i], self.downlink_rate * dl_seconds)
                    rewards[i] = transmit
                    self.data_buffer[i] -= transmit
                elif a == self.n_targets + 2:
                    # spin wheel
                    self.wheel_boost[i] = self.wheel_boost_duration
                    rewards[i] = -0.1 * self.decision_dt
                else:
                    # idle
                    rewards[i] = -0.01 * self.decision_dt

                next_states[i] = self._build_local_state_vector(pos, vel, self.wheel_boost[i])

        else:
            mu = 398600.4418e9
            n = math.sqrt(mu / (self.sma**3))
            theta = n * self.decision_dt
            c = math.cos(theta); s = math.sin(theta)
            Rz = np.array([[c, -s, 0],[s, c, 0],[0,0,1]])
            K = max(2, int(max(2, self.decision_dt / self.sample_dt)))

            pos_new = np.zeros_like(self._mock_pos)
            vel_new = np.zeros_like(self._mock_vel)
            for i in range(self.n_agents):
                pos_new[i] = Rz.dot(self._mock_pos[i])
                vel_new[i] = (pos_new[i] - self._mock_pos[i]) / self.decision_dt

                # effective fov
                efov = self.fov_nom * (self.wheel_boost_factor if self.wheel_boost[i] > 0.0 else 1.0)

                a = actions[i]
                if 0 <= a < self.n_targets:
                    tgt = self.target_ecef[a]
                    visseconds = 0.0
                    for k in range(K):
                        alpha = k / float(K-1)
                        p = (1-alpha) * self._mock_pos[i] + alpha * pos_new[i]
                        boresight = -p
                        sat2t = tgt - p
                        cosang = np.dot(sat2t, boresight) / ((np.linalg.norm(sat2t)+1e-12) * (np.linalg.norm(boresight)+1e-12))
                        cosang = float(np.clip(cosang, -1.0, 1.0))
                        ang = math.acos(cosang)
                        if ang <= (efov / 2.0):
                            visseconds += self.decision_dt / K
                    rewards[i] = visseconds
                    self.data_buffer[i] += visseconds * 0.5
                    if visseconds > 0.0:
                        observed_targets[i] = a
                elif a == self.n_targets + 1:
                    # downlink
                    visseconds = 0.0
                    for k in range(K):
                        alpha = k / float(K-1)
                        p = (1-alpha) * self._mock_pos[i] + alpha * pos_new[i]
                        boresight = -p
                        gs_vec = self.gs_ecef - p
                        cosang = np.dot(gs_vec, boresight) / ((np.linalg.norm(gs_vec)+1e-12) * (np.linalg.norm(boresight)+1e-12))
                        cosang = float(np.clip(cosang, -1.0, 1.0))
                        ang = math.acos(cosang)
                        if ang <= (self.fov_nom/2.0):
                            visseconds += self.decision_dt / K
                    transmit = min(self.data_buffer[i], self.downlink_rate * visseconds)
                    rewards[i] = transmit
                    self.data_buffer[i] -= transmit
                elif a == self.n_targets + 2:
                    self.wheel_boost[i] = self.wheel_boost_duration
                    rewards[i] = -0.1 * self.decision_dt
                else:
                    rewards[i] = -0.01 * self.decision_dt

                next_states[i] = self._build_local_state_vector(pos_new[i], vel_new[i], self.wheel_boost[i])

            self._mock_pos = pos_new
            self._mock_vel = vel_new

        unique_targets = set([t for t in observed_targets if t >= 0])
        team_coverage = len(unique_targets)
        if self.n_targets > 0:
            coverage_bonus_each = self.coverage_coeff * (team_coverage / float(self.n_targets))
        else:
            coverage_bonus_each = 0.0

        rewards = rewards + coverage_bonus_each

        # decrement wheel boost timers
        self.wheel_boost = np.clip(self.wheel_boost - self.decision_dt, 0.0, None)

        next_states = np.stack(next_states, axis=0)
        done = False
        info = {
            'observed_targets': observed_targets,
            'team_coverage': team_coverage,
            'data_buffer': self.data_buffer.copy()
        }
        return next_states.astype(np.float32), rewards.astype(np.float32), done, info

    def action_space(self):
        # imaging targets + idle + downlink + wheel
        return self.n_targets + 3

    def local_state_dim(self):
        # pos(3), vel(3), (cos,range)*n_targets, wheel_boost(1)
        return 6 + self.n_targets * 2 + 1

    def global_state_dim(self):
        return self.local_state_dim() * self.n_agents

# GAE 

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

# MAPPO training 
def train_mappo(env: MultiSatScheduleEnv,
                n_agents: int = 3,
                epochs: int = 200,
                batch_steps: int = 2048,
                minibatch_size: int = 256,
                ppo_epochs: int = 4,
                gamma: float = 0.99,
                lam: float = 0.95,
                clip_eps: float = 0.2,
                vf_coeff: float = 0.5,
                ent_coeff: float = 0.01,
                lr: float = 3e-4,
                device: str = 'cpu'):

    local_state_dim = env.local_state_dim()
    action_dim = env.action_space()
    model = MAPPOModel(local_state_dim, action_dim, n_agents).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_rewards = []

    try:
        for epoch in range(1, epochs+1):
            local_states_buf = []
            global_states_buf = []
            actions_buf = []
            logps_buf = []
            rewards_buf = []
            dones_buf = []
            values_buf = []

            steps_collected = 0
            ep_rewards = []

            while steps_collected < batch_steps:
                obs = env.reset()
                done = False
                ep_r = np.zeros(n_agents, dtype=np.float32)

                while not done and steps_collected < batch_steps:
                    global_obs = obs.reshape(-1)
                    local_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                    logits_tensor = model.forward_actor(local_obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits_tensor)
                    acts = dist.sample()
                    logps_t = dist.log_prob(acts).detach().cpu().numpy()

                    global_obs_tensor = torch.tensor(global_obs.reshape(1, -1), dtype=torch.float32, device=device)
                    vals = model.forward_critic(global_obs_tensor).detach().cpu().numpy().reshape(-1)

                    actions = []
                    values_list = []
                    for i in range(n_agents):
                        a = int(acts[i].item())
                        actions.append(a)
                        values_list.append(float(vals[i]))

                    next_obs, rewards, done, info = env.step(actions)

                    local_states_buf.append(obs.copy())
                    global_states_buf.append(global_obs.copy())
                    actions_buf.append(np.array(actions, dtype=np.int64))
                    logps_buf.append(np.array(logps_t, dtype=np.float32))
                    rewards_buf.append(rewards.copy())
                    dones_buf.append(float(done))
                    values_buf.append(np.array(values_list, dtype=np.float32))

                    obs = next_obs
                    steps_collected += 1
                    ep_r += rewards

                ep_rewards.append(ep_r.sum())

            values_buf.append(np.zeros(n_agents, dtype=np.float32))

            T = len(rewards_buf)
            local_states_np = np.array(local_states_buf, dtype=np.float32)
            global_states_np = np.array(global_states_buf, dtype=np.float32)
            actions_np = np.array(actions_buf, dtype=np.int64)
            logps_np = np.array(logps_buf, dtype=np.float32)
            rewards_np = np.array(rewards_buf, dtype=np.float32)
            dones_np = np.array(dones_buf, dtype=np.float32)
            values_np = np.array(values_buf, dtype=np.float32)

            advantages = np.zeros_like(rewards_np, dtype=np.float32)
            returns = np.zeros_like(rewards_np, dtype=np.float32)
            for agent_idx in range(n_agents):
                agent_rewards = rewards_np[:, agent_idx]
                agent_values = values_np[:, agent_idx]
                advs, rets = compute_gae(agent_rewards, agent_values, dones_np, gamma=gamma, lam=lam)
                advantages[:, agent_idx] = advs
                returns[:, agent_idx] = rets

            dataset_size = T * n_agents
            local_states_flat = local_states_np.reshape(T * n_agents, local_state_dim)
            global_states_rep = np.repeat(global_states_np, n_agents, axis=0)
            actions_flat = actions_np.reshape(T * n_agents)
            old_logps_flat = logps_np.reshape(T * n_agents)
            advantages_flat = advantages.reshape(T * n_agents)
            returns_flat = returns.reshape(T * n_agents)
            agent_idx_flat = np.tile(np.arange(n_agents, dtype=np.int64), T)

            adv_mean = advantages_flat.mean()
            adv_std = advantages_flat.std() + 1e-8
            advantages_flat = (advantages_flat - adv_mean) / adv_std

            tensor_local_states = torch.tensor(local_states_flat, dtype=torch.float32, device=device)
            tensor_global_states = torch.tensor(global_states_rep, dtype=torch.float32, device=device)
            tensor_actions = torch.tensor(actions_flat, dtype=torch.int64, device=device)
            tensor_old_logps = torch.tensor(old_logps_flat, dtype=torch.float32, device=device)
            tensor_returns = torch.tensor(returns_flat, dtype=torch.float32, device=device)
            tensor_advantages = torch.tensor(advantages_flat, dtype=torch.float32, device=device)
            tensor_agent_idx = torch.tensor(agent_idx_flat, dtype=torch.long, device=device)

            indices = np.arange(dataset_size)
            for _ in range(ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, minibatch_size):
                    mb_idx = indices[start:start+minibatch_size]
                    mb_local = tensor_local_states[mb_idx]
                    mb_global = tensor_global_states[mb_idx]
                    mb_actions = tensor_actions[mb_idx]
                    mb_old_logps = tensor_old_logps[mb_idx]
                    mb_returns = tensor_returns[mb_idx]
                    mb_adv = tensor_advantages[mb_idx]
                    mb_agent_idx = tensor_agent_idx[mb_idx]

                    logits = model.forward_actor(mb_local)
                    dist = torch.distributions.Categorical(logits=logits)
                    mb_logps = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    vals_all = model.forward_critic(mb_global)
                    batch_range = torch.arange(len(mb_agent_idx), device=device)
                    mb_values = vals_all[batch_range, mb_agent_idx]

                    ratio = torch.exp(mb_logps - mb_old_logps)
                    surrogate1 = ratio * mb_adv
                    surrogate2 = torch.clamp(ratio, 1.0-clip_eps, 1.0+clip_eps) * mb_adv
                    policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))

                    value_loss = torch.mean((mb_returns - mb_values)**2)
                    loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

            avg_ep_reward = float(np.mean(ep_rewards)) if len(ep_rewards) > 0 else 0.0
            total_rewards.append(avg_ep_reward)
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs} | AvgEpReward {avg_ep_reward:.3f} | Steps {dataset_size}")

        return model, total_rewards

    finally:
        try:
            if hasattr(env, '_cleanup_prev_sim'):
                env._cleanup_prev_sim()
        except Exception:
            gc.collect()

if __name__ == "__main__":
    USE_BASILISK = True
    VIZ = True
    N_AGENTS = 3
    N_TARGETS = 6
    DECISION_DT = 10.0
    SAMPLE_DT = 1.0

    env = MultiSatScheduleEnv(n_agents=N_AGENTS,
                              n_targets=N_TARGETS,
                              decision_dt=DECISION_DT,
                              sample_dt=SAMPLE_DT,
                              fov_deg=30.0,
                              use_basilisk=USE_BASILISK,
                              viz=VIZ,
                              downlink_rate_mb_s=2.0,
                              wheel_boost_factor=1.5,
                              wheel_boost_duration=20.0,
                              coverage_coeff=3.0)

    device = 'cpu'
    model, rewards = train_mappo(env,
                                n_agents=N_AGENTS,
                                epochs=30,
                                batch_steps=512,
                                minibatch_size=256,
                                device=device)
    print("Training done. Last 5 avg rewards:", rewards[-5:])

    obs = env.reset()
    total = 0.0
    for step in range(60):
        local_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        logits = model.forward_actor(local_obs_tensor)
        actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy().tolist()
        obs, rewards_step, done, info = env.step(actions)
        total += float(np.sum(rewards_step))
        print(f"Step {step:02d} | actions {actions} | rewards {rewards_step} | team_coverage {info.get('team_coverage')} | buffers {info.get('data_buffer')}")

    try:
        env._cleanup_prev_sim()
    except Exception:
        gc.collect()

    print("Playback total team reward:", total)
