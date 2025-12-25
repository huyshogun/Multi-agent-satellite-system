import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# --- 1. BASILISK IMPORTS ---
try:
    from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, vizSupport, simIncludeRW
    from Basilisk.simulation import spacecraft, reactionWheelStateEffector
    BASILISK_AVAILABLE = True
    print("[INIT] Basilisk libraries loaded.")
except ImportError:
    print("[ERROR] Basilisk not found.")
    sys.exit(1)

# --- 2. MAPPO NETWORK (CTDE) ---
class MultiAgentActorCritic(nn.Module):
    def __init__(self, obs_dim, global_state_dim, action_dim, hidden=256):
        super().__init__()
        
        # ACTOR (Decentralized): Chỉ nhận Observation cục bộ của riêng Agent
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        
        # CRITIC (Centralized): Nhận Global State (Thông tin của toàn bộ đội hình)
        self.critic = nn.Sequential(
            nn.Linear(global_state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def act(self, obs):
        # Hàm này dùng khi chạy thực tế (Execution)
        return self.actor(obs)

    def evaluate(self, global_state):
        # Hàm này chỉ dùng khi huấn luyện (Training)
        return self.critic(global_state)

# --- 3. MULTI-AGENT ENVIRONMENT ---
class BasiliskMultiSatEnv:
    def __init__(self, n_sats=4, n_targets=50, decision_dt=10.0, viz=False):
        self.n_sats = n_sats
        self.n_targets = n_targets
        self.decision_dt = decision_dt
        self.viz = viz
        
        # Hằng số vật lý
        self.earth_radius = 6371e3
        self.mu = 398600.4418e9
        self.max_slew_rate = math.radians(3.0)
        self.fuel_max = 100.0
        
        # Action Space: [Target 0...N-1, Idle, Thrust_Up, Thrust_Down]
        self.action_dim = self.n_targets + 3
        
        # Observation Dim (Local): [Pos(3), Vel(3), Fuel(1)] = 7 (Rút gọn cho demo)
        self.obs_dim = 7 
        # Global State Dim: [Obs * n_sats] (Ghép tất cả obs của các vệ tinh lại)
        self.global_state_dim = self.obs_dim * self.n_sats 

        self._build_targets()
        self._init_simulator()

    def _build_targets(self):
        rng = np.random.RandomState(42)
        self.targets_ecef = []
        self.targets_lld = []
        for _ in range(self.n_targets):
            lat = rng.uniform(-60, 60) * macros.D2R
            lon = rng.uniform(-180, 180) * macros.D2R
            x = self.earth_radius * math.cos(lat) * math.cos(lon)
            y = self.earth_radius * math.cos(lat) * math.sin(lon)
            z = self.earth_radius * math.sin(lat)
            self.targets_ecef.append(np.array([x, y, z]))
            self.targets_lld.append([lat, lon, 0.0])

    def _cleanup(self):
        if hasattr(self, 'scSim'): self.scSim = None
        gc.collect()

    def _init_simulator(self):
        self._cleanup()
        self.scSim = SimulationBaseClass.SimBaseClass()
        self.scSim.SetProgressBar(False)
        self.processName = "dynProcess"
        self.taskName = "dynTask"
        self.dt = macros.sec2nano(1.0)
        
        dynProc = self.scSim.CreateNewProcess(self.processName)
        dynProc.addTask(self.scSim.CreateNewTask(self.taskName, self.dt))

        # --- Tạo 4 Vệ tinh (Constellation) ---
        self.sats = []     # Danh sách object vệ tinh Basilisk
        self.rw_effs = []  # Danh sách bộ bánh đà
        self.states = []   # Lưu trạng thái nội tại (Fuel, Attitude) cho từng vệ tinh
        
        gravFactory = simIncludeGravBody.gravBodyFactory()
        earth = gravFactory.createEarth()
        earth.isCentralBody = True
        self.planet_mu = earth.mu
        
        rwFactory = simIncludeRW.rwFactory()

        # Tạo đội hình Walker Delta (phân bố đều các mặt phẳng quỹ đạo)
        for i in range(self.n_sats):
            sc = spacecraft.Spacecraft()
            sc.ModelTag = f"Sat_{i}"
            self.scSim.AddModelToTask(self.taskName, sc)
            gravFactory.addBodiesTo(sc)
            
            # Gắn RW
            rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
            rwFactory.addToSpacecraft(f"RW_Array_{i}", rwStateEffector, sc)
            self.scSim.AddModelToTask(self.taskName, rwStateEffector)
            self.rw_effs.append(rwStateEffector)
            
            # Cài đặt quỹ đạo (Lệch nhau góc RAAN để phủ toàn cầu)
            oe = orbitalMotion.ClassicElements()
            oe.a = 7000e3
            oe.e = 0.001
            oe.i = 45.0 * macros.D2R
            oe.Omega = (i * 360.0 / self.n_sats) * macros.D2R # RAAN lệch đều
            oe.omega = 0.0
            oe.f = 0.0
            rN, vN = orbitalMotion.elem2rv(self.planet_mu, oe)
            
            sc.hub.r_CN_NInit = np.array(rN)
            sc.hub.v_CN_NInit = np.array(vN)
            
            self.sats.append(sc)
            
            # Khởi tạo trạng thái nội tại (Fuel, Hướng nhìn)
            bore_vec = -np.array(rN) / np.linalg.norm(rN)
            self.states.append({
                'fuel': self.fuel_max,
                'bore_vec': bore_vec,
                'captured_targets': set() # Mỗi vệ tinh nhớ nó đã chụp gì
            })

        # Viz Setup (ĐÃ SỬA LỖI)
        if self.viz:
            try:
                viz = vizSupport.enableUnityVisualization(
                    self.scSim, self.taskName, self.sats, # Truyền list vệ tinh vào
                    saveFile="mappo_constellation.bin"
                )
                
                # [FIXED] Bọc hàm tạo Cone trong try-except để tránh lỗi phiên bản
                try:
                    if hasattr(vizSupport, 'createCone'):
                        for i in range(self.n_sats):
                            vizSupport.createCone(viz, fromBodyName=f"Sat_{i}", fov=30*macros.D2R, color=[0, 1, 0, 0.5])
                except Exception:
                    pass # Bỏ qua nếu không hỗ trợ vẽ nón
                
                # [FIXED] Bọc hàm tạo GroundStation
                try:
                    for k, lld in enumerate(self.targets_lld):
                        if hasattr(vizSupport, 'createGroundStation'):
                            vizSupport.createGroundStation(viz, groundPos=lld, stationName=f"TGT_{k}", color=[1, 0, 0, 1])
                except Exception: 
                    pass
                
                self.vizObj = viz
            except Exception as e:
                print(f"Viz Error (Ignored): {e}")
                self.vizObj = None

        self.scSim.InitializeSimulation()
        self.sim_time = 0.0

    def reset(self):
        self._init_simulator()
        return self._get_all_obs() # Trả về list các obs

    def _get_all_obs(self):
        obs_list = []
        for i, sat in enumerate(self.sats):
            r = np.array(sat.hub.r_CN_NInit).flatten() / 1e7
            v = np.array(sat.hub.v_CN_NInit).flatten() / 1e4
            f = [self.states[i]['fuel'] / self.fuel_max]
            # Obs cục bộ: [Pos(3), Vel(3), Fuel(1)] = 7
            obs = np.concatenate([r, v, f]).astype(np.float32)
            obs_list.append(obs)
        return obs_list

    def step(self, actions):
        # actions: list gồm n_sats hành động
        rewards = np.zeros(self.n_sats)
        
        # 1. Xử lý hành động cho từng vệ tinh
        for i, action in enumerate(actions):
            sat = self.sats[i]
            state = self.states[i]
            act = int(action)
            
            # Lấy vector vận tốc hiện tại
            v_curr = np.array(sat.hub.v_CN_NInit).flatten()
            v_dir = v_curr / (np.linalg.norm(v_curr) + 1e-9)
            
            # Logic Thrust (Động cơ)
            if act == self.n_targets + 1 and state['fuel'] > 0: # Thrust Up
                sat.hub.v_CN_NInit = v_curr + v_dir * 10.0
                state['fuel'] -= 1.0
                rewards[i] -= 0.5
            elif act == self.n_targets + 2 and state['fuel'] > 0: # Thrust Down
                sat.hub.v_CN_NInit = v_curr - v_dir * 10.0
                state['fuel'] -= 1.0
                rewards[i] -= 0.5
            
            # Logic Slew & Capture
            if 0 <= act < self.n_targets:
                tgt_pos = self.targets_ecef[act]
                sat_pos = np.array(sat.hub.r_CN_NInit).flatten()
                
                req_vec = tgt_pos - sat_pos
                dist = np.linalg.norm(req_vec)
                req_vec /= (dist + 1e-9)
                
                cos_theta = np.dot(state['bore_vec'], req_vec)
                angle = math.acos(np.clip(cos_theta, -1.0, 1.0))
                
                t_slew = angle / self.max_slew_rate
                
                if t_slew < self.decision_dt:
                    state['bore_vec'] = req_vec # Cập nhật hướng nhìn
                    if dist < 3000e3:
                        rewards[i] += 10.0
                        state['captured_targets'].add(act)
                else:
                    rewards[i] -= 0.1 # Phạt vì xoay chậm

        # 2. Chạy mô phỏng vật lý
        stop_time = self.sim_time + self.decision_dt
        self.scSim.ConfigureStopTime(int(macros.sec2nano(stop_time)))
        self.scSim.ExecuteSimulation()
        self.sim_time = stop_time
        
        # 3. Check done
        done = False
        if self.sim_time > 5400: done = True
        
        next_obs = self._get_all_obs()
        return next_obs, rewards, [done]*self.n_sats, {}

# --- 4. MAPPO TRAINER ---
def train_mappo():
    N_SATS = 4
    N_TARGETS = 50
    OBS_DIM = 7
    ACT_DIM = N_TARGETS + 3
    # Global State = Ghép obs của tất cả vệ tinh lại
    GLOBAL_STATE_DIM = OBS_DIM * N_SATS 
    
    env = BasiliskMultiSatEnv(n_sats=N_SATS, n_targets=N_TARGETS, viz=True)
    
    # Chia sẻ tham số (Parameter Sharing): Tất cả vệ tinh dùng chung 1 mạng MAPPO
    mappo_agent = MultiAgentActorCritic(OBS_DIM, GLOBAL_STATE_DIM, ACT_DIM)
    optimizer = optim.Adam(mappo_agent.parameters(), lr=1e-3)
    
    print("--- BẮT ĐẦU HUẤN LUYỆN MAPPO (4 Vệ tinh) ---")
    
    for epoch in range(10): # Demo 10 epochs
        obs_list = env.reset() # List [Sat1_obs, Sat2_obs, ...]
        ep_rewards = np.zeros(N_SATS)
        done = False
        
        # Buffer lưu trữ
        batch_obs = []
        batch_global_state = []
        batch_actions = []
        batch_rewards = []
        
        while not done:
            # 1. Decentralized Execution
            actions_t = []
            
            # Tạo Global State cho Critic
            global_state_t = np.concatenate(obs_list) 
            
            for i in range(N_SATS):
                obs_tensor = torch.FloatTensor(obs_list[i]).unsqueeze(0)
                logits = mappo_agent.act(obs_tensor) # Actor dùng obs cục bộ
                
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                actions_t.append(action.item())
            
            # 2. Môi trường phản hồi
            next_obs_list, rewards, dones, _ = env.step(actions_t)
            
            # Lưu vào buffer
            batch_obs.append(obs_list)
            batch_global_state.append(global_state_t)
            batch_actions.append(actions_t)
            batch_rewards.append(rewards)
            
            obs_list = next_obs_list
            ep_rewards += rewards
            done = all(dones)

        # 3. Centralized Training
        # Tính Returns
        returns_batch = np.zeros_like(batch_rewards)
        running_add = np.zeros(N_SATS)
        for t in reversed(range(len(batch_rewards))):
            running_add = batch_rewards[t] + 0.99 * running_add
            returns_batch[t] = running_add
            
        # Update
        optimizer.zero_grad()
        loss = 0
        
        for t in range(len(batch_obs)):
            g_state = torch.FloatTensor(batch_global_state[t]).unsqueeze(0)
            target_value = torch.FloatTensor([np.mean(returns_batch[t])]).unsqueeze(0)
            
            # Critic Loss
            predicted_value = mappo_agent.evaluate(g_state)
            critic_loss = nn.MSELoss()(predicted_value, target_value)
            
            # Actor Loss
            actor_loss_sum = 0
            for i in range(N_SATS):
                obs = torch.FloatTensor(batch_obs[t][i]).unsqueeze(0)
                act = torch.tensor([batch_actions[t][i]])
                ret = returns_batch[t][i]
                
                advantage = ret - predicted_value.item()
                
                logits = mappo_agent.act(obs)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(act)
                
                actor_loss_sum += -log_prob * advantage
            
            loss += critic_loss + actor_loss_sum
            
        loss.backward()
        optimizer.step()
        
        avg_r = np.mean(ep_rewards)
        print(f"Epoch {epoch+1} | Avg Team Reward: {avg_r:.1f}")

    print("--- XONG ---")
    print("Mở file 'mappo_constellation.bin' trong Vizard để xem 4 vệ tinh phối hợp.")

if __name__ == "__main__":
    train_mappo()