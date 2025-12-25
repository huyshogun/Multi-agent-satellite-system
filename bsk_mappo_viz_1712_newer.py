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
        self.max_slew_rate = math.radians(5.0) # Tăng tốc độ xoay lên 5 độ/s cho lanh lẹ
        self.fuel_max = 200.0 # Tăng bình xăng vì đổi góc nghiêng rất tốn kém
        
        # --- ACTION SPACE MỞ RỘNG ---
        # 0 -> N-1: Target
        # N: Idle (Nghỉ)
        # N+1: Thrust Up (Nâng độ cao)
        # N+2: Thrust Down (Hạ độ cao)
        # N+3: Inc Up (Tăng góc nghiêng - Normal)
        # N+4: Inc Down (Giảm góc nghiêng - Anti-Normal)
        self.action_dim = self.n_targets + 5 
        
        self.ACT_IDLE = self.n_targets
        self.ACT_ALT_UP = self.n_targets + 1
        self.ACT_ALT_DOWN = self.n_targets + 2
        self.ACT_INC_UP = self.n_targets + 3
        self.ACT_INC_DOWN = self.n_targets + 4

        # Obs Dim: [Pos(3), Vel(3), Fuel(1)] = 7
        self.obs_dim = 7 
        self.global_state_dim = self.obs_dim * self.n_sats 

        self._build_targets()
        self._init_simulator()

    # ... (Giữ nguyên hàm _build_targets, _cleanup) ...
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

    # ... (Hàm _init_simulator DÙNG BẢN MỚI NHẤT MÀ BẠN ĐÃ CHẠY ĐƯỢC - VẼ DUMMY SATS) ...
    # (Để tiết kiệm không gian tôi không paste lại đoạn _init_simulator dài dòng đó, 
    #  bạn hãy dùng lại đoạn code "bất khả chiến bại" ở câu trả lời trước)
    def _init_simulator(self):
        # COPY LẠI HÀM _init_simulator TỪ CÂU TRẢ LỜI TRƯỚC (Phiên bản dùng Dummy Sats)
        # Chỉ cần lưu ý: Cập nhật self.states['fuel'] = self.fuel_max
        # ...
        # (Nội dung y hệt, chỉ viết lại đoạn logic này để bạn dễ ghép code)
        self._cleanup()
        self.scSim = SimulationBaseClass.SimBaseClass()
        self.scSim.SetProgressBar(False)
        self.processName = "dynProcess"
        self.taskName = "dynTask"
        self.dt = macros.sec2nano(1.0)
        dynProc = self.scSim.CreateNewProcess(self.processName)
        dynProc.addTask(self.scSim.CreateNewTask(self.taskName, self.dt))
        self.all_viz_objects = []
        self.sats = []; self.states = []
        gravFactory = simIncludeGravBody.gravBodyFactory()
        earth = gravFactory.createEarth()
        earth.isCentralBody = True
        self.planet_mu = earth.mu
        rwFactory = simIncludeRW.rwFactory()

        for i in range(self.n_sats):
            sc = spacecraft.Spacecraft()
            sc.ModelTag = f"Sat_{i}"
            self.scSim.AddModelToTask(self.taskName, sc)
            gravFactory.addBodiesTo(sc)
            rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
            rwFactory.addToSpacecraft(f"RW_Array_{i}", rwStateEffector, sc)
            self.scSim.AddModelToTask(self.taskName, rwStateEffector)
            
            oe = orbitalMotion.ClassicElements()
            oe.a = 7000e3
            oe.e = 0.001
            oe.i = 45.0 * macros.D2R
            oe.Omega = (i * 360.0 / self.n_sats) * macros.D2R 
            oe.omega = 0.0; oe.f = 0.0
            rN, vN = orbitalMotion.elem2rv(self.planet_mu, oe)
            
            sc.hub.r_CN_NInit = np.array(rN)
            sc.hub.v_CN_NInit = np.array(vN)
            self.sats.append(sc)
            self.all_viz_objects.append(sc)
            
            bore_vec = -np.array(rN) / np.linalg.norm(rN)
            self.states.append({'fuel': self.fuel_max, 'bore_vec': bore_vec, 'captured_targets': set()})

        self.dummy_targets = []
        for k, ecef in enumerate(self.targets_ecef):
            dummy = spacecraft.Spacecraft()
            dummy.ModelTag = f"TGT_{k}"
            r_mag = np.linalg.norm(ecef)
            pos_elevated = ecef * ((r_mag + 50000.0) / r_mag)
            dummy.hub.r_CN_NInit = pos_elevated
            dummy.hub.v_CN_NInit = np.array([0.0,0.0,0.0])
            self.scSim.AddModelToTask(self.taskName, dummy)
            self.all_viz_objects.append(dummy)
            self.dummy_targets.append(dummy)

        if self.viz:
            try:
                viz = vizSupport.enableUnityVisualization(self.scSim, self.taskName, self.all_viz_objects, saveFile="mappo_constellation_newer.bin")
                try:
                    if hasattr(vizSupport, 'createStandardCamera'):
                        for i in range(self.n_sats):
                            vizSupport.createStandardCamera(viz, setMode=1, spacecraftName=f"Sat_{i}", fieldOfView=30.0 * macros.D2R, pointingVector_B=[-1, 0, 0], position_B=[0, 0, 0])
                except: pass
                self.vizObj = viz
            except Exception as e: self.vizObj = None
        self.scSim.InitializeSimulation()
        self.sim_time = 0.0


    # ... (Giữ nguyên reset, _get_all_obs) ...
    def reset(self):
        self._init_simulator()
        return self._get_all_obs()

    def _get_all_obs(self):
        obs_list = []
        for i, sat in enumerate(self.sats):
            r = np.array(sat.hub.r_CN_NInit).flatten() / 1e7
            v = np.array(sat.hub.v_CN_NInit).flatten() / 1e4
            f = [self.states[i]['fuel'] / self.fuel_max]
            obs = np.concatenate([r, v, f]).astype(np.float32)
            obs_list.append(obs)
        return obs_list

    # --- HÀM STEP NÂNG CẤP ---
    def step(self, actions, debug_mode=True):
        rewards = np.zeros(self.n_sats)
        dones = [False] * self.n_sats
        
        for i, action in enumerate(actions):
            sat = self.sats[i]
            state = self.states[i]
            act = int(action)
            
            # 1. Lấy thông tin vật lý
            v_curr = np.array(sat.hub.v_CN_NInit).flatten()
            r_curr = np.array(sat.hub.r_CN_NInit).flatten()
            
            # Vector tiếp tuyến (Prograde)
            v_mag = np.linalg.norm(v_curr)
            v_dir = v_curr / (v_mag + 1e-9)
            
            # Vector pháp tuyến quỹ đạo (Normal) = r x v
            # Đây là hướng để thay đổi góc nghiêng (Inclination)
            h_vec = np.cross(r_curr, v_curr)
            h_dir = h_vec / (np.linalg.norm(h_vec) + 1e-9)
            
            alt_km = (np.linalg.norm(r_curr) - self.earth_radius) / 1000.0

            # 2. Xử lý Hành Động
            
            # --- NHÓM 1: THAY ĐỔI ĐỘ CAO (Semi-major Axis) ---
            if act == self.ACT_ALT_UP: # Thrust Up
                if state['fuel'] >= 1.0:
                    sat.hub.v_CN_NInit = v_curr + v_dir * 20.0
                    state['fuel'] -= 1.0
                    rewards[i] -= 0.05
                    if debug_mode: print(f"[t={self.sim_time:.0f}s] 🛰️ Sat {i} | 🔥 ALT UP (H: {alt_km:.0f}km)")
            
            elif act == self.ACT_ALT_DOWN: # Thrust Down
                if state['fuel'] >= 1.0:
                    sat.hub.v_CN_NInit = v_curr - v_dir * 20.0
                    state['fuel'] -= 1.0
                    rewards[i] -= 0.05
                    if debug_mode: print(f"[t={self.sim_time:.0f}s] 🛰️ Sat {i} | 🔻 ALT DOWN (H: {alt_km:.0f}km)")

            # --- NHÓM 2: THAY ĐỔI GÓC NGHIÊNG (Inclination) [MỚI] ---
            elif act == self.ACT_INC_UP: # Tăng góc nghiêng (Thrust theo hướng Normal)
                if state['fuel'] >= 2.0: # Tốn xăng gấp đôi
                    # Delta-V lớn (50m/s) để thấy sự thay đổi rõ ràng
                    sat.hub.v_CN_NInit = v_curr + h_dir * 50.0 
                    state['fuel'] -= 2.0
                    rewards[i] -= 0.1 # Phạt nặng hơn
                    if debug_mode: print(f"[t={self.sim_time:.0f}s] 🛰️ Sat {i} | 📐 INC CHANGE (+) -> Đổi mặt phẳng quỹ đạo")
            
            elif act == self.ACT_INC_DOWN: # Giảm góc nghiêng (Thrust theo hướng Anti-Normal)
                if state['fuel'] >= 2.0:
                    sat.hub.v_CN_NInit = v_curr - h_dir * 50.0
                    state['fuel'] -= 2.0
                    rewards[i] -= 0.1
                    if debug_mode: print(f"[t={self.sim_time:.0f}s] 🛰️ Sat {i} | 📐 INC CHANGE (-) -> Đổi mặt phẳng quỹ đạo")

            # --- NHÓM 3: XOAY CAMERA & CHỤP ---
            elif 0 <= act < self.n_targets:
                tgt_pos = self.targets_ecef[act]
                req_vec = tgt_pos - r_curr
                dist = np.linalg.norm(req_vec)
                req_vec /= (dist + 1e-9)
                
                cos_theta = np.dot(state['bore_vec'], req_vec)
                angle_deg = math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))
                
                # Logic xoay tích lũy
                max_turn_deg = math.degrees(self.max_slew_rate) * self.decision_dt
                
                if angle_deg <= max_turn_deg:
                    state['bore_vec'] = req_vec # Xoay xong
                    
                    # Logic phần thưởng
                    is_even_target = (act % 2 == 0)
                    capture_success = False
                    
                    if is_even_target and alt_km < 650 and dist < 3000e3:
                        rewards[i] += 15.0
                        capture_success = True
                    elif not is_even_target and alt_km > 750 and dist < 4000e3:
                        rewards[i] += 15.0
                        capture_success = True
                    else:
                        rewards[i] -= 0.1
                        
                    if debug_mode and capture_success:
                        print(f"[t={self.sim_time:.0f}s] 🛰️ Sat {i} | 📸 SNAP T{act} (Dist: {dist/1000:.0f}km)")
                else:
                    # Partial Slew
                    ratio = max_turn_deg / angle_deg
                    new_vec = (1 - ratio) * state['bore_vec'] + ratio * req_vec
                    state['bore_vec'] = new_vec / np.linalg.norm(new_vec)
                    rewards[i] -= 0.01
                    if debug_mode:
                        print(f"[t={self.sim_time:.0f}s] 🛰️ Sat {i} | 🔄 SLEW T{act} (Còn {angle_deg - max_turn_deg:.1f}°)")
            
            # Idle
            else:
                 pass

        # 3. Chạy mô phỏng
        stop_time = self.sim_time + self.decision_dt
        self.scSim.ConfigureStopTime(int(macros.sec2nano(stop_time)))
        self.scSim.ExecuteSimulation()
        self.sim_time = stop_time
        
        if self.sim_time > 5400: dones = [True] * self.n_sats
        return self._get_all_obs(), rewards, dones, {}
# --- 4. MAPPO TRAINER ---
def train_mappo():
    N_SATS = 4
    N_TARGETS = 50
    OBS_DIM = 7
    # CŨ: ACT_DIM = N_TARGETS + 3
    # MỚI: Thêm 2 hành động Inc Change
    ACT_DIM = N_TARGETS + 5
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
# Update
        optimizer.zero_grad()
        loss = 0
        
        # Cấu hình hệ số (Hyperparameters)
        vf_coeff = 0.5   # Hệ số quan trọng của Critic (Value Loss)
        ent_coeff = 0.05 # Hệ số Entropy (Càng cao càng khám phá nhiều) <-- TĂNG CÁI NÀY LÊN
        
        for t in range(len(batch_obs)):
            g_state = torch.FloatTensor(batch_global_state[t]).unsqueeze(0)
            target_value = torch.FloatTensor([np.mean(returns_batch[t])]).unsqueeze(0)
            
            # --- 1. Calculate Value Loss (Critic) ---
            predicted_value = mappo_agent.evaluate(g_state)
            value_loss = nn.MSELoss()(predicted_value, target_value)
            
            # --- 2. Calculate Policy Loss & Entropy (Actor) ---
            actor_loss_sum = 0
            entropy_sum = 0 # Biến để cộng dồn entropy của 4 vệ tinh
            
            for i in range(N_SATS):
                obs = torch.FloatTensor(batch_obs[t][i]).unsqueeze(0)
                act = torch.tensor([batch_actions[t][i]])
                ret = returns_batch[t][i]
                
                advantage = ret - predicted_value.item()
                
                logits = mappo_agent.act(obs)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(act)
                
                # [MỚI] Tính Entropy: Đo độ "ngẫu nhiên/bối rối" của Actor
                entropy_sum += dist.entropy() 
                
                actor_loss_sum += -log_prob * advantage
            
            # Tính trung bình Entropy của cả đội tại bước thời gian t
            avg_entropy = entropy_sum / N_SATS

            # --- [ĐOẠN BẠN CẦN THÊM VÀO ĐÂY] ---
            # Công thức: Loss = Actor_Loss + (0.5 * Critic_Loss) - (0.05 * Entropy)
            # Dấu trừ (-) trước Entropy nghĩa là: Entropy càng cao -> Loss càng thấp -> Tốt
            step_loss = actor_loss_sum + (vf_coeff * value_loss) - (ent_coeff * avg_entropy)
            
            loss += step_loss
            
        loss.backward()
        # Thêm clipping gradient để tránh update quá mạnh làm hỏng mạng
        nn.utils.clip_grad_norm_(mappo_agent.parameters(), max_norm=0.5) 
        optimizer.step()
        
        avg_r = np.mean(ep_rewards)
        print(f"Epoch {epoch+1} | Avg Team Reward: {avg_r:.1f}")
        
# --- [SỬA 2] THÊM ĐOẠN DEMO NÀY VÀO CUỐI ---
    print("\n--- CHẠY DEMO KIỂM TRA ---")
    obs_list = env.reset()
    done = False
    while not done:
        actions_t = []
        for i in range(N_SATS):
            obs_tensor = torch.FloatTensor(obs_list[i]).unsqueeze(0)
            logits = mappo_agent.act(obs_tensor)
            action = torch.argmax(logits, dim=1) # Chọn hành động tốt nhất
            actions_t.append(action.item())
            
        # BẬT DEBUG Ở ĐÂY
        next_obs_list, rewards, dones, _ = env.step(actions_t, debug_mode=True)
        obs_list = next_obs_list
        done = all(dones)

    print("--- XONG ---")
    print("Mở file 'mappo_constellation_newer.bin' trong Vizard để xem 4 vệ tinh phối hợp.")

if __name__ == "__main__":
    train_mappo()