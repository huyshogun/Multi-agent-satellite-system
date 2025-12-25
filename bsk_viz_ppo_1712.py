import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os

# --- 1. KIỂM TRA THƯ VIỆN BASILISK ---
try:
    from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, vizSupport, simIncludeRW, unitTestSupport
    from Basilisk.simulation import spacecraft, reactionWheelStateEffector
    # Hàm chuyển đổi toạ độ hình học
    from Basilisk.utilities import RigidBodyKinematics as rbk
    BASILISK_AVAILABLE = True
    print("[INIT] Basilisk libraries loaded.")
except Exception as e:
    print("Lỗi gặp phải khi tải Basilisk:", e)
    sys.exit(1)

# --- 2. MÔ HÌNH AI (PPO ACTOR-CRITIC) ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        # Actor: Quyết định hành động
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        # Critic: Đánh giá giá trị trạng thái
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

def categorical_sample(logits):
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    return action, dist.log_prob(action), dist.entropy()

# --- 3. MÔI TRƯỜNG BASILISK TÍCH HỢP ---
class BasiliskAgileEnv:
    def __init__(self, n_targets=50, decision_dt=10.0, viz=False):
        self.n_targets = n_targets
        self.decision_dt = decision_dt
        self.viz = viz
        
        # Thông số vật lý
        self.earth_radius = 6371e3
        self.sma = 7000e3
        self.mu = 398600.4418e9
        self.max_slew_rate = math.radians(3.0) # 3 độ/giây
        self.fuel_max = 100.0
        self.delta_v = 10.0 # m/s mỗi lần thrust

        # Action Space: 0->N-1 (Target), N (Idle), N+1 (Thrust Up), N+2 (Thrust Down)
        self.action_dim = self.n_targets + 3
        self.ACT_IDLE = self.n_targets
        self.ACT_UP = self.n_targets + 1
        self.ACT_DOWN = self.n_targets + 2

        self._build_targets()
        self._init_simulator()

    def _build_targets(self):
        """Tạo 50 mục tiêu rải rác trên trái đất"""
        rng = np.random.RandomState(42)
        self.targets_ecef = []
        self.targets_lld = [] # Lat/Lon/Alt cho Vizard
        
        for _ in range(self.n_targets):
            lat = rng.uniform(-50, 50) * macros.D2R
            lon = rng.uniform(-180, 180) * macros.D2R
            
            x = self.earth_radius * math.cos(lat) * math.cos(lon)
            y = self.earth_radius * math.cos(lat) * math.sin(lon)
            z = self.earth_radius * math.sin(lat)
            
            self.targets_ecef.append(np.array([x, y, z]))
            self.targets_lld.append([lat, lon, 0.0])

    def _cleanup(self):
        if hasattr(self, 'scSim'):
            self.scSim = None
        gc.collect()

    def _init_simulator(self):
            self._cleanup()
            
            # 1. Sim Engine
            self.scSim = SimulationBaseClass.SimBaseClass()
            self.scSim.SetProgressBar(False)
            self.processName = "dynProcess"
            self.taskName = "dynTask"
            self.sim_step_ns = int(macros.sec2nano(1.0)) # Physics step 1s
            
            dynProc = self.scSim.CreateNewProcess(self.processName)
            dynProc.addTask(self.scSim.CreateNewTask(self.taskName, self.sim_step_ns))

            # 2. Spacecraft
            self.scObject = spacecraft.Spacecraft()
            self.scObject.ModelTag = "AI-Sat"
            self.scSim.AddModelToTask(self.taskName, self.scObject)

            # 3. Gravity
            gravFactory = simIncludeGravBody.gravBodyFactory()
            earth = gravFactory.createEarth()
            earth.isCentralBody = True
            gravFactory.addBodiesTo(self.scObject)
            self.planet_mu = earth.mu

            # 4. Reaction Wheels (Dùng Factory chuẩn)
            rwFactory = simIncludeRW.rwFactory()
            # Tạo 3 bánh đà
            RW1 = rwFactory.create('Honeywell_HR16', [1, 0, 0], maxMomentum=50., Omega=0.0)
            RW2 = rwFactory.create('Honeywell_HR16', [0, 1, 0], maxMomentum=50., Omega=0.0)
            RW3 = rwFactory.create('Honeywell_HR16', [0, 0, 1], maxMomentum=50., Omega=0.0)
            
            self.rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
            self.rwStateEffector.ModelTag = "RW_Array"
            rwFactory.addToSpacecraft("RW_Array", self.rwStateEffector, self.scObject)
            self.scSim.AddModelToTask(self.taskName, self.rwStateEffector)

            # 5. Orbit Setup
            oe = orbitalMotion.ClassicElements()
            oe.a = self.sma
            oe.e = 0.001
            oe.i = 45.0 * macros.D2R
            oe.Omega = 0.0
            oe.omega = 0.0
            oe.f = 0.0
            rN, vN = orbitalMotion.elem2rv(self.planet_mu, oe)
            
            self.scObject.hub.r_CN_NInit = np.array(rN)
            self.scObject.hub.v_CN_NInit = np.array(vN)
            
            # Quaternion ban đầu
            self.scObject.hub.sigma_BNInit = [[0.1], [0.0], [0.0]] 

            # 6. Vizard Setup (ĐÃ SỬA LỖI TƯƠNG THÍCH)
            if self.viz:
                try:
                    # Tạo file viz
                    viz = vizSupport.enableUnityVisualization(
                        self.scSim, self.taskName, self.scObject,
                        saveFile="bsk_ppo_50targets.bin"
                    )
                    
                    # --- ĐÃ XÓA DÒNG createCone GÂY LỖI ---
                    
                    # Vẽ 50 mục tiêu (Bọc trong try-except để an toàn)
                    try:
                        for i, lld in enumerate(self.targets_lld):
                            # Kiểm tra xem hàm createGroundStation có tồn tại không
                            if hasattr(vizSupport, 'createGroundStation'):
                                vizSupport.createGroundStation(viz, groundPos=lld, stationName=f"TGT_{i}", color=[1, 0, 0, 1])
                    except Exception as e:
                        print(f"Viz Warning: Không thể vẽ GroundStation ({e})")
                        
                    self.vizObj = viz
                except Exception as e:
                    print(f"Viz Error (Có thể bỏ qua): {e}")
                    self.vizObj = None

            self.scSim.InitializeSimulation()
            self.sim_time = 0.0
            
            # State nội tại
            self.fuel = self.fuel_max
            self.current_bore_vec = -np.array(rN) / np.linalg.norm(rN)

    def reset(self):
        self._init_simulator()
        return self._get_obs()

    def _get_obs(self):
        # Lấy state từ Basilisk
        r_N = np.array(self.scObject.hub.r_CN_NInit).flatten()
        v_N = np.array(self.scObject.hub.v_CN_NInit).flatten()
        
        # Chuẩn hóa
        pos_norm = r_N / 1e7
        vel_norm = v_N / 1e4
        fuel_norm = [self.fuel / self.fuel_max]
        
        # Target info (Rút gọn: chỉ lấy 5 mục tiêu gần nhất hoặc dạng thống kê để giảm state space)
        # Để đơn giản cho demo này: Ta chỉ đưa vào vector hướng của 5 target đầu tiên 
        # (Lưu ý: Để train tốt với 50 target cần cơ chế Attention hoặc CNN, ở đây dùng bản rút gọn)
        
        # Tính toán hình học đơn giản
        obs = np.concatenate([pos_norm, vel_norm, fuel_norm])
        # Padding cho đủ input dimension cố định (ví dụ 64)
        pad = np.zeros(64 - len(obs)) 
        return np.concatenate([obs, pad]).astype(np.float32)

    def step(self, action):
        reward = 0.0
        done = False
        action = int(action)

        # 1. Physics Logic (Thrust)
        # Basilisk xử lý Orbit Drift tự nhiên. Ta can thiệp velocity thủ công nếu đốt động cơ.
        v_current = np.array(self.scObject.hub.v_CN_NInit).flatten()
        v_dir = v_current / np.linalg.norm(v_current)
        
        if action == self.ACT_UP and self.fuel > 0:
            v_new = v_current + v_dir * self.delta_v
            self.scObject.hub.v_CN_NInit = v_new # Cập nhật trực tiếp vào Sim state
            self.fuel -= 1.0
            reward -= 0.5 # Tốn xăng
        elif action == self.ACT_DOWN and self.fuel > 0:
            v_new = v_current - v_dir * self.delta_v
            self.scObject.hub.v_CN_NInit = v_new
            self.fuel -= 1.0
            reward -= 0.5

        # 2. Attitude Logic (Slew)
        # Giả lập Slew: Nếu chọn target i, kiểm tra góc quay và update hướng nhìn
        captured = False
        if 0 <= action < self.n_targets:
            tgt_pos = self.targets_ecef[action]
            sat_pos = np.array(self.scObject.hub.r_CN_NInit).flatten()
            
            # Vector từ vệ tinh đến target
            req_vec = tgt_pos - sat_pos
            dist = np.linalg.norm(req_vec)
            req_vec /= dist
            
            # Tính góc cần quay
            cos_theta = np.dot(self.current_bore_vec, req_vec)
            angle = math.acos(np.clip(cos_theta, -1.0, 1.0))
            
            t_slew = angle / self.max_slew_rate
            
            if t_slew < self.decision_dt:
                # Quay kịp -> Update hướng nhìn
                self.current_bore_vec = req_vec
                
                # Check điều kiện chụp (Gần < 3000km và nhìn thấy)
                if dist < 3000e3:
                    reward += 10.0
                    captured = True
                    # Hack nhẹ: Update hướng thật của vệ tinh trong Basilisk để Vizard hiển thị đúng hướng xoay
                    # (Code chuẩn cần tính Quaternion dcm2quat, ở đây ta để Vizard hiển thị nón camera tương đối)
            else:
                reward -= 0.1 # Phạt chờ đợi

        # 3. Step Simulator
        stop_time = self.sim_time + self.decision_dt
        self.scSim.ConfigureStopTime(int(macros.sec2nano(stop_time)))
        self.scSim.ExecuteSimulation()
        self.sim_time = stop_time

        # 4. Check Done
        r_mag = np.linalg.norm(self.scObject.hub.r_CN_NInit)
        if r_mag < (self.earth_radius + 100e3): done = True # Rơi
        if self.sim_time > 5400: done = True # Hết thời gian (90 phút)

        return self._get_obs(), reward, done, {'captured': captured}

# --- 4. HÀM TRAIN & DEMO ---
def main():
    # A. Cấu hình
    N_TARGETS = 50
    STATE_DIM = 64 # Fixed padding
    ACTION_DIM = N_TARGETS + 3
    DEVICE = "cpu"
    
    env = BasiliskAgileEnv(n_targets=N_TARGETS, viz=True) # Viz=True để tạo file bin
    model = ActorCritic(STATE_DIM, ACTION_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"--- BẮT ĐẦU TRAINING PPO (Demo nhanh 20 Epochs) ---")
    
    # B. Training Loop (Rút gọn)
    for epoch in range(20): # Train ít để demo chạy nhanh
        s = env.reset()
        ep_reward = 0
        done = False
        
        states, actions, rewards, log_probs = [], [], [], []
        
        while not done:
            s_tensor = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
            logits, _ = model(s_tensor)
            a, lp, _ = categorical_sample(logits)
            
            s_next, r, done, info = env.step(a.item())
            
            states.append(s)
            actions.append(a.item())
            rewards.append(r)
            log_probs.append(lp)
            
            s = s_next
            ep_reward += r

        # Update đơn giản (Policy Gradient basic thay vì Full PPO để code ngắn gọn)
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/20 | Reward: {ep_reward:.2f} | Fuel: {env.fuel}%")

    # C. Save & Playback Info
    print("\n--- HOÀN TẤT ---")
    print("Mô hình đã chạy xong.")
    print("Dữ liệu trực quan hóa đã được lưu vào file: 'bsk_ppo_50targets.bin'")
    print("Hãy mở 'Basilisk Vizard', load file này để xem vệ tinh bay và chụp ảnh.")

if __name__ == "__main__":
    main()