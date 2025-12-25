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
        # Actor: Obs cục bộ -> Action
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        # Critic: Global State -> Value
        self.critic = nn.Sequential(
            nn.Linear(global_state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def act(self, obs):
        return self.actor(obs)

    def evaluate(self, global_state):
        return self.critic(global_state)

# --- 3. MULTI-AGENT ENVIRONMENT (NÂNG CẤP) ---
class BasiliskFullMissionEnv:
    def __init__(self, n_sats=4, n_targets=50, decision_dt=10.0, viz=False):
        self.n_sats = n_sats
        self.n_targets = n_targets
        self.decision_dt = decision_dt
        self.viz = viz
        
        # Hằng số vật lý & Tài nguyên
        self.earth_radius = 6371e3
        self.mu = 398600.4418e9
        self.max_slew_rate = math.radians(5.0) 
        self.fuel_max = 200.0
        self.batt_max = 100.0  # Dung lượng pin tối đa
        self.buffer_max = 5    # Chỉ chứa tối đa 5 ảnh, phải về xả bớt mới chụp tiếp được
        
        # --- CẤU HÌNH TRẠM MẶT ĐẤT (GS) ---
        # 4 Trạm: Hà Nội, New York, London, Sydney
        self.gs_coords = [
            (21.0285, 105.8542),   # Hanoi
            (40.7128, -74.0060),   # New York
            (51.5074, -0.1278),    # London
            (-33.8688, 151.2093)   # Sydney
        ]
        self.n_gs = len(self.gs_coords)
        
        # --- ACTION SPACE ---
        # 0 -> 49: Targets (Chụp)
        # 50 -> 53: Ground Stations (Downlink)
        # 54: Charge (Sạc pin)
        # 55: Thrust Up (Nâng cao)
        # 56: Thrust Down (Hạ thấp)
        # 57: Inc Up (Đổi góc nghiêng +)
        # 58: Inc Down (Đổi góc nghiêng -)
        
        self.ACT_CHARGE = self.n_targets + self.n_gs
        self.ACT_ALT_UP = self.ACT_CHARGE + 1
        self.ACT_ALT_DOWN = self.ACT_CHARGE + 2
        self.ACT_INC_UP = self.ACT_CHARGE + 3
        self.ACT_INC_DOWN = self.ACT_CHARGE + 4
        
        self.action_dim = self.n_targets + self.n_gs + 5
        
        # --- OBSERVATION SPACE ---
        # [Pos(3), Vel(3), Fuel(1), Battery(1), Buffer(1)] = 9
        self.obs_dim = 9
        self.global_state_dim = self.obs_dim * self.n_sats 

        self._build_locations()
        self._init_simulator()

    def _build_locations(self):
        rng = np.random.RandomState(42)
        # 1. Tạo Targets (Mục tiêu chụp ảnh)
        self.targets_ecef = []
        for _ in range(self.n_targets):
            lat = rng.uniform(-60, 60) * macros.D2R
            lon = rng.uniform(-180, 180) * macros.D2R
            self.targets_ecef.append(self._lld_to_ecef(lat, lon, 0))
            
        # 2. Tạo Ground Stations (Trạm thu dữ liệu)
        self.gs_ecef = []
        for lat_deg, lon_deg in self.gs_coords:
            self.gs_ecef.append(self._lld_to_ecef(lat_deg*macros.D2R, lon_deg*macros.D2R, 0))

    def _lld_to_ecef(self, lat, lon, alt):
        r = self.earth_radius + alt
        x = r * math.cos(lat) * math.cos(lon)
        y = r * math.cos(lat) * math.sin(lon)
        z = r * math.sin(lat)
        return np.array([x, y, z])

    def _cleanup(self):
        if hasattr(self, 'scSim'): self.scSim = None
        gc.collect()
    
    def _init_simulator(self):
        self._cleanup()
        self.scSim = SimulationBaseClass.SimBaseClass()
        self.scSim.SetProgressBar(False)
        self.taskName = "dynTask"
        self.dt = macros.sec2nano(1.0)
        self.scSim.CreateNewProcess("dynProcess").addTask(self.scSim.CreateNewTask(self.taskName, self.dt))

        self.all_viz_objects = [] # List gửi Vizard
        self.sats = []
        self.states = []
        
        gravFactory = simIncludeGravBody.gravBodyFactory()
        earth = gravFactory.createEarth(); earth.isCentralBody = True
        self.planet_mu = earth.mu
        
        # --- A. TẠO VỆ TINH ---
        for i in range(self.n_sats):
            sc = spacecraft.Spacecraft()
            sc.ModelTag = f"Sat_{i}"
            self.scSim.AddModelToTask(self.taskName, sc)
            gravFactory.addBodiesTo(sc)
            
            # Quỹ đạo Walker Delta
            oe = orbitalMotion.ClassicElements()
            oe.a = 7000e3; oe.e = 0.001; oe.i = 45.0 * macros.D2R
            oe.Omega = (i * 360.0 / self.n_sats) * macros.D2R; oe.omega = 0.0; oe.f = 0.0
            rN, vN = orbitalMotion.elem2rv(self.planet_mu, oe)
            
            sc.hub.r_CN_NInit = np.array(rN); sc.hub.v_CN_NInit = np.array(vN)
            self.sats.append(sc)
            self.all_viz_objects.append(sc)
            
            # State mở rộng: Thêm Battery và Data Buffer
            bore_vec = -np.array(rN) / np.linalg.norm(rN)
            self.states.append({
                'fuel': self.fuel_max,
                'batt': self.batt_max,      # Pin 100%
                'buffer': 0,                # Chưa có ảnh nào
                'bore_vec': bore_vec,
                'captured_targets': set()   # Để thống kê
            })

        # --- B. TẠO DUMMY OBJECTS CHO VIZARD (Targets & GS) ---
        self.dummy_objs = []
        
        # 1. Targets (Màu Đỏ - Red)
        for k, ecef in enumerate(self.targets_ecef):
            dummy = spacecraft.Spacecraft()
            dummy.ModelTag = f"TGT_{k}"
            # Nâng cao 50km
            r_mag = np.linalg.norm(ecef)
            dummy.hub.r_CN_NInit = ecef * ((r_mag + 50000.0) / r_mag)
            self.scSim.AddModelToTask(self.taskName, dummy)
            self.all_viz_objects.append(dummy)
            self.dummy_objs.append({'obj': dummy, 'type': 'target'})

        # 2. Ground Stations (Màu Xanh Lá - Green)
        gs_names = ["Hanoi", "NewYork", "London", "Sydney"]
        for k, ecef in enumerate(self.gs_ecef):
            dummy = spacecraft.Spacecraft()
            dummy.ModelTag = f"GS_{gs_names[k]}"
            # Nâng cao 60km (cao hơn target xíu cho dễ phân biệt)
            r_mag = np.linalg.norm(ecef)
            dummy.hub.r_CN_NInit = ecef * ((r_mag + 60000.0) / r_mag)
            self.scSim.AddModelToTask(self.taskName, dummy)
            self.all_viz_objects.append(dummy)
            self.dummy_objs.append({'obj': dummy, 'type': 'gs'})

        # --- C. VIZARD ---
        if self.viz:
            try:
                viz = vizSupport.enableUnityVisualization(self.scSim, self.taskName, self.all_viz_objects, saveFile="mission_sim.bin")
                # Vẽ camera
                '''
                try:
                    if hasattr(vizSupport, 'createStandardCamera'):
                        for i in range(self.n_sats):
                            vizSupport.createStandardCamera(viz, setMode=1, spacecraftName=f"Sat_{i}", 
                                fieldOfView=30.0*macros.D2R, pointingVector_B=[-1,0,0], position_B=[0,0,0])
                except: pass
                '''
                self.vizObj = viz
                print(f"--- Đã tạo {self.n_targets} Targets và {self.n_gs} GroundStations ---")
            except Exception as e: self.vizObj = None

        self.scSim.InitializeSimulation()
        self.sim_time = 0.0
    '''
    def _init_simulator(self):
            self._cleanup()
            self.scSim = SimulationBaseClass.SimBaseClass()
            self.scSim.SetProgressBar(False)
            self.taskName = "dynTask"
            self.dt = macros.sec2nano(1.0)
            self.scSim.CreateNewProcess("dynProcess").addTask(self.scSim.CreateNewTask(self.taskName, self.dt))

            self.all_viz_objects = [] 
            self.sats = []
            self.states = []
            
            gravFactory = simIncludeGravBody.gravBodyFactory()
            earth = gravFactory.createEarth(); earth.isCentralBody = True
            self.planet_mu = earth.mu
            
            # --- A. TẠO 4 VỆ TINH THẬT ---
            for i in range(self.n_sats):
                sc = spacecraft.Spacecraft()
                sc.ModelTag = f"Sat_{i}"
                self.scSim.AddModelToTask(self.taskName, sc)
                gravFactory.addBodiesTo(sc)
                
                # Quỹ đạo Walker Delta
                oe = orbitalMotion.ClassicElements()
                oe.a = 7000e3; oe.e = 0.001; oe.i = 45.0 * macros.D2R
                oe.Omega = (i * 360.0 / self.n_sats) * macros.D2R; oe.omega = 0.0; oe.f = 0.0
                rN, vN = orbitalMotion.elem2rv(self.planet_mu, oe)
                sc.hub.r_CN_NInit = np.array(rN); sc.hub.v_CN_NInit = np.array(vN)
                
                self.sats.append(sc)
                self.all_viz_objects.append(sc)
                
                bore_vec = -np.array(rN) / np.linalg.norm(rN)
                self.states.append({'fuel': self.fuel_max, 'batt': self.batt_max, 'buffer': 0, 'bore_vec': bore_vec, 'captured_targets': set()})

            # --- B. TẠO DUMMY OBJECTS ---
            self.dummy_targets = []
            self.dummy_gs = []

            # 1. Targets (Mục tiêu nhỏ, bay thấp)
            for k, ecef in enumerate(self.targets_ecef):
                dummy = spacecraft.Spacecraft()
                dummy.ModelTag = f"TGT_{k}"
                r_mag = np.linalg.norm(ecef)
                # Độ cao 50km
                dummy.hub.r_CN_NInit = ecef * ((r_mag + 50000.0) / r_mag)
                self.scSim.AddModelToTask(self.taskName, dummy)
                self.all_viz_objects.append(dummy)
                self.dummy_targets.append(dummy)

            # 2. Ground Stations (Trạm to, bay cao)
            gs_names = ["Hanoi", "NewYork", "London", "Sydney"]
            for k, ecef in enumerate(self.gs_ecef):
                dummy = spacecraft.Spacecraft()
                dummy.ModelTag = f"GS_{gs_names[k]}" 
                r_mag = np.linalg.norm(ecef)
                # [CHIẾN THUẬT]: Treo thật cao (300km) để tách biệt hoàn toàn với TGT
                dummy.hub.r_CN_NInit = ecef * ((r_mag + 300000.0) / r_mag)
                self.scSim.AddModelToTask(self.taskName, dummy)
                self.all_viz_objects.append(dummy)
                self.dummy_gs.append(dummy)

            # --- C. VIZARD SETUP (AN TOÀN TUYỆT ĐỐI) ---
            if self.viz:
                try:
                    viz = vizSupport.enableUnityVisualization(self.scSim, self.taskName, self.all_viz_objects, saveFile="mission_sim.bin")
                    
                    # 1. Vẽ Camera (Phân biệt bằng hình dáng nón)
                    # Chúng ta dùng createStandardCamera chuẩn (không color)
                    try:
                        if hasattr(vizSupport, 'createStandardCamera'):
                            # a. VỆ TINH THẬT: Nón Hẹp (20 độ)
                            for i in range(self.n_sats):
                                vizSupport.createStandardCamera(viz, setMode=1, spacecraftName=f"Sat_{i}", 
                                    fieldOfView=20.0*macros.D2R, pointingVector_B=[-1,0,0])

                            # b. MỤC TIÊU: Nón Vừa (45 độ)
                            for tgt in self.dummy_targets:
                                vizSupport.createStandardCamera(viz, setMode=1, spacecraftName=tgt.ModelTag,
                                    fieldOfView=45.0*macros.D2R, pointingVector_B=[0,1,0])

                            # c. TRẠM MẶT ĐẤT: Nón Phẳng (160 độ - Trông như cái đĩa)
                            for gs in self.dummy_gs:
                                vizSupport.createStandardCamera(viz, setMode=1, spacecraftName=gs.ModelTag,
                                    fieldOfView=160.0*macros.D2R, pointingVector_B=[0,1,0])
                    except: pass

                    # 2. Đổi màu đường quỹ đạo (Orbit Line)
                    # Lệnh này thường có trong vizSupport, nếu không có thì nó sẽ bỏ qua (pass)
                    try:
                        if hasattr(vizSupport, 'setOrbitColor'):
                            # GS -> Đường màu VÀNG [1, 1, 0, 1]
                            for gs in self.dummy_gs:
                                vizSupport.setOrbitColor(viz, spacecraftName=gs.ModelTag, color=[1, 1, 0, 1])
                            
                            # TGT -> Đường màu ĐỎ [1, 0, 0, 1]
                            for tgt in self.dummy_targets:
                                vizSupport.setOrbitColor(viz, spacecraftName=tgt.ModelTag, color=[1, 0, 0, 1])
                    except: pass

                    self.vizObj = viz
                    print("--- Đã tạo Vizard (Phân biệt bằng Độ cao & Hình dáng nón) ---")
                except Exception as e: self.vizObj = None

            self.scSim.InitializeSimulation()
            self.sim_time = 0.0
    '''
    def reset(self):
        self._init_simulator()
        return self._get_all_obs()

    def _get_all_obs(self):
        obs_list = []
        for i, sat in enumerate(self.sats):
            r = np.array(sat.hub.r_CN_NInit).flatten() / 1e7
            v = np.array(sat.hub.v_CN_NInit).flatten() / 1e4
            # Normalize các tài nguyên về [0, 1]
            f = [self.states[i]['fuel'] / self.fuel_max]
            b = [self.states[i]['batt'] / self.batt_max]
            d = [self.states[i]['buffer'] / self.buffer_max]
            
            # Obs = [Pos(3), Vel(3), Fuel(1), Batt(1), Buffer(1)] -> Size 9
            obs = np.concatenate([r, v, f, b, d]).astype(np.float32)
            obs_list.append(obs)
        return obs_list

    def step(self, actions, debug_mode=True):
        rewards = np.zeros(self.n_sats)
        dones = [False] * self.n_sats
        
        for i, action in enumerate(actions):
            sat = self.sats[i]
            state = self.states[i]
            act = int(action)
            
            # 0. Kiểm tra Sống/Chết (Hết pin là Game Over)
            if state['batt'] <= 0:
                rewards[i] -= 10.0 # Phạt nặng
                dones[i] = True
                if debug_mode: print(f"[t={self.sim_time:.0f}] 💀 Sat {i} DEAD (Hết pin)")
                continue

            # Tiêu thụ pin cơ bản (Idle)
            base_drain = 0.1
            active_drain = 0.5 # Khi xoay/chụp/downlink
            thrust_drain = 2.0 # Khi dùng động cơ
            
            # Lấy vector trạng thái
            v_curr = np.array(sat.hub.v_CN_NInit).flatten()
            r_curr = np.array(sat.hub.r_CN_NInit).flatten()
            v_dir = v_curr / (np.linalg.norm(v_curr) + 1e-9)
            h_vec = np.cross(r_curr, v_curr); h_dir = h_vec / (np.linalg.norm(h_vec) + 1e-9)
            alt_km = (np.linalg.norm(r_curr) - self.earth_radius) / 1000.0

            # --- XỬ LÝ HÀNH ĐỘNG ---

            # 1. SẠC PIN (CHARGE)
            if act == self.ACT_CHARGE:
                # Giả lập quay tấm pin về hướng mặt trời -> Tăng pin
                # Trong code này, ta coi như action này luôn thành công hồi năng lượng
                charge_rate = 3.0 
                state['batt'] = min(self.batt_max, state['batt'] + charge_rate)
                rewards[i] += 0.1 # Thưởng nhẹ vì biết giữ gìn sự sống
                if debug_mode and state['batt'] < 90: # Chỉ log khi không đầy
                    print(f"[t={self.sim_time:.0f}] 🔋 Sat {i} SẠC PIN -> {state['batt']:.1f}%")

            # 2. ĐỘNG CƠ (THRUST) - Tốn nhiều pin + xăng
            elif act in [self.ACT_ALT_UP, self.ACT_ALT_DOWN, self.ACT_INC_UP, self.ACT_INC_DOWN]:
                if state['fuel'] >= 1.0:
                    delta_v = 20.0
                    dv_vec = np.zeros(3)
                    
                    if act == self.ACT_ALT_UP: dv_vec = v_dir
                    elif act == self.ACT_ALT_DOWN: dv_vec = -v_dir
                    elif act == self.ACT_INC_UP: 
                        dv_vec = h_dir; delta_v = 50.0; state['fuel'] -= 1.0 # Tốn thêm xăng
                    elif act == self.ACT_INC_DOWN: 
                        dv_vec = -h_dir; delta_v = 50.0; state['fuel'] -= 1.0
                    
                    sat.hub.v_CN_NInit = v_curr + dv_vec * delta_v
                    state['fuel'] -= 1.0
                    state['batt'] -= thrust_drain
                    rewards[i] -= 0.1 # Phạt chi phí
                    
                    if debug_mode: print(f"[t={self.sim_time:.0f}] 🔥 Sat {i} Thrust Act {act}")
                else:
                    rewards[i] -= 0.5 # Hết xăng mà đòi bay

            # 3. CHỤP ẢNH MỤC TIÊU (0 -> N_TARGETS-1)
            elif 0 <= act < self.n_targets:
                state['batt'] -= active_drain
                
                # Logic xoay camera (như cũ)
                tgt_pos = self.targets_ecef[act]
                req_vec = tgt_pos - r_curr
                dist = np.linalg.norm(req_vec)
                req_vec /= (dist + 1e-9)
                
                cos_theta = np.dot(state['bore_vec'], req_vec)
                angle_deg = math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))
                max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                
                if angle_deg <= max_turn:
                    state['bore_vec'] = req_vec # Xoay xong
                    
                    # Logic Chụp:
                    # 1. Buffer còn chỗ
                    # 2. Đúng độ cao/khoảng cách
                    if state['buffer'] < self.buffer_max:
                        # Điều kiện đơn giản: < 4000km
                        if dist < 4000e3:
                            state['buffer'] += 1 # Lưu vào bộ nhớ
                            rewards[i] += 5.0    # Thưởng chụp được (nhưng chưa bằng downlink)
                            if debug_mode:
                                print(f"[t={self.sim_time:.0f}] 📸 Sat {i} CHỤP T{act} -> Buffer: {state['buffer']}/{self.buffer_max}")
                        else:
                            rewards[i] -= 0.1 # Xa quá
                    else:
                        rewards[i] -= 0.5 # Buffer đầy! Phải đi downlink ngay!
                        if debug_mode and i==0: print(f"[t={self.sim_time:.0f}] ⚠️ Sat {i} BUFFER FULL!")
                else:
                    # Partial Slew
                    ratio = max_turn / angle_deg
                    new_vec = (1 - ratio) * state['bore_vec'] + ratio * req_vec
                    state['bore_vec'] = new_vec / np.linalg.norm(new_vec)
                    if debug_mode: print(f"[t={self.sim_time:.0f}] 🔄 Sat {i} Slewing T{act} ({angle_deg:.0f}°)")

            # 4. DOWNLINK TẠI TRẠM MẶT ĐẤT (GS) (N_TARGETS -> N_TARGETS + N_GS - 1)
            elif self.n_targets <= act < self.n_targets + self.n_gs:
                state['batt'] -= active_drain
                gs_idx = act - self.n_targets
                gs_pos = self.gs_ecef[gs_idx]
                
                # Logic xoay tới GS y hệt Target
                req_vec = gs_pos - r_curr
                dist = np.linalg.norm(req_vec)
                req_vec /= (dist + 1e-9)
                
                cos_theta = np.dot(state['bore_vec'], req_vec)
                angle_deg = math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))
                max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                
                if angle_deg <= max_turn:
                    state['bore_vec'] = req_vec # Lock vào trạm
                    
                    # Logic Downlink:
                    # 1. Có dữ liệu trong Buffer
                    # 2. Khoảng cách gần (< 2500km)
                    if state['buffer'] > 0:
                        if dist < 2500e3:
                            data_downlinked = 1 # Xả 1 ảnh mỗi bước
                            state['buffer'] -= data_downlinked
                            rewards[i] += 20.0 # THƯỞNG LỚN: Hoàn thành nhiệm vụ
                            if debug_mode:
                                gs_names = ["Hanoi", "NY", "London", "Sydney"]
                                print(f"[t={self.sim_time:.0f}] 📡 Sat {i} DOWNLINK @ {gs_names[gs_idx]} -> +20đ (Còn {state['buffer']} ảnh)")
                        else:
                            rewards[i] -= 0.1 # Chưa tới trạm
                    else:
                        rewards[i] -= 0.2 # Không có gì để gửi mà cũng kết nối
                else:
                     # Partial Slew tới GS
                    ratio = max_turn / angle_deg
                    new_vec = (1 - ratio) * state['bore_vec'] + ratio * req_vec
                    state['bore_vec'] = new_vec / np.linalg.norm(new_vec)
                    if debug_mode: 
                        gs_names = ["Hanoi", "NY", "London", "Sydney"]
                        print(f"[t={self.sim_time:.0f}] 🔄 Sat {i} Slewing GS_{gs_names[gs_idx]}")

            else:
                # Idle drain
                state['batt'] -= base_drain

        # Step Sim
        stop_time = self.sim_time + self.decision_dt
        self.scSim.ConfigureStopTime(int(macros.sec2nano(stop_time)))
        self.scSim.ExecuteSimulation()
        self.sim_time = stop_time
        
        if self.sim_time > 8000: dones = [True] * self.n_sats
        return self._get_all_obs(), rewards, dones, {}

# --- 4. TRAINER ---
def train_mission():
    N_SATS = 4
    N_TARGETS = 50
    # Env tự tính Action Dim và Obs Dim
    env = BasiliskFullMissionEnv(n_sats=N_SATS, n_targets=N_TARGETS, viz=True)
    
    OBS_DIM = env.obs_dim
    ACT_DIM = env.action_dim
    GLOBAL_STATE_DIM = OBS_DIM * N_SATS
    
    mappo_agent = MultiAgentActorCritic(OBS_DIM, GLOBAL_STATE_DIM, ACT_DIM)
    optimizer = optim.Adam(mappo_agent.parameters(), lr=1e-3)
    
    print(f"--- START MISSION TRAINING ---")
    print(f"Actions: {ACT_DIM} | Obs: {OBS_DIM}")
    print("Nhiệm vụ: Sạc Pin -> Chụp Ảnh -> Bay về Trạm (Hanoi/NY/...) -> Downlink")
    
    for epoch in range(15): # Demo 15 epochs
        obs_list = env.reset()
        ep_rewards = np.zeros(N_SATS)
        done = False
        
        # Buffer
        batch_obs, batch_gs, batch_acts, batch_rews = [], [], [], []
        
        while not done:
            actions_t = []
            global_state_t = np.concatenate(obs_list)
            
            # 1. Act
            for i in range(N_SATS):
                obs_t = torch.FloatTensor(obs_list[i]).unsqueeze(0)
                logits = mappo_agent.act(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                actions_t.append(action.item())
            
            # 2. Env Step (Tắt debug khi train)
            next_obs, rewards, dones, _ = env.step(actions_t, debug_mode=True)
            
            # Store
            batch_obs.append(obs_list)
            batch_gs.append(global_state_t)
            batch_acts.append(actions_t)
            batch_rews.append(rewards)
            
            obs_list = next_obs
            ep_rewards += rewards
            done = any(dones) # Chỉ cần 1 sat chết là reset epoch (hoặc all, tùy logic)
        
        # 3. Update (Simple PG for demo brevity)
        optimizer.zero_grad()
        loss = 0
        
        # Tính Discounted Returns
        returns = np.zeros_like(batch_rews)
        running_add = np.zeros(N_SATS)
        for t in reversed(range(len(batch_rews))):
            running_add = batch_rews[t] + 0.99 * running_add
            returns[t] = running_add
            
        # Loss Calculation
        for t in range(len(batch_obs)):
            g_state = torch.FloatTensor(batch_gs[t]).unsqueeze(0)
            target_val = torch.FloatTensor([np.mean(returns[t])]).unsqueeze(0)
            
            val_pred = mappo_agent.evaluate(g_state)
            val_loss = nn.MSELoss()(val_pred, target_val)
            
            act_loss = 0
            ent_loss = 0
            for i in range(N_SATS):
                obs = torch.FloatTensor(batch_obs[t][i]).unsqueeze(0)
                act = torch.tensor([batch_acts[t][i]])
                adv = returns[t][i] - val_pred.item()
                
                logits = mappo_agent.act(obs)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(act)
                
                act_loss += -log_prob * adv
                ent_loss += dist.entropy()
            
            # Loss tổng hợp: Actor + Critic - Entropy
            loss += act_loss + 0.5 * val_loss - 0.05 * (ent_loss/N_SATS)
            
        loss.backward()
        nn.utils.clip_grad_norm_(mappo_agent.parameters(), 0.5)
        optimizer.step()
        
        print(f"Epoch {epoch+1} | Reward: {np.mean(ep_rewards):.1f}")

    # --- DEMO PLAYBACK ---
    print("\n--- CHẠY DEMO SỨ MỆNH ---")
    obs_list = env.reset()
    done = False
    try:
        while not done:
            actions_t = []
            for i in range(N_SATS):
                obs_t = torch.FloatTensor(obs_list[i]).unsqueeze(0)
                logits = mappo_agent.act(obs_t)
                # Greedy action
                action = torch.argmax(logits, dim=1)
                actions_t.append(action.item())
            
            # Bật Debug để xem log Sạc/Chụp/Downlink
            next_obs, rewards, dones, _ = env.step(actions_t, debug_mode=True)
            obs_list = next_obs
            done = any(dones)
    except KeyboardInterrupt:
        print("Stop Demo.")

if __name__ == "__main__":
    train_mission()
    print("\n--- MISSION COMPLETE ---")