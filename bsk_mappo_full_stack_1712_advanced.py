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
    # Thêm vào danh sách import
    from Basilisk.simulation import dragDynamicEffector, exponentialAtmosphere
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
        self.batt_max = 100.0
        self.buffer_max = 5 
        
        # Biến theo dõi toàn cục (Global Memory)
        # Lưu các ID mục tiêu đã được chụp bởi BẤT KỲ vệ tinh nào
        self.global_captured_targets = set()

        # Cấu hình Trạm Mặt đất
        self.gs_coords = [
            (21.0285, 105.8542),   # Hanoi
            (40.7128, -74.0060),   # New York
            (51.5074, -0.1278),    # London
            (-33.8688, 151.2093)   # Sydney
        ]
        self.n_gs = len(self.gs_coords)
        
        # ACTION SPACE
        self.ACT_CHARGE = self.n_targets + self.n_gs
        self.ACT_ALT_UP = self.ACT_CHARGE + 1
        self.ACT_ALT_DOWN = self.ACT_CHARGE + 2
        self.ACT_INC_UP = self.ACT_CHARGE + 3
        self.ACT_INC_DOWN = self.ACT_CHARGE + 4
        
        self.action_dim = self.n_targets + self.n_gs + 5
        self.obs_dim = 9
        self.global_state_dim = self.obs_dim * self.n_sats 

        self._build_locations()
        self._init_simulator()

    def _build_locations(self):
        rng = np.random.RandomState(42)
        self.targets_ecef = []
        for _ in range(self.n_targets):
            lat = rng.uniform(-60, 60) * macros.D2R
            lon = rng.uniform(-180, 180) * macros.D2R
            self.targets_ecef.append(self._lld_to_ecef(lat, lon, 0))
            
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
    '''
    def _init_simulator(self):
        self._cleanup()
        self.scSim = SimulationBaseClass.SimBaseClass()
        self.scSim.SetProgressBar(False)
        self.taskName = "dynTask"
        self.dt = macros.sec2nano(2.0) #Xem thời gian nào phù hợp
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
            
            oe = orbitalMotion.ClassicElements()
            oe.a = 7000e3; oe.e = 0.001; oe.i = 45.0 * macros.D2R
            oe.Omega = (i * 360.0 / self.n_sats) * macros.D2R; oe.omega = 0.0; oe.f = 0.0
            rN, vN = orbitalMotion.elem2rv(self.planet_mu, oe)
            sc.hub.r_CN_NInit = np.array(rN); sc.hub.v_CN_NInit = np.array(vN)
            
            self.sats.append(sc)
            self.all_viz_objects.append(sc)
            
            bore_vec = -np.array(rN) / np.linalg.norm(rN)
            self.states.append({'fuel': self.fuel_max, 'batt': self.batt_max, 'buffer': 0, 'bore_vec': bore_vec})

        # --- B. TẠO DUMMY OBJECTS ---
        self.dummy_targets = []
        self.dummy_gs = []

        # 1. Targets (Nhỏ, Thấp)
        for k, ecef in enumerate(self.targets_ecef):
            dummy = spacecraft.Spacecraft()
            dummy.ModelTag = f"TGT_{k}"
            r_mag = np.linalg.norm(ecef)
            dummy.hub.r_CN_NInit = ecef * ((r_mag + 50000.0) / r_mag)
            self.scSim.AddModelToTask(self.taskName, dummy)
            self.all_viz_objects.append(dummy)
            self.dummy_targets.append(dummy)

        # 2. Ground Stations (To, Cao)
        gs_names = ["Hanoi", "NewYork", "London", "Sydney"]
        for k, ecef in enumerate(self.gs_ecef):
            dummy = spacecraft.Spacecraft()
            dummy.ModelTag = f"GS_{gs_names[k]}" 
            r_mag = np.linalg.norm(ecef)
            dummy.hub.r_CN_NInit = ecef * ((r_mag + 300000.0) / r_mag)
            self.scSim.AddModelToTask(self.taskName, dummy)
            self.all_viz_objects.append(dummy)
            self.dummy_gs.append(dummy)

        # --- C. VIZARD SETUP (Safe Mode) ---
        if self.viz:
            try:
                viz = vizSupport.enableUnityVisualization(self.scSim, self.taskName, self.all_viz_objects, saveFile="mission_sim_final.bin")
                try:
                    if hasattr(vizSupport, 'createStandardCamera'):
                        # Sat: 20 deg
                        for i in range(self.n_sats):
                            vizSupport.createStandardCamera(viz, setMode=1, spacecraftName=f"Sat_{i}", 
                                fieldOfView=20.0*macros.D2R, pointingVector_B=[-1,0,0])
                        # Target: 45 deg
                        for tgt in self.dummy_targets:
                            vizSupport.createStandardCamera(viz, setMode=1, spacecraftName=tgt.ModelTag,
                                fieldOfView=45.0*macros.D2R, pointingVector_B=[0,1,0])
                        # GS: 160 deg
                        for gs in self.dummy_gs:
                            vizSupport.createStandardCamera(viz, setMode=1, spacecraftName=gs.ModelTag,
                                fieldOfView=160.0*macros.D2R, pointingVector_B=[0,1,0])
                except: pass
                
                # Màu đường quỹ đạo (nếu có)
                try:
                    if hasattr(vizSupport, 'setOrbitColor'):
                        for gs in self.dummy_gs: vizSupport.setOrbitColor(viz, spacecraftName=gs.ModelTag, color=[1, 1, 0, 1])
                        for tgt in self.dummy_targets: vizSupport.setOrbitColor(viz, spacecraftName=tgt.ModelTag, color=[1, 0, 0, 1])
                except: pass
                self.vizObj = viz
            except Exception as e: self.vizObj = None

        self.scSim.InitializeSimulation()
        self.sim_time = 0.0
    '''
    
    def _init_simulator(self):
            self._cleanup()
            self.scSim = SimulationBaseClass.SimBaseClass()
            self.scSim.SetProgressBar(False)
            self.taskName = "dynTask"
            
            self.decision_dt = 10.0      
            self.dt = macros.sec2nano(1.0) 
            
            self.scSim.CreateNewProcess("dynProcess").addTask(self.scSim.CreateNewTask(self.taskName, self.dt))

            self.all_viz_objects = [] 
            self.sats = []
            self.states = []
            
            # --- 1. TRỌNG LỰC (GRAVITY - J2) ---
            gravFactory = simIncludeGravBody.gravBodyFactory()
            earth = gravFactory.createEarth()
            earth.isCentralBody = True
            self.planet_mu = earth.mu
            
            # Load GGM03S (Thử tìm file, nếu ko thấy dùng Point Mass)
            import os
            import Basilisk
            bskPath = os.path.dirname(Basilisk.__file__)
            gravFilePath = os.path.join(bskPath, "supportData", "LocalGravData", "GGM03S.txt")
            
            try:
                if os.path.exists(gravFilePath):
                    earth.useSphericalHarmonicsGravityModel(gravFilePath, 10)
                    print("--- [OK] Đã load Trọng lực J2 (GGM03S) ---")
                else:
                    print("--- [INFO] Không thấy file GGM03S, dùng Point Mass ---")
            except: pass

            # --- 2. KHÍ QUYỂN (ATMOSPHERE) ---
            atmo = exponentialAtmosphere.ExponentialAtmosphere()
            atmo.ModelTag = "ExpAtmo"
            atmo.planetRadius = self.earth_radius
            atmo.scaleHeight = 7200.0
            atmo.baseDensity = 4e-13
            self.scSim.AddModelToTask(self.taskName, atmo)

            # --- A. TẠO VỆ TINH THẬT ---
            for i in range(self.n_sats):
                sc = spacecraft.Spacecraft()
                sc.ModelTag = f"Sat_{i}"
                self.scSim.AddModelToTask(self.taskName, sc)
                
                gravFactory.addBodiesTo(sc)
                
                # [FIX LỖI TRƯỚC]: Truyền State Message, không truyền Object
                atmo.addSpacecraftToModel(sc.scStateOutMsg)
                
                # Tạo Drag Effector
                dragEffector = dragDynamicEffector.DragDynamicEffector()
                dragEffector.ModelTag = f"DragEff_{i}"
                dragEffector.coreParams.projectedArea = 2.0 
                dragEffector.coreParams.dragCoeff = 2.2
                
                # Kết nối mật độ khí (Safe mode)
                try:
                    dragEffector.setDensityMessage(atmo.envOutMsgs[i])
                except:
                    if hasattr(dragEffector, 'atmoDensInMsg'):
                        dragEffector.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[i])
                
                sc.addDynamicEffector(dragEffector)
                self.scSim.AddModelToTask(self.taskName, dragEffector)
                
                # Quỹ đạo Walker Delta
                oe = orbitalMotion.ClassicElements()
                oe.a = 7000e3; oe.e = 0.001; oe.i = 45.0 * macros.D2R
                oe.Omega = (i * 360.0 / self.n_sats) * macros.D2R; oe.omega = 0.0; oe.f = 0.0
                rN, vN = orbitalMotion.elem2rv(self.planet_mu, oe)
                sc.hub.r_CN_NInit = np.array(rN); sc.hub.v_CN_NInit = np.array(vN)
                
                self.sats.append(sc)
                self.all_viz_objects.append(sc)
                
                bore_vec = -np.array(rN) / np.linalg.norm(rN)
                self.states.append({
                    'fuel': self.fuel_max,
                    'batt': self.batt_max,
                    'buffer': 0,
                    'bore_vec': bore_vec
                })

            # --- B. DUMMY OBJECTS ---
            
            # 1. Targets
            for k, ecef in enumerate(self.targets_ecef):
                dummy = spacecraft.Spacecraft()
                dummy.ModelTag = f"TGT_{k}" 
                r_mag = np.linalg.norm(ecef)
                dummy.hub.r_CN_NInit = ecef * ((r_mag + 50000.0) / r_mag)
                self.scSim.AddModelToTask(self.taskName, dummy)
                self.all_viz_objects.append(dummy)

            # 2. Ground Stations
            # [FIX LỖI NAME ERROR]: Đã thêm lại dòng định nghĩa tên trạm ở đây
            gs_names = ["Hanoi", "NY", "London", "Sydney"] 
            
            for k, ecef in enumerate(self.gs_ecef):
                dummy = spacecraft.Spacecraft()
                # Bây giờ biến gs_names đã tồn tại, code sẽ chạy đúng
                dummy.ModelTag = f"GS_{gs_names[k]}"
                r_mag = np.linalg.norm(ecef)
                dummy.hub.r_CN_NInit = ecef * ((r_mag + 400000.0) / r_mag)
                self.scSim.AddModelToTask(self.taskName, dummy)
                self.all_viz_objects.append(dummy)

            # --- C. VIZARD ---
            if self.viz:
                try:
                    viz = vizSupport.enableUnityVisualization(
                        self.scSim, self.taskName, self.all_viz_objects, 
                        saveFile="mission_sim_advanced.bin"
                    )
                    viz.settings.showSpacecraftLabels = 1 
                    self.vizObj = viz
                except Exception as e: self.vizObj = None

            self.scSim.InitializeSimulation()
            self.sim_time = 0.0
        
    def reset(self):
        # Reset lại danh sách Global
        self.global_captured_targets = set()
        self._init_simulator()
        return self._get_all_obs()

    def _get_all_obs(self):
        obs_list = []
        for i, sat in enumerate(self.sats):
            r = np.array(sat.hub.r_CN_NInit).flatten() / 1e7
            v = np.array(sat.hub.v_CN_NInit).flatten() / 1e4
            f = [self.states[i]['fuel'] / self.fuel_max]
            b = [self.states[i]['batt'] / self.batt_max]
            d = [self.states[i]['buffer'] / self.buffer_max]
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
            
            # Check chết
            if state['batt'] <= 0 or state['fuel'] <= 0:
                rewards[i] -= 10.0; dones[i] = True
                if debug_mode: print(f"[t={self.sim_time:.0f}] 💀 Sat {i} DEAD Because " + ("BATTERY" if state['batt'] <= 0 else "FUEL"))
                continue

            base_drain = 0.1; active_drain = 0.5; thrust_drain = 2.0
            v_curr = np.array(sat.hub.v_CN_NInit).flatten()
            r_curr = np.array(sat.hub.r_CN_NInit).flatten()
            v_dir = v_curr / (np.linalg.norm(v_curr) + 1e-9)
            h_vec = np.cross(r_curr, v_curr); h_dir = h_vec / (np.linalg.norm(h_vec) + 1e-9)
            alt_km = (np.linalg.norm(r_curr) - self.earth_radius) / 1000.0

            # --- 1. SẠC PIN ---
            if act == self.ACT_CHARGE:
                state['batt'] = min(self.batt_max, state['batt'] + 3.0)
                rewards[i] += 0.1
                if debug_mode and state['batt'] < 90: print(f"[t={self.sim_time:.0f}] 🔋 Sat {i} CHARGE -> {state['batt']:.1f}%")

            # --- 2. ĐỘNG CƠ (ALT / INC) ---
            elif act in [self.ACT_ALT_UP, self.ACT_ALT_DOWN, self.ACT_INC_UP, self.ACT_INC_DOWN]:
                if state['fuel'] >= 1.0:
                    #delta_v = 20.0
                    delta_v = 20.0 #Demo
                    dv_vec = np.zeros(3)
                    cost = 1.0
                    if act == self.ACT_ALT_UP: dv_vec = v_dir
                    elif act == self.ACT_ALT_DOWN: dv_vec = -v_dir
                    elif act == self.ACT_INC_UP: dv_vec = h_dir; delta_v = 50.0; cost = 2.0
                    elif act == self.ACT_INC_DOWN: dv_vec = -h_dir; delta_v = 50.0; cost = 2.0
                    
                    if state['fuel'] >= cost:
                        sat.hub.v_CN_NInit = v_curr + dv_vec * delta_v
                        state['fuel'] -= cost
                        state['batt'] -= thrust_drain
                        rewards[i] -= 0.1 # Test thử tí
                        #rewards[i] += 7.0 - cost
                        if debug_mode: 
                            lbl = "INC" if cost > 1 else "ALT"
                            print(f"[t={self.sim_time:.0f}] 🔥 Sat {i} {lbl} CHANGE")
                    else: 
                        rewards[i] -= 0.5
                        if debug_mode: print(f"[t={self.sim_time:.0f}] ⚠️ Sat {i} Not Enough Fuel for Orbit Change")
                else: 
                    rewards[i] -= 0.5
                    if debug_mode: print(f"[t={self.sim_time:.0f}] ⚠️ Sat {i} Not Enough Fuel for Orbit Change")

            # --- 3. CHỤP MỤC TIÊU (LOGIC MỚI: ONE-TIME CAPTURE) ---
            elif 0 <= act < self.n_targets:
                state['batt'] -= active_drain
                
                # [MỚI] KIỂM TRA TOÀN CỤC: Nếu đã chụp rồi thì bỏ qua
                if act in self.global_captured_targets:
                    rewards[i] -= 0.2 # Phạt nhẹ vì chọn mục tiêu đã xong
                    if debug_mode: print(f"[t={self.sim_time:.0f}] ⚠️ Sat {i} Target {act} ALREADY DONE")
                
                else:
                    # Logic xoay & chụp bình thường
                    tgt_pos = self.targets_ecef[act]
                    req_vec = tgt_pos - r_curr
                    dist = np.linalg.norm(req_vec); req_vec /= (dist + 1e-9)
                    cos_theta = np.dot(state['bore_vec'], req_vec)
                    angle = math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))
                    max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                    
                    if angle <= max_turn:
                        state['bore_vec'] = req_vec
                        if state['buffer'] < self.buffer_max:
                            if dist < 4000e3:
                                # THÀNH CÔNG!
                                self.global_captured_targets.add(act) # Đánh dấu đã xong toàn cục
                                state['buffer'] += 1
                                rewards[i] += 5.0
                                if debug_mode:
                                    print(f"[t={self.sim_time:.0f}] 📸 Sat {i} CAPTURED T{act} -> Buffer: {state['buffer']}/{self.buffer_max} (Global Done: {len(self.global_captured_targets)})")
                            else: rewards[i] -= 0.1
                        else:
                            rewards[i] -= 0.5 # Full buffer
                            if debug_mode and i==0: print(f"[t={self.sim_time:.0f}] ⚠️ Sat {i} BUFFER FULL")
                    else:
                        ratio = max_turn / angle
                        new_vec = (1 - ratio) * state['bore_vec'] + ratio * req_vec
                        state['bore_vec'] = new_vec / np.linalg.norm(new_vec)
                        if debug_mode: print(f"[t={self.sim_time:.0f}] 🔄 Sat {i} Slew T{act} ({angle:.0f}°)")

            # --- 4. DOWNLINK ---
            elif self.n_targets <= act < self.n_targets + self.n_gs:
                state['batt'] -= active_drain
                gs_idx = act - self.n_targets
                gs_pos = self.gs_ecef[gs_idx]
                req_vec = gs_pos - r_curr
                dist = np.linalg.norm(req_vec); req_vec /= (dist + 1e-9)
                
                cos = np.dot(state['bore_vec'], req_vec)
                angle = math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))
                max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                
                # [DEBUG] In ra để xem nó đang cố làm gì
                if debug_mode: 
                    print(f"Sat {i} try Downlink GS_{gs_idx}: Dist={dist/1000:.0f}km, Buff={state['buffer']}")

                if angle <= max_turn:
                    state['bore_vec'] = req_vec
                    
                    # [NỚI LỎNG] Tăng khoảng cách lên 3500km (dễ trúng hơn)
                    if dist < 3500e3: 
                        if state['buffer'] > 0:
                            state['buffer'] -= 1
                            rewards[i] += 20.0
                            if debug_mode: print(f"[t={self.sim_time:.0f}] 📡 Sat {i} DOWNLINK SUCCESS (+20)!!!")
                        else:
                            # Đã đến nơi nhưng không có hàng
                            rewards[i] -= 0.5 
                            if debug_mode and i==0: print(f"[t={self.sim_time:.0f}] ⚠️ Sat {i} Over GS but Buffer Empty")
                    else:
                        # Đã hướng về trạm nhưng còn xa quá
                        rewards[i] -= 0.1
                        if debug_mode: 
                            print(f"[t={self.sim_time:.0f}] ⏳ Sat {i} Too Far for Downlink ({dist/1000:.0f} km)")
                else:
                    ratio = max_turn / angle
                    state['bore_vec'] = (1-ratio)*state['bore_vec'] + ratio*req_vec
                    state['bore_vec'] /= np.linalg.norm(state['bore_vec'])
                    if debug_mode: 
                        print(f"[t={self.sim_time:.0f}] 🔄 Sat {i} Slew GS ({angle:.0f}°)")
            else: 
                state['batt'] -= base_drain

        # Step Sim
        stop_time = self.sim_time + self.decision_dt
        self.scSim.ConfigureStopTime(int(macros.sec2nano(stop_time)))
        self.scSim.ExecuteSimulation()
        self.sim_time = stop_time
        
        # Kết thúc nếu hết thời gian HOẶC đã chụp hết 50 mục tiêu
        if self.sim_time > 8000: dones = [True] * self.n_sats
        if len(self.global_captured_targets) == self.n_targets:
            if debug_mode: print("--- MISSION COMPLETED (ALL TARGETS) ---")
            # dones = [True] * self.n_sats # Tùy chọn: Dừng ngay khi xong hết
            
        return self._get_all_obs(), rewards, dones, {}
    '''
    
    def step(self, actions, debug_mode=True):
            rewards = np.zeros(self.n_sats)
            dones = [False] * self.n_sats
            
            for i, action in enumerate(actions):
                sat = self.sats[i]
                state = self.states[i]
                act = int(action)
                
                # --- Xử lý Pin/Chết ---
                if state['batt'] <= 0 or state['fuel'] <= 0:
                    rewards[i] -= 10.0; dones[i] = True
                    print(f"[t={self.sim_time:.0f}] 💀 Sat {i} DEAD Because " + ("BATTERY" if state['batt'] <= 0 else "FUEL"))
                    continue
                
                # Lấy vector hiện tại
                v_curr = np.array(sat.hub.v_CN_NInit).flatten()
                r_curr = np.array(sat.hub.r_CN_NInit).flatten()
                
                # Tính toán các vector hướng
                v_mag = np.linalg.norm(v_curr)
                v_dir = v_curr / (v_mag + 1e-9)
                
                # --- NHÓM 1: SẠC PIN ---
                if act == self.ACT_CHARGE:
                    state['batt'] = min(self.batt_max, state['batt'] + 3.0)
                    rewards[i] += 0.1
                    if debug_mode and state['batt'] < 90:
                        print(f"[t={self.sim_time:.0f}] 🔋 Sat {i} CHARGE -> {state['batt']:.1f}%")

                # --- NHÓM 2: THAY ĐỔI QUỸ ĐẠO (Đã nâng cấp) ---
                elif act in [self.ACT_ALT_UP, self.ACT_ALT_DOWN, self.ACT_INC_UP, self.ACT_INC_DOWN]:
                    if state['fuel'] >= 2.0:
                        cost = 2.0
                        
                        # A. THAY ĐỔI ĐỘ CAO (Dùng lực đẩy vector thường)
                        if act == self.ACT_ALT_UP:
                            sat.hub.v_CN_NInit = v_curr + v_dir * 50.0 # Tăng tốc 50m/s
                            if debug_mode: print(f"[t={self.sim_time:.0f}] 🚀 Sat {i} ALT UP (Boost)")
                        
                        elif act == self.ACT_ALT_DOWN:
                            sat.hub.v_CN_NInit = v_curr - v_dir * 50.0 # Hãm tốc 50m/s
                            if debug_mode: print(f"[t={self.sim_time:.0f}] 🔻 Sat {i} ALT DOWN (Brake)")

                        # B. THAY ĐỔI GÓC NGHIÊNG (Dùng biến đổi Kepler - Visual cực mạnh)
                        elif act in [self.ACT_INC_UP, self.ACT_INC_DOWN]:
                            # 1. Chuyển đổi r,v sang Kepler Elements (oe)
                            oe = orbitalMotion.rv2elem(self.planet_mu, r_curr, v_curr)
                            
                            old_i_deg = oe.i * macros.R2D
                            change_amount = 15.0 * macros.D2R # Thay đổi hẳn 15 độ cho máu!
                            
                            if act == self.ACT_INC_UP:
                                # Tăng độ nghiêng -> Giúp bao phủ vùng cực (Polar regions)
                                # Giới hạn max 90 độ (quỹ đạo cực)
                                if oe.i < 85.0 * macros.D2R:
                                    oe.i += change_amount
                                    if debug_mode: print(f"[t={self.sim_time:.0f}] 📐 Sat {i} INC UP: {old_i_deg:.0f}° -> {oe.i*macros.R2D:.0f}°")
                                else:
                                    rewards[i] -= 0.5 # Đã kịch kim, ko tăng được nữa
                            
                            elif act == self.ACT_INC_DOWN:
                                # Giảm độ nghiêng -> Giúp bao phủ vùng xích đạo
                                # Giới hạn min 0 độ (quỹ đạo xích đạo)
                                if oe.i > 5.0 * macros.D2R:
                                    oe.i -= change_amount
                                    if debug_mode: print(f"[t={self.sim_time:.0f}] 📐 Sat {i} INC DOWN: {old_i_deg:.0f}° -> {oe.i*macros.R2D:.0f}°")
                                else:
                                    rewards[i] -= 0.5

                            # 2. Chuyển ngược lại r,v mới từ oe đã sửa
                            r_new, v_new = orbitalMotion.elem2rv(self.planet_mu, oe)
                            
                            # 3. Cập nhật cho vệ tinh
                            sat.hub.r_CN_NInit = r_new
                            sat.hub.v_CN_NInit = v_new
                            
                            # Phạt nặng vì đổi góc nghiêng cực tốn kém
                            state['fuel'] -= 5.0 
                            state['batt'] -= 5.0

                        state['fuel'] -= cost
                        state['batt'] -= 2.0
                    else:
                        rewards[i] -= 0.5 # Hết xăng

                # --- NHÓM 3: CHỤP ẢNH (Logic cũ) ---
                elif 0 <= act < self.n_targets:
                    # ... (Giữ nguyên logic chụp ảnh của bạn) ...
                    # Copy lại phần chụp ảnh từ code cũ vào đây
                    state['batt'] -= 0.5
                    if act in self.global_captured_targets:
                        rewards[i] -= 0.1
                    else:
                        tgt_pos = self.targets_ecef[act]
                        req_vec = tgt_pos - r_curr
                        dist = np.linalg.norm(req_vec); req_vec /= dist
                        cos = np.dot(state['bore_vec'], req_vec)
                        angle = math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))
                        max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                        
                        if angle <= max_turn:
                            state['bore_vec'] = req_vec
                            if state['buffer'] < self.buffer_max and dist < 4000e3:
                                self.global_captured_targets.add(act)
                                state['buffer'] += 1
                                rewards[i] += 5.0
                                if debug_mode: print(f"[t={self.sim_time:.0f}] 📸 Sat {i} CAPTURE T{act}")
                            else: rewards[i] -= 0.1
                        else:
                            ratio = max_turn / angle
                            state['bore_vec'] = (1-ratio)*state['bore_vec'] + ratio*req_vec
                            state['bore_vec'] /= np.linalg.norm(state['bore_vec'])

                # --- NHÓM 4: DOWNLINK (Logic cũ) ---
                elif self.n_targets <= act < self.n_targets + self.n_gs:
                    # ... (Giữ nguyên logic downlink của bạn) ...
                    # Copy lại phần downlink từ code cũ vào đây
                    state['batt'] -= 0.5
                    gs_idx = act - self.n_targets
                    gs_pos = self.gs_ecef[gs_idx]
                    req_vec = gs_pos - r_curr
                    dist = np.linalg.norm(req_vec); req_vec /= dist
                    cos = np.dot(state['bore_vec'], req_vec)
                    angle = math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))
                    max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                    
                    if angle <= max_turn:
                        state['bore_vec'] = req_vec
                        if state['buffer'] > 0 and dist < 2500e3:
                            state['buffer'] -= 1
                            rewards[i] += 20.0
                            if debug_mode: print(f"[t={self.sim_time:.0f}] 📡 Sat {i} DOWNLINK (+20)")
                        else: rewards[i] -= 0.1
                    else:
                        ratio = max_turn / angle
                        state['bore_vec'] = (1-ratio)*state['bore_vec'] + ratio*req_vec
                        state['bore_vec'] /= np.linalg.norm(state['bore_vec'])

                else:
                    state['batt'] -= 0.1

            # Step Sim & Check Done
            stop_time = self.sim_time + self.decision_dt
            self.scSim.ConfigureStopTime(int(macros.sec2nano(stop_time)))
            self.scSim.ExecuteSimulation()
            self.sim_time = stop_time
            
            if self.sim_time > 6000: dones = [True] * self.n_sats
                
            return self._get_all_obs(), rewards, dones, {}
    '''
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