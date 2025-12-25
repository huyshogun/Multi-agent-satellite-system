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
    from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, vizSupport
    from Basilisk.simulation import spacecraft
    
    # [QUAN TRỌNG] Import module Ngoại lực và Messaging
    from Basilisk.simulation import extForceTorque, dragDynamicEffector, exponentialAtmosphere
    from Basilisk.architecture import messaging 
    
    print("[INIT] Basilisk libraries loaded.")
except ImportError:
    print("[ERROR] Basilisk not found.")
    sys.exit(1)

# --- 2. MAPPO NETWORK (CTDE) ---
class MultiAgentActorCritic(nn.Module):
    def __init__(self, obs_dim, global_state_dim, action_dim, hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(global_state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def act(self, obs): return self.actor(obs)
    def evaluate(self, global_state): return self.critic(global_state)

# --- 3. MULTI-AGENT ENVIRONMENT ---
class BasiliskFullMissionEnv:
    def __init__(self, n_sats=4, n_targets=50, viz=False):
        self.n_sats = n_sats
        self.n_targets = n_targets
        self.viz = viz
        
        # Physics Constants
        self.earth_radius = 6371e3
        self.mu = 398600.4418e9
        
        # Config
        self.max_slew_rate = math.radians(5.0) 
        self.fuel_max = 200.0
        self.batt_max = 100.0
        self.buffer_max = 5 
        
        self.global_captured_targets = set()
        self.gs_coords = [(21.02, 105.85), (40.71, -74.00), (51.50, -0.12), (-33.86, 151.20)]
        self.n_gs = len(self.gs_coords)
        
        # Actions
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
            lat, lon = rng.uniform(-60, 60), rng.uniform(-180, 180)
            self.targets_ecef.append(self._lld_to_ecef(lat*macros.D2R, lon*macros.D2R, 0))
        self.gs_ecef = []
        for lat, lon in self.gs_coords:
            self.gs_ecef.append(self._lld_to_ecef(lat*macros.D2R, lon*macros.D2R, 0))

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
        
        # AI quyết định mỗi 10s. Lực đẩy sẽ được duy trì trong 10s này.
        self.decision_dt = 10.0      
        self.dt = macros.sec2nano(1.0) 
        
        self.scSim.CreateNewProcess("dynProcess").addTask(self.scSim.CreateNewTask(self.taskName, self.dt))

        self.all_viz_objects = [] 
        self.sats = []
        self.states = []
        
        # [MỚI] Danh sách quản lý message lực đẩy cho từng vệ tinh
        self.force_msgs = []
        self.force_payloads = []
        
        # --- 1. GRAVITY ---
        gravFactory = simIncludeGravBody.gravBodyFactory()
        earth = gravFactory.createEarth(); earth.isCentralBody = True
        self.planet_mu = earth.mu
        
        # Thử load J2
        import os, Basilisk
        bskPath = os.path.dirname(Basilisk.__file__)
        gravFilePath = os.path.join(bskPath, "supportData", "LocalGravData", "GGM03S.txt")
        try:
            if os.path.exists(gravFilePath): earth.useSphericalHarmonicsGravityModel(gravFilePath, 10)
        except: pass

        # --- 2. ATMOSPHERE ---
        atmo = exponentialAtmosphere.ExponentialAtmosphere()
        atmo.ModelTag = "ExpAtmo"
        atmo.planetRadius = self.earth_radius
        atmo.scaleHeight = 7200.0; atmo.baseDensity = 4e-13
        self.scSim.AddModelToTask(self.taskName, atmo)

        # --- A. TẠO VỆ TINH ---
        for i in range(self.n_sats):
            sc = spacecraft.Spacecraft()
            sc.ModelTag = f"Sat_{i}"
            sc.hub.mHub = 500.0 # [QUAN TRỌNG] Đặt khối lượng để tính lực F=ma
            
            self.scSim.AddModelToTask(self.taskName, sc)
            gravFactory.addBodiesTo(sc)
            atmo.addSpacecraftToModel(sc.scStateOutMsg)
            
            # 1. DRAG EFFECTOR
            dragEffector = dragDynamicEffector.DragDynamicEffector()
            dragEffector.ModelTag = f"Drag_{i}"
            dragEffector.coreParams.projectedArea = 2.0; dragEffector.coreParams.dragCoeff = 2.2
            try: dragEffector.setDensityMessage(atmo.envOutMsgs[i])
            except: 
                if hasattr(dragEffector, 'atmoDensInMsg'): dragEffector.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[i])
            sc.addDynamicEffector(dragEffector)
            self.scSim.AddModelToTask(self.taskName, dragEffector)
            
            # 2. [MỚI] EXTERNAL FORCE EFFECTOR (Động cơ ảo)
            extForce = extForceTorque.ExtForceTorque()
            extForce.ModelTag = f"ExtForce_{i}"
            sc.addDynamicEffector(extForce)
            self.scSim.AddModelToTask(self.taskName, extForce)
            
            # Tạo Message điều khiển lực
            cmdPayload = messaging.CmdForceInertialMsgPayload()
            cmdPayload.forceRequestInertial = [0.0, 0.0, 0.0] # Mặc định tắt
            cmdMsg = messaging.CmdForceInertialMsg().write(cmdPayload)
            
            # Nối Message vào Effector
            extForce.cmdForceInertialInMsg.subscribeTo(cmdMsg)
            
            # Lưu lại để dùng trong hàm step()
            self.force_msgs.append(cmdMsg)
            self.force_payloads.append(cmdPayload)
            
            # Quỹ đạo Init
            oe = orbitalMotion.ClassicElements()
            oe.a = 7000e3; oe.e = 0.001; oe.i = 45.0 * macros.D2R
            oe.Omega = (i * 360.0 / self.n_sats) * macros.D2R; oe.omega = 0.0; oe.f = 0.0
            rN, vN = orbitalMotion.elem2rv(self.planet_mu, oe)
            sc.hub.r_CN_NInit = np.array(rN); sc.hub.v_CN_NInit = np.array(vN)
            
            self.sats.append(sc)
            self.all_viz_objects.append(sc)
            
            bore_vec = -np.array(rN) / np.linalg.norm(rN)
            self.states.append({'fuel': self.fuel_max, 'batt': self.batt_max, 'buffer': 0, 'bore_vec': bore_vec})

        # --- B. DUMMY OBJECTS ---
        # Targets
        for k, ecef in enumerate(self.targets_ecef):
            dummy = spacecraft.Spacecraft()
            dummy.ModelTag = f"TGT_{k}"
            r_mag = np.linalg.norm(ecef)
            dummy.hub.r_CN_NInit = ecef * ((r_mag + 50000.0) / r_mag)
            self.scSim.AddModelToTask(self.taskName, dummy)
            self.all_viz_objects.append(dummy)

        # Ground Stations
        gs_names = ["Hanoi", "NY", "London", "Sydney"]
        for k, ecef in enumerate(self.gs_ecef):
            dummy = spacecraft.Spacecraft()
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
                    saveFile="mission_sim_advanced_final.bin"
                )
                viz.settings.showSpacecraftLabels = 1 
                # Bật đường quỹ đạo để thấy rõ khi thay đổi
                viz.settings.showOrbitLines = 1
                self.vizObj = viz
            except Exception as e: self.vizObj = None

        self.scSim.InitializeSimulation()
        self.sim_time = 0.0

    def reset(self):
        self.global_captured_targets = set()
        self._init_simulator()
        return self._get_all_obs()

    def _get_all_obs(self):
        obs_list = []
        for i, sat in enumerate(self.sats):
            # Đọc state message an toàn
            stateMsg = sat.scStateOutMsg.read()
            r = np.array(stateMsg.r_BN_N) / 1e7
            v = np.array(stateMsg.v_BN_N) / 1e4
            f = [self.states[i]['fuel'] / self.fuel_max]
            b = [self.states[i]['batt'] / self.batt_max]
            d = [self.states[i]['buffer'] / self.buffer_max]
            obs = np.concatenate([r, v, f, b, d]).astype(np.float32)
            obs_list.append(obs)
        return obs_list

    def step(self, actions, debug_mode=True):
        rewards = np.zeros(self.n_sats)
        dones = [False] * self.n_sats
        
        # Thời gian hiện tại (nanosecond) cho message
        current_nano = macros.sec2nano(self.sim_time)
        
        for i, action in enumerate(actions):
            sat = self.sats[i]
            state = self.states[i]
            act = int(action)
            
            # 0. MẶC ĐỊNH: TẮT ĐỘNG CƠ (Force = 0)
            # Mỗi step phải reset lại, nếu không vệ tinh bay mất tích
            self.force_payloads[i].forceRequestInertial = [0.0, 0.0, 0.0]
            self.force_msgs[i].write(self.force_payloads[i], current_nano)

            # Check chết
            if state['batt'] <= 0 or state['fuel'] <= 0:
                rewards[i] -= 10.0; dones[i] = True
                continue

            # Lấy vector hiện tại từ Message (Cách chuẩn mới)
            scMsg = sat.scStateOutMsg.read()
            v_curr = np.array(scMsg.v_BN_N)
            r_curr = np.array(scMsg.r_BN_N)
            
            v_mag = np.linalg.norm(v_curr)
            v_dir = v_curr / (v_mag + 1e-9)
            
            # --- 1. SẠC PIN ---
            if act == self.ACT_CHARGE:
                state['batt'] = min(self.batt_max, state['batt'] + 5.0)
                rewards[i] += 0.1
                if debug_mode and state['batt'] < 90:
                    print(f"[t={self.sim_time:.0f}] 🔋 Sat {i} Charging...")

            # --- 2. THAY ĐỔI QUỸ ĐẠO (DÙNG NGOẠI LỰC - CONTINUOUS FORCE) ---
            elif act in [self.ACT_ALT_UP, self.ACT_ALT_DOWN, self.ACT_INC_UP, self.ACT_INC_DOWN]:
                if state['fuel'] >= 2.0:
                    delta_v = 0.0
                    dv_vec = np.zeros(3)
                    cost = 2.0
                    
                    # Tính toán hướng vector
                    h_vec = np.cross(r_curr, v_curr)
                    h_dir = h_vec / np.linalg.norm(h_vec)
                    
                    # A. Đổi độ cao
                    if act == self.ACT_ALT_UP:
                        # Tăng tốc 10 m/s trong vòng 10 giây
                        dv_vec = v_dir; delta_v = 10.0; 
                        if debug_mode: print(f"[t={self.sim_time:.0f}] 🚀 Sat {i} ALT UP (Burn)")
                    
                    elif act == self.ACT_ALT_DOWN:
                        dv_vec = -v_dir; delta_v = 10.0
                        if debug_mode: print(f"[t={self.sim_time:.0f}] 🔻 Sat {i} ALT DOWN (Burn)")

                    # B. Đổi góc nghiêng (Visual hoành tráng)
                    elif act in [self.ACT_INC_UP, self.ACT_INC_DOWN]:
                        # Tính delta-v cần thiết để đổi 1 độ trong 10s (Demo)
                        # Thực tế phức tạp hơn, nhưng đây dùng lực đẩy vector Normal
                        delta_v = 50.0 
                        cost = 5.0
                        if act == self.ACT_INC_UP: 
                            dv_vec = h_dir # Hướng Normal (Lên cực)
                            if debug_mode: print(f"[t={self.sim_time:.0f}] 📐 Sat {i} INC UP (Normal Burn)")
                        else: 
                            dv_vec = -h_dir # Hướng Anti-Normal (Xuống xích đạo)
                            if debug_mode: print(f"[t={self.sim_time:.0f}] 📐 Sat {i} INC DOWN (Anti-Normal)")

                    # [CỐT LÕI] TÍNH LỰC ĐẨY CẦN THIẾT
                    # F = (m * delta_v) / dt
                    # Lực này sẽ được áp dụng liên tục trong suốt 10 giây của bước nhảy
                    mass = sat.hub.mHub # 500kg
                    force_mag = (mass * delta_v) / self.decision_dt
                    force_vector = dv_vec * force_mag
                    
                    # Gửi lệnh Lực vào Message
                    self.force_payloads[i].forceRequestInertial = force_vector.tolist()
                    self.force_msgs[i].write(self.force_payloads[i], current_nano)
                    
                    state['fuel'] -= cost
                    state['batt'] -= 2.0
                    rewards[i] -= 0.1
                    if debug_mode:
                        print(f"-> Force Applied: {force_vector} N for Δv={delta_v} m/s")
                else:
                    rewards[i] -= 0.5 # Hết xăng
                    if debug_mode: print(f"[t={self.sim_time:.0f}] ⚠️ Sat {i} Not Enough Fuel for Orbit Change")

            # --- 3. CHỤP ẢNH ---
            elif 0 <= act < self.n_targets:
                state['batt'] -= 0.5
                if act in self.global_captured_targets:
                    rewards[i] -= 0.1
                    if debug_mode: print(f"[t={self.sim_time:.0f}] ⚠️ Sat {i} T{act} already captured by another sat.")
                else:
                    tgt_pos = self.targets_ecef[act]
                    req_vec = tgt_pos - r_curr
                    dist = np.linalg.norm(req_vec); req_vec /= (dist + 1e-9)
                    
                    # Slew Logic
                    cos = np.dot(state['bore_vec'], req_vec)
                    angle = math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))
                    max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                    
                    if angle <= max_turn:
                        state['bore_vec'] = req_vec
                        if state['buffer'] < self.buffer_max and dist < 4000e3:
                            self.global_captured_targets.add(act)
                            state['buffer'] += 1
                            rewards[i] += 5.0
                            if debug_mode: print(f"[t={self.sim_time:.0f}] 📸 Sat {i} CAPTURED T{act}")
                        else: rewards[i] -= 0.1
                    else:
                        ratio = max_turn / angle
                        state['bore_vec'] = (1-ratio)*state['bore_vec'] + ratio*req_vec
                        state['bore_vec'] /= np.linalg.norm(state['bore_vec'])

            # --- 4. DOWNLINK ---
            elif self.n_targets <= act < self.n_targets + self.n_gs:
                state['batt'] -= 0.1
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
                        print(f"[t={self.sim_time:.0f}] 🔄 Sat {i} Slew GS {gs_idx} ({angle:.0f}°)")
            else: 
                state['batt'] -= 0.1

        # Step Sim
        stop_time = self.sim_time + self.decision_dt
        self.scSim.ConfigureStopTime(int(macros.sec2nano(stop_time)))
        self.scSim.ExecuteSimulation()
        self.sim_time = stop_time
        
        if self.sim_time > 8000: dones = [True] * self.n_sats
            
        return self._get_all_obs(), rewards, dones, {}

# --- 4. TRAINER & MAIN ---
def train_mission():
    N_SATS = 4
    N_TARGETS = 50
    
    print("--- 1. TRAINING (No Viz) ---")
    env = BasiliskFullMissionEnv(n_sats=N_SATS, n_targets=N_TARGETS, viz=True)
    agent = MultiAgentActorCritic(env.obs_dim, env.global_state_dim, env.action_dim)
    opt = optim.Adam(agent.parameters(), lr=1e-3)
    
    for ep in range(5): # Demo 5 ep
        obs = env.reset(); done = False; ep_rw = 0
        while not done:
            acts = []
            for i in range(N_SATS):
                l = agent.act(torch.FloatTensor(obs[i]).unsqueeze(0))
                acts.append(torch.distributions.Categorical(logits=l).sample().item())
            obs, r, d, _ = env.step(acts, debug_mode=False)
            ep_rw += np.sum(r); done = any(d)
        print(f"Epoch {ep+1} | Reward: {ep_rw:.1f}")

    print("\n--- 2. DEMO (Viz Mode) ---")
    env = None; gc.collect()
    demo_env = BasiliskFullMissionEnv(n_sats=N_SATS, n_targets=N_TARGETS, viz=True)
    obs = demo_env.reset(); done = False
    
    try:
        while not done:
            acts = []
            for i in range(N_SATS):
                with torch.no_grad():
                    l = agent.act(torch.FloatTensor(obs[i]).unsqueeze(0))
                    acts.append(torch.argmax(l, dim=1).item())
            obs, _, d, _ = demo_env.step(acts, debug_mode=True)
            done = any(d)
    except KeyboardInterrupt: pass
    print("Done. Check 'mission_sim_advanced_final.bin'")

if __name__ == "__main__":
    train_mission()