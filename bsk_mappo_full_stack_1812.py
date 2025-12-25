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
    from Basilisk.simulation import extForceTorque, dragDynamicEffector, exponentialAtmosphere
    from Basilisk.architecture import messaging 
    print("[INIT] Basilisk libraries loaded.")
except ImportError:
    print("[ERROR] Basilisk not found.")
    sys.exit(1)

# --- 2. MAPPO NETWORK ---
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

# --- 3. ENVIRONMENT (STRATEGIC ORBIT CHANGE) ---
class BasiliskFullMissionEnv:
    def __init__(self, n_sats=4, n_targets=50, viz=False):
        self.n_sats = n_sats
        self.n_targets = n_targets
        self.viz = viz
        
        # Physics
        self.earth_radius = 6371e3
        self.mu = 398600.4418e9
        
        # Config
        self.max_slew_rate = math.radians(5.0) 
        self.fuel_max = 200.0
        self.batt_max = 100.0
        self.buffer_max = 5 
        
        self.global_captured_targets = set()
        
        # Trạm mặt đất
        self.gs_coords = [(21.02, 105.85), (40.71, -74.00), (51.50, -0.12), (-33.86, 151.20)]
        self.n_gs = len(self.gs_coords)
        
        # --- [NEW] CÁC LỰA CHỌN QUỸ ĐẠO CỐ ĐỊNH ---
        # AI sẽ chọn nhảy vào 1 trong 3 quỹ đạo này
        self.ORBIT_CHOICES = [10.0, 45.0, 90.0] 
        
        # ACTION SPACE
        # 0..49: Capture Target
        # 50..53: Downlink GS
        # 54: Charge
        # 55: Alt Up (+10km)
        # 56: Alt Down (-10km)
        # 57: Go to 10 deg
        # 58: Go to 45 deg
        # 59: Go to 90 deg
        
        self.ACT_CHARGE = self.n_targets + self.n_gs
        self.ACT_ALT_UP = self.ACT_CHARGE + 1
        self.ACT_ALT_DOWN = self.ACT_CHARGE + 2
        
        # Các hành động chọn quỹ đạo
        self.ACT_GOTO_INC_START = self.ACT_ALT_DOWN + 1
        self.n_orbit_choices = len(self.ORBIT_CHOICES)
        
        self.action_dim = self.ACT_GOTO_INC_START + self.n_orbit_choices
        
        # Observation: [Pos(3), Vel(3), Fuel(1), Batt(1), Buffer(1), Current_Inc(1)]
        # Thêm Current Inclination để AI biết trạng thái hiện tại
        self.obs_dim = 10 
        self.global_state_dim = self.obs_dim * self.n_sats 

        self._build_locations()
        self._init_simulator()

    def _build_locations(self):
        rng = np.random.RandomState(42)
        self.targets_ecef = []
        for _ in range(self.n_targets):
            # Rải target từ Xích đạo (-10) đến Cực (90) để bắt AI phải đổi quỹ đạo
            lat = rng.uniform(-10, 85) # Latitude range
            lon = rng.uniform(-180, 180)
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
        self.decision_dt = 10.0      
        self.dt = macros.sec2nano(2.0) 
        self.scSim.CreateNewProcess("dynProcess").addTask(self.scSim.CreateNewTask(self.taskName, self.dt))

        self.all_viz_objects = [] 
        self.sats = []
        self.states = []
        self.force_msgs = []
        self.force_payloads = []
        
        gravFactory = simIncludeGravBody.gravBodyFactory()
        earth = gravFactory.createEarth(); earth.isCentralBody = True
        self.planet_mu = earth.mu
        
        # Load Gravity & Atmosphere (Giản lược để code gọn, giữ nguyên logic cũ)
        atmo = exponentialAtmosphere.ExponentialAtmosphere()
        atmo.ModelTag = "ExpAtmo"; atmo.planetRadius = self.earth_radius; atmo.scaleHeight = 7200.0; atmo.baseDensity = 4e-13
        self.scSim.AddModelToTask(self.taskName, atmo)

        # --- A. TẠO VỆ TINH ---
        for i in range(self.n_sats):
            sc = spacecraft.Spacecraft()
            sc.ModelTag = f"Sat_{i}"
            sc.hub.mHub = 500.0 
            self.scSim.AddModelToTask(self.taskName, sc)
            gravFactory.addBodiesTo(sc)
            atmo.addSpacecraftToModel(sc.scStateOutMsg)
            
            # Drag
            dragEffector = dragDynamicEffector.DragDynamicEffector()
            dragEffector.ModelTag = f"Drag_{i}"
            dragEffector.coreParams.projectedArea = 2.0; dragEffector.coreParams.dragCoeff = 2.2
            try: dragEffector.setDensityMessage(atmo.envOutMsgs[i])
            except: 
                if hasattr(dragEffector, 'atmoDensInMsg'): dragEffector.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[i])
            sc.addDynamicEffector(dragEffector)
            self.scSim.AddModelToTask(self.taskName, dragEffector)
            
            # External Force (Động cơ ảo)
            extForce = extForceTorque.ExtForceTorque()
            extForce.ModelTag = f"ExtForce_{i}"
            sc.addDynamicEffector(extForce)
            self.scSim.AddModelToTask(self.taskName, extForce)
            
            # Messaging Lực
            cmdPayload = messaging.CmdForceInertialMsgPayload()
            cmdPayload.forceRequestInertial = [0.0, 0.0, 0.0]
            cmdMsg = messaging.CmdForceInertialMsg().write(cmdPayload)
            extForce.cmdForceInertialInMsg.subscribeTo(cmdMsg)
            self.force_msgs.append(cmdMsg); self.force_payloads.append(cmdPayload)
            
            # Init Orbit (Bắt đầu ở 45 độ)
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
        for k, ecef in enumerate(self.targets_ecef):
            dummy = spacecraft.Spacecraft()
            dummy.ModelTag = f"TGT_{k}"
            r_mag = np.linalg.norm(ecef)
            dummy.hub.r_CN_NInit = ecef * ((r_mag + 50000.0) / r_mag)
            self.scSim.AddModelToTask(self.taskName, dummy)
            self.all_viz_objects.append(dummy)

        gs_names = ["Hanoi", "NY", "London", "Sydney"]
        for k, ecef in enumerate(self.gs_ecef):
            dummy = spacecraft.Spacecraft()
            dummy.ModelTag = f"GS_{gs_names[k]}"
            r_mag = np.linalg.norm(ecef)
            dummy.hub.r_CN_NInit = ecef * ((r_mag + 400000.0) / r_mag)
            self.scSim.AddModelToTask(self.taskName, dummy)
            self.all_viz_objects.append(dummy)

        if self.viz:
            try:
                viz = vizSupport.enableUnityVisualization(self.scSim, self.taskName, self.all_viz_objects, saveFile="mission_sim_strategic.bin")
                viz.settings.showSpacecraftLabels = 1; viz.settings.showOrbitLines = 1
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
            stateMsg = sat.scStateOutMsg.read()
            r = np.array(stateMsg.r_BN_N) / 1e7
            v = np.array(stateMsg.v_BN_N) / 1e4
            
            # Tính góc nghiêng hiện tại để đưa vào Observation
            # AI cần biết: "Mình đang ở 10 độ hay 90 độ?"
            oe = orbitalMotion.rv2elem(self.planet_mu, np.array(stateMsg.r_BN_N), np.array(stateMsg.v_BN_N))
            inc_norm = oe.i / (90.0 * macros.D2R) # Chuẩn hóa 0..1
            
            f = [self.states[i]['fuel'] / self.fuel_max]
            b = [self.states[i]['batt'] / self.batt_max]
            d = [self.states[i]['buffer'] / self.buffer_max]
            
            # Obs mới có 10 chiều
            obs = np.concatenate([r, v, f, b, d, [inc_norm]]).astype(np.float32)
            obs_list.append(obs)
        return obs_list

    def step(self, actions, debug_mode=True):
        rewards = np.zeros(self.n_sats)
        dones = [False] * self.n_sats
        current_nano = macros.sec2nano(self.sim_time)
        
        for i, action in enumerate(actions):
            sat = self.sats[i]
            state = self.states[i]
            act = int(action)
            
            # Reset Lực = 0
            self.force_payloads[i].forceRequestInertial = [0.0, 0.0, 0.0]
            self.force_msgs[i].write(self.force_payloads[i], current_nano)

            if state['batt'] <= 0 or state['fuel'] <= 0:
                rewards[i] -= 10.0; dones[i] = True
                print(f"[t={self.sim_time:.0f}] 💀 Sat {i} DEAD Because " + ("BATTERY" if state['batt'] <= 0 else "FUEL"))
                continue

            scMsg = sat.scStateOutMsg.read()
            v_curr = np.array(scMsg.v_BN_N)
            r_curr = np.array(scMsg.r_BN_N)
            v_dir = v_curr / (np.linalg.norm(v_curr) + 1e-9)
            
            # --- 1. SẠC PIN ---
            if act == self.ACT_CHARGE:
                state['batt'] = min(self.batt_max, state['batt'] + 5.0)
                rewards[i] += 0.1
                if debug_mode and state['batt'] < 90:
                    print(f"[t={self.sim_time:.0f}] 🔋 Sat {i} CHARGE -> {state['batt']:.1f}%")

            # --- 2. ALTITUDE CHANGE ---
            elif act in [self.ACT_ALT_UP, self.ACT_ALT_DOWN]:
                 if state['fuel'] >= 1.0:
                    delta_v = 10.0
                    dv_vec = v_dir if act == self.ACT_ALT_UP else -v_dir
                    
                    mass = sat.hub.mHub
                    force_mag = (mass * delta_v) / self.decision_dt
                    force_vector = dv_vec * force_mag
                    
                    self.force_payloads[i].forceRequestInertial = force_vector.tolist()
                    self.force_msgs[i].write(self.force_payloads[i], current_nano)
                    
                    state['fuel'] -= 1.0; state['batt'] -= 1.0
                    if debug_mode: print(f"[t={self.sim_time:.0f}] Sat {i} ALT Change")

            # --- 3. [NEW] STRATEGIC ORBIT CHANGE (GOTO INC) ---
            elif self.ACT_GOTO_INC_START <= act < self.ACT_GOTO_INC_START + self.n_orbit_choices:
                if state['fuel'] >= 5.0:
                    # Lấy góc mục tiêu từ Action
                    choice_idx = act - self.ACT_GOTO_INC_START
                    target_inc_deg = self.ORBIT_CHOICES[choice_idx] # 10, 45, hoặc 90
                    target_inc_rad = target_inc_deg * macros.D2R
                    
                    # Tính góc hiện tại
                    oe = orbitalMotion.rv2elem(self.planet_mu, r_curr, v_curr)
                    current_inc_rad = oe.i
                    
                    # Tính độ chênh lệch
                    inc_diff = target_inc_rad - current_inc_rad
                    
                    # Nếu chênh lệch quá nhỏ (< 2 độ), coi như đã đến nơi -> Không làm gì (Tiết kiệm xăng)
                    if abs(inc_diff) < 2.0 * macros.D2R:
                        rewards[i] += 0.1 # Thưởng nhẹ vì đang ở đúng chỗ
                        print(f"[t={self.sim_time:.0f}] ✅ Sat {i} REACHED {target_inc_deg}° Orbit")
                    else:
                        # Cần bẻ lái
                        # Tính hướng Normal
                        h_vec = np.cross(r_curr, v_curr)
                        h_dir = h_vec / np.linalg.norm(h_vec)
                        
                        # Nếu cần tăng góc (diff > 0) -> Đốt Normal (h_dir)
                        # Nếu cần giảm góc (diff < 0) -> Đốt Anti-Normal (-h_dir)
                        burn_dir = h_dir if inc_diff > 0 else -h_dir
                        
                        # Tính delta_v cần thiết
                        # Công thức xấp xỉ: dv = 2*v*sin(d_inc/2)
                        # Ở đây mô phỏng, ta dùng lực đẩy tỷ lệ thuận với độ lệch để AI hội tụ từ từ
                        # Giới hạn max delta_v mỗi lần quyết định là 50 m/s để không giật cục quá
                        v_mag = np.linalg.norm(v_curr)
                        req_dv = 2 * v_mag * math.sin(abs(inc_diff) / 2)
                        apply_dv = min(req_dv, 50.0) 
                        
                        force_mag = (sat.hub.mHub * apply_dv) / self.decision_dt
                        force_vector = burn_dir * force_mag
                        
                        self.force_payloads[i].forceRequestInertial = force_vector.tolist()
                        self.force_msgs[i].write(self.force_payloads[i], current_nano)
                        
                        cost = 5.0
                        state['fuel'] -= cost
                        state['batt'] -= 2.0
                        if debug_mode: 
                            print(f"[t={self.sim_time:.0f}] 📐 Sat {i} GOTO {target_inc_deg}° (Curr: {current_inc_rad*macros.R2D:.1f}°) -> Burn dV={apply_dv:.1f}m/s")
                else:
                    rewards[i] -= 0.5 # Hết xăng

            # --- 4. CAPTURE ---
            elif 0 <= act < self.n_targets:
                state['batt'] -= 0.5
                if act in self.global_captured_targets:
                    rewards[i] -= 0.1
                else:
                    tgt_pos = self.targets_ecef[act]
                    req_vec = tgt_pos - r_curr
                    dist = np.linalg.norm(req_vec); req_vec /= (dist + 1e-9)
                    
                    cos = np.dot(state['bore_vec'], req_vec)
                    angle = math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))
                    max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                    
                    if angle <= max_turn:
                        state['bore_vec'] = req_vec
                        # [LOGIC QUAN TRỌNG] Kiểm tra góc nhìn
                        # Nếu vệ tinh ở Xích đạo (0 độ) mà đòi chụp Bắc Cực (90 độ)
                        # Khoảng cách sẽ rất xa và góc nhìn (nadir) sẽ bị khuất
                        if state['buffer'] < self.buffer_max and dist < 4000e3:
                            self.global_captured_targets.add(act)
                            state['buffer'] += 1
                            rewards[i] += 10.0 # Tăng thưởng lên để AI ham chụp
                            if debug_mode: print(f"[t={self.sim_time:.0f}] 📸 Sat {i} CAPTURED T{act}")
                        else: rewards[i] -= 0.1 # Xa quá không chụp được (Bắt buộc phải đổi quỹ đạo mới chụp dc)
                    else:
                        ratio = max_turn / angle
                        state['bore_vec'] = (1-ratio)*state['bore_vec'] + ratio*req_vec
                        state['bore_vec'] /= np.linalg.norm(state['bore_vec'])

            # --- 5. DOWNLINK ---
            elif self.n_targets <= act < self.n_targets + self.n_gs:
                state['batt'] -= 0.5
                gs_idx = act - self.n_targets
                gs_pos = self.gs_ecef[gs_idx]
                req_vec = gs_pos - r_curr
                dist = np.linalg.norm(req_vec); req_vec /= (dist + 1e-9)
                
                cos = np.dot(state['bore_vec'], req_vec)
                angle = math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))
                max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                
                if angle <= max_turn:
                    state['bore_vec'] = req_vec
                    if state['buffer'] > 0 and dist < 3000e3:
                        state['buffer'] -= 1
                        rewards[i] += 20.0
                        if debug_mode: print(f"[t={self.sim_time:.0f}] 📡 Sat {i} DOWNLINK (+20)")
                    else: rewards[i] -= 0.1
                else:
                    ratio = max_turn / angle
                    state['bore_vec'] = (1-ratio)*state['bore_vec'] + ratio*req_vec
                    state['bore_vec'] /= np.linalg.norm(state['bore_vec'])
            
            else: state['batt'] -= 0.1

        # Step Sim
        stop_time = self.sim_time + self.decision_dt
        self.scSim.ConfigureStopTime(int(macros.sec2nano(stop_time)))
        self.scSim.ExecuteSimulation()
        self.sim_time = stop_time
        
        if self.sim_time > 6000: dones = [True] * self.n_sats
        return self._get_all_obs(), rewards, dones, {}

# --- 4. TRAINER ---
def train_mission():
    N_SATS = 4
    N_TARGETS = 50
    
    print("--- 1. TRAINING (AI ĐANG HỌC CÁCH CHỌN QUỸ ĐẠO) ---")
    env = BasiliskFullMissionEnv(n_sats=N_SATS, n_targets=N_TARGETS, viz=False)
    agent = MultiAgentActorCritic(env.obs_dim, env.global_state_dim, env.action_dim)
    opt = optim.Adam(agent.parameters(), lr=1e-3)
    
    for ep in range(5): 
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
    print("Done. Check 'mission_sim_strategic.bin'")

if __name__ == "__main__":
    train_mission()