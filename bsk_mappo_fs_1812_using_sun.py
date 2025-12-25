import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import Basilisk

# --- 1. BASILISK IMPORTS ---
try:
    from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion, vizSupport
    from Basilisk.simulation import spacecraft
    from Basilisk.simulation import extForceTorque # Chỉ dùng ExtForce, tắt Drag/Atmo để an toàn
    from Basilisk.architecture import messaging 
    from Basilisk.simulation import spiceInterface
    print("[INIT] Basilisk libraries loaded.")
except Exception as e:
    print("[ERROR] Basilisk not found. ", e)
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

# --- 3. ENVIRONMENT (FINAL VERSION) ---
class BasiliskFullMissionEnv:
    def __init__(self, n_sats=4, n_targets=50, viz=False):
        self.n_sats = n_sats
        self.n_targets = n_targets
        self.viz = viz
        
        self.earth_radius = 6371e3
        self.mu = 398600.4418e9
        
        self.SLEW_RATE_DEG = 8.0 
        self.max_slew_rate = math.radians(self.SLEW_RATE_DEG)
        self.fuel_max = 200.0
        self.batt_max = 100.0
        self.buffer_max = 5 
        
        self.global_captured_targets = set()
        self.gs_coords = [(21.02, 105.85), (40.71, -74.00), (51.50, -0.12), (-33.86, 151.20)]
        self.n_gs = len(self.gs_coords)

        self.CAPTURE_MAX_DIST = 4500e3   # Cũ: 4000km -> Mới: 4500km
        self.CAPTURE_MAX_OFF_NADIR = 60.0 # Cũ: 45 độ -> Mới: 60 độ (dễ thở hơn nhiều)
        # Chiến lược quỹ đạo: 10, 45, 75 độ
        self.ORBIT_CHOICES = [10.0, 45.0, 75.0] 
        
        # Actions
        self.ACT_CHARGE = self.n_targets + self.n_gs
        self.ACT_ALT_UP = self.ACT_CHARGE + 1
        self.ACT_ALT_DOWN = self.ACT_CHARGE + 2
        self.ACT_GOTO_INC_START = self.ACT_ALT_DOWN + 1
        self.n_orbit_choices = len(self.ORBIT_CHOICES)
        self.action_dim = self.ACT_GOTO_INC_START + self.n_orbit_choices
        
        # Obs Dim 14: [Pos(3), Vel(3), Fuel(1), Batt(1), Buff(1), Inc(1), DistGS(1), DirGS(3)]
        self.obs_dim = 14 
        self.global_state_dim = self.obs_dim * self.n_sats 

        self._build_locations()
        self._init_simulator()

    def _build_locations(self):
        rng = np.random.RandomState(42)
        self.targets_ecef = []
        for _ in range(self.n_targets):
            lat, lon = rng.uniform(-10, 85), rng.uniform(-180, 180)
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
        self.decision_dt = 2.0  
        self.dt = macros.sec2nano(2.0) 
        self.scSim.CreateNewProcess("dynProcess").addTask(self.scSim.CreateNewTask(self.taskName, self.dt))

        self.all_viz_objects = [] 
        self.sats = []
        self.states = []
        
        # [QUAN TRỌNG] Reset danh sách này ngay đầu hàm
        self.force_msgs = []
        self.force_payloads = [] 
        
        self.SUN_DIRECTION = np.array([1.0, 0.0, 0.0])
    
# --- CÀI ĐẶT SPICE (Lấy dữ liệu thiên văn thật) ---
# --- [FIXED] SPICE SETUP (ĐƯỜNG DẪN CỨNG) ---
        self.spiceObject = spiceInterface.SpiceInterface()
        self.spiceObject.ModelTag = "SpiceInterface"
        
        self.spiceObject.zeroBase = "Earth"
        # Đường dẫn chính xác bạn đã cung cấp
        # Lưu ý: Dùng r"..." để Python hiểu là raw string (tránh lỗi dấu \)
        spiceDataPath = r"D:\ProjectIII\Code\Multi-Sat-MARL-2025\basilisk\supportData\EphemerisData"
        kernelName = "de430.bsp"
        
        # Kiểm tra lần cuối cho chắc ăn
        full_path = os.path.join(spiceDataPath, kernelName)
        if not os.path.exists(full_path):
            print(f"[CRITICAL ERROR] Vẫn không thấy file tại: {full_path}")
            print("Hãy kiểm tra lại xem file de430.bsp có thực sự nằm ở đó không!")
            sys.exit(1)
            
        print(f"[INFO] Đang load SPICE kernel từ: {full_path}")
        self.spiceObject.SPICEDataPath = spiceDataPath
        self.spiceObject.addPlanetNames(["sun", "earth"])
        self.spiceObject.loadSpiceKernel(kernelName, spiceDataPath)
        
        # Cài đặt thời gian bắt đầu (Ví dụ: Năm 2025)
        self.spiceObject.UTCCalInit = "2025 December 18 00:00:00.0"
        
        self.scSim.AddModelToTask(self.taskName, self.spiceObject)
        
        gravFactory = simIncludeGravBody.gravBodyFactory()
        earth = gravFactory.createEarth(); earth.isCentralBody = True
        self.planet_mu = earth.mu
        
        earth.planetBodyInMsg.subscribeTo(self.spiceObject.planetStateOutMsgs[1])
        
        # --- A. TẠO VỆ TINH ---
        for i in range(self.n_sats):
            sc = spacecraft.Spacecraft()
            sc.ModelTag = f"Sat_{i}"
            sc.hub.mHub = 500.0 
            self.scSim.AddModelToTask(self.taskName, sc)
            gravFactory.addBodiesTo(sc)
            
            # EXT FORCE SETUP
            extForce = extForceTorque.ExtForceTorque()
            extForce.ModelTag = f"ExtForce_{i}"
            sc.addDynamicEffector(extForce)
            self.scSim.AddModelToTask(self.taskName, extForce)
            
            # [FIX LỖI INDEX ERROR] Append ngay trong vòng lặp
            cmdPayload = messaging.CmdForceInertialMsgPayload()
            cmdPayload.forceRequestInertial = [0.0, 0.0, 0.0]
            cmdMsg = messaging.CmdForceInertialMsg().write(cmdPayload)
            
            extForce.cmdForceInertialInMsg.subscribeTo(cmdMsg)
            
            self.force_msgs.append(cmdMsg)
            self.force_payloads.append(cmdPayload)
            
            # ORBIT SETUP
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
                self.vizTaskName = "vizTask"
                self.scSim.CreateNewProcess("vizProcess").addTask(self.scSim.CreateNewTask(self.vizTaskName, macros.sec2nano(10.0)))
                viz = vizSupport.enableUnityVisualization(
                    self.scSim, 
                    self.vizTaskName,  # <--- SỬA CHỖ NÀY
                    self.all_viz_objects, 
                    saveFile="mission_sim_optimize_reward_sun.bin"
                )
                                
                viz.settings.showSpacecraftLabels = 1
                viz.settings.showOrbitLines = 1                                
                # (Tùy chọn) Hiển thị ngày giờ thực từ SPICE
                # viz.epochInMsg.subscribeTo(self.spiceObject.epochProtoOutMsg)         
                self.vizObj = viz
            except Exception as e: 
                print(f"Viz Error: {e}")
                self.vizObj = None
                
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
            r_sat = np.array(stateMsg.r_BN_N)
            v_sat = np.array(stateMsg.v_BN_N)
            
            # --- SMART OBS: TÌM TRẠM GẦN NHẤT ---
            min_dist = 1e9
            nearest_gs_vec = np.zeros(3)
            for gs_pos in self.gs_ecef:
                rel_vec = gs_pos - r_sat
                dist = np.linalg.norm(rel_vec)
                if dist < min_dist:
                    min_dist = dist
                    nearest_gs_vec = rel_vec
            
            dist_norm = [min_dist / 10000e3] 
            if min_dist > 0: dir_norm = nearest_gs_vec / min_dist
            else: dir_norm = [0, 0, 0]

            try:
                oe = orbitalMotion.rv2elem(self.planet_mu, r_sat, v_sat)
                inc_norm = [oe.i / (90.0 * macros.D2R)]
            except: inc_norm = [0.5]
            
            r_norm = r_sat / 1e7; v_norm = v_sat / 1e4
            f = [self.states[i]['fuel'] / self.fuel_max]
            b = [self.states[i]['batt'] / self.batt_max]
            d = [self.states[i]['buffer'] / self.buffer_max]
            
            obs = np.concatenate([r_norm, v_norm, f, b, d, inc_norm, dist_norm, dir_norm]).astype(np.float32)
            obs_list.append(obs)
        return obs_list

    def step(self, actions, debug_mode=True):
        rewards = np.zeros(self.n_sats)
        dones = [False] * self.n_sats
        current_nano = macros.sec2nano(self.sim_time)
        
        # --- [NEW] CẬP NHẬT VECTOR MẶT TRỜI TỪ SPICE ---
        # SPICE trả về tọa độ trong hệ Inertial
        # msg[0] là Sun, msg[1] là Earth (do thứ tự addPlanetNames)
        sunMsg = self.spiceObject.planetStateOutMsgs[0].read()
        earthMsg = self.spiceObject.planetStateOutMsgs[1].read()
        
        r_Sun_N = np.array(sunMsg.PositionVector)
        r_Earth_N = np.array(earthMsg.PositionVector)
        
        # Vector từ Trái đất -> Mặt trời
        real_sun_vec = r_Sun_N - r_Earth_N
        # Cập nhật biến class để dùng cho logic bên dưới
        self.SUN_DIRECTION = real_sun_vec / (np.linalg.norm(real_sun_vec) + 1e-9)
        
        for i, action in enumerate(actions):
            sat = self.sats[i]
            state = self.states[i]
            act = int(action)
            
            # 1. Reset Lực = 0
            self.force_payloads[i].forceRequestInertial = [0.0, 0.0, 0.0]
            self.force_msgs[i].write(self.force_payloads[i], current_nano)

            # --- CHECK CHẾT & AN TOÀN ---
            scMsg = sat.scStateOutMsg.read()
            r_curr = np.array(scMsg.r_BN_N)
            v_curr = np.array(scMsg.v_BN_N)
            
            # Phạt nặng nếu rơi hoặc NaN
            if np.isnan(r_curr).any() or (np.linalg.norm(r_curr) - self.earth_radius) < 200e3:
                dones[i] = True; rewards[i] -= 50.0
                if debug_mode: print(f"[t={self.sim_time:.0f}] 💥 Sat {i} DESTROYED (r={np.linalg.norm(r_curr)/1e3:.1f} km)")
                continue
                
            
            if state['batt'] <= 0 or state['fuel'] <= 0:
                dones[i] = True; rewards[i] -= 20.0
                if debug_mode: print(f"[t={self.sim_time:.0f}] ⚠️ Sat {i} OUT OF RESOURCES (Fuel: {state['fuel']:.1f}, Batt: {state['batt']:.1f})")
                continue

            v_mag = np.linalg.norm(v_curr)
            v_dir = v_curr / (v_mag + 1e-9)

            # --- LOGIC HÀNH ĐỘNG ---
            
            # 1. CHARGE
# 1. CHARGE
            if act == self.ACT_CHARGE:
                # Kiểm tra xem VỆ TINH có đang ở ngoài sáng không
                sat_pos_dir = r_curr / np.linalg.norm(r_curr)
                sun_dir = self.SUN_DIRECTION
                sat_in_sun = np.dot(sat_pos_dir, sun_dir) > -0.1
                
                if sat_in_sun:
                    state['batt'] = min(self.batt_max, state['batt'] + 5.0)
                    if state['batt'] < 30.0: rewards[i] += 1.0
                    if debug_mode: print(f"[t={self.sim_time:.0f}] ⚡ Sat {i} CHARGE (+1) from {state['batt']-5:.1f} to {state['batt']:.1f}")
                else:
                    # Sạc trong bóng tối -> Vô dụng -> Phạt
                    rewards[i] -= 0.2
                    if debug_mode and i==0: print(f"[t={self.sim_time:.0f}] 🌑 Sat {i} Cannot Charge in Eclipse!")

            # 2. ORBIT CHANGE
            elif act in [self.ACT_ALT_UP, self.ACT_ALT_DOWN] or \
                 (self.ACT_GOTO_INC_START <= act < self.ACT_GOTO_INC_START + self.n_orbit_choices):
                
                # [STRATEGY] Nếu Buffer đầy, thưởng cho việc di chuyển (khuyến khích về trạm)
                if state['buffer'] >= self.buffer_max: 
                    rewards[i] += 2.0 
                    if debug_mode: print(f"[t={self.sim_time:.0f}] 🚀 Sat {i} Move (full buffer)")
                else: 
                    rewards[i] -= 0.1 # Phạt nhẹ nếu di chuyển lung tung khi rảnh
                    if debug_mode: print(f"[t={self.sim_time:.0f}] 🔄 Sat {i} Move (no buffer)")

                # Xử lý Lực
                if act in [self.ACT_ALT_UP, self.ACT_ALT_DOWN]:
                     if state['fuel'] >= 1.0:
                        dv_vec = v_dir if act == self.ACT_ALT_UP else -v_dir
                        force_vec = (sat.hub.mHub * 10.0) / self.decision_dt * dv_vec
                        self.force_payloads[i].forceRequestInertial = force_vec.tolist()
                        self.force_msgs[i].write(self.force_payloads[i], current_nano)
                        state['fuel'] -= 0.5 # Cost thấp để AI dám thử
                        if debug_mode: print(f"[t={self.sim_time:.0f}] 🔼🔽 Sat {i} Burn ALT")
                
                elif self.ACT_GOTO_INC_START <= act:
                    if state['fuel'] >= 5.0:
                        choice_idx = act - self.ACT_GOTO_INC_START
                        target_inc_rad = self.ORBIT_CHOICES[choice_idx] * macros.D2R
                        
                        oe = orbitalMotion.rv2elem(self.planet_mu, r_curr, v_curr)
                        inc_diff = target_inc_rad - oe.i
                        
                        if abs(inc_diff) < 2.0 * macros.D2R: rewards[i] += 0.5 # Đã ở đúng quỹ đạo
                        else:
                            h_vec = np.cross(r_curr, v_curr)
                            h_dir = h_vec / np.linalg.norm(h_vec)
                            burn_dir = h_dir if inc_diff > 0 else -h_dir
                            
                            req_dv = 2 * v_mag * math.sin(abs(inc_diff) / 2)
                            apply_dv = min(req_dv, 50.0)
                            
                            force_vec = (sat.hub.mHub * apply_dv) / self.decision_dt * burn_dir
                            self.force_payloads[i].forceRequestInertial = force_vec.tolist()
                            self.force_msgs[i].write(self.force_payloads[i], current_nano)
                            state['fuel'] -= 2.0 # Cost trung bình
                            if debug_mode: print(f"[t={self.sim_time:.0f}] 📐 Sat {i} Burn INC")

            # 3. CAPTURE

            elif 0 <= act < self.n_targets:
                state['batt'] -= 0.5
                
                if state['buffer'] >= self.buffer_max:
                    rewards[i] -= 2.0
                    if debug_mode: print(f"[t={self.sim_time:.0f}] ⚠️ Sat {i} Buffer FULL, cannot capture (-2.0)")
                else:
                    if act in self.global_captured_targets: rewards[i] -= 0.5
                    else:
                        tgt_pos = self.targets_ecef[act]
                        req_vec = tgt_pos - r_curr
                        dist = np.linalg.norm(req_vec); req_vec /= (dist + 1e-9)
                        
                        # --- [LOGIC MỚI] CHECK ÁNH SÁNG (DAY/NIGHT) ---
                        # Giả định Mặt trời ở hướng [1, 0, 0] (Hệ quán tính)
                        # Trong thực tế, bạn cần dùng module SPICE để lấy vị trí Mặt trời chính xác theo ngày giờ
                        sun_dir = self.SUN_DIRECTION
                        
                        # Vector từ Tâm Trái Đất -> Mục tiêu
                        target_normal = tgt_pos / np.linalg.norm(tgt_pos)
                        
                        # Tính tích vô hướng: Dương = Cùng phía Mặt trời (Ngày), Âm = Ngược phía (Đêm)
                        # Ta cho phép chụp lúc hoàng hôn/bình minh một chút (ngưỡng -0.1 thay vì 0.0)
                        is_daylight = np.dot(target_normal, sun_dir) > -0.1
                        
                        if not is_daylight:
                            # Phạt nhẹ vì chụp ảnh đen thùi lùi
                            rewards[i] -= 0.1 
                            if debug_mode: print(f"Sat {i} Target {act} is in DARKNESS (Night side)")
                        else:
                            # --- CÁC ĐIỀU KIỆN CŨ (Góc nhìn, Khoảng cách) ---
                            nadir_vec = -r_curr / np.linalg.norm(r_curr)
                            off_nadir = math.degrees(math.acos(np.clip(np.dot(req_vec, nadir_vec), -1, 1)))
                            
                            cos = np.dot(state['bore_vec'], req_vec)
                            angle = math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))
                            max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                            
                            if angle <= max_turn:
                                state['bore_vec'] = req_vec
                                # Tổng hợp 3 điều kiện: Gần + Góc đẹp + Ban ngày
                                if dist < 4500e3 and off_nadir < 60.0:
                                    self.global_captured_targets.add(act)
                                    state['buffer'] += 1
                                    rewards[i] += 10.0
                                    if debug_mode: print(f"[t={self.sim_time:.0f}] 📸 Sat {i} CAPTURED T{act} (Daylight)")
                                else: 
                                    rewards[i] -= 0.1
                                    if debug_mode: print(f"[t={self.sim_time:.0f}] ⚠️ Sat {i} T{act} Capture FAILED (Dist: {dist/1e3:.0f} km, Off-Nadir: {off_nadir:.0f}°)")
                            else:
                                ratio = max_turn / angle
                                state['bore_vec'] = (1-ratio)*state['bore_vec'] + ratio*req_vec
                                state['bore_vec'] /= np.linalg.norm(state['bore_vec'])
                                if debug_mode: print(f"[t={self.sim_time:.0f}] 🔄 Sat {i} Slewing to T{act} ({angle:.0f}°)")

            # 4. DOWNLINK
            elif self.n_targets <= act < self.n_targets + self.n_gs:
                state['batt'] -= 0.5
                gs_idx = act - self.n_targets
                gs_pos = self.gs_ecef[gs_idx]
                req_vec = gs_pos - r_curr
                dist = np.linalg.norm(req_vec); req_vec /= (dist + 1e-9)
                
                cos = np.dot(state['bore_vec'], req_vec)
                angle = math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))
                max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                
                is_full = (state['buffer'] >= self.buffer_max)
                if is_full: 
                    rewards[i] += 1.0 # Thưởng ý chí khi cố gắng downlink
                    if debug_mode: print(f"[t={self.sim_time:.0f}] 🚀 Sat {i} Move (full buffer)")
                
                if angle <= max_turn:
                    state['bore_vec'] = req_vec
                    if state['buffer'] > 0:
                        # SUCCESS
                        if dist < 4500e3:
                            state['buffer'] -= 1
                            rewards[i] += 50.0 # JACKPOT
                            if debug_mode: print(f"[t={self.sim_time:.0f}] 📡 Sat {i} DOWNLINK SUCCESS (+50)!!! CONGRATULATION ###")
                        # APPROACHING (Shaping Reward)
                        elif dist < 5000e3:
                            score = (5000e3 - dist) / (5000e3 - 4500e3)
                            rewards[i] += score * (2.0 if is_full else 0.5)
                            if debug_mode: print(f"[t={self.sim_time:.0f}] 📡 Sat {i} DOWNLINK APPROACHING (+{score * (2.0 if is_full else 0.5):.2f}) ")
                        else: 
                            rewards[i] -= 0.1
                            if debug_mode: print(f"[t={self.sim_time:.0f}] 📡 Sat {i} DOWNLINK TOO FAR (-0.1) THE DISTANCE IS {dist:.0f}")
                    else: 
                        rewards[i] -= 0.5 # Buffer rỗng
                        if debug_mode: print(f"[t={self.sim_time:.0f}] ⚠️ Sat {i} Buffer EMPTY, cannot downlink (-0.5)")
                else:
                    ratio = max_turn / angle
                    state['bore_vec'] = (1-ratio)*state['bore_vec'] + ratio*req_vec
                    state['bore_vec'] /= np.linalg.norm(state['bore_vec'])
                    rewards[i] += 0.05 
                    if debug_mode: print(f"[t={self.sim_time:.0f}] 🔄 Sat {i} Slewing to GS{gs_idx} ({angle:.0f}°)")
            
            # Drain nhẹ
            else:
                state['batt'] -= 0.05

        stop_time = self.sim_time + self.decision_dt
        self.scSim.ConfigureStopTime(int(macros.sec2nano(stop_time)))
        self.scSim.ExecuteSimulation()
        self.sim_time = stop_time
        
        if self.sim_time > 15000: dones = [True] * self.n_sats
        return self._get_all_obs(), rewards, dones, {}

# --- 4. TRAINER ---
def train_mission():
    N_SATS = 5
    N_TARGETS = 100
    
    print("--- 1. TRAINING (FINAL) ---")
    env = BasiliskFullMissionEnv(n_sats=N_SATS, n_targets=N_TARGETS, viz=False)
    agent = MultiAgentActorCritic(env.obs_dim, env.global_state_dim, env.action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4) # LR thấp, ổn định
    
    for ep in range(20): # Tăng Epoch
        obs_list = env.reset()
        ep_rewards = np.zeros(N_SATS)
        done = False
        
        # Buffer
        batch_obs, batch_gs, batch_acts, batch_rews = [], [], [], []
        
        while not done:
            actions_t = []
            global_state_t = np.concatenate(obs_list)
            
            for i in range(N_SATS):
                obs_t = torch.FloatTensor(obs_list[i]).unsqueeze(0)
                logits = agent.act(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                actions_t.append(action.item())
            
            next_obs, rewards, dones, _ = env.step(actions_t, debug_mode=True)
            
            batch_obs.append(obs_list)
            batch_gs.append(global_state_t)
            batch_acts.append(actions_t)
            batch_rews.append(rewards)
            
            obs_list = next_obs
            ep_rewards += rewards
            done = any(dones)
        
        # UPDATE
        optimizer.zero_grad()
        loss = 0
        returns = np.zeros_like(batch_rews)
        running_add = np.zeros(N_SATS)
        for t in reversed(range(len(batch_rews))):
            running_add = batch_rews[t] + 0.99 * running_add
            returns[t] = running_add
            
        # [THÊM] Chuẩn hóa Returns để việc học ổn định hơn
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        ent_loss = 0
        act_loss = 0
        val_loss = 0
        
        for t in range(len(batch_obs)):
            g_state = torch.FloatTensor(batch_gs[t]).unsqueeze(0)
            
            # Target Value lúc này là trung bình Return của cả đội (vì Critic là Global)
            # Hoặc tổng Return (tùy cách bạn định nghĩa Value)
            # Ở đây ta lấy trung bình Returns của các vệ tinh tại thời điểm t
            avg_return_t = np.mean(returns[t]) 
            target_val = torch.FloatTensor([avg_return_t]).unsqueeze(0)
            
            val_pred = agent.evaluate(g_state)
            val_loss += nn.MSELoss()(val_pred, target_val)
            
            for i in range(N_SATS):
                obs = torch.FloatTensor(batch_obs[t][i]).unsqueeze(0)
                act = torch.tensor([batch_acts[t][i]])
                
                logits = agent.act(obs)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(act)
                ent_loss += dist.entropy()
                
                # Advantage = Return thực tế (của riêng vệ tinh đó) - Value dự đoán (của cả đội)
                # Điều này giúp vệ tinh biết nó đang làm tốt hơn hay tệ hơn mặt bằng chung
                adv = returns[t][i] - val_pred.item()
                
                act_loss += -log_prob * adv
        
        # [ENTROPY REGULARIZATION] Thêm Entropy bonus để kích thích khám phá
        total_loss = act_loss + 0.5 * val_loss - 0.01 * ent_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
        
        print(f"Epoch {ep+1} | Reward: {np.mean(ep_rewards):.1f}")

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
    print("Done. Check 'mission_sim_optimize_reward.bin'")

if __name__ == "__main__":
    train_mission()