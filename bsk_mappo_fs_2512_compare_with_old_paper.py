import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import Basilisk
import pandas as pd # Thêm pandas để lưu log

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
    def __init__(self, n_sats=4, n_targets=50, viz=False, hurricane_mode=True):
        self.n_sats = n_sats
        self.n_targets = n_targets
        self.viz = viz
        self.hurricane_mode = hurricane_mode # Kích hoạt chế độ Bão
        
        self.earth_radius = 6371e3
        self.mu = 398600.4418e9
        
        # --- 1. ĐỒNG BỘ THAM SỐ VỚI BÀI BÁO (Agile EOS Model) ---
        # Bài báo thường dùng vệ tinh LEO 600-700km, xoay nhanh
        self.SAT_ALTITUDE = 600e3  # 600 km (Chuẩn LEO)
        self.SLEW_RATE_DEG = 10.0   # 10 độ/s (Agile Satellite standard)
        self.max_slew_rate = math.radians(self.SLEW_RATE_DEG)
        
        # Năng lượng & Bộ nhớ (Giả lập EOS-1/Pleiades)
        self.fuel_max = 100.0
        self.batt_max = 100.0
        self.buffer_max = 100       # Tăng buffer lên để phù hợp mission chụp bão
        
        # Cấu hình Camera & Downlink
        self.CAPTURE_MAX_DIST = 1500e3    # Cho phép chụp xa hơn chút
        self.CAPTURE_MAX_OFF_NADIR = 45.0 # Góc chuẩn chụp ảnh vệ tinh (45 độ)
        
        # Trạm mặt đất (Svalbard, Matera, McMurdo, v.v...) - Chuẩn ESA/NASA
        self.gs_coords = [
            (78.22, 15.40),   # Svalbard (Na Uy) - Trạm cực quan trọng nhất
            (40.65, 16.70),   # Matera (Ý)
            (21.02, 105.85),  # Hanoi (Vietnam)
            (-33.86, 151.20), # Sydney
            (40.71, -74.00)   # New York
        ]
        self.n_gs = len(self.gs_coords)
        self.global_captured_targets = set()
        
        # Action & Observation Space
        self.ORBIT_CHOICES = [10.0, 45.0, 97.0] # 97 là Sun-Synchronous Orbit (SSO)
        self.ACT_CHARGE = self.n_targets + self.n_gs
        self.ACT_ALT_UP = self.ACT_CHARGE + 1
        self.ACT_ALT_DOWN = self.ACT_CHARGE + 2
        self.ACT_GOTO_INC_START = self.ACT_ALT_DOWN + 1
        self.n_orbit_choices = len(self.ORBIT_CHOICES)
        self.action_dim = self.ACT_GOTO_INC_START + self.n_orbit_choices
        
        self.obs_dim = 14 
        self.global_state_dim = self.obs_dim * self.n_sats 
        
        # Cấu hình mặt trời mặc định
        self.SUN_DIRECTION = np.array([1.0, 0.0, 0.0])

        self._build_locations()
        self._init_simulator()

    def _build_locations(self):
            rng = np.random.RandomState(42)
            self.targets_ecef = []
            
            if self.hurricane_mode:
                print("[INFO] Kích hoạt kịch bản: HURRICANE (Tập trung mục tiêu)")
                # Tạo vùng bão tập trung (Ví dụ: Vùng biển Caribe/Đại Tây Dương)
                # Lat: 15 đến 35 độ Bắc, Lon: -90 đến -60 độ Tây
                center_lat, center_lon = 25.0, -75.0
                
                for _ in range(self.n_targets):
                    # Phân phối Gaussian xung quanh tâm bão để tạo cluster
                    lat = np.clip(rng.normal(center_lat, 5.0), -90, 90)
                    lon = np.clip(rng.normal(center_lon, 10.0), -180, 180)
                    self.targets_ecef.append(self._lld_to_ecef(lat*macros.D2R, lon*macros.D2R, 0))
            else:
                print("[INFO] Kích hoạt kịch bản: GLOBAL RANDOM")
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
        self.decision_dt = 10.0  # Thời gian giữa các quyết định hành động  
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
        
        self.sats, self.states, self.force_msgs, self.force_payloads, self.all_viz_objects = [], [], [], [], []
        
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
# [CẬP NHẬT] Orbit Parameter chuẩn bài báo (LEO ~600km)
            oe = orbitalMotion.ClassicElements()
            oe.a = self.earth_radius + self.SAT_ALTITUDE # 600km Altitude
            oe.e = 0.001
            oe.i = 45.0 * macros.D2R # Inclination quét qua vùng bão tốt
            oe.Omega = (i * 360.0 / self.n_sats) * macros.D2R
            oe.omega = 0.0; oe.f = 0.0
            
            rN, vN = orbitalMotion.elem2rv(self.planet_mu, oe)
            sc.hub.r_CN_NInit = np.array(rN); sc.hub.v_CN_NInit = np.array(vN)
            
            self.sats.append(sc); self.all_viz_objects.append(sc)
            bore_vec = -np.array(rN) / np.linalg.norm(rN)
            self.states.append({'fuel': self.fuel_max, 'batt': self.batt_max, 'buffer': 0, 'bore_vec': bore_vec})

        # --- B. DUMMY OBJECTS ---
# --- B. DUMMY OBJECTS ---
        for k, ecef in enumerate(self.targets_ecef):
            dummy = spacecraft.Spacecraft()
            dummy.ModelTag = f"TGT_{k}"
            r_mag = np.linalg.norm(ecef)
            dummy.hub.r_CN_NInit = ecef * ((r_mag + 50000.0) / r_mag)
            self.scSim.AddModelToTask(self.taskName, dummy)
            self.all_viz_objects.append(dummy)

        # [FIX LỖI INDEX ERROR]
        # Cập nhật tên cho đúng với 5 trạm trong __init__ (Svalbard, Matera, Hanoi, Sydney, NY)
        gs_names = ["Svalbard", "Matera", "Hanoi", "Sydney", "NY"]
        
        for k, ecef in enumerate(self.gs_ecef):
            dummy = spacecraft.Spacecraft()
            
            # Cơ chế an toàn: Nếu k vượt quá danh sách tên, dùng số thứ tự để không bị crash
            if k < len(gs_names):
                tag_name = gs_names[k]
            else:
                tag_name = f"{k}"
            
            dummy.ModelTag = f"GS_{tag_name}"
            
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
            
# ==================================================================
            # [NEW] PHẦN 1: SHAPING REWARD (THƯỞNG DẪN ĐƯỜNG)
            # ==================================================================
            # Tự động thưởng nếu vệ tinh đang hướng camera về mục tiêu gần nhất
            
            # 1.1 Tìm mục tiêu gần nhất chưa chụp
            best_target_idx = -1
            min_dist_to_tgt = 1e9
            for t_idx, t_ecef in enumerate(self.targets_ecef):
                if t_idx in self.global_captured_targets: continue
                d = np.linalg.norm(t_ecef - r_curr)
                if d < min_dist_to_tgt:
                    min_dist_to_tgt = d
                    best_target_idx = t_idx
            
            # 1.2 Nếu có mục tiêu trong tầm "Cảm nhận" (3000km)
            if best_target_idx != -1 and min_dist_to_tgt < 3000e3:
                tgt_vec = self.targets_ecef[best_target_idx] - r_curr
                tgt_vec /= np.linalg.norm(tgt_vec)
                
                # Tính độ khớp hướng (Alignment)
                alignment = np.dot(state['bore_vec'], tgt_vec) # 1.0 = Trùng khít
                
                # Nếu hướng khá đúng (> 60 độ), thưởng nhẹ liên tục
                if alignment > 0.5: 
                    rewards[i] += 0.05 * alignment # Cộng dồn mỗi bước

            # 1.3 Tương tự với Trạm mặt đất (Nếu buffer có ảnh)
            if state['buffer'] > 0:
                # Tìm trạm gần nhất
                min_gs_dist = 1e9
                best_gs_vec = None
                for gs_pos in self.gs_ecef:
                    vec = gs_pos - r_curr
                    d = np.linalg.norm(vec)
                    if d < min_gs_dist:
                        min_gs_dist = d
                        best_gs_vec = vec / d
                
                if min_gs_dist < 4000e3 and best_gs_vec is not None:
                    align_gs = np.dot(state['bore_vec'], best_gs_vec)
                    if align_gs > 0.5:
                        rewards[i] += 0.05 * align_gs

            # ==================================================================
            # PHẦN 2: XỬ LÝ HÀNH ĐỘNG (ACTION)
            # ==================================================================

            # 1. CHARGE
            if act == self.ACT_CHARGE:
                sat_pos_dir = r_curr / np.linalg.norm(r_curr)
                if np.dot(sat_pos_dir, self.SUN_DIRECTION) > -0.1:
                    state['batt'] = min(self.batt_max, state['batt'] + 5.0)
                    if state['batt'] < 30.0: rewards[i] += 1.0
                else:
                    rewards[i] -= 0.1 # Phạt nhẹ sạc lúc tối

            # 2. ORBIT CHANGE (NÂNG CẤP: SMART SWITCHING)
            elif act in [self.ACT_ALT_UP, self.ACT_ALT_DOWN] or \
                 (self.ACT_GOTO_INC_START <= act < self.ACT_GOTO_INC_START + self.n_orbit_choices):
                
                # --- A. LOGIC CŨ (BUFFER) ---
                if state['buffer'] >= self.buffer_max: rewards[i] += 1.0
                else: rewards[i] -= 0.05 

                # --- B. LOGIC THÔNG MINH: TÍNH TOÁN QUỸ ĐẠO TỐT NHẤT ---
                # Tìm xem quỹ đạo nào (Inclination) phù hợp nhất với các mục tiêu còn lại
                # Heuristic: Inclination tối ưu thường xấp xỉ vĩ độ (Latitude) lớn nhất của đám mục tiêu
                
                remaining_lats = []
                for t_idx, t_pos in enumerate(self.targets_ecef):
                    if t_idx not in self.global_captured_targets:
                        # Chuyển ECEF sang vĩ độ (đơn giản hóa)
                        z = t_pos[2]
                        r = np.linalg.norm(t_pos)
                        lat = math.degrees(math.asin(z / r))
                        remaining_lats.append(abs(lat))
                
                # Nếu còn mục tiêu, tính vĩ độ trung bình hoặc max của tụi nó
                if len(remaining_lats) > 0:
                    avg_target_lat = np.mean(remaining_lats) # Ví dụ: Bão ở vĩ độ 20
                else:
                    avg_target_lat = 45.0 # Mặc định
                
                # --- C. XỬ LÝ ALTITUDE (ĐỘ CAO) ---
                if act in [self.ACT_ALT_UP, self.ACT_ALT_DOWN]:
                    if state['fuel'] >= 1.0: 
                        dv_vec = v_dir if act == self.ACT_ALT_UP else -v_dir
                        force_vec = (sat.hub.mHub * 10.0) / self.decision_dt * dv_vec
                        self.force_payloads[i].forceRequestInertial = force_vec.tolist()
                        self.force_msgs[i].write(self.force_payloads[i], current_nano)
                        state['fuel'] -= 0.5 
                        
                        # [SMART] Nếu đang ở quá cao (>800km) hoặc quá thấp (<400km) mà chỉnh về chuẩn -> Thưởng
                        r_mag = np.linalg.norm(r_curr)
                        alt = r_mag - self.earth_radius
                        if (alt > 700e3 and act == self.ACT_ALT_DOWN) or (alt < 500e3 and act == self.ACT_ALT_UP):
                            rewards[i] += 2.0 # Thưởng vì biết giữ độ cao chuẩn
                
                # --- D. XỬ LÝ INCLINATION (GÓC NGHIÊNG QUỸ ĐẠO - SMART LOGIC) ---
                elif self.ACT_GOTO_INC_START <= act:
                    if state['fuel'] >= 5.0:
                        choice_idx = act - self.ACT_GOTO_INC_START
                        target_inc_deg = self.ORBIT_CHOICES[choice_idx]
                        target_inc_rad = target_inc_deg * macros.D2R
                        
                        oe = orbitalMotion.rv2elem(self.planet_mu, r_curr, v_curr)
                        current_inc_deg = math.degrees(oe.i)
                        inc_diff = target_inc_rad - oe.i

                        # ======================================================
                        # BƯỚC 1: XÁC ĐỊNH NHU CẦU (MOTIVATION)
                        # ======================================================
                        
                        # Tính vĩ độ trung bình của các Trạm mặt đất (GS)
                        # (GS thường cố định, nhưng ta cứ tính lại cho chắc)
                        gs_lats = [abs(coord[0]) for coord in self.gs_coords]
                        avg_gs_lat = np.mean(gs_lats) if gs_lats else 45.0
                        
                        # Tính vĩ độ trung bình của các Mục tiêu còn lại (Targets)
                        remaining_lats = []
                        for t_idx, t_pos in enumerate(self.targets_ecef):
                            if t_idx not in self.global_captured_targets:
                                z = t_pos[2]; r = np.linalg.norm(t_pos)
                                remaining_lats.append(abs(math.degrees(math.asin(z / r))))
                        avg_target_lat = np.mean(remaining_lats) if remaining_lats else 45.0

                        # QUYẾT ĐỊNH: Vệ tinh đang cần gì nhất?
                        is_buffer_critical = (state['buffer'] / self.buffer_max) > 0.8
                        
                        if is_buffer_critical:
                            # --- ƯU TIÊN DOWNLINK (Tìm GS) ---
                            optimal_lat = avg_gs_lat
                            mode_name = "DOWNLINK_SEEKING"
                            # Mẹo: Với GS, quỹ đạo càng cao (gần 90 độ) càng dễ quét trúng nhiều trạm
                            # Nên nếu buffer đầy, ta ưu tiên quỹ đạo có góc nghiêng CAO NHẤT trong list
                            best_inc_option = max(self.ORBIT_CHOICES) 
                        else:
                            # --- ƯU TIÊN CAPTURE (Tìm Bão) ---
                            optimal_lat = avg_target_lat
                            mode_name = "TARGET_HUNTING"
                            # Tìm trong các lựa chọn, cái nào gần vĩ độ bão nhất
                            best_inc_option = min(self.ORBIT_CHOICES, key=lambda x: abs(x - optimal_lat))

                        # ======================================================
                        # BƯỚC 2: SO SÁNH VÀ TÍNH THƯỞNG
                        # ======================================================
                        
                        # Vệ tinh chọn quỹ đạo nào? (target_inc_deg)
                        # Quỹ đạo tốt nhất là gì? (best_inc_option)
                        
                        is_correct_decision = (target_inc_deg == best_inc_option)
                        
                        # Logic thực hiện đốt động cơ (Burn)
                        if abs(inc_diff) < 2.0 * macros.D2R: 
                            # Đã ở đúng quỹ đạo mình chọn
                            if is_correct_decision:
                                rewards[i] += 1.0 # Tốt, hãy ở yên đây
                            else:
                                rewards[i] -= 0.5 # Đang ở sai chỗ, nên đổi đi
                        else:
                            # Thực hiện đổi quỹ đạo
                            h_vec = np.cross(r_curr, v_curr); h_dir = h_vec/np.linalg.norm(h_vec)
                            burn_dir = h_dir if inc_diff > 0 else -h_dir
                            req_dv = 2 * v_mag * math.sin(abs(inc_diff) / 2)
                            apply_dv = min(req_dv, 50.0)
                            force_vec = (sat.hub.mHub * apply_dv) / self.decision_dt * burn_dir
                            self.force_payloads[i].forceRequestInertial = force_vec.tolist()
                            self.force_msgs[i].write(self.force_payloads[i], current_nano)
                            state['fuel'] -= 2.0
                            
                            # [SMART REWARD]
                            if is_correct_decision:
                                # Thưởng ĐẬM nếu quyết định chuyển đúng hướng
                                rewards[i] += 15.0 
                                if debug_mode: 
                                    print(f"[SMART] Sat {i} ({mode_name}) switching to {target_inc_deg}° (Optimal: {best_inc_option}°)")
                            else:
                                # Phạt nếu Buffer đang đầy mà lại chui vào quỹ đạo không có trạm
                                rewards[i] -= 5.0
                                if debug_mode: 
                                    print(f"[BAD] Sat {i} Wrong Switch! Mode: {mode_name}, Picked: {target_inc_deg}, Needed: {best_inc_option}")

            # 3. CAPTURE (CHỤP ẢNH - ĐÃ NÂNG CẤP LOGIC)
            elif 0 <= act < self.n_targets:
                if state['buffer'] >= self.buffer_max:
                    rewards[i] -= 1.0 # Phạt đầy bộ nhớ
                elif act in self.global_captured_targets:
                    rewards[i] -= 0.1 # Phạt nhẹ trùng lặp
                else:
                    tgt_pos = self.targets_ecef[act]
                    req_vec = tgt_pos - r_curr
                    dist = np.linalg.norm(req_vec); req_vec /= (dist + 1e-9)
                    
                    # Check Ánh sáng
                    sun_dir = self.SUN_DIRECTION
                    tgt_norm = tgt_pos / np.linalg.norm(tgt_pos)
                    is_daylight = np.dot(tgt_norm, sun_dir) > -0.1
                    
                    if not is_daylight:
                        rewards[i] -= 0.1 # Phạt chụp đêm
                    else:
                        # Tính góc xoay
                        cos = np.dot(state['bore_vec'], req_vec)
                        angle = math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))
                        max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                        
                        # A. ĐÃ HƯỚNG VÀO MỤC TIÊU -> THỬ CHỤP
                        if angle <= max_turn:
                            state['bore_vec'] = req_vec
                            state['batt'] -= 1.0 # Chụp tốn pin
                            
                            # Tính Off-Nadir
                            nadir_vec = -r_curr / np.linalg.norm(r_curr)
                            off_nadir = math.degrees(math.acos(np.clip(np.dot(req_vec, nadir_vec), -1, 1)))
                            
                            # [LOGIC MỚI] BỎ PHẠT NẾU TRƯỢT, CHỈ THƯỞNG NẾU TRÚNG
                            # Điều kiện: < 1500km và < 45 độ
                            if dist < 1500e3 and off_nadir < 45.0:
                                self.global_captured_targets.add(act)
                                state['buffer'] += 1
                                rewards[i] += 15.0 # Thưởng đậm (+15)
                                if debug_mode: print(f"Sat {i} CAPTURE T{act} SUCCESS")
                            else:
                                # KHÔNG trừ điểm reward ở đây nữa (chỉ mất pin)
                                pass 
                                
                        # B. ĐANG XOAY (Slewing)
                        else:
                            state['batt'] -= 0.2 # Xoay tốn ít pin
                            ratio = max_turn / angle
                            state['bore_vec'] = (1-ratio)*state['bore_vec'] + ratio*req_vec
                            state['bore_vec'] /= np.linalg.norm(state['bore_vec'])
                            rewards[i] += 0.05 # Thưởng công xoay

            # 4. DOWNLINK (TRUYỀN TIN - ĐÃ NÂNG CẤP)
            elif self.n_targets <= act < self.n_targets + self.n_gs:
                gs_idx = act - self.n_targets
                gs_pos = self.gs_ecef[gs_idx]
                req_vec = gs_pos - r_curr
                dist = np.linalg.norm(req_vec); req_vec /= (dist + 1e-9)
                
                cos = np.dot(state['bore_vec'], req_vec)
                angle = math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))
                max_turn = math.degrees(self.max_slew_rate) * self.decision_dt
                
                if state['buffer'] >= self.buffer_max: rewards[i] += 0.5
                
                # A. ĐÃ HƯỚNG VỀ TRẠM
                if angle <= max_turn:
                    state['bore_vec'] = req_vec
                    if state['buffer'] > 0:
                        # Điều kiện Downlink: < 2200km
                        if dist < 2200e3:
                            state['buffer'] -= 1
                            state['batt'] -= 1.0 
                            rewards[i] += 99.0 # JACKPOT
                            if debug_mode: print(f"Sat {i} DOWNLINK SUCCESS")
                        # Tiếp cận: < 3000km
                        elif dist < 3000e3:
                            score = (3000e3 - dist)/(3000e3 - 2200e3)
                            rewards[i] += score 
                        else:
                            # Xa quá thì thôi, không phạt
                            pass
                    else:
                         rewards[i] -= 0.2 # Buffer rỗng mà đòi gửi
                
                # B. ĐANG XOAY
                else:
                    state['batt'] -= 0.2
                    ratio = max_turn / angle
                    state['bore_vec'] = (1-ratio)*state['bore_vec'] + ratio*req_vec
                    state['bore_vec'] /= np.linalg.norm(state['bore_vec'])
                    rewards[i] += 0.05

            # Idle Drain
            else:
                state['batt'] -= 0.05

        stop_time = self.sim_time + self.decision_dt
        self.scSim.ConfigureStopTime(int(macros.sec2nano(stop_time)))
        self.scSim.ExecuteSimulation()
        self.sim_time = stop_time
        
        if self.sim_time > 7200: dones = [True] * self.n_sats
        return self._get_all_obs(), rewards, dones, {}

# --- 4. TRAINER ---
# --- CÁC HÀM PHỤ TRỢ LƯU/TẢI MODEL ---
def save_checkpoint(path, agent, optimizer, epoch, history):
    torch.save({
        'epoch': epoch,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, path)
    print(f"[CHECKPOINT] Đã lưu model tại epoch {epoch} vào '{path}'")

def load_checkpoint(path, agent, optimizer):
    if os.path.exists(path):
        # [SỬA LỖI TẠI ĐÂY] Thêm weights_only=False để cho phép load dữ liệu numpy/list
        try:
            checkpoint = torch.load(path, weights_only=False)
        except TypeError:
            # Fallback cho các phiên bản PyTorch cũ hơn (chưa có tham số weights_only)
            checkpoint = torch.load(path)
            
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', {
            'epoch': [], 'total_reward': [], 'targets_captured': [], 
            'data_downlinked': [], 'avg_fuel_remaining': []
        })
        print(f"[CHECKPOINT] Đã tìm thấy file lưu! Tiếp tục train từ Epoch {start_epoch}")
        return start_epoch, history
    else:
        print("[CHECKPOINT] Không tìm thấy file lưu. Bắt đầu train mới.")
        return 0, {
            'epoch': [], 'total_reward': [], 'targets_captured': [], 
            'data_downlinked': [], 'avg_fuel_remaining': []
        }

# --- 4. TRAINER (CÓ TÍNH NĂNG SAVE/LOAD) ---
def train_mission():
    N_SATS = 4
    N_TARGETS = 100 
    MAX_EPOCHS = 1000
    CHECKPOINT_FILE = "mappo_hurricane_checkpoint.pth" # Tên file lưu
    
    print("--- 1. TRAINING (HURRICANE SCENARIO + CHECKPOINT) ---")
    # Kích hoạt Hurricane Mode
    env = BasiliskFullMissionEnv(n_sats=N_SATS, n_targets=N_TARGETS, viz=False, hurricane_mode=True)
    agent = MultiAgentActorCritic(env.obs_dim, env.global_state_dim, env.action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    
    # --- [NEW] LOAD CHECKPOINT NẾU CÓ ---
    start_epoch, history = load_checkpoint(CHECKPOINT_FILE, agent, optimizer)
    
    for ep in range(start_epoch, MAX_EPOCHS): 
        obs_list = env.reset()
        ep_rewards = np.zeros(N_SATS)
        done = False
        
        # Metrics
        targets_captured_count = 0
        data_downlinked_count = 0
        
        batch_obs, batch_gs, batch_acts, batch_rews = [], [], [], []
        
        while not done:
            actions_t = []
            global_state_t = np.concatenate(obs_list)
            
            for i in range(N_SATS):
                obs_t = torch.FloatTensor(obs_list[i]).unsqueeze(0)
                logits = agent.act(obs_t)
                action = torch.distributions.Categorical(logits=logits).sample().item()
                actions_t.append(action)
            
            # Step
            next_obs, rewards, dones, info = env.step(actions_t, debug_mode=False)
            
            # Metrics đếm tạm qua reward (để log)
            for r in rewards:
                if r >= 15.0: targets_captured_count += 1 
                if r >= 99.0: data_downlinked_count += 1
            
            batch_obs.append(obs_list)
            batch_gs.append(global_state_t)
            batch_acts.append(actions_t)
            batch_rews.append(rewards)
            
            obs_list = next_obs
            ep_rewards += rewards
            done = any(dones)
        
        # --- PPO UPDATE ---
        optimizer.zero_grad()
        returns = np.zeros_like(batch_rews)
        running_add = np.zeros(N_SATS)
        for t in reversed(range(len(batch_rews))):
            running_add = batch_rews[t] + 0.999 * running_add
            returns[t] = running_add
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        ent_loss, act_loss, val_loss = 0, 0, 0
        for t in range(len(batch_obs)):
            g_state = torch.FloatTensor(batch_gs[t]).unsqueeze(0)
            avg_return = np.mean(returns[t])
            target_val = torch.FloatTensor([avg_return]).unsqueeze(0)
            val_pred = agent.evaluate(g_state)
            val_loss += nn.MSELoss()(val_pred, target_val)
            
            for i in range(N_SATS):
                obs = torch.FloatTensor(batch_obs[t][i]).unsqueeze(0)
                act = torch.tensor([batch_acts[t][i]])
                logits = agent.act(obs)
                dist = torch.distributions.Categorical(logits=logits)
                ent_loss += dist.entropy()
                act_loss += -dist.log_prob(act) * (returns[t][i] - val_pred.item())
        
        total_loss = act_loss + 0.5 * val_loss - 0.01 * ent_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
        
        # --- LOGGING ---
        avg_fuel = np.mean([s['fuel'] for s in env.states])
        total_rw = np.sum(ep_rewards)
        real_capture_count = len(env.global_captured_targets)
        
        # 1. Cập nhật vào biến History (để vẽ biểu đồ và lưu checkpoint)
        history['epoch'].append(ep+1)
        history['total_reward'].append(total_rw)
        history['targets_captured'].append(real_capture_count)
        history['data_downlinked'].append(data_downlinked_count)
        history['avg_fuel_remaining'].append(avg_fuel)
        
        print(f"Epoch {ep+1} | Reward: {total_rw:.1f} | Captured: {real_capture_count} | Downlink: {data_downlinked_count} | Fuel: {avg_fuel:.1f}%")
        
        # 2. LƯU CHECKPOINT (Quan trọng để lần sau load lại)
        save_checkpoint(CHECKPOINT_FILE, agent, optimizer, ep, history)

        # 3. [MỚI] LƯU CSV THEO CHẾ ĐỘ APPEND (GHI NỐI TIẾP)
        # Tạo một DataFrame chỉ chứa thông tin của Epoch hiện tại
        current_epoch_data = {
            'epoch': [ep+1],
            'total_reward': [total_rw],
            'targets_captured': [real_capture_count],
            'data_downlinked': [data_downlinked_count],
            'avg_fuel_remaining': [avg_fuel]
        }
        df_epoch = pd.DataFrame(current_epoch_data)
        
        # Kiểm tra xem file đã tồn tại chưa
        csv_file = "comparison_results.csv"
        file_exists = os.path.isfile(csv_file)
        
        try:
            # mode='a' là append (ghi nối đuôi)
            # header=not file_exists: Chỉ ghi tên cột nếu file chưa tồn tại
            df_epoch.to_csv(csv_file, mode='a', header=not file_exists, index=False)
        except PermissionError:
            print(f"[WARNING] Không thể lưu CSV epoch {ep+1} vì file đang mở! Đừng lo, Checkpoint vẫn đã được lưu.")

    # --- CUỐI HÀM TRAIN ---
    # Không cần lưu CSV tổng ở đây nữa vì đã lưu từng bước rồi
    print("\n[INFO] Hoàn thành huấn luyện.")
    
    # Vẽ biểu đồ từ lịch sử đầy đủ
    # (Lưu ý: history vẫn chứa đủ dữ liệu từ lúc load checkpoint)
    df_full = pd.DataFrame(history)
    plot_comparison(df_full)

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

import matplotlib.pyplot as plt

def plot_comparison(df):
    """
    Vẽ biểu đồ so sánh kết quả huấn luyện với Benchmark của bài báo.
    """
    plt.figure(figsize=(15, 5))
    
    # --- 1. Total Reward Comparison ---
    plt.subplot(1, 3, 1)
    plt.plot(df['epoch'], df['total_reward'], label='Our MAPPO Model', color='blue', linewidth=2)
    plt.title('Total Reward over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # --- 2. Targets Captured Comparison ---
    # Giả sử mô hình bài báo (Heuristic/MIP) đạt được khoảng 75% số mục tiêu trong kịch bản Bão
    PAPER_BASELINE_CAPTURE = 75.0 
    
    plt.subplot(1, 3, 2)
    plt.plot(df['epoch'], df['targets_captured'], label='Our MAPPO Model', marker='o', color='green')
    plt.axhline(y=PAPER_BASELINE_CAPTURE, color='red', linestyle='--', label='Paper Model (Benchmark)', linewidth=2)
    plt.title('Targets Captured (Hurricane Scenario)')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Targets')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 3. Data Downlinked vs Fuel ---
    plt.subplot(1, 3, 3)
    plt.plot(df['epoch'], df['data_downlinked'], label='Downlinked Packets', color='purple')
    
    # Tạo trục Y thứ 2 cho nhiên liệu
    ax2 = plt.gca().twinx()
    ax2.plot(df['epoch'], df['avg_fuel_remaining'], label='Fuel Remaining', color='orange', linestyle=':')
    ax2.set_ylabel('Fuel (%)')
    
    plt.title('Downlink Efficiency & Fuel')
    plt.xlabel('Epoch')
    plt.ylabel('Packets Downlinked')
    
    # Gộp legend
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hurricane_comparison_chart.png")
    print("[INFO] Đã lưu biểu đồ so sánh vào 'hurricane_comparison_chart.png'")
    plt.show()

if __name__ == "__main__":
    train_mission()