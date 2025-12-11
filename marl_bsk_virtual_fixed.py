import numpy as np
import gymnasium as gym

from Basilisk.architecture import bskLogging
from bsk_rl import act, obs, sats, scene, data
from bsk_rl.sim import dyn, fsw, world
from bsk_rl.utils.orbital import random_orbit

# Tắt log rác
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

# --- 1. ĐỊNH NGHĨA CLASS (SỬA LỖI MODEL TẠI ĐÂY) ---
class OpticalSat(sats.ImagingSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction")
        ),
        obs.Eclipse(),
        obs.Time(),
    ]
    action_spec = [
        act.Image(n_ahead_image=3), # Giảm xuống 3 để tính toán nhanh hơn
        act.Charge(duration=600.0),
    ]
    
    # --- SỬA LỖI: Dùng model chuẩn hỗ trợ xoay ngắm (Target Pointing) ---
    dyn_type = dyn.ImagingDynModel      # <--- Đã sửa (bỏ Continuous)
    fsw_type = fsw.ImagingFSWModel      # <--- Đã sửa (bỏ Continuous)

    @classmethod
    def default_sat_args(cls, **kwargs):
        extra_elev = kwargs.pop("imageTargetMinimumElevation", None)
        sat_args = super().default_sat_args(**kwargs)
        if extra_elev is not None:
            sat_args["imageTargetMinimumElevation"] = extra_elev
        return sat_args

class RadarSat(sats.ImagingSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction")
        ),
        obs.Eclipse(),
        obs.Time(),
    ]
    action_spec = [
        act.Image(n_ahead_image=3),
        act.Charge(duration=600.0),
    ]
    
    # --- SỬA LỖI: Dùng model chuẩn ---
    dyn_type = dyn.ImagingDynModel      # <--- Đã sửa
    fsw_type = fsw.ImagingFSWModel      # <--- Đã sửa

    @classmethod
    def default_sat_args(cls, **kwargs):
        extra_elev = kwargs.pop("imageTargetMinimumElevation", None)
        sat_args = super().default_sat_args(**kwargs)
        if extra_elev is not None:
            sat_args["imageTargetMinimumElevation"] = extra_elev
        return sat_args

# --- 2. TẠO VỆ TINH ---
def build_heterogeneous_satellites():
    sats_list = []

    # Optical 1
    opt1_args = OpticalSat.default_sat_args(
        oe=random_orbit(i=45, a=7000), 
        imageTargetMinimumElevation=np.radians(45),
        batteryStorageCapacity=80.0 * 3600 * 2,
        storedCharge_Init=lambda: np.random.uniform(0.4, 1.0) * 80.0 * 3600 * 2,
        nHat_B=np.array([0, 0, -1]),
    )
    
    # Optical 2
    opt2_args = OpticalSat.default_sat_args(
        oe=random_orbit(i=98, a=7000),
        imageTargetMinimumElevation=np.radians(45),
        batteryStorageCapacity=60.0 * 3600 * 2,
        storedCharge_Init=lambda: np.random.uniform(0.3, 1.0) * 60.0 * 3600 * 2,
        nHat_B=np.array([0, 0, -1]),
    )

    # Radar 1
    radar_args = RadarSat.default_sat_args(
        oe=random_orbit(i=20, a=7200),
        imageTargetMinimumElevation=np.radians(30),
        batteryStorageCapacity=100.0 * 3600 * 2,
        storedCharge_Init=lambda: np.random.uniform(0.5, 1.0) * 100.0 * 3600 * 2,
        nHat_B=np.array([0, 0, -1]),
    )

    sats_list.append(OpticalSat("Optical_1", opt1_args))
    sats_list.append(OpticalSat("Optical_2", opt2_args))
    sats_list.append(RadarSat("Radar_1", radar_args))

    return sats_list
'''
# --- 3. TẠO MÔI TRƯỜNG ---
def make_env(satellites):
    # Scenario: 50 mục tiêu ngẫu nhiên
    scenario = scene.UniformTargets(n_targets=50) 
    
    # Sử dụng BasicWorldModel để tránh lỗi GroundStation
    world_type = world.BasicWorldModel 
    world_args = {} 

    env = gym.make(
        "GeneralSatelliteTasking-v1",
        satellites=satellites,
        scenario=scenario,
        rewarder=data.UniqueImageReward(),
        world_type=world_type,
        world_args=world_args,
        # Giảm thời gian chạy xuống 90 phút (khoảng 1 vòng quỹ đạo) để test nhanh
        time_limit=95.0 * 60.0, 
        terminate_on_time_limit=True,
        log_level="INFO",
        disable_env_checker=True 
    )
    return env
'''
# --- 3. TẠO MÔI TRƯỜNG (SỬA ĐỂ BẬT VIZARD) ---
def make_env(satellites):
    # Scenario: 50 mục tiêu
    scenario = scene.UniformTargets(n_targets=50) 
    
    # Dùng BasicWorldModel (Không có trạm mặt đất)
    world_type = world.BasicWorldModel 
    world_args = {} 

    env = gym.make(
        "GeneralSatelliteTasking-v1",
        satellites=satellites,
        scenario=scenario,
        rewarder=data.UniqueImageReward(),
        world_type=world_type,
        world_args=world_args,
        time_limit=95.0 * 60.0,
        terminate_on_time_limit=True,
        log_level="INFO",
        disable_env_checker=True,
        
        # --- THÊM DÒNG NÀY ĐỂ BẬT VIZARD ---
        render_mode="human" 
    )
    return env
# --- 4. TEST ---
def test_env():
    print("1. Khởi tạo vệ tinh...")
    sats = build_heterogeneous_satellites()
    
    print("2. Tạo môi trường Basic (Imaging Mode)...")
    env = make_env(sats)
    
    try:
        print("3. Reset môi trường...")
        obs, info = env.reset(seed=42)
        print(">>> RESET THÀNH CÔNG!")
        
        done = False
        total_reward = 0.0
        step = 0
        
        print("4. Chạy vòng lặp mô phỏng...")
        # Chạy thử 50 bước
        while not done and step < 50:
            
            # Lấy mẫu hành động (1 Tuple gồm 3 hành động)
            actions = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(actions)
            
            if isinstance(reward, (list, tuple, np.ndarray)):
                r = sum(reward)
            else:
                r = reward
            total_reward += r
            
            done = terminated or truncated
            step += 1
            
            # In ra mỗi 10 bước
            if step % 10 == 0:
                print(f"Step {step}: Reward tích lũy = {total_reward:.4f}")
            
        print(f">>> TEST HOÀN TẤT. Tổng reward: {total_reward}")

    except Exception as e:
        print("\nLỖI:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_env()