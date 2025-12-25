import numpy as np
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion

# --- IMPORT MỚI: HỆ THỐNG MESSAGING ---
from Basilisk.architecture import messaging

from Basilisk.simulation import spacecraft
from Basilisk.simulation import extForceTorque
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import vizSupport

def runContinuousManeuver():
    # 1. Setup Simulation
    scSim = SimulationBaseClass.SimBaseClass()
    simTaskName = "simTask"
    proc = scSim.CreateNewProcess("simProcess")
    timeStep = 1.0 
    proc.addTask(scSim.CreateNewTask(simTaskName, macros.sec2nano(timeStep)))

    # 2. Setup Earth
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True

    # 3. Setup Spacecraft
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "satellite"
    scObject.hub.mHub = 500.0 
    scObject.hub.IHubPntBc_B = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]
    
    scSim.AddModelToTask(simTaskName, scObject)
    scObject.gravField.gravBodies = spacecraft.GravBodyVector([earth])

    # 4. Setup External Force Effector
    extForce = extForceTorque.ExtForceTorque()
    extForce.ModelTag = "externalDisturbance"
    scObject.addDynamicEffector(extForce)
    scSim.AddModelToTask(simTaskName, extForce)

    # --- THIẾT LẬP MESSAGING ---
    cmdPayload = messaging.CmdForceInertialMsgPayload()
    cmdPayload.forceRequestInertial = [0.0, 0.0, 0.0]
    
    # Tạo message ban đầu
    cmdMsg = messaging.CmdForceInertialMsg().write(cmdPayload)
    
    # Kết nối message vào module ngoại lực
    extForce.cmdForceInertialInMsg.subscribeTo(cmdMsg)

    # 5. Init Orbit (20 độ)
    initialInclination = 20.0 * macros.D2R
    oe = orbitalMotion.ClassicElements()
    oe.a = 7000e3; oe.e = 0.0001; oe.i = initialInclination
    oe.Omega = 0.0; oe.omega = 0.0; oe.f = 0.0
    rN, vN = orbitalMotion.elem2rv(earth.mu, oe)

    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN

    # 6. Setup Vizard
    viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject, saveFile="continuous_orbit.bin")
    viz.settings.showOrbitLines = 1
    viz.settings.showHillFrame = 1

    # ===============================================
    # PHASE 1: CHẠY QUỸ ĐẠO CŨ (10 Phút)
    # ===============================================
    print("--- Phase 1: Bay 10 phút (Góc nghiêng 20 độ) ---")
    scSim.InitializeSimulation()
    
    phase1_time = 600.0 
    scSim.ConfigureStopTime(macros.sec2nano(phase1_time))
    scSim.ExecuteSimulation()

    # ===============================================
    # MANEUVER: TÍNH TOÁN VÀ UPDATE MESSAGE
    # ===============================================
    print("--- Maneuver: Kích hoạt động cơ bẻ góc! ---")
    
    # Lấy trạng thái hiện tại
    currState = scObject.scStateOutMsg.read()
    rVec = np.array(currState.r_BN_N)
    vVec = np.array(currState.v_BN_N)

    # Tính toán Vector bẻ góc
    hVec = np.cross(rVec, vVec)
    hHat = hVec / np.linalg.norm(hVec)
    rHat = rVec / np.linalg.norm(rVec)
    vHat = np.cross(hHat, rHat)

    targetInc = 90.0 * macros.D2R
    deltaInc = targetInc - initialInclination
    v0 = np.dot(vVec, vHat)
    
    vNew = (vVec - (1.0 - np.cos(deltaInc))*v0*vHat + np.sin(deltaInc)*v0*hHat)

    # Tính Lực cần thiết
    delta_v_vec = vNew - vVec
    mass = scObject.hub.mHub
    force_vec = (delta_v_vec * mass) / timeStep
    
    # --- [FIX LỖI] CẬP NHẬT MESSAGE ---
    # Ghi đè lực mới vào message
    cmdPayload.forceRequestInertial = force_vec.tolist()
    
    # Thay vì dùng scSim.TotalSimNanos, ta dùng chính biến thời gian ta đang có
    # Chuyển đổi giây sang nanosecond
    currentTimeNs = macros.sec2nano(phase1_time)
    
    cmdMsg.write(cmdPayload, currentTimeNs) 

    # Chạy đúng 1 bước (1 giây) để Lực tác dụng
    maneuver_time = phase1_time + timeStep
    scSim.ConfigureStopTime(macros.sec2nano(maneuver_time))
    scSim.ExecuteSimulation()

    # --- TẮT LỰC NGAY LẬP TỨC ---
    cmdPayload.forceRequestInertial = [0.0, 0.0, 0.0]
    
    # Cập nhật thời gian cho lần ghi này
    maneuverTimeNs = macros.sec2nano(maneuver_time)
    cmdMsg.write(cmdPayload, maneuverTimeNs)

    # ===============================================
    # PHASE 2: CHẠY QUỸ ĐẠO MỚI (10 Phút nữa)
    # ===============================================
    print("--- Phase 2: Bay tiếp 10 phút (Góc nghiêng 90 độ) ---")
    
    phase2_time = maneuver_time + 600.0
    scSim.ConfigureStopTime(macros.sec2nano(phase2_time))
    scSim.ExecuteSimulation()

    print("Done. Mở file 'continuous_orbit.bin' để xem.")

if __name__ == "__main__":
    runContinuousManeuver()