import MPCC
import numpy as np
from numpy.linalg import inv
from math import pi
import time

import matplotlib.pyplot as plt
from srmt2.planning_scene import PlanningScene
import argparse
import scipy.io

## ROS library
import rclpy
from nav_msgs.msg import Path
from std_msgs.msg import Float32

from main_utils import create_path_message, create_pred_path_message

from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration


np.set_printoptions(suppress=True, precision=3)

## MPCC parameters
param_value = {'cost': {
                    "qC" : 1000.0,
                    "qCNmult": 10,
                    "qL" : 500.0,
                    "qVs" : 10.0,

                    "qOri": 100,
                    
                    "qSing": 1,

                    "rdq"  : 0.002,
                    "rddq"  : 50,
                    "rdVs" : 0.1,
                    },
            'param': {
                    "desired_s_velocity": 0.1,
                    "tol_sing": 0.018,
                    "tol_selcol": 1.0,
                    "tol_envcol": 1.0,
                    }
            }

# ## Obstacle information
# obs_positions = np.array([0.48,  0.218, 0.521]) # unit: [m]
# obs_limit = np.array([[0.48,  0.218, 0.421],   # lower limit
#                       [0.48,  0.218, 0.621]])  # upper limit
# obs_radius = 5         # unit: [cm]
# obs_speed = 0.05       # unit: [m/s]



# 모든 장애물의 반지름과 속도가 같다고 가정
obs_radius = 5         # unit: [cm]
obs_speed = 0.05       # unit: [m/s]

def print_stats(data_dict, name):
    """
    data_dict: {'key1': array1, 'key2': array2, …}
    name: 'debug_data' 또는 'time_data' 등의 식별 문자열
    """
    print(f"\n=== Statistics for {name} ===")
    for key, arr in data_dict.items():
        # scipy.io.loadmat 로드 시에는 내부에 meta 정보가 함께 들어오기 때문에,
        # 실제 배열만 뽑아 쓰려면 아래처럼 검사해주면 안전합니다.
        if not isinstance(arr, np.ndarray):
            continue

        flat = arr.flatten()
        print(f"{key:15s} | mean: {flat.mean():10.6f} | min: {flat.min():10.6f} | max: {flat.max():10.6f} | var: {flat.var():10.6f} | std: {flat.std():10.6f} |")
    print()

def main(args):
    ## Create Planning Scene
    pc = PlanningScene(arm_names=["fr3"], arm_dofs=[7], base_link="world")
    
    ## Obstacle information
    num_obstacles = 3  # <-- 생성할 장애물 개수

    # ================================================
    #                     Case 3
    # ================================================
    # 각 장애물의 초기 위치 (num_obstacles, 3) 형태로 정의
    obs_positions = np.array([
        [0.48,  0.218, 0.521],
        [0.40, -0.200, 0.450],
        [0.55, -0.200, 0.55]
    ])

    # 각 장애물의 이동 한계와 속도도 배열로 관리
    obs_limits = np.zeros((num_obstacles, 2, 3))
    # Obstacle 1 & 2: 상한/하한을 초기 위치와 같게 설정하여 고정
    obs_limits[0] = np.array([obs_positions[0], obs_positions[0]]) 
    obs_limits[1] = np.array([obs_positions[1], obs_positions[1]])
    # Obstacle 3: Y축(좌우)으로 -0.2에서 0.2까지 움직이도록 설정
    obs_limits[2] = np.array([[0.55, -0., 0.450], [0.55, 20, 0.450]])

    # Create publishers
    node = rclpy.create_node('mpcc_node')
    path_pub = node.create_publisher(Path, '/mpcc/path', 10)
    splined_path_pub = node.create_publisher(Path, '/mpcc/splined_path', 10)
    local_path_pub = node.create_publisher(Path, '/mpcc/local_path', 10)
    ref_local_path_pub = node.create_publisher(Path, '/mpcc/ref_local_path', 10)
    ee_speed_pub = node.create_publisher(Float32, '/mpcc/ee_speed', 1)
    mani_pub = node.create_publisher(Float32, '/mpcc/mani', 1)
    sel_min_dist_pub = node.create_publisher(Float32, '/mpcc/sel_min_dist', 1)
    env_min_dist_pub = node.create_publisher(Float32, '/mpcc/env_min_dist', 1)
    contour_error_pub = node.create_publisher(Float32, '/mpcc/contour_error', 1)

    ## Create mpc controller
    mpc = MPCC.MPCC()
    N = mpc.pred_horizon
    panda_num_links = mpc.num_links
    robot_dof = mpc.robot_dof

    ## Create robot data processor
    integrator = MPCC.Integrator()
    robot = MPCC.RobotModel()
    selcolNN = MPCC.SelfCollisionNN()
    selcolNN.setNeuralNetwork(input_size=robot_dof, output_size=1, hidden_layer_size=np.array([256, 64]), is_nerf=True)
    envcolNN = MPCC.EnvCollisionNN()
    envcolNN.setNeuralNetwork(input_size=robot_dof+3, output_size=panda_num_links, hidden_layer_size=np.array([256, 256, 256, 256]), is_nerf=True)

    # 1) 퍼블리셔 생성 (while 루프 밖)
    marker_pub = node.create_publisher(Marker, 'obstacle_marker', 10)

    ## Set initial state and control input
    state = np.array([0., 0., 0., -pi/2, 0., pi/2, pi/4, 0., 0.])
    input = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    mpc.setTrack(state)
    mpc.setParam(param_value)
    path_pos, path_ori = mpc.getPath()
    spline_pos, spline_ori, spline_arc_length = mpc.getSplinePath()
    spline_T = np.zeros([spline_pos.shape[0], 4, 4])
    for i, (posi, rot) in enumerate(zip(spline_pos, spline_ori)):
        spline_T[i, :3, :3] = rot
        spline_T[i, :3, 3] = posi

    path_msg = create_path_message(node, path_pos, path_ori)
    splined_path_msg = create_path_message(node, spline_pos, spline_ori)

    ## Debugging data  
    debug_data = {}
    debug_data["q"] = []                    # joint angle
    debug_data["qdot"] = []                 # joint velocity
    debug_data["qddot"] = []                # joint acceleration
    debug_data["s"] = []                    # path parameter
    debug_data["vs"] = []                   # velocity of path parameter
    debug_data["dVs"] = []                  # acceleration of path parameter
    debug_data["ee_vel"] = []               # End-Effector velocity
    debug_data["ee_speed"] = []             # End-Effector speed
    debug_data["sel_min_dist"] = []         # Minium distance for self collision
    debug_data["env_min_dist"] = []         # Minium distance for environment collision
    debug_data["mani"] = []                 # Manipulability
    debug_data["contour_error"] = []        # Contouring error
    debug_data["spline_ee_pose"] = spline_T # splined track path pose (track length, 4, 4)
    debug_data["pred_ee_pose"] = []         # predicted End-Effector path pose (N+1, 4, 4)
    debug_data["ref_ee_pose"] = []          # predicted reference path pose (N+1, 4, 4)
    ## Added
    debug_data["iter_count"] = []             # Iteration count

    time_data = {}
    time_data["total"] = []
    time_data["set_env"] = []
    time_data["set_qp"] = []
    time_data["solve_qp"] = []
    time_data["get_alpha"] = []

    SLECOL_BUFFER = param_value["param"]["tol_selcol"]
    MANI_BUFFER = param_value["param"]["tol_sing"]

    obs_step = obs_speed * mpc.Ts
    
    time_idx=0
    qdot_pre = np.zeros(robot_dof)
    
    obs_steps = np.zeros((num_obstacles, 3))
    obs_steps[0, 2] = obs_speed * mpc.Ts  # 첫 번째 장애물은 Z축으로 이동
    obs_steps[1, 2] = obs_speed * mpc.Ts  # 두 번째 장애물은 Z축으로 이동
    obs_steps[2, 1] = obs_speed * mpc.Ts  # 세 번째 장애물은 Y축으로 이동

    while rclpy.ok():
        start = time.time()
        
        # ==========================================================
        # ▼▼▼▼▼▼▼▼▼▼ 이 블록 전체를 아래 코드로 교체하세요 ▼▼▼▼▼▼▼▼▼▼
        # ==========================================================
        # while 루프 내부
        if args.is_obs:
            # --- 1. 장애물 이동 로직 (수정된 버전) ---
            # 세 번째 장애물이 오른쪽으로 움직이고 있고(obs_steps > 0), 오른쪽 경계선을 넘었을 때
            if obs_steps[2, 1] > 0 and obs_positions[2, 1] >= obs_limits[2, 1, 1]:
                obs_steps[2, 1] *= -1 # 방향을 왼쪽으로 전환
            # 세 번째 장애물이 왼쪽으로 움직이고 있고(obs_steps < 0), 왼쪽 경계선을 넘었을 때
            elif obs_steps[2, 1] < 0 and obs_positions[2, 1] <= obs_limits[2, 0, 1]:
                obs_steps[2, 1] *= -1 # 방향을 오른쪽으로 전환

            # 모든 장애물 위치 업데이트
            obs_positions += obs_steps

            # --- 2. 시각화 및 Planning Scene 업데이트 로직 ---
            for i in range(num_obstacles):
                # Planning Scene에 i번째 장애물 추가
                pc.add_sphere(f"obs_{i}", 
                            0.01*obs_radius, 
                            obs_positions[i] + np.array([0.3, 0, 0.256]),
                            np.array([1,0,0,0]))

                # RViz 시각화용 Marker 메시지 작성
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp    = node.get_clock().now().to_msg()
                m.ns              = 'obstacles'
                m.id              = i 
                m.type            = Marker.SPHERE
                m.action          = Marker.ADD

                m.pose.position.x = float(obs_positions[i, 0])
                m.pose.position.y = float(obs_positions[i, 1])
                m.pose.position.z = float(obs_positions[i, 2])
                
                m.pose.orientation.w = 1.0
                m.scale.x = obs_radius * 0.02 # cm -> m, diameter
                m.scale.y = obs_radius * 0.02
                m.scale.z = obs_radius * 0.02
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
                m.color.a = 0.8
                m.lifetime = Duration(sec=0, nanosec=0)

                # 각 마커를 루프 안에서 즉시 퍼블리시
                marker_pub.publish(m)
        # if time_idx == 500:
        #     path = np.array([[[1, 0, 0, 0],
        #                       [0, 1, 0, 0],
        #                       [0, 0, 1, 0],
        #                       [0, 0, 0, 1]],
        #                      [[0, 0, -1, 0.3],
        #                       [0, 1, 0, 0],
        #                       [1, 0, 0, 0.3],
        #                       [0, 0, 0, 1]]])
        #     mpc.setTrack(state, path)
        #     spline_pos, spline_ori, spline_arc_length = mpc.getSplinePath()
        #     spline_T = np.zeros([spline_pos.shape[0], 4, 4])
        #     for i, (posi, rot) in enumerate(zip(spline_pos, spline_ori)):
        #         spline_T[i, :3, :3] = rot
        #         spline_T[i, :3, 3] = posi
        #     splined_path_msg = create_path_message(node, spline_pos, spline_ori)
        # if time_idx == 900:
        #     path = np.array([[[1, 0, 0, 0],
        #                       [0, 1, 0, 0],
        #                       [0, 0, 1, 0],
        #                       [0, 0, 0, 1]],
        #                      [[0, 0, -1, 0],
        #                       [0, 1, 0, 0],
        #                       [1, 0, 0, 0],
        #                       [0, 0, 0, 1]]])
        #     mpc.setTrack(state, path)
        #     spline_pos, spline_ori, spline_arc_length = mpc.getSplinePath()
        #     spline_T = np.zeros([spline_pos.shape[0], 4, 4])
        #     for i, (posi, rot) in enumerate(zip(spline_pos, spline_ori)):
        #         spline_T[i, :3, :3] = rot
        #         spline_T[i, :3, 3] = posi
        #     splined_path_msg = create_path_message(node, spline_pos, spline_ori)
        # if time_idx == 1200:
        #     path = np.array([[[1, 0, 0, 0],
        #                       [0, 1, 0, 0],
        #                       [0, 0, 1, 0],
        #                       [0, 0, 0, 1]],
        #                      [[0, 0, 1, 0],
        #                       [0, 1, 0, 0],
        #                       [-1, 0, 0, 0],
        #                       [0, 0, 0, 1]]])
        #     mpc.setTrack(state, path)
        #     spline_pos, spline_ori, spline_arc_length = mpc.getSplinePath()
        #     spline_T = np.zeros([spline_pos.shape[0], 4, 4])
        #     for i, (posi, rot) in enumerate(zip(spline_pos, spline_ori)):
        #         spline_T[i, :3, :3] = rot
        #         spline_T[i, :3, 3] = posi
        #     splined_path_msg = create_path_message(node, spline_pos, spline_ori)

        ## run MPCC
        status, state, input, mpc_horizon, compute_time, iter_count = mpc.runMPC(state, input, obs_positions, obs_radius) if args.is_obs else mpc.runMPC(state, input)
        if status == False:
            print("MPC did not solve properly!!")
            break
        ## Virtual contact scenario
        # if time_idx > 500 and time_idx < 600:
        #     j = robot.getEEJacobian(state[:robot_dof])
        #     xdot = j @ input[:robot_dof]
        #     j_pinv = j.T.dot(inv(j.dot(j.T)))
        #     input[:robot_dof] = 0.5 * j_pinv @ xdot + (np.identity(robot_dof) - j_pinv.dot(j)) @ input[:robot_dof]
        state = integrator.simTimeStep(state, input)

        ## get robot information
        q = state[:robot_dof] # Puts state into q
        qdot = input[:robot_dof]
        qddot = (qdot - qdot_pre)/mpc.Ts
        x = robot.getEEPosition(q)
        xdot = robot.getEEJacobianv(q) @ qdot # matrix multiplication
        x_speed = np.linalg.norm(xdot)
        rotation = robot.getEEOrientation(q)
        s = state[-2]
        vs = state[-1]
        dVs = input[-1]
        sel_min_dist, _ = selcolNN.calculateMlpOutput(q)
        # env_min_dist, _ = envcolNN.calculateMlpOutputBatch(np.concatenate((q, obs_positions), axis=0))
        
        # 1. 장애물 개수(N) 가져오기
        num_obstacles = obs_positions.shape[0]

        # 2. q를 (7, 1) 형태로 바꾸고, N번 만큼 가로로 복제 -> (7, N) 행렬 생성
        q_batch = np.tile(q.reshape(-1, 1), (1, num_obstacles))

        # 3. obs_positions를 전치(Transpose)하여 (3, N) 형태로 만듦
        obs_batch = obs_positions.T

        # 4. 두 행렬을 위아래(수직)로 합쳐서 최종 입력 데이터 생성
        # (7, N) 행렬과 (3, N) 행렬을 합쳐 (10, N) 행렬이 됨
        batch_input = np.vstack([q_batch, obs_batch])

        # 5. 올바르게 만들어진 batch_input을 신경망에 전달
        env_min_dist, _ = envcolNN.calculateMlpOutputBatch(batch_input)
        
        mani = robot.getEEManipulability(q)
        contour_error = mpc.getContourError(s, x)
        pred_ee_T = np.zeros([mpc.pred_horizon + 1, 4, 4])
        ref_ee_T = np.zeros([mpc.pred_horizon + 1, 4, 4])
        for i in range(mpc.pred_horizon + 1):
            pred_ee_T[i, :3, :3] = robot.getEEOrientation(mpc_horizon[i]["state"][:robot_dof])
            pred_ee_T[i, :3, 3]  = robot.getEEPosition(mpc_horizon[i]["state"][:robot_dof])
            ref_ee_T[i, :3, 3], ref_ee_T[i, :3, :3] = mpc.getRefPose(mpc_horizon[i]["state"][-2])


        ## visualize
        pc.display(q)

        ## Print robot information
        # print("===============================================================")
        # print("time step       : ",time_idx)
        # print("q               : ", q)
        # print("qdot            : ", qdot)
        # print("qddot           : ", qddot)
        # print("x               : ", x)
        # print("xdot            : {:0.5f}".format(x_speed))
        # print("xdot            : ", xdot)
        # print("R               :\n", rotation)
        # print("s               : {:0.6f}".format(s))
        # print("vs              : {:0.6f}".format(vs))
        # print("dVs             : {:0.5f}".format(input[-1]))
        # print("mani            : {:0.5f}".format(mani))
        # print("sel min dist[cm]: {:0.5f}".format(sel_min_dist))
        # print("env min dist[cm]: ",env_min_dist)
        # print("MPC time        : {:0.5f}".format(compute_time["total"]))
        # print("===============================================================")

        ## Save data 
        debug_data["q"].append(q) 
        debug_data["qdot"].append(qdot)
        debug_data["qddot"].append(qdot)
        debug_data["s"].append(s)
        debug_data["vs"].append(vs)
        debug_data["dVs"].append(dVs)
        debug_data["ee_vel"].append(xdot)
        debug_data["ee_speed"].append(x_speed)
        debug_data["sel_min_dist"].append(sel_min_dist)
        debug_data["env_min_dist"].append(env_min_dist)
        debug_data["mani"].append(mani)
        debug_data["contour_error"].append(contour_error)
        debug_data["pred_ee_pose"].append(pred_ee_T)
        debug_data["ref_ee_pose"].append(ref_ee_T)
        debug_data["iter_count"].append(iter_count)

        time_data["total"].append(compute_time["total"])
        time_data["set_env"].append(compute_time["set_env"])
        time_data["set_qp"].append(compute_time["set_qp"])
        time_data["solve_qp"].append(compute_time["solve_qp"])
        time_data["get_alpha"].append(compute_time["get_alpha"])

        ## Publish data
        local_path_msg = create_pred_path_message(node, pred_ee_T[:, :3, 3], pred_ee_T[:, :3, :3])
        ref_local_path_msg = create_pred_path_message(node, ref_ee_T[:, :3, 3], ref_ee_T[:, :3, :3])
        ee_speed_msg = Float32()
        mani_msg = Float32()
        sel_min_dist_msg = Float32()
        env_min_dist_msg = Float32()
        contour_error_msg = Float32()
        ee_speed_msg.data = x_speed
        mani_msg.data = mani
        sel_min_dist_msg.data = sel_min_dist
        env_min_dist_msg.data = np.min(env_min_dist)
        contour_error_msg.data = contour_error*100 # [m] -> [cm]
        
        path_pub.publish(path_msg)
        splined_path_pub.publish(splined_path_msg)
        local_path_pub.publish(local_path_msg)
        ref_local_path_pub.publish(ref_local_path_msg)
        ee_speed_pub.publish(ee_speed_msg)
        mani_pub.publish(mani_msg)
        sel_min_dist_pub.publish(sel_min_dist_msg)
        env_min_dist_pub.publish(env_min_dist_msg)
        contour_error_pub.publish(contour_error_msg)
        
        
        ## End condition 
        if np.linalg.norm((spline_pos[-1] - x), 2) < 1E-2 and np.linalg.norm(MPCC.Log(spline_ori[-1].T @ rotation), 2) < 1E-2 and abs(state[-2] - 1.) < 1E-2:
            print("End point reached!!!")
            break
        end = time.time()
        elapsed = end - start
        if elapsed < mpc.Ts:
            time.sleep(mpc.Ts - elapsed)# rate.sleep()
        qdot_pre = qdot
        time_idx += 1

        rclpy.spin_once(node, timeout_sec=0.0)
        time_idx += 1
        ## End of while loop

    node.destroy_node()
    rclpy.shutdown()
    
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 블록을 추가하세요 ▼▼▼▼▼▼▼▼▼▼▼▼▼
    # C++ 객체에서 저장된 추론 시간 목록을 가져옵니다.
    inference_times = envcolNN.NNmodel.getInferenceTimes()
    
    # 추론 시간 분포를 히스토그램으로 시각화합니다.
    plt.figure(figsize=(14, 8))
    plt.hist(inference_times, bins=50, alpha=0.75, color='coral', edgecolor='black')
    plt.title("Distribution of Batch Inference Times")
    plt.xlabel("Inference Time (ms)")
    plt.ylabel("Frequency (count)")
    
    mean_time = np.mean(inference_times)
    plt.axvline(mean_time, color='r', linestyle='dashed', linewidth=2)
    min_ylim, max_ylim = plt.ylim()
    plt.text(mean_time*1.1, max_ylim*0.9, f'Mean: {mean_time:.3f} ms')
    
    plt.grid(True)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    ## Convert lists to NumPy arrays
    for key in debug_data:
        debug_data[key] = np.array(debug_data[key])
    for key in time_data:
        time_data[key] = np.array(time_data[key])

    ## Save the data to a .mat file
    scipy.io.savemat("debug_data.mat", debug_data)
    print("Data written to debug.mat")

    scipy.io.savemat("time_data.mat", time_data)
    print("Data written to debug.mat")


    print("mean nmpc time[sec]: {:0.6f} ".format(np.mean(time_data["total"])))
    print("max nmpc time[sec]: {:0.6f} ".format(np.max(time_data["total"])))

    ## Plotting the computation times
    plt.figure(figsize=(14, 8))

    plt.plot(time_data["total"], label="Total Time", color='b')
    plt.plot(time_data["set_env"], label="Set Env Time", color='m')
    plt.plot(time_data["set_qp"], label="Set QP Time", color='g')
    plt.plot(time_data["solve_qp"], label="Solve QP Time", color='r')
    plt.plot(time_data["get_alpha"], label="Get Alpha Time", color='c')
    plt.axhline(y=mpc.Ts, color='black', linestyle='--', label="Ts")

    plt.xlabel("Time Step")
    plt.ylabel("Time (s)")
    plt.title("Computation Times per Time Step")
    plt.ylim(-0.01, 0.2)  
    plt.xlim(0, len(time_data["total"]))
    plt.legend()
    plt.grid(True)

    ## Plotting the s/vs, ee_speed
    fig=plt.figure(figsize=(14, 8))
    fig.subplots_adjust(hspace=1)
    plt.subplot(411)
    # plt.plot("vs", data=debug_data, label="vs", color='b')
    plt.plot("ee_speed", data=debug_data, label="ee_speed", color='r')
    # plt.axhline(y=param_value["param"]["desired_ee_velocity"], color='black', linestyle='--', label="desired")
    plt.xlabel("s (m)")
    plt.ylabel("Speed (m/s)")
    plt.title("EE Speed per Arc length")
    plt.ylim(-0.01, max(max(debug_data["ee_speed"]), max(debug_data["vs"]))*1.2)  
    # plt.xlim(0, debug_data["s"][-1])
    plt.legend()
    plt.grid(True)

    ## Plotting the s/min_dist
    plt.subplot(412)
    plt.plot("sel_min_dist", data=debug_data, label="minimum distance", color='b')
    plt.axhline(y=SLECOL_BUFFER, color='black', linestyle='--', label="buffer")
    plt.xlabel("s (m)")
    plt.ylabel("distance (cm)")
    plt.title("Minimum distance per Arc length")
    plt.ylim(-0.01, max(debug_data["sel_min_dist"])*1.2)  
    # plt.xlim(0, debug_data["s"][-1])
    plt.legend()
    plt.grid(True)

    ## Plotting the s/manipulability
    plt.subplot(413)
    plt.plot("mani", data=debug_data, label="manip", color='b')
    plt.axhline(y=MANI_BUFFER, color='black', linestyle='--', label="buffer")
    plt.xlabel("s (m)")
    plt.ylabel("Manipulability")
    plt.title("Manipulability per Arc length")
    plt.ylim(-0.01, max(debug_data["mani"])*1.2)  
    # plt.xlim(0, debug_data["s"][-1])
    plt.legend()
    plt.grid(True)

    ## Plotting the s/contouring error
    plt.subplot(414)
    plt.plot("contour_error", data=debug_data, label="Contour Error", color='b')
    plt.xlabel("s (m)")
    plt.ylabel("Error (m)")
    plt.title("Contouring Error per Arc length")
    plt.ylim(-max(debug_data["contour_error"])*0.3, max(debug_data["contour_error"])*1.2)  
    # plt.xlim(0, debug_data["s"][-1])
    plt.legend()
    plt.grid(True)

    print_stats(debug_data, "debug_data")
    print_stats(time_data, "time_data")

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_obs", type=bool, default=True)
    
    args = parser.parse_args()
    main(args)