import MPCC
import numpy as np
from numpy.linalg import inv
from math import pi
import time

import matplotlib.pyplot as plt
from srmt2.planning_scene import PlanningScene
import argparse
import scipy.io
import os 
import sys

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

obs_radius = 5         # unit: [cm]
obs_speed = 0.05       # unit: [m/s]

def print_stats(data_dict, name, build_options=None):
    """
    data_dict: {'key1': array1, 'key2': array2, …}
    name: 'debug_data' 또는 'time_data' 등의 식별 문자열
    build_options: BuildOptions 객체 (optional)
    """
    # Print build options first if provided
    if build_options is not None:
        print(f"\n=== Build Options ===")
        print(f"  osqp_use_float:           {build_options.osqp_use_float}")
        print(f"  osqp_use_truncate:        {build_options.osqp_use_truncate}")
        print(f"  nn_use_truncate:          {build_options.nn_use_truncate}")
        print(f"  constraints_use_truncate: {build_options.constraints_use_truncate}")
        print(f"  ffp_contract_off:         {build_options.ffp_contract_off}")
        print()

    print(f"\n=== Statistics for {name} ===")

    # Keys that should only show final value (cumulative/tracking variables)
    final_value_keys = ['solve_count', 'max_iter_reached_count', 'max_iter_solve_failed_count']

    for key, arr in data_dict.items():
        # Skip non-array types
        if not isinstance(arr, np.ndarray):
            continue

        # Special handling for top3 data - show nicely formatted
        if key == 'top3_total_iter_counts':
            if len(arr) > 0:
                top3_counts = arr[-1]
                print(f"{key:15s} | Top 3 values: {top3_counts}")
            else:
                print(f"{key:15s} | [empty]")
            continue

        if key == 'top3_solve_nums':
            if len(arr) > 0:
                top3_nums = arr[-1]
                print(f"{key:15s} | Top 3 solve numbers:")
                for i, nums in enumerate(top3_nums):
                    print(f"                  Rank {i+1}: {nums}")
            else:
                print(f"{key:15s} | [empty]")
            continue

        # Special handling for cumulative/tracking variables - only show final value
        if key in final_value_keys:
            if len(arr) > 0:
                last_value = arr[-1]
                print(f"{key:15s} | final value: {last_value}")
            else:
                print(f"{key:15s} | [empty]")
            continue

        # Special handling for object arrays (like nested lists)
        if arr.dtype == object:
            print(f"{key:15s} | [object array - skipped]")
            continue

        flat = arr.flatten()
        print(f"{key:15s} | mean: {flat.mean():10.6f} | min: {flat.min():10.6f} | max: {flat.max():10.6f} | var: {flat.var():10.6f} | std: {flat.std():10.6f} |")
    print()

### NEW FUNCTION START ###
def save_stats_as_txt(data_dict, name, filename, build_options=None):
    """
    print_stats 함수의 콘솔 출력을 그대로 txt 파일에 저장합니다.
    """
    # try...finally 구문을 사용하면途中で에러가 발생해도
    # 반드시 원래의 표준 출력으로 복원되므로 안전합니다.
    original_stdout = sys.stdout  # 원래의 표준 출력(콘솔)을 저장
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            sys.stdout = f  # 표준 출력을 파일로 변경
            print_stats(data_dict, name, build_options) # 이제 이 함수의 print문은 파일에 쓰여집니다.
    finally:
        sys.stdout = original_stdout # 표준 출력을 다시 원래대로(콘솔) 복원

    print(f"Statistics data saved to {filename}")
### NEW FUNCTION END ###

### NEW FUNCTION START ###
def generate_stats_data(data_dict):
    """
    통계 데이터를 계산하고 Matplotlib table을 위한 형식으로 반환합니다.
    """
    col_labels = ['Metric', 'Mean', 'Min', 'Max', 'Var', 'Std']
    table_data = []
    
    for key, arr in data_dict.items():
        if not isinstance(arr, np.ndarray):
            continue
        
        flat = arr.flatten()
        # 데이터를 문자열로 포맷팅하여 리스트에 추가
        row_data = [
            key,
            f"{flat.mean():.4f}",
            f"{flat.min():.4f}",
            f"{flat.max():.4f}",
            f"{flat.var():.4f}",
            f"{flat.std():.4f}"
        ]
        table_data.append(row_data)
        
    return table_data, col_labels

def save_stats_as_image(data_dict, title, filename):
    """
    통계 데이터를 표 이미지로 저장합니다.
    """
    table_data, col_labels = generate_stats_data(data_dict)
    
    if not table_data:
        print(f"No data to generate table for {title}")
        return

    # 표의 크기를 내용에 맞게 동적으로 조절
    num_rows = len(table_data)
    fig_height = max(4, num_rows * 0.5) # 기본 높이 4, 행마다 0.5인치 추가
    
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('tight')
    ax.axis('off')

    # 표 생성
    the_table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')

    # 스타일링
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.1, 1.5) # 너비, 높이 스케일 조절

    plt.title(title, fontsize=16, pad=20)
    fig.tight_layout()
    
    # 이미지 저장
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close(fig) # 메모리 해제를 위해 figure를 닫아줍니다.
    print(f"Statistics table saved to {filename}")
### NEW FUNCTION END ###


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
        [0.40, -0.200, 0.350],
        [0.55, -0.200, 0.55]
    ])

    # 각 장애물의 이동 한계와 속도도 배열로 관리
    obs_limits = np.zeros((num_obstacles, 2, 3))
    # Obstacle 1 & 2: 상한/하한을 초기 위치와 같게 설정하여 고정
    obs_limits[0] = np.array([[0.48,  0.218, 0.421],   # lower limit
                    [0.48,  0.218, 0.621]])  # upper limit
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

    ## Get build options for logging
    build_options = MPCC.BuildOptions.get()

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
    debug_data["iter_count"] = []             # Iteration count
    debug_data["sqp_iter_count"] = []         # SQP Iteration count
    debug_data["total_iter_count"] = []       # Total QP Iteration count across SQP iterations
    debug_data["solve_count"] = []            # Which solveOCP call this is
    debug_data["top3_total_iter_counts"] = []   # Top 3 max total_iter_count values
    debug_data["top3_solve_nums"] = []  # Solve numbers for each top 3 value
    debug_data["max_iter_reached_count"] = []  # Total count of times max_iter (250) was reached
    debug_data["max_iter_solve_failed_count"] = []  # Count of solve failures among max_iter cases

    time_data = {}
    time_data["total"] = []
    time_data["set_env"] = []
    time_data["set_qp"] = []
    time_data["init_solver"] = []
    time_data["solve_qp"] = []
    time_data["get_alpha"] = []
    time_data["scaling_time"] = []
    time_data["permutation_time"] = []
    time_data["factorization_time"] = []
    time_data["rho_updates"] = []

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
        
        if args.is_obs:
            # --- 1. 장애물 이동 로직 (수정된 버전) ---
            # 첫 번째 장애물이 위쪽으로 움직이고 있고(obs_steps > 0), 위쪽 경계선을 넘었을 때
            if obs_steps[0, 2] > 0 and obs_positions[0, 2] >= obs_limits[0, 1, 2]:
                obs_steps[0, 2] *= -1  # 방향을 아래쪽으로 전환
            # 첫 번째 장애물이 아래쪽으로 움직이고 있고(obs_steps < 0), 아래쪽 경계선을 넘었을 때
            elif obs_steps[0, 2] < 0 and obs_positions[0, 2] <= obs_limits[0, 0, 2]:
                obs_steps[0, 2] *= -1  # 방향을 위쪽으로 전환
    
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

        status, state, input, mpc_horizon, compute_time, iter_count, sqp_iter_count, total_iter_count, solve_count, top3_total_iter_counts, top3_solve_nums, max_iter_reached_count, max_iter_solve_failed_count = mpc.runMPC(state, input, obs_positions, obs_radius) if args.is_obs else mpc.runMPC(state, input)
        if status == False:
            print("MPC did not solve properly!!")
            break

        state = integrator.simTimeStep(state, input)

        q = state[:robot_dof]
        qdot = input[:robot_dof]
        qddot = (qdot - qdot_pre)/mpc.Ts
        x = robot.getEEPosition(q)
        xdot = robot.getEEJacobianv(q) @ qdot
        x_speed = np.linalg.norm(xdot)
        rotation = robot.getEEOrientation(q)
        s = state[-2]
        vs = state[-1]
        dVs = input[-1]
        sel_min_dist, _ = selcolNN.calculateMlpOutput(q)
        num_obstacles = obs_positions.shape[0]
        env_min_dists = []
        for obs_idx in range(num_obstacles):
            input_vec = np.concatenate([q, obs_positions[obs_idx]])
            dist, _ = envcolNN.calculateMlpOutput(input_vec)
            env_min_dists.append(dist)
        env_min_dist = np.array(env_min_dists)

        mani = robot.getEEManipulability(q)
        contour_error = mpc.getContourError(s, x)
        pred_ee_T = np.zeros([mpc.pred_horizon + 1, 4, 4])
        ref_ee_T = np.zeros([mpc.pred_horizon + 1, 4, 4])
        for i in range(mpc.pred_horizon + 1):
            pred_ee_T[i, :3, :3] = robot.getEEOrientation(mpc_horizon[i]["state"][:robot_dof])
            pred_ee_T[i, :3, 3]  = robot.getEEPosition(mpc_horizon[i]["state"][:robot_dof])
            ref_ee_T[i, :3, 3], ref_ee_T[i, :3, :3] = mpc.getRefPose(mpc_horizon[i]["state"][-2])

        pc.display(q)

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
        debug_data["sqp_iter_count"].append(sqp_iter_count)
        debug_data["total_iter_count"].append(total_iter_count)
        debug_data["solve_count"].append(solve_count)
        debug_data["top3_total_iter_counts"].append(top3_total_iter_counts)
        debug_data["top3_solve_nums"].append(top3_solve_nums)
        debug_data["max_iter_reached_count"].append(max_iter_reached_count)
        debug_data["max_iter_solve_failed_count"].append(max_iter_solve_failed_count)
        time_data["total"].append(compute_time["total"])
        time_data["set_env"].append(compute_time["set_env"])
        time_data["set_qp"].append(compute_time["set_qp"])
        time_data["init_solver"].append(compute_time["init_solver"])
        time_data["solve_qp"].append(compute_time["solve_qp"])
        time_data["get_alpha"].append(compute_time["get_alpha"])
        time_data["scaling_time"].append(compute_time["scaling_time"])
        time_data["permutation_time"].append(compute_time["permutation_time"])
        time_data["factorization_time"].append(compute_time["factorization_time"])
        time_data["rho_updates"].append(compute_time["rho_updates"])

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
        contour_error_msg.data = contour_error*100
        path_pub.publish(path_msg)
        splined_path_pub.publish(splined_path_msg)
        local_path_pub.publish(local_path_msg)
        ref_local_path_pub.publish(ref_local_path_msg)
        ee_speed_pub.publish(ee_speed_msg)
        mani_pub.publish(mani_msg)
        sel_min_dist_pub.publish(sel_min_dist_msg)
        env_min_dist_pub.publish(env_min_dist_msg)
        contour_error_pub.publish(contour_error_msg)
        
        if np.linalg.norm((spline_pos[-1] - x), 2) < 1E-2 and np.linalg.norm(MPCC.Log(spline_ori[-1].T @ rotation), 2) < 1E-2 and abs(state[-2] - 1.) < 1E-2:
            print("End point reached!!!")
            break
        end = time.time()
        elapsed = end - start
        if elapsed < mpc.Ts:
            time.sleep(mpc.Ts - elapsed)
        qdot_pre = qdot
        time_idx += 1
        rclpy.spin_once(node, timeout_sec=0.0)

    node.destroy_node()
    rclpy.shutdown()

    # ▼▼▼▼▼ ReLU deactivation 통계를 가져와서 출력합니다 ▼▼▼▼▼
    relu_deactivation_ratios = envcolNN.NNmodel.getReluDeactivationRatios()
    relu_total_units = envcolNN.NNmodel.getReluTotalUnits()
    relu_avg_deactivated = envcolNN.NNmodel.getReluAvgDeactivatedCounts()
    relu_min_deactivated = envcolNN.NNmodel.getReluMinDeactivatedCounts()
    relu_max_deactivated = envcolNN.NNmodel.getReluMaxDeactivatedCounts()

    print("\n" + "="*80)
    print("=== ReLU Deactivation Statistics ===")
    print("="*80)
    print(f"{'Layer':<8} {'Total Units':<15} {'Avg Deact':<15} {'Min Deact':<15} {'Max Deact':<15} {'Deact %':<15}")
    print("-"*80)
    for i in range(len(relu_deactivation_ratios)):
        print(f"{i:<8} {relu_total_units[i]:<15} {relu_avg_deactivated[i]:<15.2f} {relu_min_deactivated[i]:<15} {relu_max_deactivated[i]:<15} {relu_deactivation_ratios[i]*100:<15.2f}")
    print("="*80)
    print()

    inference_times = envcolNN.NNmodel.getInferenceTimes()
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
    
    if args.name:
        # 1. 기본 출력 폴더 경로를 정의합니다.
        output_folder = os.path.join("../result/main_w_sim_three", args.name)
        
        # 2. 파일이 저장될 전체 경로를 만듭니다.
        stats_path = os.path.join(output_folder, f"{args.name}_inference_times.png")
        
        # 3. (★★가장 중요★★) 저장할 폴더가 없으면 자동으로 생성합니다.
        # exist_ok=True 옵션은 폴더가 이미 있어도 에러를 발생시키지 않습니다.
        os.makedirs(output_folder, exist_ok=True)
        
        fig_name = stats_path
        plt.savefig(fig_name)
        print(f"Saved figure to {fig_name}")
    
    # scipy.io.savemat과 호환되지 않는 가변 길이 리스트는 제외
    excluded_keys = ["top3_solve_nums", "top3_total_iter_counts"]
    for key in excluded_keys:
        if key in debug_data:
            del debug_data[key]

    for key in debug_data:
        debug_data[key] = np.array(debug_data[key])
    for key in time_data:
        time_data[key] = np.array(time_data[key])

    if args.name:
        # 1. 기본 출력 폴더 경로를 정의합니다.
        output_folder = os.path.join("../result/main_w_sim_three", args.name)

        # 2. 파일이 저장될 전체 경로를 만듭니다.
        stats_path_1 = os.path.join(output_folder, f"{args.name}_debug_data.mat")
        stats_path_2 = os.path.join(output_folder, f"{args.name}_time_data.mat")
    else:
        stats_path_1 = "debug_data.mat"
        stats_path_2 = "time_data.mat"

    scipy.io.savemat(stats_path_1, debug_data)
    print(f"Data written to {stats_path_1}")
    scipy.io.savemat(stats_path_2, time_data)
    print(f"Data written to {stats_path_2}")

    print("mean nmpc time[sec]: {:0.6f} ".format(np.mean(time_data["total"])))
    print("max nmpc time[sec]: {:0.6f} ".format(np.max(time_data["total"])))

    plt.figure(figsize=(14, 8))
    plt.plot(time_data["total"], label="Total Time", color='b')
    plt.plot(time_data["set_env"], label="Set Env Time", color='m')
    plt.plot(time_data["set_qp"], label="Set QP Time", color='g')
    plt.plot(time_data["init_solver"], label="Init Solver Time", color='orange')
    plt.plot(time_data["solve_qp"], label="Solve QP (ADMM+Fact)", color='r')
    plt.plot(time_data["scaling_time"], label="Scaling Time", color='purple', linestyle='--')
    plt.plot(time_data["permutation_time"], label="Permutation Time", color='brown', linestyle='--')
    plt.plot(time_data["factorization_time"], label="Factorization Time", color='pink', linestyle='--')
    plt.plot(time_data["get_alpha"], label="Get Alpha Time", color='c')
    plt.axhline(y=mpc.Ts, color='black', linestyle='--', label="Ts")
    plt.xlabel("Time Step")
    plt.ylabel("Time (s)")
    plt.title("Computation Times per Time Step")
    plt.ylim(-0.01, 0.2)
    plt.xlim(0, len(time_data["total"]))
    plt.legend()
    plt.grid(True)

    if args.name:
        # 1. 기본 출력 폴더 경로를 정의합니다.
        output_folder = os.path.join("../result/main_w_sim_three", args.name)
        
        # 2. 파일이 저장될 전체 경로를 만듭니다.
        stats_path = os.path.join(output_folder, f"{args.name}_computation_times.png")

        plt.savefig(stats_path)
        print(f"Saved figure to {stats_path}")

    fig=plt.figure(figsize=(14, 8))
    fig.subplots_adjust(hspace=1)
    plt.subplot(411)
    plt.plot("ee_speed", data=debug_data, label="ee_speed", color='r')
    plt.xlabel("s (m)")
    plt.ylabel("Speed (m/s)")
    plt.title("EE Speed per Arc length")
    plt.ylim(-0.01, max(max(debug_data["ee_speed"]), max(debug_data["vs"]))*1.2)  
    plt.legend()
    plt.grid(True)

    plt.subplot(412)
    plt.plot("sel_min_dist", data=debug_data, label="minimum distance", color='b')
    plt.axhline(y=SLECOL_BUFFER, color='black', linestyle='--', label="buffer")
    plt.xlabel("s (m)")
    plt.ylabel("distance (cm)")
    plt.title("Minimum distance per Arc length")
    plt.ylim(-0.01, max(debug_data["sel_min_dist"])*1.2)  
    plt.legend()
    plt.grid(True)

    plt.subplot(413)
    plt.plot("mani", data=debug_data, label="manip", color='b')
    plt.axhline(y=MANI_BUFFER, color='black', linestyle='--', label="buffer")
    plt.xlabel("s (m)")
    plt.ylabel("Manipulability")
    plt.title("Manipulability per Arc length")
    plt.ylim(-0.01, max(debug_data["mani"])*1.2)  
    plt.legend()
    plt.grid(True)

    plt.subplot(414)
    plt.plot("contour_error", data=debug_data, label="Contour Error", color='b')
    plt.xlabel("s (m)")
    plt.ylabel("Error (m)")
    plt.title("Contouring Error per Arc length")
    plt.ylim(-max(debug_data["contour_error"])*0.3, max(debug_data["contour_error"])*1.2)  
    plt.legend()
    plt.grid(True)
    
    ### === NEWLY ADDED SECTION START === ###
    # 마지막 4-subplot 그림을 저장합니다.
    if args.name:
        # 1. 기본 출력 폴더 경로를 정의합니다.
        output_folder = os.path.join("../result/main_w_sim_three", args.name)

        # 2. 저장할 폴더가 (아직) 없으면 자동으로 생성합니다.
        #    (위의 다른 저장 로직에서 이미 생성했을 수도 있지만, 안전을 위해 확인)
        os.makedirs(output_folder, exist_ok=True)
        
        # 3. 그림 파일의 전체 경로를 만듭니다.
        stats_path = os.path.join(output_folder, f"{args.name}_performance_metrics.png")

        # 4. 현재 figure (plt)를 저장합니다.
        plt.savefig(stats_path)
        print(f"Saved figure to {stats_path}")
    ### === NEWLY ADDED SECTION END === ###

    ### MODIFIED SECTION START ###
    if args.name:
        # 1. 기본 출력 폴더 경로를 정의합니다.
        output_folder = os.path.join("../result/main_w_sim_three", args.name)

        # 2. 저장할 폴더가 없으면 자동으로 생성합니다.
        os.makedirs(output_folder, exist_ok=True)

        # 3. debug_data와 time_data를 합칩니다.
        combined_stats_data = {**debug_data, **time_data}

        # 4. 저장될 텍스트 파일의 전체 경로를 만듭니다.
        stats_path = os.path.join(output_folder, f"{args.name}_debug_stats.txt") # 확장자를 .txt로 변경

        # 5. 새로 만든 함수를 호출하여 텍스트 파일로 저장합니다.
        save_stats_as_txt(
            combined_stats_data,
            name="Combined Statistics (Debug & Time Data)",
            filename=stats_path,
            build_options=build_options
        )

        # ▼▼▼▼▼ 6. ReLU deactivation 통계를 텍스트 파일로 저장합니다 ▼▼▼▼▼
        relu_stats_path = os.path.join(output_folder, f"{args.name}_relu_deactivation_stats.txt")
        with open(relu_stats_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("=== ReLU Deactivation Statistics ===\n")
            f.write("="*80 + "\n")
            f.write(f"{'Layer':<8} {'Total Units':<15} {'Avg Deact':<15} {'Min Deact':<15} {'Max Deact':<15} {'Deact %':<15}\n")
            f.write("-"*80 + "\n")
            for i in range(len(relu_deactivation_ratios)):
                f.write(f"{i:<8} {relu_total_units[i]:<15} {relu_avg_deactivated[i]:<15.2f} {relu_min_deactivated[i]:<15} {relu_max_deactivated[i]:<15} {relu_deactivation_ratios[i]*100:<15.2f}\n")
            f.write("="*80 + "\n")
        print(f"ReLU deactivation statistics saved to {relu_stats_path}")

    ### MODIFIED SECTION END ###

    # 기존의 콘솔 출력은 그대로 유지하거나, 필요 없다면 주석 처리/삭제 가능
    print_stats(debug_data, "debug_data")
    print_stats(time_data, "time_data")
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_obs", type=bool, default=True)
    parser.add_argument("--name", type=str, default=None, help="A name for the run, used as a prefix for saved files.")
    
    args = parser.parse_args()
    main(args)