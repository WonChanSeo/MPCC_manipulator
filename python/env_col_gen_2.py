from __future__ import annotations
import argparse, time, numpy as np
from math import pi
import rclpy, tf2_ros
from rclpy.node     import Node
from srmt2.planning_scene import PlanningScene, VisualSimulator, add_table
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import sys
import os
import datetime as dt
from multiprocessing import Process, Queue
import multiprocessing as mp
import pickle

title_font = {
                'fontsize': 16,
                'fontweight': 'bold'
            }

np.printoptions(precision=3, suppress=True, linewidth=100, threshold=10000)
np.set_printoptions(threshold=sys.maxsize)
title_font = {
    'fontsize': 16,
    'fontweight': 'bold'
}
    
# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def get_static_tfs():
    rclpy.init(args=None)
    node = rclpy.create_node("tf_collector")
    buf  = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)
    
    def lookup(tgt, src):
            while rclpy.ok():
                try:
                    tf = buf.lookup_transform(tgt, src, rclpy.time.Time(),
                                            rclpy.duration.Duration(seconds=1.0))
                    return tf.transform
                except (tf2_ros.LookupException,
                        tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException):
                    rclpy.spin_once(node, timeout_sec=0.05)

    base2fr3_tf  = lookup("base_link", "fr3_link0")
    base2cam_tf  = lookup("base_link", "depth_camera_link")

    base2fr3 = np.array([base2fr3_tf.translation.x,
                         base2fr3_tf.translation.y,
                         base2fr3_tf.translation.z])
    base2cam = np.array([base2cam_tf.translation.x,
                         base2cam_tf.translation.y,
                         base2cam_tf.translation.z])
    cam2view = R.from_quat([base2cam_tf.rotation.x,
                            base2cam_tf.rotation.y,
                            base2cam_tf.rotation.z,
                            base2cam_tf.rotation.w]).apply([0,0,1])

    node.destroy_node()
    return base2fr3, base2cam, cam2view

def work(proc_id: int,
         result: Queue,
         base2fr3: np.ndarray,
         base2cam: np.ndarray,
         cam2view: np.ndarray,
         num_envs_per_thread: list[int],
         voxel_res: float,
         num_q_per_env: int,
         joint_limit: np.ndarray,
         obs_size_limit: np.ndarray,
         obs_num_limit: np.ndarray,
         workspace: np.ndarray):
    
    rclpy.init(args=None)
    
    t0 = time.time()
    np.random.seed(proc_id)
    dataset = []
    
    # --------------------- Planning scene & camera ------------------------- #
    pc = PlanningScene(arm_names=["fr3"],arm_dofs=[7], base_link="base_link", topic_name="planning_scene_" + str(proc_id))
    vs = VisualSimulator(640, 576, 504.118, 504.12, 0.25, 2.21) # Azure Kinect SDK (mode: NFOV_UNBINNED)
    vs.set_cam_and_target_pose(base2cam, base2cam + cam2view)
    vs.set_scene_bounds(base2fr3 + [-.9, -.9, 0.0], base2fr3 + [.9, .9, 1.8])
    vs.set_grid_resolution(int(1.8 / voxel_res))
    print("Thread {}: Planning scene and camera created.".format(proc_id))
    
    link_list = ["fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3",
                     "fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7", "fr3_hand"]

    # -------------------------- Create environments ------------------------ #
    for env_iter in range(num_envs_per_thread[proc_id]):
        q_set = []
        min_dist_set = []

        env_idx = (sum(num_envs_per_thread[0:proc_id]) if proc_id != 0 else 0) + env_iter    
        num_obs = np.random.randint(obs_num_limit[0].item(), obs_num_limit[1].item())
        for obs_idx in range(num_obs):
            obs_pos = np.random.uniform(workspace[0], workspace[1]) # angle, radius, height
            obs_angle = obs_pos[0]
            obs_radius = obs_pos[1]
            obs_height = obs_pos[2]
            obs_x, obs_y,obs_z = obs_radius*np.cos(obs_angle)+base2fr3[0], obs_radius*np.sin(obs_angle)+base2fr3[1], obs_height+base2fr3[2]
            pc.add_box(name = "obs_"+str(obs_idx), 
                        dim = np.random.uniform(obs_size_limit[0], obs_size_limit[1]),
                        pos = [obs_x, obs_y, obs_z],
                        quat = R.random().as_quat()
                        )
        vs.load_scene(pc)
        voxel_grid = vs.generate_voxel_occupancy()
        voxel_grid = voxel_grid.reshape(int(1.8 / voxel_res), int(1.8 / voxel_res), int(1.8 / voxel_res))
        
        # random configuration
        for joint_iter in range(num_q_per_env):
            q = np.random.uniform(low=joint_limit[0,:], high=joint_limit[1,:])
            pc.display(q)
            env_min_dist = pc.get_links_min_distances(link_list, q, False, True)*100 # [m] -> [cm]
            ## Save data
            q_set.append(q)
            min_dist_set.append(env_min_dist)
            # print("env: {}, q: {}, min_dist: {}".format(env_iter, q, env_min_dist))
        
        #  Reset planning scene
        pc.remove_all_objects()

        # Save data
        env_dataset = {"env_idx": env_idx,
                        "q": np.array(q_set),
                        "min_dist": np.array(min_dist_set),
                        "occupancy": voxel_grid}
        dataset.append(env_dataset)

        if ((env_iter+1) / num_envs_per_thread[proc_id]) % 0.01 == 0 : 
            print("{0:.1f} sec {1:.1f}% completed on th:{2}.".format(time.time()-t0, 
                                                                        (env_iter+1) / num_envs_per_thread[proc_id] * 100, 
                                                                        proc_id))
    result.put(dataset)
    return
# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main(args):
    # Number of threads
    if args.num_th > os.cpu_count():
        args.num_th = os.cpu_count()
    print("core: {}".format(args.num_th))
    
    # -------------------------- Parameter setting -------------------------- #
    joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],  # min
                            [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]]) # max

    workspace = np.array([[-pi/3, 0.4, 0.0],       # min (Angle, Radius, Height)
                          [pi/3,  0.75, 0.8]])     # max (Angle, Radius, Height)
    obs_size_limit = np.array([[0.1, 0.1, 0.1],    # min (Width, Depth, Height)
                               [0.2, 0.2, 0.2]])   # max (Width, Depth, Height)
    obs_num_limit = np.array([5,     # min
                              10])    # max

    # -------------------------- static TF fetch ---------------------------- #
    base2fr3, base2cam, cam2view = get_static_tfs()
    base2fr3.flags.writeable = False
    
    num_envs_per_thread = [args.num_env // args.num_th + (1 if i < args.num_env % args.num_th else 0) for i in range(args.num_th)]
    print("num_envs_per_thread: ", num_envs_per_thread)
    
    
    result = Queue()
    threads = []
    dataset = []
    
    # -------------------------- Multi threading --------------------------- #
    print("Create multi-threading!")
    for i in range(args.num_th):
        th = Process(target=work,
                     args=(i, 
                           result, 
                           base2fr3, 
                           base2cam, 
                           cam2view, 
                           num_envs_per_thread, 
                           args.voxel_res, 
                           args.num_q_per_env, 
                           joint_limit, 
                           obs_size_limit, 
                           obs_num_limit, 
                           workspace))
        threads.append(th)
    
    print("Start multi-threading!")
    for i in range(args.num_th):
        threads[i].start()

    for i in range(args.num_th):
        data = result.get()
        dataset = dataset + data

    for i in range(args.num_th):
        threads[i].join()
    print("Finish multi-threading!")
        
    
    # -------------------------- Save dataset ------------------------------- #
    data_dir = "env_data"
    if not os.path.exists(data_dir): os.mkdir(data_dir)
    
    date = dt.datetime.now()
    data_dir = data_dir + "/{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    os.mkdir(data_dir)

    with open(data_dir + "/dataset.pickle", "wb") as f:
        pickle.dump(dataset,f)

    with open(data_dir + "/param_setting.txt", "w", encoding='UTF-8') as f:
        params = {"num_env": args.num_env,
                  "num_q_per_env": args.num_q_per_env,
                  "voxel_res": args.voxel_res,
                  "seed": args.seed}
        for param, value in params.items():
            f.write(f'{param} : {value}\n')


    # import shutil
    # folder_path = "data/"
    # num_save = 3
    # order_list = sorted(os.listdir(folder_path), reverse=True)[1:]
    # remove_folder_list = order_list[num_save:]
    # for rm_folder in remove_folder_list:
    #     shutil.rmtree(folder_path+rm_folder)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--num_th", type=int, default=30)
    p.add_argument("--num_env", type=int, default=2000)
    p.add_argument("--num_q_per_env", type=int, default=500)
    p.add_argument("--voxel_res",type=float, default=0.05)
    p.add_argument("--seed",     type=int,   default=0)
    main(p.parse_args())
    rclpy.shutdown()
