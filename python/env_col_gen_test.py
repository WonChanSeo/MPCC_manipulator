#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast demo of PlanningScene + TF query + sphere‑tracking.
"""

from __future__ import annotations
import argparse, time, numpy as np
from math import pi
import rclpy, tf2_ros
from rclpy.node     import Node
from srmt2.planning_scene import PlanningScene, VisualSimulator, add_table
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

title_font = {
                'fontsize': 16,
                'fontweight': 'bold'
            }

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def lookup_once(buf: tf2_ros.Buffer,
                tgt: str, src: str,
                node: Node,
                timeout: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """
    TF(tgt←src) 를 timeout 안에 찾아서 (xyz, quat) 로 반환.
    두 프레임이 반대 방향으로만 존재하면 역변환으로 대체.
    """
    t_end = time.time() + timeout
    while rclpy.ok() and time.time() < t_end:
        try:                             # ① 정방향 시도
            tf = buf.lookup_transform(tgt, src, rclpy.time.Time())
            break
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            try:                         # ② 역방향 시도 → 역행렬
                tf_inv = buf.lookup_transform(src, tgt, rclpy.time.Time())
                # 역변환 구하기
                q = np.array([tf_inv.transform.rotation.x,
                              tf_inv.transform.rotation.y,
                              tf_inv.transform.rotation.z,
                              tf_inv.transform.rotation.w])
                Rmat = R.from_quat(q).as_matrix().T
                p_inv = np.array([tf_inv.transform.translation.x,
                                  tf_inv.transform.translation.y,
                                  tf_inv.transform.translation.z])
                p = -Rmat @ p_inv
                q_inv = R.from_matrix(Rmat).as_quat()
                return p, q_inv
            except tf2_ros.LookupException:
                pass                     # 아직 프레임이 안 들어옴
            rclpy.spin_once(node, timeout_sec=0.05)

    else:
        raise RuntimeError(f"TF {tgt}←{src} not available in {timeout}s")

    # 정방향 성공 → xyz, quat 반환
    xyz = np.array([tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z])
    quat = np.array([tf.transform.rotation.x,
                     tf.transform.rotation.y,
                     tf.transform.rotation.z,
                     tf.transform.rotation.w])
    return xyz, quat
# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main(args):
    # ----------------------------- ROS init -------------------------------- #
    rclpy.init()
    node = Node("scene_bench")
    buf  = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)

    # -------------------------- static TF fetch ---------------------------- #
    base2fr3, _ = lookup_once(buf, "base_link", "fr3_link0", node)
    base2cam, cam_quat = lookup_once(buf, "base_link", "depth_camera_link", node)
    cam2view = R.from_quat(cam_quat).apply([0, 0, 1])
    
    # -------------------------- Parameter setting -------------------------- #
    joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],  # min
                            [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]]) # max

    workspace = np.array([[-pi/3, 0.4, 0.0],       # min (Angle, Radius, Height)
                          [pi/3,  0.75, 0.8]])     # max (Angle, Radius, Height)
    obs_size_limit = np.array([[0.1, 0.1, 0.1],    # min (Width, Depth, Height)
                               [0.2, 0.2, 0.2]])   # max (Width, Depth, Height)
    obs_num_limit = np.array([2,     # min
                              5])    # max

    # --------------------- Planning scene & camera ------------------------- #
    pc = PlanningScene(arm_names=["fr3"],arm_dofs=[7], base_link="base_link", topic_name="planning_scene")
    vs = VisualSimulator(640, 576, 504.118, 504.12, 0.25, 2.21) # Azure Kinect SDK (mode: NFOV_UNBINNED)
    vs.set_cam_and_target_pose(base2cam, base2cam + cam2view)
    vs.set_scene_bounds(base2fr3 + [-.9, -.9, 0.0], base2fr3 + [.9, .9, 1.8])
    vs.set_grid_resolution(int(1.8 / args.voxel_res))
    
    # -------------------------- Create environments ------------------------ #
    for env_iter in range(args.num_env):
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
                        # quat = R.random().as_quat()
                        quat = [0, 0, 0, 1]
                        )
        pc.display(np.array([0, 0, 0, 0, 0, 0, 0]))
        vs.load_scene(pc)
        voxel_grid = vs.generate_voxel_occupancy()
        voxel_grid = voxel_grid.reshape(int(1.8 / args.voxel_res), int(1.8 / args.voxel_res), int(1.8 / args.voxel_res))
        
        ax = plt.figure(1).add_subplot(projection='3d')
        ax.voxels(voxel_grid)
        ax.set_title("voxel grid", fontsize=16, fontweight='bold', pad=20)
        plt.show()
        
        link_list = ["fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3",
                     "fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7", "fr3_hand"]
        
        # random configuration
        for joint_iter in range(args.num_q_per_env):
            q = np.random.uniform(low=joint_limit[0,:], high=joint_limit[1,:])
            pc.display(q)
            env_min_dist = pc.get_links_min_distances(link_list, q, False, True)*100 # [m] -> [cm]
            print("min_dist: {}".format( env_min_dist))
            time.sleep(10)
        
        pc.remove_all_objects()
        
    rclpy.shutdown()

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_env", type=int, default=100)
    p.add_argument("--num_q_per_env", type=int, default=10)
    p.add_argument("--voxel_res",type=float, default=0.05)
    p.add_argument("--seed",     type=int,   default=0)
    main(p.parse_args())