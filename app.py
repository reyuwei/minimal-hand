import cv2
import keyboard
import numpy as np
import open3d as o3d
import pygame
from transforms3d.axangles import axangle2mat

from mri import HandMRI
import config
from capture import OpenCVCapture
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from utils import OneEuroFilter, imresize
from wrappers import ModelPipeline
from utils import *
import sys
from tiny_nerf.runnerf import TinyNerf
import SimpleITK as sitk

# mri_spacing = [0.5, 0.5, 0.5]
mri_spacing = [0.8, 0.8, 0.8]


def live_application(capture):
    """
    Launch an application that reads from a webcam and estimates hand pose at
    real-time.

    The captured hand must be the right hand, but will be flipped internally
    and rendered.

    Parameters
    ----------
    capture : object
      An object from `capture.py` to read capture stream from.
    """
    ############ output visualization ############
    view_mat = axangle2mat([1, 0, 0], np.pi)  # align different coordinate systems
    window_size = 1080

    hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
    hand_mri = HandMRI()
    tnerf = TinyNerf()

    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = \
        o3d.utility.Vector3dVector(np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
    mesh.compute_vertex_normals()

    bone_pcl = o3d.geometry.PointCloud()
    mri_pcl = o3d.geometry.PointCloud()

    # viewer = o3d.visualization.Visualizer()
    # viewer.create_window(
    #     width=window_size + 1, height=window_size + 1,
    #     window_name='Minimal Hand - output'
    # )
    # viewer.add_geometry(mesh)
    #
    # view_control = viewer.get_view_control()
    # cam_params = view_control.convert_to_pinhole_camera_parameters()
    # extrinsic = cam_params.extrinsic.copy()
    # extrinsic[0:3, 3] = 0
    # cam_params.extrinsic = extrinsic
    # cam_params.intrinsic.set_intrinsics(
    #     window_size + 1, window_size + 1, config.CAM_FX, config.CAM_FY,
    #     window_size // 2, window_size // 2
    # )
    # view_control.convert_from_pinhole_camera_parameters(cam_params)
    # view_control.set_constant_z_far(1000)

    # render_option = viewer.get_render_option()
    # render_option.load_from_json('./render_option.json')
    # viewer.update_renderer()

    ############ input visualization ############
    # pygame.init()
    # display = pygame.display.set_mode((window_size, window_size))
    # pygame.display.set_caption('Minimal Hand - input')

    ############ misc ############
    mesh_smoother = OneEuroFilter(4.0, 0.0)
    clock = pygame.time.Clock()
    # model = ModelPipeline()
    frameid = 0
    while True:
        frame_large = capture.read()
        if frame_large is None:
            break

        frameid += 1
        if frameid % 2 != 0:
            continue
        print(frameid)
        if os.path.exists("../video/ev_20200708_151823/{:06d}_mri_vol.nii.gz".format(frameid)):
            continue
        # if frameid != 68:
        #     continue

        if frame_large.shape[0] > frame_large.shape[1]:
            margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
            frame_large = frame_large[margin:-margin]
        else:
            margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
            frame_large = frame_large[:, margin:-margin]
        frame_large = np.flip(frame_large, axis=1).copy()
        frame = imresize(frame_large, (128, 128))
        # _, theta_mpii = model.process(frame)
        # theta_mano = mpii_to_mano(theta_mpii)
        # np.save("../video/ev_20200708_151823/{:06d}_abs_quat.npy".format(frameid), theta_mano)
        theta_mano = np.load("../video/ev_20200708_151823/{:06d}_abs_quat.npy".format(frameid))
        v, j = hand_mesh.set_abs_quat(theta_mano)

        # generate mri
        print("=======generate mri")
        pts_vol = hand_mri.mano_to_vol(v, spacing=mri_spacing)
        vol_size = pts_vol.shape[:3]
        # np.save("../video/ev_20200708_151823/{:06d}_mri_vol_q_size.npy".format(frameid), vol_size)

        # transfer weights and back to mano canonical
        print("=======transfer weights")
        pts_vol_flatten = pts_vol.reshape(-1, 3)
        pts_weights = hand_mri.transfer_weights(v, hand_mesh.weights, pts_vol_flatten)
        print("========back to mano canonical")
        pts_vol_mano_c, j_mano_c = hand_mesh.reverse_abs_quat(pts_vol_flatten, j, pts_weights, theta_mano)
        # v_c, v_j_c = hand_mesh.reverse_abs_quat(v, j, hand_mesh.weights, theta_mano)
        # np.savetxt("../video/ev_20200708_151823/{:06d}_mri_vol_manoc.xyz".format(frameid), v_c)
        # continue

        # warp to mri canonical
        print("=======warp to mri canonical")
        mri_v = hand_mri.warp_pts(pts_vol_mano_c, pts_weights)
        # np.save("../video/ev_20200708_151823/{:06d}_mri_vol_q.npy".format(frameid), mri_v)
        # mri_v_hand = hand_mri.warp_pts(v_c, hand_mesh.weights)
        # np.savetxt("../video/ev_20200708_151823/{:06d}_mri_vol_mano.xyz".format(frameid), mri_v_hand)
        # continue

        # run tiny nerf
        print("=======run nerf")
        inside_mask = np.ones(mri_v.shape[0]).astype(np.int)
        inside_mask = inside_mask > 0
        mri, label, mri_nii, label_nii = tnerf.render_pts(mri_v, inside_mask, vol_size, spacing=mri_spacing)
        sitk.WriteImage(mri_nii, "../video/ev_20200708_151823/{:06d}_mri_vol.nii.gz".format(frameid))
        sitk.WriteImage(mri_nii,
                        "/data/new_disk/liyuwei/nnUNet/Hand_MRI_segdata/tracking_test/{:06d}_mri_vol.nii.gz".format(
                            frameid))
        sitk.WriteImage(label_nii, "../video/ev_20200708_151823/{:06d}_mri_vol_label.nii.gz".format(frameid))

        mri_color = np.tile(mri.reshape(-1, 1), 3)
        label_color = np.zeros_like(mri_color)
        label = label.reshape(-1)
        label_color[label == 1] = [1, 0, 0]
        label_color[label == 2] = [0, 1, 0]
        label_mask = label > 0

        label_meshes = hand_mri.labelvol2mesh(pts_vol, label.reshape(vol_size), mri_spacing)

        v *= 2  # for better visualization
        v = v * 1000 + np.array([0, 0, 400])
        v = mesh_smoother.process(v)
        mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
        mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
        mesh.paint_uniform_color(config.HAND_COLOR)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        pts_vol_flatten *= 2
        pts_vol_flatten = pts_vol_flatten * 1000 + np.array([0, 0, 400])
        pts_vol_flatten = np.matmul(view_mat, pts_vol_flatten.T).T
        mri_pcl.points = o3d.utility.Vector3dVector(pts_vol_flatten)
        mri_pcl.colors = o3d.utility.Vector3dVector(mri_color)

        bone_pcl.points = o3d.utility.Vector3dVector(pts_vol_flatten[label_mask])
        bone_pcl.colors = o3d.utility.Vector3dVector(label_color[label_mask])
        # marching cube
        for m in range(len(label_meshes)):
            label_meshes[m].vertices *= 2
            label_meshes[m].vertices = label_meshes[m].vertices * 1000 + np.array([0, 0, 400])
            label_meshes[m].vertices = np.matmul(view_mat, label_meshes[m].vertices.T).T
            label_meshes[m].export("../video/ev_20200708_151823/{:06d}_mri_label{:d}.ply".format(frameid, m + 1))

        # save mesh to file
        # np.save("../video/ev_20200708_151823/{:06d}_mri_vol.npy".format(frameid), pts_vol_flatten)
        o3d.io.write_triangle_mesh("../video/ev_20200708_151823/{:06d}.ply".format(frameid), mesh)
        o3d.io.write_point_cloud("../video/ev_20200708_151823/{:06d}_mri.ply".format(frameid), mri_pcl)
        o3d.io.write_point_cloud("../video/ev_20200708_151823/{:06d}_bone.ply".format(frameid), bone_pcl)
        cv2.imwrite("../video/ev_20200708_151823/{:06d}.png".format(frameid), frame_large)

        # for some version of open3d you may need `viewer.update_geometry(mesh)`
        # viewer.update_geometry()

        # viewer.poll_events()
        #
        # display.blit(
        #   pygame.surfarray.make_surface(
        #     np.transpose(
        #       imresize(frame_large, (window_size, window_size)
        #     ), (1, 0, 2))
        #   ),
        #   (0, 0)
        # )
        # pygame.display.update()

        # if keyboard.is_pressed("esc"):
        #     break

        clock.tick(30)


import os

if __name__ == '__main__':
    videoname = sys.argv[1] if len(sys.argv) > 1 else None
    gpu = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    live_application(OpenCVCapture(videoname))
