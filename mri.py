import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
import trimesh
from tiny_mano.armatures import *
from tiny_mano.models import *
from skimage import measure


def t2mat(t_vec):
    I = np.eye(4)
    I[0:3, 3] = t_vec
    return I  # 4,4


def r2mat(r_axisangle):
    rot = Rotation.from_rotvec(r_axisangle)
    I = np.eye(4)
    I[0:3, 0:3] = rot.as_dcm()
    return I  # 4,4


class HandMRI:
    def __init__(self):
        self.mri_canonical_global_scale = 1000.0
        self.mri_canonical_global_trans = np.array([137.6553, 232.4672, 69.1232])
        self.mri_canonical_shape_param = np.array(
            [0.5259, -0.0180, 0.1837, 0.0747, 0.0008, -0.0520, -0.0781, 0.0091, -0.0019,
             -0.0037])
        self.mri_canonical_global_rot = np.array([0.9898, 1.1716, 1.3019])
        self.mri_canonical_pose_param = np.array([-5.7645e-01, -4.2942e-01, -3.3077e-01, -2.8438e-01, -5.9076e-01,
                                                  -5.5435e-01, 2.4739e-01, 8.8679e-01, -1.7700e+00, -1.6026e-01,
                                                  6.0865e-02, -2.8645e-01, -1.4039e+00, -9.1460e-01, 3.7365e-01,
                                                  2.1662e-01, 8.8742e-01, -5.6921e-01, -7.0107e-01, -3.6534e-01,
                                                  2.8090e-01, 7.7969e-01, 4.4456e-01, 4.5719e-01, -1.8439e+00,
                                                  -5.5264e-01, 1.7728e+00, -3.2953e-02, -8.0070e-01, 2.1393e-01,
                                                  3.1896e-03, -9.4712e-01, 4.7687e-01, -4.6999e-02, -9.4202e-03,
                                                  4.0925e-01, 1.3740e-01, -6.4839e-02, -4.6928e-01, 6.1413e-01,
                                                  -1.0510e-01, -1.4899e-03, -1.8446e-01, 1.2987e-01,
                                                  -8.2856e-02])
        self.mri_canonical_pose_param_abs = np.array([
            [9.8980e-01, 1.1716e+00, 1.3019e+00],
            [6.4388e-02, 2.0755e-01, 3.0672e-01],
            [3.5124e-02, -1.2438e-01, -1.1941e-02],
            [-6.1480e-02, -8.8771e-03, 1.0302e-01],
            [-4.9296e-02, -2.0554e-01, 3.5221e-01],
            [-9.0540e-02, -8.5413e-02, 7.6242e-02],
            [1.0099e-01, -1.2438e-03, 1.6378e-01],
            [-4.7319e-01, -2.9746e-01, 4.3383e-01],
            [4.4658e-01, 6.5839e-02, -1.4871e-01],
            [-3.2916e-02, 7.1311e-02, 6.5473e-02],
            [-1.1794e-01, -3.1688e-01, 3.2602e-01],
            [-1.2744e-02, -4.7282e-02, 6.6444e-02],
            [1.9754e-02, 1.2303e-01, 2.3759e-01],
            [2.6414e-01, -8.0720e-02, 8.5048e-02],
            [3.6949e-02, -1.1148e-01, -1.7685e-01],
            [6.1712e-02, -1.9793e-01, 4.0913e-01]
        ]).reshape(-1, 3)
        self.mano_mesh = KinematicModel("tiny_mano/mano_right.pkl", MANOArmature)
        # s global position
        self.mano_mesh_verts = self.mano_mesh.verts
        self.mano_mesh_skinning_weights = self.mano_mesh.skinning_weights
        self.mano_mri_c_joints_0 = np.array([0.09331443,0.006153, 0.0060891 ])
        # _, self.mano_mri_c_joints = self.mano_mesh.set_params(pose_abs=self.mri_canonical_pose_param_abs,
        #                                                       pose_glb=self.mri_canonical_global_rot,
        #                                                       shape=self.mri_canonical_shape_param)
        # print(self.mano_mri_c_joints[0])

    def mano_to_vol(self, mano_verts, spacing=[0.5, 0.5, 0.5]):
        pt = mano_verts.squeeze() * 1000.0
        xyz = np.array(pt).astype(np.int)
        max_xyz = np.max(xyz, axis=0) + 1
        min_xyz = np.min(xyz, axis=0) - 1
        D, H, W = max_xyz
        SD, SH, SW = min_xyz
        ds, hs, ws = spacing
        x_ = np.arange(SD, D + 1e-5, step=ds, dtype=np.float32)
        y_ = np.arange(SH, H + 1e-5, step=hs, dtype=np.float32)
        z_ = np.arange(SW, W + 1e-5, step=ws, dtype=np.float32)
        px, py, pz = np.meshgrid(x_, y_, z_, indexing='ij')
        all_pts = np.stack([px, py, pz], -1)  # [D, H, W, 3]
        all_pts /= 1000.0
        return all_pts

    def transfer_weights(self, hand_verts, hand_weights, pts_vol):
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(hand_verts)
        indices = neigh.kneighbors(pts_vol, return_distance=False)
        pts_weights = hand_weights[indices.squeeze()]
        return pts_weights

    def warp_pts(self, pts_vol_mano_c, pts_weights):
        # transfer to right hand
        pts_vol_mano_c_right = pts_vol_mano_c
        pts_vol_mano_c_right[:, 0] *= -1

        new_weights = self.transfer_weights(self.mano_mesh_verts, self.mano_mesh_skinning_weights, pts_vol_mano_c_right)
        verts, _ = self.mano_mesh.set_params(pose_abs=self.mri_canonical_pose_param_abs,
                                             pose_glb=self.mri_canonical_global_rot,
                                             shape=self.mri_canonical_shape_param,
                                             additional_verts=pts_vol_mano_c_right,
                                             additional_verts_weights=new_weights)
        # scaling
        center_joint = self.mano_mri_c_joints_0
        verts = verts - center_joint
        verts = verts * self.mri_canonical_global_scale

        # translation
        offset = self.mri_canonical_global_trans - center_joint
        verts = verts + offset
        return verts

    def labelvol2mesh(self, pts_vol, label_vol, spacing):
        # d, h, w
        label_meshes = []
        for l in np.unique(label_vol):
            if l == 0:
                continue
            label_mask = label_vol.copy()
            label_mask[label_mask != l] = 0
            verts, faces, normals, values = measure.marching_cubes(label_mask, spacing=spacing)
            verts = verts / 1000.
            faces = np.flip(faces, axis=-1)
            label_mesh = trimesh.Trimesh(vertices=verts,
                                      faces=faces,
                                      normals=normals,
                                      process=True,
                                      validate=True)
            label_mesh.apply_translation(pts_vol[0, 0, 0])
            label_meshes.append(label_mesh)
        return label_meshes