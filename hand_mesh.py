import numpy as np
from transforms3d.quaternions import quat2mat
from scipy.spatial.transform import Rotation
from config import *
from kinematics import *
from utils import *


class HandMesh():
    """
    Wrapper for the MANO hand model.
    """
    def __init__(self, model_path):
        """
        Init.

        Parameters
        ----------
        model_path : str
          Path to the MANO model file. This model is converted by `prepare_mano.py`
          from official release.
        """
        params = load_pkl(model_path)
        self.verts = params['verts']
        self.faces = params['faces']
        self.weights = params['weights']
        self.joints = params['joints']

        self.n_verts = self.verts.shape[0]
        self.n_faces = self.faces.shape[0]

        self.ref_pose = []
        self.ref_T = []
        for j in range(MANOHandJoints.n_joints):
            parent = MANOHandJoints.parents[j]
            if parent is None:
                self.ref_T.append(self.verts)
                self.ref_pose.append(self.joints[j])
            else:
                self.ref_T.append(self.verts - self.joints[parent])
                self.ref_pose.append(self.joints[j] - self.joints[parent])
        self.ref_pose = np.expand_dims(np.stack(self.ref_pose, 0), -1)
        self.ref_T = np.expand_dims(np.stack(self.ref_T, 1), -1)

    def set_abs_quat(self, quat):
        """
        Set absolute (global) rotation for the hand.

        Parameters
        ----------
        quat : np.ndarray, shape [J, 4]
          Absolute rotations for each joint in quaternion.

        Returns
        -------
        np.ndarray, shape [V, 3]
          Mesh vertices after posing.

        np.ndarray, shape [J, 3]
          Mesh joints after posing.
        """
        mats = []
        for j in range(MANOHandJoints.n_joints):
            mats.append(quat2mat(quat[j]))
        mats = np.stack(mats, 0)

        pose = np.matmul(mats, self.ref_pose)
        joint_xyz = [None] * MANOHandJoints.n_joints
        for j in range(MANOHandJoints.n_joints):
            joint_xyz[j] = pose[j]
            parent = MANOHandJoints.parents[j]
            if parent is not None:
                joint_xyz[j] += joint_xyz[parent]
        joint_xyz = np.stack(joint_xyz, 0)[..., 0]

        T = np.matmul(np.expand_dims(mats, 0), self.ref_T)[..., 0]
        self.verts = [None] * MANOHandJoints.n_joints
        for j in range(MANOHandJoints.n_joints):
            self.verts[j] = T[:, j]
            parent = MANOHandJoints.parents[j]
            if parent is not None:
                self.verts[j] += joint_xyz[parent]
        self.verts = np.stack(self.verts, 1)
        self.verts = np.sum(self.verts * self.weights, 1)

        return self.verts.copy(), joint_xyz

    @staticmethod
    def reverse_abs_quat(verts, joints, weights, quat):
        mats = []
        # quat [w, x, y, z]
        for j in range(MANOHandJoints.n_joints):
            r_q = np.array([quat[j][1], quat[j][2], quat[j][3], quat[j][0]])  # [x,y,z,w]
            axisang = Rotation.from_quat(r_q).as_rotvec()
            angle = np.linalg.norm(axisang)
            axis = axisang / angle
            mats.append(Rotation.from_rotvec(axis * (2 * np.pi - angle)).as_dcm())
        mats = np.stack(mats, 0)

        ref_pose = []
        ref_T = []
        for j in range(MANOHandJoints.n_joints):
            parent = MANOHandJoints.parents[j]
            if parent is None:
                ref_T.append(verts)
                ref_pose.append(joints[j])
            else:
                ref_T.append(verts - joints[parent])
                ref_pose.append(joints[j] - joints[parent])
        ref_pose = np.expand_dims(np.stack(ref_pose, 0), -1)
        ref_T = np.expand_dims(np.stack(ref_T, 1), -1)

        pose = np.matmul(mats, ref_pose)
        joint_xyz = [None] * MANOHandJoints.n_joints
        for j in range(MANOHandJoints.n_joints):
            joint_xyz[j] = pose[j]
            parent = MANOHandJoints.parents[j]
            if parent is not None:
                joint_xyz[j] += joint_xyz[parent]
        joint_xyz = np.stack(joint_xyz, 0)[..., 0]

        T = np.matmul(np.expand_dims(mats, 0), ref_T)[..., 0]
        verts = [None] * MANOHandJoints.n_joints
        for j in range(MANOHandJoints.n_joints):
            verts[j] = T[:, j]
            parent = MANOHandJoints.parents[j]
            if parent is not None:
                verts[j] += joint_xyz[parent]
        verts = np.stack(verts, 1)
        verts = np.sum(verts * weights, 1)

        return verts, joint_xyz
