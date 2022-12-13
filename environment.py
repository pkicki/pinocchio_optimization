import os

import pybullet
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
import numpy as np



class Environment:
    def __init__(self, urdf_path, timestep, q0l, xyz, r):
        self._client = BulletClient(connection_mode=pybullet.GUI, options='--background_color_red=0.57 --background_color_green=0.88 --background_color_blue=0.5')
        self._client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        self._client.setTimeStep(timestep)
        self._client.setGravity(0, 0, -9.81)
        self._client.setAdditionalSearchPath(pybullet_data.getDataPath())
        base_path = "/".join(os.path.abspath(__file__).split("/")[:-1])
        print(base_path)
        self.left_arm = self._client.loadURDF(os.path.join(base_path, urdf_path),
                                              flags=pybullet.URDF_USE_IMPLICIT_CYLINDER | pybullet.URDF_USE_INERTIA_FROM_FILE,
                                              basePosition=[0., 0., 0.],
                                              useFixedBase=True)
        self._client.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=135.0, cameraPitch=-45.,
                                                cameraTargetPosition=[0., 0., 0.])
        self.left_arm_num_joints = self._client.getNumJoints(self.left_arm)
        self.joint_num = 7
        self.joint_idx = [1, 2, 3, 4, 5, 6, 7]
        for i in range(3):
            v = [0.] * 3
            v[i] = 1.
            self._client.addUserDebugLine([0., 0., 0.], v, v)

        dyn = []
        for i in range(8):
            dyn.append(self._client.getDynamicsInfo(self.left_arm, i))

        for i in range(self.joint_num):
            self._client.changeDynamics(self.left_arm, self.joint_idx[i], lateralFriction=0., linearDamping=0., angularDamping=0.)

        sphere = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=r, rgbaColor=[1.0, .0, .0, 0.4])
        idx = pybullet.createMultiBody(baseVisualShapeIndex=sphere, basePosition=xyz)
        self.reset_joints_state(q0l, np.zeros_like(q0l))

    def set_control(self, u):
        assert len(u) == self.joint_num
        for i in range(self.joint_num):
            self._client.setJointMotorControl2(self.left_arm, self.joint_idx[i], pybullet.TORQUE_CONTROL, force=u[i])

    def reset_joints_state(self, q0l, qdot0l):
        assert len(q0l) == len(qdot0l) == self.joint_num
        for i in range(self.joint_num):
            self._client.setJointMotorControl2(self.left_arm, self.joint_idx[i], pybullet.POSITION_CONTROL, force=0)
        for i in range(self.joint_num):
            self._client.resetJointState(self.left_arm, self.joint_idx[i], q0l[i], qdot0l[i])

    def acc_to_ctrl_action(self, ddq):
        #ddq = np.clip(ddq, -10., 10.)
        q, dq = self.get_state()
        tau = self._client.calculateInverseDynamics(self.left_arm, q.tolist(), dq.tolist(), ddq.tolist())
        return tau

    def simulation_step(self):
        self._client.stepSimulation()

    def get_state(self):
        q = np.zeros(self.joint_num)
        qdot = np.zeros(self.joint_num)
        for i in range(self.joint_num):
            q[i], qdot[i], _, _ = self._client.getJointState(self.left_arm, self.joint_idx[i])
        return q, qdot

    def to_pybullet(self, v):
        r = np.zeros(self.left_arm_num_joints)
        for i in range(len(self.joint_idx)):
            r[self.joint_idx[i]] = v[i]
        return r
