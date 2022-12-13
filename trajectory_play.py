from time import sleep

import numpy as np
from scipy.interpolate import interp1d
import pinocchio as pino

from environment import Environment

Tp = 1. / 500
end_time = 3.
urdf_path = "iiwa_cup.urdf"

pino_model = pino.buildModelFromUrdf(urdf_path)
pino_data = pino_model.createData()

q = np.array([[0., 0., 0., 0., 0., 0., 0.],
              [1., 1., 1., 1., 1., 1., 1.],
              [0., 0., 0., 0., 0., 0., 0.],
              ])
t = np.linspace(0., end_time, q.shape[0])

qi = interp1d(t, q, axis=0)

env = Environment(urdf_path, Tp, q[0], [0.5, 0.5, 0.5], 0.15)

for i in range(int(end_time/Tp)):
    t_act = i * Tp

    q_act, dq_act = env.get_state()

    q_i = qi(t_act)
    env.reset_joints_state(q_i, np.zeros_like(q_i))
    env.simulation_step()
    sleep(Tp)