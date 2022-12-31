import numpy as np
import matplotlib.pyplot as plt
import math
import logging


logging.basicConfig(
    format='|%(asctime)s-%(name)s-%(levelname)s-%(funcName)s| %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class GenerateTraj():
    '''
    Generate Trajectory Object
    state: [x,y,theta]; v: max velocity; ω: max angular velocity; T: horzion time; t: sample time
    '''

    def __init__(self, v, ω, T, t, state=[0, 0, math.pi/2],):
        self.init_state = state
        self.max_v = v
        self.max_ω = ω
        self.horizon_t = T
        self.sample_t = t
        self.traj_arr = None

    def __call__(self, is_save=False, is_display=False):
        self.__generate_trajs()
        if is_save:
            file_name = str(self.max_v)+'-'+str(round(math.degrees(self.max_ω), 2)) + \
                '-'+str(self.horizon_t)+'-'+str(self.sample_t)
            waypoint_num = int(self.horizon_t / self.sample_t) + 1
            np.save(file_name, np.reshape(self.traj_arr, (-1, waypoint_num, 3)))
        if is_display:
            self.__display_trajs()

    def __state_shift(self, state, control):
        '''
        compute state shift with state, control and t
        state: [x,y,theta]; control: [v, ω]; t: delta time
        '''
        assert len(state) == 3
        assert len(control) == 2
        theta = state[2]
        v = control[0]
        if v == 0:
            return state
        ω = control[1]
        delta = np.array([v*math.cos(theta),
                          v*math.sin(theta),
                          ω])
        return state + delta * self.sample_t

    def __generate_trajs(self):
        '''
        generate trajectories with the params given
        '''
        x0 = self.init_state[0]
        y0 = self.init_state[1]
        theta0 = self.init_state[2]
        logger.debug("agent_state: ({},{},{})".format(x0, y0, theta0))
        v_step = 0.5
        v_arr = np.arange(0, self.max_v + v_step, v_step)
        v_num = v_arr.shape[0]
        logger.debug("v_arr:\n{}".format(v_arr))
        ω_step = math.pi / 30
        left = np.arange(-self.max_ω, 0, ω_step)
        right = np.arange(0, self.max_ω + ω_step, ω_step)
        ω_arr = np.concatenate((left, right))
        ω_num = ω_arr.shape[0]
        logger.debug("omega_arr:\n{}".format(ω_arr))
        control_mat = np.zeros((v_num, ω_num, 2))
        for i in range(v_num):
            for j in range(ω_num):
                control_mat[i, j][0] = v_arr[i]
                control_mat[i, j][1] = ω_arr[j]
        traj_point_num = int(self.horizon_t / self.sample_t) + 1
        self.traj_arr = np.zeros(
            (v_arr.shape[0], ω_arr.shape[0], traj_point_num, 3))
        self.traj_arr[:,:,0] = self.init_state
        for i in range(v_num):
            for j in range(ω_num):
                state = self.init_state
                for k in range(1, traj_point_num):
                    state = self.__state_shift(state, control_mat[i, j])
                    self.traj_arr[i, j, k] = state

    def __display_trajs(self):
        '''
        display the generated trajectories
        '''
        if self.traj_arr is None:
            logger.error("Trajectory array is empty!")
            return
        # display all traj
        for i in range(self.traj_arr.shape[0]):
            for j in range(self.traj_arr.shape[1]):
                plt.plot(self.traj_arr[i, j, :, 0], self.traj_arr[i, j, :, 1])
                plt.scatter(self.traj_arr[i, j, :, 0],
                            self.traj_arr[i, j, :, 1])
        # display same v trajs
        # v_idx = 9
        # for i in range(traj_arr.shape[1]):
        #     plt.plot(traj_arr[v_idx, i, :, 0], traj_arr[v_idx, i, :, 1])
        #     plt.scatter(traj_arr[v_idx, i, :, 0], traj_arr[v_idx, i, :, 1])
        # display same ω trajs
        # ω_idx = 4
        # for i in range(traj_arr.shape[1]):
        #     plt.plot(traj_arr[i,ω_idx,:,0],traj_arr[i,ω_idx,:,1])
        #     plt.scatter(traj_arr[i,ω_idx,:,0],traj_arr[i,ω_idx,:,1])
        plt.show()


def main():
    gt = GenerateTraj(4, math.pi/6, 4, 1)
    gt(is_save=True, is_display=True)


if __name__ == '__main__':
    main()
