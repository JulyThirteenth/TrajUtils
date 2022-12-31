import numpy as np
import matplotlib.pyplot as plt
import math
import logging
from tqdm import tqdm

logging.basicConfig(
    format='|%(asctime)s-%(name)s-%(levelname)s-%(funcName)s| %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class EvaluateTraj():
    '''
    Evaluate traj templates with traj dataset given
    '''
    def __init__(self, traj_templates: np.array):
        self.traj_templates = traj_templates

    def __call__(self, input: np.array):
        if input.shape[1:] == self.traj_templates.shape[1:]:
            self.__evaluate_with_trajs(input)
        elif input.shape == self.traj_templates.shape[1:]:
            self.__evaluate_with_traj(input)
        else:
            logger.error("Input of EvaluateTraj Object has wrong dimension!")

    @staticmethod
    def draw_traj(traj: np.array, arrow_length=1):
        '''
        display a traj
        '''
        ax = plt.axes()
        for x, y, theta in traj:
            dx = arrow_length*math.cos(theta)
            dy = arrow_length*math.sin(theta)
            ax.arrow(x, y, dx, dy,
                     head_width=0.1,
                     head_length=0.2)
        plt.xlim(min(traj[:, 0])-5, max(traj[:, 0]+5))
        plt.ylim(min(traj[:, 1])-5, max(traj[:, 1]+5))
        plt.gca().set_aspect(1)
        plt.show()

    def __to_origin(self, traj: np.array):
        '''
        transform traj to start == [0,0,math.pi/2]
        '''
        logger.debug('before transform:\n{}'.format(traj))
        x0 = traj[0, 0]
        y0 = traj[0, 1]
        t0 = traj[0, 2]
        traj[:, 0:2] = traj[:, 0:2]-[x0, y0]
        dt = math.pi/2 - t0
        trans_matrix = [[math.cos(dt), -math.sin(dt)],
                        [math.sin(dt), math.cos(dt)]]
        traj[:, 0:2] = np.dot(trans_matrix, traj[:, 0:2].T).T
        traj[:, 2] = np.mod(traj[:, 2] + dt, 2*math.pi)
        logger.debug('after transform:\n{}'.format(traj))
        return traj

    def __calc_minADE(self, traj: np.array):
        traj = self.__to_origin(traj)
        diff = self.traj_templates - traj
        dist = np.sqrt(diff[:,:,0]**2 + diff[:,:,1]**2)
        ades = np.sum(dist, axis=1)/dist.shape[1]
        min_idx, min_ade = np.argmin(ades), np.min(ades)
        logger.debug('minADE:{}'.format(min_ade))    
        return min_idx, min_ade

    def __evaluate_with_trajs(self, trajs: np.array):
        miniADEs = []
        for idx in tqdm(range(trajs.shape[0]), desc='Evaluating'):
            miniADEs.append(self.__calc_minADE(trajs[idx])[1])
        miniADEs = np.array(miniADEs)
        minimum = min(miniADEs)
        maximum = max(miniADEs)
        logger.info("maximum minADE:{:.3f}m, minimum minADE:{:.3f}m".format(maximum, minimum))
        minimum = round(minimum)
        maximum = round(maximum)
        x = np.arange(minimum, maximum, 0.1)
        y = np.zeros_like(x)
        for idx, value in enumerate(x):
            y[idx] = np.sum(np.where((miniADEs >= value) &
                            (miniADEs < value+0.1), 1, 0))/trajs.shape[0]
        plt.plot(x, y)
        plt.xlabel('minADE')
        plt.ylabel('Proportion')
        plt.show()

    def __evaluate_with_traj(self, traj: np.array, is_display=True):
        traj = self.__to_origin(traj)
        min_idx, min_ade = self.__calc_minADE(traj)
        logger.info("min_idx:{}, min_ade:{:.3f}".format(min_idx, min_ade))
        def draw_fitness(template_traj, traj):
            fig = plt.figure()
            ax = fig.add_subplot()
            for x, y, theta in template_traj:
                dx = 1*math.cos(theta)
                dy = 1*math.sin(theta)
                ax.arrow(x, y, dx, dy,
                            head_width=0.1,
                            head_length=0.2,
                            fc ='r', ec ='r')
            for x, y, theta in traj:
                dx = 1*math.cos(theta)
                dy = 1*math.sin(theta)
                ax.arrow(x, y, dx, dy,
                            head_width=0.1,
                            head_length=0.2,
                            fc ='g', ec ='g')
            plt.xlim(min(traj[:, 0])-5, max(traj[:, 0]+5))
            plt.ylim(min(traj[:, 1])-5, max(traj[:, 1]+5))
            fig.gca().set_aspect(1)
            plt.show()
        if is_display:
            draw_fitness(self.traj_templates[min_idx], traj)

def main():
    traj_templates = np.load('./traj_templates/4-30.0-4-1.npy')
    waypoints = np.reshape(np.load('./traj_datasets/lbc_waypoints.npy'), (-1, 5, 3))
    et = EvaluateTraj(traj_templates)
    et(waypoints)
    


if __name__ == '__main__':
    main()
