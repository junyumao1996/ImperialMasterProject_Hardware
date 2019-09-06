from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import serial
import time
import matplotlib.animation as animation
from envs.serial_utilis import available_port_check, send_data
from envs.serial_utilis import receive_data_one_shoot, receive_data_waiting, received_data_process

class RealCircularPendulumEnv(object):
    def __init__(self, g=10.0, dt=0.01, episode_length=200, live_plot=False):
        self.max_speed_theta = 60         # not a restriction here, just for reference
        self.max_speed_alpha = 200
        self.max_voltage = 1              # max pwm : 6900
        self.dt = dt
        self.g = g
        self.viewer = None
        self.action_pwm_ratio = 3500.
        self.data_valid = False

        # self variables relates to pendulum-PC communication
        self.portname = available_port_check()
        self.BAUDRATE = 128000
        self.TIMEOUT = 0.002
        self.ser = serial.Serial(self.portname, self.BAUDRATE)
        print('Successful Connection!')
        print("Serial Port Info:", self.ser)
        self.received_data_valid_time = 1

        high = np.array([1, 1, self.max_speed_theta, self.max_speed_alpha])  # mind the angle conversion here
        low = np.array([-1, -1, -self.max_speed_theta, -self.max_speed_alpha])  # mind the angle conversion here
        self.action_space = spaces.Box(low=-self.max_voltage, high=self.max_voltage, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.theta_trajectory = []
        self.alpha_trajectory = []
        self.time_window = episode_length * dt
        self.state_dim = 3           # raw state dimension (not observation dimension)

        self.seed()
        if live_plot:
          plt.ion()  ## Note this correction

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u, step_counter):
        alpha, theta_dot, alpha_dot = self.state  # th := theta
        u = np.clip(u, -self.max_voltage, self.max_voltage)
        pwm = int(u * self.action_pwm_ratio)
        if pwm == 0:
            pwm = 1
        dt = self.dt
        ####### impose action ######
        # t1 = int(round(time.time() * 1000))
        self.send_action_to_hardware(pwm)
        # t2 = int(round(time.time() * 1000))
        # print("t: ", t2 - t1)

        last_state = self.state
        ####### capture next observation ######
        # t1 = int(round(time.time() * 1000))
        self.state, last_action, step_counter = self.receive_and_process_obs_from_hardware(pwm, step_counter)
        # t2 = int(round(time.time() * 1000))
        # print("t: ", t2 - t1)
        self.data_valid = (self.state_validation(self.state)) and (self.state_validation(last_state))

        ####### reward function ######
        costs = 0
        # costs = 0.1 * theta**2
        # print("1:", costs)
        costs += (alpha-np.pi)**2
        # print("2:", (alpha-np.pi)**2)
        costs += .5 * (theta_dot/10)**2 + .05 * (alpha_dot/20)**2
        # costs += .5 * (theta_dot / 10) ** 2 + 0.1 * (alpha_dot / 20) ** 2
        # print("3:", 0 * (theta_dot**2 + alpha_dot**2)
        costs += .001*(u**2)
        # print("4:", .01*(u**2))

        # if np.absolute(alpha-np.pi) < 0.3:
        #     if np.absolute(alpha_dot/20) < 1.5:
        #         costs -= 2
        #     elif np.absolute(alpha_dot/20) > 4.5:
        #         costs += 1.0
        #     else:
        #         pass

            # ap = .1 * (np.absolute(alpha - np.pi)) * np.absolute(alpha_dot / 20)
        # ap = .1 * (np.absolute(alpha-np.pi)) * np.absolute(alpha_dot/20)
        # costs += ap
        # print("5:", ap)
        costs = float(costs)


        ###### append the state trajectory ######
        # self.theta_trajectory.append(theta)
        self.alpha_trajectory.append(alpha)

        # return self.get_obs(self.state), last_action, -costs, False, {}, step_counter
        # print("state: ", self.state, "reward", -costs)

        # print("STEP:", step_counter, "ACTION:", pwm, "LAST_ACTION:", int(last_action * self.action_pwm_ratio),
        #       "STATE: [{:.3f}, {:.3f}, {:.3f}]".format(last_state[0], last_state[1], last_state[2]),
        #       "REWARD: {:7.2f}".format(-costs), "DATA VALID:", self.data_valid)

        # return self.state, last_action, -costs, False, {}, step_counter, self.data_valid
        return self.get_obs(self.state), last_action, -costs, False, {}, step_counter, self.data_valid

    def state_validation(self, s):
        if np.absolute(s[-1]) > self.max_speed_alpha or np.absolute(s[-2]) > self.max_speed_theta:
            data_valid = False
        else:
            data_valid = True
        return data_valid

    def stop_pendulum(self):
        self.send_action_to_hardware(0)

    def send_action_to_hardware(self, pwm):
        for i in range(1):
            send_data(self.ser, str(pwm))
            # time.sleep(1/1000)

    def receive_and_process_obs_from_hardware(self, executed_pwm, step_number):
        correct_data_counter = 0
        received_data_buffer = []
        correctness = False

        while correct_data_counter < self.received_data_valid_time:
            # t1 = int(round(time.time() * 1000))
            received_data = receive_data_waiting(self.ser)
            # print(received_data)
            # t2 = int(round(time.time() * 1000))
            # print("t: ", t2 - t1)

            # raw_received_data = receive_data_one_shoot(self.ser)
            # received_data = received_data_process(raw_received_data)
            # data_length = len(received_data)
            # print(received_data)
            # print(data_length)
            if received_data[-1] > step_number:
                correct_data_counter += 1
                received_data_buffer.append(np.array(received_data))
                # print('get', correct_data_counter)

        received_data_buffer = np.array(received_data_buffer)
        mode_received_data = stats.mode(received_data_buffer)[0][0]  # get mode observation

        state = np.array(mode_received_data[1:-2], dtype=float)
        action = np.array(mode_received_data[-2]/self.action_pwm_ratio, dtype=float)
        state[0] = angle_normalize_alpha(state[0])
        return state, action, received_data[-1]

    def get_obs(self, state):
        return np.array([np.cos(state[0]), np.sin(state[0]),
                         state[1]/10, state[2]/20])

    def reset(self, is_invert=False):
        self.send_action_to_hardware(0)
        time.sleep(2)
        while 1:
            state, action, _ = self.receive_and_process_obs_from_hardware(0, -2)
            # print("state: ", state)

            if np.absolute(state[0] - np.pi) > (np.pi - 0.15) and state[1] == 0 and state[2] == 0 and is_invert == False:
                break
            elif np.absolute(state[0] - np.pi) < 0.1 and state[1] == 0 and state[2] == 0 and is_invert == True:
                break
            else:
                pass
            # if np.absolute(state[1] - 3085) < 200 and state[2] == 0 and state[3] == 0:
            #     break
        self.state = state
        print("Initial state: ", self.state)

        self.last_u = None
        self.theta_trajectory = []
        self.alpha_trajectory = []

        plt.close()
        self.fig = plt.figure()
        self.ax_theta = self.fig.add_subplot(121)
        self.ax_alpha = self.fig.add_subplot(122)
        self.ax_theta.grid()
        self.ax_alpha.grid()
        return self.get_obs(self.state)

    def render(self):
        time = len(self.theta_trajectory) * self.dt
        time_length = np.arange(len(self.theta_trajectory)) * self.dt
        self.ax_theta.set_xlim([0, self.time_window])
        self.ax_alpha.set_xlim([0, self.time_window])
        # self.ax_theta.axis([0, 10, -1.5 * np.pi, 1.5 * np.pi])
        # self.ax_alpha.axis([0, 10, -2.5 * np.pi, 1.5 * np.pi])
        theta_tra = np.unwrap(self.theta_trajectory)
        alpha_tra = np.unwrap(self.alpha_trajectory)
        self.ax_theta.plot(time_length, theta_tra, 'b')
        self.ax_theta.set_xlabel("time (s)")
        self.ax_theta.set_ylabel("theta (rad)")

        self.ax_alpha.plot(time_length, alpha_tra, 'r')
        self.ax_alpha.set_xlabel("time (s)")
        self.ax_alpha.set_ylabel("alpha (rad)")
        plt.show()
        plt.pause(0.00001)

##############################################
######### changed version for MB task ########
##############################################
class RealCircularPendulumEnv2(object):
    def __init__(self, g=10.0, dt=0.01, episode_length=200, live_plot=False):
        self.max_speed_theta = 60         # not a restriction here, just for reference
        self.max_speed_alpha = 200
        self.max_voltage = 1              # max pwm : 6900
        self.dt = dt
        self.g = g
        self.viewer = None
        self.action_pwm_ratio = 3000.
        self.data_valid = False

        # self variables relates to pendulum-PC communication
        self.portname = available_port_check()
        self.BAUDRATE = 128000
        self.TIMEOUT = 0.002
        self.ser = serial.Serial(self.portname, self.BAUDRATE)
        print('Successful Connection!')
        print("Serial Port Info:", self.ser)
        self.received_data_valid_time = 1

        high = np.array([1, 1, 1, 1, self.max_speed_theta, self.max_speed_alpha])  # mind the angle conversion here
        low = np.array([-1, -1, -1, -1, -self.max_speed_theta, -self.max_speed_alpha])  # mind the angle conversion here
        self.action_space = spaces.Box(low=-self.max_voltage, high=self.max_voltage, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.theta_trajectory = []
        self.alpha_trajectory = []
        self.time_window = episode_length * dt
        self.state_dim = 4           # raw state dimension (not observation dimension)

        self.seed()
        if live_plot:
          plt.ion()  ## Note this correction

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u, step_counter):
        theta, alpha, theta_dot, alpha_dot = self.state  # th := theta
        u = np.clip(u, -self.max_voltage, self.max_voltage)
        pwm = int(u * self.action_pwm_ratio)
        if pwm == 0:
            pwm = 1

        dt = self.dt
        ####### impose action ######
        # t1 = int(round(time.time() * 1000))
        self.send_action_to_hardware(pwm)
        # t2 = int(round(time.time() * 1000))
        # print("t: ", t2 - t1)

        last_state = self.state
        ####### capture next observation ######
        # t1 = int(round(time.time() * 1000))
        self.state, last_action, step_counter = self.receive_and_process_obs_from_hardware(pwm, step_counter)
        # t2 = int(round(time.time() * 1000))
        # print("t: ", t2 - t1)
        self.data_valid = (self.state_validation(self.state)) and (self.state_validation(last_state))

        ####### reward function ######
        costs = 0
        costs = 0.1 * theta**2
        # print("1:", costs)
        costs += (alpha-np.pi)**2
        # print("2:", (alpha-np.pi)**2)
        costs += .5 * (theta_dot/10)**2 + .05 * (alpha_dot/20)**2
        # print("3:", 0 * (theta_dot**2 + alpha_dot**2)
        costs += .001*(u**2)
        # print("4:", .01*(u**2))
        if (alpha-np.pi) < 0.1 and ((alpha_dot/20) < 0.1):
            costs -= 1.0
        costs = float(costs)


        ###### append the state trajectory ######
        self.theta_trajectory.append(theta)
        self.alpha_trajectory.append(alpha)

        # return self.get_obs(self.state), last_action, -costs, False, {}, step_counter
        # print("state: ", self.state, "reward", -costs)
        print("STEP:", step_counter, "ACTION:", pwm, "LAST_ACTION:", int(last_action * self.action_pwm_ratio),
              "STATE: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]".format(last_state[0], last_state[1], last_state[2], last_state[3]),
              "REWARD: {:7.2f}".format(-costs), "DATA VALID:", self.data_valid)

        return self.state, last_action, -costs, False, {}, step_counter, self.data_valid
        # return self.get_obs(self.state), last_action, -costs, False, {}, step_counter, self.data_valid

    def state_validation(self, s):
        if np.absolute(s[-1]) > self.max_speed_alpha or np.absolute(s[-2]) > self.max_speed_theta:
            data_valid = False
        else:
            data_valid = True
        return data_valid

    def stop_pendulum(self):
        self.send_action_to_hardware(0)

    def send_action_to_hardware(self, pwm):
        for i in range(1):
            send_data(self.ser, str(pwm))
            # time.sleep(1/1000)

    def receive_and_process_obs_from_hardware(self, executed_pwm, step_number):
        correct_data_counter = 0
        received_data_buffer = []
        correctness = False

        while correct_data_counter < self.received_data_valid_time:
            # t1 = int(round(time.time() * 1000))
            received_data = receive_data_waiting(self.ser)
            # print(received_data)
            # t2 = int(round(time.time() * 1000))
            # print("t: ", t2 - t1)

            # raw_received_data = receive_data_one_shoot(self.ser)
            # received_data = received_data_process(raw_received_data)
            # data_length = len(received_data)
            # print(received_data)
            # print(data_length)
            if received_data[-1] > step_number:
                correct_data_counter += 1
                received_data_buffer.append(np.array(received_data))
                # print('get', correct_data_counter)

        received_data_buffer = np.array(received_data_buffer)
        mode_received_data = stats.mode(received_data_buffer)[0][0]  # get mode observation

        state = np.array(mode_received_data[0:-2], dtype=float)
        action = np.array(mode_received_data[-2]/self.action_pwm_ratio, dtype=float)
        # state[0] = angle_normalize_theta(state[0])
        # state[1] = angle_normalize_alpha(state[1])
        return state, action, received_data[-1]

    def get_obs(self, state):
        return np.array([np.cos(state[0]), np.sin(state[0]), np.cos(state[1]), np.sin(state[1]),
                         state[2]/10, state[3]/20])

    def reset(self, is_invert=False):
        self.send_action_to_hardware(0)
        time.sleep(2)
        while 1:
            state, action, _ = self.receive_and_process_obs_from_hardware(0, -2)
            # print("state: ", state)

            # if np.absolute(state[1] - np.pi) > (np.pi - 0.15) and state[2] == 0 and state[3] == 0 and is_invert == False:
            #     break
            # elif np.absolute(state[1] - np.pi) < 0.1 and state[2] == 0 and state[3] == 0 and is_invert == True:
            #     break
            # else:
            #     pass
            if np.absolute(state[1] - 3085) < 200 and state[2] == 0 and state[3] == 0:
                break
        self.state = state
        print("Initial state: ", self.state)

        self.last_u = None
        self.theta_trajectory = []
        self.alpha_trajectory = []

        plt.close()
        self.fig = plt.figure()
        self.ax_theta = self.fig.add_subplot(121)
        self.ax_alpha = self.fig.add_subplot(122)
        self.ax_theta.grid()
        self.ax_alpha.grid()
        return self.get_obs(self.state)

    def render(self):
        time = len(self.theta_trajectory) * self.dt
        time_length = np.arange(len(self.theta_trajectory)) * self.dt
        self.ax_theta.set_xlim([0, self.time_window])
        self.ax_alpha.set_xlim([0, self.time_window])
        # self.ax_theta.axis([0, 10, -1.5 * np.pi, 1.5 * np.pi])
        # self.ax_alpha.axis([0, 10, -2.5 * np.pi, 1.5 * np.pi])
        theta_tra = np.unwrap(self.theta_trajectory)
        alpha_tra = np.unwrap(self.alpha_trajectory)
        self.ax_theta.plot(time_length, theta_tra, 'b')
        self.ax_theta.set_xlabel("time (s)")
        self.ax_theta.set_ylabel("theta (rad)")

        self.ax_alpha.plot(time_length, alpha_tra, 'r')
        self.ax_alpha.set_xlabel("time (s)")
        self.ax_alpha.set_ylabel("alpha (rad)")
        plt.show()
        plt.pause(0.00001)



def angle_normalize_theta(theta):
    return ((theta - 10000 + 1040/2) % 1040 - 1040/2) * (2*np.pi/1040)

def angle_normalize_alpha(alpha):
    alpha_rad = (alpha - 3085)*(2*np.pi/4100) + np.pi
    return (alpha_rad % (2*np.pi))

def state_normalize(state):
    s = state
    s[0] = angle_normalize_theta(s[0])
    s[1] = angle_normalize_alpha(s[1])
    return s