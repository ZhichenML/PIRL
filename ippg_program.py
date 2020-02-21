from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import logging
import sys
import datetime
import time
import functools
import copy
from bayes_opt import BayesianOptimization
from scipy import spatial
from neural_update import NeuralAgent
from controllers import Controller
from utils import *

class ParameterFinder():
    def __init__(self, inputs, actions, steer_prog, accel_prog, brake_prog):
        self.inputs = inputs
        self.actions = actions
        self.steer = steer_prog
        self.accel = accel_prog
        self.brake = brake_prog

    def find_distance_paras(self, sp0, sp1, sp2, spt, ap0, ap1, ap2, apt, api, apc, bp0, bp1, bp2, bpt):
        self.steer.update_parameters([sp0, sp1, sp2], spt)
        self.accel.update_parameters([ap0, ap1, ap2], apt, api, apc)
        self.brake.update_parameters([bp0, bp1, bp2], bpt)
        steer_acts = []
        accel_acts = []
        brake_acts = []
        for window_list in self.inputs:
            steer_acts.append(clip_to_range(self.steer.pid_execute(window_list), -1, 1))
            accel_acts.append(clip_to_range(self.accel.pid_execute(window_list), 0, 1))
            brake_acts.append(clip_to_range(self.brake.pid_execute(window_list), 0, 1))
        steer_diff = spatial.distance.euclidean(steer_acts, np.array(self.actions)[:, 0])
        accel_diff = spatial.distance.euclidean(accel_acts, np.array(self.actions)[:, 1])
        brake_diff = spatial.distance.euclidean(brake_acts, np.array(self.actions)[:, 2])
        diff_total = -(steer_diff + accel_diff + brake_diff)/float(len(self.actions))
        return diff_total

    def pid_parameters(self, info_list):

        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 10}  # Optimizer configuration
        logging.info('Optimizing Controller')
        bo_pid = BayesianOptimization(self.find_distance_paras,
                                        {'sp0': info_list[0][0], 'sp1': info_list[0][1], 'sp2': info_list[0][2], 'spt': info_list[0][3],
                                         'ap0': info_list[1][0], 'ap1': info_list[1][1], 'ap2': info_list[1][2], 'apt': info_list[1][3], 'api': info_list[1][4], 'apc': info_list[1][5],
                                         'bp0': info_list[2][0], 'bp1': info_list[2][1], 'bp2': info_list[2][2], 'bpt': info_list[2][3]}, verbose=0)

        bo_pid.maximize(init_points=20, n_iter=20, kappa=5, **gp_params)
        logging.info(bo_pid.max['params'])
        logging.info(bo_pid.max['target'])

        return bo_pid.max['params']


def programmatic_game(steer, accel, brake, track_name='practice.xml'):
    episode_count = 1
    max_steps = 10000
    window = 5

    # Generate a Torcs environment
    env = TorcsEnv(vision=False, throttle=True, gear_change=False, track_name=track_name)

    logging.info("TORCS Experiment Start with Priors on " + track_name)

    observation_list = []
    actions_list = []

    for i_episode in range(episode_count):
        ob = env.reset(relaunch=True)
        tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                   list(ob.wheelSpinVel / 100.0), list(ob.track), [0, 0, 0]]
        window_list = [tempObs[:] for _ in range(window)]

        total_reward = 0
        sp = []
        lastLapTime = []



        for j in range(max_steps):
            steer_action = clip_to_range(steer.pid_execute(window_list), -1, 1)
            accel_action = clip_to_range(accel.pid_execute(window_list), 0, 1)
            brake_action = clip_to_range(brake.pid_execute(window_list), 0, 1)
            action_prior = [steer_action, accel_action, brake_action]


            observation_list.append(window_list[:])
            actions_list.append(action_prior) #(mixed_act[:])


            tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                       list(ob.wheelSpinVel / 100.0), list(ob.track), action_prior]
            window_list.pop(0)
            window_list.append(tempObs[:])

            ob, r_t, done, info = env.step(action_prior)
            #if np.mod(j, 1000) == 0:

            total_reward += r_t
            sp.append(info['speed'])

            if lastLapTime == []:
                if info['lastLapTime']>0:
                    lastLapTime.append(info['lastLapTime'])
            elif info['lastLapTime']>0 and lastLapTime[-1] != info['lastLapTime']:
                lastLapTime.append(info['lastLapTime'])

            if done:
                print('Done. Steps: ', j)
                break

        #logging.info("Episode: " + str(i_episode) + " step: " + str(j+1) + " Distance: " + str(ob.distRaced) + ' ' + str(ob.distFromStart) + " Lap Times: " + str(ob.lastLapTime))
        logging.info(" step: " + str(j+1) + " " + str(i_episode) + "-th Episode Reward: " + str(total_reward) +
                         " Ave Reward: " + str(total_reward/(j+1)) +
                         "\n Distance: " + str(info['distRaced']) + ' ' + str(info['distFromStart']) +
                         "\n Last Lap Times: " + str(info['lastLapTime']) + " Cur Lap Times: " + str(info['curLapTime']) + " lastLaptime: " + str(lastLapTime) +
                         "\n ave sp: " + str(np.mean(sp)) + " max sp: " + str(np.max(sp)) )

        env.end()  # This is for shutting down TORCS
        logging.info("Finish.")

    return observation_list, actions_list

def test_policy(track_name, seed):

    vision = False

    env = TorcsEnv(vision=vision, throttle=True, gear_change=False, track_name=track_name)
    nn_agent = NeuralAgent(track_name=track_name)
    #Now load the weight
    logging.info("Now we load the weight")
    try:
        nn_agent.actor.model.load_weights("./model_1343/actormodel_"+str(seed)+'_'+str(900)+".h5")
        nn_agent.critic.model.load_weights("./model_1343/criticmodel_"+str(seed)+'_'+str(900)+".h5")
        nn_agent.actor.target_model.load_weights("./model_1343/actormodel_"+str(seed)+'_'+str(900)+".h5")
        nn_agent.critic.target_model.load_weights("./model_1343/criticmodel_"+str(seed)+'_'+str(900)+".h5")
        logging.info("Weight load successfully")
    except:
        logging.info("Cannot find the weight")
    nn_agent.rollout(env)
    return None

def learn_policy(track_name, test_program, seed, program_from_file):

    # Define Pi_0
    # def __init__(self, pid_constants=(0, 0, 0), pid_target=0.0, pid_sensor=0, pid_sub_sensor=0, pid_increment=0.0, para_condition=0.0, condition='False')
    #steer_prog = Controller(pid_constants=[0.97, 0.05, 49.98], pid_target=0, pid_sensor=2, pid_sub_sensor=0)
    #accel_prog = Controller(pid_constants=[3.97, 0.01, 48.79], pid_target=0.30, pid_sensor=5, pid_sub_sensor=0, pid_increment=0.0, para_condition=0.01, condition='obs[-1][2][0] > -self.para_condition and obs[-1][2][0] < self.para_condition')
    #brake_prog = Controller(pid_constants=[0, 0, 0], pid_target=0, pid_sensor=2, pid_sub_sensor=0)

    steer_prog = Controller(pid_constants=[0.8811106985926564, 0.01712974054406783, 49.947819771350105], pid_target=-0.000664433050917396, pid_sensor=2, pid_sub_sensor=0)
    accel_prog = Controller(pid_constants=[4.1120530179121175, 0.13995816823737148, 48.684311417901206], pid_target=2.168666263543196, pid_sensor=5, pid_sub_sensor=0, pid_increment=0.20990457893971481, para_condition=0.0038142532010616106, condition='obs[-1][2][0] > -self.para_condition and obs[-1][2][0] < self.para_condition')
    brake_prog = Controller(pid_constants=[-0.06600550019147762, 0.01192179945036169, -0.03761014893630601], pid_target=0.0012019017967644296, pid_sensor=2, pid_sub_sensor=0)

    if test_program == 1:
        for seeds in {1337, 1338, 1339, 1340, 1341, 1342, 1343}:
            random.seed(seeds)
            programmatic_game(steer_prog, accel_prog, brake_prog, track_name=track_name)
            print('Finish')
        return

    nn_agent = NeuralAgent(track_name=track_name)

    if program_from_file:
        logging.info("Now we load the weight")
        try:
            nn_agent.actor.model.load_weights("./model_1343/actormodel_"+str(seed)+'_'+str(900)+".h5")
            nn_agent.critic.model.load_weights("./model_1343/criticmodel_"+str(seed)+'_'+str(900)+".h5")
            nn_agent.actor.target_model.load_weights("./model_1343/actormodel_"+str(seed)+'_'+str(900)+".h5")
            nn_agent.critic.target_model.load_weights("./model_1343/criticmodel_"+str(seed)+'_'+str(900)+".h5")
            logging.info("Weight load successfully")
        except:
            logging.info("Cannot find the weight")
    else:
        # 1. train the neural network
        nn_agent.update_neural([steer_prog, accel_prog, brake_prog], episode_count=1000, tree=False, seed=seed)

    # 2. Collect data
    all_observations = []
    all_actions = []

    relabel_count = 100
    for relabel_ind in range(relabel_count):
        logging.info("\n Iteration {}".format(relabel_ind))
        for i_iter in range(1): # optimize controller parameters
            # Learn/Update Neural Policy
            #if i_iter == 0:
            #    nn_agent.update_neural([steer_prog, accel_prog, brake_prog], episode_count=2000)
            #else:
            #    nn_agent.update_neural([steer_prog, accel_prog, brake_prog], episode_count=100)

            # Collect Trajectories

            #if np.mod(i_iter, 3) == 0:
            #    relaunch=True  # relaunch TORCS every 3 episode because of the memory leak error
            #else:
            #    relaunch=False
            observation_list, action_list = nn_agent.collect_data([steer_prog, accel_prog, brake_prog])
            #print('observation_list', observation_list[0])
            #print('\n action_list', action_list[0])

            all_observations += observation_list
            all_actions += action_list
            # Relabel Observations
            #_, _, all_actions = nn_agent.label_data([steer_prog, accel_prog, brake_prog], all_observations)
            #print('\n all_actions', all_actions[0])

        # 3. Learn new programmatic policy
        logging.info("Learn programmatic policy! \n")
        #print('observations: ', np.array(all_observations)[0])
        #print('actions', np.array(all_actions).shape)
        param_finder = ParameterFinder(all_observations, all_actions, steer_prog, accel_prog, brake_prog)
        #print('observations: ', np.array(all_observations).shape())
        #print('actions', np.array(all_actions).shape())

        #steer_ranges = [[create_interval(steer_prog.pid_info()[0][const], 0.05) for const in range(3)], create_interval(steer_prog.pid_info()[1], 0.01)]
        #accel_ranges = [[create_interval(accel_prog.pid_info()[0][const], 0.05) for const in range(3)], create_interval(accel_prog.pid_info()[1], 0.5), create_interval(accel_prog.pid_info()[2], 0.1), create_interval(accel_prog.pid_info()[3], 0.01)]
        #brake_ranges = [[create_interval(brake_prog.pid_info()[0][const], 0.05) for const in range(3)], create_interval(brake_prog.pid_info()[1], 0.001)]
        steer_ranges = [create_interval(steer_prog.pid_info()[0][const], 0.05) for const in range(3)]
        steer_ranges.append(create_interval(steer_prog.pid_info()[1], 0.01))

        accel_ranges = [create_interval(accel_prog.pid_info()[0][const], 0.05) for const in range(3)]
        accel_ranges.append(create_interval(accel_prog.pid_info()[1], 0.5))
        accel_ranges.append(create_interval(accel_prog.pid_info()[2], 0.1))
        accel_ranges.append(create_interval(accel_prog.pid_info()[3], 0.01))

        brake_ranges = [create_interval(brake_prog.pid_info()[0][const], 0.05) for const in range(3)]
        brake_ranges.append(create_interval(brake_prog.pid_info()[1], 0.001))

        pid_ranges = [steer_ranges, accel_ranges, brake_ranges]
        new_paras = param_finder.pid_parameters(pid_ranges)

        steer_prog.update_parameters([new_paras[i] for i in ['sp0', 'sp1', 'sp2']], new_paras['spt'])
        accel_prog.update_parameters([new_paras[i] for i in ['ap0', 'ap1', 'ap2']], new_paras['apt'], new_paras['api'], new_paras['apc'])
        brake_prog.update_parameters([new_paras[i] for i in ['bp0', 'bp1', 'bp2']], new_paras['bpt'])

        for i_iter in range(1):
            #logging.info("\n Program Iteration {}".format(i_iter))
            program_observations, program_actions = programmatic_game(steer_prog, accel_prog, brake_prog, track_name=track_name)

            # Relabel Observations
            _, _, program_actions = nn_agent.label_data([steer_prog, accel_prog, brake_prog], program_observations)
            #print('\n all_actions', all_actions[0])

            all_observations += program_observations
            all_actions += program_actions


    logging.info("Steering Controller" + str(steer_prog.pid_info()))
    logging.info("Acceleration Controller" + str(accel_prog.pid_info()))
    logging.info("Brake Controller" + str(brake_prog.pid_info()))

    programmatic_game(steer_prog, accel_prog, brake_prog, track_name=track_name)



#[
#    [(0.9199999999999999, 1.02), (0.0, 0.1), (49.93, 50.029999999999994), (-0.01, 0.01)],
#    [(3.9200000000000004, 4.0200000000000005), (-0.04, 0.060000000000000005), (48.74, 48.839999999999996), (-0.2, 0.8), (-0.1, 0.1), (0.0, 0.02)],
#    [(-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), (-0.001, 0.001)]
# ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trackfile', default='practice.xml') # practgt2 practice
    parser.add_argument('--seed', default=1337)
    parser.add_argument('--logname', default='PIRL')
    parser.add_argument('--test_program', default=0)
    parser.add_argument('--train_indicator', default=1)
    parser.add_argument('--program_from_file', default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    logPath = 'logs'
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    logFileName = args.logname + args.trackfile[:-4] + str(args.seed) + now
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] %(module)s  %(funcName)s %(lineno)d [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(logPath, logFileName)),
            logging.StreamHandler(sys.stdout)
        ])
    logging.info("Logging started with level: INFO")
    if args.train_indicator == 1:
        learn_policy(track_name=args.trackfile, test_program=args.test_program, seed = args.seed, program_from_file=args.program_from_file)
    else:
        test_policy(track_name=args.trackfile, seed = args.seed)
