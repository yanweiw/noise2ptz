import os
import sys
import time
import pickle
import random
import numpy as np
import numpy.linalg as la 
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
import math
sys.path.append('gen_data')
from settings import default_sim_settings, make_cfg
from settings import make_cfg
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.markers import MarkerStyle
from matplotlib import gridspec
from torch.nn.functional import softmax as softmax
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
import seaborn as sns
import scipy
from PIL import Image
import alphashape
from descartes import PolygonPatch
sys.path.append('../PerceptualSimilarity/')
import models

sys.path.append('../data_aug_pred')
from aug_data import *
import train_nav as tn

class NavInfer:

    default_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                  std=[0.5, 0.5, 0.5])])    

    def __init__(self, base_dir, infer_h=128, sample_deviation=360):

        self.sim = None 
        self._sim_settings = None
        self.action_list = ['stop', 'move_forward', 'turn_left', 'turn_right']
        self.perceptual_model = None # for goal recognizing
        # self.policy = {'random': None} # navigation policy, dict to store multiple policies
        self.policy = {}
        self.thumbnail = True # if scene observation is 120 x 160 (thumbnail) or 480 x 640
        self.all_states = None # states ([x, y, z, r] sensor posistion & rotation) visited in all data collected
        self.floor = None # floor map approximated by concave hull of all the states
        self.curr_state = None # [x, y, z, r] of sensor
        self.curr_obs = None # np array shape of (120, 160, 4) 0 ~ 255, required shape for inference input
        self.goal_state = None
        self.goal_obs = None
        self.hidden = None # hidden states for LSTM
        self.prev_act_idx = None # previous action index
        self.save_dir = None # dir to log trajectory info
        self.base_dir = base_dir
        # all save_dir stored inside base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        assert infer_h in [120, 480, 128]
        if infer_h == 120:
            self.infer_h = 120
            self.infer_w = 160
        elif infer_h == 128:
            self.infer_h = 128
            self.infer_w = 128
        else:
            self.infer_h = 256
            self.infer_w = 256

        self.use_goal_recognizer = True
        self.distance_threshold = 0.5
        self.sample_deviation = sample_deviation

        
    def load_perceptual_model(self, confidence_threshold=0.6):
        # load perceptual model
        if torch.cuda.is_available(): # assuming 2 gpus, set gpu_id yourself otherwise
            self.perceptual_model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[2])
        else:
            self.perceptual_model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=False)
        self.perceptual_model.eval()
        self.confidence_threshold = confidence_threshold


    def load_sim(self, scene, mode='nav', thumbnail=False):
        '''
        make a simulator with a particular scene (.glb file)
        mode: default 'nav', meaning data collection by robot moving around, as opposed to 'ptz' which rotates 360 deg
        '''
        if self.sim is not None:
            self.sim.close()
        self.thumbnail = thumbnail

        # make configuration file for the 
        settings = default_sim_settings.copy()
        settings["max_frames"] = 100
        settings['mode'] = mode
        if thumbnail:
            settings["width"] = 128#160
            settings["height"] = 128#120
        else:
            settings["width"] = 256#640
            settings["height"] = 256#480
        settings["scene"] = scene
        settings["save_png"] = False
        settings["sensor_height"] = 1.5
        settings["color_sensor"] = True
        settings["semantic_sensor"] = False
        settings["depth_sensor"] = False
        settings["print_semantic_scene"] = False
        settings["print_semantic_mask_stats"] = False
        settings["compute_shortest_path"] = False
        settings["compute_action_shortest_path"] = False
        settings["seed"] = 800
        settings["silent"] = False
        settings["enable_physics"] = False
        settings["physics_config_file"] = default_sim_settings['physics_config_file']

        cfg = make_cfg(settings.copy())
        self.sim = habitat_sim.Simulator(config=cfg)
        self._sim_settings = settings
        print('\n\nmaking scene of: ', scene)
        print('mode: ', mode, ' thumbnail: ', thumbnail)
        print('action space: ', list(cfg.agents[0].action_space.keys()), '\n')


    def load_policy(self, tag, weight_path=None, state_size=128, with_lstm=False, img_h=128, img_w=128, 
                    use_PTZ=None):
        '''
        load navigation policy, which is a pretrained model from train.py
        original model is trained with data parallelization
        we modify it to be run on cpu
        tag is the key to access specific policy in the policy dict
        '''
        print('\nLoading policy', tag, ' with weight_path: ', weight_path, '\n')

        if tag == 'random':
            self.policy[tag] = None
        else: 
            policy = tn.Siamese(max_seq_len=1, state_size=state_size, PTZ_weights=use_PTZ,
                        with_lstm=with_lstm, img_h=img_h, img_w=img_w).float().cuda()
            # original saved file with DataParallel
            state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
            # create new OrderedDict that does not contain `module.`
            if 'module' in list(state_dict.keys())[0]:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                state_dict = new_state_dict        
            policy.load_state_dict(state_dict)
            _ = policy.eval() 

            self.policy[tag] = policy


    def load_all_states(self, data_source, show_img=False):
        '''
        load states from all trajectories collected
        e.g. load_all_states('data/test/hometown')
        be sure to check the scatter plot of all states to rid off spurious dispersive pattern, 
        which is typically caused by agent landing not on the floor, e.g. on bed
        can be prevented by height checking during agent initialization in random_explore.py
        '''
        states = []
        for folder in os.listdir(data_source):
            if 'cam' in folder:
                continue
            folder = os.path.join(data_source, folder)
            for d in sorted(os.listdir(folder)):
                if '_state.txt' in d:
                    states.append(np.loadtxt(os.path.join(folder, d)))
        states = np.vstack(states)
        if show_img:
            plt.figure()
            plt.scatter(states[:, 0], states[:, 2])

        self.all_states = states


    def load_floor(self, load_floor_path=None, save_floor_path=None, show_img=False):
        '''
        load floor map from running a concave hull algorithm on states
        '''
        if self.all_states is None:
            raise ValueError('Need to load states first!')

        if show_img:
            fig, ax = plt.subplots()
            ax.set_aspect('equal', adjustable='box')
            ax.scatter(self.all_states[:, 0], self.all_states[:, 2])
            plt.pause(0.01) # display scatter plot before calculating concave hull

        if load_floor_path is None:
            since = time.time()
            alpha_shape = alphashape.alphashape(self.all_states[:, [0, 2]], 3)
            alpha_time = time.time() - since
            print('Concave Hull compute time: {:.0f}m {:.0f}s\n'.format(alpha_time // 60, alpha_time % 60))
        else:
            with open(load_floor_path, 'rb') as f:
                alpha_shape = pickle.load(f)
        if show_img:
            ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))

        if save_floor_path is not None:
            with open(save_floor_path, 'wb') as f:
                pickle.dump(alpha_shape, f)

        self.floor = alpha_shape


    def load_agent(self, state):
        '''
        load agent to a particular sensor state, [x, y, z, r]
        '''
        state = np.array(state.copy()) # avoid in-place changes as we transform sensor state to agent state
        agent_state = habitat_sim.AgentState()
        pos_list = state[:3]
        rot_scalar = state[3]
        pos_list[1] = pos_list[1] - self._sim_settings['sensor_height']
        agent_state.position = pos_list
        agent_state.rotation = quat_from_angle_axis(math.radians(rot_scalar), np.array([0, 1.0, 0]))
        # init for new inference process
        self.sim.initialize_agent(0, agent_state)
        self.hidden = (torch.zeros(1, 1, 512), torch.zeros(1, 1, 512)) # init hidden states for LSTM
        self.prev_act_idx = torch.zeros(1).long() # init stop action
        curr_obs = self.get_obs()
        curr_state = self.get_state()
        return curr_obs, curr_state


    def get_obs(self):
        '''
        get observation of shape (height, width, 4)
        '''
        obs = self.sim.get_sensor_observations()['color_sensor']
        self.curr_obs = self.resize_obs(obs)
        return obs 


    def resize_obs(self, obs):
        '''
        resize np array (h, w, 4) to target shape
        '''
        if obs.shape != (self.infer_h, self.infer_w, 4):
            target_obs = Image.fromarray(obs)
            target_obs = target_obs.resize((self.infer_w, self.infer_h))
            target_obs = np.asarray(target_obs, dtype=np.uint8)
        else:
            target_obs = obs.copy()
        return target_obs


    def get_state(self):
        '''
        get sensor state
        '''
        pos = self.sim.agents[0].state.sensor_states['color_sensor'].position
        rot = self.sim.agents[0].state.sensor_states['color_sensor'].rotation
        angle_along_y_axis = np.arctan2(rot.y, rot.w) * 2 / np.pi * 180 % 360 # convert from quaternion
        self.curr_state = np.append(pos, angle_along_y_axis)
        return self.curr_state


    def infer_traj(self, save_dir, start_state, goal_image_path=None, goal_state=None, 
                                                max_steps=40, save_img=True, policy_tag='random'):
        '''
        infer trajectory from current observation to goal image
        '''
        if goal_image_path is None and goal_state is None:
            raise ValueError('goal_image_path and goal_state cannot both be None')
        self.save_dir = os.path.join(self.base_dir, save_dir)
        
        if goal_state is None:
            goal_state = [0, 0, 0, 0]
            goal_image = np.asarray(Image.open(goal_image_path))
        else:
            goal_image, goal_state = self.load_agent(goal_state)

        self.log_obs_state(0, goal_image, goal_state, save_img=save_img)
        self.goal_obs = self.resize_obs(goal_image)
        self.goal_state = goal_state
        start_obs, start_state = self.load_agent(start_state)
        self.log_obs_state(1, start_obs, start_state, save_img=save_img)

        step_count = 0 
        reach_goal_count = 0

        # start inference 
        while True:
            reached = self.reach_goal()
            if reached:
                reach_goal_count = 1
                break

            if step_count >= max_steps:
                break

            # one step action prediction
            action = self.pred_action(policy_tag)
            curr_obs, curr_state = self.step(action)
            self.log_obs_state(step_count+2, curr_obs, curr_state, save_img=save_img) # +2 cause first 2 idx are taken

            step_count += 1
            if self.prev_act_idx == 0: # stop action is predicted
                _ = self.reach_goal()
                break 

        return reach_goal_count, step_count


    def np2tensor(self, img):
        '''
        convert numpy image [h, w, c] to tensor equivalent [c, h, w] plus default_transform
        '''
        return torch.unsqueeze(self.default_transform(img[:, :, :3].copy()), 0)


    def tensor2np(self, img):
        img = img.detach().numpy()
        img = img / 2 + 0.5
        img = np.transpose(img, (1, 2, 0))
        return img 


    def pred_action(self, tag):
        '''
        recursively present one-step action using a inv-LSTM policy
        '''
        if self.policy[tag] is None: # random policy 
            act_idx = torch.tensor(random.choice([1, 2, 3]))
        else:
            curr = self.np2tensor(self.curr_obs)
            goal = self.np2tensor(self.goal_obs)
            # curr = torch.unsqueeze(NavInfer.default_transform(self.curr_obs[:, :, :3].copy()), 0)
            # goal = torch.unsqueeze(NavInfer.default_transform(self.goal_obs[:, :, :3].copy()), 0)
            # imgs = torch.stack([curr, goal], dim=1)#.cuda()
            # assert imgs.size() == (1, 2, 3, 120, 160)

            curr = curr.unsqueeze(0)
            goal = goal.unsqueeze(0)
            imgs = torch.cat((curr, goal), dim=2)
            imgs2 = torch.cat((goal, goal), dim=2)
            imgs = torch.cat((imgs, imgs2), dim=1).cuda()
            assert imgs.size() == (1, 2, 6, self.infer_h, self.infer_w)

            prev_act_idx = torch.unsqueeze(self.prev_act_idx.clone().detach(), 0).cuda()
            if self.policy[tag].with_lstm: # need recurrency, verified that recurrency does not affect lstm_1
                hidden = (self.hidden[0].cuda(), self.hidden[1].cuda())
                # hidden = self.hidden
            else:
                # print('single action initing hidden for lstm')
                hidden = (torch.zeros(1, 1, 512).cuda(), torch.zeros(1, 1, 512).cuda()) # init hidden states for LSTM

            with torch.no_grad():
                inv_prob, hidden = self.policy[tag](imgs, act_lens=torch.ones(1).int(), hidden=hidden, phase='infer', prev_action=prev_act_idx)

            h, c = hidden
            h = torch.transpose(h, 0, 1).contiguous()
            c = torch.transpose(c, 0, 1).contiguous()
            self.hidden = (h, c)
            if self.policy[tag].with_lstm:
                act_idx = torch.argmax(inv_prob, dim=2)[0][0]
            else:
                act_idx = torch.argmax(inv_prob, dim=1)[0]

            # inv_prob = torch.nn.Softmax(dim=1)(inv_prob)
            # print(inv_prob)
            # if random.random() < 0.2:
            #     act_idx = torch.distributions.categorical.Categorical(probs=inv_prob).sample()

        action = self.action_list[act_idx]
        with open(self.save_dir + '_action.txt', 'a') as file:
            file.write('%s\n' % action)        
        self.prev_act_idx = act_idx.clone().detach().unsqueeze(0)

        return action 


    def step(self, action):
        '''
        forward simulation by one step
        '''
        assert action in self.action_list
        _ = self.sim.step(action)
        curr_obs = self.get_obs() # necessary to call this function to update self.curr_obs
        curr_state = self.get_state() # necessary for similar reasons        

        return curr_obs, curr_state 


    def reach_goal(self):
        '''
        check if the agent has reached goal by comparing current and goal observation
        '''
        if not self.use_goal_recognizer:
            return False # never reaching goal

        curr = self.np2tensor(self.curr_obs)
        goal = self.np2tensor(self.goal_obs)
        # curr = torch.unsqueeze(NavInfer.default_transform(self.curr_obs[:, :, :3].copy()), 0)
        # goal = torch.unsqueeze(NavInfer.default_transform(self.goal_obs[:, :, :3].copy()), 0)
        with torch.no_grad():
            logits = self.perceptual_model(goal, curr)
            logits = logits.cpu().detach().flatten().numpy()[0]
            confidence = 1 - logits # logtis here denotes perceptual loss
            dist = math.hypot(self.goal_state[0] - self.curr_state[0], self.goal_state[2] - self.curr_state[2])
            if dist < self.distance_threshold and confidence > self.confidence_threshold:
                reached = True
            else:
                reached = False

        with open(self.save_dir + '_conf.txt', 'a') as file:
            file.write('%3f\n' % confidence)

        return reached


    def log_obs_state(self, obs_idx, obs, state, save_img=True):
        '''
        save image and state info
        '''
        if save_img:
            img_dir = self.save_dir + '_orig'
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            obs = Image.fromarray(obs)
            obs.save(os.path.join(img_dir, '%06d.png' % (obs_idx)))

        with open(self.save_dir + '_state.txt', 'a') as file:
            file.write('%.2f %.2f %.2f %.0f\n' % (state[0], state[1], state[2], state[3]))


    def plot_floor_states(self, save_path=None, plot_axis=False, ax=None):
        '''
        plot floor plan with states
        floor plan is imperfect due to imperfection of concave hull algorithm
        plotting states helps clarify the accessible region
        '''
        if self.all_states is None or self.floor is None:
            raise ValueError('states or floor map is None!')

        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 15))
        if not plot_axis:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

        ax.scatter(self.all_states[:, 0], self.all_states[:, 2], c='k', alpha=0.01)
        plt.gca().set_aspect('equal', adjustable='box')
        ax.add_patch(PolygonPatch(self.floor, fill=False))
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')

        return ax


    def plot_single_traj(self, traj_dir, start=None, steps=None, ax=None, inferred=False, plot_axis=False):
        '''
        plotting a single state trajectory, no need of images
        traj_dir e.g. 'data/test/hometown/rob0/000'
        purple marker denotes start, red marker denotes end
        inferred=True means the first image in the sequence is a goal image
        '''
        states, images = self.load_single_traj(traj_dir, start=start, steps=steps, inferred=inferred)
        colors = cm.rainbow(np.linspace(0, 1, len(states)))        
        # plot floor
        if ax is None:
            ax = self.plot_floor_states(plot_axis=plot_axis)
        # plot goal marker
        if inferred:
            self.plot_marker(states[-1], colors[-1], ax, alpha=1)
            states = states[:-1]
            colors = colors[:-1]
        # plot the rest of markers
        for state, color in zip(states, colors):
            self.plot_marker(state, color, ax)
        # plot trajecotry
        ax.plot(states[:,0], states[:,2], color='g')

        return ax


    def load_single_traj(self, traj_dir, start=None, steps=None, inferred=False, load_img=False, thumbnail=False):
        '''
        loading states of a single trajectory
        inferred=True means the first image in the sequence is a goal image
        this function always returns a sequence where the first state/image is the start state/image
        '''
        state_file_path = traj_dir + '_state.txt'
        traj_states = np.loadtxt(state_file_path)
        if start is None:
            start = 0 # starting state index
        else:
            assert start < len(traj_states)
        if steps is None:
            steps = len(traj_states)
            if start + steps > len(traj_states):
                steps = len(traj_states) - start

        traj_states = traj_states[start:start+steps, :]
        if inferred:
            traj_states = np.roll(traj_states, -1, axis=0)

        traj_images = None
        if load_img:
            image_dir = traj_dir + '_orig'
            traj_images = np.zeros((steps, 256, 256, 3))
            if thumbnail:
                traj_images = np.zeros((steps, 128, 128, 3))
            for step in range(steps):
                img_name = os.path.join(image_dir, str(start+step).zfill(6) + '.png')
                if thumbnail:
                    img = np.asarray(Image.open(img_name).resize((128, 128)).convert('RGB'), dtype=np.uint8)
                else:
                    img = np.asarray(Image.open(img_name).resize((256, 256)).convert('RGB'), dtype=np.uint8)
                traj_images[step] = img / 255
            if inferred:
                traj_images = np.roll(traj_images, -1, axis=0)

        return traj_states, traj_images


    def plot_marker(self, state, color, ax, alpha=0.5):
        # make triangular marker to indicate agent state
        m = MarkerStyle('^')
        m._transform.scale(2, 5) # squash equilateral triangle to indicate orientation
        m._transform.rotate_deg(180-state[3])
        ax.scatter(state[0], state[2], color=color, alpha=alpha, marker=m,)


    def setup_obs_window(self, ax):
        # set up observation window for displaying images in replay
        plt.setp(ax.get_xticklabels(), visible=False) # make ticks go away
        plt.setp(ax.get_yticklabels(), visible=False) 
        ax.tick_params(axis='both', which='both', length=0)
        obs_display = ax.imshow(np.zeros((480, 640))) # placeholder for observation display
        return obs_display


    def plot_update(self, state, color, obs, marker_ax, marker_alpha, display_ax, display):
        # update marker (agent state) and display (agent observation)
        display.set_data(obs)
        plt.setp(display_ax.spines.values(), color=color, linewidth=10, alpha=1) # draw edge: edgecolor='k', linewidths=2
        self.plot_marker(state, color, marker_ax, alpha=marker_alpha)


    def replay(self, traj_dir, start=None, steps=None, inferred=True):
        '''
        replay image sequence of a single trajectory
        in addition to single state trajectory, images are shown too
        e.g. replay('hometown/test/rob0/000')
        if it's inferred sequence, then goal_first is True
        goal_first means the first image in the folder is goal image
        '''
        # set up axes to plot on 
        fig = plt.figure(figsize=(30,15))
        fig.tight_layout()
        ax1 = fig.add_subplot(1,3,(1,2)) # floor map occupies 2/3 of the plot
        ax2 = fig.add_subplot(233)       # curr observation
        ax3 = fig.add_subplot(236)       # goal observation
        ax1.set_aspect('equal',)
        ax2.set_aspect('equal',)
        ax3.set_aspect('equal',)
        # plotting floor
        ax1 = self.plot_floor_states(plot_axis=True, ax=ax1)
        # set up observation display
        curr_obs = self.setup_obs_window(ax2)
        goal_obs = self.setup_obs_window(ax3)

        # loading states and images (goal assumed to be last in the outputs)
        states, images = self.load_single_traj(traj_dir, start, steps, inferred=inferred, load_img=True)        

        # load marker colors, and confidence, these do not need to check goal_first
        colors = cm.rainbow(np.linspace(0, 1, len(states)))
        if inferred: # only inferred sequence will have a confidence log
            conf = np.loadtxt(traj_dir + '_conf.txt')
            assert len(colors) == len(conf) + 1

        # plot goal state and image
        self.plot_update(states[-1], colors[-1], images[-1], marker_ax=ax1, marker_alpha=1, display_ax=ax3, display=goal_obs) 
        ax3.set_title('GOAL IMAGE    POS: (%.2f, %0.2f, %0.2f)  ROT: %.0f\n' 
            % (states[-1, 0], states[-1, 1], states[-1, 2], states[-1, 3]), fontsize=15, fontweight='bold')

        # plot rest of the states and images one pair at a time
        for step, (state, color, obs) in enumerate(zip(states[:-1], colors[:-1], images[:-1])):
            self.plot_update(state, color, obs, marker_ax=ax1, marker_alpha=0.5, display_ax=ax2, display=curr_obs) 
            ax1.plot(states[:step+1, 0], states[:step+1, 2], color='g') # plot trajectory connecting markers

            if inferred:
                ax2.set_title('STEP %d  CONF: %0.2f    POS: (%.2f, %0.2f, %0.2f)  ROT: %.0f\n' 
                    % (step, conf[step], state[0], state[1], state[2], state[3]), fontsize=15, fontweight='bold')
            else:
                ax2.set_title('STEP %d    POS: (%.2f, %0.2f, %0.2f)  ROT: %.0f\n' 
                    % (step, state[0], state[1], state[2], state[3]), fontsize=15, fontweight='bold')

            plt.pause(0.3)


    def sample_goal_cone(self, x_min=None, x_max=None, y_min=None, y_max=None, dist_min=2, dist_max=3):
        '''
        sample start and goal locations x meters apart, where dist_min < x < dist_max
        start is in a cone region behind the goal
        return start_state, goal_state
        '''
        # Hambleton x ~ (-8, -2), y ~ (0, 2.1)
        while True:
            # x1 = random.uniform(x_min, x_max)
            # y1 = random.uniform(y_min, y_max)
            # x2 = random.uniform(x_min, x_max)
            # y2 = random.uniform(y_min, y_max)
            x1, z1, y1 = self.sim.pathfinder.get_random_navigable_point()
            x2, z2, y2 = self.sim.pathfinder.get_random_navigable_point()
            o1 = int(random.uniform(0, 360))
            o2 = int(random.uniform(0, 360))        
            dist  = math.hypot(x2 - x1, y2 - y1)
            ang2goal = np.arctan2(y2-y1, x2-x1) / np.pi * 180 % 360 
            ang2goal = 270 - ang2goal # convert to env coordinate system(orientation 0-360 clockwise, 0 pointing downwards)
            ang_diff = abs(o2 - ang2goal)
            if dist > dist_min and dist < dist_max and min(ang_diff, 360-ang_diff) < self.sample_deviation:
                break
        # ax = nav.plot_floor_states(plot_axis=True)
        # nav.plot_marker([x1, 1.68, y1, o1], 'g', ax)
        # nav.plot_marker([x2, 1.68, y2, o2], 'r', ax)
        return [x1, z1+1.5, y1, o1], [x2, z2+1.5, y2, o2]


    def sample_start_cone(self, x_min, x_max, y_min, y_max, steps=7):
        '''
        goal is in a cone region in front of the start
        testing the ability of the agent to move forward
        return start state and goal state
        '''
        while True:
            x2 = random.uniform(x_min, x_max)
            y2 = random.uniform(y_min, y_max)
            o2 = int(random.uniform(0, 360))        
            obs, state = self.load_agent([x2, 1.68, y2, o2])
            obses, states = [obs], [state]
            # steps = random.randint(max_steps, max_steps) # typically sample btw [3, 7]
            for _ in range(steps):
                collided = self.sim._default_agent.act("move_backward")
                if collided:
                    break
                obs = self.get_obs()
                state = self.get_state()
                obses.append(obs)
                states.append(state)
            if collided:
                continue
            # sample a different orientation for start 
            ang_deviation = random.randint(-self.sample_deviation, self.sample_deviation)
            o1 = (o2 + ang_deviation) % 360
            state[3] = o1 # changes the orientation of last state in states list
            break

        # ax = self.plot_floor_states(plot_axis=True)
        # for state in states:
        #     self.plot_marker(state, 'r', ax)
        # fig, axes = plt.subplots(1, 8, figsize=(25, 2.5))
        # for i in range(8):
        #     ax = axes[i]
        #     plt.setp(ax.get_xticklabels(), visible=False)
        #     plt.setp(ax.get_yticklabels(), visible=False)
        #     ax.tick_params(axis='both', which='both', length=0)
        # for i in range(len(obses)):
        #     axes[i].imshow(obses[i])
        # plt.tight_layout()

        return states[-1], states[0]         


    def compare_infer(self, env, seed, sample_boundary, dist_min=2, dist_max=3, 
                        sample_scheme='start', max_steps=50, init_sep_steps=7, tags=None):
        '''
        run inference on four models 'random', 'r50k', 'c50k', 'r200'
        save trajectories at save_dir made up from env, idx and policy, e.g. 'Hambleton/r1k/03'
        init_sep_steps denotes how apart are goal and init observation measured in agent steps
        '''
        random.seed(seed)
        np.random.seed(seed)

        x_min, x_max, y_min, y_max = sample_boundary

        if sample_scheme == 'start': # sample start cone
            start_state, goal_state = self.sample_start_cone(x_min, x_max, y_min, y_max, init_sep_steps)
        else:
            start_state, goal_state = self.sample_goal_cone(x_min, x_max, y_min, y_max, 
                                        dist_min=dist_min, dist_max=dist_max)

        if tags is None:
            tags = self.policy.keys()
        for tag in tags:
            save_dir = os.path.join(env, tag, str(seed).zfill(3))
            self.infer_traj(save_dir, start_state, None, goal_state, max_steps=max_steps, policy_tag=tag)


    def plot_obs_traj(self, traj_dir, plot_num=80):
        '''
        visualize observation traj, max capacity for 80 images
        '''
        _, obs_traj = self.load_single_traj(traj_dir, inferred=False, load_img=True)
        fig, axes = plt.subplots(8, 10, figsize=(30, 18))
        fig.suptitle(traj_dir)
        for i in range(80):
            ax = axes[i//10, i%10]
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)        
        for i in range(plot_num):
            ax = axes[i//10, i%10]
            ax.imshow(obs_traj[i]) # show every other obs
        plt.tight_layout()


def compare_traj(nav, env, idx, figsize=(10, 8), nor=2): # nor denotes num of rows to display
    '''
    compare inferred trajectories generated by the four models 'random', 'r50k', 'c50k', 'r200'
    traj_dir e.g. 'tmp/Hambleton/03'
    '''
    noc = len(nav.policy.keys())//nor+len(nav.policy.keys())%nor # noc denotes num of cols
    # fig = plt.figure(figsize=figsize)
    fig, axes = plt.subplots(nor, noc, figsize=figsize)
    for i, tag in enumerate(nav.policy.keys()):
        traj_dir_tag = os.path.join(nav.base_dir, env, tag, str(idx).zfill(3))
        if nor == 1:
            ax = axes[i]
        else:
            ax = axes[i//nor, i%nor]
        # ax = fig.add_subplot(len(nav.policy.keys())//noc+1, noc, i+1)
        ax.scatter(nav.all_states[:, 0], nav.all_states[:, 2], c='k', alpha=0.01)
        ax.add_patch(PolygonPatch(nav.floor, fill=False))
        ax = nav.plot_single_traj(traj_dir_tag, inferred=True, ax=ax)
        steps = len(np.loadtxt(traj_dir_tag + '_action.txt', 'str'))
        # ax.text(0, 0, tag, transform=ax.transAxes)
        ax.set_title(tag + ', steps: ' + str(steps), fontsize=8)
        ax.set_aspect('equal')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.tight_layout()

    
def init(scene_name, base_dir='tmp', policies=None, sample_deviation=360):
    import matplotlib.pyplot as plt
    import nav as nv 
    nav = nv.NavInfer(base_dir, sample_deviation=sample_deviation)
    nav.load_all_states('data/nav_test/' + scene_name)
    nav.load_floor('data/floor_maps/' + scene_name)
    nav.load_sim('data/gibson/' + scene_name + '.glb')
    nav.use_goal_recognizer = True 
    nav.load_perceptual_model(0.60)

    if policies is not None:
        assert type(policies) is list 
        for policy in policies:
            weight_path = os.path.join('weights', policy+'.pth')
            if 'ptz' in policy:
                use_PTZ = True
            else:
                use_PTZ = None
            if 'lstm' in policy:
                with_lstm = True
            else:
                with_lstm = False
            nav.load_policy(policy, weight_path, with_lstm=with_lstm, use_PTZ=use_PTZ)                   
    return nav


def calc_dist(traj_dir):
    states = np.loadtxt(traj_dir + '_state.txt')
    steps = len(states) - 2
    if steps >= 100:
        success = 0
    else:
        success = 1

    goal_state = states[0]
    start_state = states[1]
    end_state = states[-1]

    dist1 = math.hypot(goal_state[0] - start_state[0], goal_state[2] - start_state[2])
    dist2 = math.hypot(goal_state[0] - end_state[0], goal_state[2] - end_state[2])

    return success, dist1, dist2, steps


def batch_calc(nav, base_dir, start, end):
    results = {}
    for tag in nav.policy.keys():
        res = []
        for i in range(start, end):
            res.append(calc_dist(os.path.join(base_dir, str(i) + '_' + tag)))
        results[tag] = np.array(res)
    return results


def calc_stats(nav, base_dir, start, end):
    results = batch_calc(nav, base_dir, start, end)
    stats = {}
    for tag in results.keys():
        res = results[tag]
        res[:, 2] = res[:, 2] / res[:, 1]
        res = res.mean(axis=0)
        stats[tag] = res
    return stats


def calc_feats(nav, model_tag, folder, num=30):
    '''
    calc the features from images 
    folder etc: 'tmp/Hambleton/363_c50k_for'
    '''
    states, images = nav.load_single_traj(folder, inferred=False, load_img=True, thumbnail=True)
    goal = nav.np2tensor(images[0])
    feats = []
    m = nav.policy[model_tag]
    for i in range(num):
        img = torch.cat((nav.np2tensor(images[i]), goal), dim=1)
        assert img.size() == (1, 6, 120, 160)
        feats.append(m.base_model(img.float()))
    return feats 


def calc_feat_diff(feats):
    '''
    calc l1, l2 feature difference, and cosine sim between curr image and goal image
    '''
    l1, l2, cos = [], [], []
    for i in range(1, len(feats)):
        l1.append(torch.nn.L1Loss()(feats[i], feats[0]).detach().numpy())
        l2.append(torch.nn.MSELoss()(feats[i], feats[0]).detach().numpy())
        cos.append(torch.nn.CosineSimilarity()(feats[i], feats[0]).detach().numpy())
    l1 = np.stack(l1).flatten()
    l2 = np.stack(l2).flatten()
    cos = np.stack(cos).flatten()
    # plt.figure()
    # plt.bar(np.arange(1, len(feats))+0.00, l1, alpha=1, width=0.25)
    # plt.bar(np.arange(1, len(feats))+0.25, l2, alpha=1, width=0.25)
    # plt.bar(np.arange(1, len(feats))+0.50, cos, alpha=1, width=0.25)
    # plt.plot(np.vstack((l1, l2, cos)).T)
    # plt.legend(['l1', 'l2', 'cos'])
    return l1, l2, cos


def plot_feat_diff(nav, folder, num=30):
    feats1 = calc_feats(nav, 'r50k_for', folder, num)
    feats2 = calc_feats(nav, 'c50k_for', folder, num)
    l11, l21, cos1 = calc_feat_diff(feats1)
    l12, l22, cos2 = calc_feat_diff(feats2)
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(np.vstack((l11, l12, cos1)).T)
    axes[0].set_title('r50k_for feature diff')
    axes[0].legend(['l1', 'l2', 'cos'])
    axes[1].plot(np.vstack((l12, l22, cos2)).T)
    axes[1].set_title('c50k_for feature diff')
    axes[1].legend(['l1', 'l2', 'cos'])
    return feats1, feats2

def for_simulate(fmodel, feats, loss='l2',):
    '''
    simulate states forward and compare their features against ground truth features
    first feature in feats is the one corresponding to the goal image
    feats is a list
    '''
    feats = torch.cat(feats[1:]).clone().detach() # discarding the first goal feat
    act0 = torch.tensor([1, 0, 0, 0]).expand(len(feats)-1, -1).float() # discard the last feat
    act1 = torch.tensor([0, 1, 0, 0]).expand(len(feats)-1, -1).float()
    act2 = torch.tensor([0, 0, 1, 0]).expand(len(feats)-1, -1).float()
    act3 = torch.tensor([0, 0, 0, 1]).expand(len(feats)-1, -1).float()
    fa0 = torch.cat((feats[:-1].clone(), act0), dim=1)
    fa1 = torch.cat((feats[:-1].clone(), act1), dim=1)
    fa2 = torch.cat((feats[:-1].clone(), act2), dim=1)
    fa3 = torch.cat((feats[:-1].clone(), act3), dim=1)
    p0 = fmodel(fa0) # feature prediction
    p1 = fmodel(fa1)
    p2 = fmodel(fa2)
    p3 = fmodel(fa3)
    if loss == 'l1':
        l = torch.nn.L1Loss(reduction='none')
    elif loss == 'l2':
        l = torch.nn.MSELoss(reduction='none')
    else: # CosineSimilarity
        l = torch.nn.CosineSimilarity()
    if loss in ['l1', 'l2']:
        diff0 = l(p0, feats[1:]).mean(dim=1).detach().numpy()
        diff1 = l(p1, feats[1:]).mean(dim=1).detach().numpy()
        diff2 = l(p2, feats[1:]).mean(dim=1).detach().numpy()
        diff3 = l(p3, feats[1:]).mean(dim=1).detach().numpy()
    else:
        diff0 = l(p0, feats[1:]).detach().numpy()
        diff1 = l(p1, feats[1:]).detach().numpy()
        diff2 = l(p2, feats[1:]).detach().numpy()
        diff3 = l(p3, feats[1:]).detach().numpy()
    # plt.figure()
    # plt.plot(np.vstack((diff0, diff1, diff2, diff3)).T)
    # plt.legend(['stop', 'forward', 'left', 'right'])

    return diff0, diff1, diff2, diff3


def plot_for_sim(nav, feats1, feats2, loss):
    m1 = nav.policy['r50k_for'].for_model
    m2 = nav.policy['c50k_for'].for_model
    d01, d11, d21, d31 = for_simulate(m1, feats1, loss)
    d02, d12, d22, d32 = for_simulate(m2, feats2, loss)
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(np.vstack((d01, d11, d21, d31)).T)
    axes[0].set_title('r50k_for forward simulate ' + loss + ' loss')
    axes[0].legend(['stop', 'forward', 'left', 'right'])
    axes[1].plot(np.vstack((d02, d12, d22, d32)).T)
    axes[1].set_title('c50k_for forward simulate ' + loss + ' loss')
    axes[1].legend(['stop', 'forward', 'left', 'right'])


def pair(images):
    # pair single sequence of numpy images into curr_goal image pairs in tensor
    # assuming the first image in the sequence is goal image
    # assert images.shape[1:] == (120, 160, 3)
    goal = np.expand_dims(images[0], 0).repeat(repeats=len(images)-1, axis=0)
    image_pair = np.concatenate((images[1:], goal), axis=3)
    # assert image_pair.shape[1:] == (120, 160, 6)
    image_pair = torch.tensor((image_pair - 0.5) * 2).permute(0, 3, 1, 2)
    # assert image_pair.size()[1:] == (6, 120, 160)
    return image_pair.float()


def pred_single_traj_bbox(nav, env, idx, policy_tag=None, steps=None, thumbnail=True):
    # given a sequence of images, predict bounding boxes
    traj_dir = os.path.join(nav.base_dir, env, policy_tag, str(idx).zfill(3))
    _, images = nav.load_single_traj(traj_dir, steps=steps, inferred=False, load_img=True, thumbnail=thumbnail)
    image_pair = pair(images) # images are numpy, image_pair tensor for a single traj
    if policy_tag is None:
        policy_tag = list(nav.policy.keys())[0]
    model = nav.policy[policy_tag]
    # bbox = model.pred_coord(model.base_model(image_pair)).unsqueeze(0)
    bbox = model.base_model(image_pair).unsqueeze(0)
    image_pair = image_pair.unsqueeze(0)
    actions = np.append(np.loadtxt(traj_dir + '_action.txt', dtype=str), 'stop')
    conf = np.loadtxt(traj_dir + '_conf.txt')

    for i in range(len(bbox)):
        fig = plt.figure()
        plt.tight_layout()
        gs = gridspec.GridSpec(2, len(bbox[0]), hspace=0, wspace=0.1)
        for j in range(len(bbox[i])):
            im0 = nav.tensor2np(image_pair[i, j, :3])
            ax = setup_ax(fig, gs[0, j])
            ax.imshow(im0)
            if thumbnail:
                draw_bbox(ax, (bbox[i, j, 0]*128, bbox[i, j, 1]*128), 'r', scaling=bbox[i, j, -1], 
                            target_w=128, target_h=128)
            else:
                draw_bbox(ax, (bbox[i, j, 0]*256, bbox[i, j, 1]*256), 'r', scaling=bbox[i, j, -1], 
                            target_w=256, target_h=256)
            ax.set_title([round(bbox[i, j][0].item(), 2), round(bbox[i, j][1].item(), 2), 
                          round(bbox[i, j][2].item(), 2)], fontsize=8)

            im1 = nav.tensor2np(image_pair[i, j, 3:])
            ax = setup_ax(fig, gs[1, j])
            ax.imshow(im1)
            if actions[j] == 'stop':
                atext = 'stp'
            elif actions[j] == 'move_forward':
                atext = 'fwd'
            elif actions[j] == 'turn_left':
                atext = 'lft'
            else:
                atext = 'rgt'
            ax.set_title(atext + f"{conf[j]: .3f}", fontsize=8)
