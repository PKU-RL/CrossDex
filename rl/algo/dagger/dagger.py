from datetime import datetime
#from importlib.resources import path
import os
#import os.path as osp
#import pdb
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time

from matplotlib.patches import FancyArrow
from gym import spaces

from gym.spaces import Space

import numpy as np
import statistics
import copy
import yaml
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from .storage import RolloutStorage, PPORolloutStorage


class DAGGER:
    def __init__(self, vec_env, actor_critic_class, train_param, log_dir, apply_reset=False, pred_arm_dof_pos=False):
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.student_observation_space, Space):
            raise TypeError("vec_env.student_observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.student_observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.expert_observation_space = vec_env.observation_space
        print("\nDAgger spaces: obs space: {}, expert obs space {}, act space {}".format(
            self.observation_space, self.expert_observation_space, self.action_space))

        # DAgger
        self.is_vision = train_param.is_vision
        self.device = vec_env.device
        self.asymmetric = train_param.asymmetric
        self.schedule = train_param.schedule
        self.learning_rate = train_param.optim_stepsize
        self.buffer_size = train_param.nsteps #train_param.buffer_size
        self.vec_env = vec_env
        self.num_learning_iterations = train_param.max_iterations
        self.num_learning_epochs = train_param.noptepochs
        self.num_mini_batches = train_param.nminibatches
        self.num_transitions_per_env = train_param.nsteps
        init_noise_std = train_param.init_noise_std
        self.sampler = train_param.get("sampler", "sequential")
        self.pred_arm_dof_pos = pred_arm_dof_pos

        # Log
        self.save_interval = train_param.save_interval
        self.log_dir = log_dir
        self.print_log = train_param.print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = train_param.test
        self.test_expert = train_param.test_expert
        self.current_learning_iteration = 0
        self.apply_reset = apply_reset
        
        # create student
        self.value_loss_cfg = train_param['value_loss']
        self.apply_value_net = self.value_loss_cfg['apply']
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.action_space.shape, 
            init_noise_std, model_cfg=train_param.policy, asymmetric=self.asymmetric, use_pc = self.is_vision,
            pred_arm_dof_pos=self.pred_arm_dof_pos)
        self.actor_critic.to(self.device)
        print("\nDAgger actor critic:", self.actor_critic)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        
        if (not self.is_testing) or self.test_expert:
            # create replay buffers
            if self.apply_value_net:
                self.ppo_buffer = PPORolloutStorage(self.vec_env.num_envs, self.num_transitions_per_env, self.observation_space.shape,
                    self.state_space.shape, self.action_space.shape, self.device, self.sampler)
                self.gamma = self.value_loss_cfg['gamma']
                self.lam = self.value_loss_cfg['lam']
                self.use_clipped_value_loss = self.value_loss_cfg['use_clipped_value_loss']
                self.clip_range = self.value_loss_cfg['clip_range']
                self.value_loss_coef = self.value_loss_cfg['value_loss_coef']
            self.storage = RolloutStorage(self.vec_env.num_envs, self.buffer_size, self.observation_space.shape,
                self.state_space.shape, self.action_space.shape, self.device, self.sampler, pred_arm_dof_pos=self.pred_arm_dof_pos)
        
            # load experts
            from omegaconf import DictConfig, OmegaConf
            from algo import ppo
            import json
            self.expert_cfg_dict = train_param['expert']
            self.expert_list = [] # load expert for each object in the env
            for object_name in self.vec_env.object_names:
                expert_cfg = self.expert_cfg_dict[object_name]
                load_cfg_path = os.path.join(expert_cfg['path'], "config.json")
                with open(load_cfg_path, "r") as f:
                    cfg_load = json.load(f)
                cfg_load = OmegaConf.create(cfg_load)
                expert = ppo.ActorCritic(
                    self.expert_observation_space.shape,
                    self.state_space.shape,
                    self.action_space.shape,
                    init_noise_std,
                    cfg_load.train.params.policy,
                    asymmetric=self.asymmetric,
                    use_pc=False,
                )
                expert.to(self.device)
                expert.load_state_dict(torch.load(os.path.join(expert_cfg['path'], expert_cfg['ckpt']), 
                    map_location=self.device))
                expert.eval()
                self.expert_list.append(expert)
                print('Expert {} loaded from {}'.format(object_name, os.path.join(expert_cfg['path'], expert_cfg['ckpt'])))
            self.expert_to_env_ids = [torch.tensor(x, dtype=torch.int64, device=self.device) \
                for x in self.vec_env.object_to_env_ids]
            #print(self.vec_env.object_names, self.expert_list, self.expert_to_env_ids)


    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.current_learning_iteration = 0 #int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    @torch.no_grad()
    def multi_expert_inference_batch(self, obs_buf, act_buf):
        B, N, d = obs_buf.shape # buffer size * num envs * dim
        for expert, env_ids in zip(self.expert_list, self.expert_to_env_ids):
            act_buf[:,env_ids] = expert.act_inference(obs_buf[:,env_ids].view(-1,d)).view(B,len(env_ids),-1)

    @torch.no_grad()
    def multi_expert_inference(self, current_obs):
        action = torch.zeros((current_obs.shape[0], self.action_space.shape[0]), device=self.device) # num envs * dim
        for expert, env_ids in zip(self.expert_list, self.expert_to_env_ids):
            action[env_ids] = expert.act_inference(current_obs[env_ids])
        return action


    def run(self):
        num_learning_iterations = self.num_learning_iterations
        self.vec_env.reset_idx(np.arange(0,self.vec_env.num_envs))
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        ### test
        if self.is_testing:
            n_successes, n_dones = 0, 0
            n_successes_per_env, n_dones_per_env = \
                torch.zeros_like(self.vec_env.successes), torch.zeros_like(self.vec_env.successes)
            n_test_timesteps = self.vec_env.max_episode_length
            if "screenshot" in self.vec_env.cfg:
                n_test_timesteps = 1000000
            for i in range(n_test_timesteps):
                with torch.no_grad():
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                    # Compute the action
                    if not self.test_expert:
                        actions = self.actor_critic.act_inference(current_obs["student_obs"])
                    else:
                        actions = self.multi_expert_inference(current_obs["obs"])
                    #print(current_obs, actions)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    current_obs = next_obs
                    n_dones += self.vec_env.reset_buf.sum()
                    n_dones_per_env += self.vec_env.reset_buf
                    tmp = torch.where(self.vec_env.reset_buf>0, 
                        self.vec_env.current_successes,
                        torch.zeros_like(self.vec_env.current_successes))
                    n_successes += tmp.sum()
                    n_successes_per_env += tmp
            
            print("success_rate: {}, n_dones: {}\n".format((n_successes/n_dones).item(), n_dones.item()))
            # success per embodiment
            if hasattr(self.vec_env, "hand_names"):
                N = len(self.vec_env.hand_names)
                for i, hn in enumerate(self.vec_env.hand_names):
                    print("{}: {}".format(hn, 
                        n_successes_per_env[i::N].sum()/n_dones_per_env[i::N].sum()))
                print()
            # success per object
            if hasattr(self.vec_env, "object_names"):
                success_per_object = {}
                N = len(self.vec_env.object_names)
                for i, obj_name in enumerate(self.vec_env.object_names):
                    success_per_object[obj_name] = \
                        (n_successes_per_env[i::N].sum()/n_dones_per_env[i::N].sum()).cpu().numpy()
                success_per_object = sorted(success_per_object.items(), key=lambda item: item[1], reverse=True)
                print(success_per_object)
                print()
            # save raw test data
            if hasattr(self.vec_env, "hand_names") and hasattr(self.vec_env, "object_names"):
                import pickle
                save_dict = {'hand_names': self.vec_env.hand_names, 'object_names': self.vec_env.object_names,
                             'n_successes_per_env': n_successes_per_env, 'n_dones_per_env': n_dones_per_env}
                with open('test_results.pkl', 'wb') as f:
                    pickle.dump(save_dict, f)
                
            exit()
        
        ### train
        else:
            # multi_expert
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    with torch.no_grad():
                        #print((current_obs['student_obs']-current_obs['obs']).abs().max())
                        if self.apply_value_net:
                            actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs["student_obs"])
                        else:
                            actions = self.actor_critic.act_inference(current_obs["student_obs"])
                        # multi_expert
                        actions_expert = self.multi_expert_inference(current_obs["obs"])
                        #print((actions_expert - actions).abs().max())
                        #actions_expert = torch.clamp(actions_expert, -self.vec_env.clip_actions, self.vec_env.clip_actions)
                        # Step the vec_environment
                        next_obs, rews, dones, infos = self.vec_env.step(actions) # DAgger: actions; Online BC: actions_expert
                        next_states = self.vec_env.get_state()
                        if self.pred_arm_dof_pos:
                            arm_dof_pos_diff = self.vec_env.arm_dof_pos_diff_buf.clone()
                            #print(arm_dof_pos_diff, arm_dof_pos_diff.shape)
                            # Record the transition
                        else:
                            arm_dof_pos_diff = None
                        self.storage.add_transitions(current_obs["student_obs"], actions_expert, rews, dones, arm_dof_pos_diff)

                        # value_net
                        if self.apply_value_net:
                            self.ppo_buffer.add_transitions(current_obs["student_obs"], current_states, actions, rews, dones, values, actions_log_prob, mu, sigma)
                        current_obs = next_obs
                        current_states = next_states
                        # Book keeping
                        ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)
                
                # value_net: compute last value
                if self.apply_value_net:
                    with torch.no_grad():
                        actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs["student_obs"])

                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                if self.apply_value_net:
                    self.ppo_buffer.compute_returns(values, self.gamma, self.lam)
                    mean_policy_loss, mean_value_loss, mean_arm_pred_loss = self.update()
                    self.ppo_buffer.clear()
                else:
                    mean_policy_loss, mean_arm_pred_loss = self.update()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        if not 'mean_value_loss' in locs:
            locs['mean_value_loss'] = 0
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/policy', locs['mean_policy_loss'], locs['it'])
        self.writer.add_scalar('Loss/predictor', locs['mean_arm_pred_loss'], locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Arm prediction loss:':>{pad}} {locs['mean_arm_pred_loss']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Arm prediction loss:':>{pad}} {locs['mean_arm_pred_loss']:.4f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self):
        mean_policy_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        if self.apply_value_net:
            mean_value_loss = 0
            batch_value = self.ppo_buffer.mini_batch_generator(self.num_mini_batches)
        mean_arm_pred_loss = 0
        
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                actions_expert_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                actions_batch = self.actor_critic.act_inference(obs_batch)
                #print((actions_batch - actions_expert_batch).abs().max())
                # Policy loss
                dagger_loss = F.huber_loss(actions_batch, actions_expert_batch) #F.mse_loss(actions_batch, actions_expert_batch)
                loss = dagger_loss
                mean_policy_loss += dagger_loss.item()
        
                if self.apply_value_net:
                    obs_batch = self.ppo_buffer.observations.view(-1, *self.ppo_buffer.observations.size()[2:])[indices]
                    actions_batch = self.ppo_buffer.actions.view(-1, self.ppo_buffer.actions.size(-1))[indices]
                    target_values_batch = self.ppo_buffer.values.view(-1, 1)[indices]
                    returns_batch = self.ppo_buffer.returns.view(-1, 1)[indices]

                    actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch, actions_batch)

                    if self.use_clipped_value_loss:
                        clip_range = self.clip_range
                        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-clip_range, clip_range)
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2).mean()
                    value_loss = value_loss * self.value_loss_coef
                    loss += value_loss
                    mean_value_loss += value_loss.item()
                
                # arm dof pos prediction
                if self.pred_arm_dof_pos:
                    arm_dof_pos_diff_batch = self.storage.arm_dof_pos_diff.view(-1, self.storage.arm_dof_pos_diff.size(-1))[indices]
                    arm_dof_pos_diff_pred = self.actor_critic.predict_arm_dof_pos(obs_batch)
                    arm_pred_loss = F.huber_loss(arm_dof_pos_diff_pred*10., arm_dof_pos_diff_batch*10.)
                    loss += arm_pred_loss
                    mean_arm_pred_loss += arm_pred_loss.item()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_policy_loss /= num_updates

        if self.apply_value_net:
            mean_value_loss /= num_updates
        
            return mean_policy_loss, mean_value_loss, mean_arm_pred_loss
        else:
            return mean_policy_loss, mean_arm_pred_loss
