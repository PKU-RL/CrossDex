from datetime import datetime
import os
import os.path as osp
from pickle import FALSE
import time
from turtle import done
import sys
from matplotlib.patches import FancyArrow

from gym.spaces import Space

import numpy as np
import statistics
import copy
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .storage import RolloutStorage


class PPOFinetune:
    def __init__(
        self,
        vec_env,
        actor_critic_class,
        train_param,
        log_dir="run",
        apply_reset=False
    ):
        # PPO parameters
        self.clip_param = train_param["cliprange"]
        self.num_learning_epochs = train_param["noptepochs"]
        self.num_mini_batches = train_param["nminibatches"]
        self.num_learning_iterations = train_param["max_iterations"]
        self.num_transitions_per_env = train_param["nsteps"]
        self.value_loss_coef = train_param.get("value_loss_coef", 2.0)
        self.entropy_coef = train_param["ent_coef"]
        self.gamma = train_param["gamma"]
        self.lam = train_param["lam"]
        self.max_grad_norm = train_param.get("max_grad_norm", 2.0)
        self.use_clipped_value_loss = train_param.get("use_clipped_value_loss", False)
        self.init_noise_std = train_param.get("init_noise_std", 0.3)

        self.model_cfg = train_param.policy
        self.sampler = train_param.get("sampler", "sequential")
        self.is_vision = train_param.is_vision

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = vec_env.device
        self.asymmetric = vec_env.num_states > 0

        self.desired_kl = train_param.get("desired_kl", None)
        self.schedule = train_param.get("schedule", "fixed")
        self.step_size = train_param["optim_stepsize"]

        # PPO components
        self.vec_env = vec_env
        self.actor_critic = actor_critic_class(
            self.observation_space.shape,
            self.state_space.shape,
            self.action_space.shape,
            self.init_noise_std,
            self.model_cfg,
            asymmetric=self.asymmetric,
            use_pc=self.is_vision,
        )
        self.actor_critic.to(self.device)
        print(self.actor_critic)
        self.storage = RolloutStorage(
            self.vec_env.num_envs,
            self.num_transitions_per_env,
            self.observation_space.shape,
            self.state_space.shape,
            self.action_space.shape,
            self.device,
            self.sampler,
        )
        # Finetuning components
        self.finetune_ckpt = train_param.finetune_ckpt
        self.finetune_kl_loss_coef = train_param.finetune_kl_loss_coef
        self.finetune_reverse_kl = train_param.finetune_reverse_kl
        if len(self.finetune_ckpt)>1:
            self.actor_critic.load_state_dict(torch.load(self.finetune_ckpt, map_location=self.device))
        self.actor_critic.log_std.data = np.log(self.init_noise_std) * torch.ones_like(self.actor_critic.log_std.data) # set the trainable logstd
        self.actor_critic_pretrained = copy.deepcopy(self.actor_critic)
        self.actor_critic_pretrained.eval()
        print("Pre-trained actor-critic loaded from:", self.finetune_ckpt)
        for param in self.actor_critic_pretrained.parameters():
            param.requires_grad = False
        if self.model_cfg.freeze_backbone:
            for param in self.actor_critic.backbone.parameters():
                param.requires_grad = False
            print("Policy backbone fixed!")
        #self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.step_size)
        # keep the backbone fixed if needed
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.step_size)
        print("Finetuning params:", sum(p.numel() for p in self.optimizer.param_groups[0]['params']))
        self.print_opt_params()
        #print(self.actor_critic_pretrained.log_std)

        # Log
        self.save_interval = train_param["save_interval"]
        self.log_dir = log_dir
        self.print_log = train_param["print_log"]
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = train_param["test"]
        self.save_traj = False  # need to be modified
        self.current_learning_iteration = 0
        if not self.is_testing:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.apply_reset = apply_reset

    def print_opt_params(self):
        model_params = {id(p): name for name, p in self.actor_critic.named_parameters()}
        for i, param_group in enumerate(self.optimizer.param_groups):
            print(f"Param group {i}:")
            for param in param_group['params']:
                param_id = id(param)
                if param_id in model_params:
                    print(f" - {model_params[param_id]}")  # 打印参数的名字
                else:
                    print(" - Unknown parameter (not in model)") 

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.current_learning_iteration = 0  # int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self):
        num_learning_iterations = self.num_learning_iterations
        self.vec_env.reset_idx(np.arange(0,self.vec_env.num_envs))
        current_obs = self.vec_env.reset()["obs"]
        current_states = self.vec_env.get_state()

        ### test
        if self.is_testing:
            n_successes, n_dones = 0, 0
            n_successes_per_env, n_dones_per_env = \
                torch.zeros_like(self.vec_env.successes), torch.zeros_like(self.vec_env.successes)
            for i in range(self.vec_env.max_episode_length):
                with torch.no_grad():
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()["obs"]
                    # Compute the action
                    actions = self.actor_critic.act_inference(current_obs)
                    # Step the vec_environment
                    next_obs_dict, rews, dones, infos = self.vec_env.step(actions)
                    next_obs = next_obs_dict["obs"]
                    current_obs.copy_(next_obs)
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
                        current_obs = self.vec_env.reset()["obs"]
                        current_states = self.vec_env.get_state()
                    with torch.no_grad():
                        # Compute the action
                        actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
                        # Step the vec_environment
                        next_obs_dict, rews, dones, infos = self.vec_env.step(actions)
                        next_obs = next_obs_dict["obs"]
                        next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma,)
                    current_obs.copy_(next_obs)
                    current_states.copy_(next_states)
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

                with torch.no_grad():
                    _, _, last_values, _, _ = self.actor_critic.act(current_obs, current_states)
                stop = time.time()
                collection_time = stop - start
                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss, mean_finetune_kl_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, "model_{}.pt".format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        self.writer.add_scalar(
            "Loss/optimizer_lr", self.step_size, locs["it"]
        )
        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/finetune_kl", locs["mean_finetune_kl_loss"], locs["it"]
        )
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        self.writer.add_scalar(
            "Train2/mean_reward/step", locs["mean_reward"], locs["it"]
        )
        self.writer.add_scalar(
            "Train2/mean_episode_length/episode",
            locs["mean_trajectory_length"],
            locs["it"],
        )

        fps = int(
            self.num_transitions_per_env
            * self.vec_env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Finetune KL loss:':>{pad}} {locs['mean_finetune_kl_loss']:.4f}\n"""
                f"""{'Optimizer lr:':>{pad}} {self.step_size:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Finetune KL loss:':>{pad}} {locs['mean_finetune_kl_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_finetune_kl_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch, states_batch, actions_batch)
                # get the distribution of the initial policy
                with torch.no_grad():
                    _, _, _, mu_prior, sigma_prior = self.actor_critic_pretrained.evaluate(obs_batch, states_batch, actions_batch)

                # update lr
                if self.desired_kl != None and self.schedule == "adaptive":
                    kl = torch.sum(
                        sigma_batch
                        - old_sigma_batch
                        + (
                            torch.square(old_sigma_batch.exp())
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch.exp()))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.step_size

                # Surrogate loss
                ratio = torch.exp(
                    actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
                )
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    raise NotImplementedError
                    # value_clipped = target_values_batch + (
                    #     value_batch - target_values_batch
                    # ).clamp(-self.clip_param, self.clip_param)
                    # value_losses = (value_batch - returns_batch).pow(2)
                    # value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    # value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                
                # finetune KL loss
                if self.finetune_reverse_kl: # KL(now||prior)
                    finetune_kl_loss = torch.sum(
                        (sigma_prior - sigma_batch)
                        + (torch.square(sigma_batch.exp()) + torch.square(mu_prior - mu_batch)) / (2.0 * torch.square(sigma_prior.exp()))
                        - 0.5, dim=-1
                    )
                else: #KL(prior||now)
                    finetune_kl_loss = torch.sum(
                        (sigma_batch - sigma_prior)
                        + (torch.square(sigma_prior.exp()) + torch.square(mu_prior - mu_batch)) / (2.0 * torch.square(sigma_batch.exp()))
                        - 0.5, dim=-1
                    )
                finetune_kl_loss = finetune_kl_loss.mean()

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + self.finetune_kl_loss_coef * finetune_kl_loss
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_finetune_kl_loss += finetune_kl_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_finetune_kl_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, mean_finetune_kl_loss
