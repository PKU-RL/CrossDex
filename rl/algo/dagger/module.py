import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import Optional
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNetWithInstanceInfo
# import pytorch_lightning as pl
from typing import List, Optional, Tuple
import copy

class PointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int,
        feature_dim: int,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.backbone = getPointNet({
                'input_feature_dim': self.pc_dim,
                'feat_dim': self.feature_dim
            })

        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            
    def forward(self, input_pc):
        return self.backbone(input_pc)


# class TransPointNetBackbone(nn.Module):
#     def __init__(
#         self,
#         pc_dim: int,
#         feature_dim: int,
#         state_dim: int,
#         use_seg: bool = True,
#     ):
#         super().__init__()

#         cfg = {}
#         cfg["state_dim"] = state_dim
#         cfg["feature_dim"] = feature_dim
#         cfg["pc_dim"] = pc_dim
#         cfg["output_dim"] = feature_dim
#         if use_seg:
#             cfg["mask_dim"] = 2
#         else:
#             cfg["mask_dim"] = 0

#         self.transpn = getPointNetWithInstanceInfo(cfg)

#     def forward(self, input_pc):
#         others = {}
#         input_pc["pc"] = torch.cat([input_pc["pc"], input_pc["mask"]], dim = -1)
#         return self.transpn(input_pc), others


class ActorCriticDagger(nn.Module): # for dagger_value

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, asymmetric=False, use_pc = True, pred_arm_dof_pos = False):
        super(ActorCriticDagger, self).__init__()

        self.asymmetric = asymmetric
        self.use_pc = use_pc
        self.backbone_type = model_cfg['backbone_type']
        self.freeze_backbone = model_cfg["freeze_backbone"]

        actor_hidden_dim = model_cfg['pi_hid_sizes']
        critic_hidden_dim = model_cfg['vf_hid_sizes']
        activation = get_activation(model_cfg['activation'])
        
        if self.use_pc:
            self.pc_shape = model_cfg['pc_shape'] # [512,3]
            self.pc_emb_dim = model_cfg["pc_emb_dim"]
            if self.backbone_type == "pn":
                self.backbone = PointNetBackbone(pc_dim=self.pc_shape[-1], feature_dim=self.pc_emb_dim)
            else:
                raise ValueError(f"Invalid backbone type: {self.backbone_type}")
            #print(self.backbone)
        else:
            self.backbone = None
            self.pc_emb_dim = 0
            self.pc_shape = [0,0]

        self.num_obs = obs_shape[0]
        self.num_state_based_obs = self.num_obs - np.prod(self.pc_shape) + self.pc_emb_dim # replace N*3 pc with pn embedding
        self.pc_start_idx = self.num_obs - np.prod(self.pc_shape)

        actor_layers = []
        critic_layers = []
            
        actor_layers.append(nn.Linear(self.num_state_based_obs, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        
        self.pred_arm_dof_pos = pred_arm_dof_pos
        if pred_arm_dof_pos:
            predictor_layers = []
            predictor_layers.append(nn.Linear(self.num_state_based_obs, actor_hidden_dim[0]))
            predictor_layers.append(activation)
            for l in range(len(actor_hidden_dim)):
                if l == len(actor_hidden_dim) - 1:
                    predictor_layers.append(nn.Linear(actor_hidden_dim[l], 6))
                else:
                    predictor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                    predictor_layers.append(activation)
            self.predictor = nn.Sequential(*predictor_layers)

        critic_layers.append(nn.Linear(self.num_state_based_obs, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)
        if self.pred_arm_dof_pos:
            self.init_weights(self.predictor, actor_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def act(self, observations):
        if self.use_pc and not self.freeze_backbone:
            if self.backbone_type =="pn":
                pc = observations[:, self.pc_start_idx:].reshape(-1, *self.pc_shape)
                pc_feature = self.backbone(pc).reshape(-1, self.pc_emb_dim)
            else:
                raise NotImplementedError
            observations = torch.cat([observations[:, :self.pc_start_idx], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_pc and self.freeze_backbone:
            with torch.no_grad():
                raise NotImplementedError
        else:
            actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        # note: dagger only use actions_mean!!!
        actions = actions_mean #distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # value and action share the vision backbone!
        #observations_copy = observations.clone()
        #observations_copy.detach()
        value = self.critic(observations) #observations_copy)
        
        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()


    def act_inference(self, observations):
        if self.use_pc and not self.freeze_backbone:
            if self.backbone_type =="pn":
                pc = observations[:, self.pc_start_idx:].reshape(-1, *self.pc_shape)
                pc_feature = self.backbone(pc).reshape(-1, self.pc_emb_dim)
            else:
                raise NotImplementedError
            observations = torch.cat([observations[:, :self.pc_start_idx], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_pc and self.freeze_backbone:
            with torch.no_grad():
                raise NotImplementedError
        else:
            actions_mean = self.actor(observations)
        return actions_mean


    def evaluate(self, observations, actions):
        if self.use_pc and not self.freeze_backbone:
            if self.backbone_type =="pn":
                pc = observations[:, self.pc_start_idx:].reshape(-1, *self.pc_shape)
                pc_feature = self.backbone(pc).reshape(-1, self.pc_emb_dim)
            else:
                raise NotImplementedError
            observations = torch.cat([observations[:, :self.pc_start_idx], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_pc and self.freeze_backbone:
            with torch.no_grad():
                raise NotImplementedError
        else:
            actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # value and action share the vision backbone!
        #observations_copy = observations.clone()
        #observations_copy.detach()
        value = self.critic(observations) #observations_copy)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

    ### 10/28: use an arm state predictor
    # predict next arm state
    def predict_arm_dof_pos(self, observations):
        if self.use_pc and not self.freeze_backbone:
            if self.backbone_type =="pn":
                pc = observations[:, self.pc_start_idx:].reshape(-1, *self.pc_shape)
                pc_feature = self.backbone(pc).reshape(-1, self.pc_emb_dim)
            else:
                raise NotImplementedError
            observations = torch.cat([observations[:, :self.pc_start_idx], pc_feature], dim=1)
        elif self.use_pc and self.freeze_backbone:
            raise NotImplementedError
        
        observations_copy = observations.clone()
        observations_copy.detach()
        arm_dof_pos = self.predictor(observations_copy)
        return arm_dof_pos

    # infer action, use predicted arm state as arm action
    def act_inference_using_arm_pred(self, observations, arm_dof_pos):
        if self.use_pc and not self.freeze_backbone:
            if self.backbone_type =="pn":
                pc = observations[:, self.pc_start_idx:].reshape(-1, *self.pc_shape)
                pc_feature = self.backbone(pc).reshape(-1, self.pc_emb_dim)
            else:
                raise NotImplementedError
            observations = torch.cat([observations[:, :self.pc_start_idx], pc_feature], dim=1)
        elif self.use_pc and self.freeze_backbone:
            with torch.no_grad():
                raise NotImplementedError
        
        actions_mean = self.actor(observations)
        arm_dof_pos_diff = self.predictor(observations)
        arm_action = arm_dof_pos + arm_dof_pos_diff
        actions_mean[:, :6] = arm_action
        #print(arm_action, actions_mean)
        return actions_mean
    

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
