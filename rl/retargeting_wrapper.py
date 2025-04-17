'''
Env wrapper with eigengrasp action space. Use PCA and NN retargeting to compute robot action.
'''

import os, sys
import random
import torch
import numpy as np
import gym
from gym import spaces
#from dex_retargeting.constants import RobotName, ROBOT_NAME_MAP
from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, tensor_clamp
sys.path.append('..')
from retargeting.retargeting_nn_utils import EigenRetargetModel

class RetargetingWrapper:
    def __init__(self, env, robot_name, dataset:str="grab", add_random_dataset:bool=False):
        self.env = env
        self.retargeting_model = EigenRetargetModel(dataset, robot_name, add_random_dataset, device=self.env.device)
        self.num_eigengrasp_actions = self.retargeting_model.principal_vectors.shape[0]
        self.num_actions = self.env.num_actions = self.env.num_actions - self.retargeting_model.robot_dim \
             + self.num_eigengrasp_actions # replace robot hand dofs with eigengrasps
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
        self.eigengrasp_action_indices = [i for i in range(6,6+self.num_eigengrasp_actions)]
        print("Eigengrasp action indices:", self.eigengrasp_action_indices)
        self.retarget2isaac = [self.retargeting_model.robot_joint_names.index(n) for n in self.env.hand_dof_names]
        print("isaac hand dof names:", self.env.hand_dof_names)
        print("retarget2isaac dof idx mapping:", self.retarget2isaac)
        self.prev_eigengrasp_targets = torch.zeros(
            (self.env.num_envs, self.num_eigengrasp_actions), dtype=torch.float, device=self.env.device
        ) # for relative control

        # 修改底层环境的pre_physics
        self.env.pre_physics_step = self.pre_physics_step

        # 添加env的所有属性和方法到 Wrapper 中
        for attr_name in dir(env):
            if not hasattr(self, attr_name):
                setattr(self, attr_name, getattr(env, attr_name))

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    def reset(self):
        self.prev_eigengrasp_targets = torch.zeros(
            (self.num_envs, self.num_eigengrasp_actions), dtype=torch.float, device=self.device
        ) # for relative control
        return self.env.reset()

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.env.actions = actions.clone().to(self.device)
        
        # process hand dof actions
        '''
        if self.use_relative_control:
            eigengrasp_targets = self.prev_eigengrasp_targets + \
                self.dof_speed_scale * self.dt * self.env.actions[:, self.eigengrasp_action_indices]
            self.prev_eigengrasp_targets = tensor_clamp(eigengrasp_targets, 
                self.retargeting_model.min_values, self.retargeting_model.max_values)
            actions_hand_dof = self.retargeting_model.retarget(self.prev_eigengrasp_targets)[:, self.retarget2isaac] # isaac hand dofs
            self.cur_targets[:, self.hand_dof_indices] = tensor_clamp(
                actions_hand_dof, 
                self.robot_dof_lower_limits[self.hand_dof_indices],
                self.robot_dof_upper_limits[self.hand_dof_indices],
            )
        '''
        if True: # hand always use direct control
            # convert eigengrasp into robot dofs
            actions_eigengrasp = self.env.actions[:, self.eigengrasp_action_indices] # -1~1
            actions_eigengrasp = scale(actions_eigengrasp, 
                self.retargeting_model.min_values, self.retargeting_model.max_values) # min_value~max_value
            actions_hand_dof = self.retargeting_model.retarget(actions_eigengrasp)[:, self.retarget2isaac] # isaac hand dofs

            self.cur_targets[:, self.hand_dof_indices] = tensor_clamp(
                actions_hand_dof, #self.env.actions[:, self.hand_dof_indices],
                self.robot_dof_lower_limits[self.hand_dof_indices],
                self.robot_dof_upper_limits[self.hand_dof_indices],
            )
            #print((actions_hand_dof - self.cur_targets[:, self.hand_dof_indices]).abs().max())
            self.cur_targets[:, self.hand_dof_indices] = (
                self.act_moving_average * self.cur_targets[:, self.hand_dof_indices]
                + (1.0 - self.act_moving_average)
                * self.prev_targets[:, self.hand_dof_indices]
            )
            self.cur_targets[:, self.hand_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.hand_dof_indices],
                self.robot_dof_lower_limits[self.hand_dof_indices],
                self.robot_dof_upper_limits[self.hand_dof_indices],
            )

        # process arm dof actions
        if self.arm_controller == "qpos":
            if self.use_relative_control:
                targets = (
                    self.prev_targets[:, self.arm_dof_indices]
                    + self.dof_speed_scale * self.dt * self.env.actions[:, self.arm_dof_indices]
                )
                self.cur_targets[:, self.arm_dof_indices] = tensor_clamp(
                    targets,
                    self.robot_dof_lower_limits[self.arm_dof_indices],
                    self.robot_dof_upper_limits[self.arm_dof_indices],
                )
            else:
                self.cur_targets[:, self.arm_dof_indices] = scale(
                    self.env.actions[:, self.arm_dof_indices],
                    self.robot_dof_lower_limits[self.arm_dof_indices],
                    self.robot_dof_upper_limits[self.arm_dof_indices],
                )
        elif self.arm_controller == "ik":  # direct qpos control
            pos_err = (
                self.env.actions[:, 0:3]
                - self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:3]
            )
            orn_err = orientation_error(
                matrix_to_quaternion(
                    rotation_6d_to_matrix(
                        torch.torch.concat(
                            (self.env.actions[:, 3:6], self.env.actions[:, -3:]),
                            dim=-1,
                        )
                    )
                ),
                self.rigid_body_states.view(-1, 13)[self.eef_idx, 3:7],
            )
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            u = self._control_ik(dpose)
            self.cur_targets[:, self.arm_dof_indices] = (
                self.robot_dof_pos[:, self.arm_dof_indices] + u
            )

        self.prev_targets[:, self.robot_dof_indices] = self.cur_targets[
            :, self.robot_dof_indices
        ]

        # print(self.hand_dof_indices, self.hand_dof_names)
        # print(self.robot_dof_state[:,self.hand_dof_indices,0][0])
        # self.cur_targets[:, self.hand_dof_indices] = torch.tensor([
        #         1.2684, -0.0867,  0.8991,  0.5065,  0.6873,  0.8256,  1.2283, -1.3400,
        #         0.9989, -0.3568,  0.7487,  0.6715,  1.4569,  0.3157,  0.6957,  0.7189
        #     ]).to(self.device)

        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets)
        )