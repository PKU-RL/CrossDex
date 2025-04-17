# unified dex grasping env for all embodiments

import os
import random
import torch
import numpy as np
from torch.nn import functional as F

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class CrossDexGrasp(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.init_configs(cfg)

        super().__init__(
            self.cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        print("num obs: {}, num actions: {}".format(self.num_obs, self.num_acts))


        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(
                round(self.reset_time / (control_freq_inv * self.dt))
            )
            print("Reset time: ", self.reset_time)
            print("Max episode length: ", self.max_episode_length)

        # viewer camera setup
        if self.viewer != None:
            cam_pos = gymapi.Vec3(1.8, -3.2, 3.0)
            cam_target = gymapi.Vec3(1.8, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        if self.arm_controller == "ik":
            _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
            jacobian = gymtorch.wrap_tensor(_jacobian)
            # jacobian entries corresponding to eef
            self.j_eef = jacobian[:, self.arm_eef_index - 1, :, :6]

        # create some wrapper tensors for different slices
        self.robot_default_dof_pos = torch.zeros(
            self.num_robot_dofs, dtype=torch.float, device=self.device
        )
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.robot_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_robot_dofs
        ]
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.current_successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.consecutive_successes = torch.zeros(
            1, dtype=torch.float, device=self.device
        )
        self.total_successes = 0
        self.total_resets = 0


    def init_configs(self, cfg):
        self.cfg = cfg
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]

        # scale factor of velocity based observations
        self.vel_obs_scale = 0.2
        # NOTE don't use, scale factor of force and torque based observations
        #self.force_torque_obs_scale = 10.0

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.arm_controller = self.cfg["env"]["armController"]
        self.dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_vis = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = -1 #self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)
        self.goal_height = self.cfg["env"].get("goalHeight", 0.6)

        self.obs_type = self.cfg["env"]["observationType"]
        self.multi_task = self.cfg["env"]["multiTask"]
        self.multi_task_label = self.cfg["env"]["multiTaskLabel"]

        assert self.arm_controller in ["ik", "qpos"]
        #assert self.obs_type in ["full_no_vel", "full", "full_state", "pos"]

        ## hand-specific settings
        if "hand_name" in self.cfg:
            self.hand_specific_cfg = self.cfg["hand_specific"][self.cfg["hand_name"]]
            self.palm_offset = self.hand_specific_cfg["palm_offset"]
            self.robot_asset_file = self.hand_specific_cfg["robot_asset_file"]
            # need to set the number of observations according to the robot
            self.num_obs_dict = self.hand_specific_cfg["num_obs_dict"]

            #self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
            self.cfg["env"]["numObservations"] = \
                sum([(self.num_obs_dict[i] if i in self.obs_type else 0) for i in self.num_obs_dict]) #self.num_obs_dict[self.obs_type]
            if self.multi_task:
                raise Exception("TODO: add observation dims for multitask")
                self.obs_type += "+onehottask"
            self.cfg["env"]["numStates"] = 0 #(self.num_obs_dict[self.obs_type] if self.asymmetric_obs else 0)
            self.cfg["env"]["numActions"] = self.hand_specific_cfg["numActions"] 

            if self.arm_controller == "ik":  # use rotation 6D representation
                self.cfg["env"]["numObservations"] += 3
                #self.cfg["env"]["numStates"] += 3 if self.asymmetric_obs else 0
                self.cfg["env"]["numActions"] += 3


    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = 2 if self.cfg["sim"]["up_axis"] == "z" else 1

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

        # if randamizing, apply once immediately on startup before the first sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]

        robot_asset, robot_dof_props, robot_start_pose = self._prepare_robot_asset(
            asset_root, self.robot_asset_file
        )
        if self.multi_task: # multitask: add different object assets in different envs
            object_assets = []
            object_start_poses = []
            object_asset_dir = self.cfg["env"]["asset"]["objectAssetDir"]
            object_asset_fn_list = sorted(os.listdir(os.path.join(asset_root, object_asset_dir)))
            for asset in object_asset_fn_list:
                object_asset, _, object_start_pose = self._prepare_object_asset(
                    asset_root, os.path.join(object_asset_dir, asset)
                )
                object_assets.append(object_asset)
                object_start_poses.append(object_start_pose)
        else: # single task: one object
            object_asset_file = self.cfg["env"]["asset"]["objectAssetFile"]
            object_asset, object_dof_props, object_start_pose = self._prepare_object_asset(
                asset_root, object_asset_file
            )
        table_asset, table_start_pose, side_panel_asset, side_panel_start_pose = self._prepare_table_asset()

        self.envs = []
        self.robots = []
        self.objects = []

        if self.arm_controller == "ik":
            self.eef_idx = []

        self.robot_indices = []
        self.object_indices = []
        self.robot_start_states = []
        self.object_init_states = []

        if self.multi_task:
            self.task_label_onehot = torch.zeros((num_envs, len(object_assets))).to(self.device)
            self.num_tasks = len(object_assets)

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # aggregate size
            max_agg_bodies = self.num_robot_bodies + self.num_object_bodies + 2
            max_agg_shapes = self.num_robot_shapes + self.num_object_shapes + 2
            if self.aggregate_mode > 0:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # create robot actor
            robot_actor = self.gym.create_actor(
                env_ptr, robot_asset, robot_start_pose, "robot", i, -1, 0
            )
            self.robot_start_states.append(
                [robot_start_pose.p.x,robot_start_pose.p.y,robot_start_pose.p.z,
                robot_start_pose.r.x,robot_start_pose.r.y,robot_start_pose.r.z,
                robot_start_pose.r.w,0,0,0,0,0,0,])
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            robot_idx = self.gym.get_actor_index(
                env_ptr, robot_actor, gymapi.DOMAIN_SIM
            )
            self.robot_indices.append(robot_idx)

            # add obejct
            if self.multi_task:
                idx = i % len(object_assets) # idx = random.randint(0, len(object_assets) - 1)
                object_asset = object_assets[idx]
                object_start_pose = object_start_poses[idx]
                self.task_label_onehot[i, idx]=1.

            object_handle = self.gym.create_actor(
                env_ptr, object_asset, object_start_pose, "object", i, -1, 0
            )
            self.object_init_states.append(
                [object_start_pose.p.x,object_start_pose.p.y,object_start_pose.p.z,
                object_start_pose.r.x,object_start_pose.r.y,object_start_pose.r.z,
                object_start_pose.r.w,0,0,0,0,0,0,])
            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            self.object_indices.append(object_idx)

            # add table
            table_actor = self.gym.create_actor(
                env_ptr, table_asset, table_start_pose, "table", i, -1, 0
            )
            side_panel_actor = self.gym.create_actor(
                env_ptr, side_panel_asset, side_panel_start_pose, "side_panel", i+num_envs, -1, 0
            )

            # enable DOF force sensors, if needed
            #if self.obs_type == "full_state" or self.asymmetric_obs:
            #    self.gym.enable_actor_dof_force_sensors(env_ptr, robot_actor)
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robots.append(robot_actor)

            if self.arm_controller == "ik":
                eef_idx = self.gym.find_actor_rigid_body_index(
                    env_ptr, robot_actor, self.palm, gymapi.DOMAIN_SIM
                )
                self.eef_idx.append(eef_idx)

        self.robot_start_states = to_torch(
            self.robot_start_states, device=self.device
        ).view(num_envs, 13)
        self.object_init_states = to_torch(
            self.object_init_states, device=self.device
        ).view(num_envs, 13)

        self.fingertip_handles = to_torch(
            self.fingertip_handles, dtype=torch.long, device=self.device
        )
        self.palm_handle = to_torch(
            self.palm_handle, dtype=torch.long, device=self.device
        )

        self.robot_indices = to_torch(
            self.robot_indices, dtype=torch.long, device=self.device
        )
        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device
        )
        if self.arm_controller == "ik":
            self.eef_idx = to_torch(self.eef_idx, dtype=torch.long, device=self.device)


    def _prepare_robot_asset(self, asset_root, asset_file):
        # load arm hand asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        if self.cfg["env"]["useRobotVhacd"]:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 300000

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        # drive_mode: 0: none, 1: position, 2: velocity, 3: force
        asset_options.default_dof_drive_mode = 0

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        # get arm hand asset info
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_robot_actuators = self.num_robot_dofs
        print("self.num_robot_bodies: ", self.num_robot_bodies)
        print("self.num_robot_shapes: ", self.num_robot_shapes)
        print("self.num_robot_dofs: ", self.num_robot_dofs)
        print("self.num_robot_actuators: ", self.num_robot_actuators)

        # need to set the names according to the robot
        self.palm = self.hand_specific_cfg["palm_link"] #"palm_lower"
        self.fingertips = self.hand_specific_cfg["fingertips_link"]
        self.arm_dof_names = ["arm_joint1","arm_joint2","arm_joint3",
            "arm_joint4","arm_joint5","arm_joint6",]
        self.num_fingers = len(self.fingertips)
        self.hand_dof_names = []
        for i in range(self.num_robot_dofs):
            joint_name = self.gym.get_asset_dof_name(robot_asset, i)
            if joint_name not in self.arm_dof_names:
                self.hand_dof_names.append(joint_name)
        self.robot_dof_names = self.arm_dof_names + self.hand_dof_names

        self.palm_handle = self.gym.find_asset_rigid_body_index(robot_asset, self.palm)
        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(robot_asset, fingertip)
            for fingertip in self.fingertips
        ]
        if -1 in self.fingertip_handles or self.palm_handle==-1:
            raise Exception("Fingertip names or palm name not found!")
        self.arm_dof_indices = [
            self.gym.find_asset_dof_index(robot_asset, name)
            for name in self.arm_dof_names
        ]
        self.hand_dof_indices = [
            self.gym.find_asset_dof_index(robot_asset, name)
            for name in self.hand_dof_names
        ]
        self.robot_dof_indices = self.arm_dof_indices + self.hand_dof_indices
        self.robot_dof_indices = to_torch(
            self.robot_dof_indices, dtype=torch.long, device=self.device
        )
        # create fingertip force sensors, if needed
        #if self.obs_type == "full_state" or self.asymmetric_obs:
        #    sensor_pose = gymapi.Transform()
        #    for ft_handle in self.fingertip_handles:
        #        self.gym.create_asset_force_sensor(robot_asset, ft_handle, sensor_pose)
        self.num_arm_dofs = len(self.arm_dof_indices)
        self.num_hand_dofs = len(self.hand_dof_indices)
        assert self.num_arm_dofs+self.num_hand_dofs==self.num_robot_dofs

        # get eef index
        robot_link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.arm_eef_index = robot_link_dict[self.palm]

        # arm hand dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        self.robot_dof_default_pos = []
        self.robot_dof_default_vel = []

        for i in range(self.num_robot_dofs):
            self.robot_dof_lower_limits.append(robot_dof_props["lower"][i])
            self.robot_dof_upper_limits.append(robot_dof_props["upper"][i])
            self.robot_dof_default_pos.append(0.0)
            self.robot_dof_default_vel.append(0.0)
            if i in self.arm_dof_indices:
                robot_dof_props["driveMode"][i] = 1
                robot_dof_props["stiffness"][i] = 1000
                robot_dof_props["damping"][i] = 20
                robot_dof_props["friction"][i] = 0.01
                robot_dof_props["armature"][i] = 0.001
            elif i in self.hand_dof_indices:
                robot_dof_props["driveMode"][i] = 1
                robot_dof_props["stiffness"][i] = 3
                robot_dof_props["damping"][i] = 0.5
                robot_dof_props["friction"][i] = 0.01
                robot_dof_props["armature"][i] = 0.001
            print('DoF {} effort {:.2} stiffness {:.2} damping {:.2} friction {:.2} armature {:.2} limit {:.2}~{:.2}'.format(
                self.robot_dof_names[(self.arm_dof_indices + self.hand_dof_indices).index(i)], 
                robot_dof_props['effort'][i], robot_dof_props['stiffness'][i],
                robot_dof_props['damping'][i], robot_dof_props['friction'][i],
                robot_dof_props['armature'][i], robot_dof_props['lower'][i], 
                robot_dof_props['upper'][i]))
        print("\nArm dofs: {}. Hand dofs: {}.\n".format(self.arm_dof_names, self.hand_dof_names))

        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)
        self.robot_dof_default_pos = to_torch(self.robot_dof_default_pos, device=self.device)
        self.robot_dof_default_vel = to_torch(self.robot_dof_default_vel, device=self.device)

        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(-0.5, 0.0, 0.82)
        robot_start_pose.r = gymapi.Quat.from_euler_zyx(0.0, -np.pi / 2, np.pi)
        return robot_asset, robot_dof_props, robot_start_pose

    def _prepare_object_asset(self, asset_root, asset_file):
        # load object asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.collapse_fixed_joints = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        if self.cfg["env"]["useObjectVhacd"]:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 300000

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        # drive_mode: 0: none, 1: position, 2: velocity, 3: force
        asset_options.default_dof_drive_mode = 0

        object_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        # get object asset info
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)

        object_dof_props = self.gym.get_asset_dof_properties(object_asset)
        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []
        for i in range(self.num_object_dofs):
            self.object_dof_lower_limits.append(object_dof_props["lower"][i])
            self.object_dof_upper_limits.append(object_dof_props["upper"][i])

        self.object_dof_lower_limits = to_torch(
            self.object_dof_lower_limits, device=self.device
        )
        self.object_dof_upper_limits = to_torch(
            self.object_dof_upper_limits, device=self.device
        )

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.3)
        object_start_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)

        return object_asset, object_dof_props, object_start_pose

    def _prepare_table_asset(self):
        # create table asset
        table_dims = gymapi.Vec3(1, 1.5, 0.3)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(
            self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options
        )
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0.0, 0.0, table_dims.z / 2)

        side_panel_dims = gymapi.Vec3(0.06, 1.5, 1.1)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        side_panel_asset = self.gym.create_box(
            self.sim,
            side_panel_dims.x,
            side_panel_dims.y,
            side_panel_dims.z,
            asset_options,
        )
        side_panel_start_pose = gymapi.Transform()
        side_panel_start_pose.p = gymapi.Vec3(-0.53, 0.0, side_panel_dims.z / 2)

        return table_asset, table_start_pose, side_panel_asset, side_panel_start_pose

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.current_successes[:],
            self.consecutive_successes[:],
            reward_info,
        ) = compute_task_rewards(
            self.reset_buf,
            self.progress_buf,
            self.successes,
            self.current_successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.goal_height,
            self.palm_center_pos,
            self.fingertip_pos,
            torch.as_tensor(self.num_fingers).to(self.device),
            self.actions,
            self.dist_reward_scale,
            self.object_init_states,
            self.action_penalty_scale,
            self.success_tolerance,
            self.av_factor,
        )

        self.extras.update(reward_info)
        self.extras["successes"] = self.successes
        self.extras["current_successes"] = self.current_successes
        self.extras["consecutive_successes"] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = (
                self.total_successes + (self.successes * self.reset_buf).sum()
            )
            # The direct average shows the overall result more quickly, but slightly undershoots long term policy performance.
            print(
                "Direct average consecutive successes = {:.1f}".format(
                    direct_average_successes / (self.total_resets + self.num_envs)
                )
            )
            if self.total_resets > 0:
                print(
                    "Post-Reset average consecutive successes = {:.1f}".format(
                        self.total_successes / self.total_resets
                    )
                )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        #if self.obs_type == "full_state" or self.asymmetric_obs:
        #    self.gym.refresh_force_sensor_tensor(self.sim)
        #    self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.palm_state = self.rigid_body_states[:, self.palm_handle][..., :13]
        self.palm_pos = self.palm_state[..., :3]
        self.palm_rot = self.palm_state[..., 3:7]
        self.palm_center_pos = self.palm_pos + quat_apply(
            self.palm_rot, to_torch(self.palm_offset).repeat(self.num_envs, 1)
        )

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][
            ..., :13
        ]
        self.fingertip_pose = self.fingertip_state[..., :7]
        self.fingertip_pos = self.fingertip_state[..., :3]
        self.fingertip_rot = self.fingertip_state[..., 3:7]

        self.compute_required_observations()

    # self.obs_type: element choices: armdof,handdof,ftpos,lastact,objpose,objvel. separated with "+"
    def compute_required_observations(self):
        obs_end = 0

        if "armdof" in self.obs_type:
            self.obs_buf[:, obs_end: obs_end+self.num_arm_dofs] = unscale(
                self.robot_dof_pos[:, self.arm_dof_indices],
                self.robot_dof_lower_limits[self.arm_dof_indices],
                self.robot_dof_upper_limits[self.arm_dof_indices],
            )
            obs_end += self.num_arm_dofs

        if "handdof" in self.obs_type:
            self.obs_buf[:, obs_end: obs_end+self.num_hand_dofs] = unscale(
                self.robot_dof_pos[:, self.hand_dof_indices],
                self.robot_dof_lower_limits[self.hand_dof_indices],
                self.robot_dof_upper_limits[self.hand_dof_indices],
            )
            obs_end += self.num_hand_dofs

        if "dofvel" in self.obs_type:
            self.obs_buf[:, obs_end: obs_end+self.num_robot_dofs] = (
                self.vel_obs_scale * self.robot_dof_vel
            )
            obs_end += self.num_robot_dofs

        if "ftpos" in self.obs_type: # fingertip positions, N*3
            num_ft_states = self.num_fingers * 3
            self.obs_buf[:, obs_end: obs_end+num_ft_states] = (
                self.fingertip_pos.reshape(self.num_envs, num_ft_states)
            )
            obs_end += num_ft_states
        
        if "objpose" in self.obs_type: # object pose: pos, rot (7)
            self.obs_buf[:, obs_end: obs_end+7] = self.object_pose
            obs_end += 7

        if "objvel" in self.obs_type: # object vel, angvel
            self.obs_buf[:, obs_end: obs_end+3] = self.object_linvel
            self.obs_buf[:, obs_end+3:obs_end+6] = self.object_angvel
            obs_end += 6

        if "lastact" in self.obs_type: # last action
            self.obs_buf[:, obs_end : obs_end+self.num_actions] = self.actions
            obs_end += self.num_actions

        if "onehottask" in self.obs_type: # onehot task label
            self.obs_buf[:, obs_end: obs_end+self.num_tasks] = self.task_label_onehot
            obs_end += self.num_tasks


    def reset_idx(self, env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), self.num_robot_dofs * 2), device=self.device
        )

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_states[
            env_ids
        ].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = (
            self.object_init_states[env_ids, 0:2]
            + self.reset_position_noise * rand_floats[:, 0:2]
        )
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = (
            self.object_init_states[env_ids, self.up_axis_idx]
            + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        )
        object_indices = self.object_indices[env_ids]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices.to(torch.int32)),
            len(object_indices),
        )

        # reset robot
        delta_max = self.robot_dof_upper_limits - self.robot_dof_default_pos
        delta_min = self.robot_dof_lower_limits - self.robot_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (
            rand_floats[:, : self.num_robot_dofs] + 1.0
        )
        pos = self.robot_default_dof_pos + self.reset_dof_pos_noise * rand_delta

        self.robot_dof_pos[env_ids, :] = pos
        self.robot_dof_vel[env_ids, :] = (
            self.robot_dof_default_vel
            + self.reset_dof_vel_noise
            * rand_floats[:, self.num_robot_dofs : 2 * self.num_robot_dofs]
        )
        self.prev_targets[env_ids, : self.num_robot_dofs] = pos
        self.cur_targets[env_ids, : self.num_robot_dofs] = pos

        robot_indices = self.robot_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(robot_indices),
            len(env_ids),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.robot_dof_state),
            gymtorch.unwrap_tensor(robot_indices),
            len(env_ids),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = (
                self.prev_targets[:, self.hand_dof_indices]
                + self.dof_speed_scale * self.dt * self.actions[:, self.hand_dof_indices]
            )
            self.cur_targets[:, self.hand_dof_indices] = tensor_clamp(
                targets,
                self.robot_dof_lower_limits[self.hand_dof_indices],
                self.robot_dof_upper_limits[self.hand_dof_indices],
            )
        else:
            self.cur_targets[:, self.hand_dof_indices] = scale(
                self.actions[:, self.hand_dof_indices],
                self.robot_dof_lower_limits[self.hand_dof_indices],
                self.robot_dof_upper_limits[self.hand_dof_indices],
            )
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

        if self.arm_controller == "qpos":
            if self.use_relative_control:
                targets = (
                    self.prev_targets[:, self.arm_dof_indices]
                    + self.dof_speed_scale * self.dt * self.actions[:, self.arm_dof_indices]
                )
                self.cur_targets[:, self.arm_dof_indices] = tensor_clamp(
                    targets,
                    self.robot_dof_lower_limits[self.arm_dof_indices],
                    self.robot_dof_upper_limits[self.arm_dof_indices],
                )
            else:
                self.cur_targets[:, self.arm_dof_indices] = scale(
                    self.actions[:, self.arm_dof_indices],
                    self.robot_dof_lower_limits[self.arm_dof_indices],
                    self.robot_dof_upper_limits[self.arm_dof_indices],
                )
                self.cur_targets[:, self.arm_dof_indices] = tensor_clamp(
                    self.cur_targets[:, self.arm_dof_indices],
                    self.robot_dof_lower_limits[self.arm_dof_indices],
                    self.robot_dof_upper_limits[self.arm_dof_indices],
                )
        elif self.arm_controller == "ik":  # direct qpos control
            pos_err = (
                self.actions[:, 0:3]
                - self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:3]
            )
            orn_err = orientation_error(
                matrix_to_quaternion(
                    rotation_6d_to_matrix(
                        torch.torch.concat(
                            (self.actions[:, 3:6], self.actions[:, -3:]),
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
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets)
        )

        # TODO apply new forces
        # if self.force_scale > 0.0:
        #     self.rb_forces *= torch.pow(
        #         self.force_decay, self.dt / self.force_decay_interval
        #     )

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_vis:
            # draw axes to debug
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            object_state = self.root_state_tensor[self.object_indices, :]
            for i in range(self.num_envs):
                self._add_debug_lines(
                    self.envs[i], object_state[i, :3], object_state[i, 3:7]
                )
                self._add_debug_lines(
                    self.envs[i], self.palm_center_pos[i], self.palm_rot[i]
                )
                for j in range(self.num_fingers):
                    self._add_debug_lines(
                        self.envs[i],
                        self.fingertip_pos[i][j],
                        self.fingertip_rot[i][j],
                    )

    def _add_debug_lines(self, env, pos, rot, line_len=0.2):
        posx = (
            (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * line_len))
            .cpu()
            .numpy()
        )
        posy = (
            (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * line_len))
            .cpu()
            .numpy()
        )
        posz = (
            (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * line_len))
            .cpu()
            .numpy()
        )

        p0 = pos.cpu().numpy()
        self.gym.add_lines(
            self.viewer,
            env,
            1,
            [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]],
            [0.85, 0.1, 0.1],
        )
        self.gym.add_lines(
            self.viewer,
            env,
            1,
            [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]],
            [0.1, 0.85, 0.1],
        )
        self.gym.add_lines(
            self.viewer,
            env,
            1,
            [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]],
            [0.1, 0.1, 0.85],
        )

    def _control_ik(self, dpose):
        damping = 0.1
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping**2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(
            self.num_envs, 6
        )
        return u


@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


@torch.jit.script
def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


@torch.jit.script
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    out = standardize_quaternion(out)
    return torch.cat([out[..., 3:4], out[..., 0:3]], dim=-1)


@torch.jit.script
def compute_task_rewards(
    reset_buf,
    progress_buf,
    successes,
    current_successes,
    consecutive_successes,
    max_episode_length: float,
    object_pos,
    goal_height: float,
    palm_pos,
    fingertip_pos,
    num_fingers, 
    actions,
    dist_reward_scale: float,
    object_init_states,
    action_penalty_scale: float,
    success_tolerance: float,
    av_factor: float,
):
    info = {}
    goal_object_dist = torch.abs(goal_height - object_pos[:, 2])
    palm_object_dist = torch.norm(object_pos - palm_pos, dim=-1)
    palm_object_dist = torch.where(palm_object_dist >= 0.5, 0.5, palm_object_dist)
    horizontal_offset = torch.norm(
        object_pos[:, 0:2] - object_init_states[:, 0:2], dim=-1
    )

    fingertips_object_dist = torch.zeros_like(goal_object_dist)
    for i in range(fingertip_pos.shape[-2]):
        fingertips_object_dist += torch.norm(
            fingertip_pos[:, i, :] - object_pos, dim=-1
        )
    fingertips_object_dist = torch.where(
        fingertips_object_dist >= 3.0, 3.0, fingertips_object_dist
    )

    flag = (fingertips_object_dist <= 0.12 * num_fingers) + (palm_object_dist <= 0.15)
    # stage 1: approach reward
    hand_approach_reward = torch.zeros_like(goal_object_dist)
    hand_approach_reward = torch.where(
        flag >= 1, 1 * (0.9 - 2 * goal_object_dist), hand_approach_reward
    )
    # stage 2: lift reward
    object_height = object_pos[:, 2]
    hand_up = torch.zeros_like(goal_object_dist)
    hand_up = torch.where(flag >= 1, 1 * (object_height - goal_height), hand_up)

    # stage 3: lift to goal bonus
    bonus = torch.zeros_like(goal_object_dist)
    bonus = torch.where(
        flag >= 1,
        torch.where(
            goal_object_dist <= success_tolerance, 1.0 / (1 + goal_object_dist), bonus
        ),
        bonus,
    )

    # TODO reward shaping
    reward = (
        -dist_reward_scale * fingertips_object_dist
        - 2 * dist_reward_scale * palm_object_dist
        + hand_approach_reward
        + hand_up
        + bonus
        - 0.3 * horizontal_offset
    )

    info["fingertips_object_dist"] = fingertips_object_dist
    info["palm_object_dist"] = palm_object_dist
    info["hand_approach_reward"] = hand_approach_reward
    info["hand_up"] = hand_up
    info["bonus"] = bonus
    info["horizontal_offset"] = horizontal_offset
    info["reward"] = reward
    info["hand_approach_flag"] = flag

    resets = reset_buf.clone()
    resets = torch.where(
        progress_buf >= max_episode_length, torch.ones_like(resets), resets
    )
    resets = torch.where(object_height <= 0.3, torch.ones_like(resets), resets)
    successes = torch.where(
        goal_object_dist <= success_tolerance,
        torch.where(
            flag >= 1, torch.ones_like(successes), successes
        ),
        torch.zeros_like(successes),
    )
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    current_successes = torch.where(resets, successes, current_successes)
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return (
        reward,
        resets,
        progress_buf,
        successes,
        current_successes,
        cons_successes,
        info,
    )
