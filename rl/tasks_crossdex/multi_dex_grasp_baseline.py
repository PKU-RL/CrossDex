# create multiple envs for different hands

import os
import math
import random
import torch
import numpy as np
from torch.nn import functional as F

from isaacgym import gymtorch
from isaacgym import gymapi
import isaacgymenvs
from isaacgymenvs.utils.torch_jit_utils import *
#from isaacgymenvs.tasks.base.vec_task import VecTask
from .vec_task import VecTask

from copy import deepcopy
import gym
from gym import spaces
import sys
from .reward import REWARD_DICT
from .utils import load_robot_randomization_dict, load_object_point_clouds, transform_points
from vision.utils import farthest_point_sample, index_points
sys.path.append('..')
from retargeting.retargeting_nn_utils import EigenRetargetModel


class MultiDexGraspBaseline(VecTask):
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
        print("\nnum obs: {}, num actions: {}\n".format(self.num_obs, self.num_acts))
        if hasattr(self, "num_student_observations"):
            print("num student obs: {}\n".format(self.num_student_observations))

        if self.use_eigengrasp_action:
            self.init_retargeting()

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
            raise NotImplementedError
            _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
            jacobian = gymtorch.wrap_tensor(_jacobian)
            # jacobian entries corresponding to eef
            self.j_eef = jacobian[:, self.arm_eef_index - 1, :, :6]

        
        # Because we assign robots into envs with i_env%N_robot, 
        # we aggregate each N_robot envs together to deal with tensors
        self.agg_num_robot_dofs = sum(self.num_robot_dofs)
        self.agg_num_envs = self.num_envs // len(self.hand_names)

        # create some wrapper tensors for different slices
        self.robot_default_dof_pos = torch.zeros(
            self.agg_num_robot_dofs, dtype=torch.float, device=self.device
        )
        self.robot_default_dof_vel = torch.zeros(
            self.agg_num_robot_dofs, dtype=torch.float, device=self.device
        )
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        #print(self.dof_state, self.dof_state.shape, self.robot_default_dof_pos.shape)
        self.robot_dof_state = self.dof_state.view(self.agg_num_envs, -1, 2)[
            :, : self.agg_num_robot_dofs
        ]
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]
        #print(self.robot_dof_pos.shape, self.robot_dof_vel.shape, self.robot_dof_state.shape)

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.agg_num_envs, -1, 13
        )
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        #print(self.num_bodies, self.rigid_body_states.shape, self.root_state_tensor.shape)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.agg_num_envs
        self.prev_targets = torch.zeros(
            (self.agg_num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.agg_num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        #print(self.num_dofs, self.prev_targets.shape, self.cur_targets.shape)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cont_success_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0

        # forced lift in test stage
        self.test_forced_lift = False
        if "test_forced_lift" in self.cfg["env"] and self.cfg["is_test"]:
            self.test_forced_lift = self.cfg["env"]["test_forced_lift"]["apply"]
            #self.lift_arm_target = self.cfg["env"]["test_forced_lift"]["arm_dof_pos"]
            #self.lift_arm_target = to_torch(self.lift_arm_target, dtype=torch.float32, device=self.device)
            self.n_lift_steps = self.cfg["env"]["test_forced_lift"]["n_lift_steps"]
            self.lift_arm_dof_per_robot = [torch.zeros((self.agg_num_envs,n),dtype=torch.float32).to(self.device) \
                for n in self.num_arm_dofs] #[self.lift_arm_target.unsqueeze(0).repeat(self.agg_num_envs,1) for n in self.num_arm_dofs]
            self.lift_hand_dof_per_robot = [torch.zeros((self.agg_num_envs,n),dtype=torch.float32).to(self.device) \
                for n in self.num_hand_dofs]
            #for a,b in zip(self.lift_arm_dof_per_robot, self.lift_hand_dof_per_robot):
            #    print(a.shape, b.shape)
            self.lift_step_count = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
            self.is_lifting_stage = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
            self.reward_additional_params = dict(
                test_forced_lift = self.test_forced_lift,
                n_lift_steps = self.n_lift_steps,
                lift_step_count = self.lift_step_count,
                is_lifting_stage = self.is_lifting_stage,
            )
        else:
            self.reward_additional_params = dict(test_forced_lift = self.test_forced_lift)

    def init_configs(self, cfg):
        self.cfg = cfg

        self.reward_function = REWARD_DICT[self.cfg["env"]["reward_function"]]

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        #self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        #self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
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
        #self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)
        self.goal_height = self.cfg["env"].get("goalHeight", 0.6)

        self.obs_type = self.cfg["env"]["observationType"]
        self.multi_task = self.cfg["env"]["multiTask"]
        self.multi_task_label = self.cfg["env"]["multiTaskLabel"]

        assert self.arm_controller in ["ik", "qpos"]
        #assert self.obs_type in ["full_no_vel", "full", "full_state", "pos"]

        ## hand-specific settings
        self.hand_names = self.cfg["multi_dex"]["hand_names"]
        self.num_robots = len(self.hand_names)
        self.hand_specific_cfgs = [self.cfg["hand_specific"][hn] for hn in self.hand_names]
        self.palm_offsets = [self.hand_specific_cfgs[i]["palm_offset"] for i in range(len(self.hand_names))]
        self.robot_asset_files = [self.hand_specific_cfgs[i]["robot_asset_file"] for i in range(len(self.hand_names))]
        self.robot_asset_root = self.cfg["env"]["asset"]["robotAssetRoot"]

        self.use_robot_randomization = self.cfg["env"]["randomizeRobot"]
        if self.use_robot_randomization:
            self.robot_asset_root = self.cfg["env"]["randomizeRobotAssetRoot"]
            self.robot_asset_files = sorted(os.listdir(self.robot_asset_root))
            self.robot_asset_prefixs = [self.hand_specific_cfgs[i]["robot_name"] for i in range(len(self.hand_names))]
            print("Use robot randomization, loaded {} urdf files from {}".format(len(self.robot_asset_files),self.robot_asset_root))
            self.robot_randomization_data = load_robot_randomization_dict(os.path.join(self.robot_asset_root,"results.json"))

        # action space
        self.use_eigengrasp_action = self.cfg["multi_dex"]["use_eigengrasp_action"]
        if self.use_eigengrasp_action:
            self.num_eigengrasp_actions = self.cfg["multi_dex"]["n_eigenvecs"]
            self.cfg["env"]["numActions"] = 6 + self.num_eigengrasp_actions
            self.eigengrasp_action_indices = [i for i in range(6,6+self.num_eigengrasp_actions)]
        else:
            self.num_actions = self.cfg["multi_dex"]["max_action_dim"]
            self.cfg["env"]["numActions"] = self.num_actions
        self.position_baseline = self.cfg["multi_dex"]["position_baseline"]
        if self.position_baseline:
            self.num_eigengrasp_actions = 15
            self.cfg["env"]["numActions"] = 6 + self.num_eigengrasp_actions
            self.eigengrasp_action_indices = [i for i in range(6,6+self.num_eigengrasp_actions)]
            self.num_actions = 6 + self.num_eigengrasp_actions
        # obs space
        if self.multi_task:
            if self.multi_task_label=="onehot":
                self.obs_type += "+onehottask"
        self.num_obs_dict = self.cfg["env"]["num_obs_dict"]
        if not self.use_eigengrasp_action or self.position_baseline:
            self.num_obs_dict['lastact'] = self.num_actions
        self.cfg["env"]["numObservations"] = \
            sum([(self.num_obs_dict[i] if i in self.obs_type else 0) for i in self.num_obs_dict]) #self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = 0 #(self.num_obs_dict[self.obs_type] if self.asymmetric_obs else 0)
        #self.cfg["env"]["numActions"] = sum([x["numActions"] for x in self.hand_specific_cfgs])

        if self.arm_controller == "ik":  # use rotation 6D representation
            self.cfg["env"]["numObservations"] += 3
            self.cfg["env"]["numActions"] += 3

        # point clouds
        self.enable_pcl = self.cfg["env"]["enablePointCloud"]
        if self.enable_pcl:
            self.max_points_per_object = self.cfg["env"]["point_cloud"]["max_points_per_object"]
            self.points_per_object = self.cfg["env"]["point_cloud"]["points_per_object"]
            self.n_resample_steps = self.cfg["env"]["point_cloud"]["n_resample_steps"]
            self.student_observation_type = self.cfg["env"]["studentObservationType"]
            self.num_student_observations = sum([(self.num_obs_dict[i] if i in self.student_observation_type else 0) for i in self.num_obs_dict]) 
            self.cfg["env"]["numStudentObservations"] = self.num_student_observations
        
        # baseline
        self.max_num_hand_dof = self.cfg["env"]["num_obs_dict"]["handdof"]
        self.max_num_robot_dof = self.cfg["multi_dex"]["max_action_dim"]

    def init_retargeting(self):
        if "retargeting_type" in self.cfg["multi_dex"]:
            retargeting_type = self.cfg["multi_dex"]["retargeting_type"]
        else:
            retargeting_type = "dexpilot"

        self.retargeting_models, self.retarget2isaacs = [], []
        for i_robot, hand_name in enumerate(self.hand_names):
            retargeting_model = EigenRetargetModel(
                self.cfg["multi_dex"]["dataset"], 
                hand_name, 
                self.cfg["multi_dex"]["add_random_dataset"], 
                retargeting_type,
                self.position_baseline,
                device=self.device
            )
            self.retargeting_models.append(retargeting_model)
            retarget2isaac = [retargeting_model.robot_joint_names.index(n) for n in self.hand_dof_names[i_robot]]
            self.retarget2isaacs.append(retarget2isaac)
            print("\nRobot:", hand_name)
            print("Retarget2isaac dof idx mapping:", retarget2isaac)

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
        

        self.robot_assets, self.robot_dof_props, self.robot_start_poses = self._prepare_robot_asset(
            self.robot_asset_root, self.robot_asset_files
        )
        self.n_assets_per_robot = [len(a) for a in self.robot_assets]
        if self.multi_task: # multitask: add different object assets in different envs
            object_assets = []
            object_start_poses = []
            object_asset_dir = self.cfg["env"]["asset"]["objectAssetDir"]
            object_asset_fn_list = sorted(os.listdir(os.path.join(asset_root, object_asset_dir)))
            self.object_fns = [os.path.join(object_asset_dir, fn) for fn in object_asset_fn_list]
            for fn in self.object_fns:
                object_asset, _, object_start_pose = self._prepare_object_asset(asset_root, fn)
                object_assets.append(object_asset)
                object_start_poses.append(object_start_pose)
            self.task_label_onehot = torch.zeros((num_envs, len(object_assets))).to(self.device)
            self.num_tasks = len(object_assets)
            assert math.gcd(self.num_tasks, self.num_robots)==1 # ensure all <robot, object> are visited
        else: # single task: one object
            object_asset_file = self.cfg["env"]["asset"]["objectAssetFile"]
            object_asset, object_dof_props, object_start_pose = self._prepare_object_asset(asset_root, object_asset_file)
            self.object_fns = [object_asset_file]
        table_asset, table_start_pose, side_panel_asset, side_panel_start_pose = self._prepare_table_asset()

        if "onehotrobot" in self.obs_type:
            self.robot_label_onehot = torch.zeros((num_envs, self.num_obs_dict["onehotrobot"])).to(self.device)
        if "robotranddata" in self.obs_type:
            self.robot_rand_data_obs = torch.zeros((num_envs, self.num_obs_dict["robotranddata"])).to(self.device)
        if self.enable_pcl:
            self.object_pcls = load_object_point_clouds(self.object_fns, asset_root)
            self.obj_pcl_buf = torch.zeros((num_envs, self.max_points_per_object, 3), device=self.device, dtype=torch.float)
        self.object_names = [fn.split('/')[-1].split('.')[0] for fn in self.object_fns]
        self.object_to_env_ids = [[] for n in self.object_names] # maintain env ids for each object class
        print('object names:', self.object_names)

        self.envs = []
        self.robots = []
        self.objects = []
        if self.arm_controller == "ik":
            self.eef_idx = []
        self.robot_indices = []
        self.object_indices = []
        self.robot_start_states = []
        self.object_init_states = []

        assert num_envs % len(self.hand_names) == 0
        for i in range(num_envs):
            i_robot = i%len(self.hand_names)
            i_agg_env = i//len(self.hand_names)
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if "onehotrobot" in self.obs_type:
                self.robot_label_onehot[i] = torch.tensor(self.hand_specific_cfgs[i_robot]["onehot"],dtype=torch.float32).to(self.device)
            if "robotranddata" in self.obs_type:
                self.robot_rand_data_obs[i] = torch.tensor(self.robot_randomization_data[i_robot][i_agg_env%self.n_assets_per_robot[i_robot]],
                    dtype=torch.float32).to(self.device)

            # aggregate size ## TODO: handle different obj shapes for multi-task
            max_agg_bodies = self.num_robot_bodies[i_robot] + self.num_object_bodies + 2
            max_agg_shapes = 256 #self.num_robot_shapes[i_robot] + self.num_object_shapes + 2
            if self.aggregate_mode > 0:
                #print(self.num_robot_bodies[i_robot], self.num_object_bodies, self.num_robot_shapes[i_robot], self.num_object_shapes)
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            #print(self.num_robot_bodies[i_robot],self.num_object_bodies,self.num_robot_shapes[i_robot],self.num_object_shapes)
            
            # create robot actor
            asset = self.robot_assets[i_robot][i_agg_env%self.n_assets_per_robot[i_robot]] # alter randomized assets
            #print(self.n_assets_per_robot, i_agg_env%self.n_assets_per_robot[i_robot])
            robot_actor = self.gym.create_actor(
                env_ptr, asset, self.robot_start_poses[i_robot], self.hand_names[i_robot], i, -1, 0
            )
            self.robot_start_states.append(
                [self.robot_start_poses[i_robot].p.x,self.robot_start_poses[i_robot].p.y,self.robot_start_poses[i_robot].p.z,
                self.robot_start_poses[i_robot].r.x,self.robot_start_poses[i_robot].r.y,self.robot_start_poses[i_robot].r.z,
                self.robot_start_poses[i_robot].r.w,0,0,0,0,0,0,])
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, self.robot_dof_props[i_robot])
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
                self.object_to_env_ids[idx].append(i)
            else:
                self.object_to_env_ids[0].append(i)

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

            # add object point cloud to buffer
            if self.enable_pcl:
                self.obj_pcl_buf[i] = to_torch(self.object_pcls[i % len(self.object_pcls)], dtype=torch.float32, device=self.device)

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
                    env_ptr, robot_actor, self.palms[i_robot], gymapi.DOMAIN_SIM
                )
                self.eef_idx.append(eef_idx)

        self.robot_start_states = to_torch(
            self.robot_start_states, device=self.device
        ).view(num_envs, 13)
        self.object_init_states = to_torch(
            self.object_init_states, device=self.device
        ).view(num_envs, 13)

        self.fingertip_handles = [to_torch(x, dtype=torch.long, device=self.device) for x in self.fingertip_handles]
        self.palm_handles = [to_torch(x, dtype=torch.long, device=self.device) for x in self.palm_handles]
        self.robot_indices = to_torch(
            self.robot_indices, dtype=torch.long, device=self.device
        )
        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device
        )
        if self.arm_controller == "ik":
            self.eef_idx = to_torch(self.eef_idx, dtype=torch.long, device=self.device)
        #print(self.fingertip_handles, self.palm_handles, self.robot_indices, self.object_indices)

    def _prepare_robot_asset(self, asset_root, asset_files):
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

        self.robot_assets, self.robot_dof_props = [[] for hn in self.hand_names],[]
        self.num_robot_bodies, self.num_robot_shapes, self.num_robot_actuators, self.num_robot_dofs = [],[],[],[]
        self.palms, self.fingertips, self.arm_dof_names, self.hand_dof_names, self.robot_dof_names = [],[],[],[],[]
        self.palm_handles, self.fingertip_handles, self.arm_dof_indices, self.hand_dof_indices, self.robot_dof_indices = [],[],[],[],[]
        self.num_fingers, self.num_arm_dofs, self.num_hand_dofs, self.arm_eef_indices = [],[],[],[]
        self.robot_dof_lower_limits,self.robot_dof_upper_limits,self.robot_dof_default_pos,self.robot_dof_default_vel = [],[],[],[]
        self.robot_start_poses = []

        if self.use_robot_randomization:
            asset_files_ = [[] for hn in self.hand_names]
            robot_randomization_data_ = [[] for hn in self.hand_names]
            for fn in asset_files:
                for i, rn in enumerate(self.robot_asset_prefixs):
                    if fn.startswith(rn):
                        asset_files_[i].append(fn)
                        robot_randomization_data_[i].append(self.robot_randomization_data[fn])
            asset_files = asset_files_
            self.robot_randomization_data = robot_randomization_data_
            #print(self.robot_randomization_data)
            #print(asset_files)
        else:
            asset_files = [[fn] for fn in asset_files]


        for i_robot, asset_file in enumerate(asset_files):
            for fn in asset_file:
                print("Loading robot asset:", fn)
                robot_asset = self.gym.load_asset(self.sim, asset_root, fn, asset_options)
                # get arm hand asset info
                num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
                num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
                num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
                num_robot_actuators = num_robot_dofs
                print("self.num_robot_bodies: ", num_robot_bodies)
                print("self.num_robot_shapes: ", num_robot_shapes)
                print("self.num_robot_dofs: ", num_robot_dofs)
                print("self.num_robot_actuators: ", num_robot_actuators)

                # need to set the names according to the robot
                palm = self.hand_specific_cfgs[i_robot]["palm_link"] #"palm_lower"
                fingertips = self.hand_specific_cfgs[i_robot]["fingertips_link"]
                arm_dof_names = ["arm_joint1","arm_joint2","arm_joint3","arm_joint4","arm_joint5","arm_joint6",]
                num_fingers = len(fingertips)
                hand_dof_names = []
                for i in range(num_robot_dofs):
                    joint_name = self.gym.get_asset_dof_name(robot_asset, i)
                    if joint_name not in arm_dof_names:
                        hand_dof_names.append(joint_name)
                robot_dof_names = arm_dof_names + hand_dof_names

                palm_handle = self.gym.find_asset_rigid_body_index(robot_asset, palm)
                fingertip_handles = [
                    self.gym.find_asset_rigid_body_index(robot_asset, fingertip)
                    for fingertip in fingertips
                ]
                if -1 in fingertip_handles or palm_handle==-1:
                    raise Exception("Fingertip names or palm name not found!")
                arm_dof_indices = [
                    self.gym.find_asset_dof_index(robot_asset, name)
                    for name in arm_dof_names
                ]
                hand_dof_indices = [
                    self.gym.find_asset_dof_index(robot_asset, name)
                    for name in hand_dof_names
                ]
                robot_dof_indices = arm_dof_indices + hand_dof_indices
                robot_dof_indices = to_torch(
                    robot_dof_indices, dtype=torch.long, device=self.device
                )
                num_arm_dofs = len(arm_dof_indices)
                num_hand_dofs = len(hand_dof_indices)
                assert num_arm_dofs+num_hand_dofs==num_robot_dofs

                # get eef index
                robot_link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
                arm_eef_index = robot_link_dict[palm]

                # arm hand dof properties
                robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
                robot_dof_lower_limits = []
                robot_dof_upper_limits = []
                robot_dof_default_pos = []
                robot_dof_default_vel = []
                for i in range(num_robot_dofs):
                    robot_dof_lower_limits.append(robot_dof_props["lower"][i])
                    robot_dof_upper_limits.append(robot_dof_props["upper"][i])
                    robot_dof_default_pos.append(0.0)
                    robot_dof_default_vel.append(0.0)
                    if i in arm_dof_indices:
                        robot_dof_props["driveMode"][i] = 1
                        robot_dof_props["stiffness"][i] = 1000
                        robot_dof_props["damping"][i] = 20
                        robot_dof_props["friction"][i] = 0.01
                        robot_dof_props["armature"][i] = 0.001
                    elif i in hand_dof_indices:
                        robot_dof_props["driveMode"][i] = 1
                        robot_dof_props["stiffness"][i] = 3
                        robot_dof_props["damping"][i] = 0.5
                        robot_dof_props["friction"][i] = 0.01
                        robot_dof_props["armature"][i] = 0.001
                    print('DoF {} effort {:.2} stiffness {:.2} damping {:.2} friction {:.2} armature {:.2} limit {:.2}~{:.2}'.format(
                        robot_dof_names[(arm_dof_indices + hand_dof_indices).index(i)], 
                        robot_dof_props['effort'][i], robot_dof_props['stiffness'][i],
                        robot_dof_props['damping'][i], robot_dof_props['friction'][i],
                        robot_dof_props['armature'][i], robot_dof_props['lower'][i], 
                        robot_dof_props['upper'][i]))
                print("\nArm dofs: {}. Hand dofs: {}.\n".format(arm_dof_names, hand_dof_names))

                self.robot_assets[i_robot].append(robot_asset)

            robot_dof_lower_limits = to_torch(robot_dof_lower_limits, device=self.device)
            robot_dof_upper_limits = to_torch(robot_dof_upper_limits, device=self.device)
            robot_dof_default_pos = to_torch(robot_dof_default_pos, device=self.device)
            robot_dof_default_vel = to_torch(robot_dof_default_vel, device=self.device)

            robot_start_pose = gymapi.Transform()
            robot_start_pose.p = gymapi.Vec3(-0.5, 0.0, 0.82)
            robot_start_pose.r = gymapi.Quat.from_euler_zyx(0.0, -np.pi / 2, np.pi)

            self.robot_dof_props.append(robot_dof_props)
            self.num_robot_bodies.append(num_robot_bodies) 
            self.num_robot_shapes.append(num_robot_shapes) 
            self.num_robot_actuators.append(num_robot_actuators) 
            self.num_robot_dofs.append(num_robot_dofs)
            self.palms.append(palm) 
            self.fingertips.append(fingertips) 
            self.arm_dof_names.append(arm_dof_names) 
            self.hand_dof_names.append(hand_dof_names) 
            self.robot_dof_names.append(robot_dof_names)
            self.palm_handles.append(palm_handle) 
            self.fingertip_handles.append(fingertip_handles) 
            assert arm_dof_indices==[0,1,2,3,4,5]
            self.arm_dof_indices.append(arm_dof_indices) 
            self.hand_dof_indices.append(hand_dof_indices) 
            self.robot_dof_indices.append(robot_dof_indices)
            self.num_fingers.append(num_fingers) 
            self.num_arm_dofs.append(num_arm_dofs) 
            self.num_hand_dofs.append(num_hand_dofs) 
            self.arm_eef_indices.append(arm_eef_index)
            self.robot_dof_lower_limits.append(robot_dof_lower_limits) 
            self.robot_dof_upper_limits.append(robot_dof_upper_limits) 
            self.robot_dof_default_pos.append(robot_dof_default_pos) 
            self.robot_dof_default_vel.append(robot_dof_default_vel)
            self.robot_start_poses.append(robot_start_pose)


        self.agg_robot_dof_lower_limits = torch.cat(self.robot_dof_lower_limits)
        self.agg_robot_dof_upper_limits = torch.cat(self.robot_dof_upper_limits)
        #print(self.agg_robot_dof_lower_limits.shape, self.agg_robot_dof_upper_limits.shape)
        self.agg_hand_dof_indices, self.agg_arm_dof_indices, self.agg_robot_dof_indices, \
            self.per_robot_dof_indices, self.per_robot_hand_dof_indices = [], [], [], [], []
        c=0
        for hand_ids, arm_ids, robot_ids, n_dof in zip(self.hand_dof_indices, self.arm_dof_indices, self.robot_dof_indices, self.num_robot_dofs):
            for i in hand_ids:
                self.agg_hand_dof_indices.append(i+c)
            for i in arm_ids:
                self.agg_arm_dof_indices.append(i+c)
            for i in robot_ids:
                self.agg_robot_dof_indices.append(i+c)
            self.per_robot_dof_indices.append(to_torch([i+c for i in robot_ids], dtype=torch.long, device=self.device))
            self.per_robot_hand_dof_indices.append(to_torch([i+c for i in hand_ids], dtype=torch.long, device=self.device))
            c+=n_dof
        self.agg_robot_dof_indices = to_torch(self.agg_robot_dof_indices, dtype=torch.long, device=self.device)
        #print(self.agg_robot_dof_indices, self.agg_hand_dof_indices, self.agg_arm_dof_indices, self.num_robot_dofs)
        self.agg_palm_handles, self.agg_fingertip_handles = [], []
        c=0
        for palm, fingertip, n_body in zip(self.palm_handles, self.fingertip_handles, self.num_robot_bodies):
            self.agg_palm_handles.append(palm+c)
            for i in fingertip:
                self.agg_fingertip_handles.append(i+c)
            c+=(n_body+3)
        self.agg_num_fingers = np.cumsum([0]+self.num_fingers[:-1])

        return self.robot_assets, self.robot_dof_props, self.robot_start_poses

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
            #asset_options.vhacd_params.max_convex_hulls = 40

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



    def reset_idx(self, env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        #rand_floats = torch_rand_float(
        #    -1.0, 1.0, (len(env_ids), self.num_robot_dofs * 2), device=self.device
        #)

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_states[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = (
            self.object_init_states[env_ids, 0:2]
            + self.reset_position_noise *  torch_rand_float(-1.0,1.0,(len(env_ids),2),device=self.device)
        )
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = (
            self.object_init_states[env_ids, self.up_axis_idx]
            + self.reset_position_noise * torch_rand_float(-1.0,1.0,(len(env_ids),1),device=self.device).reshape(-1)
        )
        object_indices = self.object_indices[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices.to(torch.int32)),
            len(object_indices),
        )

        # reset robot
        #print(self.robot_dof_state.shape, self.prev_targets.shape, self.robot_indices, self.num_dofs)
        delta_max = self.agg_robot_dof_upper_limits - self.robot_default_dof_pos
        delta_min = self.agg_robot_dof_lower_limits - self.robot_default_dof_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (
            torch_rand_float(-1.0,1.0,(len(env_ids),self.num_dofs),device=self.device) + 1.0
        )
        pos = self.robot_default_dof_pos + self.reset_dof_pos_noise * rand_delta # [num_reset_envs, agg_num_dofs]
        vel = self.robot_default_dof_vel + self.reset_dof_vel_noise * torch_rand_float(-1.0,1.0,(len(env_ids),self.num_dofs),device=self.device)

        for i, idx in enumerate(env_ids):
            i_robot = idx.item()%self.num_robots
            i_agg_env = idx.item()//self.num_robots
            self.robot_dof_pos[i_agg_env, self.per_robot_dof_indices[i_robot]] = pos[i, self.per_robot_dof_indices[i_robot]]
            self.robot_dof_vel[i_agg_env, self.per_robot_dof_indices[i_robot]] = vel[i, self.per_robot_dof_indices[i_robot]]
            self.prev_targets[i_agg_env, self.per_robot_dof_indices[i_robot]] = pos[i, self.per_robot_dof_indices[i_robot]]
            self.cur_targets[i_agg_env, self.per_robot_dof_indices[i_robot]] = pos[i, self.per_robot_dof_indices[i_robot]]

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
        self.cont_success_steps[env_ids] = 0
        if self.test_forced_lift:
            self.lift_step_count[env_ids] = 0
            self.is_lifting_stage[env_ids] = 0


    def pre_physics_step(self, actions):
        # reset when done
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            #print(env_ids)
            self.reset_idx(env_ids)

        #print("actions:", actions, actions.shape)
        self.actions = actions.clone().to(self.device)
        
        actions_per_robot = [self.actions[i::self.num_robots] for i in range(self.num_robots)]
        dof_actions_per_robot = [torch.zeros((self.agg_num_envs,n),dtype=torch.float32).to(self.device) \
            for n in self.num_robot_dofs]
        
        for i in range(self.num_robots):
            # prepare arm dof actions
            dof_actions_per_robot[i][:,self.arm_dof_indices[i]] = actions_per_robot[i][:,self.arm_dof_indices[i]]
            # prepare hand dof actions
            if self.use_eigengrasp_action:
                # retarget eigengrasp actions to robot dof targets
                actions_eigengrasp = scale(actions_per_robot[i][:,self.eigengrasp_action_indices], 
                    self.retargeting_models[i].min_values, self.retargeting_models[i].max_values) # min_value~max_value
                actions_hand_dof = self.retargeting_models[i].retarget(actions_eigengrasp)[:, self.retarget2isaacs[i]]
                dof_actions_per_robot[i][:,self.hand_dof_indices[i]] = actions_hand_dof
            else:
                dof_actions_per_robot[i][:,self.hand_dof_indices[i]] = scale(
                    actions_per_robot[i][:,self.hand_dof_indices[i]],
                    self.agg_robot_dof_lower_limits[self.per_robot_hand_dof_indices[i]],
                    self.agg_robot_dof_upper_limits[self.per_robot_hand_dof_indices[i]],
                )


        agg_actions = torch.cat(dof_actions_per_robot, dim=-1)

        # hand: always use absolute pos control
        self.cur_targets[:, self.agg_hand_dof_indices] = tensor_clamp( # hand_dofs in agg_actions are already in the dofs raw range
            agg_actions[:, self.agg_hand_dof_indices],
            self.agg_robot_dof_lower_limits[self.agg_hand_dof_indices],
            self.agg_robot_dof_upper_limits[self.agg_hand_dof_indices],
        )
        #print(self.cur_targets[:,self.agg_hand_dof_indices]-agg_actions[:,self.agg_hand_dof_indices])
        self.cur_targets[:, self.agg_hand_dof_indices] = (
            self.act_moving_average * self.cur_targets[:, self.agg_hand_dof_indices]
            + (1.0 - self.act_moving_average)
            * self.prev_targets[:, self.agg_hand_dof_indices]
        )
        self.cur_targets[:, self.agg_hand_dof_indices] = tensor_clamp(
            self.cur_targets[:, self.agg_hand_dof_indices],
            self.agg_robot_dof_lower_limits[self.agg_hand_dof_indices],
            self.agg_robot_dof_upper_limits[self.agg_hand_dof_indices],
        )

        # arm: different control methods
        if self.arm_controller == "qpos":
            if self.use_relative_control:
                targets = (
                    self.prev_targets[:, self.agg_arm_dof_indices]
                    + self.dof_speed_scale * self.dt * agg_actions[:, self.agg_arm_dof_indices]
                )
                self.cur_targets[:, self.agg_arm_dof_indices] = tensor_clamp(
                    targets,
                    self.agg_robot_dof_lower_limits[self.agg_arm_dof_indices],
                    self.agg_robot_dof_upper_limits[self.agg_arm_dof_indices],
                )
            else:
                self.cur_targets[:, self.agg_arm_dof_indices] = scale(
                    agg_actions[:, self.agg_arm_dof_indices],
                    self.agg_robot_dof_lower_limits[self.agg_arm_dof_indices],
                    self.agg_robot_dof_upper_limits[self.agg_arm_dof_indices],
                )
                self.cur_targets[:, self.agg_arm_dof_indices] = (
                    self.act_moving_average * self.cur_targets[:, self.agg_arm_dof_indices]
                    + (1.0 - self.act_moving_average)
                    * self.prev_targets[:, self.agg_arm_dof_indices]
                ) # moving average
                self.cur_targets[:, self.agg_arm_dof_indices] = tensor_clamp(
                    self.cur_targets[:, self.agg_arm_dof_indices],
                    self.agg_robot_dof_lower_limits[self.agg_arm_dof_indices],
                    self.agg_robot_dof_upper_limits[self.agg_arm_dof_indices],
                )
        elif self.arm_controller == "ik":  # direct qpos control
            raise NotImplementedError

        ### 9-8: forced lift test 
        if self.test_forced_lift:
            for i in range(self.num_robots):
                agg_arm_indices = self.per_robot_dof_indices[i][self.arm_dof_indices[i]]
                agg_hand_indices = self.per_robot_dof_indices[i][self.hand_dof_indices[i]]
                ## for all envs not in the forced lifting stage, save the current dof target
                # save the current hand dof target
                self.lift_hand_dof_per_robot[i] = torch.where(
                    (self.is_lifting_stage[i::self.num_robots]>0)[:,None], # check stage for robot i. broadcast to (agg_num_envs,1)
                    self.lift_hand_dof_per_robot[i], # forced lift stage: keep the hand action unchanged
                    self.cur_targets[:, agg_hand_indices] # not: save the recent hand action
                ) 
                # save the current arm dof target
                self.lift_arm_dof_per_robot[i] = torch.where(
                    (self.is_lifting_stage[i::self.num_robots]>0)[:,None], # check stage for robot i. broadcast to (agg_num_envs,1)
                    self.lift_arm_dof_per_robot[i], # forced lift stage: keep the arm action unchanged
                    self.robot_dof_pos[:, agg_arm_indices] # not: save the recent arm dof pos as action
                ) 

                ## for all envs in the forced lifting stage, set the arm-hand action
                # set arm action
                self.cur_targets[:, agg_arm_indices] = torch.where(
                    (self.is_lifting_stage[i::self.num_robots]>0)[:,None], # check stage for robot i. broadcast to (agg_num_envs,1)
                    self.lift_arm_dof_per_robot[i], # forced lift stage: set the arm action
                    self.cur_targets[:, agg_arm_indices] # not: use the policy action
                )
                # set hand action
                self.cur_targets[:, agg_hand_indices] = torch.where(
                    (self.is_lifting_stage[i::self.num_robots]>0)[:,None], # check stage for robot i. broadcast to (agg_num_envs,1)
                    self.lift_hand_dof_per_robot[i], # forced lift stage: set the hand action
                    self.cur_targets[:, agg_hand_indices] # not: use the policy action
                )

        self.prev_targets[:, self.agg_robot_dof_indices] = self.cur_targets[
            :, self.agg_robot_dof_indices
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
            for i in range(self.agg_num_envs):
                for j in range(self.num_robots):
                    i_env = self.num_robots*i+j
                    self._add_debug_lines(
                        self.envs[i_env], object_state[i_env, :3], object_state[i_env, 3:7]
                    )
                    self._add_debug_lines(
                        self.envs[i_env], self.palm_center_pos[i,j], self.palm_rot[i,j]
                    )
                    offset = self.agg_num_fingers[j]
                    for k in range(self.num_fingers[j]):
                        self._add_debug_lines(
                            self.envs[i_env],
                            self.fingertip_pos[i, offset+k],
                            self.fingertip_rot[i, offset+k],
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

        self.palm_state = self.rigid_body_states[:, self.agg_palm_handles][..., :13]
        self.palm_pos = self.palm_state[..., :3]
        self.palm_rot = self.palm_state[..., 3:7]
        #print(self.palm_pos.shape)
        #print(to_torch(self.palm_offsets).repeat(self.agg_num_envs, 1).reshape(self.agg_num_envs,self.num_robots,-1), self.palm_rot.shape)
        self.palm_center_pos = self.palm_pos + quat_apply(
            self.palm_rot, to_torch(self.palm_offsets).repeat(self.agg_num_envs, 1).reshape(self.agg_num_envs,self.num_robots,-1)
        )
        #print(self.palm_center_pos, self.palm_center_pos.shape)

        self.fingertip_state = self.rigid_body_states[:, self.agg_fingertip_handles][..., :13]
        #self.fingertip_pose = self.fingertip_state[..., :7]
        self.fingertip_pos = self.fingertip_state[..., :3]
        self.fingertip_rot = self.fingertip_state[..., 3:7]

        self.compute_required_observations(self.obs_buf, self.obs_type, self.num_observations)
        if self.enable_pcl:
            self.compute_required_observations(self.student_obs_buf, self.student_observation_type, self.num_student_observations)


    # compute obs for either priviliged policy or student vision-based policy
    # depends on the selected obs_buf and obs_type
    def compute_required_observations(self, obs_buf, obs_type, num_obs):
        obs_end = 0

        if "armdof" in obs_type:
            obs_buf[:, obs_end: obs_end+self.num_arm_dofs[0]] = unscale(
                self.robot_dof_pos[:, self.agg_arm_dof_indices],
                self.agg_robot_dof_lower_limits[self.agg_arm_dof_indices],
                self.agg_robot_dof_upper_limits[self.agg_arm_dof_indices],
            ).reshape(self.num_envs,self.num_arm_dofs[0])
            obs_end += self.num_arm_dofs[0]

        if "keypts" in obs_type: # keypoints: palm pos + 4 fingertip pos, 5*3
            num_keypts_states = (1+4)*3
            keypos = []
            for i in range(self.num_robots):
                offset = self.agg_num_fingers[i]
                keypos.append(self.fingertip_pos[:, offset:offset+4].reshape(self.agg_num_envs,1,4,3))
            keypos = torch.cat(keypos, dim=1)
            keypos = torch.cat([self.palm_center_pos.reshape(self.agg_num_envs,self.num_robots,1,3), 
                keypos], dim=2)
            obs_buf[:, obs_end: obs_end+num_keypts_states] = keypos.reshape(self.num_envs, num_keypts_states)
            obs_end += num_keypts_states
        
        if "objpose" in obs_type: # object pose: pos, rot (7)
            obs_buf[:, obs_end: obs_end+7] = self.object_pose
            obs_end += 7

        if "lastact" in obs_type: # last action
            obs_buf[:, obs_end : obs_end+self.num_actions] = self.actions
            obs_end += self.num_actions

        if "onehotrobot" in obs_type:
            dim_onehotrobot = self.num_obs_dict["onehotrobot"]
            obs_buf[:, obs_end: obs_end+dim_onehotrobot] = self.robot_label_onehot
            obs_end += dim_onehotrobot

        if "robotranddata" in obs_type:
            dim_robotranddata = self.num_obs_dict["robotranddata"]
            obs_buf[:, obs_end: obs_end+dim_robotranddata] = self.robot_rand_data_obs
            obs_end += dim_robotranddata
            #print(self.robot_rand_data_obs)

        if "onehottask" in obs_type: # onehot task label
            obs_buf[:, obs_end: obs_end+self.num_tasks] = self.task_label_onehot
            obs_end += self.num_tasks

        if "objpcl" in obs_type: # object point cloud
            self.sampled_pcl = self.transform_obj_pcl_2_world()
            #self.sampled_pcl_abs = sampled_pcl.clone()
            obs_buf[:, obs_end: obs_end+self.points_per_object*3] = self.sampled_pcl.reshape(self.num_envs,-1)
            obs_end += self.points_per_object*3
        
        # baseline
        if 'handdof' in obs_type: # hand dof padded to 22-dim
            agg_hand_dof_pos = torch.zeros((self.agg_num_envs, self.num_robots, self.max_num_hand_dof), dtype=torch.float32).to(self.device)
            for i in range(self.num_robots):
                agg_hand_dof_pos[:,i,0:self.num_hand_dofs[i]] = unscale(
                    self.robot_dof_pos[:, self.per_robot_hand_dof_indices[i]],
                    self.agg_robot_dof_lower_limits[self.per_robot_hand_dof_indices[i]],
                    self.agg_robot_dof_upper_limits[self.per_robot_hand_dof_indices[i]],
                )
            obs_buf[:, obs_end:obs_end+self.max_num_hand_dof] = agg_hand_dof_pos.reshape(-1,self.max_num_hand_dof)
            obs_end += self.max_num_hand_dof

        assert obs_end==num_obs


    # sample object point cloud & transform within the world coordinate
    def transform_obj_pcl_2_world(self):
        o2w_pos = self.object_pos[:,:3].clone()
        o2w_pos = o2w_pos.resize(o2w_pos.size(0),1,3)
        o2w_quat = self.object_rot.clone()
        o2w_quat = o2w_quat.resize(o2w_quat.size(0),1,4)

        # resample point cloud per k steps
        if self.control_steps % self.n_resample_steps == 0:
            self.sampled_point_idxs = farthest_point_sample(self.obj_pcl_buf, self.points_per_object, self.device)
        sampled_pcl = index_points(self.obj_pcl_buf, self.sampled_point_idxs, self.device)
        append_pos = torch.zeros([sampled_pcl.size(0),self.points_per_object,1]).to(self.device)
        sampled_pcl = torch.cat([sampled_pcl,append_pos],2)

        o2w_quat = o2w_quat.expand_as(sampled_pcl)
        sampled_pcl = transform_points(o2w_quat, sampled_pcl)
        o2w_pos = o2w_pos.expand_as(sampled_pcl)
        sampled_pcl = sampled_pcl + o2w_pos

        return sampled_pcl


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


    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.current_successes[:],
            self.consecutive_successes[:],
            self.cont_success_steps[:],
            reward_info,
        ) = self.reward_function(
            reset_buf = self.reset_buf,
            progress_buf = self.progress_buf,
            successes = self.successes,
            current_successes = self.current_successes,
            consecutive_successes = self.consecutive_successes,
            max_episode_length = self.max_episode_length,
            object_pos = self.object_pos,
            goal_height = self.goal_height,
            palm_pos = self.palm_center_pos,
            fingertip_pos = self.fingertip_pos,
            num_fingers = self.num_fingers,
            agg_num_fingers = self.agg_num_fingers,
            num_envs = self.num_envs,
            agg_num_envs = self.agg_num_envs,
            num_robots = self.num_robots,
            actions = self.actions,
            #self.dist_reward_scale,
            object_init_states = self.object_init_states,
            #self.action_penalty_scale,
            success_tolerance = self.success_tolerance,
            av_factor = self.av_factor,
            cont_success_steps = self.cont_success_steps,
            ### 9-8: forced lift test
            **self.reward_additional_params
        )

        self.extras.update(reward_info)
        self.extras["successes"] = self.successes
        self.extras["current_successes"] = self.current_successes
        self.extras["consecutive_successes"] = self.consecutive_successes
        self.extras["cont_success_steps"] = self.cont_success_steps

        # if self.print_success_stat:
        #     self.total_resets = self.total_resets + self.reset_buf.sum()
        #     direct_average_successes = self.total_successes + self.successes.sum()
        #     self.total_successes = (
        #         self.total_successes + (self.successes * self.reset_buf).sum()
        #     )
        #     # The direct average shows the overall result more quickly, but slightly undershoots long term policy performance.
        #     print(
        #         "Direct average consecutive successes = {:.1f}".format(
        #             direct_average_successes / (self.total_resets + self.num_envs)
        #         )
        #     )
        #     if self.total_resets > 0:
        #         print(
        #             "Post-Reset average consecutive successes = {:.1f}".format(
        #                 self.total_successes / self.total_resets
        #             )
        #         )

