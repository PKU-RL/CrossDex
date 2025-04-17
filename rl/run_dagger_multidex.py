'''
distillation into a vision-based policy
'''

import os
import json
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import gym
from isaacgym import gymapi
from isaacgym import gymutil
import isaacgymenvs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.torch_jit_utils import *
import tasks_crossdex
import glob
import re
import time

def build_runner(cfg, env):
    train_param = cfg.train.params
    is_testing = cfg.test  # train_param["test"]
    ckpt_path = cfg.checkpoint

    if not is_testing:
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{cfg.task_name}_{time_str}"
        log_dir = os.path.join(train_param.log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=4)
    else:
        log_dir = None

    if train_param.name == "ppo":
        from algo import ppo
        runner = ppo.PPO(
            vec_env=env,
            actor_critic_class=ppo.ActorCritic,
            train_param=train_param,
            log_dir=log_dir,
            apply_reset=False,
            is_vision=False,
        )
    elif train_param.name == "dagger":
        from algo import dagger
        runner = dagger.DAGGER(
            vec_env=env,
            actor_critic_class=dagger.ActorCriticDagger,
            train_param=train_param,
            log_dir=log_dir,
            apply_reset=False,
            pred_arm_dof_pos=train_param.pred_arm_dof_pos,
        )
    else:
        raise ValueError("Unrecognized algorithm!")

    if is_testing and ckpt_path != "":
        print(f"Loading model from {ckpt_path}")
        runner.test(ckpt_path)
    elif ckpt_path != "":
        print(f"\nWarning: load pre-trained policy. Loading model from {ckpt_path}\n")
        runner.load(ckpt_path)

    return runner


@hydra.main(version_base="1.3", config_path="./tasks_crossdex", config_name="config")
def main(cfg: DictConfig) -> None:
    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank
    )

    ### use 'test=True num_envs=xxx +path=runs/xxx' for test. config and checkpoint will be auto loaded.
    if cfg.test and "path" in cfg: #(not cfg.test_expert):
        load_cfg_path = os.path.join(cfg.path, "config.json")
        with open(load_cfg_path, "r") as f:
            cfg_load = json.load(f)
        cfg_load = OmegaConf.create(cfg_load)
        cfg_load.task.name = cfg_load.task_name = cfg.task_name
        cfg_load.test = True
        cfg_load.test_expert = cfg.test_expert
        cfg_load.headless = cfg.headless
        cfg_load.task.task.randomize = cfg.task.task.randomize
        #cfg_load.capture_video = cfg.capture_video
        checkpoints = glob.glob(cfg.path+'/*.pt')
        checkpoints = sorted(checkpoints, key=lambda path: int(re.search(r'model_(\d+)\.pt', path).group(1)))
        cfg_load.checkpoint = checkpoints[-1] # use last rl checkpoint
        if "ckpt" in cfg:
            cfg_load.checkpoint = os.path.join(cfg.path, cfg.ckpt) # use the specified checkpoint
        cfg_load.num_envs = cfg_load.task.env.numEnvs = cfg.num_envs
        if "save_data" in cfg:
            cfg_load.save_data = cfg.save_data
        if "test_hands" in cfg:
            cfg_load.task.multi_dex.hand_names = cfg.test_hands
        if "multi_task" in cfg:
            cfg_load.task.env.multiTask = True
            cfg_load.task.env.multiTaskLabel = ""
        if "randomize_robot" in cfg:
            cfg_load.task.env.randomizeRobot = cfg.randomize_robot
        if "randomize_robot_asset_root" in cfg:
            cfg_load.task.env.randomizeRobotAssetRoot = cfg.randomize_robot_asset_root
        if "object_dir" in cfg: # change a test set of objects
            cfg_load.task.env.asset.objectAssetDir = cfg.object_dir
        if "single_object" in cfg: # test on a single object
            cfg_load.task.env.multiTask = False
            cfg_load.task.env.asset.objectAssetFile = cfg.single_object
        if "screenshot" in cfg: # take some pictures
            cfg_load.task.screenshot = cfg.screenshot
        if "no_robot_reset_noise" in cfg:
            cfg_load.task.env.resetDofPosRandomInterval = 0.0
            cfg_load.task.env.resetDofVelRandomInterval = 0.0
        cfg = cfg_load

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{cfg.task_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env = create_isaacgym_env()

    if "screenshot" in cfg.task:
        # look at 1st table
        if cfg.task.screenshot==1:
            cam_pos = gymapi.Vec3(0.6, -0.4, 0.7)
            cam_target = gymapi.Vec3(0, 0, 0.7)
        elif cfg.task.screenshot==2:
            cam_pos = gymapi.Vec3(0.9, -0.6, 0.7)
            cam_target = gymapi.Vec3(0, 0, 0.7)
        elif cfg.task.screenshot==3:
            cam_pos = gymapi.Vec3(3.5, 1.2, 3)
            cam_target = gymapi.Vec3(1.2, 1.2, 0)
        env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)

    if "debug" in cfg: # debug the environment
        # look at 1st table
        cam_pos = gymapi.Vec3(2, 0.4, 1.5)
        cam_target = gymapi.Vec3(0, -0.4, 0.3)
        env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)
        # load a teacher policy to execute
        if "test_teacher" in cfg or "test_student" in cfg:
            load_cfg_path = os.path.join(cfg.path, "config.json")
            with open(load_cfg_path, "r") as f:
                cfg_load = json.load(f)
            cfg_load = OmegaConf.create(cfg_load)
            cfg_load.test = True
            checkpoints = glob.glob(cfg.path+'/*.pt')
            checkpoints = sorted(checkpoints, key=lambda path: int(re.search(r'model_(\d+)\.pt', path).group(1)))
            cfg_load.checkpoint = checkpoints[-1] # use last rl checkpoint
            runner = build_runner(cfg_load, env)

        from vision.utils import vis_pointcloud_realtime
        import queue
        pcl_queue = queue.Queue(1)
        vis_pointcloud_realtime(pcl_queue, coord_len=1)

        obs = env.reset()
        for i in range(10000):
            if "test_teacher" in cfg:
                action = runner.actor_critic.act_inference(obs["obs"])
            elif "test_student" in cfg:
                action = runner.actor_critic.act_inference(obs["student_obs"])
            else:
                action = torch.zeros((env.num_envs, env.num_acts))
                if (i//100)%2==0:
                    action[:, 6:15] = 1 #torch.rand(9)*2-1
                else:
                    action[:, 6:15] = 0
            obs, _, _, _ = env.step(action)
            pcl = env.sampled_pcl[0].cpu().numpy()
            if pcl_queue.full():
                pcl_queue.get()
            pcl_queue.put(pcl)
            time.sleep(0.05)

    else:
        runner = build_runner(cfg, env)
        runner.run()


if __name__ == "__main__":
    main()
