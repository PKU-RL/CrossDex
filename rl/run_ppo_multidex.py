'''
PPO for crossdex grasping, using eigengrasp action space
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
        if not cfg.task.env.multiTask:
            run_name = f"{env.object_names[0]}_{time_str}"
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
    elif train_param.name == "ppo_finetune":
        from algo import ppo
        runner = ppo.PPOFinetune(
            vec_env=env,
            actor_critic_class=ppo.ActorCritic,
            train_param=train_param,
            log_dir=log_dir,
            apply_reset=False,
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
    if cfg.test:
        load_cfg_path = os.path.join(cfg.path, "config.json")
        with open(load_cfg_path, "r") as f:
            cfg_load = json.load(f)
        cfg_load = OmegaConf.create(cfg_load)
        cfg_load.test = True
        cfg_load.headless = cfg.headless
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

    if "debug" in cfg: # debug the environment
        for i in range(100000):
            action = torch.zeros((env.num_envs, env.num_acts))
            #action[:, 6:env.num_acts] = torch.tensor(np.random.uniform(-1,1,env.num_acts-6))
            #action[:, env.agg_arm_dof_indices] = np.random.uniform(-0.1, 0.1)
            if (i//100)%2==0:
                action[:, 6:] = 1 #torch.rand(9)*2-1
            else:
                action[:, 6:] = -1
            _, _, _, _ = env.step(action)
            #if i % 100 == 0:
            #    print(f"state: {env.rigid_body_states[:, 7, 0:3]}")
    
    elif "save_data" in cfg: # save a trajectory
        import pickle
        assert cfg.test
        runner = build_runner(cfg, env)
        runner.vec_env.random_time = False
        current_obs = runner.vec_env.reset()["obs"]
        current_states = runner.vec_env.get_state()
        data = {'arm_states': [], 'arm_actions': [], 
            'hand_states': [], 'hand_actions': []}
        for i in range(60): #(runner.vec_env.max_episode_length):
            with torch.no_grad():
                actions = runner.actor_critic.act_inference(current_obs)
                next_obs_dict, rews, dones, infos = runner.vec_env.step(actions)
                next_obs = next_obs_dict["obs"]
                current_obs.copy_(next_obs)
                data['arm_states'].append(runner.vec_env.robot_dof_state[:,runner.vec_env.arm_dof_indices,0][0].cpu().numpy())
                data['arm_actions'].append(runner.vec_env.cur_targets[:,runner.vec_env.arm_dof_indices][0].cpu().numpy())
                data['hand_states'].append(runner.vec_env.robot_dof_state[:,runner.vec_env.hand_dof_indices,0][0].cpu().numpy())
                data['hand_actions'].append(runner.vec_env.cur_targets[:,runner.vec_env.hand_dof_indices][0].cpu().numpy())
                print(data['arm_states'][-1], data['arm_actions'][-1],
                    data['hand_states'][-1], data['hand_actions'][-1])
                time.sleep(0.1)
        with open('data.pkl', 'wb') as f:
            pickle.dump(data, f)

    else:
        #cfg.task_name += 'Retarget'
        runner = build_runner(cfg, env)
        runner.run()


if __name__ == "__main__":
    main()
