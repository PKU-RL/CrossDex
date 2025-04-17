import torch

'''
hand approach object + object lift 
problems in success criterion: 
1. only account for success when episode time out
2. reward in the final stage is not consistent with success
3. sometimes regard some none-grasping behaviors as success (e.g. balancing the object with palm)
'''
def reward_v0(
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
    agg_num_fingers,
    num_envs, 
    agg_num_envs,
    num_robots,
    actions,
    #dist_reward_scale: float,
    object_init_states,
    #action_penalty_scale: float,
    success_tolerance: float,
    av_factor: float,
    cont_success_steps,
    **kwargs,
):
    info = {}
    
    goal_object_dist = torch.abs(goal_height - object_pos[:, 2])
    palm_object_dist = torch.norm(object_pos - palm_pos.reshape(num_envs,-1), dim=-1)
    palm_object_dist = torch.where(palm_object_dist >= 0.5, 0.5, palm_object_dist)
    horizontal_offset = torch.norm(
        object_pos[:, 0:2] - object_init_states[:, 0:2], dim=-1
    )

    fingertips_object_dist = [] #torch.zeros_like(goal_object_dist)
    for i in range(num_robots):
        offset = agg_num_fingers[i]
        n_f = num_fingers[i]
        dists = torch.norm(fingertip_pos[:,offset:offset+n_f] - object_pos.view(agg_num_envs,-1,3)[:,i:i+1].repeat(1,n_f,1), dim=-1) # [agg_env,n_finger]
        fingertips_object_dist.append(torch.mean(dists, dim=-1).view(-1,1)) # n_robots * [agg_env,1]. avg distance over fingers
    fingertips_object_dist = torch.cat(fingertips_object_dist, dim=1).reshape(-1) # [num_envs]
    fingertips_object_dist = torch.where(fingertips_object_dist >= 0.5, 0.5, fingertips_object_dist)

    flag = (fingertips_object_dist <= 0.12) + (palm_object_dist <= 0.15)

    # stage 1: lift reward 1
    hand_approach_reward = torch.zeros_like(goal_object_dist)
    hand_approach_reward = torch.where(flag >= 1, (0.9 - 2 * goal_object_dist), hand_approach_reward)
    # stage 2: lift reward 2
    object_height = object_pos[:, 2]
    hand_up = torch.zeros_like(goal_object_dist)
    hand_up = torch.where(flag >= 1, 1 * (object_height - goal_height), hand_up)
    # stage 3: lift to goal bonus
    bonus = torch.zeros_like(goal_object_dist)
    bonus = torch.where(flag >= 1, 
        torch.where(goal_object_dist <= success_tolerance, 1.0 / (1 + goal_object_dist), bonus), 
        bonus)

    # reward shaping
    reward = (
        - 2.0 * fingertips_object_dist
        - 1.0 * palm_object_dist
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
    current_successes = torch.where(resets > 0, successes, current_successes)
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
        cont_success_steps,
        info,
    )


'''
- Revise success criterion: keeping the object at the goal height for K steps
'''
def reward_v1(
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
    agg_num_fingers,
    num_envs, 
    agg_num_envs,
    num_robots,
    actions,
    object_init_states,
    success_tolerance: float,
    av_factor: float,
    cont_success_steps,
    K_cont_success_steps = 60,
    **kwargs,
):
    info = {}
    
    goal_object_dist = torch.abs(goal_height - object_pos[:, 2])
    palm_object_dist = torch.norm(object_pos - palm_pos.reshape(num_envs,-1), dim=-1)
    palm_object_dist = torch.where(palm_object_dist >= 0.5, 0.5, palm_object_dist)
    horizontal_offset = torch.norm(
        object_pos[:, 0:2] - object_init_states[:, 0:2], dim=-1
    )

    fingertips_object_dist = [] #torch.zeros_like(goal_object_dist)
    for i in range(num_robots):
        offset = agg_num_fingers[i]
        n_f = num_fingers[i]
        dists = torch.norm(fingertip_pos[:,offset:offset+n_f] - object_pos.view(agg_num_envs,-1,3)[:,i:i+1].repeat(1,n_f,1), dim=-1) # [agg_env,n_finger]
        fingertips_object_dist.append(torch.mean(dists, dim=-1).view(-1,1)) # n_robots * [agg_env,1]. avg distance over fingers
    fingertips_object_dist = torch.cat(fingertips_object_dist, dim=1).reshape(-1) # [num_envs]
    fingertips_object_dist = torch.where(fingertips_object_dist >= 0.5, 0.5, fingertips_object_dist)

    flag = (fingertips_object_dist <= 0.12) + (palm_object_dist <= 0.15)

    # stage 1: lift reward 1
    object_goal_reward = torch.zeros_like(goal_object_dist)
    object_goal_reward = torch.where(flag >= 1, (0.9 - 2 * goal_object_dist), object_goal_reward)
    # stage 1: lift reward 2
    object_height = object_pos[:, 2]
    object_up = torch.zeros_like(goal_object_dist)
    object_up = torch.where(flag >= 1, 1 * (object_height - goal_height), object_up)
    
    # stage 2: lift to goal bonus
    bonus = torch.zeros_like(goal_object_dist)
    bonus = torch.where(flag >= 1, 
        torch.where(goal_object_dist <= success_tolerance, 1.0 / (1 + goal_object_dist), bonus), 
        bonus)

    # stage 3: success
    successes = torch.where(
        goal_object_dist <= success_tolerance,
        torch.where(
            flag >= 1, torch.ones_like(successes), successes
        ),
        torch.zeros_like(successes),
    )
    cont_success_steps = torch.where(successes>0, cont_success_steps+1, cont_success_steps)
    task_successes = torch.zeros_like(goal_object_dist)
    task_successes = torch.where(cont_success_steps>=K_cont_success_steps, torch.ones_like(task_successes), task_successes)

    resets = reset_buf.clone()
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets) # timeout
    resets = torch.where(object_height <= 0.3, torch.ones_like(resets), resets) # object fall
    resets = torch.where(cont_success_steps>=K_cont_success_steps, torch.ones_like(resets), resets) # task success
    
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(task_successes * resets.float())
    current_successes = torch.where(resets > 0, task_successes, current_successes)
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )


    # reward shaping & information log
    reward = (
        - 2.0 * fingertips_object_dist
        - 1.0 * palm_object_dist
        + object_goal_reward
        + object_up
        + bonus
        + 200 * task_successes
        - 0.3 * horizontal_offset
    )

    info["fingertips_object_dist"] = fingertips_object_dist
    info["palm_object_dist"] = palm_object_dist
    info["object_goal_reward"] = object_goal_reward
    info["object_up"] = object_up
    info["bonus"] = bonus
    info["success_reward"] = task_successes
    info["horizontal_offset"] = horizontal_offset
    info["reward"] = reward
    info["hand_approach_flag"] = flag

    return (
        reward,
        resets,
        progress_buf,
        successes,
        current_successes,
        cons_successes,
        cont_success_steps,
        info,
    )


'''
highlight horizontal_offset
- Success criterion: keep K steps, consider both x,y,z
'''
def reward_v2(
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
    agg_num_fingers,
    num_envs, 
    agg_num_envs,
    num_robots,
    actions,
    object_init_states,
    success_tolerance: float,
    av_factor: float,
    cont_success_steps,
    K_cont_success_steps = 60,
    max_horizontal_offset = 0.4,
    **kwargs,
):
    info = {}

    ### 9-8: forced lift test
    test_forced_lift = kwargs["test_forced_lift"]
    if test_forced_lift:
        n_lift_steps = kwargs["n_lift_steps"]
        lift_step_count = kwargs["lift_step_count"]
        is_lifting_stage = kwargs["is_lifting_stage"]
    
    goal_object_dist = torch.abs(goal_height - object_pos[:, 2])
    palm_object_dist = torch.norm(object_pos - palm_pos.reshape(num_envs,-1), dim=-1)
    palm_object_dist = torch.where(palm_object_dist >= 0.5, 0.5, palm_object_dist)
    horizontal_offset = torch.norm(object_pos[:, 0:2], dim=-1) # horizontally close to (0,0)

    fingertips_object_dist = [] #torch.zeros_like(goal_object_dist)
    for i in range(num_robots):
        offset = agg_num_fingers[i]
        n_f = num_fingers[i]
        dists = torch.norm(fingertip_pos[:,offset:offset+n_f] - object_pos.view(agg_num_envs,-1,3)[:,i:i+1].repeat(1,n_f,1), dim=-1) # [agg_env,n_finger]
        fingertips_object_dist.append(torch.mean(dists, dim=-1).view(-1,1)) # n_robots * [agg_env,1]. avg distance over fingers
    fingertips_object_dist = torch.cat(fingertips_object_dist, dim=1).reshape(-1) # [num_envs]
    fingertips_object_dist = torch.where(fingertips_object_dist >= 0.5, 0.5, fingertips_object_dist)

    flag = torch.logical_and((fingertips_object_dist <= 0.12) + (palm_object_dist <= 0.15), horizontal_offset <= max_horizontal_offset)
    flag_object_hand = torch.logical_or(fingertips_object_dist <= 0.12, palm_object_dist <= 0.15)

    # stage 1: lift reward 1
    object_goal_reward = torch.zeros_like(goal_object_dist)
    object_goal_reward = torch.where(flag >= 1, (0.9 - 2 * goal_object_dist), object_goal_reward)
    # stage 1: lift reward 2
    object_height = object_pos[:, 2]
    object_up = torch.zeros_like(goal_object_dist)
    object_up = torch.where(flag >= 1, 1 * (object_height - goal_height), object_up)
    
    # stage 2: lift to goal bonus
    bonus = torch.zeros_like(goal_object_dist)
    bonus = torch.where(flag >= 1, 
        torch.where(goal_object_dist <= success_tolerance, 1.0 / (1 + goal_object_dist), bonus), 
        bonus)

    # stage 3: success
    successes = torch.where(
        goal_object_dist <= success_tolerance,
        torch.where(
            flag >= 1, torch.ones_like(successes), successes
        ),
        torch.zeros_like(successes),
    )

    # penalty
    # object_fall = torch.where(
    #     (horizontal_offset>max_horizontal_offset)+(object_height<=0.3),
    #     torch.ones_like(task_successes),
    #     torch.zeros_like(task_successes)
    # )

    ### 9-8: forced lift test
    if test_forced_lift:
        # envs in the lifting stage: step count +1
        lift_step_count[:] = torch.where(is_lifting_stage>0, lift_step_count+1, lift_step_count) 
        # envs trigger the success criterion: go to lifting stage
        is_lifting_stage[:] = torch.where(successes>0, torch.ones_like(is_lifting_stage), is_lifting_stage) 
        # envs at the last step of lifting stage and hand-obj still close: task success
        task_successes = torch.zeros_like(goal_object_dist)
        task_successes = torch.where((lift_step_count>=n_lift_steps) & (flag_object_hand>=1), torch.ones_like(task_successes), task_successes) 
        # reset criterion
        resets = reset_buf.clone()
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets) # timeout
        resets = torch.where(object_height<=0.3, torch.ones_like(resets), resets) # object fall
        resets = torch.where(lift_step_count>=n_lift_steps, torch.ones_like(resets), resets) # lifting stage end
    else:
        ### default / training case: hold several steps to accomplish the task
        cont_success_steps = torch.where(successes>0, cont_success_steps+1, cont_success_steps)
        task_successes = torch.zeros_like(goal_object_dist)
        task_successes = torch.where(cont_success_steps>=K_cont_success_steps, torch.ones_like(task_successes), task_successes)
        resets = reset_buf.clone()
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets) # timeout
        resets = torch.where(object_height<=0.3, torch.ones_like(resets), resets) # object fall
        resets = torch.where(cont_success_steps>=K_cont_success_steps, torch.ones_like(resets), resets) # task success
    
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(task_successes * resets.float())
    current_successes = torch.where(resets > 0, task_successes, current_successes)
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    # reward shaping & information log
    reward = (
        - 2.0 * fingertips_object_dist
        - 1.0 * palm_object_dist
        + object_goal_reward
        + object_up
        + bonus
        + 200 * task_successes
        - 0.3 * horizontal_offset
        #- 100 * object_fall
    )

    info["fingertips_object_dist"] = fingertips_object_dist
    info["palm_object_dist"] = palm_object_dist
    info["object_goal_reward"] = object_goal_reward
    info["object_up"] = object_up
    info["bonus"] = bonus
    info["success_reward"] = task_successes
    info["horizontal_offset"] = horizontal_offset
    info["reward"] = reward
    info["hand_approach_flag"] = flag
    #info["object_fall"] = object_fall

    return (
        reward,
        resets,
        progress_buf,
        successes,
        current_successes,
        cons_successes,
        cont_success_steps,
        info,
    )



'''
Add safety rewards for sim2real
- consider table force on the z axis
'''
def reward_sim2real(
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
    agg_num_fingers,
    num_envs, 
    agg_num_envs,
    num_robots,
    actions,
    object_init_states,
    success_tolerance: float,
    av_factor: float,
    cont_success_steps,
    table_force_z, 
    init_table_force_z,
    K_cont_success_steps = 60,
    max_horizontal_offset = 0.4,
    **kwargs,
):
    info = {}

    ### 9-8: forced lift test
    test_forced_lift = kwargs["test_forced_lift"]
    if test_forced_lift:
        n_lift_steps = kwargs["n_lift_steps"]
        lift_step_count = kwargs["lift_step_count"]
        is_lifting_stage = kwargs["is_lifting_stage"]
    
    goal_object_dist = torch.abs(goal_height - object_pos[:, 2])
    palm_object_dist = torch.norm(object_pos - palm_pos.reshape(num_envs,-1), dim=-1)
    palm_object_dist = torch.where(palm_object_dist >= 0.5, 0.5, palm_object_dist)
    horizontal_offset = torch.norm(object_pos[:, 0:2], dim=-1) # horizontally close to (0,0)

    fingertips_object_dist = [] #torch.zeros_like(goal_object_dist)
    for i in range(num_robots):
        offset = agg_num_fingers[i]
        n_f = num_fingers[i]
        dists = torch.norm(fingertip_pos[:,offset:offset+n_f] - object_pos.view(agg_num_envs,-1,3)[:,i:i+1].repeat(1,n_f,1), dim=-1) # [agg_env,n_finger]
        fingertips_object_dist.append(torch.mean(dists, dim=-1).view(-1,1)) # n_robots * [agg_env,1]. avg distance over fingers
    fingertips_object_dist = torch.cat(fingertips_object_dist, dim=1).reshape(-1) # [num_envs]
    fingertips_object_dist = torch.where(fingertips_object_dist >= 0.5, 0.5, fingertips_object_dist)

    flag = torch.logical_and((fingertips_object_dist <= 0.12) + (palm_object_dist <= 0.15), horizontal_offset <= max_horizontal_offset)
    flag_object_hand = torch.logical_or(fingertips_object_dist <= 0.12, palm_object_dist <= 0.15)

    # stage 1: lift reward 1
    object_goal_reward = torch.zeros_like(goal_object_dist)
    object_goal_reward = torch.where(flag >= 1, (0.9 - 2 * goal_object_dist), object_goal_reward)
    # stage 1: lift reward 2
    object_height = object_pos[:, 2]
    object_up = torch.zeros_like(goal_object_dist)
    object_up = torch.where(flag >= 1, 1 * (object_height - goal_height), object_up)
    
    # stage 2: lift to goal bonus
    bonus = torch.zeros_like(goal_object_dist)
    bonus = torch.where(flag >= 1, 
        torch.where(goal_object_dist <= success_tolerance, 1.0 / (1 + goal_object_dist), bonus), 
        bonus)

    # stage 3: success
    successes = torch.where(
        goal_object_dist <= success_tolerance,
        torch.where(
            flag >= 1, torch.ones_like(successes), successes
        ),
        torch.zeros_like(successes),
    )

    # penalty
    # object_fall = torch.where(
    #     (horizontal_offset>max_horizontal_offset)+(object_height<=0.3),
    #     torch.ones_like(task_successes),
    #     torch.zeros_like(task_successes)
    # )

    # save the table force before the hand reaches the table
    table_force_z = table_force_z.clone()
    init_table_force_z[:] = torch.where(progress_buf<5, table_force_z[:], init_table_force_z[:])
    # table force penalty
    table_force_z_change = torch.where(table_force_z-init_table_force_z<-500, table_force_z-init_table_force_z, torch.zeros_like(table_force_z)) / 200.
    #print(table_force_z_change[1])

    ### 9-8: forced lift test
    if test_forced_lift:
        # envs in the lifting stage: step count +1
        lift_step_count[:] = torch.where(is_lifting_stage>0, lift_step_count+1, lift_step_count) 
        # envs trigger the success criterion: go to lifting stage
        is_lifting_stage[:] = torch.where(successes>0, torch.ones_like(is_lifting_stage), is_lifting_stage) 
        # envs at the last step of lifting stage and hand-obj still close: task success
        task_successes = torch.zeros_like(goal_object_dist)
        task_successes = torch.where((lift_step_count>=n_lift_steps) & (flag_object_hand>=1), torch.ones_like(task_successes), task_successes) 
        # reset criterion
        resets = reset_buf.clone()
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets) # timeout
        resets = torch.where(object_height<=0.3, torch.ones_like(resets), resets) # object fall
        resets = torch.where(lift_step_count>=n_lift_steps, torch.ones_like(resets), resets) # lifting stage end
    else:
        ### default / training case: hold several steps to accomplish the task
        cont_success_steps = torch.where(successes>0, cont_success_steps+1, cont_success_steps)
        task_successes = torch.zeros_like(goal_object_dist)
        task_successes = torch.where(cont_success_steps>=K_cont_success_steps, torch.ones_like(task_successes), task_successes)
        resets = reset_buf.clone()
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets) # timeout
        resets = torch.where(object_height<=0.3, torch.ones_like(resets), resets) # object fall
        resets = torch.where(cont_success_steps>=K_cont_success_steps, torch.ones_like(resets), resets) # task success
    
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(task_successes * resets.float())
    current_successes = torch.where(resets > 0, task_successes, current_successes)
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    # reward shaping & information log
    reward = (
        - 2.0 * fingertips_object_dist
        - 1.0 * palm_object_dist
        + object_goal_reward
        + object_up
        + bonus
        + 200 * task_successes
        - 0.3 * horizontal_offset
        #- 100 * object_fall
        + table_force_z_change
    )

    info["fingertips_object_dist"] = fingertips_object_dist
    info["palm_object_dist"] = palm_object_dist
    info["object_goal_reward"] = object_goal_reward
    info["object_up"] = object_up
    info["bonus"] = bonus
    info["success_reward"] = task_successes
    info["horizontal_offset"] = horizontal_offset
    info["reward"] = reward
    info["hand_approach_flag"] = flag
    #info["object_fall"] = object_fall
    info["table_force_z_change"] = table_force_z_change

    return (
        reward,
        resets,
        progress_buf,
        successes,
        current_successes,
        cons_successes,
        cont_success_steps,
        info,
    )


REWARD_DICT = {"v0": reward_v0, "v1": reward_v1, "v2": reward_v2, "sim2real": reward_sim2real}
