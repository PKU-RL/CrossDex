# CrossDex
Official code for **"Cross-Embodiment Dexterous Grasping with Reinforcement Learning"** *(ICLR 2025)*

## TODO List
- [x] Code for processing eigengrasps and training retargeting networks.
- [x] Code for embodiment randomization.
- [ ] Code for RL and DAgger.

## Requirements
- `pip install -r requirements.txt`.
- Install manopth according to [this](https://github.com/dexsuite/dex-retargeting/tree/main).
- Install dex-retargeting: `cd dex-retargeting & pip install -e .`. This codebase is developed upon [this](https://github.com/dexsuite/dex-retargeting/).


## Eigengrasp
Files `results/pca_$N_grab.pkl` are eigengrasps with $N PCA eigen vectors from the [GRAB dataset](https://github.com/otaheri/GRAB). Data format in each .pkl is:
```
'eigen_vectors': (N,45) numpy array, the eigengrasps corresponding to 45-dim finger axis-angles in Mano
'min_values': (N,) numpy array, min values on each axis
'max_values': (N,) numpy array, max values on each axis
'D_mean': (45,) numpy array, mean of the original data
'D_std': (45,) numpy array, std of the original data
```
Run `results/vis_pca_data.py` to control the 9-dim coordinates and visualize the corresponding Mano hand pose. 


## Hand Retargeting
We use dexpilot to retarget Mano pose to dexterous hand joint angles. `cd retargeting` and run `vis_eigengrasp_to_dexhand.py` to visualize Mano-to-all-hands retargeting.

To accelerate batch computation for parallel RL training, we train retargeting neural networks. 
- Download [GRAB dataset](https://github.com/otaheri/GRAB), place `s1.pkl`~`s10.pkl` files under `../GRAB/hand_dataset/`. Run `generate_dataset.py` to generate paired training data of 45-dim mano pose and X-dim robot pose. Dataset saved in `dataset/`. Use the option `--robot_name` to specify the robot hand.
- Run `train_retartgeting_nn.py` to train the retargeting neural network. The checkpoint, config, tensorboard log will be saved in `models/`. Also use the option `--robot_name`.
- Run `vis_nn_retargeting.py` to qualitatively check the performance of the learned model.

After that, we can use the class `retargeting_nn_utils.EigenRetargetModel` to do retargeting.

## Policy Learning

### Add Robot Randomization
In `robot_randomization/`, run `create_random_robots.py` to randomize the xyz offsets of the hand-arm mounting joint, generating 20 variants for each robot.

### Cross-Embodiment training 
(coming soon)

`cd rl/`, follow the scripts in `run.sh` to run the experiment with randomized robots.

We implement parallel environment for multiple robots in `tasks_crossdex/multi_dex_grasp.py`. The config file in `tasks_crossdex/task/MultiDexGrasp.yaml`.
- Action space: 6-dim arm dofs + 9-dim eigengrasp space, using pre-trained NNs to retarget to each robot's dof targets.
- Observation space: arm dof pos + keypoints pos (palm + first 4 ordered fingers) + object pose + last act.

### Vision-Based Distillation 
(coming soon)

## Citation
```bibtex
@article{yuan2024cross,
  title={Cross-embodiment dexterous grasping with reinforcement learning},
  author={Yuan, Haoqi and Zhou, Bohan and Fu, Yuhui and Lu, Zongqing},
  journal={arXiv preprint arXiv:2410.02479},
  year={2024}
}
```
