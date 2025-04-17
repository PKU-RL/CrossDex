### RL

# train. 
# use task.env.asset.objectAssetFile to specify the object
# use headless=True when running on a headless machine
python run_ppo_multidex.py \
num_envs=8192 \
task=MultiDexGrasp \
train.params.max_iterations=40000 \
task.env.observationType="armdof+keypts+objpose+lastact" \
task.env.asset.objectAssetFile="ycb_assets/urdf/077_rubiks_cube.urdf" \
task.env.randomizeRobot=True 

# test. 
# use +path to specify the checkpoint path
# use +test_hands to specify the testing embodiments
#   training hands are: ["shadow_hand","allegro_hand","ability_hand","schunk_svh_hand"]
#   unseen hands are: ["leap_hand","inspire_hand"]
python run_ppo_multidex.py \
task=MultiDexGrasp \
test=True \
num_envs=2400 \
+randomize_robot=False \
+test_hands='["shadow_hand","allegro_hand","ability_hand","schunk_svh_hand"]' \
+path="runs_multidex/077_rubiks_cube_2024-09-03_11-57-05" 



### Vision-based Distillation

# train.
# use expert=XXX to specify config file for objects and their corresponding expert policies
# use headless=True when running on a headless machine
python run_dagger_multidex.py \
num_envs=16384 \
task=MultiDexGrasp \
train=MultiDexGraspDAGGER \
task.env.enablePointCloud=True \
task.env.observationType="armdof+keypts+objpose+lastact" \
task.env.studentObservationType="armdof+keypts+lastact+objpcl" \
task.env.multiTask=True \
task.env.multiTaskLabel="no" \
task.env.asset.objectAssetDir="ycb_assets/urdf" \
expert=expert \
train.params.max_iterations=20000 \
task.env.randomizeRobot=True 

