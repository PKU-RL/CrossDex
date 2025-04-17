from isaacgymenvs.tasks import isaacgym_task_map
from .cross_dex_grasp import CrossDexGrasp
from .multi_dex_grasp import MultiDexGrasp
from .multi_dex_grasp_baseline import MultiDexGraspBaseline
from .multi_dex_grasp_sim2real import MultiDexGraspSim2real

isaacgym_task_map["CrossDexGrasp"] = CrossDexGrasp
isaacgym_task_map["MultiDexGrasp"] = MultiDexGrasp
isaacgym_task_map["MultiDexGraspBaseline"] = MultiDexGraspBaseline
isaacgym_task_map["MultiDexGraspSim2real"] = MultiDexGraspSim2real
