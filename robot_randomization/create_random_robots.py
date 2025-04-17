import xml.etree.ElementTree as ET
import random
import os
import json

def add_random_perturbation_to_xyz(xyz_str, perturbation_ranges=
    [(-0.03, 0.03), (-0.02,0.02), (-0.01,0.01)]):
    # 将 xyz 字符串转换为浮点数列表
    xyz = list(map(float, xyz_str.split()))

    # 对每个坐标添加随机扰动
    perturbed_xyz = [coord + random.uniform(*(perturbation_ranges[i])) for i, coord in enumerate(xyz)]

    # 将列表转换回字符串
    return ' '.join(map(str, perturbed_xyz)), perturbed_xyz

def randomize_urdf(input_dir, input_urdf, output_dir, save_dict, idx, joint_name="arm_hand_joint"):
    # 解析 URDF 文件
    tree = ET.parse(os.path.join(input_dir, input_urdf))
    root = tree.getroot()

    # 查找指定 joint
    for joint in root.findall('joint'):
        if joint.get('name') == joint_name:
            origin = joint.find('origin')
            assert origin is not None
            # 获取原始的 xyz 值
            original_xyz = origin.get('xyz')
            # 对 xyz 添加随机扰动
            perturbed_xyz_str, perturbed_xyz = add_random_perturbation_to_xyz(original_xyz)
            # 更新 origin 的 xyz 值
            origin.set('xyz', perturbed_xyz_str)
            print(f"{input_urdf}: updated {joint_name} xyz from {original_xyz} to {perturbed_xyz_str}")
            break
    
    # 将修改后的 URDF 保存为新的文件
    output_urdf = f"{input_urdf.split('.')[0]}_{idx}.urdf"
    tree.write(os.path.join(output_dir, output_urdf), encoding="utf-8", xml_declaration=True)
    save_dict[output_urdf] = {
        'x': perturbed_xyz[0],
        'y': perturbed_xyz[1],
        'z': perturbed_xyz[2]
    }


if __name__=='__main__':
    input_dir = 'urdf_origin'
    input_urdfs = os.listdir(input_dir)
    output_dir = 'urdf_rand_xyz'
    N_per_robot = 20
    save_dict = {}

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for fn in input_urdfs:
        if not fn.endswith(".urdf"):
            continue
        for i in range(N_per_robot):
            randomize_urdf(input_dir, fn, output_dir, save_dict, i)

    json_filename = f"{output_dir}/results.json"
    with open(json_filename, 'w') as json_file:
        json.dump(save_dict, json_file, indent=4)
