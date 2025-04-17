import os
import re
import yaml
import argparse

def find_latest_checkpoint(folder_path):
    """
    Find the latest checkpoint file in the given folder.
    """
    pt_file_pattern = re.compile(r"model_(\d+)\.pt")
    max_index = -1
    latest_pt_file = None

    for filename in os.listdir(folder_path):
        if filename.endswith(".pt"):
            match = pt_file_pattern.match(filename)
            if match:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
                    latest_pt_file = filename

    return latest_pt_file

def generate_expert_yaml(path, output_file):
    """
    Generate the expert.yaml configuration file from the experiments in the given path.
    """
    expert_config = {}
    datetime_pattern = re.compile(r"^2024")

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)

        if os.path.isdir(folder_path):
            # Find the index of '2024' in folder_name
            match = re.search(r'2024', folder_name)
            if match:
                datetime_start_index = match.start()
                objname = folder_name[:datetime_start_index].rstrip('_')
                datetime_part = folder_name[datetime_start_index:]
                
                # Verify datetime_part starts with '2024'
                if datetime_pattern.match(datetime_part):
                    latest_ckpt = find_latest_checkpoint(folder_path)
                    if latest_ckpt:
                        expert_config[objname] = {
                            'path': folder_path,
                            'ckpt': latest_ckpt
                        }

    with open(output_file, 'w') as yaml_file:
        yaml.dump(expert_config, yaml_file, default_flow_style=False)

    print(f"Configuration file {output_file} has been generated.")

def main():
    parser = argparse.ArgumentParser(description="Generate expert.yaml configuration file.")
    parser.add_argument("--path", type=str, help="Path to the experiments folder.")

    args = parser.parse_args()
    generate_expert_yaml(args.path, 'expert.yaml')

if __name__ == "__main__":
    main()
