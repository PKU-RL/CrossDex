import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import tyro
from torch.utils.tensorboard import SummaryWriter
from retargeting_nn_utils import *

class Trainer:
    def __init__(self, dataset="grab", add_random_dataset=False, robot_name=RobotName.leap, 
                 retargeting_type="dexpilot", position_baseline=False, device='cuda'):
        self.position_baseline = position_baseline
        data = load_dataset(dataset, add_random_dataset, robot_name, retargeting_type, position_baseline)
        if not position_baseline:
            mano_pose45 = torch.tensor(data['mano_pose45'], dtype=torch.float32)
        else:
            mano_pose45 = torch.tensor(data['fingertip_pos'], dtype=torch.float32)
        robot_joint_pos = torch.tensor(data['robot_joint_pos'], dtype=torch.float32)
        mano_train, mano_val, robot_train, robot_val = train_test_split(
            mano_pose45, robot_joint_pos, test_size=0.025, random_state=42)
        self.train_dataset = TensorDataset(mano_train, robot_train)
        self.val_dataset = TensorDataset(mano_val, robot_val)
        self.robot_dim = robot_joint_pos.shape[1]
        self.robot_name = data["robot_name"]
        self.robot_joint_names = data["joint_names"]
        self.robot_urdf_path = data["urdf_path"]
        self.device = torch.device(device)
        print("Loaded train dataset: {}, val dataset {}".format(len(self.train_dataset), len(self.val_dataset)))

    def save_config(self, save_path):
        import yaml
        data = {"robot_name": self.robot_name,
            "robot_dim": self.robot_dim, 
            "robot_joint_names": self.robot_joint_names,
            "robot_urdf_path": self.robot_urdf_path,
        }
        with open(save_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
        
    def train(self, batch_size=256, num_epochs=200, save_path=None, log_dir=None):
        writer = SummaryWriter(log_dir=log_dir)

        device = self.device
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        model = RetargetingNN(robot_dim=self.robot_dim, mano_dim=45 if not self.position_baseline else 15).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        best_val_loss = float('inf')
        n_step = 0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                loss = criterion(outputs, targets.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                writer.add_scalar('Loss/Train-step', loss.item(), n_step)
                n_step += 1
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch}/{num_epochs - 1}, Training Loss: {epoch_loss:.4f}")
            writer.add_scalar('Loss/Train-epoch', epoch_loss, epoch)
        
            # 验证模型
            val_loss = self.evaluate_model(model, val_loader, criterion)
            print(f"Epoch {epoch}/{num_epochs - 1}, Validation Loss: {val_loss:.4f}")
            writer.add_scalar('Loss/Val', val_loss, epoch)
        
            # 保存验证集上最好的模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print("Saved best model with loss:", best_val_loss)

    def evaluate_model(self, model, val_loader, criterion):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs.to(self.device))
                loss = criterion(outputs, targets.to(self.device))
                running_loss += loss.item() * inputs.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        return val_loss

def main(dataset:str="grab", add_random_dataset:bool=False, robot_name:RobotName=RobotName.leap, retargeting_type:str="dexpilot",
         position_baseline:bool=False):
    trainer = Trainer(dataset, add_random_dataset, robot_name, retargeting_type, position_baseline)
    if position_baseline:
        save_config_path = "models/position_baseline_retargeting_nn_{}_{}_{}.yaml".format(ROBOT_NAME_MAP[robot_name], dataset, retargeting_type)
        save_model_path = "models/position_baseline_retargeting_nn_{}_{}_{}.pth".format(ROBOT_NAME_MAP[robot_name], dataset, retargeting_type)
        log_dir = 'models/log_position_baseline_{}_{}_{}'.format(ROBOT_NAME_MAP[robot_name], dataset, retargeting_type)
    elif add_random_dataset:
        save_config_path = "models/retargeting_nn_{}_{}_{}_random.yaml".format(ROBOT_NAME_MAP[robot_name], dataset, retargeting_type)
        save_model_path = "models/retargeting_nn_{}_{}_{}_random.pth".format(ROBOT_NAME_MAP[robot_name], dataset, retargeting_type)
        log_dir = 'models/log_{}_{}_{}_random'.format(ROBOT_NAME_MAP[robot_name], dataset, retargeting_type)
    else:
        save_config_path = "models/retargeting_nn_{}_{}_{}.yaml".format(ROBOT_NAME_MAP[robot_name], dataset, retargeting_type)
        save_model_path = "models/retargeting_nn_{}_{}_{}.pth".format(ROBOT_NAME_MAP[robot_name], dataset, retargeting_type)
        log_dir = 'models/log_{}_{}_{}'.format(ROBOT_NAME_MAP[robot_name], dataset, retargeting_type)

    os.makedirs(log_dir, exist_ok=True)
    trainer.save_config(save_config_path)
    trainer.train(save_path=save_model_path, log_dir=log_dir)

if __name__=='__main__':
    tyro.cli(main)
