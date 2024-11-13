import argparse, json, os, mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from model.Yolo import TinyYOLOv2
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from utils.training_utils import train_epoch, validate_epoch, top_k_checkpoints
from dataset.Image_Dataloader import Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from yolo_loss import YOLOLoss
from load_weight import load_weights
import os
import urllib.request
import numpy as np

from models.yolo import Model



def train(model, train_loader, val_loader, criterion, optimizer, args):
    best_val_loss = float("inf")
    for epoch in range(args.num_epochs):
        model, train_loss, train_acc, box_l, conf_l, class_l = train_epoch(model, train_loader, criterion, optimizer, args)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)

        if args.val_interval > 0 and (epoch + 1) % args.val_interval == 0:
            val_loss, val_acc, box_v, conf_v, class_v = validate_epoch(model, val_loader, criterion, args)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(model.state_dict(), os.path.join(mlflow.get_artifact_uri(), \
                            f"model_best_ep{epoch}_val_loss_{val_loss:.4f}.pth"))
                top_k_checkpoints(args, mlflow.get_artifact_uri())

            print(f"[Validation] at Epoch {epoch+1}/{args.num_epochs}: Val Loss: {val_loss:.4f}, box_v:{box_v:.4f}, conf_v:{conf_v:.4f}, class_v:{class_v:.4f}")
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
        torch.cuda.empty_cache()
        # Print progress
        print(f"Epoch {epoch+1}/{args.num_epochs}: Train Loss: {train_loss:.4f}, box_l:{box_l:.4f}, conf_l:{conf_l:.4f}, class_l:{class_l:.4f}")
    return model


def main(args):
    # Load hyperparameters from JSON configuration file
    if args.config_file:
        with open(os.path.join('./configs', args.config_file), 'r') as f:
            config = json.load(f)
        # Overwrite hyperparameters with command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        args = argparse.Namespace(**config)
    else:
        raise ValueError("Please provide a JSON configuration file.")

    # Set up MLflow logging
    mlflow.set_tracking_uri(args.mlflow_path)
    mlflow.set_experiment(experiment_name=args.experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name):
        # dump this training file to MLflow artifact
        mlflow.log_artifact(__file__)

        # Log all hyperparameters to MLflow
        mlflow.log_params(vars(args))
        # also dump the args to a JSON file in MLflow artifact
        with open(os.path.join(mlflow.get_artifact_uri(), "args.json"), 'w') as f:
            json.dump(vars(args), f)

        if args.architecture == "YOLOv5n":
            # Use specified newer YOLO model
            model = Model('/Users/benediktwitteler/Code/Python/NeuromorphicHackathon/Object-tracking/models/yolov5n.yaml').to('mps')


            # # Only train last layer (for nano n version number 23)
            # for name, param in model.model.named_parameters():
            #     # Unfreeze only the parameters in the last layer (e.g., 'head' in YOLO models)
            #     if 'model.model.23' in name:
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False

        else:
            model = eval(args.architecture)(args).to(args.device)

            # Use initial YOLOv2 model
            file_path = "yolov2-tiny-voc.weights"
            url = "https://pjreddie.com/media/files/yolov2-tiny-voc.weights"

            if not os.path.exists(file_path):
                try:
                    urllib.request.urlretrieve(url, file_path)
                    print(f"Downloaded {file_path} from {url}")
                except Exception as e:
                    print(f"An error occurred: {e}")
            else:
                print(f"{file_path} already exists.")
            load_weights(model)

            # Freeze all layers except the last layer
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze the last two layers
            # for param in model.conv7.parameters():
            #     param.requires_grad = True
            # for param in model.conv8.parameters():
            #     param.requires_grad = True
            for param in model.conv9.parameters():
                param.requires_grad = True

        # Print to verify which layers are trainable
        print("Trainable Parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)


        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.loss == "mse":
            criterion = nn.MSELoss()
        elif args.loss == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        elif args.loss == "yolo":
            criterion = YOLOLoss()
        else:
            raise ValueError("Invalid loss name")

        
        train_dataset = Data(split="train")
        val_dataset = Data(split="valid")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)

        model = train(model, train_loader, val_loader, criterion, optimizer, args)

        torch.save(model.state_dict(), os.path.join(mlflow.get_artifact_uri(), f"model_last_epoch{args.num_epochs}.pth"))

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))  # Filter out None values
    if len(batch) == 0:
        return None  # Handle case where all items in the batch are invalid
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()   
    parser.add_argument("--mlflow_path", type=str, help="path to MLflow tracking server")
    parser.add_argument("--experiment_name", type=str, help="name of the experiment")
    parser.add_argument("--run_name", type=str, help="name of the run")
    parser.add_argument("--config_file", type=str, default=None, help="path to JSON configuration file")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--num_epochs", type=int, help="number of epochs")
    args = parser.parse_args()
    main(args)
