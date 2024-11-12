import numpy as np
import torch

def load_weights(network, filename="yolov2-tiny-voc.weights"):
    with open(filename, "rb") as file:
        version = np.fromfile(file, count=3, dtype=np.int32)
        seen_so_far = np.fromfile(file, count=1, dtype=np.int32)
        weights = np.fromfile(file, dtype=np.float32)
        idx = 0
        print(f"Total weights read: {len(weights)}")
        
        for layer in network.children():
            if isinstance(layer, torch.nn.Conv2d):
                if layer.bias is not None:
                    n = layer.bias.numel()
                    layer.bias.data[:] = torch.from_numpy(weights[idx:idx + n]).view_as(layer.bias.data)
                    idx += n
                n = layer.weight.numel()
                if len(weights[idx:idx + n]) == n:
                    layer.weight.data[:] = torch.from_numpy(weights[idx:idx + n]).view_as(layer.weight.data)
                    print(f"Loaded Conv2d layer with weight shape: {layer.weight.shape}")
                else:
                    print(f"Warning: Conv2d layer with weight shape {layer.weight.shape} has a mismatch.")
                idx += n
            if isinstance(layer, torch.nn.BatchNorm2d):
                n = layer.bias.numel()
                layer.bias.data[:] = torch.from_numpy(weights[idx:idx + n]).view_as(layer.bias.data)
                idx += n
                layer.weight.data[:] = torch.from_numpy(weights[idx:idx + n]).view_as(layer.weight.data)
                idx += n
                layer.running_mean.data[:] = torch.from_numpy(weights[idx:idx + n]).view_as(layer.running_mean.data)
                idx += n
                layer.running_var.data[:] = torch.from_numpy(weights[idx:idx + n]).view_as(layer.running_var.data)
                idx += n
            if isinstance(layer, torch.nn.Linear):
                n = layer.bias.numel()
                layer.bias.data[:] = torch.from_numpy(weights[idx:idx + n]).view_as(layer.bias.data)
                idx += n
                n = layer.weight.numel()
                layer.weight.data[:] = torch.from_numpy(weights[idx:idx + n]).view_as(layer.weight.data)
                idx += n

        print(f"Index after loading weights: {idx}")
        print(f"Total weights in file: {len(weights)}")
