'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2023-12-15 18:05:25
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2023-12-15 18:37:42
FilePath: /UNET/2dunet_test.py
Description: 
I Love IU
Copyright (c) 2023 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
import monai
import torch
import glob
from monai.networks.nets import UNet
import numpy as np
from omegaconf import OmegaConf
from utils import make_if_dont_exist
import argparse
import os
import resource
from tqdm import tqdm
import json
from monai.data import DataLoader, Dataset
from metrics import dice_score, average_surface_distance, hausdorff_distance, surface_dice, average_normal_error, average_normalized_lap_distance
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes

def parse_command():
    """
    The function `parse_command` is a Python function that uses the `argparse` module to parse command
    line arguments and returns the parsed arguments.
    :return: the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, type=str, help='path to config file')
    args = parser.parse_args()
    return args

def dataset(cfg, test_dir):
    """
    The function `dataset` takes in a configuration object and a directory path, and returns a
    DataLoader object that loads and transforms test data for a machine learning model.
    
    :param cfg: The parameter `cfg` is a configuration object that contains various settings for the
    dataset and data loader. It likely includes properties such as `test_bs` (test batch size) and
    `num_workers` (number of worker processes for data loading)
    :param test_dir: The `test_dir` parameter is the directory path where the test data is located. It
    is expected to have two subdirectories: "images" and "labels". The "images" directory should contain
    the input images in NIfTI format (with the extension ".nii.gz"), and the "
    :return: a DataLoader object.
    """
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    test_data = []
    for i in sorted(glob.glob(os.path.join(test_dir, 'images', '*.nii.gz'))):
        test_data.append({
            'img': i,
            'seg': i.replace('images', 'labels')
        })
    test = test_data
    transform = monai.transforms.Compose(
        transforms=[
            monai.transforms.LoadImageD(keys=['img', 'seg'], image_only=False),
            monai.transforms.TransposeD(keys=["img", "seg"], indices=(2, 1, 0)),
            monai.transforms.EnsureChannelFirstD(keys=['img', 'seg']),
        ]
    )
    
    test_dataset = Dataset(
        data=test,
        transform=transform
    )
    return DataLoader(
        test_dataset,
        batch_size=cfg.test_bs,
        num_workers=cfg.num_workers,
        shuffle=False
    )

if __name__ == '__main__':
    args = parse_command()
    cfg = args.cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg is not None:
        if os.path.exists(cfg):
            cfg = OmegaConf.load(cfg)
        else:
            raise FileNotFoundError(f'config file {cfg} not found')
    else:
        raise ValueError('config file not specified')
    
    # setup folders
    exp = cfg.experiment
    root_dir = os.path.join(cfg.dataset.dataset_dir, '2D')
    test_dir = os.path.join(cfg.dataset.dataset_dir, '3D', 'test')
    exp_path = os.path.join(root_dir, exp)
    test_path = os.path.join(exp_path, 'inference')
    model_path = os.path.join(exp_path, 'model')
    make_if_dont_exist(test_path)
    
    test_loader = dataset(cfg, test_dir)
    
    # load model
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=cfg.model.class_num,
        channels=cfg.model.channels,
        strides=cfg.model.strides,
    ).to(device)
    best_model = torch.load(os.path.join(model_path, 'model.pth'), map_location=device)
    model.load_state_dict(best_model['weights'])
    
    model.eval()
    results = {}
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader,desc='inference',unit='batch')):
            results[str(i)] = {}
            img = batch['img'].to(device)
            seg = batch['seg'].to(device)
            pred = torch.zeros(seg.shape).to(device)
            for j in range(img.shape[-1]):
                if len(torch.unique(seg[...,j])) > 1:
                    output = model(img[..., j])
                    output = torch.argmax(output.softmax(dim=1), dim=1, keepdim=True)
                    pred[..., j] = output
                    del output
                    torch.cuda.empty_cache()
            seg = monai.networks.one_hot(seg, 2)
            onehot_pred = monai.networks.one_hot(pred, 2)
            pred_verts, pred_faces = marching_cubes(onehot_pred[:,1,...].float())
            gt_verts, gt_faces = marching_cubes(seg[:,1,...])
            pred_mesh = Meshes(verts=pred_verts, faces=pred_faces)
            gt_mesh = Meshes(verts=gt_verts, faces=gt_faces)
            spacing = batch['seg_meta_dict']['pixdim'][0,1:4]
            # origin = np.array([0., 0., 0.])
            # direction = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            # ants_seg = ants.from_numpy(output_save, origin=origin, spacing=spacing, direction=direction)
            results[str(i)]['dsc'] = dice_score(onehot_pred, seg).detach().cpu().numpy()[0,0]
            results[str(i)]['asd'] = average_surface_distance(onehot_pred, seg).detach().cpu().numpy()[0,0]
            results[str(i)]['hd'] = hausdorff_distance(onehot_pred, seg).detach().cpu().numpy()[0,0]
            results[str(i)]['sd'] = surface_dice(onehot_pred[:,1,...], seg[:,1,...], spacing)
            results[str(i)]['cd'] = chamfer_distance(gt_verts[0].unsqueeze(0), pred_verts[0].unsqueeze(0))[0].item()
            results[str(i)]['ane'] = average_normal_error(gt_mesh, pred_mesh).item()
            results[str(i)]['anld'] = average_normalized_lap_distance(pred_mesh).item()
    
    metrics = np.zeros((len(results), 7))
    for i, j in enumerate(results.values()):
        metrics[i, 0] = j['dsc']
        metrics[i, 1] = j['asd']
        metrics[i, 2] = j['hd']
        metrics[i, 3] = j['sd']
        metrics[i, 4] = j['cd']
        metrics[i, 5] = j['ane']
        metrics[i, 6] = j['anld']
    
    ret = {}
    ret['mean_dsc'] = np.mean(metrics[:,0])
    ret['mean_asd'] = np.mean(metrics[:,1])
    ret['mean_hd'] = np.mean(metrics[:,2])
    ret['mean_sd'] = np.mean(metrics[:,3])
    ret['mean_cd'] = np.mean(metrics[:,4])
    ret['mean_ane'] = np.mean(metrics[:,5])
    ret['mean_anld'] = np.mean(metrics[:,6])

    
    with open(os.path.join(test_path, 'results.json'), 'w') as f:
        json.dump(ret, f, indent=4, sort_keys=False)
            
    torch.cuda.empty_cache()      