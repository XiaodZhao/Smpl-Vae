import sys
import numpy as np
import os
import glob
from meters import Meters
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from train import train
from transform import transform
from save import save_pic, save_params
from load import load
from easydict import EasyDict as edict
import json
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def get_config(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)
    cfg = edict(data.copy())
    return cfg
def set_device(USE_GPU):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
def getNum(item):
    return int(''.join(filter(str.isdigit, item)))

root = "./gt/"
paths = glob.glob(os.path.join(root, '*.npy'))
paths = sorted(paths, key=getNum)

dataset_name = "Shelf"
config_path = './Config/Shelf.json'
colors = [(1,0,0),
          (0,0,1),
          (1,0.5,0),
          (1,1,0)]
params = {}
params["pose_params"] = None
params["shape_params"] = None
params["scale"] = None
OriginNum = 0
for frame, dataPath in enumerate(paths):
    d_colors = []
    points = np.load(dataPath, allow_pickle=True)
    Points = []
    Origin = []
    labels = []
    center_idx =0
    for m in range(points.shape[0]):
        p = points[m]
        if p is not None:
            labels.append(m)
            Origin.append(p[center_idx] * np.array([1., 1., 1.]))
            d_colors.append(colors[m])
            p[:, [1, 2]] = p[:, [2, 1]]
            p[:, [0, 2]] = p[:, [2, 0]]
            Points.append(p)
    pNum = len(Points)
    print(labels)
    if pNum > 0:
        Points3D = np.array(Points)
        cfg = get_config(config_path)
        # print(cfg)
        device = set_device(USE_GPU=1)
        smpl_layer = SMPL_Layer(
            center_idx=0,
            gender="neutral",
            model_root='./smplpytorch/native/models')
        target = torch.from_numpy(transform(dataset_name, Points3D)).float()
        print(target.size(),'target.size')
        meters = Meters()


        if pNum > OriginNum:
            makeup_pose = torch.zeros(pNum - OriginNum, 72,device='cuda:0')
            makeup_shape = torch.zeros(pNum - OriginNum, 10,device='cuda:0')
            res[0] = res[0].detach()
            res[1] = res[1].detach()
            new_pose = torch.cat((res[0],makeup_pose),dim=0)
            new_shape = torch.cat((res[1],makeup_shape),dim=0)
            params["pose_params"] = new_pose
            params["shape_params"] =new_shape
            params["scale"] = torch.ones([1]) 
        elif pNum <  OriginNum:
            deal_pose = res[0].detach()
            deal_shape = res[1].detach()
            params["pose_params"] = deal_pose[:pNum]
            params["shape_params"] = deal_shape[:pNum]
            print('pNum <  OriginNum')
            params["scale"] = torch.ones([1]) 
        elif pNum ==  OriginNum:
            params["pose_params"] = res[0]
            params["shape_params"] = res[1]
            params["scale"] = res[2]
            data111 = np.load('zzzzz.npz',allow_pickle=True)
            pose_z = torch.squeeze(torch.tensor(data111['arr_1']))
            s = torch.zeros(1,72)
            pose_end = torch.cat((pose_z[0].reshape(1,72), s),dim=0)
            params["pose_params"] = torch.flip(pose_end, dims=[-1])
            params["shape_params"] =torch.squeeze(torch.tensor(data111['arr_0']))
        res = train(smpl_layer, target, device, cfg, meters,params)
        meters.update_avg(meters.min_loss, k=target.shape[0])
        meters.reset_early_stop()
        save_pic(res, smpl_layer,Origin, frame, d_colors,pNum)
        torch.cuda.empty_cache()
        print("Fitting finished! Average loss:     {:.9f}".format(meters.avg))
    else:
        fig_size = [9, 6.8]
        fig = plt.figure(figsize=fig_size)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 2)

        plt.savefig("./Pictures/frame_{:0>4d}.jpg".format(frame),bbox_inches='tight', pad_inches=0)
        plt.close()
    OriginNum = pNum


