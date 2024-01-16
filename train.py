import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.functional as F
import torch.optim as optim
import sys
import os

from tqdm import tqdm
sys.path.append(os.getcwd())
from save import save_single_pic

class VAE(nn.Module):
    def __init__(self, input_size=72, hidden_size=400, latent_size=20):
        super(VAE, self).__init__()


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        # 解码
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    BCE = F.smooth_l1_loss(recon_x, x)

    # Kullback-Leibler散度项
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def init(smpl_layer, target, device, cfg, params):
    # params = {}
    # params["pose_params"] = torch.zeros(target.shape[0], 72)
    # params["shape_params"] = torch.zeros(target.shape[0], 10)
    # params["scale"] = torch.ones([1])

    smpl_layer = smpl_layer.to(device)
    params["pose_params"] = params["pose_params"].to(device)
    params["shape_params"] = params["shape_params"].to(device)
    target = target.to(device)
    params["scale"] = params["scale"].to(device)

    params["pose_params"].requires_grad = True
    params["shape_params"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SHAPE)
    params["scale"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SCALE)

    optim_params = [{'params': params["pose_params"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    {'params': params["shape_params"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    {'params': params["scale"], 'lr': cfg.TRAIN.LEARNING_RATE*10},]
    optimizer = optim.Adam(optim_params)
    # optimizer = optim.SGD(optim_params)
    index = {}
    smpl_index = []
    dataset_index = []
    for tp in cfg.DATASET.DATA_MAP:
        smpl_index.append(tp[0])
        dataset_index.append(tp[1])

    index["smpl_index"] = torch.tensor(smpl_index).to(device)
    index["dataset_index"] = torch.tensor(dataset_index).to(device)

    return smpl_layer, params, target, optimizer, index


def train(smpl_layer, target, device, cfg, meters,params):
    res = []
    smpl_layer, params, target, optimizer_smpl, index = \
        init(smpl_layer, target, device, cfg,params)
    pose_params = params["pose_params"]
    shape_params = params["shape_params"]
    scale = params["scale"]

    # 创建VAE模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_model = VAE().to(device)
    # 定义优化器
    optimizer_vae  = optim.Adam(vae_model.parameters(), lr=0.001)
    input_data = target.index_select(1, index["dataset_index"]).reshape(-1,72)

    with torch.no_grad():
        recon_data, mu, logvar = vae_model(input_data)
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        params["scale"]*=(torch.max(torch.abs(target))/torch.max(torch.abs(Jtr)))
    print('---------------------------------------------------------vae------------------------------------------------------------------------------------------------------------')
    for epoch_vae in range(1000):
        recon_data, mu, logvar = vae_model(input_data)
        lossvae =  loss_function(recon_data, input_data, mu, logvar)
        optimizer_vae.zero_grad()
        lossvae.backward()
        optimizer_vae.step()
        
        if epoch_vae % cfg.TRAIN.WRITE == 0 or epoch_vae<10:
            print("Epoch {}, lossPerBatch={:.6f}".format( epoch_vae, float(lossvae)))

    print('-------------------------------------------------------smpl--------------------------------------------------------------------------------------')
    for epoch in tqdm(range(cfg.TRAIN.MAX_EPOCH)):
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        loss_smpl = F.smooth_l1_loss(params["scale"]*Jtr.index_select(1, index["smpl_index"]), target.index_select(1, index["dataset_index"]))
        optimizer_smpl.zero_grad()
        loss_smpl.backward()
        optimizer_smpl.step()
        meters.update_early_stop(float(loss_smpl))
        if meters.update_res:
            res = [pose_params, shape_params, scale ,verts, Jtr]
            print(verts.shape)
            print(Jtr.shape)
            # torch.Size([2, 6890, 3])
            # torch.Size([2, 24, 3])
        if meters.early_stop:
            break
        print
        if epoch % cfg.TRAIN.WRITE == 0 or epoch<10:
            # logger.info("Epoch {}, lossPerBatch={:.6f}, scale={:.4f}".format(
            #         epoch, float(loss),float(scale)))
            print("Epoch {}, lossPerBatch={:.6f}, scale={:.4f}".format(
                    epoch, float(loss_smpl),float(scale)))
    
    print('----------------------------------------------------lianhe-------------------------------------------------------------------')
    for epoch_sv in range(10000):
        end_recon_data, mu, logvar = vae_model(input_data)
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        loss_sv = F.smooth_l1_loss(scale*Jtr.index_select(1, index["smpl_index"]),end_recon_data.reshape(-1,24,3))
        optimizer_vae.zero_grad()
        optimizer_smpl.zero_grad()
        loss_sv.backward()
        optimizer_vae.step()
        optimizer_smpl.step()
        if loss_sv.item() <= 0.002:
            break
        if epoch_sv % cfg.TRAIN.WRITE == 0 or epoch_sv<10:
            print("Epoch {}, lossPerBatch={:.6f}".format( epoch_sv, float(loss_sv)))

    res = [pose_params, shape_params, scale ,verts, Jtr]
    return res
