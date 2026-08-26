import pandas as pd
import torch
import torch.nn as nn
import os
from network import noise
from network.encoder import Encoder
from network.decoder import Decoder
from tabulate import tabulate
from config import training_config as cfg
from utils.Quality import psnr, ssim
import random
from random import randint
from torch import optim
from datetime import datetime
# from utils.DataLoad_highpass import *
from utils.torch_utils import decoded_message_error_rate_batch
from dataloader import train_dataloader, val_dataloader
import json
from tqdm import tqdm
import warnings
from network.gnn import GNNModel, build_graph
import numpy as np
from utils import DTCWT_highpass
import pytorch_ssim
import sys
import os

# ✅ 添加hififace模块路径（必须指向含有model的目录）
sys.path.append(os.path.join(os.path.dirname(__file__), 'network', 'noise', 'hififace'))


warnings.filterwarnings("ignore")
history = []
indices_encoder = torch.tensor([0, 2]).to(cfg.device)
indices_decoder_d = torch.tensor([0, 2]).to(cfg.device)
indices_decoder_t = torch.tensor([0, 2, 3, 5]).to(cfg.device)

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def preprocess(images):
    images_R = images[:, [0], :, :]
    images_G = images[:, [1], :, :]
    images_B = images[:, [2], :, :]
    low_pass, high_pass = DTCWT_highpass.images_U_dtcwt_with_low(images_B)
    return images_R, images_G, images_B, low_pass, high_pass


#图像从空间域到频率域的转换，为后续模块提供子带信息。
# def preprocess(images):
#     images_Y = images[:, [0], :, :]
#     images_U = images[:, [1], :, :]
#     images_V = images[:, [2], :, :]
#     low_pass, high_pass = DTCWT_highpass.images_U_dtcwt_with_low(images_U)
#     return images_Y, images_U, images_V, low_pass, high_pass

def lr_decay(lr, epoch, opt):
    if epoch == 3:
        for param_group in opt.param_groups:
            param_group["lr"] = 5e-5
    elif epoch == 5:
        for param_group in opt.param_groups:
            param_group["lr"] = 1e-5
    elif epoch == 7:
        for param_group in opt.param_groups:
            param_group["lr"] = 1e-6

class IWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder().to(cfg.device)
        self.decoder_t = Decoder(type="tracer").to(cfg.device)
        self.decoder_d = Decoder(type="detector").to(cfg.device)
        self.gnn = GNNModel().to(cfg.device)

    def fit(self, log_dir=False, batch_size=cfg.batch_size, lr=float(cfg.lr), epochs=cfg.epochs):
        if not log_dir:
            log_dir = f'exp_highpass/{(datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))}'
        os.makedirs(log_dir, exist_ok=True)

        train = train_dataloader
        val = val_dataloader

        optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=0.00001)
        optimizer_decoder_t = optim.Adam(self.decoder_t.parameters(), lr=lr, weight_decay=0.00001)
        optimizer_decoder_d = optim.Adam(self.decoder_d.parameters(), lr=lr, weight_decay=0.00001)

        start_epoch = 1

        # 从配置文件指定的 checkpoint 继续训练
        # 支持两种格式：
        #   1. checkpoint.pth：含 model_state + optimizer + epoch + history（完整续训）
        #   2. model_state_{epoch}.pth / model_{epoch}.pth：纯权重（从头训，仅加载权重）
        resume_path = getattr(cfg, "resume_path", None)
        if resume_path and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=cfg.device)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                # checkpoint 格式，完整恢复
                self.load_state_dict(checkpoint["model_state"], strict=False)
                optimizer_encoder.load_state_dict(checkpoint["optimizer_encoder"])
                optimizer_decoder_t.load_state_dict(checkpoint["optimizer_decoder_t"])
                optimizer_decoder_d.load_state_dict(checkpoint["optimizer_decoder_d"])
                start_epoch = checkpoint["epoch"] + 1
                if "history" in checkpoint:
                    history.extend(checkpoint["history"])
                # 如果配置了 resume_lr，覆盖所有 optimizer 的学习率
                resume_lr = getattr(cfg, "resume_lr", None)
                if resume_lr is not None and resume_lr != "":
                    for opt in [optimizer_encoder, optimizer_decoder_t, optimizer_decoder_d]:
                        for pg in opt.param_groups:
                            pg["lr"] = float(resume_lr)
                    print(f"[Resume] 从 checkpoint (epoch {checkpoint['epoch']}) 恢复，学习率覆盖为 {resume_lr}，继续从 epoch {start_epoch} 训练")
                else:
                    print(f"[Resume] 从 checkpoint (epoch {checkpoint['epoch']}) 恢复，继续从 epoch {start_epoch} 训练")
            else:
                # 纯 state_dict 或完整模型，仅加载权重
                if hasattr(checkpoint, "state_dict"):
                    checkpoint = checkpoint.state_dict()
                self.load_state_dict(checkpoint, strict=False)
                print(f"[Resume] 从权重文件加载模型，从头开始训练")

        with open(os.path.join(log_dir, "config.json"), "wt") as out:
            out.write(json.dumps(cfg,indent=2, default=lambda o: str(o)))

        ssim_module = pytorch_ssim.SSIM().to(cfg.device)

        identity = noise.Identity()
        jpeg = noise.jpeg_compression_train
        resize = noise.Resize()
        medianblur = noise.MedianBlur()
        gau_noise = noise.GaussianNoise()
        gau_blur = noise.GaussianBlur()
        dropout_noise = noise.Dropout()
        salt_pepper_noise = noise.SaltPepper()
        stargan = noise.stargan_noise
        ganimation = noise.ganimation_noise
        simswap = noise.simswap_noise
        uniface_swap = noise.unifaceswap_noise
        fsrt = noise.fsrt_noise
        cscs = noise.cscs_noise
        hififace = noise.hififace_noise
        
        def add_noise(input, u_embedded, type):
            if type == "all":
                choice = randint(0, 14)
            elif type == "common":
                choice = randint(0, 7)
            elif type == "deepfake":
                choice = randint(8, 14)

            if choice == 0:
                return u_embedded + identity(input)
            if choice == 1:
                return u_embedded + jpeg(input)
            if choice == 2:
                return u_embedded + resize(input)
            if choice == 3:
                return u_embedded + medianblur(input)
            if choice == 4:
                return u_embedded + gau_noise(input)
            if choice == 5:
                return u_embedded + gau_blur(input)
            if choice == 6:
                return u_embedded + dropout_noise(input)
            if choice == 7:
                return u_embedded + salt_pepper_noise(input)
            if choice == 8:
                return u_embedded + stargan(input, type)
            if choice == 9:
                return u_embedded + ganimation(input, type)
            if choice == 10:
                return u_embedded + simswap(input, type)
            if choice == 11:
                return u_embedded + uniface_swap(input, type)
            if choice == 12:
                return u_embedded + fsrt(input, type)  
            if choice == 13:
                return u_embedded + cscs(input, type)  
            if choice == 14:
                return u_embedded + hififace(input, type)  

        def decode(u_embedded, decoder, indices_decoder):       
         # 对水印图像进行 DT-CWT 分解（不包含低频）
            high_pass_extract = DTCWT_highpass.images_U_dtcwt_without_low(u_embedded)
             # 步骤 2：从高频子带中提取所需的子带方向
            selected_areas_extract = torch.index_select(high_pass_extract[1], 2, indices_decoder)
            # 步骤 3：去除冗余维度，准备输入解码器
            selected_areas_extract = selected_areas_extract[:, :, :, :, :, 0].squeeze(1)
            return decoder(selected_areas_extract)

        def validation_attack(input, u_embedded, noise, decoder, indices_decoder, type="default"):
            noised = noise(input, type)
            wm = decode(u_embedded + noised, decoder, indices_decoder)
            return wm

        for epoch in range(start_epoch, epochs + 1):
            metrics = {
                "train_loss": [],
                "train_vis": [],
                "train_er_all": [],
                "train_er_common": [],
                "train_er_df": [],
                "val_psnr": [],
                "val_ssim": [],
                "val_jpeg_er_all": [],
                "val_resize_er_all": [],
                "val_medianblur_er_all": [],
                "val_gaublur_er_all": [],
                "val_gauNoise_er_all": [],
                "val_dropout_er_all": [],
                "val_saltPepper_er_all": [],
                "val_identity_er_all": [],
                "val_simswap_er_all": [],
                "val_stargan_er_all": [],
                "val_ganimation_er_all": [],
                "val_jpeg_er_common": [],
                "val_resize_er_common": [],
                "val_medianblur_er_common": [],
                "val_gaublur_er_common": [],
                "val_gauNoise_er_common": [],
                "val_dropout_er_common": [],
                "val_saltPepper_er_common": [],
                "val_identity_er_common": [],
                "val_simswap_er_df": [],
                "val_stargan_er_df": [],
                "val_ganimation_er_df": [],
                "val_unifaceswap_er_all": [],
                "val_unifaceswap_er_df": [],
                "val_fsrt_er_all": [],
                "val_fsrt_er_df": [],
                "val_cscs_er_all": [],
                "val_cscs_er_df": [], 
                "val_hififace_er_all": [],
                "val_hififace_er_df": [],

            }
            self.encoder.train()
            self.decoder_t.train()
            self.decoder_d.train()
            iterator = tqdm(train)

            cur_lr = 0.0

            for step, (cover_images, mask) in enumerate(iterator):
                cover_images = cover_images.to(cfg.device)
                mask = mask.to(cfg.device)

                R, G, B, low_pass, high_pass = preprocess(cover_images)

                for param_group in optimizer_encoder.param_groups:
                    cur_lr = param_group["lr"]

                lr_decay(cur_lr, epoch, optimizer_encoder)
                lr_decay(cur_lr, epoch, optimizer_decoder_t)
                lr_decay(cur_lr, epoch, optimizer_decoder_d)

                """
                嵌入
                """
                watermark = torch.Tensor(np.random.choice([-cfg.message_range, cfg.message_range], (cover_images.shape[0], cfg.message_length))).to(cfg.device)
                selected_areas_embed = torch.index_select(high_pass[1], 2, indices_encoder)[:, :, :, :, :, 0].squeeze(1)
                high_pass[1][:, :, indices_encoder, :, :, 0] = self.encoder(selected_areas_embed, watermark).unsqueeze(1)
                u_embedded = DTCWT_highpass.dtcwt_images_U(low_pass, high_pass)
                
                """
                图特征提取
                """
                u_graphs = build_graph(B)
                u_embeded_graphs = build_graph(u_embedded)
                u_features = []
                u_embeded_features = []
                for carrier_graph, watermarked_graph in zip(u_graphs, u_embeded_graphs):
                    carrier_feature = self.gnn(carrier_graph)
                    watermarked_feature = self.gnn(watermarked_graph)
                    u_features.append(carrier_feature)
                    u_embeded_features.append(watermarked_feature)

                watermarked_images = torch.cat([R,G,u_embedded], dim=1)

                forward_u_embedded = u_embedded.clone().detach()
                forward_watermarked_images = watermarked_images.clone().detach()
                forward_cover_images = cover_images.clone().detach()
                forward_mask = mask.clone().detach()

                input = [forward_u_embedded, forward_watermarked_images, forward_cover_images, forward_mask]
                
                u_embedded_attack_type_all = add_noise(input, u_embedded, type="all")
                u_embedded_attack_type_common = add_noise(input, u_embedded, type="common")
                # DeepFake 路径使用 .detach() 切断对 encoder 的梯度，
                # 使 loss_noise_df 只训练 decoder_d，学会对 DeepFake 输入输出 zeros
                u_embedded_attack_type_df = add_noise(input, u_embedded.detach(), type="deepfake")

                extract_wm_all = decode(u_embedded_attack_type_all, self.decoder_t, indices_decoder_t)
                extract_wm_commom = decode(u_embedded_attack_type_common, self.decoder_d, indices_decoder_d)
                extract_wm_df = decode(u_embedded_attack_type_df, self.decoder_d, indices_decoder_d)

                mse = nn.MSELoss().to(cfg.device)

                loss_gnn = 0
                for carrier_feature, watermarked_feature in zip(u_features, u_embeded_features):
                    loss_gnn += mse(carrier_feature, watermarked_feature)
                loss_gnn /= batch_size
                loss_encoder = mse(B, u_embedded)
                loss_noise_all = mse(extract_wm_all, watermark)
                loss_nosie_common = mse(extract_wm_commom, watermark)
                loss_noise_df = mse(extract_wm_df, torch.zeros_like(watermark))

                loss_rgb = mse(watermarked_images, cover_images) * cfg.rgb_w
                # SSIM期望输入[0,1]，需从[-1,1]转换
                wm_rgb_01 = (watermarked_images.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)
                cv_rgb_01 = (cover_images * 0.5 + 0.5).clamp(0, 1)
                loss_ssim = (1.0 - ssim_module(wm_rgb_01, cv_rgb_01)) * cfg.ssim_w

                # Tracer: all attacks should preserve the watermark.
                # Detector: common attacks should preserve it, while DeepFake
                # attacks should produce logits close to zero (random BER ~= 0.5).
                # loss_noise_df 梯度只到达 decoder_d（u_embedded 已 detach），不影响 encoder。
                loss_total = (
                    loss_encoder * cfg.encoder_w
                    + loss_noise_all * cfg.all_w
                    + loss_nosie_common * cfg.common_w
                    + loss_noise_df * cfg.df_w
                    + loss_gnn * cfg.gnn
                    + loss_rgb
                    + loss_ssim
                )

                optimizer_encoder.zero_grad()
                optimizer_decoder_t.zero_grad()
                optimizer_decoder_d.zero_grad()
                loss_total.backward()
                # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
                # torch.nn.utils.clip_grad_norm_(self.decoder_t.parameters(), 5.0)
                # torch.nn.utils.clip_grad_norm_(self.decoder_d.parameters(), 5.0)
                optimizer_encoder.step()
                optimizer_decoder_t.step()
                optimizer_decoder_d.step()

                metrics["train_loss"].append(loss_total.item())
                metrics["train_vis"].append(loss_encoder.item())
                metrics["train_er_all"].append(decoded_message_error_rate_batch(extract_wm_all, watermark).detach().cpu())
                metrics["train_er_common"].append(decoded_message_error_rate_batch(extract_wm_commom, watermark).detach().cpu())
                metrics["train_er_df"].append(decoded_message_error_rate_batch(extract_wm_df, watermark).detach().cpu())

                iterator.set_description(
                    "Epoch %s | Loss %.6f | Vis %.6f |Er_all %.6f | Er_common %.6f | Er_df %0.6f" % (
                        epoch,
                        np.mean(metrics["train_loss"]),
                        np.mean(metrics["train_vis"]),
                        np.mean(metrics["train_er_all"]),
                        np.mean(metrics["train_er_common"]),
                        np.mean(metrics["train_er_df"]),
                    )
                )
            """
            验证
            """
            self.encoder.eval()
            self.decoder_t.eval()
            self.decoder_d.eval()
            # iterator = tqdm(train, ncols=100, bar_format="{desc}: {n_fmt}/{total_fmt} | {elapsed} < {remaining}")
            iterator = tqdm(val)
            with torch.no_grad():
                for step, (images, mask) in enumerate(iterator):
                    cover_images = images.to(cfg.device)
                    mask = mask.to(cfg.device)
                    R, G, B, low_pass, high_pass = preprocess(cover_images)

                    watermark = torch.Tensor(np.random.choice([-cfg.message_range, cfg.message_range], (cover_images.shape[0], cfg.message_length))).to(cfg.device)
                    selected_areas_embed = torch.index_select(high_pass[1], 2, indices_encoder)
                    selected_areas_embed = selected_areas_embed[:, :, :, :, :, 0].squeeze(1)
                    ans = self.encoder(selected_areas_embed, watermark)
                    ans = ans.unsqueeze(1)
                    high_pass[1][:, :, indices_encoder, :, :, 0] = ans
                    u_embedded = DTCWT_highpass.dtcwt_images_U(low_pass, high_pass)
                    
                    # u_embedded_raw = DTCWT_highpass.dtcwt_images_U(low_pass,high_pass)
                    # u_embedded = u_embedded_raw.clamp(-1, 1)
                    watermarked_images = torch.cat([R, G, u_embedded], dim=1)

                    forward_u_embedded = u_embedded.clone().detach()
                    forward_watermarked_images = watermarked_images.clone().detach()
                    forward_cover_images = cover_images.clone().detach()
                    forward_mask = mask.clone().detach()
                    input = [forward_u_embedded, forward_watermarked_images, forward_cover_images, forward_mask]

                    cover_images = cover_images.detach().cpu() 
                    embedded_images = watermarked_images.clamp(-1, 1).detach().cpu() 
                    metrics["val_psnr"].append(psnr(cover_images, embedded_images))
                    metrics["val_ssim"].append(ssim(cover_images, embedded_images))

                    val_jpeg_wm_all = validation_attack(input, u_embedded, jpeg, self.decoder_t, indices_decoder_t)
                    val_jpeg_wm_common = validation_attack(input, u_embedded, jpeg, self.decoder_d, indices_decoder_d)

                    val_resize_wm_all = validation_attack(input, u_embedded, resize, self.decoder_t, indices_decoder_t)
                    val_resize_wm_common = validation_attack(input, u_embedded, resize, self.decoder_d, indices_decoder_d)

                    val_medianblur_wm_all = validation_attack(input, u_embedded, medianblur, self.decoder_t, indices_decoder_t)
                    val_medianblur_wm_common = validation_attack(input, u_embedded, medianblur, self.decoder_d, indices_decoder_d)

                    val_gaublur_wm_all = validation_attack(input, u_embedded, gau_blur, self.decoder_t, indices_decoder_t)
                    val_gaublur_wm_common = validation_attack(input, u_embedded, gau_blur, self.decoder_d, indices_decoder_d)

                    val_gauNoise_wm_all = validation_attack(input, u_embedded, gau_noise, self.decoder_t, indices_decoder_t)
                    val_gauNoise_wm_common = validation_attack(input, u_embedded, gau_noise, self.decoder_d, indices_decoder_d)

                    val_dropout_wm_all = validation_attack(input, u_embedded, dropout_noise, self.decoder_t, indices_decoder_t)
                    val_dropout_wm_common = validation_attack(input, u_embedded, dropout_noise, self.decoder_d, indices_decoder_d)

                    val_saltPepper_wm_all = validation_attack(input, u_embedded, salt_pepper_noise, self.decoder_t, indices_decoder_t)
                    val_saltPepper_wm_common = validation_attack(input, u_embedded, salt_pepper_noise, self.decoder_d, indices_decoder_d)

                    val_identity_wm_all = validation_attack(input, u_embedded, identity, self.decoder_t, indices_decoder_t)
                    val_identity_wm_common = validation_attack(input, u_embedded, identity, self.decoder_d, indices_decoder_d)

                    val_simswap_wm_all = validation_attack(input, u_embedded, simswap, self.decoder_t, indices_decoder_t, type="all")
                    val_simswap_wm_df = validation_attack(input, u_embedded, simswap, self.decoder_d, indices_decoder_d, type="deepfake")

                    val_ganimation_wm_all = validation_attack(input, u_embedded, ganimation, self.decoder_t, indices_decoder_t, type="all")
                    val_ganimation_wm_df = validation_attack(input, u_embedded, ganimation, self.decoder_d, indices_decoder_d, type="deepfake")

                    val_stargan_wm_all = validation_attack(input, u_embedded, stargan, self.decoder_t, indices_decoder_t, type="all")
                    val_stargan_wm_df = validation_attack(input, u_embedded, stargan, self.decoder_d, indices_decoder_d, type="deepfake")

                    val_unifaceswap_wm_all = validation_attack(input, u_embedded, uniface_swap, self.decoder_t, indices_decoder_t, type="all")
                    val_unifaceswap_wm_df = validation_attack(input, u_embedded, uniface_swap, self.decoder_d, indices_decoder_d, type="deepfake")

                    val_fsrt_wm_all = validation_attack(input, u_embedded, fsrt, self.decoder_t, indices_decoder_t, type="all")
                    val_fsrt_wm_df = validation_attack(input, u_embedded, fsrt, self.decoder_d, indices_decoder_d, type="deepfake")

                    val_cscs_wm_all = validation_attack(input, u_embedded, cscs, self.decoder_t, indices_decoder_t, type="all")
                    val_cscs_wm_df = validation_attack(input, u_embedded, cscs, self.decoder_d, indices_decoder_d, type="deepfake")
                    
                    val_hififace_wm_all = validation_attack(input, u_embedded, hififace, self.decoder_t, indices_decoder_t, type="all")
                    val_hififace_wm_df = validation_attack(input, u_embedded, hififace, self.decoder_d, indices_decoder_d, type="deepfake")

                    metrics["val_jpeg_er_all"].append(decoded_message_error_rate_batch(val_jpeg_wm_all, watermark).detach().cpu())
                    metrics["val_jpeg_er_common"].append(decoded_message_error_rate_batch(val_jpeg_wm_common, watermark).detach().cpu())

                    metrics["val_resize_er_all"].append(decoded_message_error_rate_batch(val_resize_wm_all, watermark).detach().cpu())
                    metrics["val_resize_er_common"].append(decoded_message_error_rate_batch(val_resize_wm_common, watermark).detach().cpu())
                    
                    metrics["val_medianblur_er_all"].append(decoded_message_error_rate_batch(val_medianblur_wm_all, watermark).detach().cpu())
                    metrics["val_medianblur_er_common"].append(decoded_message_error_rate_batch(val_medianblur_wm_common, watermark).detach().cpu())
                    
                    metrics["val_gaublur_er_all"].append(decoded_message_error_rate_batch(val_gaublur_wm_all, watermark).detach().cpu())
                    metrics["val_gaublur_er_common"].append(decoded_message_error_rate_batch(val_gaublur_wm_common, watermark).detach().cpu())
                    
                    metrics["val_gauNoise_er_all"].append(decoded_message_error_rate_batch(val_gauNoise_wm_all, watermark).detach().cpu())
                    metrics["val_gauNoise_er_common"].append(decoded_message_error_rate_batch(val_gauNoise_wm_common, watermark).detach().cpu())
                    
                    metrics["val_dropout_er_all"].append(decoded_message_error_rate_batch(val_dropout_wm_all, watermark).detach().cpu())
                    metrics["val_dropout_er_common"].append(decoded_message_error_rate_batch(val_dropout_wm_common, watermark).detach().cpu())
                    
                    metrics["val_saltPepper_er_all"].append(decoded_message_error_rate_batch(val_saltPepper_wm_all, watermark).detach().cpu())
                    metrics["val_saltPepper_er_common"].append(decoded_message_error_rate_batch(val_saltPepper_wm_common, watermark).detach().cpu())
                    
                    metrics["val_identity_er_all"].append(decoded_message_error_rate_batch(val_identity_wm_all, watermark).detach().cpu())
                    metrics["val_identity_er_common"].append(decoded_message_error_rate_batch(val_identity_wm_common, watermark).detach().cpu())
                    
                    metrics["val_stargan_er_all"].append(decoded_message_error_rate_batch(val_stargan_wm_all, watermark).detach().cpu())
                    metrics["val_stargan_er_df"].append(decoded_message_error_rate_batch(val_stargan_wm_df, watermark).detach().cpu())
                    
                    metrics["val_simswap_er_all"].append(decoded_message_error_rate_batch(val_simswap_wm_all, watermark).detach().cpu())
                    metrics["val_simswap_er_df"].append(decoded_message_error_rate_batch(val_simswap_wm_df, watermark).detach().cpu())
                    
                    metrics["val_ganimation_er_all"].append(decoded_message_error_rate_batch(val_ganimation_wm_all, watermark).detach().cpu())
                    metrics["val_ganimation_er_df"].append(decoded_message_error_rate_batch(val_ganimation_wm_df, watermark).detach().cpu())

                    metrics["val_unifaceswap_er_all"].append(decoded_message_error_rate_batch(val_unifaceswap_wm_all, watermark).detach().cpu())
                    metrics["val_unifaceswap_er_df"].append(decoded_message_error_rate_batch(val_unifaceswap_wm_df, watermark).detach().cpu())

                    metrics["val_fsrt_er_all"].append(decoded_message_error_rate_batch(val_fsrt_wm_all, watermark).detach().cpu())
                    metrics["val_fsrt_er_df"].append(decoded_message_error_rate_batch(val_fsrt_wm_df, watermark).detach().cpu())

                    metrics["val_cscs_er_all"].append(decoded_message_error_rate_batch(val_cscs_wm_all, watermark).detach().cpu())
                    metrics["val_cscs_er_df"].append(decoded_message_error_rate_batch(val_cscs_wm_df, watermark).detach().cpu())

                    metrics["val_hififace_er_all"].append(decoded_message_error_rate_batch(val_hififace_wm_all, watermark).detach().cpu())
                    metrics["val_hififace_er_df"].append(decoded_message_error_rate_batch(val_hififace_wm_df, watermark).detach().cpu())


                    print(f"val-epoch-{epoch}: \n")
                    data_vis = [
                        ["PSNR", "SSIM"],
                        [np.mean(metrics["val_psnr"]), np.mean(metrics["val_ssim"])],
                    ]
                    data_err = [
                        ["Attack", "All(er)", "Common(er)", "DF(er)"],
                        ["Jpeg", np.mean(metrics["val_jpeg_er_all"]), np.mean(metrics["val_jpeg_er_common"]), "-"],
                        ["Resize", np.mean(metrics["val_resize_er_all"]), np.mean(metrics["val_resize_er_common"]), "-"],
                        ["MedianBlur", np.mean(metrics["val_medianblur_er_all"]), np.mean(metrics["val_medianblur_er_common"]), "-"],
                        ["Gau_blur", np.mean(metrics["val_gaublur_er_all"]), np.mean(metrics["val_gaublur_er_common"]), "-"],
                        ["Gau_noise", np.mean(metrics["val_gauNoise_er_all"]), np.mean(metrics["val_gauNoise_er_common"]), "-"],
                        ["Dropout", np.mean(metrics["val_dropout_er_all"]), np.mean(metrics["val_dropout_er_common"]), "-"],
                        ["SaltPepper", np.mean(metrics["val_saltPepper_er_all"]), np.mean(metrics["val_saltPepper_er_common"]), "-"],
                        ["Identity", np.mean(metrics["val_identity_er_all"]), np.mean(metrics["val_identity_er_common"]), "-"],
                        ["Simswap", np.mean(metrics["val_simswap_er_all"]), "-", np.mean(metrics["val_simswap_er_df"])],
                        ["StarGan", np.mean(metrics["val_stargan_er_all"]), "-", np.mean(metrics["val_stargan_er_df"])],
                        ["Ganimation", np.mean(metrics["val_ganimation_er_all"]), "-", np.mean(metrics["val_ganimation_er_df"])],
                        ["UniFaceswap", np.mean(metrics["val_unifaceswap_er_all"]), "-", np.mean(metrics["val_unifaceswap_er_df"])],
                        ["FSRT", np.mean(metrics["val_fsrt_er_all"]), "-", np.mean(metrics["val_fsrt_er_df"])],
                        ["CSCS", np.mean(metrics["val_cscs_er_all"]), "-", np.mean(metrics["val_cscs_er_df"])],
                        ["hififace", np.mean(metrics["val_hififace_er_all"]), "-", np.mean(metrics["val_hififace_er_df"])],
                    ]
                    table_str = tabulate(data_vis, headers="firstrow", tablefmt="grid")
                    print(table_str)
                    with open(os.path.join(log_dir, "metrics_table_visual.json"), "at") as file0:
                        print(table_str, file=file0)

                    table_str2 = tabulate(data_err, headers="firstrow", tablefmt="grid")
                    print(table_str2)
                    with open(os.path.join(log_dir, "metrics_table_err.json"), "at") as file1:
                        print(table_str2, file=file1)
            
            metrics = {
                k: round(np.mean(v), 7) if len(v) > 0 else "NaN"
                for k, v in metrics.items()
            }
            metrics["epoch"] = epoch
            metrics["LR"] = cur_lr
            history.append(metrics)
            pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
            with open(os.path.join(log_dir, "metrics.json"), "at") as out:
                out.write(json.dumps(metrics, indent=2, default=lambda o: str(o)))

            # 每个 epoch 保存模型 + checkpoint
            torch.save(self, os.path.join(log_dir, f"model_{epoch}.pth"))
            torch.save(self.state_dict(), os.path.join(log_dir, f"model_state_{epoch}.pth"))
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.state_dict(),
                    "optimizer_encoder": optimizer_encoder.state_dict(),
                    "optimizer_decoder_t": optimizer_decoder_t.state_dict(),
                    "optimizer_decoder_d": optimizer_decoder_d.state_dict(),
                    "history": history,
                },
                os.path.join(log_dir, "checkpoint.pth"),
            )

        return history


if __name__ == "__main__":
    seed_torch(42)
    model = IWNet()
    model.fit()
