import pandas as pd
import torch
import torch.nn as nn
from network import noise
from network.encoder import Encoder
from network.decoder import Decoder
from tabulate import tabulate
from config import training_config as cfg
from utils.Quality import psnr, ssim
from random import randint
from torch import optim
from datetime import datetime
from utils.DataLoad_highpass import *
from utils.torch_utils import decoded_message_error_rate_batch
from dataloader import train_dataloader, val_dataloader
import json
import random
import numpy as np
from utils import DTCWT_highpass
from tqdm import tqdm
import warnings
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
    images_Y = images[:, [0], :, :]
    images_U = images[:, [1], :, :]
    images_V = images[:, [2], :, :]
    low_pass, high_pass = DTCWT_highpass.images_U_dtcwt_with_low(images_U)
    return images_Y, images_U, images_V, low_pass, high_pass

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

    def fit(self, log_dir=False, batch_size=cfg.batch_size, lr=float(cfg.lr), epochs=cfg.epochs):
        if not log_dir:
            log_dir = f'ablation_exp/wo_gnn/{(datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))}'
        os.makedirs(log_dir, exist_ok=True)

        train = train_dataloader
        val = val_dataloader

        optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=0.00001)
        optimizer_decoder_t = optim.Adam(self.decoder_t.parameters(), lr=lr, weight_decay=0.00001)
        optimizer_decoder_d = optim.Adam(self.decoder_d.parameters(), lr=lr, weight_decay=0.00001)

        with open(os.path.join(log_dir, "config.json"), "wt") as out:
            out.write(json.dumps(cfg,indent=2, default=lambda o: str(o)))

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

        def add_noise(input, u_embedded, type):
            if type == "all":
                choice = randint(0, 10)
            elif type == "common":
                choice = randint(0, 7)
            elif type == "deepfake":
                choice = randint(8, 10)

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

        def decode(u_embedded, decoder, indices_decoder):
            high_pass_extract = DTCWT_highpass.images_U_dtcwt_without_low(u_embedded)
            selected_areas_extract = torch.index_select(high_pass_extract[1], 2, indices_decoder)
            selected_areas_extract = selected_areas_extract[:, :, :, :, :, 0].squeeze(1)
            return decoder(selected_areas_extract)

        def validation_attack(input, u_embedded, noise, decoder, indices_decoder, type="default"):
            noised = noise(input, type)
            wm = decode(u_embedded + noised, decoder, indices_decoder)
            return wm

        for epoch in range(1, epochs + 1):
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
            }
            self.encoder.train()
            self.decoder_t.train()
            self.decoder_d.train()
            iterator = tqdm(train)
            cur_lr = 0.0

            for step, (cover_images, mask) in enumerate(iterator):
                cover_images = cover_images.to(cfg.device)
                mask = mask.to(cfg.device)

                Y, U, V, low_pass, high_pass = preprocess(cover_images)

                for param_group in optimizer_encoder.param_groups:
                    cur_lr = param_group["lr"]

                lr_decay(cur_lr, epoch, optimizer_encoder)
                lr_decay(cur_lr, epoch, optimizer_decoder_t)
                lr_decay(cur_lr, epoch, optimizer_decoder_d)

                watermark = torch.Tensor(np.random.choice([-cfg.message_range, cfg.message_range], (cover_images.shape[0], cfg.message_length))).to(cfg.device)
                selected_areas_embed = torch.index_select(high_pass[1], 2, indices_encoder)[:, :, :, :, :, 0].squeeze(1)
                high_pass[1][:, :, indices_encoder, :, :, 0] = self.encoder(selected_areas_embed, watermark).unsqueeze(1)
                u_embedded = DTCWT_highpass.dtcwt_images_U(low_pass, high_pass)

                watermarked_images = torch.cat([Y, u_embedded, V], dim=1)

                forward_u_embedded = u_embedded.clone().detach()
                forward_watermarked_images = watermarked_images.clone().detach()
                forward_cover_images = cover_images.clone().detach()
                forward_mask = mask.clone().detach()
                input = [forward_u_embedded, forward_watermarked_images, forward_cover_images, forward_mask]
                
                u_embedded_attack_type_all = add_noise(input, u_embedded, type="all")
                u_embedded_attack_type_common = add_noise(input, u_embedded, type="common")
                u_embedded_attack_type_df = add_noise(input, u_embedded, type="deepfake")

                extract_wm_all = decode(u_embedded_attack_type_all, self.decoder_t, indices_decoder_t)
                extract_wm_commom = decode(u_embedded_attack_type_common, self.decoder_d, indices_decoder_d)
                extract_wm_df = decode(u_embedded_attack_type_df, self.decoder_d, indices_decoder_d)

                mse = nn.MSELoss().to(cfg.device)
                loss_encoder = mse(U, u_embedded)
                loss_noise_all = mse(extract_wm_all, watermark)
                loss_nosie_common = mse(extract_wm_commom, watermark)
                loss_noise_df = mse(extract_wm_df, torch.zeros_like(watermark))
                loss_total = loss_encoder * cfg.encoder_w + loss_noise_all * cfg.all_w + loss_nosie_common * cfg.common_w

                optimizer_encoder.zero_grad()
                optimizer_decoder_t.zero_grad()
                optimizer_decoder_d.zero_grad()
                loss_total.backward()
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

            self.encoder.eval()
            self.decoder_t.eval()
            self.decoder_d.eval()
            iterator = tqdm(val)
            with torch.no_grad():
                for step, (images, mask) in enumerate(iterator):
                    cover_images = images.to(cfg.device)
                    mask = mask.to(cfg.device)
                    Y, U, V, low_pass, high_pass = preprocess(cover_images)

                    watermark = torch.Tensor(np.random.choice([-cfg.message_range, cfg.message_range], (cover_images.shape[0], cfg.message_length))).to(cfg.device)
                    selected_areas_embed = torch.index_select(high_pass[1], 2, indices_encoder)
                    selected_areas_embed = selected_areas_embed[:, :, :, :, :, 0].squeeze(1)
                    ans = self.encoder(selected_areas_embed, watermark)
                    ans = ans.unsqueeze(1)
                    high_pass[1][:, :, indices_encoder, :, :, 0] = ans
                    u_embedded = DTCWT_highpass.dtcwt_images_U(low_pass, high_pass)
                    watermarked_images = torch.cat([Y, u_embedded, V], dim=1)

                    forward_u_embedded = u_embedded.clone().detach()
                    forward_watermarked_images = watermarked_images.clone().detach()
                    forward_cover_images = cover_images.clone().detach()
                    forward_mask = mask.clone().detach()
                    input = [forward_u_embedded, forward_watermarked_images, forward_cover_images, forward_mask]

                    cover_images = cover_images.detach().cpu() 
                    embedded_images = watermarked_images.clamp(-1, 1).detach().cpu() 
                    metrics["val_psnr"].append(psnr(cover_images, embedded_images, batch_size))
                    metrics["val_ssim"].append(ssim(cover_images, embedded_images, batch_size))

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
            torch.save(self, os.path.join(log_dir, f"model_{epoch}.pth"))
            torch.save(self.state_dict(), os.path.join(log_dir, f"model_state_{epoch}.pth"))
        return history


if __name__ == "__main__":
    seed_torch(42)
    model = IWNet()
    # model.load_state_dict(
    #     torch.load(
    #         "/root/autodl-tmp/code_bk/exp_highpass/2025.02.06-18.23.19/model_state_5.pth",
    #         map_location="cuda:0",
    #     ),
    #     strict=False
    # )
    model.fit()
