import pandas as pd
import torch
import torch.nn as nn
from network import noise
from ablation.encoder_wo_freq import Encoder
from ablation.decoder_wo_freq import Decoder
from tabulate import tabulate
from config import training_config as cfg
from utils.Quality import psnr, ssim
from datetime import datetime
from utils.DataLoad_highpass import *
from utils.torch_utils import decoded_message_error_rate_batch
from dataloader import val_dataloader
from network.noise import stargan_for_test
import json
from tqdm import tqdm
import warnings
from network.gnn import GNNModel
import random
import numpy as np
from utils import DTCWT_highpass
warnings.filterwarnings("ignore")

history = []
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
    return images_Y, images_U, images_V

class IWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder().to(cfg.device)
        self.decoder_t = Decoder(type="tracer").to(cfg.device)
        self.decoder_d = Decoder(type="detector").to(cfg.device)
        self.gnn = GNNModel().to(cfg.device)

    def fit(self, log_dir=False, batch_size=cfg.batch_size, lr=float(cfg.lr), epochs=cfg.epochs):
        if not log_dir:
            log_dir = f'ablation_test/wo_freq/{(datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))}'
        os.makedirs(log_dir, exist_ok=True)

        val = val_dataloader

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
        stargan = stargan_for_test.stargan_noise
        ganimation = noise.ganimation_noise
        simswap = noise.simswap_noise

        def decode(u_embedded, decoder):
            return decoder(u_embedded)

        def validation_attack(input, u_embedded, noise, decoder, type="default"):
            noised = noise(input, type)
            wm = decode(u_embedded + noised, decoder)
            return wm

        def stargan_attack(input, u_embedded, noise, decoder, type="default", c_trg="3"):
            noised = noise(input, type, c_trg)
            wm = decode(u_embedded + noised, decoder)
            return wm
        
        metrics = {
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
            "val_ganimation_er_all": [],

            "val_stargan_black_hair_er_all": [],
            "val_stargan_blond_hair_er_all": [],
            "val_stargan_brown_hair_er_all": [],
            "val_stargan_male_er_all": [],
            "val_stargan_young_er_all": [],

            "val_jpeg_er_common": [],
            "val_resize_er_common": [],
            "val_medianblur_er_common": [],
            "val_gaublur_er_common": [],
            "val_gauNoise_er_common": [],
            "val_dropout_er_common": [],
            "val_saltPepper_er_common": [],
            "val_identity_er_common": [],
            "val_simswap_er_df": [],
            "val_ganimation_er_df": [],

            "val_stargan_black_hair_er_df": [],
            "val_stargan_blond_hair_er_df": [],
            "val_stargan_brown_hair_er_df": [],
            "val_stargan_male_er_df": [],
            "val_stargan_young_er_df": [],

        }
        data_robust = []
        data_vis = []

        self.encoder.eval()
        self.decoder_t.eval()
        self.decoder_d.eval()
        iterator = tqdm(val)
        with torch.no_grad():
            for step, (images, mask) in enumerate(iterator):
                cover_images = images.to(cfg.device)
                mask = mask.to(cfg.device)
                Y, U, V = preprocess(cover_images)

                watermark = torch.Tensor(np.random.choice([-cfg.message_range, cfg.message_range], (cover_images.shape[0], cfg.message_length))).to(cfg.device)
                u_embedded = self.encoder(U, watermark)
                watermarked_images = torch.cat([Y, u_embedded, V], dim=1)

                forward_u_embedded = u_embedded.clone().detach()
                forward_watermarked_images = watermarked_images.clone().detach()
                forward_cover_images = cover_images.clone().detach()
                forward_mask = mask.clone().detach()
                input = [forward_u_embedded, forward_watermarked_images, forward_cover_images, forward_mask]

                cover_images = cover_images.detach().cpu() + 1.0
                embedded_images = watermarked_images.detach().cpu() + 1.0
                metrics["val_psnr"].append(psnr(cover_images, embedded_images, batch_size))
                metrics["val_ssim"].append(ssim(cover_images, embedded_images, batch_size))

                val_jpeg_er_all = validation_attack(input, u_embedded, jpeg, self.decoder_t)
                val_jpeg_er_common = validation_attack(input, u_embedded, jpeg, self.decoder_d)

                val_resize_er_all = validation_attack(input, u_embedded, resize, self.decoder_t)
                val_resize_er_common = validation_attack(input, u_embedded, resize, self.decoder_d)

                val_medianblur_er_all = validation_attack(input, u_embedded, medianblur, self.decoder_t)
                val_medianblur_er_common = validation_attack(input, u_embedded, medianblur, self.decoder_d)

                val_gaublur_er_all = validation_attack(input, u_embedded, gau_blur, self.decoder_t)
                val_gaublur_er_common = validation_attack(input, u_embedded, gau_blur, self.decoder_d)

                val_gauNoise_er_all = validation_attack(input, u_embedded, gau_noise, self.decoder_t)
                val_gauNoise_er_common = validation_attack(input, u_embedded, gau_noise, self.decoder_d)

                val_dropout_er_all = validation_attack(input, u_embedded, dropout_noise, self.decoder_t)
                val_dropout_er_common = validation_attack(input, u_embedded, dropout_noise, self.decoder_d)

                val_saltPepper_er_all = validation_attack(input, u_embedded, salt_pepper_noise, self.decoder_t)
                val_saltPepper_er_common = validation_attack(input, u_embedded, salt_pepper_noise, self.decoder_d)

                val_identity_er_all = validation_attack(input, u_embedded, identity, self.decoder_t)
                val_identity_er_common = validation_attack(input, u_embedded, identity, self.decoder_d)

                val_simswap_er_all = validation_attack(input, u_embedded, simswap, self.decoder_t, type="all")
                val_simswap_er_df = validation_attack(input, u_embedded, simswap, self.decoder_d, type="deepfake")

                val_ganimation_er_all = validation_attack(input, u_embedded, ganimation, self.decoder_t, type="all")
                val_ganimation_er_df = validation_attack(input, u_embedded, ganimation, self.decoder_d, type="deepfake")

                val_stargan_black_hair_er_all = stargan_attack(input, u_embedded, stargan, self.decoder_t, type="all", c_trg=0)
                val_stargan_blond_hair_er_all = stargan_attack(input, u_embedded, stargan, self.decoder_t, type="all", c_trg=1)
                val_stargan_brown_hair_er_all = stargan_attack(input, u_embedded, stargan, self.decoder_t, type="all", c_trg=2)
                val_stargan_male_er_all = stargan_attack(input, u_embedded, stargan, self.decoder_t, type="all", c_trg=3)
                val_stargan_young_er_all = stargan_attack(input, u_embedded, stargan, self.decoder_t, type="all", c_trg=4)                
                val_stargan_black_hair_er_df = stargan_attack(input, u_embedded, stargan, self.decoder_d, type="deepfake", c_trg=0)
                val_stargan_blond_hair_er_df = stargan_attack(input, u_embedded, stargan, self.decoder_d, type="deepfake", c_trg=1)
                val_stargan_brown_hair_er_df = stargan_attack(input, u_embedded, stargan, self.decoder_d, type="deepfake", c_trg=2)
                val_stargan_male_er_df = stargan_attack(input, u_embedded, stargan, self.decoder_d, type="deepfake", c_trg=3)
                val_stargan_young_er_df = stargan_attack(input, u_embedded, stargan, self.decoder_d, type="deepfake", c_trg=4)

                metrics["val_jpeg_er_all"].append(decoded_message_error_rate_batch(val_jpeg_er_all, watermark).detach().cpu())
                metrics["val_jpeg_er_common"].append(decoded_message_error_rate_batch(val_jpeg_er_common, watermark).detach().cpu())

                metrics["val_resize_er_all"].append(decoded_message_error_rate_batch(val_resize_er_all, watermark).detach().cpu())
                metrics["val_resize_er_common"].append(decoded_message_error_rate_batch(val_resize_er_common, watermark).detach().cpu())
                
                metrics["val_medianblur_er_all"].append(decoded_message_error_rate_batch(val_medianblur_er_all, watermark).detach().cpu())
                metrics["val_medianblur_er_common"].append(decoded_message_error_rate_batch(val_medianblur_er_common, watermark).detach().cpu())
                
                metrics["val_gaublur_er_all"].append(decoded_message_error_rate_batch(val_gaublur_er_all, watermark).detach().cpu())
                metrics["val_gaublur_er_common"].append(decoded_message_error_rate_batch(val_gaublur_er_common, watermark).detach().cpu())
                
                metrics["val_gauNoise_er_all"].append(decoded_message_error_rate_batch(val_gauNoise_er_all, watermark).detach().cpu())
                metrics["val_gauNoise_er_common"].append(decoded_message_error_rate_batch(val_gauNoise_er_common, watermark).detach().cpu())
                
                metrics["val_dropout_er_all"].append(decoded_message_error_rate_batch(val_dropout_er_all, watermark).detach().cpu())
                metrics["val_dropout_er_common"].append(decoded_message_error_rate_batch(val_dropout_er_common, watermark).detach().cpu())
                
                metrics["val_saltPepper_er_all"].append(decoded_message_error_rate_batch(val_saltPepper_er_all, watermark).detach().cpu())
                metrics["val_saltPepper_er_common"].append(decoded_message_error_rate_batch(val_saltPepper_er_common, watermark).detach().cpu())
                
                metrics["val_identity_er_all"].append(decoded_message_error_rate_batch(val_identity_er_all, watermark).detach().cpu())
                metrics["val_identity_er_common"].append(decoded_message_error_rate_batch(val_identity_er_common, watermark).detach().cpu())
                
                metrics["val_stargan_black_hair_er_all"].append(decoded_message_error_rate_batch(val_stargan_black_hair_er_all, watermark).detach().cpu())
                metrics["val_stargan_blond_hair_er_all"].append(decoded_message_error_rate_batch(val_stargan_blond_hair_er_all, watermark).detach().cpu())
                metrics["val_stargan_brown_hair_er_all"].append(decoded_message_error_rate_batch(val_stargan_brown_hair_er_all, watermark).detach().cpu())
                metrics["val_stargan_male_er_all"].append(decoded_message_error_rate_batch(val_stargan_male_er_all, watermark).detach().cpu())
                metrics["val_stargan_young_er_all"].append(decoded_message_error_rate_batch(val_stargan_young_er_all, watermark).detach().cpu())

                metrics["val_stargan_black_hair_er_df"].append(decoded_message_error_rate_batch(val_stargan_black_hair_er_df, watermark).detach().cpu())
                metrics["val_stargan_blond_hair_er_df"].append(decoded_message_error_rate_batch(val_stargan_blond_hair_er_df, watermark).detach().cpu())
                metrics["val_stargan_brown_hair_er_df"].append(decoded_message_error_rate_batch(val_stargan_brown_hair_er_df, watermark).detach().cpu())
                metrics["val_stargan_male_er_df"].append(decoded_message_error_rate_batch(val_stargan_male_er_df, watermark).detach().cpu())
                metrics["val_stargan_young_er_df"].append(decoded_message_error_rate_batch(val_stargan_young_er_df, watermark).detach().cpu())

                
                metrics["val_simswap_er_all"].append(decoded_message_error_rate_batch(val_simswap_er_all, watermark).detach().cpu())
                metrics["val_simswap_er_df"].append(decoded_message_error_rate_batch(val_simswap_er_df, watermark).detach().cpu())
                
                metrics["val_ganimation_er_all"].append(decoded_message_error_rate_batch(val_ganimation_er_all, watermark).detach().cpu())
                metrics["val_ganimation_er_df"].append(decoded_message_error_rate_batch(val_ganimation_er_df, watermark).detach().cpu())

                data_robust = [
                    ["Noice", "All(er)", "Common(er)", "DF(er)"],
                    ["Identity", np.mean(metrics["val_identity_er_all"]), np.mean(metrics["val_identity_er_common"]), "-"],
                    ["Jpeg", np.mean(metrics["val_jpeg_er_all"]), np.mean(metrics["val_jpeg_er_common"]), "-"],
                    ["Resize", np.mean(metrics["val_resize_er_all"]), np.mean(metrics["val_resize_er_common"]), "-"],
                    ["MedianBlur", np.mean(metrics["val_medianblur_er_all"]), np.mean(metrics["val_medianblur_er_common"]), "-"],
                    ["Gau_blur", np.mean(metrics["val_gaublur_er_all"]), np.mean(metrics["val_gaublur_er_common"]), "-"],
                    ["Gau_noise", np.mean(metrics["val_gauNoise_er_all"]), np.mean(metrics["val_gauNoise_er_common"]), "-"],
                    ["Dropout", np.mean(metrics["val_dropout_er_all"]), np.mean(metrics["val_dropout_er_common"]), "-"],
                    ["SaltPepper", np.mean(metrics["val_saltPepper_er_all"]), np.mean(metrics["val_saltPepper_er_common"]), "-"],
                    ["Simswap", np.mean(metrics["val_simswap_er_all"]), "-", np.mean(metrics["val_simswap_er_df"])],
                    ["StarGan_black_hair", np.mean(metrics["val_stargan_black_hair_er_all"]), "-", np.mean(metrics["val_stargan_black_hair_er_df"])],
                    ["StarGan_blond_hair", np.mean(metrics["val_stargan_blond_hair_er_all"]), "-", np.mean(metrics["val_stargan_blond_hair_er_df"])],
                    ["StarGan_brown_hair", np.mean(metrics["val_stargan_brown_hair_er_all"]), "-", np.mean(metrics["val_stargan_brown_hair_er_df"])],
                    ["StarGan_male", np.mean(metrics["val_stargan_male_er_all"]), "-", np.mean(metrics["val_stargan_male_er_df"])],
                    ["StarGan_young", np.mean(metrics["val_stargan_young_er_all"]), "-", np.mean(metrics["val_stargan_young_er_df"])],
                    ["Ganimation", np.mean(metrics["val_ganimation_er_all"]), "-", np.mean(metrics["val_ganimation_er_df"])],
                ]
                data_vis = [
                    ["PSNR", "SSIM"],
                    [np.mean(metrics["val_psnr"]), np.mean(metrics["val_ssim"])],
                ]

                table_str_robust = tabulate(data_robust, headers="firstrow", tablefmt="grid")
                print(table_str_robust)

                table_str_vis = tabulate(data_vis, headers="firstrow", tablefmt="grid")
                print(table_str_vis)

        table_str_robust = tabulate(data_robust, headers="firstrow", tablefmt="grid")
        print(table_str_robust)
        with open(os.path.join(log_dir, "metrics_table_er_average.json"), "at") as file0:
            print(table_str_robust, file=file0)

        table_str_vis = tabulate(data_vis, headers="firstrow", tablefmt="grid")
        print(table_str_vis)
        with open(os.path.join(log_dir, "metrics_table_visual_average.json"), "at") as file1:
            print(table_str_vis, file=file1)
        
        metrics = {
            k: round(np.mean(v), 7) if len(v) > 0 else "NaN"
            for k, v in metrics.items()
        }
        history.append(metrics)
        pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
        with open(os.path.join(log_dir, "metrics.json"), "at") as out:
            out.write(json.dumps(metrics, indent=2, default=lambda o: str(o)))

        return history


if __name__ == "__main__":
    seed_torch(42)
    model = IWNet()
    model.load_state_dict(
        torch.load(
            "/root/autodl-tmp/code_bk/ablation_exp/wo_freq/2025.03.03-10.47.45/model_state_7.pth",
            map_location="cuda:0",
        ),
        strict=False
    )
    model.fit()
