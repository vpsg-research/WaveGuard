from config import training_config as cfg
from network.encoder import Encoder
from network.decoder import Decoder
from network.gnn import GNNModel
from utils.DataLoad_highpass import *
from dataloader import vis_dataloader
import torchvision.transforms as transforms
import os
import torch
from network.noise import noise_for_vis
import numpy as np
from utils import DTCWT_highpass


encoder = Encoder().to(cfg.device)
decoder_t = Decoder(type="tracer").to(cfg.device)
decoder_d = Decoder(type="detector").to(cfg.device)
gnn = GNNModel().to(cfg.device)
encoder.eval()
decoder_t.eval()
decoder_d.eval()
gnn.eval()

vis = vis_dataloader
iterator = tqdm(vis)

indices_encoder = torch.tensor([0, 2]).to(cfg.device)
indices_decoder_d = torch.tensor([0, 2]).to(cfg.device)
indices_decoder_t = torch.tensor([0, 2, 3, 5]).to(cfg.device)

identity = noise_for_vis.Identity()
gau_noise = noise_for_vis.GaussianNoise()
gau_blur = noise_for_vis.GaussianBlur()
salt_pepper_noise = noise_for_vis.SaltPepper()
dropout = noise_for_vis.Dropout()
medianblur = noise_for_vis.MedianBlur()
resize = noise_for_vis.Resize()
jpeg = noise_for_vis.jpeg_compression_train

stargan_male = noise_for_vis.stargan_male
ganimation = noise_for_vis.ganimation
simswap = noise_for_vis.simswap

save_dir = '/root/autodl-tmp/code_bk/vis_exp'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def preprocess(images):
    images_Y = images[:, [0], :, :]
    images_U = images[:, [1], :, :]
    images_V = images[:, [2], :, :]
    low_pass, high_pass = DTCWT_highpass.images_U_dtcwt_with_low(images_U)
    return images_Y, images_U, images_V, low_pass, high_pass

def common_attack(input, noise):
    img_noised_yuv = noise(input)
    img_noised_yuv = img_noised_yuv.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img_noised_np = np.clip(img_noised_yuv, 0, 255).astype(np.uint8)
    img_noised_bgr = cv2.cvtColor(img_noised_np, cv2.COLOR_YUV2BGR)
    return img_noised_bgr

for step, (cover_images, mask, cover_images_ori) in enumerate(iterator):
    cover_images = cover_images.to(cfg.device)
    cover_images_np = cover_images_ori.squeeze(0).numpy()
    Y, U, V, low_pass, high_pass = preprocess(cover_images)

    watermark = torch.Tensor(np.random.choice([-cfg.message_range, cfg.message_range], (cover_images.shape[0], cfg.message_length))).to(cfg.device)
    selected_areas_embed = torch.index_select(high_pass[1], 2, indices_encoder)[:, :, :, :, :, 0].squeeze(1)
    high_pass[1][:, :, indices_encoder, :, :, 0] = encoder(selected_areas_embed, watermark).unsqueeze(1)
    u_embedded = DTCWT_highpass.dtcwt_images_U(low_pass, high_pass)
    watermarked_images_yuv = (torch.cat([Y, u_embedded, V], dim=1) + 1.0) * 127.5
    watermarked_images_np = np.clip(watermarked_images_yuv.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), 0, 255).astype(np.uint8)
    watermarked_images_bgr = cv2.cvtColor(watermarked_images_np, cv2.COLOR_YUV2BGR)
    watermarked_images_rgb = cv2.cvtColor(watermarked_images_bgr, cv2.COLOR_BGR2RGB)
    combined_input = (watermarked_images_yuv, (cover_images + 1.0) * 127.5)

    regular_image_transform = []
    regular_image_transform.append(transforms.ToTensor())
    regular_image_transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    regular_image_transform = transforms.Compose(regular_image_transform)
    df_input = regular_image_transform(watermarked_images_rgb)
    df_input_for_stargan = [df_input, mask]

    ganimation_img = ganimation(df_input, os.path.join(save_dir, f'ganimation_image_{step}.png'))
    simswap_img = simswap(df_input, os.path.join(save_dir, f'simswap_image_{step}.png'))
    stargan_male_img = stargan_male(df_input_for_stargan, os.path.join(save_dir, f'stargan_male_image_{step}.png'))

    img_bgr = watermarked_images_bgr
    img_gau_noise_bgr = common_attack(watermarked_images_yuv, gau_noise)
    img_gau_blur_bgr = common_attack(watermarked_images_yuv, gau_blur)
    img_salt_pepper_bgr = salt_pepper_noise(watermarked_images_bgr)
    img_dropout_bgr = common_attack(combined_input, dropout)
    img_medianblur_bgr = common_attack(watermarked_images_yuv, medianblur)
    img_resize_bgr = common_attack(watermarked_images_yuv, resize)
    img_jpeg_bgr = common_attack(watermarked_images_yuv, jpeg)

    residual = cv2.absdiff(cover_images_np, img_bgr) * 20
    save_path_wm = os.path.join(save_dir, f'watermarked_image_{step}.png')
    save_path_res = os.path.join(save_dir, f'residual_image_{step}.png')
    save_path_gau_noise = os.path.join(save_dir, f'gau_noise_image_{step}.png')
    save_path_gau_blur = os.path.join(save_dir, f'gau_blur_image_{step}.png')
    save_path_salt_pepper = os.path.join(save_dir, f'salt_pepper_image_{step}.png')
    save_path_dropout = os.path.join(save_dir, f'dropout_image_{step}.png')
    save_path_medianblur = os.path.join(save_dir, f'medianblur_image_{step}.png')
    save_path_resize = os.path.join(save_dir, f'resize_image_{step}.png')
    save_path_jpeg = os.path.join(save_dir, f'jpeg_image_{step}.png')
    save_path_ganimation = os.path.join(save_dir, f'ganimation_image_{step}.png')

    cv2.imwrite(save_path_wm, img_bgr)
    cv2.imwrite(save_path_res, residual)
    cv2.imwrite(save_path_gau_noise, img_gau_noise_bgr)
    cv2.imwrite(save_path_gau_blur, img_gau_blur_bgr)
    cv2.imwrite(save_path_salt_pepper, img_salt_pepper_bgr)
    cv2.imwrite(save_path_dropout, img_dropout_bgr)
    cv2.imwrite(save_path_medianblur, img_medianblur_bgr)
    cv2.imwrite(save_path_resize, img_resize_bgr)
    cv2.imwrite(save_path_jpeg, img_jpeg_bgr)

    print(f"Watermarked/Residual/Noised image {step} saved to {save_dir}")


if __name__ == "__main__":
    pass

