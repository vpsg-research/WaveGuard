import torch
import torch.nn.functional as F
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))

from .models.face_parsing.model import BiSeNet

FACE_CLASSES = [1, 2, 3, 4, 5, 10, 11, 12, 13, 17]

_PARSER_MEAN = torch.tensor([0.485, 0.456, 0.406])
_PARSER_STD = torch.tensor([0.229, 0.224, 0.225])

_parser = None


def _get_parser(device):
    global _parser
    if _parser is None:
        ckpt = os.path.join(_current_dir, 'models', '79999_iter.pth')
        _parser = BiSeNet(n_classes=19)
        _parser.load_state_dict(torch.load(ckpt, map_location='cpu'))
        _parser.eval()
    _parser = _parser.to(device)
    return _parser


def _dilate_mask(mask, kernel_size=11):
    B, C, H, W = mask.shape
    k = kernel_size
    kernel = torch.ones(1, 1, k, k, device=mask.device)
    dilated = F.conv2d(mask, kernel, padding=k // 2)
    return (dilated > 0).float()


def get_face_mask(images):
    device = images.device
    B, C, H, W = images.shape
    parser = _get_parser(device)

    # 统一缩放到 256 以上以保证解析准确性
    if H < 256 or W < 256:
        images_256 = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
    else:
        images_256 = images

    # [-1,1] → [0,1] → ImageNet norm
    images_01 = (images_256 + 1.0) / 2.0
    mean = _PARSER_MEAN.to(device).view(1, 3, 1, 1)
    std = _PARSER_STD.to(device).view(1, 3, 1, 1)
    images_norm = (images_01 - mean) / std

    with torch.no_grad():
        out, _, _ = parser(images_norm)  # [B, 19, 256, 256]
        parsing = out.argmax(dim=1)      # [B, 256, 256]

    # 构建人脸 mask
    mask = torch.zeros(B, 256, 256, device=device)
    for cls_id in FACE_CLASSES:
        mask = mask + (parsing == cls_id).float()
    mask = mask.clamp(0, 1).unsqueeze(1)  # [B, 1, 256, 256]

    # 膨胀 mask 产生平滑过渡
    mask = _dilate_mask(mask)

    # 缩回原始尺寸
    if mask.shape[-2:] != (H, W):
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        mask = (mask > 0.5).float()

    return mask


def composite_face(original, fake, fluent=0, mask_from_fake=False):
    src = fake if mask_from_fake else original
    mask = get_face_mask(src)  # [B, 1, H, W]
    effective_mask = mask * fluent
    composited = original * (1 - effective_mask) + fake * effective_mask
    return composited.clamp(-1, 1)
