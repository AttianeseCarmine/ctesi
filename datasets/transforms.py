import random
import math
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import torch
from torch import Tensor
from torchvision.transforms import ColorJitter as _ColorJitter

from typing import Tuple, Union, Optional, Callable

# ==========================================================
# TRANSFORM PIPELINE (MERGED ZIP + CLIP-EBC)
# ==========================================================



def _crop(
    image: Tensor,
    label: Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
) -> Tuple[Tensor, Tensor]:
    image = F.crop(image, top, left, height, width)
    if len(label) > 0:
        label[:, 0] -= left
        label[:, 1] -= top
        label_mask = (label[:, 0] >= 0) & (label[:, 0] < width) & (label[:, 1] >= 0) & (label[:, 1] < height)
        label = label[label_mask]

    return image, label


def _resize(
    image: Tensor,
    label: Tensor,
    height: int,
    width: int,
) -> Tuple[Tensor, Tensor]:
    image_height, image_width = image.shape[-2:]
    image = F.resize(image, (height, width), interpolation=F.InterpolationMode.BICUBIC, antialias=True) if (image_height != height or image_width != width) else image
    if len(label) > 0 and (image_height != height or image_width != width):
        label[:, 0] = label[:, 0] * width / image_width
        label[:, 1] = label[:, 1] * height / image_height
        label[:, 0] = label[:, 0].clamp(min=0, max=width - 1)
        label[:, 1] = label[:, 1].clamp(min=0, max=height - 1)

    return image, label


class Resize(object):
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        assert len(self.size) == 2, f"size should be a tuple (h, w), got {self.size}."

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        return _resize(image, label, self.size[0], self.size[1])


class RandomCrop(object):
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        assert len(self.size) == 2, f"size should be a tuple (h, w), got {self.size}."

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        crop_height, crop_width = self.size
        image_height, image_width = image.shape[-2:]
        assert crop_height <= image_height and crop_width <= image_width, \
            f"crop size should be no larger than image size, got crop size {self.size} and image size {image.shape}."
        
        top = torch.randint(0, image_height - crop_height + 1, (1,)).item()
        left = torch.randint(0, image_width - crop_width + 1, (1,)).item()
        return _crop(image, label, top, left, crop_height, crop_width)



class Compose(object):
    """Applica una lista di trasformazioni in sequenza."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None, den=None):
        for t in self.transforms:
            img, pts, den = t(img, pts, den)
        return img, pts, den

# --- TRASFORMAZIONI BASE ---

class ToTensor(object):
    """Converte PIL/Numpy in Tensor [0,1] e gestisce la densità."""
    def __call__(self, img, pts=None, den=None):
        # 1. Immagine
        img = F.to_tensor(img)
        
        # 2. Densità (se presente, converte in tensore)
        if den is not None:
            if not isinstance(den, torch.Tensor):
                den = torch.from_numpy(den).float()
            if den.dim() == 2:
                den = den.unsqueeze(0) # [H, W] -> [1, H, W]
                
        return img, pts, den

class Normalize(object):
    """Normalizza con Mean/Std (default CLIP)."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, pts=None, den=None):
        img = F.normalize(img, self.mean, self.std)
        return img, pts, den

# --- TRASFORMAZIONI GEOMETRICHE (LOGICA CLIP-EBC) ---

class RandomCrop(object):
    """
    Esegue un ritaglio casuale di dimensione fissa.
    Se l'immagine è più piccola del crop, esegue padding con 0.
    Allinea Immagine, Punti e Mappa di Densità.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, img, pts, den=None):
        w, h = img.size
        
        # 1. Padding (se l'immagine è più piccola del crop size)
        pad_w = max(0, self.size - w)
        pad_h = max(0, self.size - h)
        
        if pad_w > 0 or pad_h > 0:
            # Pad immagine (destra, basso)
            img = F.pad(img, (0, 0, pad_w, pad_h), fill=0)
            # Pad densità (se presente)
            if den is not None:
                den = np.pad(den, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        # Aggiorna dimensioni dopo padding
        w_new, h_new = img.size
        
        # 2. Coordinate casuali per il crop
        i = random.randint(0, h_new - self.size)
        j = random.randint(0, w_new - self.size)
        
        # 3. Crop Immagine
        img = F.crop(img, i, j, self.size, self.size)
        
        # 4. Crop Densità
        if den is not None:
            den = den[i:i+self.size, j:j+self.size]
            
        # 5. Shift e Filtro Punti
        if pts is not None and len(pts) > 0:
            pts = pts.copy()
            pts[:, 0] -= j # Shift X
            pts[:, 1] -= i # Shift Y
            
            # Mantieni solo i punti che cadono nel nuovo crop
            mask = (pts[:, 0] >= 0) & (pts[:, 0] < self.size) & \
                   (pts[:, 1] >= 0) & (pts[:, 1] < self.size)
            pts = pts[mask]
            
        return img, pts, den

class RandomHorizontalFlip(object):
    """Flip orizzontale coerente per Img, Punti e Densità."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, pts, den=None):
        if random.random() < self.p:
            w, h = img.size
            
            # 1. Flip Immagine
            img = F.hflip(img)
            
            # 2. Flip Punti
            if pts is not None and len(pts) > 0:
                pts = pts.copy()
                pts[:, 0] = w - pts[:, 0] # Inverti coordinata X
                # Clip per sicurezza (punti sul bordo esatto)
                mask = (pts[:, 0] >= 0) & (pts[:, 0] < w)
                pts = pts[mask]
                
            # 3. Flip Densità
            if den is not None:
                den = np.fliplr(den).copy() # Numpy flip è su asse 1 (W)
                
        return img, pts, den

class Resize2Multiple(object):
    """
    Ridimensiona l'immagine affinché altezza e larghezza siano multipli di 'base'.
    Fondamentale per ViT (patch size 16) e CLIP.
    """
    def __init__(self, base=16):
        self.base = base

    def __call__(self, img, pts, den=None):
        w, h = img.size
        # Calcola nuove dimensioni (arrotondamento per eccesso)
        new_h = int(math.ceil(h / self.base) * self.base)
        new_w = int(math.ceil(w / self.base) * self.base)
        
        if (new_w, new_h) == (w, h):
            return img, pts, den
            
        # Resize Immagine
        img = img.resize((new_w, new_h), Image.BICUBIC)
        
        # Scala Punti
        scale_w = new_w / w
        scale_h = new_h / h
        
        if pts is not None and len(pts) > 0:
            pts = pts.copy()
            pts[:, 0] *= scale_w
            pts[:, 1] *= scale_h
            
        # Nota: La density map solitamente viene rigenerata dai punti 
        # o non usata nel validation standard, quindi qui non la scaliamo 
        # (interpolare una density map sparsa è rischioso).
        # Se serve, ZIP la rigenererà dai punti scalati.
            
        return img, pts, den

# ==========================================================
# BUILDER
# ==========================================================

def build_transforms(cfg_data, is_train=True):
    # Default: Normalizzazione CLIP (OpenAI)
    # Se nel config non c'è, usa questi valori standard di CLIP
    mean = cfg_data.get('NORM_MEAN', [0.48145466, 0.4578275, 0.40821073])
    std = cfg_data.get('NORM_STD', [0.26862954, 0.26130258, 0.27577711])
    
    transforms_list = []
    
    if is_train:
        # TRAINING:
        # 1. Random Horizontal Flip
        transforms_list.append(RandomHorizontalFlip(p=0.5))
        
        # 2. Random Crop (Fondamentale per patch counting e CLIP)
        # Usa CROP_SIZE dal config (es. 448 o 384 per CLIP)
        crop_size = cfg_data.get('CROP_SIZE', 448) 
        transforms_list.append(RandomCrop(crop_size))
        
    else:
        # VALIDATION:
        # 1. Resize intelligente per ViT (multipli di 16)
        # Non si fa crop in validation per contare tutta l'immagine
        transforms_list.append(Resize2Multiple(base=16))

    # COMUNI:
    # 3. Conversione a Tensor e Normalizzazione
    transforms_list.append(ToTensor())
    transforms_list.append(Normalize(mean, std))
    
    return Compose(transforms_list)


class Resize2Multiple(object):
    """
    Resize the image so that it satisfies:
        img_h = window_h + stride_h * n_h
        img_w = window_w + stride_w * n_w
    """
    def __init__(
        self,
        window_size: Tuple[int, int],
        stride: Tuple[int, int],
    ) -> None:
        window_size = (int(window_size), int(window_size)) if isinstance(window_size, (int, float)) else window_size
        window_size = tuple(window_size)
        stride = (int(stride), int(stride)) if isinstance(stride, (int, float)) else stride
        stride = tuple(stride)
        assert len(window_size) == 2, f"window_size should be a tuple (h, w), got {window_size}."
        assert len(stride) == 2, f"stride should be a tuple (h, w), got {stride}."
        assert all(s > 0 for s in window_size), f"window_size should be positive, got {window_size}."
        assert all(s > 0 for s in stride), f"stride should be positive, got {stride}."
        assert stride[0] <= window_size[0] and stride[1] <= window_size[1], f"stride should be no larger than window_size, got {stride} and {window_size}."
        self.window_size = window_size
        self.stride = stride

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        image_height, image_width = image.shape[-2:]
        window_height, window_width = self.window_size
        stride_height, stride_width = self.stride
        new_height = int(max(round((image_height - window_height) / stride_height), 0) * stride_height + window_height)
        new_width = int(max(round((image_width - window_width) / stride_width), 0) * stride_width + window_width)

        if new_height == image_height and new_width == image_width:
            return image, label
        else:
            return _resize(image, label, new_height, new_width)


class ZeroPad2Multiple(object):
    def __init__(
        self,
        window_size: Tuple[int, int],
        stride: Tuple[int, int],
    ) -> None:
        window_size = (int(window_size), int(window_size)) if isinstance(window_size, (int, float)) else window_size
        window_size = tuple(window_size)
        stride = (int(stride), int(stride)) if isinstance(stride, (int, float)) else stride
        stride = tuple(stride)
        assert len(window_size) == 2, f"window_size should be a tuple (h, w), got {window_size}."
        assert len(stride) == 2, f"stride should be a tuple (h, w), got {stride}."
        assert all(s > 0 for s in window_size), f"window_size should be positive, got {window_size}."
        assert all(s > 0 for s in stride), f"stride should be positive, got {stride}."
        assert stride[0] <= window_size[0] and stride[1] <= window_size[1], f"stride should be no larger than window_size, got {stride} and {window_size}."
        self.window_size = window_size
        self.stride = stride

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        image_height, image_width = image.shape[-2:]
        window_height, window_width = self.window_size
        stride_height, stride_width = self.stride
        new_height = int(max(np.ceil((image_height - window_height) / stride_height), 0) * stride_height + window_height)
        new_width = int(max(np.ceil((image_width - window_width) / stride_width), 0) * stride_width + window_width)

        if new_height == image_height and new_width == image_width:
            return image, label
        else:
            assert new_height >= image_height and new_width >= image_width, f"new size should be no less than the original size, got {new_height} and {new_width}."
            pad_height, pad_width = new_height - image_height, new_width - image_width
            return TF.pad(image, (0, 0, pad_width, pad_height), fill=0), label  # only pad the right and bottom sides so that the label coordinates are not affected


class RandomResizedCrop(object):
    def __init__(
        self,
        size: Tuple[int, int],
        scale: Tuple[float, float] = (0.75, 1.25),
    ) -> None:
        """
        Randomly crop an image and resize it to a given size. The aspect ratio is preserved during this process.
        """
        self.size = size
        self.scale = scale
        assert len(self.size) == 2, f"size should be a tuple (h, w), got {self.size}."
        assert 0 < self.scale[0] <= self.scale[1], f"scale should satisfy 0 < scale[0] <= scale[1], got {self.scale}."

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        out_height, out_width = self.size
        # out_ratio = out_width / out_height

        scale = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()  # if scale < 1, then the image will be zoomed in, otherwise zoomed out
        in_height, in_width = image.shape[-2:]

        # if in_width / in_height < out_ratio:  # Image is too tall
        #     crop_width = int(in_width * scale)
        #     crop_height = int(crop_width / out_ratio)
        # else:  # Image is too wide
        #     crop_height = int(in_height * scale)
        #     crop_width = int(crop_height * out_ratio)

        crop_height, crop_width = int(out_height * scale), int(out_width * scale)

        if crop_height <= in_height and crop_width <= in_width:  # directly crop and resize the image
            top = torch.randint(0, in_height - crop_height + 1, (1,)).item()
            left = torch.randint(0, in_width - crop_width + 1, (1,)).item()

        else:  # resize the image and then crop
            ratio = max(crop_height / in_height, crop_width / in_width)  # keep the aspect ratio
            resize_height, resize_width = int(in_height * ratio) + 1, int(in_width * ratio) + 1  # add 1 to make sure the resized image is no less than the crop size
            image, label = _resize(image, label, resize_height, resize_width)
            
            top = torch.randint(0, resize_height - crop_height + 1, (1,)).item()
            left = torch.randint(0, resize_width - crop_width + 1, (1,)).item()

        image, label = _crop(image, label, top, left, crop_height, crop_width)
        return _resize(image, label, out_height, out_width)
        

class RandomHorizontalFlip(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        assert 0 <= self.p <= 1, f"p should be in range [0, 1], got {self.p}."

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)

            if len(label) > 0:
                label[:, 0] = image.shape[-1] - 1 - label[:, 0]  # if width is 256, then 0 -> 255, 1 -> 254, 2 -> 253, etc.
                label[:, 0] = label[:, 0].clamp(min=0, max=image.shape[-1] - 1)

        return image, label
    

class ColorJitter(object):
    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0.4,
        contrast: Union[float, Tuple[float, float]] = 0.4,
        saturation: Union[float, Tuple[float, float]] = 0.4,
        hue: Union[float, Tuple[float, float]] = 0.2,
    ) -> None:
        self.color_jitter = _ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        return self.color_jitter(image), label
    

class RandomGrayscale(object):
    def __init__(self, p: float = 0.1) -> None:
        self.p = p
        assert 0 <= self.p <= 1, f"p should be in range [0, 1], got {self.p}."

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < self.p:
            image = F.rgb_to_grayscale(image, num_output_channels=3)

        return image, label
    

class GaussianBlur(object):
    def __init__(self, kernel_size: int, sigma: Optional[float] = None) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        return F.gaussian_blur(image, self.kernel_size, self.sigma), label


class RandomApply(object):
    def __init__(self, transforms: Tuple[Callable, ...], p: Union[float, Tuple[float, ...]] = 0.5) -> None:
        self.transforms = transforms
        p = [p] * len(transforms) if isinstance(p, float) else p
        assert all(0 <= p_ <= 1 for p_ in p), f"p should be in range [0, 1], got {p}."
        assert len(p) == len(transforms), f"p should be a float or a tuple of floats with the same length as transforms, got {p}."
        self.p = p

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        for transform, p in zip(self.transforms, self.p):
            if torch.rand(1) < p:
                image, label = transform(image, label)

        return image, label


class PepperSaltNoise(object):
    def __init__(self, saltiness: float = 0.001, spiciness: float = 0.001) -> None:
        self.saltiness = saltiness
        self.spiciness = spiciness
        assert 0 <= self.saltiness <= 1, f"saltiness should be in range [0, 1], got {self.saltiness}."
        assert 0 <= self.spiciness <= 1, f"spiciness should be in range [0, 1], got {self.spiciness}."

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        noise = torch.rand_like(image)
        image = torch.where(noise < self.saltiness, 1., image)  # Salt
        image = torch.where(noise > 1 - self.spiciness, 0., image)    # Pepper
        return image, label