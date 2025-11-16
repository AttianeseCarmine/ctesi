import torch
from torch import Tensor
from torchvision.transforms import ColorJitter as _ColorJitter
import torchvision.transforms.functional as TF
import numpy as np
from typing import Tuple, Union, Optional, Callable
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

#
# --- DEFINIZIONI DELLE CLASSI DI TRASFORMAZIONE (CORRETTE) ---
#

class Compose(object):
    def __init__(self, transforms: Callable[..., Tuple[Tensor, Tensor]]) -> None:
        self.transforms = transforms

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label


class ToTensor(object):
    """Questa classe è usata da 'crowd.py' ma non da 'build_transforms'"""
    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        image = TF.to_tensor(image)
        label = torch.from_numpy(label).to(torch.float32)
        return image, label


class Normalize(object):
    """Questa classe è usata da 'crowd.py' ma non da 'build_transforms'"""
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        image = TF.normalize(image, self.mean, self.std)
        return image, label


class RandomCrop(object):
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        pad_if_needed: bool = True,
        fill: float = 0,
        padding_mode: str = "constant",
    ) -> None:
        self.size = (size, size) if isinstance(size, int) else size
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img_size: Tuple[int, int], output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        h, w = img_size
        th, tw = output_size
        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")
        if w == tw and h == th:
            return 0, 0, h, w
        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        img_size = TF.get_image_size(image)
        # i = top, j = left, h = height, w = width
        i, j, h, w = self.get_params(img_size, self.size)
        
        # Crop dell'immagine
        image = TF.crop(image, i, j, h, w)
        
        # --- MODIFICA: Trasformazione dei punti (label) ---
        if label.numel() > 0:
            # Filtra i punti che cadono fuori dal crop
            mask = (label[:, 0] >= j) & (label[:, 0] < j + w) & \
                   (label[:, 1] >= i) & (label[:, 1] < i + h)
            label = label[mask]
            
            if label.numel() > 0:
                # Trasla i punti rimasti al nuovo (0,0)
                label[:, 0] -= j
                label[:, 1] -= i
        else:
            # Gestisce il caso di tensore vuoto
            label = torch.zeros((0, 2), dtype=label.dtype)

        return image, label


class Resize(object):
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        self.size = (size, size) if isinstance(size, int) else size
        self.interpolation = interpolation

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        img_size = TF.get_image_size(image) # (h, w)
        original_h, original_w = img_size
        
        # Resize dell'immagine
        image = TF.resize(image, self.size, self.interpolation)
        
        # --- MODIFICA: Scaling dei punti (label) ---
        if label.numel() > 0:
            # Calcola i rapporti di scaling
            scale_w = self.size[1] / original_w
            scale_h = self.size[0] / original_h
            
            # Applica lo scaling alle coordinate
            label[:, 0] *= scale_w  # Scala x
            label[:, 1] *= scale_h  # Scala y

        return image, label


class RandomResizedCrop(object):
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        self.size = (size, size) if isinstance(size, int) else size
        self.scale = scale
        self.interpolation = interpolation

    @staticmethod
    def get_params(
        img_size: Tuple[int, int],
        scale: Tuple[float, float],
    ) -> Tuple[int, int, int, int]:
        h, w = img_size
        area = h * w
        # Manteniamo il ratio fisso come nell'originale
        log_ratio = torch.log(torch.tensor(0.75)) / torch.log(torch.tensor(1.333))

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio, -log_ratio)).item()
            w_ = int(round(torch.sqrt(torch.tensor(target_area * aspect_ratio)).item()))
            h_ = int(round(torch.sqrt(torch.tensor(target_area / aspect_ratio)).item()))
            if 0 < w_ <= w and 0 < h_ <= h:
                i = torch.randint(0, h - h_ + 1, size=(1,)).item()
                j = torch.randint(0, w - w_ + 1, size=(1,)).item()
                return i, j, h_, w_

        # Fallback
        in_ratio = float(w) / float(h)
        if in_ratio < 0.75:
            w_ = w
            h_ = int(round(w / 0.75))
        elif in_ratio > 1.333:
            h_ = h
            w_ = int(round(h * 1.333))
        else:
            w_ = w
            h_ = h
        i = (h - h_) // 2
        j = (w - w_) // 2
        return i, j, h_, w_

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        img_size = TF.get_image_size(image)
        # i = top, j = left, h = height, w = width
        i, j, h, w = self.get_params(img_size, self.scale)
        
        # Resize-crop dell'immagine
        image = TF.resized_crop(image, i, j, h, w, self.size, self.interpolation)
        
        # --- MODIFICA: Trasformazione dei punti (label) ---
        if label.numel() > 0:
            # 1. Filtra i punti
            mask = (label[:, 0] >= j) & (label[:, 0] < j + w) & \
                   (label[:, 1] >= i) & (label[:, 1] < i + h)
            label = label[mask]
            
            if label.numel() > 0:
                # 2. Trasla i punti all'origine del crop
                label[:, 0] -= j
                label[:, 1] -= i
                
                # 3. Scala i punti alla dimensione finale
                label[:, 0] *= self.size[1] / w  # scale x
                label[:, 1] *= self.size[0] / h  # scale y
        else:
            label = torch.zeros((0, 2), dtype=label.dtype)

        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        img_size = TF.get_image_size(image)
        original_w = img_size[1] # Larghezza originale
        
        if torch.rand(1) < self.p:
            # Flip dell'immagine
            image = TF.hflip(image)
            
            # --- MODIFICA: Flip dei punti (label) ---
            if label.numel() > 0:
                # Inverti la coordinata x
                label[:, 0] = original_w - label[:, 0]
                
        return image, label


class Resize2Multiple(object):
    def __init__(
        self,
        multiple: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        self.multiple = multiple
        self.interpolation = interpolation

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        img_size = TF.get_image_size(image)
        h, w = img_size
        h_ = int(np.ceil(h / self.multiple) * self.multiple)
        w_ = int(np.ceil(w / self.multiple) * self.multiple)
        
        image = TF.resize(image, (h_, w_), self.interpolation)
        
        # --- MODIFICA: Scaling dei punti (label) ---
        if label.numel() > 0:
            scale_w = w_ / w
            scale_h = h_ / h
            label[:, 0] *= scale_w
            label[:, 1] *= scale_h

        return image, label


class ZeroPad2Multiple(object):
    def __init__(self, multiple: int, fill: float = 0, padding_mode: str = "constant") -> None:
        self.multiple = multiple
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        img_size = TF.get_image_size(image)
        h, w = img_size
        h_ = int(np.ceil(h / self.multiple) * self.multiple)
        w_ = int(np.ceil(w / self.multiple) * self.multiple)
        padding = (0, 0, w_ - w, h_ - h) # (left, top, right, bottom)
        
        image = TF.pad(image, padding, self.fill, self.padding_mode)
        
        # --- MODIFICA: Non c'è bisogno di modificare i punti ---
        # Il padding (left=0, top=0) non sposta l'origine (0,0)
        # I punti esistenti mantengono le loro coordinate
        
        return image, label


# --- Trasformazioni che modificano solo l'immagine ---

class ColorJitter(_ColorJitter):
    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        image = super().__call__(image)
        return image, label


class RandomGrayscale(object):
    def __init__(self, p: float = 0.1) -> None:
        self.p = p

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < self.p:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)
        return image, label


class GaussianBlur(object):
    def __init__(self, kernel_size: int, sigma: Optional[Tuple[float, float]] = None, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < self.p:
            image = TF.gaussian_blur(image, self.kernel_size, self.sigma)
        return image, label


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
        if torch.rand(1) < 0.5:
            salt = torch.rand(image.shape[-2:]) < self.saltiness
            image[..., salt] = 1.0
        if torch.rand(1) < 0.5:
            pepper = torch.rand(image.shape[-2:]) < self.spiciness
            image[..., pepper] = 0.0
        return image, label

#
# --- FUNZIONE BUILDER MODIFICATA ---
#

def build_transforms(input_size: int, aug_config: str, is_train: bool = True):
    """
    Costruisce la pipeline di trasformazioni in base alla configurazione.
    """
    
    # NOTA: ToTensor() e Normalize() sono state RIMOSSE da questa pipeline
    # perché 'crowd.py' le applica manualmente.
    
    if is_train:
        if aug_config == 'aug_config_1':
            transforms_list = [
                RandomResizedCrop(
                    size=(input_size, input_size),
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC
                ),
                RandomHorizontalFlip(p=0.5),
                ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
                RandomGrayscale(p=0.1),
                GaussianBlur(kernel_size=3, p=0.3),
            ]
        else:
            raise ValueError(f"Configurazione di augmentation '{aug_config}' non riconosciuta.")
            
    else:
        # Pipeline per Validazione/Test
        transforms_list = [
            Resize(
                size=(input_size, input_size),
                interpolation=InterpolationMode.BICUBIC
            ),
        ]

    return Compose(transforms_list)