# models/__init__.py

# Importa la nuova classe generica e il builder
from .backbone import Backbone, build_backbone

# Gli altri rimangono uguali
from .pi_head import ZIPHead, build_zip_head
from .clip_ebc_model import CLIPEBCModel
from .zip_model import ZIPModel
from .joint_model import ZIPCLIPJointModel

__all__ = [
    'Backbone', 
    'build_backbone',
    'ZIPHead', 
    'build_zip_head',
    'CLIPEBCModel',
    'ZIPModel',
    'ZIPCLIPJointModel',
]