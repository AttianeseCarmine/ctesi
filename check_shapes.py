import torch
from models.backbone import Backbone  # adatta se il path è diverso
from model.clip import CLIP_EBC
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run(backbone_name: str, input_size: int = 224, batch: int = 1):
    model = Backbone(backbone_name, pretrained=False, freeze_bn=True).to(DEVICE).eval()
    model_stage2 = CLIP_EBC(backbone_name,freeze_bn=True).to(DEVICE).eval()

    x = torch.randn(batch, 3, input_size, input_size, device=DEVICE)

    with torch.no_grad():
        feat = model(x)

    print(f"\n=== {backbone_name} ===")
    print("output feature map shape:", tuple(feat.shape))
    print("format: (B, C, H, W)")

    B, C, H, W = feat.shape
    print("B=", B, "C=", C, "H=", H, "W=", W)

    # “patch map” esplicita (solo per ViT ha senso chiamarla così)
    if "vit" in backbone_name:
        # ogni cella corrisponde a una patch 16x16 sull'immagine
        print(f"patch grid = {H}x{W}  (expected {input_size//16}x{input_size//16})")

    if "resnet" in backbone_name:
        print(f"stride grid = {H}x{W}  (expected ~{input_size//32}x{input_size//32})")

if __name__ == "__main__":
    run("vit_b_16", input_size=224, batch=2)
    run("resnet50", input_size=224, batch=2)
    run("clip_vit_b")