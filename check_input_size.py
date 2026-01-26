#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml

# ----------------------------
# YAML loader che supporta !!python/tuple
# ----------------------------
class SafeTupleLoader(yaml.SafeLoader):
    pass

def _construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))

SafeTupleLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple
)

# ----------------------------
# Utils
# ----------------------------
def find_yaml_in_dir(d: Path) -> Path | None:
    if not d.exists() or not d.is_dir():
        return None
    cands = list(d.glob("*.yaml")) + list(d.glob("*.yml"))
    if not cands:
        return None

    # Preferisci nomi tipo "config*.yaml"
    def score(p: Path):
        name = p.name.lower()
        return (0 if "config" in name else 1, len(name))

    cands = sorted(cands, key=score)
    return cands[0]

def locate_config_from_ckpt(ckpt_path: str) -> Path | None:
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt}")

    # Primo tentativo: stessa cartella del .pth
    d0 = ckpt.parent
    yml = find_yaml_in_dir(d0)
    if yml is not None:
        return yml

    # Secondo tentativo: parent (es: .../stage2_v2/ -> .../vit_b_16/)
    d1 = d0.parent
    yml = find_yaml_in_dir(d1)
    if yml is not None:
        return yml

    # Terzo tentativo: cerca ricorsivamente un config*.y*ml nelle due cartelle (limitato)
    for base in [d0, d1]:
        if base.exists() and base.is_dir():
            for p in sorted(base.rglob("*.yaml")) + sorted(base.rglob("*.yml")):
                if "config" in p.name.lower():
                    return p

    return None

def read_input_size_from_yaml(yml_path: Path) -> int | None:
    with open(yml_path, "r") as f:
        cfg = yaml.load(f, Loader=SafeTupleLoader)

    # supporta sia dict flat che nested
    if isinstance(cfg, dict):
        if "input_size" in cfg:
            return cfg.get("input_size")
        # se qualcuno l'ha annidato (non dovrebbe, ma per sicurezza)
        for k, v in cfg.items():
            if isinstance(v, dict) and "input_size" in v:
                return v.get("input_size")
    return None

def get_input_size_from_ckpt(ckpt_path: str):
    yml = locate_config_from_ckpt(ckpt_path)
    if yml is None:
        return None, None
    inp = read_input_size_from_yaml(yml)
    return inp, yml

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Check input_size from checkpoint config.yaml")
    ap.add_argument("--s1", required=True, help="Path checkpoint .pth (stage1)")
    ap.add_argument("--s2", required=True, help="Path checkpoint .pth (stage2)")
    args = ap.parse_args()

    s1_in, s1_yml = get_input_size_from_ckpt(args.s1)
    s2_in, s2_yml = get_input_size_from_ckpt(args.s2)

    print("==================================================")
    print("📦 CHECK INPUT SIZE FROM CONFIGS")
    print("==================================================")
    print(f"[S1] ckpt: {args.s1}")
    print(f"[S1] yaml: {s1_yml}")
    print(f"[S1] input_size: {s1_in}")
    print("--------------------------------------------------")
    print(f"[S2] ckpt: {args.s2}")
    print(f"[S2] yaml: {s2_yml}")
    print(f"[S2] input_size: {s2_in}")
    print("==================================================")

    if s1_in is None:
        raise RuntimeError(f"Non riesco a leggere input_size per S1 dal yaml: {s1_yml}")
    if s2_in is None:
        raise RuntimeError(f"Non riesco a leggere input_size per S2 dal yaml: {s2_yml}")

    if s1_in != s2_in:
        raise RuntimeError(f"❌ MISMATCH: stage1 input_size={s1_in} vs stage2 input_size={s2_in}")

    print(f"✅ OK: input_size coincide -> {s1_in}")

if __name__ == "__main__":
    main()
