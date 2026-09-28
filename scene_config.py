"""Shared, portable scene configuration (paths are relative to the JSON file)."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "data": {"reconstruction": "", "images_dir": "", "sfm_max_side": 1600, "max_side": 800},
    "scene": {"box_min": None, "box_max": None, "near": 0.3},
    "training": {"directory": "../training/test_dataset", "model_id": "test_dataset",
                 "steps": 10000, "period": 100, "batch_rays": 512,
                 "samples_per_ray": 64, "lr": 0.0001, "lmd": 0.0001, "seed": 777},
    "preview": {"enabled": True, "host": "127.0.0.1", "port": 8080,
                "width": 360, "height": 640, "downscale": 5,
                "samples_per_ray": 64, "views": [0, 1, 2],
                "save_seconds": 120, "image_seconds": 30},
    "viewer": {"width": 540, "height": 960, "block_size": 12,
               "samples_per_ray": 64, "camera": 0, "move_speed": 1.0},
}


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    cfg = copy.deepcopy(DEFAULTS)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object")
    for section, values in raw.items():
        if section not in cfg or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid config section: {section}")
        for key, value in values.items():
            if key not in cfg[section]:
                raise ValueError(f"Unknown config key: {section}.{key}")
            cfg[section][key] = value
    for section, key in [("data", "reconstruction"), ("data", "images_dir"), ("training", "directory")]:
        value = cfg[section][key]
        if value:
            cfg[section][key] = str((path.parent / value).resolve())
    if not cfg["data"]["reconstruction"]:
        raise ValueError("data.reconstruction is required")
    for section, keys in {"training": ["steps", "period", "batch_rays", "samples_per_ray"],
                          "preview": ["width", "height", "downscale", "samples_per_ray", "port"],
                          "viewer": ["width", "height", "block_size", "samples_per_ray"]}.items():
        for key in keys:
            value = cfg[section][key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for section in ("training", "preview", "viewer"):
        if cfg[section]["samples_per_ray"] > 64:
            raise ValueError(f"{section}.samples_per_ray must be <= 64 for the supplied kernels")
    for key in ("sfm_max_side", "max_side"):
        if type(cfg["data"][key]) is not int or cfg["data"][key] < 0:
            raise ValueError(f"data.{key} must be a nonnegative integer (0 disables resizing)")
    if type(cfg["viewer"]["camera"]) is not int or cfg["viewer"]["camera"] < 0:
        raise ValueError("viewer.camera must be a nonnegative integer")
    if type(cfg["preview"]["enabled"]) is not bool:
        raise ValueError("preview.enabled must be true or false")
    if not isinstance(cfg["preview"]["views"], list):
        raise ValueError("preview.views must be a list of registered-camera indices")
    for section, key in [("scene", "near"), ("training", "lmd")]:
        if not 0 <= cfg[section][key] < float("inf"):
            raise ValueError(f"{section}.{key} must be nonnegative and finite")
    if cfg["preview"]["port"] > 65535:
        raise ValueError("preview.port must be <= 65535")
    v, p, t = cfg["viewer"], cfg["preview"], cfg["training"]
    if v["width"] % v["block_size"] or v["height"] % v["block_size"]:
        raise ValueError("Viewer width and height must be multiples of block_size")
    if p["downscale"] > min(p["width"], p["height"]):
        raise ValueError("Preview downscale exceeds image dimensions")
    if not t["model_id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in t["model_id"]):
        raise ValueError("model_id may contain only letters, digits, _, . and -")
    for section, keys in {"training": ["lr"], "preview": ["save_seconds", "image_seconds"], "viewer": ["move_speed"]}.items():
        for key in keys:
            if not 0 < cfg[section][key] < float("inf"):
                raise ValueError(f"{section}.{key} must be positive and finite")
    return cfg
