"""
config_loader.py

This module just loads a config.yaml file from the configuration folder.
"""
from pathlib import Path
import yaml

def load_config(file_path="config/config.yaml"):
    """Loads a yaml configuration file and returns it"""
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config