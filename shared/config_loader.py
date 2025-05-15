"""
config_loader.py

This module just loads a config.yaml file from the configuration folder.
"""
import os
import yaml
from dotenv import load_dotenv

load_dotenv()  # Load from .env file if present

def resolve_env_vars(config):
    """Recursively replace *_evar keys with env var values."""

    if isinstance(config, dict):
        new_config = {}

        for key, value in config.items():
            if key.endswith("_evar") and isinstance(value, str):
                env_value = os.getenv(value)
                if env_value is None:
                    raise EnvironmentError(f"Missing environment variable: {value}")

                new_key = key[:-5]  # strip "_evar"
                new_config[new_key] = env_value

            elif isinstance(config, dict):
                new_config[key] = resolve_env_vars(value)

            else:
                new_config[key] = value

        return new_config

    elif isinstance(config, list):
        return [resolve_env_vars(item) for item in config]

    else:
        return config  # base case: leave other values as-is

def load_config(file_path="config/config.yaml"):
    """Loads a yaml configuration file and returns it"""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    return resolve_env_vars(raw_config)
