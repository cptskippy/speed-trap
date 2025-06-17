"""
config_loader.py

Utility module for loading and parsing a YAML configuration file.
It supports dynamic configuration using environment variables by
replacing keys that end with `_evar` with values from the environment.

Usage:
    config = load_config()  # Loads from config/config.yaml by default
"""

import os
from typing import Any, Union
import yaml
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if present

def resolve_env_vars(config: Any) -> Any:
    """
    Recursively resolves environment variable references in a config dictionary.

    Keys ending with `_evar` are treated as references to environment variables.
    The resolved key will have `_evar` stripped and the value replaced with the
    corresponding environment variable's value.

    Parameters:
        config (Any): The parsed YAML config structure (dict, list, or primitive).

    Returns:
        Any: The config structure with environment variables resolved.
    
    Raises:
        EnvironmentError: If a referenced environment variable is not set.
    """

    if isinstance(config, dict):
        new_config = {}
        for key, value in config.items():
            if key.endswith("_evar") and isinstance(value, str):
                env_value = os.getenv(value)
                if env_value is None:
                    raise EnvironmentError(f"Missing environment variable: {value}")
                new_key = key[:-5]  # strip '_evar'
                new_config[new_key] = env_value
            else:
                new_config[key] = resolve_env_vars(value)
        return new_config

    elif isinstance(config, list):
        return [resolve_env_vars(item) for item in config]

    else:
        return config  # Base case: leave primitive values as-is

def load_config(file_path: str = "config/config.yaml") -> Union[dict, list]:
    """
    Loads a YAML configuration file and resolves environment variable references.

    Parameters:
        file_path (str): Path to the YAML config file. Defaults to 'config/config.yaml'.

    Returns:
        Union[dict, list]: The fully loaded and processed configuration object.
    
    Raises:
        FileNotFoundError: If the YAML file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
        EnvironmentError: If an expected environment variable is not set.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    return resolve_env_vars(raw_config)
