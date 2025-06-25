import yaml
from pathlib import Path
from typing import Dict, Any

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f) or {}  
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing Yaml_file : {e}") from e

def run_config():
    try:
        config = load_yaml_config(Path("./config/config.yaml"))
    except:
        config = load_yaml_config(Path("./config.yaml"))
    return config

if __name__ == "__main__":
    run_config()