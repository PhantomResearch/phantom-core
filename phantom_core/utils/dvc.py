from typing import Any
import yaml
from pathlib import Path
import importlib.util

def process_config(stage: str, config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """
    Takes a dvc params.yaml filepath and a dvc stage name as input and splits the params/config
    file into base_config, config for that state, and specifically pulls out the debug_mode param
    and prints that it's in debug mode if that's the case.
    """
    if str(config_path).endswith('.py'):
        base_config, stage_config = load_params_from_py(str(config_path), stage)
    else:
        config = yaml.safe_load(Path(config_path).open())
        stage_config = config.get(stage, dict())
        base_config = config.get('base', dict())
    debug_mode = base_config.get('debug_mode', False)
    if debug_mode:
        print('========== debug mode ==========')
    return stage_config, base_config, debug_mode


def load_params_from_py(file_path, stage: str):
    spec = importlib.util.spec_from_file_location("params_module", file_path)
    assert spec is not None
    params_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(params_module)
    return getattr(params_module, 'base'), getattr(params_module, stage)


def insert_into_nested_dict(data, keys, value):
    
    if not data:
        data = dict()
    
    top_pointer = data

    # If there is only one key, insert the value directly
    if len(keys) == 1:
        data[keys[0]] = value
    else:
        # Traverse through the keys to reach the desired level
        while len(keys) > 1:
            key = keys.pop(0)
            if key not in data:
                data[key] = dict()
            data = data[key]
        data[keys[0]] = value

    return top_pointer


def write_to_metrics_yaml(keys: list[str], metrics: dict[str, Any], metrics_file_path: str) -> None:
    metrics_ex = yaml.safe_load(open(metrics_file_path, 'r'))
    updated_metrics = insert_into_nested_dict(metrics_ex, keys, metrics)
    yaml.dump(updated_metrics, open(metrics_file_path, 'w'))