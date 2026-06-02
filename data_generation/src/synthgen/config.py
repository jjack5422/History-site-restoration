import yaml


def load_config(path):
    """讀 YAML 設定, 回傳 dict。"""
    with open(path) as fh:
        return yaml.safe_load(fh)
