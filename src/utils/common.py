import yaml

def read_config(config_path):
    with open(config_path) as config_file:
        try:
            content = yaml.safe_load(config_file)
            return content
        except yaml.YAMLError as exc:
            print(exc)
