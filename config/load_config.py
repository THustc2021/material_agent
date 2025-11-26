import yaml, os

def load_config(path: str):
    ## Load the configuration file
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# def check_config(config: dict):
#     for key, value in config.items():
#         _set_if_undefined(key)
#     return 'Loaded config successfully'

config = load_config(os.path.join(os.path.dirname(__file__), "default.yaml"))

WORKING_DIRECTORY = config["WORKING_DIR"]
save_dialogure = config["SAVE_DIALOGUE"]
pseudo_dir = config["PSEUDO_DIR"]