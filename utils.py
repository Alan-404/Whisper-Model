import yaml
from yaml.loader import SafeLoader
import argparse
from dictionary import activation_dict, optimizer_dict

def load_model_config(path: str) -> dict:
    with open(f'{path}') as file:
        data = yaml.load(file, Loader=SafeLoader)
    return data

def set_parameters(args: argparse.Namespace, config: dict, parameters: list) -> argparse.Namespace:
    for param in parameters:
        if args.__dict__[param] is None:
            if param == 'eps' or param == 'learning_rate':
                args.__dict__[param] = float(config[param])
            elif param == 'activation':
                args.__dict__[param] = activation_dict[config[param]]
            elif param == 'optimizer':
                args.__dict__[param] = optimizer_dict[config[param]]
            else:
                args.__dict__[param] = config[param]
        elif param == 'activation':
            args.__dict__[param] = activation_dict[args.__dict__[param].lower()]
        elif param == 'optimizer':
            args.__dict__[param] = optimizer_dict[args.__dict__[param].lower()]
    return args 