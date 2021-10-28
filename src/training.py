import os
from utils.common import read_config
from utils.data_Management import get_data
import argparse

def training(config_path):
    """
    Training function
    """
    config = read_config(config_path)
    #print(config)
    validation_size = config['params']['validation_data_size']
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_data(validation_size)



if __name__ == '__main__':
    args= argparse.ArgumentParser()      # for parsing the arguments from yaml file
    args.add_argument('--config', "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)