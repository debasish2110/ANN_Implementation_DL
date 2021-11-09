import os
from utils.common import read_config
from utils.data_Management import get_data
from utils.model import create_model, save_model
from utils.callbacks import get_callbacks
import argparse

def training(config_path):
    """
    Training function
    """
    config = read_config(config_path)
    #print(config)
    validation_size = config['params']['validation_data_size']
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_data(validation_size)

    LOSS_FUNCTION=config['params']['loss_function']
    OPTIMIZER=config['params']['optimizer']
    METRICS=config['params']['metrics']
    NUM_CLASSES=config['params']['num_classes']
    
    ann_model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS=config['params']['epochs']
    VALIDATION_SET=(x_valid, y_valid)

    # create callbacks
    CALLBACK_LIST = get_callbacks(config, x_train)

    history = ann_model.fit(x_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_SET, callbacks=CALLBACK_LIST)

    # svving the model
    artifect_dir = config['artifects']['artifects_dir']
    model_name = config['artifects']['model_name']
    model_dir = config['artifects']['model_dir']

    model_dir_path = os.path.join(artifect_dir, model_dir)

    os.makedirs(model_dir_path, exist_ok=True)

    save_model(ann_model, model_name, model_dir_path)


if __name__ == '__main__':
    args= argparse.ArgumentParser()      # for parsing the arguments from yaml file
    args.add_argument('--config', "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)