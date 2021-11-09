import tensorflow as tf
import numpy as np
import os
import time

def get_time(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name

def get_callbacks(config, x_train):
    logs = config["logs"]
    unique_dir_name = get_time("tb_logs")
    TENSORBOARD_ROOT_LOG_DIR = os.path.join(logs["log_dir"], logs["TENSORBOARD_ROOT_LOG_DIR"], unique_dir_name)

    os.makedirs(TENSORBOARD_ROOT_LOG_DIR, exist_ok=True)

    # tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_ROOT_LOG_DIR)

    file_writer = tf.summary.create_file_writer(TENSORBOARD_ROOT_LOG_DIR)

    with file_writer.as_default():
        images = np.reshape(x_train[10:30], (-1, 28, 28, 1))  ### <<< (20, 28, 28, 1)
        tf.summary.image("20 Handwritten Digit Samples", images, max_outputs = 25, step=0)
    
    # early checkpoint callback
    params = config['params']
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                                patience = params["patience"],
                                restore_best_weights = params["restore_best_weights"])
    

    artifects = config["artifects"]
    # checkpoint callback
    CKPT_dir = os.path.join(artifects["artifects_dir"], artifects["checkpoint_dir"])
    os.makedirs(CKPT_dir, exist_ok=True)

    CKPT_path = os.path.join(CKPT_dir, "model_ckpt.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_weights_only=True, verbose=1)

    return [tensorboard_callback, checkpoint_callback, early_stopping_callback]