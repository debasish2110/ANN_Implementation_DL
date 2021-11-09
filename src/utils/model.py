import tensorflow as tf
import time
import os

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28, 28], name = "inputLayer"),
        tf.keras.layers.Dense(units=300, activation='relu', name = "hiddenLayer1"),
        tf.keras.layers.Dense(units=100, activation='relu', name = "hiddenLayer2"),
        tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax', name = "outputLayer")
    ]

    model_clf = tf.keras.Sequential(LAYERS)       #model classifier
    model_clf.summary()
    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    return model_clf



def get_unique_filename(file_name):
    """
    Get a unique filename
    """
    unique_filename= time.strftime(f"_%y%m%d_%H%M%S_{file_name}")
    # .h5 is the format used by keras to save the modeel
    return unique_filename


# Save the model
def save_model(model, model_name, model_dir):
    """
    Save the model
    """
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model) 