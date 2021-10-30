import tensorflow as tf

def get_data(validation_size):
    mnist_data = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist_data.load_data()

    # create a validation dataset from the full training dataset
    # Scale the data between 0 to 1 by deviding it by 255. its an unsigned data between 0 to 255 range
    x_train, x_valid = x_train[:validation_size] / 255.0, x_test[:validation_size] / 255.0
    y_train, y_valid = y_train[:validation_size], y_test[:validation_size]
    
    # Scling the test data between 0 to 1
    x_test=x_test/255.0

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)