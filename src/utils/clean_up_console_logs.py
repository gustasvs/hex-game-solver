import tensorflow as tf
import os

def clean_up_console_logs():
    # tf._logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['KMP_WARNINGS'] = 'off'
    # tf._logging._logger.setLevel('ERROR')