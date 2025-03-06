import tensorflow as tf
import keras.layers as layers

def build_model(board_width, board_height):

    inputs = tf.keras.Input(shape=(board_height, board_width, 2))
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    # x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    # x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    common = layers.Dense(64, activation='relu')(x)

    # Policy head
    policy_logits = layers.Dense(board_width)(common)
    policy_out = layers.Softmax(name='policy')(policy_logits)

    # Value head
    value_hidden = layers.Dense(32, activation='relu')(common)
    value_out = layers.Dense(1, activation='tanh', name='value')(value_hidden)

    model = tf.keras.Model(inputs=inputs, outputs=[policy_out, value_out])

    return model


def train(model, input_data, target_data):
    """
    expects the data to already be in the right format
    """
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'mean_squared_error'])
    model.fit(x=input_data, y=target_data, epochs=10, batch_size=32)

class Model:
    def __init__(self, board_width, board_height):

        self.model = build_model(board_width, board_height)
        self.prediction_hash_map = {}
        self.times_used_hash = 0
        self.total_predict_calls = 0

    def get_times_used_hash(self):
        return self.times_used_hash
    
    def get_total_predict_calls(self):
        return self.total_predict_calls
    
    def clear_hash_map(self):
        self.prediction_hash_map = {}
        self.times_used_hash = 0

    def predict(self, state_input, board_hash, verbose=0):
        """
        requires state_input to be of shape (batch_size, board_height, board_width, 2)
        """

        policy, value = self.model.predict(state_input, verbose=verbose)

        # use the hash map to make less network calls        
        if board_hash in self.prediction_hash_map:
            self.times_used_hash += 1
            return self.prediction_hash_map[board_hash]
        else:
            self.total_predict_calls += 1
            self.prediction_hash_map[board_hash] = (policy, value)


        return policy, value

    def train(self, input_data, target_data):
        train(self.model, input_data, target_data)

    def save(self, path="model/model.h5"):
        self.model.save(path)