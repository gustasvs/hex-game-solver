import tensorflow as tf
import keras.layers as layers

def build_model():

    inputs = tf.keras.Input(shape=(6, 7, 2))
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    common = layers.Dense(64, activation='relu')(x)

    # Policy head
    policy_logits = layers.Dense(7)(common)
    policy_out = layers.Softmax(name='policy')(policy_logits)

    # Value head
    value_hidden = layers.Dense(32, activation='relu')(common)
    value_out = layers.Dense(1, activation='tanh', name='value')(value_hidden)

    model = tf.keras.Model(inputs=inputs, outputs=[policy_out, value_out])


def train(model, data):
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'mean_squared_error'])
    model.fit(data, epochs=10, batch_size=32)