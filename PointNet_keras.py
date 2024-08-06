import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def STNkd(channel=None, k=64, dim_reduce_factor=1):
    inputs = tf.keras.Input(shape=(None, channel if channel is not None else k))
    x = layers.Conv1D(int(64//dim_reduce_factor), 1, activation='relu')(inputs)
    x = layers.Conv1D(int(128//dim_reduce_factor), 1, activation='relu')(x)
    x = layers.Conv1D(int(1024//dim_reduce_factor), 1, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(int(512//dim_reduce_factor), activation='relu')(x)
    x = layers.Dense(int(256//dim_reduce_factor), activation='relu')(x)
    x = layers.Dense(k * k)(x)
    
    iden = tf.eye(k, batch_shape=[tf.shape(inputs)[0]])
    x = layers.Reshape((k, k))(x) + iden

    model = models.Model(inputs, x)
    return model

def PointNetfeat(dimensions, dim_reduce_factor, dropout_rate):
    inputs = tf.keras.Input(shape=(None, dimensions))
    x = layers.Conv1D(64, 1, activation=None)(inputs)
    x = layers.ReLU()(layers.Dropout(dropout_rate)(x))
    x = layers.ReLU()(layers.Dropout(dropout_rate)(layers.Conv1D(int(128 / dim_reduce_factor), 1)(x)))
    x = layers.Conv1D(int(1024 / dim_reduce_factor), 1)(x)
    
    global_stats = layers.GlobalAveragePooling1D()(x)
    model = models.Model(inputs, global_stats)
    return model

def PointClassifier(n_hits, dim, dim_reduce_factor, out_dim, enc_dropout, dec_dropout):
    input_points = tf.keras.Input(shape=(n_hits, dim))
    
    def process_data_with_label(x):
        label = x[:, :, 3:4]
        t = x[:, :, 4:5]
        q = x[:, :, 5:6]
        chamfer_x = tf.concat([x[:, :, :3], t, q], axis=-1)
        label = tf.tile(label, [1, 1, chamfer_x.shape[-1]])
        return chamfer_x * label
    
    #x = layers.Lambda(process_data_with_label)(input_points)
    encoder = PointNetfeat(dim-1, dim_reduce_factor, enc_dropout)
    x = encoder(x)
    x = layers.Dense(int(512/dim_reduce_factor), activation='leaky_relu')(x)
    x = layers.Dropout(dec_dropout)(x)
    x = layers.Dense(int(128/dim_reduce_factor), activation='leaky_relu')(x)
    outputs = layers.Dense(out_dim)(x)
    
    model = models.Model(input_points, outputs)
    return model
"""
# Example usage:
# Define args with the necessary parameters
class Args:
    enc_dropout = 0.3
    dec_dropout = 0.3

args = Args()
n_hits = 10  # Example value
dim = 6  # Example value, should be the total dimensions of the data
dim_reduce_factor = 1
out_dim = 3  # Example value, should be the desired output dimensions

# Create the Keras model
keras_model = PointClassifier(n_hits, dim, dim_reduce_factor, out_dim, args.enc_dropout, args.dec_dropout)
keras_model.summary()

# Compile the model
keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Generate dummy data for training
X_train = np.random.rand(100, n_hits, dim).astype(np.float32)
y_train = np.random.randint(0, out_dim, size=(100, out_dim)).astype(np.float32)

# Train the model
keras_model.fit(X_train, y_train, epochs=10, batch_size=32)
"""
