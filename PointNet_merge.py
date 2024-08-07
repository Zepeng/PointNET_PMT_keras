import tensorflow as tf
from tensorflow.keras import layers, models
from qkeras import QConv1D, QDense, QActivation
from qkeras import quantized_bits

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
    
    # Begin PointNetfeat operations
    x = QConv1D(64, 1, activation=None, kernel_quantizer=quantized_bits(8, 0), bias_quantizer=quantized_bits(8, 0))(input_points)
    x = QActivation('quantized_relu(8,0)')(layers.Dropout(enc_dropout)(x))
    x = QActivation('quantized_relu(8,0)')(layers.Dropout(enc_dropout)(QConv1D(int(128 / dim_reduce_factor), 1, kernel_quantizer=quantized_bits(8, 0), bias_quantizer=quantized_bits(8, 0))(x)))
    x = QConv1D(int(1024 / dim_reduce_factor), 1, kernel_quantizer=quantized_bits(8, 0), bias_quantizer=quantized_bits(8, 0))(x)
    global_stats = layers.GlobalAveragePooling1D()(x)
    # End PointNetfeat operations
    
    x = global_stats
    x = QDense(int(512 / dim_reduce_factor), activation=None, kernel_quantizer=quantized_bits(8, 0), bias_quantizer=quantized_bits(8, 0))(x)
    x = QActivation('quantized_relu(8,0)')(layers.Dropout(dec_dropout)(x))
    x = QDense(int(128 / dim_reduce_factor), activation=None, kernel_quantizer=quantized_bits(8, 0), bias_quantizer=quantized_bits(8, 0))(x)
    x = QActivation('quantized_relu(8,0)')(layers.Dropout(dec_dropout)(x))
    outputs = QDense(out_dim, kernel_quantizer=quantized_bits(8, 0), bias_quantizer=quantized_bits(8, 0))(x)

    model = models.Model(input_points, outputs)
    return model
"""
# Example usage:
class Args:
    enc_dropout = 0.3
    dec_dropout = 0.3

args = Args()
n_hits = 10  # Example value
dim = 6  # Example value, should be the total dimensions of the data
dim_reduce_factor = 1
out_dim = 3  # Example value, should be the desired output dimensions

# Create the QKeras model
qkeras_model = PointClassifier(n_hits, dim, dim_reduce_factor, out_dim, args.enc_dropout, args.dec_dropout)
qkeras_model.summary()

# Compile the model
qkeras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
import numpy as np
# Generate dummy data for training
X_train = np.random.rand(100, n_hits, dim).astype(np.float32)
y_train = np.random.randint(0, out_dim, size=(100, out_dim)).astype(np.float32)

# Train the model
qkeras_model.fit(X_train, y_train, epochs=10, batch_size=32)
"""
