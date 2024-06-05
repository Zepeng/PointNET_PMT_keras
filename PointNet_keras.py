import tensorflow as tf
from tensorflow.keras import layers, models, activations, initializers
import argparse

class PointNetfeat(tf.keras.Model):
    '''
    PointNet (Encoder) Implementation
    1. STNKD (DISCARDED due to no improvement)
    2. 1x1 Conv1D layers (Literally a linear layer)
    3. Global Statistics (Mean shows superior performance than Min/Max)
    '''
    def __init__(self, dimensions, dim_reduce_factor, args):
        super(PointNetfeat, self).__init__()
        dr = args.enc_dropout

        self.conv1 = layers.Conv1D(filters=64, kernel_size=1, input_shape=(None, dimensions))
        self.conv2 = layers.Conv1D(filters=int(128 / dim_reduce_factor), kernel_size=1)
        self.conv3 = layers.Conv1D(filters=int(1024 / dim_reduce_factor), kernel_size=1)
        self.dr1 = layers.Dropout(dr)
        self.dr2 = layers.Dropout(dr)
        self.relu = layers.ReLU()

        # recording output dim for construct decoder's input dim
        self.latent_dim = self.conv3.filters

    def stats(self, x):
        meani = tf.reduce_mean(x, axis=2, keepdims=True)
        return meani

    def call(self, x):
        x = self.conv1(x)
        x = self.relu(self.dr1(x))
        x = self.relu(self.dr2(self.conv2(x)))
        x = self.conv3(x)

        global_stats = self.stats(x)
        x = tf.squeeze(global_stats, axis=-1)
        return x

class PointClassifier(tf.keras.Model):
    def __init__(self, n_hits, dim, dim_reduce_factor, out_dim, args):
        '''
        Main Model
        :param n_hits: number of points per point cloud
        :param dim: total dimensions of data (3 spatial + time and/or charge)
        '''
        super(PointClassifier, self).__init__()
        dr = args.dec_dropout
        self.n_hits = n_hits
        self.encoder = PointNetfeat(dimensions=dim-1,
                                    dim_reduce_factor=dim_reduce_factor,
                                    args=args,
                                    )
        self.latent = self.encoder.latent_dim  # dimension from enc to dec
        self.decoder = models.Sequential([
            layers.Dense(units=int(512/dim_reduce_factor)),
            layers.LeakyReLU(),
            layers.Dropout(rate=dr),
            layers.Dense(units=int(128/dim_reduce_factor)),
            layers.LeakyReLU(),
            layers.Dense(units=out_dim)
        ])

    def process_data_with_label(self, x, label):
        '''
            zero out xyz positions of sensors that are not activated
        '''
        # add dimension to allow broadcasting
        t = tf.expand_dims(x[:,4,:], axis=1)
        q = tf.expand_dims(x[:,5,:], axis=1)
        chamfer_x = tf.concat([x[:,:3,:], t, q], axis=1)
        label = tf.expand_dims(label, axis=1)
        label = tf.tile(label, [1, chamfer_x.shape[1], 1])
        # zero out sensors not activated (including the position features as well)
        x = chamfer_x * label
        return x

    def call(self, x):
        x = self.process_data_with_label(x, x[:,3,:])  # Output: [B, F-1, N]
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--reduce_lr_wait', type=int, default=20)
    parser.add_argument('--enc_dropout', type=float, default=0.2)
    parser.add_argument('--dec_dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--smaller_run', action="store_true")
    parser.add_argument('--dim_reduce_factor', type=float, default=1.5)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--save_ver', default=0)
    parser.add_argument('--mean_only', action="store_true")
    parser.add_argument('--save_best', action="store_true")
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--patience', type=int, default=15)


    args = parser.parse_args()
    # Example usage
    input_shape = (100, 4)
    model = PointClassifier(input_shape, 1, 10, args)

