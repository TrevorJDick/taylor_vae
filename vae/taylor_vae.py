# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 21:01:12 2022

@author: TD
"""
import itertools
import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers


def make_dirs_if_not_exist(directory=None, path=None):
    ### TODO check permissions are set correctly for OS
    if path is not None:
        directory = os.path.dirname(path)

    if directory is not None:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return


class VAE: 
    """
    Convolutional variational autoencoder
    
    Designed for 2d inputs coming from 3d arrays with shape
    (nsamples, dim1, dim2)
    """
    
    def __init__(self, input_shape, latent_dim=2, save_dir=None):
        # ex (npoints, ndims), or (xpixels, ypixels)
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        # init methods
        self.encoder = self.encoder_model()
        self.decoder = self.decoder_model()
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        # save directory for checkpoints and plots
        self.save_dir = save_dir
        
        if self.save_dir is not None:
            # create checkpoint and checkpoint manager
            self.checkpoint, self.ckpt_manager = self.checkpoint_manager()
    
    
    @staticmethod
    def swish(x):
        """
        Better activation than gelu
        
        Must be part of class for tensforflow decorator to work properly
        """
        return x * keras_backend.sigmoid(x)

    
    def encoder_model(self):
        npts, ndims = self.input_shape
        latent_dim = self.latent_dim
        
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(npts, ndims)))
        model.add(
            layers.Conv1D(
                filters=32,
                kernel_size=3,
                strides=2,
                # padding='same',
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(VAE.swish))
        model.add(
            layers.Conv1D(
                filters=64,
                kernel_size=3,
                strides=2,
                # padding='same',
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(VAE.swish))
        model.add(layers.Flatten())
        # needs to be double latent dim for encoder split
        model.add(layers.Dense(2 * latent_dim))
        model.summary()
        return model
    
    
    def decoder_model(self):
        ### TODO have decoder input shape different than encoder
        npts, ndims = self.input_shape
        latent_dim = self.latent_dim
        
        ### TODO ensure npts is divisible by: 2 ^ (# conv1d transpose layers)
        # b/c we are using 2 conv1d transpose
        proxy_npts = int(npts / 2 ** 2)
        proxy_ndims = 16
        
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(latent_dim,)))
        ### pay attention to this chunk
        # trying to end decoder with (None, npts, ndims) 
        # stride of 2 will double each conv transpose
        model.add(layers.Dense(proxy_npts * proxy_ndims))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(VAE.swish))
        model.add(layers.Reshape((proxy_npts, proxy_ndims)))
        model.add(
            layers.Conv1DTranspose(
                32, 
                kernel_size=3,
                strides=2,
                padding='same',
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(VAE.swish))
        model.add(
            layers.Conv1DTranspose(
                16, 
                kernel_size=3,
                strides=2,
                padding='same',
            )
        ) # output shape should yeild (None, npts, filters) 
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(VAE.swish))
        model.add(
            layers.Conv1DTranspose(
                ndims,
                kernel_size=3,
                strides=1,
                padding='same',
                # activation='tanh'
            )
        ) # output shape should yeild (None, npts, ndims)
        model.summary()
        return model

    
    def encode(self, x):
        # split into 2 smaller tensors along axis 1
        encoded_x = self.encoder(x)
        mean, logvar = tf.split(
            encoded_x,
            num_or_size_splits=2,
            axis=1
        )
        return mean, logvar
    
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    
    @staticmethod
    def reparameterize(mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    
    @staticmethod
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis
        )
    
    
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z) # reconstruction
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit,
            labels=x
        )
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
        logpz = VAE.log_normal_pdf(z, 0., 0.)
        logqz_x = VAE.log_normal_pdf(z, mean, logvar)
        loss = -tf.reduce_mean(logpx_z + logpz - logqz_x, axis=0, keepdims=True)
        return loss
    
    
    @tf.function
    def train_step(self, x_train):
        """
        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        
        NOTE: when this class inherits from tf.keras.Model class the 
        attribute self.trainable_variables is simply an extended python list
        of the models compiled who each have the attrbute
        model.trainable_variables.  For example the encoder model
        will have encoder.trainable_variables which is a list of trainable
        tf.Variables
        """
        trainable_variables = self.encoder.trainable_variables
        trainable_variables.extend(self.decoder.trainable_variables)
        
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x_train)
        
        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(
            zip(
                gradients,
                trainable_variables
            )
        )
        return
    
    
    def plot_and_save_decoded_latent_vectors(self, epoch,
                                             decoded_latent_vectors,
                                             x_test,
                                             epoch_loss=None, num_plots=None):
        if self.save_dir is not None:
            savefig_dir = (
                f'{self.save_dir}/'
                'plots/decoded_latent_vectors_test/images_by_epoch'
            )
            make_dirs_if_not_exist(directory=savefig_dir, path=None)
        
        epoch_loss_str = ''
        if epoch_loss is not None:
            epoch_loss_str = f', epoch loss: {epoch_loss}'
        
        if num_plots is None:
            num_plots = decoded_latent_vectors.shape[0]
        s = int(np.sqrt(num_plots))
        
        subplot_scale = 2
        width = 2 * s
        height = s
        plt.figure(figsize=(subplot_scale * width, subplot_scale * height))
        subplot_idx_list = list(
            itertools.chain(
                *[
                    2 * list(range(height * i, height * (i + 1)))
                    for i in range(height)
                ]
            )
        )
        for i, j in zip(range(num_plots * 2), subplot_idx_list):
            plt.subplot(height, width, i + 1)
            if (i + 1) % width in range(1, height + 1):
                f = decoded_latent_vectors[j, :, :]
            else:
                f = x_test[j, :, :]
            plt.plot(
                f[:, 1],
                f[:, 0] # y is in first col of matrix
            )
        plt.suptitle(
            f'LEFT {num_plots}: decoded latent vectors, '
            f'RIGHT {num_plots}: test sample\n'
            f'epoch: {epoch}{epoch_loss_str}',
            y=0.93
        )
        if self.save_dir is not None:
            plt.savefig(f'{savefig_dir}/image_at_epoch_{epoch}.png')
        plt.show()
        return

    
    def checkpoint_manager(self):
        checkpoint_dir = f'{self.save_dir}/training_checkpoints'
        make_dirs_if_not_exist(directory=checkpoint_dir, path=None)
        
        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            encoder=self.encoder,
            decoder=self.decoder
        )
        ckpt_manager = tf.train.CheckpointManager(
            checkpoint,
            checkpoint_dir,
            checkpoint_name='ckpt',
            max_to_keep=3
        )
        return checkpoint, ckpt_manager
    
    
    def save_checkpoint(self, epoch, epoch_save_freq=None):
        if epoch_save_freq is not None:
            epoch_save_freq = int(epoch_save_freq)
            if (epoch + 1) % epoch_save_freq == 0:
                    save_path = self.ckpt_manager.save()
                    print(
                        f'Saved checkpoint for step {epoch}: {save_path}'
                    )
        else:
            save_path = self.ckpt_manager.save()
            print(
                f'Saved checkpoint for step {epoch}: {save_path}'
            )
        return
    
    
    def restore_checkpoint(self):
        self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print(f'Restored from {self.ckpt_manager.latest_checkpoint}')
        else:
            print('Initializing from scratch.')
        return
    
    
    def generate_latent_vectors(self, x):
        latent_vectors = VAE.reparameterize(
            *self.encode(x)
        ).numpy()
        return latent_vectors
    
    
    def decode_latent_vectors(self, latent_vectors):
        x = self.decode(
            latent_vectors,
            apply_sigmoid=True
        ).numpy()
        return x
    
    
    @staticmethod
    def gen_lattice_from_latent_space(latent_vectors, num_lattice_ticks):
        latent_vectors_min = latent_vectors.min(axis=0)
        latent_vectors_max = latent_vectors.max(axis=0)
        
        lattice = np.meshgrid(
            np.linspace(
                latent_vectors_min[0],
                latent_vectors_max[0],
                num_lattice_ticks
            ),
            np.linspace(
                latent_vectors_min[1],
                latent_vectors_max[1],
                num_lattice_ticks
            )
        )
        # do flip because we need order to match subplot draw ordering
        lattice = np.vstack(
            (
                lattice[0].ravel(),
                np.flip(lattice[1], axis=0).ravel()
            )
        ).T
        return lattice
    
    
    def train(self, x_train, x_test=None, batch_size=32, epochs=10):
        if self.save_dir is not None:
            self.restore_checkpoint()
        
        batched_train = tf.data.Dataset.from_tensor_slices(
            x_train
        ).shuffle(
            x_train.shape[0]
        ).batch(
            batch_size
        )
        
        if x_test is not None:
            batched_test = tf.data.Dataset.from_tensor_slices(
                x_test
            ).shuffle(
                x_test.shape[0]
            ).batch(
                batch_size
            )
        
            # Pick a fixed subsample of one of the test batches for plotting
            # NOTA BENE: stick to perfect squares!
            num_test_plots = 16 
            if num_test_plots > batch_size:
                num_test_plots = batch_size
            
            ### TODO seed?
            for test_batch_single in batched_test.take(1):
                test_batch_plot = test_batch_single[0:num_test_plots]
        
        for epoch in range(epochs):
            start_time = time.time()
            for train_batch in batched_train:
                self.train_step(train_batch)
            end_time = time.time()
            
            if x_test is not None:
                loss = tf.keras.metrics.Mean()
                for test_batch in batched_test:
                    loss(self.compute_loss(test_batch))
                epoch_mean_loss = -loss.result()
                print(
                    'Mean loss over test batches: '
                    f'{epoch_mean_loss}, epoch: {epoch}'
                )
            
                # generate encoded then decoded latent vectors from test
                latent_vectors = self.generate_latent_vectors(test_batch_plot)
                decoded_latent_vectors = self.decode_latent_vectors(
                    latent_vectors
                )
                
                # plot decoded test vectors
                self.plot_and_save_decoded_latent_vectors(
                    epoch,
                    decoded_latent_vectors,
                    test_batch_plot,
                    epoch_loss=epoch_mean_loss
                )
            
            # Save the model every 5 epochs
            if self.save_dir is not None:
                self.save_checkpoint(epoch, epoch_save_freq=5)
        
            print(
                f'{end_time - start_time} -- time elapse for {epoch} epoch.'
            )
        
        # save again after last epoch
        if self.save_dir is not None:
            self.save_checkpoint(epoch)
        return



def map_region_to_unit_box(F, bounding_region):
    for i, (a, b) in enumerate(bounding_region):
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        # min and max are only the same when curve is constant
        ###TODO raises divide by 0 warning still
        F[:, :, i] = np.where(
            b == a,
            0,
            (F[:, :, i] - a) / (b - a)
        )
    return F


def make_poly_curves_from_coefficients(coeffs_matrix, num_points=None):  
    if num_points is None:
        num_points = 100
        """
        BIG NOTE:  the VAE NN architecture is dependant on the num points
        this is an issue for generating curve with more points.  interpolation
        is a major issue!
        """
        
    N, p = coeffs_matrix.shape # num samples, num coeffs
    m = 2 # dimension where function lives

    # x = np.linspace(0, 1, num_points)
    x = np.linspace(-1, 1, num_points)
    # design matrix for linear functions (polynomial basis here)
    A = np.vstack([x ** i for i in range(p)]).T
    F = np.einsum('ij,kj->ik', coeffs_matrix, A) # y values for N samples
    # ensure x,y matrix is represented for each of the N samples
    F = np.concatenate(
        [
            F.T.reshape(1, num_points, N),
            A[:, 1].reshape(1, num_points, 1).repeat(N, axis=2)
        ],
        axis=0
    ).T # shape (N, num_points, m)
    
    ### map to unit box, need x,y bounds for each sample!
    bounding_region = [
        (F[:, :, i].min(axis=1), F[:, :, i].max(axis=1))
        for i in range(m)
    ]
    F = map_region_to_unit_box(F, bounding_region)
    return bounding_region, F


def gen_random_coeffs_matrix(seed):
    p = 5 # taylor series highest degree polynomial minus 1
    N = 1500 # num train samples
    coeffs_matrix = np.random.default_rng(
        seed
    ).normal(loc=0.0, scale=1000, size=(N, p))
    zero_coeffs = np.random.default_rng(
        seed
    ).integers(0, 2, size=(N, p))
    coeffs_matrix = np.where(zero_coeffs == 0, 0.0, coeffs_matrix) # train
    del zero_coeffs
    return coeffs_matrix


def gen_legendre_coeffs(n):
    k_list = list(range(int(n / 2) + 1))
    pwrs = [n - 2 * k for k in k_list] # highest to lowest
    coeffs = [
        (
            (1 / (2 ** n)) 
            * ((-1) ** k) 
            * math.comb(n, k) 
            * math.comb((2 * (n - k)), n)
        )
        for k in k_list
    ] # order matches pwrs
    full_coeffs = np.zeros(n + 1)
    for i, j in zip(pwrs, coeffs):
        full_coeffs[i] = j
        del i, j
    return full_coeffs


def gen_legendre_coeffs_matrix(highest_degree=100):
    # need to pad each output of coeffs with highest degree in order to stack
    coeffs_by_deg = []
    for i in range(highest_degree + 1):
        c = np.pad(gen_legendre_coeffs(i), (0, highest_degree - i))
        coeffs_by_deg.append(c)
    coeffs_matrix = np.vstack(coeffs_by_deg)
    return coeffs_matrix


"""
seed = 37
# coeffs_matrix = gen_random_coeffs_matrix(seed)
# coeffs_matrix = np.diag(np.ones(100))
coeffs_matrix = gen_legendre_coeffs_matrix(highest_degree=100)


bounding_region, F = make_poly_curves_from_coefficients(
    coeffs_matrix, 
    num_points=1000
)


# plot the train polynomials
plt.figure(figsize=(20, 20))
for i, f in enumerate(F[: 100, :, :]):
    plt.subplot(10, 10, i + 1)
    plt.plot(
        f[:, 1],
        f[:, 0]
    )
    # plt.axis('off')
    del i, f
plt.show()


train_idx = np.random.default_rng(
    seed
).choice(
    F.shape[0],
    size=int(0.8 * F.shape[0]),
    replace=False
)
test_idx = list(set(range(F.shape[0])) - set(train_idx))

x_train = F[train_idx].astype('float32')
x_test = F[test_idx].astype('float32')



save_dir = (
    'C:/Users/the_s/Documents/python_projects/tensorflow_projects/taylor_vae'
)
input_shape = x_train[0, :, :].shape

vae_obj = VAE(input_shape, latent_dim=2, save_dir=None)

vae_obj.train(
      x_train,
      x_test=x_test,
      epochs=10
)


latent_vectors = vae_obj.generate_latent_vectors(x_test)
lattice = vae_obj.gen_lattice_from_latent_space(latent_vectors, 20)
decoded_latice = vae_obj.decode_latent_vectors(lattice)

plt.scatter(
    lattice[:, 0],
    lattice[:, 1],
    marker='X',
    s=1,
)
plt.show()

plt.figure(figsize=(20, 20))
for i in range(lattice.shape[0]):
    plt.subplot(20, 20, i+1)
    plt.plot(
        decoded_latice[i, :, 1],
        decoded_latice[i, :, 0]
    )
    # plt.axis('off')
del i
plt.show()
 

### show how latent vectors add
add_list = [0, 11]
add_test_vectors = x_test[add_list]
add_latent_vectors = latent_vectors[add_list]
add_latent_decoded_vectors = vae_obj.decode(
    add_latent_vectors, apply_sigmoid=True
).numpy()
result_add_latent_vectors = add_latent_vectors.sum(axis=0).reshape(1, -1)
result_add_decoded_vectors = vae_obj.decode(
    result_add_latent_vectors, apply_sigmoid=True
).numpy()
idx_closest_add_decoded_vectors = np.sqrt(
    np.sum((latent_vectors - result_add_latent_vectors) ** 2, axis=1)
).argmin()
closest_add_decoded_vectors = x_test[idx_closest_add_decoded_vectors]


plt.figure(figsize=(6, 4))
plt.subplot(2, 3, 1)
plt.plot(
    add_test_vectors[0, :, 1],
    add_test_vectors[0, :, 0]
)
plt.subplot(2, 3, 2)
plt.plot(
    add_test_vectors[1, :, 1],
    add_test_vectors[1, :, 0]
)

plt.subplot(2, 3, 3)
plt.plot(
    closest_add_decoded_vectors[:, 1],
    closest_add_decoded_vectors[:, 0]
)

plt.subplot(2, 3, 4)
plt.plot(
    add_latent_decoded_vectors[0, :, 1],
    add_latent_decoded_vectors[0, :, 0],
)

plt.subplot(2, 3, 5)
plt.plot(
    add_latent_decoded_vectors[1, :, 1],
    add_latent_decoded_vectors[1, :, 0],
)

plt.subplot(2, 3, 6)
plt.plot(
    result_add_decoded_vectors[0, :, 1],
    result_add_decoded_vectors[0, :, 0],
)
plt.show()

"""