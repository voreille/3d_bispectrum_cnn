import math
from itertools import product
from typing import Callable
import warnings

import tensorflow as tf
import numpy as np
from scipy import special as sp
from sympy.physics.quantum.cg import CG
from sympy import Ynm, Symbol, lambdify


def get_lri_conv2d(*args, kind="bispectrum", **kwargs):
    if kind == "bispectrum":
        return BSHConv3D(*args, **kwargs)
    else:
        raise ValueError(f"The kind {kind} is not supported")


class SEBlock(tf.keras.layers.Layer):

    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

    def build(self, input_shape):
        self.fc_1 = tf.keras.layers.Dense(input_shape[-1] // self.ratio,
                                          activation="relu")
        self.fc_2 = tf.keras.layers.Dense(input_shape[-1],
                                          activation="sigmoid")

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return tf.multiply(inputs, x)


class BSHConv3D(tf.keras.layers.Layer):

    def __init__(self,
                 streams,
                 kernel_size,
                 max_degree=3,
                 strides=1,
                 padding='SAME',
                 initializer="random_normal",
                 use_bias=True,
                 bias_initializer="zeros",
                 radial_profile_type="radial",
                 activation="linear",
                 proj_activation="relu",
                 proj_initializer="glorot_uniform",
                 is_transpose=False,
                 project=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.streams = streams
        self.max_degree = max_degree
        self.output_bispectrum_channels = self._output_bispectrum_channels()
        self._indices = None
        self._indices_inverse = None
        self.activation = tf.keras.activations.get(activation)
        self.clebschgordan_matrix = self._compute_clebschgordan_matrix()

        self.conv_sh = SHConv3D.get(name=radial_profile_type)(
            streams,
            kernel_size,
            max_degree=max_degree,
            strides=strides,
            padding=padding,
            initializer=initializer,
            is_transpose=is_transpose,
            **kwargs)

        if use_bias:
            self.bias = self.add_weight(
                shape=(self.output_bispectrum_channels * self.streams, ),
                initializer=bias_initializer,
                trainable=True,
                name="bias_bchconv3d",
            )
        else:
            self.bias = None

        if project:
            self.proj_conv = tf.keras.layers.Conv2D(
                streams,
                1,
                kernel_initializer=proj_initializer,
                activation=proj_activation,
                padding="SAME")
        else:
            self.proj_conv = None

    def _output_bispectrum_channels(self):
        n_outputs = 0
        for n1 in range(0, math.floor(self.max_degree / 2) + 1):
            for n2 in range(n1, math.ceil(self.max_degree / 2) + 1):
                for i in range(np.abs(n1 - n2), n1 + n2 + 1):
                    n_outputs += 1
        return n_outputs

    def _compute_clebschgordan_matrix(self):
        cg_mat = {}  # the set of cg matrices
        for n1 in range(self.max_degree + 1):
            for n2 in range(self.max_degree + 1):
                cg_mat[(n1, n2)] = tf.constant(
                    compute_clebschgordan_matrix(n1, n2),
                    dtype=tf.complex64,
                )
        return cg_mat

    @property
    def indices(self):
        if self._indices is None:
            self._indices = []
            for n1 in range(0, math.floor(self.max_degree / 2) + 1):
                for n2 in range(n1, math.ceil(self.max_degree / 2) + 1):
                    for i in range(np.abs(n1 - n2), n1 + n2 + 1):
                        self._indices.append((n1, n2, i))
        return self._indices

    @property
    def indices_inverse(self):
        if self._indices_inverse is None:
            self._indices_inverse = {}
            for k, (n1, n2, i) in enumerate(self.indices):
                self._indices_inverse[(n1, n2, i)] = k
        return self._indices_inverse

    # @tf.function
    def get_bisp_feature_maps(self, sh_feature_maps):
        batch_size, depth, height, width, n_streams, n_harmonics = sh_feature_maps.get_shape(
        ).as_list()
        sh_feature_maps = tf.reshape(sh_feature_maps, [-1, n_harmonics])

        bispectrum_coeffs = []
        for n1 in range(0, math.floor(self.max_degree / 2) + 1):
            for n2 in range(n1, math.ceil(self.max_degree / 2) + 1):
                kronecker_product = []
                for m1 in degree_to_indices_range(n1):
                    kronecker_product.append(
                        tf.expand_dims(sh_feature_maps[..., m1], -1) *
                        sh_feature_maps[..., degree_to_indices_slice(n2)])
                kronecker_product = tf.concat(kronecker_product, axis=-1)
                kronp_clebshgordan = tf.matmul(
                    kronecker_product, self.clebschgordan_matrix[(n1, n2)])

                for i in range(np.abs(n1 - n2), n1 + n2 + 1):
                    n_p = i**2 - (n1 - n2)**2
                    Fi = tf.math.conj(
                        sh_feature_maps[..., degree_to_indices_slice(i)])

                    if (n1 + n2 + i) % 2 == 0:
                        bispectrum_coeffs.append(
                            tf.math.real(
                                tf.reduce_sum(
                                    kronp_clebshgordan[:, n_p:n_p + 2 * i + 1]
                                    * Fi, -1)))
                    else:
                        bispectrum_coeffs.append(
                            tf.math.imag(
                                tf.reduce_sum(
                                    kronp_clebshgordan[:, n_p:n_p + 2 * i + 1]
                                    * Fi, -1)))
        bispectrum_coeffs = tf.stack(bispectrum_coeffs, -1)
        return tf.reshape(bispectrum_coeffs, [
            batch_size,
            depth,
            height,
            width,
            n_streams * self.output_bispectrum_channels,
        ])

    def call(self, inputs):
        x = self.conv_sh(inputs)
        x = self.get_bisp_feature_maps(x)
        x = tf.math.sign(x) * tf.math.log(1 + tf.math.abs(x))
        if self.bias is not None:
            x = x + self.bias
        x = self.activation(x)
        if self.proj_conv is not None:
            x = self.proj_conv(x)
        return x


class BSHConv3DComplex(BSHConv3D):

    def _get_fn(self, x, n):
        return x[..., n * n:n * n + 2 * n + 1]

    def _get_kron_product(self, f1, f2):
        kronecker_product = []
        for m1 in range(f1.shape[-1]):
            kronecker_product.append(tf.expand_dims(f1[..., m1], -1) * f2)
        return tf.concat(kronecker_product, axis=-1)

    def get_bisp_feature_maps(self, sh_feature_maps):
        batch_size, depth, height, width, n_streams, n_harmonics = tf.shape(
            sh_feature_maps)
        sh_feature_maps = tf.reshape(sh_feature_maps, [-1, n_harmonics])
        bisp_feature_maps = []
        for n1 in range(0, math.floor(self.max_degree / 2) + 1):
            for n2 in range(n1, math.ceil(self.max_degree / 2) + 1):
                x = self._get_kron_product(self._get_fn(sh_feature_maps, n1),
                                           self._get_fn(sh_feature_maps, n2))
                x = tf.matmul(x, self.clebschgordan_matrix[(n1, n2)])
                for i in range(np.abs(n1 - n2), n1 + n2 + 1):
                    n_p = i**2 - (n1 - n2)**2
                    Fi = tf.math.conj(self._get_fn(sh_feature_maps, i))
                    bisp_feature_maps.append(
                        tf.reduce_sum(x[:, n_p:n_p + 2 * i + 1] * Fi, -1))
        bisp_feature_maps = tf.stack(bisp_feature_maps, -1)
        return tf.reshape(bisp_feature_maps, [
            batch_size,
            depth,
            height,
            width,
            n_streams * self.output_bispectrum_channels,
        ])

    def call(self, inputs):
        x = self.conv_sh(inputs)
        x = self.get_bisp_feature_maps(x)
        return x


class SSHConv3D(BSHConv3D):

    @property
    def indices(self):
        if self._indices is None:
            self._indices = list(range(self.max_degree + 1))
        return self._indices

    @property
    def indices_inverse(self):
        if self._indices_inverse is None:
            self._indices_inverse = list(range(self.max_degree + 1))
        return self._indices_inverse

    def _get_fn(self, x, n):
        return x[..., n * n:n * n + 2 * n + 1]

    def _get_spectrum_feature_maps(self, sh_feature_maps):
        batch_size, depth, height, width, n_streams, n_harmonics = tf.shape(
            sh_feature_maps)
        spect_feature_maps = []
        for n in range(self.max_degree + 1):
            spect_feature_maps.append(
                1 / (2 * n + 1) *
                tf.reduce_sum(np.abs(self._get_fn(sh_feature_maps, n))**2, -1))
        spect_feature_maps = tf.stack(spect_feature_maps, -1)
        return tf.reshape(spect_feature_maps, [
            batch_size,
            depth,
            height,
            width,
            n_streams * (self.max_degree + 1),
        ])

    def call(self, inputs):
        x = self.conv_sh(inputs)
        x = self._get_spectrum_feature_maps(x)
        x = tf.math.sign(x) * tf.math.log(1 + tf.math.abs(x))
        return x


class SHConv3D(tf.keras.layers.Layer):
    _registry = {}  # class var that store the different daughter

    def __init_subclass__(cls, name, **kwargs):
        cls.name = name
        SHConv3D._registry[name] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def get(cls, name: str):
        return SHConv3D._registry[name]

    def __init__(self,
                 streams,
                 kernel_size,
                 max_degree=3,
                 strides=1,
                 padding='valid',
                 initializer="random_normal",
                 is_transpose=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.streams = streams
        self.max_degree = max_degree
        self.n_harmonics = (max_degree + 1)**2
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.initializer = initializer
        self.sh_indices = list(self._sh_indices())
        if is_transpose:
            self.convolution = conv3d_transpose_complex
        else:
            self.convolution = conv3d_complex
        self.atoms = self._atoms()

    def call(self, inputs, training=None):
        filters = self.filters
        channels = tf.shape(filters)[3]
        filters = tf.reshape(filters, (
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            channels,
            self.streams * self.n_harmonics,
        ))
        feature_maps = self.convolution(inputs,
                                        filters,
                                        self.strides,
                                        padding=self.padding)

        # tf is too dumb for tf.shape(...)[:3]
        batch_size = tf.shape(feature_maps)[0]
        depth = tf.shape(feature_maps)[1]
        height = tf.shape(feature_maps)[2]
        width = tf.shape(feature_maps)[3]

        feature_maps = tf.reshape(feature_maps, (
            batch_size,
            depth,
            height,
            width,
            self.streams,
            self.n_harmonics,
        ))
        return feature_maps

    def _atoms(self):
        raise NotImplementedError("It is an abstrac class")

    @property
    def filters(self):
        raise NotImplementedError("It is an abstrac class")

    def _get_spherical_coordinates(self):
        x_grid = np.linspace(-(self.kernel_size // 2), self.kernel_size // 2,
                             self.kernel_size)
        x, y, z = np.meshgrid(x_grid, x_grid, x_grid, indexing='xy')
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        theta = np.arccos(np.divide(z, r, out=np.zeros_like(r), where=r != 0))
        return r, theta, phi

    def _compute_spherical_harmonics(self, theta, phi):
        sh = np.zeros((self.kernel_size, self.kernel_size, self.kernel_size,
                       self.n_harmonics),
                      dtype=np.complex64)
        for n in range(self.max_degree + 1):
            for m in range(-n, n + 1):
                sh[..., self.ravel_sh_index(n, m)] = spherical_harmonics(
                    m, n, theta, phi)
        return sh

    def ravel_sh_index(self, n, m):
        if np.abs(m) > n:
            raise ValueError("m must be in [-n, n]")
        return n**2 + m + n

    def _sh_indices(self):
        for n in range(self.max_degree + 1):
            for m in range(-n, n + 1):
                yield (n, m)

    def unravel_sh_index(self, index):
        return self.sh_indices[index]


class SHConv3DRadial(SHConv3D, name="radial"):

    def __init__(self,
                 streams,
                 kernel_size,
                 max_degree=3,
                 strides=1,
                 padding='valid',
                 initializer="glorot_adapted",
                 is_transpose=False,
                 radial_function=None,
                 **kwargs):

        self.radial_function = SHConv3DRadial._get_radial_function(
            radial_function)
        # number of radial profiles used to build the filters, w/o the central one
        self.n_radial_profiles = kernel_size // 2
        # number of atoms used to build the filters, w/o the central one
        self.n_atoms = (max_degree + 1)**2 * self.n_radial_profiles
        super().__init__(
            streams,
            kernel_size,
            max_degree=max_degree,
            strides=strides,
            padding=padding,
            initializer=initializer,
            is_transpose=is_transpose,
            **kwargs,
        )

    @staticmethod
    def _get_radial_function(input):
        if input is None:
            return lambda r, i: tri(r - i)
        if input == "triangle":
            return lambda r, i: tri(r - i)
        if input == "gaussian":
            return lambda r, i: np.exp(-0.5 * ((i - r) / 0.5)**2)
        if isinstance(input, Callable):
            return input

        raise ValueError("Unknown radial function")

    # def ravel_index(self, i, n, m):
    #     return i * self.n_harmonics + self.ravel_sh_index(n, m)

    # def unravel_index(self, index):
    #     i = index // self.n_harmonics
    #     index -= i * self.n_harmonics
    #     n, m = self.unravel_sh_index(index)
    #     return i, n, m

    def _atoms(self):
        r, theta, phi = self._get_spherical_coordinates()
        kernel_profiles = self._compute_kernel_profiles(r)
        spherical_harmonics = self._compute_spherical_harmonics(theta, phi)

        atoms0 = np.zeros(
            (self.kernel_size, self.kernel_size, self.kernel_size, 1, 1, 1),
            dtype=np.csingle)
        atoms0[:, :, :, 0, 0, 0] = kernel_profiles[..., 0]
        kernel_profiles = kernel_profiles[:, :, :, np.newaxis, np.newaxis,
                                          np.newaxis, 1:]
        spherical_harmonics = spherical_harmonics[:, :, :, np.newaxis,
                                                  np.newaxis, :, np.newaxis]
        atoms = kernel_profiles * spherical_harmonics

        # norm = np.sqrt(np.sum(np.conj(atoms) * atoms, axis=(0, 1, 2)))
        # norm[norm == 0] = 1
        # atoms = atoms / norm

        # norm = np.sqrt(np.sum(np.conj(atoms0) * atoms0, axis=(0, 1, 2)))
        # norm[norm == 0] = 1
        # atoms0 = atoms0 / norm

        return (tf.constant(atoms0, dtype=tf.complex64),
                tf.constant(atoms, dtype=tf.complex64))

    def _compute_kernel_profiles(self, radius):
        n_profiles = self.kernel_size // 2 + 1
        kernel_profiles = np.zeros(
            (self.kernel_size, self.kernel_size, self.kernel_size, n_profiles),
            dtype=np.float32)
        for i in range(n_profiles):
            kernel_profiles[:, :, :, i] = self.radial_function(radius, i)
        return kernel_profiles

    def build(self, input_shape):
        if self.initializer == "glorot_adapted":
            limit = limit_glorot(input_shape[-1], self.streams)
            initializer = tf.keras.initializers.RandomUniform(minval=-limit,
                                                              maxval=limit),
        else:
            initializer = self.initializer

        self.w = self.add_weight(
            shape=(
                1,
                1,
                1,
                input_shape[-1],
                self.streams,
                self.max_degree + 1,
                self.n_radial_profiles,
            ),
            initializer=initializer,
            trainable=True,
            name="w_profile",
        )
        self.w0 = self.add_weight(
            shape=(
                1,
                1,
                1,
                input_shape[-1],
                self.streams,
                1,
            ),
            initializer=initializer,
            trainable=True,
            name="w0_profile",
        )

    @property
    def filters(self):
        atoms0, atoms = self.atoms
        w0 = tf.complex(self.w0, tf.zeros_like(self.w0))
        w = tf.complex(self.w, tf.zeros_like(self.w))
        factor = tf.concat(
            [
                tf.ones((1, ), dtype=tf.complex64),
                tf.zeros((self.n_harmonics - 1, ), dtype=tf.complex64)
            ],
            axis=0,
        )
        factor = tf.reshape(factor, (1, 1, 1, 1, self.n_harmonics))
        return w0 * atoms0 * factor + tf.reduce_sum(
            tf.repeat(
                w,
                [2 * k + 1 for k in range(self.max_degree + 1)],
                axis=5,
            ) * atoms,
            axis=-1,
        )


class ResidualLayer3D(tf.keras.layers.Layer):

    def __init__(self, *args, activation='relu', **kwargs):
        super().__init__()
        self.filters = args[0]
        self.conv = tf.keras.layers.Conv3D(*args,
                                           **kwargs,
                                           activation=activation)
        self.activation = activation
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = None
        self.proj = None
        self.strides = kwargs.get("strides", 1)

    def build(self, input_shape):
        self.c_in = input_shape[1]
        if input_shape[-1] != self.filters:
            self.proj = tf.keras.layers.Conv3D(self.filters,
                                               1,
                                               activation=self.activation,
                                               strides=self.strides)
            self.bn_2 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        if self.proj:
            return self.bn_1(self.conv(x)) + self.bn_2(self.proj(x))
        else:
            return self.bn_1(self.conv(x)) + x


class ResidualLRILayer3D(tf.keras.layers.Layer):

    def __init__(self, *args, kind="bispectrum", activation="relu", **kwargs):
        super().__init__()
        self.filters = args[0]
        self.conv = get_lri_conv2d(*args,
                                   **kwargs,
                                   activation=activation,
                                   kind=kind)
        self.strides = kwargs.get("strides", 1)
        self.activation = activation
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = None
        self.proj = None

    def build(self, input_shape):
        self.c_in = input_shape[1]
        if input_shape[-1] != self.filters:  # or self.strides == 2 ??
            self.proj = tf.keras.layers.Conv3D(
                self.filters,
                1,
                activation=self.activation,
                strides=self.strides,
                padding="SAME",
            )
            self.bn_2 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        if self.proj:
            return self.bn_1(self.conv(x)) + self.bn_2(self.proj(x))
        else:
            return self.bn_1(self.conv(x)) + x


class MaskedConv3D(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 trainable=True,
                 activation=None,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 mask=None,
                 **kwargs):
        super().__init__(trainable=trainable,
                         name=name,
                         dtype=dtype,
                         dynamic=dynamic,
                         **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.bias = self.add_weight(
            shape=(self.filters, ),
            initializer="zeros",
            trainable=True,
            name="bias_masked_conv2d",
        )
        self.activation = tf.keras.activations.get(activation)
        self.mask = tf.reshape(tf.Variable(mask, dtype=tf.float32),
                               mask.shape + (1, 1))
        if mask is None:
            raise ValueError("HEY!!, provide a mask")

    def build(self, input_shape):
        limit = limit_glorot(input_shape[-1], self.filters)
        self.w = self.add_weight(
            shape=(
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
                input_shape[-1],
                self.filters,
            ),
            initializer=tf.keras.initializers.RandomUniform(minval=-limit,
                                                            maxval=limit),
            trainable=True,
            name="kernel_masked_conv2d",
        )

    @property
    def kernel(self):
        return self.mask * self.w

    def call(self, inputs, training=None):
        return self.activation(
            tf.nn.conv3d(
                inputs,
                self.kernel,
                self.strides,
                self.padding,
            ) + self.bias)


# @tf.function
def conv3d_complex_alternative(input, filters, strides, **kwargs):
    real_filter = tf.math.real(filters)
    imag_filter = tf.math.imag(filters)

    if type(strides) is int:
        strides = 5 * (strides, )

    real_output = tf.nn.conv3d(input, real_filter, strides, **kwargs)
    imag_output = tf.nn.conv3d(input, imag_filter, strides, **kwargs)
    return tf.complex(real_output, imag_output)


def conv3d_complex(input, filters, strides, **kwargs):
    out_channels = tf.shape(filters)[-1]
    filters_expanded = tf.concat(
        [
            tf.math.real(filters),
            tf.math.imag(filters),
        ],
        axis=-1,
    )

    if type(strides) is int:
        strides = 5 * (strides, )

    output = tf.nn.conv3d(input, filters_expanded, strides, **kwargs)
    return tf.complex(output[..., :out_channels], output[..., out_channels:])


# @tf.function
def conv3d_transpose_complex(input, filters, strides, **kwargs):
    out_channels = tf.shape(filters)[-1]
    filters_expanded = tf.concat(
        [
            tf.math.real(filters),
            tf.math.imag(filters),
        ],
        axis=3,
    )

    output = conv3d_transpose(input, filters_expanded, strides, **kwargs)
    return tf.complex(output[..., :out_channels], output[..., out_channels:])


# @tf.function
def conv3d_transpose(input, filters, strides, **kwargs):
    filter_depth, filter_height, filter_width, _, out_channels = filters.get_shape(
    ).as_list()
    batch_size = tf.shape(input)[0]
    in_depth = tf.shape(input)[1]
    in_height = tf.shape(input)[2]
    in_width = tf.shape(input)[3]
    if type(strides) is int:
        stride_d = strides
        stride_h = strides
        stride_w = strides
    elif len(strides) == 3:
        stride_d, stride_h, stride_w = strides

    padding = kwargs.get("padding", "SAME")
    if padding == 'VALID':
        output_size_d = (in_depth - 1) * stride_d + filter_depth
        output_size_h = (in_height - 1) * stride_h + filter_height
        output_size_w = (in_width - 1) * stride_w + filter_width
    elif padding == 'SAME':
        output_size_d = in_depth * stride_d
        output_size_h = in_height * stride_h
        output_size_w = in_width * stride_w
    else:
        raise ValueError("unknown padding")
    output_shape = (batch_size, output_size_d, output_size_h, output_size_w,
                    out_channels)

    return tf.nn.conv3d_transpose(input, tf.transpose(filters,
                                                      (0, 1, 2, 4, 3)),
                                  output_shape, strides, **kwargs)


def is_approx_equal(x, y, epsilon=1e-3):
    return np.abs(x - y) / (np.sqrt(np.abs(x) * np.abs(y)) + epsilon) < epsilon


def tri(x):
    return np.where(np.abs(x) <= 1, np.where(x < 0, x + 1, 1 - x), 0)


def limit_glorot(c_in, c_out):
    return np.sqrt(6 / (c_in + c_out))


def legendre(n, X):
    '''
    Legendre polynomial used to define the SHs for degree n
    '''
    res = np.zeros(((n + 1, ) + (X.shape)))
    for m in range(n + 1):
        res[m] = sp.lpmv(m, n, X)
    return res


def spherical_harmonics_old(m, n, p_legendre, phi):
    '''
    Returns the SH of degree n, order m
    '''
    P_n_m = np.squeeze(p_legendre[np.abs(m)])
    sign = (-1)**((m + np.abs(m)) / 2)
    # Normalization constant
    A = sign * np.sqrt(
        (2 * n + 1) / (4 * np.pi) * np.math.factorial(n - np.abs(m)) /
        np.math.factorial(n + np.abs(m)))
    # Spherical harmonics
    sh = A * np.exp(1j * m * phi) * P_n_m
    # Normalize the SH to unit norm
    sh /= np.sqrt(np.sum(sh * np.conj(sh)))
    return sh.astype(np.complex64)


def spherical_harmonics(m, n, theta, phi):
    '''
    Returns the SH of degree n, order m
    '''
    theta_s = Symbol('theta')
    phi_s = Symbol('phi')
    ynm = Ynm(n, m, theta_s, phi_s).expand(func=True)
    f = lambdify([theta_s, phi_s], ynm, 'numpy')
    return f(theta, phi).astype(np.complex64)


def compute_clebschgordan_matrix(k, l):
    '''
    Computes the matrix that block-diagonilizes the Kronecker product of
    Wigned D matrices of degree k and l respectively
    Output size (2k+1)(2l+1)x(2k+1)(2l+1)
    '''
    c_kl = np.zeros([(2 * k + 1) * (2 * l + 1), (2 * k + 1) * (2 * l + 1)])

    n_off = 0
    for J in range(abs(k - l), k + l + 1):
        m_off = 0
        for m1_i in range(2 * k + 1):
            m1 = m1_i - k
            for m2_i in range(2 * l + 1):
                m2 = m2_i - l
                for n_i in range(2 * J + 1):
                    n = n_i - J
                    if m1 + m2 == n:
                        c_kl[m_off + m2_i,
                             n_off + n_i] = CG(k, m1, l, m2, J,
                                               m1 + m2).doit()
            m_off = m_off + 2 * l + 1
        n_off = n_off + 2 * J + 1

    return c_kl


def degree_to_indices_range(n):
    return range(n * n, n * n + 2 * n + 1)


def degree_to_indices_slice(n):
    return slice(n * n, n * n + 2 * n + 1)