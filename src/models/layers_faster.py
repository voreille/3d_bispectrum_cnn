import math
from typing import Callable
import logging

import tensorflow as tf
import numpy as np
from scipy import special as sp
from sympy.physics.quantum.cg import CG
from sympy import Ynm, Symbol, lambdify

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def get_lri_conv3d(*args, kind="bispectrum", **kwargs):
    if kind == "bispectrum":
        return BSHConv3D(*args, **kwargs)
    elif kind == "spectrum":
        return SSHConv3D(*args, **kwargs)
    else:
        raise ValueError(f"The kind {kind} is not supported")


class BSHConv3D(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 max_degree=3,
                 strides=1,
                 padding='SAME',
                 kernel_initializer="glorot_uniform",
                 use_bias=True,
                 bias_initializer="zeros",
                 radial_profile_type="radial",
                 activation="linear",
                 proj_activation="relu",
                 proj_initializer="glorot_uniform",
                 project=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.max_degree = max_degree
        self._indices = (
            (0, 0, 0),
            (0, 1, 1),
            (0, 2, 2),
            (1, 1, 2),
            (1, 2, 1),
            (1, 2, 3),
        )
        self.output_bispectrum_channels = self._output_bispectrum_channels()

        self._indices_inverse = None
        self.activation = tf.keras.activations.get(activation)
        self.clebschgordan_matrix = self._compute_clebschgordan_matrix()

        self.conv_sh = SHConv3D.get(name=radial_profile_type)(
            filters,
            kernel_size,
            max_degree=max_degree,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            **kwargs)

        if use_bias:
            self.bias = self.add_weight(
                shape=(self.output_bispectrum_channels * self.filters, ),
                initializer=bias_initializer,
                trainable=True,
                name="bias_bchconv3d",
            )
        else:
            self.bias = None

        if project:
            self.proj_conv = tf.keras.layers.Conv3D(
                filters,
                1,
                kernel_initializer=proj_initializer,
                activation=proj_activation,
                padding="SAME")
        else:
            self.proj_conv = None

    def _output_bispectrum_channels(self):
        return len(self._indices)
        # return self.max_degree + 1
        # n_outputs = 0
        # for n1 in range(0, math.floor(self.max_degree / 2) + 1):
        #     for n2 in range(n1, math.ceil(self.max_degree / 2) + 1):
        #         for i in range(np.abs(n1 - n2), n1 + n2 + 1):
        #             n_outputs += 1
        # return n_outputs

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

    def get_bisp_feature_maps_old(self, sh_feature_maps):
        _, depth, height, width, filters, n_harmonics = sh_feature_maps.get_shape(
        ).as_list()
        batch_size = tf.shape(sh_feature_maps)[0]
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
            filters * self.output_bispectrum_channels,
        ])

    # @tf.function
    def get_bisp_feature_maps(self, sh_feature_maps):
        _, depth, height, width, filters, n_harmonics = sh_feature_maps.get_shape(
        ).as_list()
        batch_size = tf.shape(sh_feature_maps)[0]
        sh_feature_maps = tf.reshape(sh_feature_maps, [-1, n_harmonics])

        bispectrum_coeffs = []
        for n1, n2, i in self.indices:
            kronecker_product = []
            f_n1 = self._get_fn(sh_feature_maps, n1)
            for m1 in range(2 * n1 + 1):
                kronecker_product.append(
                    tf.expand_dims(f_n1[..., m1], -1) *
                    self._get_fn(sh_feature_maps, n2))
            kronecker_product = tf.concat(kronecker_product, axis=-1)
            kronp_clebshgordan = tf.matmul(kronecker_product,
                                           self.clebschgordan_matrix[(n1, n2)])

            n_p = i**2 - (n1 - n2)**2
            Fi = tf.math.conj(self._get_fn(sh_feature_maps, i))

            if (n1 + n2 + i) % 2 == 0:
                bispectrum_coeffs.append(
                    tf.math.real(
                        tf.reduce_sum(
                            kronp_clebshgordan[:, n_p:n_p + 2 * i + 1] * Fi,
                            -1)))
            else:
                bispectrum_coeffs.append(
                    tf.math.imag(
                        tf.reduce_sum(
                            kronp_clebshgordan[:, n_p:n_p + 2 * i + 1] * Fi,
                            -1)))
        bispectrum_coeffs = tf.stack(bispectrum_coeffs, -1)
        return tf.reshape(bispectrum_coeffs, [
            batch_size,
            depth,
            height,
            width,
            filters * self.output_bispectrum_channels,
        ])

    def _get_spectrum_feature_maps(
        self,
        x,
    ):
        # Iterating over a symbolic `tf.Tensor` is not allowed: AutoGraph did convert this function
        batch_size = tf.shape(x)[0]
        depth = tf.shape(x)[1]
        height = tf.shape(x)[2]
        width = tf.shape(x)[3]
        filters = tf.shape(x)[4]

        spect_feature_maps = []
        for n in range(self.max_degree + 1):
            spect_feature_maps.append(1 / (2 * n + 1) * tf.reduce_sum(
                self._get_fn(tf.math.real(x), n)**2 +
                self._get_fn(tf.math.imag(x), n)**2, -1))
        spect_feature_maps = tf.stack(spect_feature_maps, -1)
        return tf.reshape(spect_feature_maps, [
            batch_size,
            depth,
            height,
            width,
            filters * (self.max_degree + 1),
        ])

    def _get_fn(self, x, n):
        return x[..., n * n:n * n + 2 * n + 1]

    def call(self, inputs, training=None):
        real_x, imag_x = self.conv_sh(inputs, training=training)
        # x = self.get_bisp_feature_maps(
        #     tf.complex(tf.cast(real_x, tf.float32),
        #                tf.cast(imag_x, tf.float32)))
        x = tf.complex(real_x, imag_x)
        x = self.get_bisp_feature_maps(x)
        x = tf.math.sign(x) * tf.math.log(1 + tf.math.abs(x))
        # x = tf.cast(x, inputs.dtype)
        if self.bias is not None:
            x = x + self.bias
        x = self.activation(x)
        if self.proj_conv is not None:
            x = self.proj_conv(x)
        return x


class SSHConv3D(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 max_degree=3,
                 strides=1,
                 padding='SAME',
                 kernel_initializer="glorot_uniform",
                 use_bias=True,
                 bias_initializer="zeros",
                 radial_profile_type="radial",
                 activation="linear",
                 proj_activation="relu",
                 proj_initializer="glorot_uniform",
                 project=True,
                 **kwargs):
        super().__init__(**kwargs)
        logger.info(f"Initializing SSHConv3D layer with filters: {filters}")
        self.filters = filters
        self.max_degree = max_degree
        self._indices = None
        self._indices_inverse = None
        self.activation = tf.keras.activations.get(activation)

        self.conv_sh = SHConv3D.get(name=radial_profile_type)(
            filters,
            kernel_size,
            max_degree=max_degree,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            **kwargs)

        self.n_radial_profiles = self.conv_sh.n_radial_profiles
        self.n_harmonics = self.conv_sh.n_harmonics

        if use_bias:
            self.bias = self.add_weight(
                shape=((self.max_degree + 1) * self.filters, ),
                initializer=bias_initializer,
                trainable=True,
                name="bias_schconv3d",
            )
        else:
            self.bias = None

        if project:
            self.proj_conv = tf.keras.layers.Conv3D(
                filters,
                1,
                kernel_initializer=proj_initializer,
                activation=proj_activation,
                padding="SAME")
        else:
            self.proj_conv = None
        logger.info(
            f"Initializing SSHConv3D layer with filters: {filters} - done")

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

    def _get_spectrum_feature_maps(
        self,
        real_sh_feature_maps,
        imag_sh_feature_maps,
    ):
        # Iterating over a symbolic `tf.Tensor` is not allowed: AutoGraph did convert this function
        batch_size = tf.shape(real_sh_feature_maps)[0]
        depth = tf.shape(real_sh_feature_maps)[1]
        height = tf.shape(real_sh_feature_maps)[2]
        width = tf.shape(real_sh_feature_maps)[3]
        filters = tf.shape(real_sh_feature_maps)[4]

        spect_feature_maps = []
        for n in range(self.max_degree + 1):
            spect_feature_maps.append(1 / (2 * n + 1) * tf.reduce_sum(
                self._get_fn(real_sh_feature_maps, n)**2 +
                self._get_fn(imag_sh_feature_maps, n)**2, -1))
        spect_feature_maps = tf.stack(spect_feature_maps, -1)
        return tf.reshape(spect_feature_maps, [
            batch_size,
            depth,
            height,
            width,
            filters * (self.max_degree + 1),
        ])

    def call(self, inputs):
        real_x, imag_x = self.conv_sh(inputs)

        x = self._get_spectrum_feature_maps(real_x, imag_x)

        if self.bias is not None:
            x += self.bias

        x = tf.math.sign(x) * tf.math.log(1 + tf.math.abs(x))
        if self.proj_conv is not None:
            x = self.proj_conv(x)
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
                 filters,
                 kernel_size,
                 max_degree=3,
                 strides=1,
                 padding='valid',
                 kernel_initializer="glorot_adapted",
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.max_degree = max_degree
        self.n_harmonics = (max_degree + 1)**2
        self.kernel_size = np.max(kernel_size)
        self.strides = strides if type(strides) is not int else 5 * (strides, )
        self.padding = padding.upper()
        self.sh_indices = list(self._sh_indices())
        self.atoms = self._atoms()
        self.n_radial_profiles = self.atoms.shape[-2]
        self.kernel_initializer = kernel_initializer
        if padding.lower() == "same":
            self.conv_central_pixel = tf.keras.layers.Conv3D(filters,
                                                             1,
                                                             strides=strides,
                                                             padding="SAME")
        else:
            crop = self.kernel_size // 2
            self.conv_central_pixel = tf.keras.Sequential([
                tf.keras.layers.Conv3D(filters,
                                       1,
                                       strides=strides,
                                       padding="SAME"),
                tf.keras.layers.Cropping3D(cropping=(
                    (crop, crop),
                    (crop, crop),
                    (crop, crop),
                ))
            ])

    def build(self, input_shape):
        if self.kernel_initializer == "glorot_adapted":
            limit = limit_glorot(input_shape[-1], self.filters)
            kernel_initializer = tf.keras.initializers.RandomUniform(
                minval=-limit, maxval=limit)
        else:
            kernel_initializer = self.kernel_initializer

        # limit = limit_glorot(input_shape[-1], self.filters)
        # limit = 0.005
        # kernel_initializer = tf.keras.initializers.RandomUniform(minval=-limit,
        #                                                          maxval=limit)

        self.w = self.add_weight(
            shape=(
                1,  # batch size
                1,  # depth
                1,  # height
                1,  # width
                input_shape[-1],  # input channels
                self.filters,  # output channels
                self.n_radial_profiles,
                self.max_degree + 1,
            ),
            initializer=kernel_initializer,
            trainable=True,
            name="w_profile",
        )

    def call(self, inputs, training=None):
        filters = self.atoms
        channels = tf.shape(inputs)[-1]
        filters = tf.reshape(filters, (
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            1,
            self.n_radial_profiles * self.n_harmonics,
        ))

        real_filters = tf.cast(tf.math.real(filters), inputs.dtype)
        imag_filters = tf.cast(tf.math.imag(filters), inputs.dtype)
        reals = list()
        imags = list()
        xs = tf.unstack(inputs, axis=-1)
        for x in xs:
            x = tf.expand_dims(x, -1)
            reals.append(
                tf.nn.conv3d(
                    x,
                    real_filters,
                    self.strides,
                    self.padding,
                    name="real_shconv",
                ))
            imags.append(
                tf.nn.conv3d(
                    x,
                    imag_filters,
                    self.strides,
                    self.padding,
                    name="imag_shconv",
                ))

        real_feature_maps = tf.stack(reals, axis=4)
        imag_feature_maps = tf.stack(imags, axis=4)

        # tf is too dumb for tf.shape(...)[:3]
        batch_size = tf.shape(real_feature_maps)[0]
        depth = tf.shape(real_feature_maps)[1]
        height = tf.shape(real_feature_maps)[2]
        width = tf.shape(real_feature_maps)[3]

        real_feature_maps = tf.reshape(real_feature_maps, (
            batch_size,
            depth,
            height,
            width,
            channels,
            1,
            self.n_radial_profiles,
            self.n_harmonics,
        ))
        imag_feature_maps = tf.reshape(imag_feature_maps, (
            batch_size,
            depth,
            height,
            width,
            channels,
            1,
            self.n_radial_profiles,
            self.n_harmonics,
        ))
        w = tf.repeat(
            self.w,
            [2 * k + 1 for k in range(self.max_degree + 1)],
            axis=-1,
        )
        # real_feature_maps = tf.reduce_sum(w * real_feature_maps, axis=(4, 6)),
        real_feature_maps = tf.unstack(
            tf.reduce_sum(w * real_feature_maps, axis=(4, 6)),
            axis=-1,
        )
        real_feature_maps[0] = real_feature_maps[0] + self.conv_central_pixel(
            inputs)
        real_feature_maps = tf.stack(real_feature_maps, axis=-1)

        imag_feature_maps = tf.reduce_sum(w * imag_feature_maps, axis=(4, 6))

        return real_feature_maps, imag_feature_maps

    def _atoms(self):
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
                 filters,
                 kernel_size,
                 max_degree=3,
                 strides=1,
                 padding='valid',
                 radial_function=None,
                 **kwargs):

        self.radial_function = SHConv3DRadial._get_radial_function(
            radial_function)
        # number of radial profiles used to build the filters, w/o the central one
        self.n_radial_profiles = np.max(kernel_size) // 2
        # number of atoms used to build the filters, w/o the central one
        self.n_atoms = (max_degree + 1)**2 * self.n_radial_profiles
        super().__init__(
            filters,
            kernel_size,
            max_degree=max_degree,
            strides=strides,
            padding=padding,
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

    def _atoms(self):
        r, theta, phi = self._get_spherical_coordinates()
        kernel_profiles = self._compute_kernel_profiles(r)
        spherical_harmonics = self._compute_spherical_harmonics(theta, phi)

        norm = np.sqrt(np.sum(kernel_profiles**2, axis=(0, 1, 2)))
        kernel_profiles = kernel_profiles / norm
        kernel_profiles = kernel_profiles[:, :, :, :, np.newaxis]

        spherical_harmonics = spherical_harmonics[:, :, :, np.newaxis, :]
        atoms = kernel_profiles * spherical_harmonics

        # norm = np.sqrt(np.sum(np.conj(atoms) * atoms, axis=(0, 1, 2)))
        # norm[norm == 0] = 1
        # atoms = atoms / norm

        return tf.constant(atoms, dtype=tf.complex64)

    def _compute_kernel_profiles(self, radius):
        n_profiles = self.kernel_size // 2
        kernel_profiles = np.zeros(
            (
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
                n_profiles,
            ),
            dtype=np.float32,
        )
        r0s = np.arange(1, n_profiles + 1)
        for i, r0 in enumerate(r0s):
            kernel_profiles[:, :, :, i] = self.radial_function(radius, r0)
        return kernel_profiles


# class SHConv3DCrossed(SHConv3D, name="crossed"):

#     def __init__(self,
#                  filters,
#                  kernel_size,
#                  max_degree=3,
#                  strides=1,
#                  padding='valid',
#                  radial_function=None,
#                  **kwargs):

#         # number of radial profiles used to build the filters, w/o the central one
#         self.n_radial_profiles = kernel_size // 2
#         # number of atoms used to build the filters, w/o the central one
#         self.n_atoms = (max_degree + 1)**2 * self.n_radial_profiles
#         super().__init__(
#             filters,
#             kernel_size,
#             max_degree=max_degree,
#             strides=strides,
#             padding=padding,
#             **kwargs,
#         )

#     @staticmethod
#     def _get_radial_function(input):
#         if input is None:
#             return lambda r, i: tri(r - i)
#         if input == "triangle":
#             return lambda r, i: tri(r - i)
#         if input == "gaussian":
#             return lambda r, i: np.exp(-0.5 * ((i - r) / 0.5)**2)
#         if isinstance(input, Callable):
#             return input

#         raise ValueError("Unknown radial function")

#     def _atoms(self):
#         r, theta, phi = self._get_spherical_coordinates()
#         kernel_profiles = self._compute_kernel_profiles(r)
#         spherical_harmonics = self._compute_spherical_harmonics(theta, phi)

#         kernel_profiles = kernel_profiles[:, :, :, :, np.newaxis]
#         spherical_harmonics = spherical_harmonics[:, :, :, np.newaxis, :]
#         atoms = kernel_profiles * spherical_harmonics

#         # norm = np.sqrt(np.sum(np.conj(atoms) * atoms, axis=(0, 1, 2)))
#         # norm[norm == 0] = 1
#         # atoms = atoms / norm

#         return tf.constant(atoms, dtype=tf.complex64)

#     def _compute_kernel_profiles(self, radius):
#         n_profiles = self.kernel_size // 2
#         kernel_profiles = np.zeros(
#             (
#                 self.kernel_size,
#                 self.kernel_size,
#                 self.kernel_size,
#                 n_profiles,
#             ),
#             dtype=np.float32,
#         )
#         r0s = np.arange(1, n_profiles + 1)
#         for i, r0 in enumerate(r0s):
#             kernel_profiles[:, :, :, i] = self.radial_function(radius, r0)
#         return kernel_profiles


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
        self.conv = get_lri_conv3d(*args,
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


class LinearUpsampling3D(tf.keras.layers.Layer):

    def __init__(self, size=(2, 2, 2), **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.kernel = self._get_kernel(size)

    @staticmethod
    def _get_kernel(size):
        k1 = tri(np.linspace(-1, 1, 2 * size[0] + 1))
        k1 = k1[1:-1]
        k2 = tri(np.linspace(-1, 1, 2 * size[1] + 1))
        k2 = k2[1:-1]
        k3 = tri(np.linspace(-1, 1, 2 * size[2] + 1))
        k3 = k3[1:-1]
        k = np.tensordot(k1, k2, axes=0)
        k = np.tensordot(k, k3, axes=0)
        k = np.reshape(k, k.shape + (
            1,
            1,
        ))
        return tf.constant(k, dtype=tf.float32)

    def call(self, inputs, training=None):
        xs = tf.unstack(inputs, axis=-1)
        out = []
        kernel = tf.cast(self.kernel, inputs.dtype)
        for x in xs:
            x = tf.expand_dims(x, axis=-1)
            x = conv3d_transpose(x, kernel, self.size, padding="SAME")
            out.append(x)
        return tf.concat(out, axis=-1)


# @tf.function
def conv3d_complex(input, filters, strides, **kwargs):
    filters_expanded = tf.concat(
        [
            tf.math.real(filters),
            tf.math.imag(filters),
        ],
        axis=-1,
    )

    if type(strides) is int:
        strides = 5 * (strides, )

    return tf.nn.conv3d(input, filters_expanded, strides, **kwargs)


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