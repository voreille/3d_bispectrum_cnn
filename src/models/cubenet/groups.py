from math import isclose

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class V_group(object):

    def __init__(self):
        self.group_dim = 4
        self.cayleytable = self.get_cayleytable()

    def get_cayleytable(self):
        print("...Computing Cayley table")
        cayley = np.asarray([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1],
                             [3, 2, 1, 0]])
        return cayley

    def get_Grotations(self, x):
        """Rotate the tensor x with all 4 Klein Vierergruppe rotations

        Args:
            x: [h,w,d,n_channels]
        Returns:
            list of 4 rotations of x [[h,w,d,n_channels],....]
        """
        xsh = x.get_shape().as_list()
        angles = [0., np.pi]
        rx = []
        for i in range(2):
            # 2x 180. rotations about the z axis
            perm = [1, 0, 2, 3]
            y = tf.transpose(x, perm=perm)
            y = tfa.image.rotate(y, angles[i])
            y = tf.transpose(y, perm=perm)

            # 2x 180. rotations about another axis
            for j in range(2):
                perm = [2, 1, 0, 3]
                z = tf.transpose(y, perm=perm)
                z = tfa.image.rotate(z, angles[j])
                z = tf.transpose(z, perm=perm)
                rx.append(z)
        return rx

    def G_permutation(self, W):
        """Permute the outputs of the group convolution"""
        Wsh = W.get_shape().as_list()
        cayley = self.cayleytable
        U = []
        for i in range(4):
            perm_mat = self.get_permutation_matrix(cayley, i)
            w = W[:, :, :, :, :, :, i]
            w = tf.transpose(w, [0, 1, 2, 3, 5, 4])
            w = tf.reshape(w, [-1, 4])
            w = w @ perm_mat
            w = tf.reshape(w, Wsh[:4] + [-1, 4])
            U.append(tf.transpose(w, [0, 1, 2, 3, 5, 4]))
        return U

    def get_permutation_matrix(self, perm, dim):
        ndim = perm.shape[0]
        mat = np.zeros((ndim, ndim))
        for j in range(ndim):
            mat[j, perm[j, dim]] = 1
        return mat


class T4_group(object):

    def __init__(self):
        self.group_dim = 12
        self.cayleytable = self.get_cayleytable()

    def rotate(self, x, axis, shift):
        angles = [0., np.pi / 2., np.pi, 3. * np.pi / 2.]
        perm = ([2, 1, 0, 3], [0, 2, 1, 3], [1, 0, 2, 3])
        x = tf.transpose(x, perm=perm[axis])
        x = tfa.image.rotate(x, angles[shift])
        return tf.transpose(x, perm=perm[axis])

    def r1(self, x):
        x = self.rotate(x, 0, -1)
        return self.rotate(x, 1, 1)

    def r2(self, x):
        x = self.rotate(x, 0, -1)
        return self.rotate(x, 1, -1)

    def r3(self, x):
        return self.rotate(x, 0, 2)

    def get_Grotations(self, x):
        """Rotate the tensor x with all 12 T4 rotations

        Args:
            x: [h,w,d,n_channels]
        Returns:
            list of 12 rotations of x [[h,w,d,n_channels],....]
        """
        Z = []
        for i in range(3):
            y = x
            for __ in range(i):
                y = self.r1(y)
            for j in range(3):
                z = y
                for __ in range(j):
                    z = self.r2(z)
                Z.append(z)
        for i in range(3):
            z = self.r3(x)
            for __ in range(i):
                z = self.r2(z)
            Z.append(z)
        return Z

    def G_permutation(self, W):
        """Permute the outputs of the group convolution"""
        Wsh = W.get_shape().as_list()
        cayley = self.cayleytable
        U = []
        for i in range(12):
            perm_mat = self.get_permutation_matrix(cayley, i)
            w = W[:, :, :, :, :, :, i]
            w = tf.transpose(w, [0, 1, 2, 3, 5, 4])
            w = tf.reshape(w, [-1, 12])
            w = w @ perm_mat
            w = tf.reshape(w, Wsh[:4] + [-1, 12])
            U.append(tf.transpose(w, [0, 1, 2, 3, 5, 4]))
        return U

    def get_permutation_matrix(self, perm, dim):
        ndim = perm.shape[0]
        mat = np.zeros((ndim, ndim))
        for j in range(ndim):
            mat[j, perm[j, dim]] = 1
        return mat

    def get_t4mat(self):
        Z = []
        for i in range(3):
            y = np.eye(3)
            for __ in range(i):
                y = y @ self.get_3Drotmat(1, 0, 0)
                y = y @ self.get_3Drotmat(0, -1, 0)
            for j in range(3):
                z = y
                for __ in range(j):
                    #z = r2(z)
                    z = z @ self.get_3Drotmat(1, 0, 0)
                    z = z @ self.get_3Drotmat(0, 1, 0)
                Z.append(z)
        for i in range(3):
            #z = r3(x)
            z = self.get_3Drotmat(2, 0, 0)
            for __ in range(i):
                #z = r2(z)
                z = z @ self.get_3Drotmat(1, 0, 0)
                z = z @ self.get_3Drotmat(0, 1, 0)
            Z.append(z)
        return Z

    def get_3Drotmat(self, x, y, z):
        c = [1., 0., -1., 0.]
        s = [0., 1., 0., -1]

        Rx = np.asarray([[c[x], -s[x], 0.], [s[x], c[x], 0.], [0., 0., 1.]])
        Ry = np.asarray([[c[y], 0., s[y]], [0., 1., 0.], [-s[y], 0., c[y]]])
        Rz = np.asarray([[1., 0., 0.], [0., c[z], -s[z]], [0., s[z], c[z]]])
        return Rz @ Ry @ Rx

    def get_cayleytable(self):
        Z = self.get_t4mat()
        cayley = []
        for y in Z:
            for z in Z:
                r = z @ y
                for i, el in enumerate(Z):
                    if np.sum(np.square(el - r)) < 1e-6:
                        cayley.append(i)
        cayley = np.stack(cayley)
        return np.reshape(cayley, [12, 12]).T


class S4_group(object):

    def __init__(self):
        self.group_dim = 24
        self.cayleytable = self.get_cayleytable()

    def get_Grotations(self, x):
        """Rotate the tensor x with all 24 S4 rotations

        Args:
            x: [h,w,d,n_channels]
        Returns:
            list of 24 rotations of x [[h,w,d,n_channels],....]
        """
        xsh = x.get_shape().as_list()
        angles = [0., np.pi / 2., np.pi, 3. * np.pi / 2.]
        rx = []
        for i in range(4):
            # Z4 rotations about the z axis
            perm = [1, 0, 2, 3]
            y = tf.transpose(x, perm=perm)
            y = tfa.image.rotate(y, angles[i])
            y = tf.transpose(y, perm=perm)
            # Rotations in the quotient space (sphere S^2)
            # i) Z4 rotations about y axis
            for j in range(4):
                perm = [2, 1, 0, 3]
                z = tf.transpose(y, perm=perm)
                z = tfa.image.rotate(z, angles[-j])
                z = tf.transpose(z, perm=perm)

                rx.append(z)
            # ii) 2 rotations to the poles about the x axis
            perm = [0, 2, 1, 3]
            z = tf.transpose(y, perm=perm)
            z = tfa.image.rotate(z, angles[3])
            z = tf.transpose(z, perm=perm)
            rx.append(z)

            z = tf.transpose(y, perm=perm)
            z = tfa.image.rotate(z, angles[1])
            z = tf.transpose(z, perm=perm)
            rx.append(z)

        return rx

    def G_permutation(self, W):
        """Permute the outputs of the group convolution"""
        Wsh = W.get_shape().as_list()
        cayley = self.cayleytable
        U = []
        for i in range(24):
            perm_mat = self.get_permutation_matrix(cayley, i)
            w = W[:, :, :, :, :, :, i]
            w = tf.transpose(w, [0, 1, 2, 3, 5, 4])
            w = tf.reshape(w, [-1, 24])
            w = w @ perm_mat
            w = tf.reshape(w, Wsh[:4] + [-1, 24])
            U.append(tf.transpose(w, [0, 1, 2, 3, 5, 4]))
        return U

    def get_permutation_matrix(self, perm, dim):
        ndim = perm.shape[0]
        mat = np.zeros((ndim, ndim))
        for j in range(ndim):
            mat[j, perm[j, dim]] = 1
        return mat

    def get_cayleytable(self):
        Z = self.get_s4mat()
        cayley = []
        for y in Z:
            for z in Z:
                r = z @ y
                for i, el in enumerate(Z):
                    if np.sum(np.square(el - r)) < 1e-6:
                        cayley.append(i)
        cayley = np.stack(cayley)
        return np.reshape(cayley, [24, 24])

    def get_s4mat(self):
        Z = []
        for i in range(4):
            # Z_4 rotation about Y
            # S^2 rotation
            for j in range(4):
                z = self.get_3Drotmat(i, j, 0)
                Z.append(z)
            # Residual pole rotations
            Z.append(self.get_3Drotmat(i, 0, 1))
            Z.append(self.get_3Drotmat(i, 0, 3))
        return Z

    def get_3Drotmat(self, x, y, z):
        c = [1., 0., -1., 0.]
        s = [0., 1., 0., -1]

        Rx = np.asarray([[c[x], -s[x], 0.], [s[x], c[x], 0.], [0., 0., 1.]])
        Ry = np.asarray([[c[y], 0., s[y]], [0., 1., 0.], [-s[y], 0., c[y]]])
        Rz = np.asarray([[1., 0., 0.], [0., c[z], -s[z]], [0., s[z], c[z]]])
        return Rz @ Ry @ Rx


def right_angle_rotations(x, angle):
    angle = angle % (2 * np.pi)
    if isclose(angle, 0.0):
        return x
    if isclose(angle, np.pi / 2.):
        return tf.reverse(tf.transpose(x, perm=[0, 2, 1, 3]), axis=[1])
    if isclose(angle, np.pi):
        return tf.reverse(x, axis=[1, 2])
    if isclose(angle, 3 * np.pi / 2.):
        return tf.reverse(tf.transpose(x, perm=[0, 2, 1, 3]), axis=[2])

    raise ValueError('Angle must be a multiple of pi/2')


class S4_group_faster(S4_group):

    def __init__(self):
        self.group_dim = 24
        self.cayleytable = self.get_cayleytable()
        self.permutation_matrices = self.get_permutation_matrices()

    def get_Grotations(self, x):
        """Rotate the tensor x with all 24 S4 rotations

        Args:
            x: [h,w,d,n_channels]
        Returns:
            list of 24 rotations of x [[h,w,d,n_channels],....]
        """
        xsh = x.get_shape().as_list()
        angles = [0., np.pi / 2., np.pi, 3. * np.pi / 2.]
        rx = []
        for i in range(4):
            # Z4 rotations about the z axis
            perm = [1, 0, 2, 3]
            y = tf.transpose(x, perm=perm)
            # y = tfa.image.rotate(y, angles[i])
            y = right_angle_rotations(y, angles[i])
            y = tf.transpose(y, perm=perm)
            # Rotations in the quotient space (sphere S^2)
            # i) Z4 rotations about y axis
            for j in range(4):
                perm = [2, 1, 0, 3]
                z = tf.transpose(y, perm=perm)
                z = right_angle_rotations(z, angles[-j])
                z = tf.transpose(z, perm=perm)

                rx.append(z)
            # ii) 2 rotations to the poles about the x axis
            perm = [0, 2, 1, 3]
            z = tf.transpose(y, perm=perm)
            z = right_angle_rotations(z, angles[3])
            z = tf.transpose(z, perm=perm)
            rx.append(z)

            z = tf.transpose(y, perm=perm)
            z = right_angle_rotations(z, angles[1])
            z = tf.transpose(z, perm=perm)
            rx.append(z)

        return rx

    def G_permutation(self, W):
        """Permute the outputs of the group convolution"""
        Wsh = W.get_shape().as_list()
        U = []
        for i in range(24):
            perm_mat = self.permutation_matrices[:, :, i]
            w = W[:, :, :, :, :, :, i]
            w = tf.transpose(w, [0, 1, 2, 3, 5, 4])
            w = tf.reshape(w, [-1, 24])
            w = w @ perm_mat
            w = tf.reshape(w, Wsh[:4] + [-1, 24])
            U.append(tf.transpose(w, [0, 1, 2, 3, 5, 4]))
        return U

    def get_permutation_matrix(self, perm, dim):
        ndim = perm.shape[0]
        mat = np.zeros((ndim, ndim))
        for j in range(ndim):
            mat[j, perm[j, dim]] = 1
        return mat

    def get_permutation_matrices(self):
        ndim = self.cayleytable.shape[0]
        mat = np.zeros((ndim, ndim, 24))
        for i in range(24):
            mat[:, :, i] = self.get_permutation_matrix(self.cayleytable, i)
        return mat
