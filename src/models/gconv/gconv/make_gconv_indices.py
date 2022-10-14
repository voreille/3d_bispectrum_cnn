# Code for generating indices used in G-convolutions for various groups G.
# The indices created by these functions are used to rotate and flip filters on the plane or on a group.
# These indices depend only on the filter size, so they are created only once at the beginning of training.

import numpy as np
from src.models.gconv.garray.C4_array import C4
from src.models.gconv.garray.D4_array import D4
from src.models.gconv.garray.O_array import O
from src.models.gconv.garray.Oh_array import Oh
from src.models.gconv.garray.C4h_array import C4h
from src.models.gconv.garray.D4h_array import D4h
from src.models.gconv.garray.p4_array import C4_halfshift
from src.models.gconv.gfunc.otfunc_array import OtFuncArray
from src.models.gconv.gfunc.ohtfunc_array import OhtFuncArray
from src.models.gconv.gfunc.c4htfunc_array import C4htFuncArray
from src.models.gconv.gfunc.d4htfunc_array import D4htFuncArray
from src.models.gconv.gfunc.p4func_array import P4FuncArray
from src.models.gconv.gfunc.p4mfunc_array import P4MFuncArray
from src.models.gconv.gfunc.z2func_array import Z2FuncArray
from src.models.gconv.gfunc.z3func_array import Z3FuncArray
def make_c4_z2_indices(ksize):
    x = np.random.randn(1, ksize, ksize)
    f = Z2FuncArray(v=x)

    if ksize % 2 == 0:
        uv = f.left_translation_indices(C4_halfshift[:, None, None, None])
    else:
        a = C4[:, None, None, None]
        uv = f.left_translation_indices(a)
    r = np.zeros(uv.shape[:-1] + (1,))
    ruv = np.c_[r, uv]
    return ruv.astype('int32')


def make_c4_p4_indices(ksize):
    x = np.random.randn(4, ksize, ksize)
    f = P4FuncArray(v=x)

    if ksize % 2 == 0:
        li = f.left_translation_indices(C4_halfshift[:, None, None, None])
    else:
        li = f.left_translation_indices(C4[:, None, None, None])
    return li.astype('int32')


def make_d4h_z3_indices(ksize):
    assert ksize % 2 == 1
    x = np.random.randn(1, ksize, ksize, ksize)
    f = Z3FuncArray(v=x)
    a = D4h[:, None, None, None, None]
    uvw = f.left_translation_indices(a)
    i = np.zeros(uvw.shape[:-1] + (1,))
    iuvw = np.c_[i, uvw]
    return iuvw.astype('int32')


def make_d4h_d4ht_indices(ksize):
    assert ksize % 2 == 1
    x = np.random.randn(16, ksize, ksize, ksize)
    f = D4htFuncArray(v=x)
    li = f.left_translation_indices(D4h[:, None, None, None, None])
    return li.astype('int32')


def make_c4h_z3_indices(ksize):
    assert ksize % 2 == 1
    x = np.random.randn(1, ksize, ksize, ksize)
    f = Z3FuncArray(v=x)
    a = C4h[:, None, None, None, None]
    uvw = f.left_translation_indices(a)
    i = np.zeros(uvw.shape[:-1] + (1,))
    iuvw = np.c_[i, uvw]
    return iuvw.astype('int32')


def make_c4h_c4ht_indices(ksize):
    assert ksize % 2 == 1
    x = np.random.randn(8, ksize, ksize, ksize)
    f = C4htFuncArray(v=x)
    li = f.left_translation_indices(C4h[:, None, None, None, None])
    return li.astype('int32')


def make_o_z3_indices(ksize):
    assert ksize % 2 == 1
    x = np.random.randn(1, ksize, ksize, ksize)
    f = Z3FuncArray(v=x)
    a = O[:, None, None, None, None]
    uvw = f.left_translation_indices(a)
    i = np.zeros(uvw.shape[:-1] + (1,))
    iuvw = np.c_[i, uvw]
    #import pdb;pdb.set_trace()
    return iuvw.astype('int32')

def make_o_ot_indices(ksize):
    assert ksize % 2 == 1
    x = np.random.randn(24, ksize, ksize, ksize)
    f = OtFuncArray(v=x)
    li = f.left_translation_indices(O[:, None, None, None, None])
    return li.astype('int32')


def make_oh_z3_indices(ksize):
    assert ksize % 2 == 1
    x = np.random.randn(1, ksize, ksize, ksize)
    f = Z3FuncArray(v=x)
    a = Oh[:, None, None, None, None]
    uvw = f.left_translation_indices(a)
    i = np.zeros(uvw.shape[:-1] + (1,))
    iuvw = np.c_[i, uvw]
    return iuvw.astype('int32')


def make_oh_oht_indices(ksize):
    assert ksize % 2 == 1
    x = np.random.randn(48, ksize, ksize, ksize)
    f = OhtFuncArray(v=x)
    li = f.left_translation_indices(Oh[:, None, None, None, None])
    return li.astype('int32')


def make_d4_z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(1, ksize, ksize)
    f = Z2FuncArray(v=x)
    uv = f.left_translation_indices(D4.flatten()[:, None, None, None])
    mr = np.zeros(uv.shape[:-1] + (1,))
    mruv = np.c_[mr, uv]
    return mruv.astype('int32')


def make_d4_p4m_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(8, ksize, ksize)
    f = P4MFuncArray(v=x)
    li = f.left_translation_indices(D4.flatten()[:, None, None, None])
    return li.astype('int32')
'''
def make_z3_z3_indices(ksize):
    assert ksize % 2 == 1
    x = np.random.randn(1, ksize, ksize, ksize)
    f = Z3FuncArray(v=x)
    a = Z3[:, None, None, None, None]
    uvw = f.left_translation_indices(a)
    import pdb;pdb.set_trace()
    i = np.zeros(uvw.shape[:-1] + (1,))
    iuvw = np.c_[i, uvw]
    return iuvw.astype('int32')

def make_z3_z3_indices(ksize):
    assert ksize % 2 == 1
    x = np.random.randn(1, ksize, ksize, ksize)
    f = Z3FuncArray(v=x)
    li = f.left_translation_indices(Z3[:, None, None, None, None])
    return li.astype('int32')
'''
def make_z3_z3_indices(ksize):
    assert ksize % 2 == 1
    x = np.random.randn(1, ksize, ksize, ksize)
    f = Z3FuncArray(v=x)
    a = O[:, None, None, None, None]
    uvw = f.left_translation_indices(a)
    i = np.zeros(uvw.shape[:-1] + (1,))
    iuvw = np.c_[i, uvw]
    #import pdb;pdb.set_trace()
    return np.expand_dims(iuvw[0,...], axis=0).astype('int32')



def flatten_indices(inds):
    """
    The Chainer implementation of G-Conv uses indices into a 5D filter tensor (with an additional axis for the
    transformations H. For the tensorflow implementation it was more convenient to flatten the filter tensor into
    a 3D tensor with shape (output channels, input channels, transformations * width * height).
    This function takes indices in the format required for Chainer and turns them into indices into the flat array
    used by tensorflow.
    :param inds: np.ndarray of shape (output transformations, input transformations, n, n, 3), as output by
    the functions like make_d4_p4m_indices(n).
    :return: np.ndarray of shape (output transformations, input transformations, n, n)
    """
    n = inds.shape[-2]
    nti = inds.shape[1]
    T = inds[..., 0]  # shape (nto, nti, n, n)
    U = inds[..., 1]  # shape (nto, nti, n, n)
    V = inds[..., 2]  # shape (nto, nti, n, n)
    # inds_flat = T * n * n + U * n + V
    inds_flat = U * n * nti + V * nti + T

    return inds_flat


def flatten_indices_3d(inds):
    """
    The Chainer implementation of G-Conv uses indices into a 5D filter tensor (with an additional axis for the
    transformations H. For the tensorflow implementation it was more convenient to flatten the filter tensor into
    a 3D tensor with shape (output channels, input channels, transformations * width * height).
    This function takes indices in the format required for Chainer and turns them into indices into the flat array
    used by tensorflow.
    :param inds: np.ndarray of shape (output transformations, input transformations, n, n, 3), as output by
    the functions like make_d4_p4m_indices(n).
    :return: np.ndarray of shape (output transformations, input transformations, n, n)
    """
    n = inds.shape[-2]
    nti = inds.shape[1]
    T = inds[..., 0]  # shape (nto, nti, n, n, n)
    U = inds[..., 1]  # shape (nto, nti, n, n, n)
    V = inds[..., 2]  # shape (nto, nti, n, n, n)
    W = inds[..., 3]  # shape (nto, nti, n, n, n)
    inds_flat = U * n * n * nti + V * n * nti + W * nti + T
    return inds_flat
