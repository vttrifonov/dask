import numpy as np
from numbers import  Number
from dask.array import chunk
from dask.utils import deepmap
from dask.array.core import _concatenate2
from dask.array.reductions import divide

def numel(x, **kwargs):
    """A reduction to count the number of elements"""

    if hasattr(x, "mask"):
        return chunk.sum(np.ones_like(x), **kwargs)

    shape = x.shape
    keepdims = kwargs.get("keepdims", False)
    axis = kwargs.get("axis", None)
    dtype = kwargs.get("dtype", np.float64)

    if axis is None:
        prod = np.prod(shape, dtype=dtype)
        return (
            np.full((1,) * len(shape), prod, dtype=dtype)
            if keepdims is True
            else prod
        )

    if not isinstance(axis, tuple or list):
        axis = [axis]

    prod = np.prod([shape[dim] for dim in axis])
    if keepdims is True:
        new_shape = tuple(
            shape[dim] if dim not in axis else 1 for dim in range(len(shape))
        )
    else:
        new_shape = tuple(shape[dim] for dim in range(len(shape)) if dim not in axis)
    return np.full(new_shape, prod, dtype=dtype)


# calculations for moments about the mean, calculate moments about the origin
# in the chunk/combine phases and then combine those to produce moments about the mean
# in the agg phase

def moment_chunk(
    A, order=2, sum=chunk.sum, numel=numel, dtype="f8", computing_meta=False, **kwargs
):
    if computing_meta:
        return A
    n = numel(A, **kwargs)
    n = n.astype(np.int64)
    M = [sum(A**i, dtype=dtype, **kwargs) for i in range(1, order + 1)]
    M = np.stack(M, axis=-1)
    return {"n": n, "M": M}


def moment_combine(
    pairs,
    order=2,
    ddof=0,
    dtype="f8",
    sum=np.sum,
    axis=None,
    computing_meta=False,
    **kwargs,
):
    if not isinstance(pairs, list):
        pairs = [pairs]

    kwargs["dtype"] = dtype
    kwargs["keepdims"] = True

    n = deepmap(lambda pair: pair["n"], pairs) if not computing_meta else pairs
    n = _concatenate2(n, axes=axis)
    n = n.sum(axis=axis, **kwargs)

    if computing_meta:
        return n

    M = deepmap(lambda pair: pair["M"], pairs)
    M = _concatenate2(M, axes=axis)
    M = M.sum(axis=axis, **kwargs)

    return {"n": n, "M": M}

def moment_agg(
    pairs,
    order=2,
    ddof=0,
    dtype="f8",
    sum=np.sum,
    axis=None,
    computing_meta=False,
    **kwargs,
):
    if not isinstance(pairs, list):
        pairs = [pairs]

    kwargs["dtype"] = dtype
    # To properly handle ndarrays, the original dimensions need to be kept for
    # part of the calculation.
    keepdim_kw = kwargs.copy()
    keepdim_kw["keepdims"] = True

    n = deepmap(lambda pair: pair["n"], pairs) if not computing_meta else pairs
    n = _concatenate2(n, axes=axis)
    n = n.sum(axis=axis, **keepdim_kw)

    if computing_meta:
        return n

    Ms = deepmap(lambda pair: pair["M"], pairs)
    Ms = _concatenate2(Ms, axes=axis)
    Ms = Ms.sum(axis=axis, **kwargs)        
    
    n = n.sum(axis=axis, **kwargs)

    # this is based on the formula for converting moments about the origin to moments about the mean
    # see https://en.wikipedia.org/wiki/Central_moment
    mu = Ms[...,0]/n
    M = 0
    coef = 1
    for k in range(order):
        M = M + (Ms[...,order-k-1]/n) * coef
        coef = - (coef * (order-k))/(k+1) * mu
    M = M + coef
    M = M*n

    d = n - ddof
    if isinstance(d, Number):
        if d < 0:
            d = np.nan
    elif d is not np.ma.masked:
        d = np.where(d<0, np.nan, d)    
    M = divide(M, d, dtype=dtype)

    return M
