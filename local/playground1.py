#%%
import dask
import dask.array as da
import sparse
import numpy as np

#%%
x = (da.random.random((7,7,7), chunks=(2,2,2))<0.3).astype(np.float)
x = x.map_blocks(sparse.COO.from_numpy)
y = x.compute().todense()

# %%
t = lambda x: [x.mean(), x.mean(axis=1), x.mean(axis=(0,2)), x.std(), x.std(axis=1), x.std(axis=(0,2))]
tx = dask.compute(t(x))[0]
ty = t(y)
[np.sum((x-y)*2) for x, y in zip(tx, ty)]

# %%
from dask.array.tests.test_reductions import assert_eq

def moment(x, n, axis=None):
    return ((x - x.mean(axis=axis, keepdims=True)) ** n).sum(
        axis=axis
    ) / np.ones_like(x).sum(axis=axis)

# Poorly conditioned
x = np.array([1.0, 2.0, 3.0] * 10).reshape((3, 10)) + 1e8
a = da.from_array(x, chunks=5)
assert_eq(a.moment(2), moment(x, 2))
assert_eq(a.moment(3), moment(x, 3))
assert_eq(a.moment(4), moment(x, 4))

(a.moment(2)-moment(x, 2)).compute()
(a.moment(3)-moment(x, 3)).compute()
(a.moment(4)-moment(x, 4)).compute()

x = np.random.random(100)+1e8
x1 = x.mean()
x2 = (x**2).mean()
x3 = ((x-x1)**2).mean()
(x2 - x1**2) - x3

x = np.arange(1, 122).reshape((11, 11)).astype("f8")
a = da.from_array(x, chunks=(4, 4))
assert_eq(a.moment(4, axis=1), moment(x, 4, axis=1))
assert_eq(a.moment(4, axis=(1, 0)), moment(x, 4, axis=(1, 0)))

# Tree reduction
assert_eq(a.moment(order=4, split_every=4), moment(x, 4))
assert_eq(a.moment(order=4, axis=0, split_every=4), moment(x, 4, axis=0))
assert_eq(a.moment(order=4, axis=1, split_every=4), moment(x, 4, axis=1))# %%
