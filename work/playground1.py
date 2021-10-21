import dask.array as da
import sparse

x = da.random.random((2, 3, 4), chunks=(1, 2, 2))
x[x < 0.8] = 0

y = x.map_blocks(sparse.COO.from_numpy)

(x.mean()-y.mean()).compute()
(x.var()-y.var()).compute()

(x.var(axis=0)-y.var(axis=0)).compute()