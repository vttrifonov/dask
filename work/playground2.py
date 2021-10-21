from dask_ml.linear_model import LogisticRegression
import numpy as np
import dask.array as da
import sparse

x1 = da.random.random((10, 5), chunks=(2, 'auto'))
x1[x1 < 0.8] = 0
x1 = da.concatenate([da.ones((x1.shape[0],1), chunks=(x1.shape[0],1)), x1], axis=1)
x1 = x1.rechunk((x1.chunks[0], 'auto'))

x1.mean(axis=0).compute()

x2 = x1.map_blocks(sparse.COO.from_numpy)

y = np.where(da.random.random((x1.shape[0],), chunks=(x1.chunks[0],))<0.5, 1, 0)

lr = LogisticRegression(penalty='l1', solver='proximal_grad', fit_intercept=False)

#z1 = lr.fit(x1, y)

#z2 = lr.fit(x2, y)


x1 = np.random.random((5,))
#x1 = sparse.COO.from_numpy(x1)
#x1[x1 < 0.8] = 0
np.where(x1==0)
