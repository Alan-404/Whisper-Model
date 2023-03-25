#%%
import torch
import numpy as np
# %%
ref = np.array([[1,2,3,4,5,6], [1,2,3, 0, 0, 0]])
pred = np.array([[1,3,4,7,8,6], [1,2,3, 0, 0, 0]])
#%%
ref.shape
# %%

# %%
ref = torch.tensor(ref)
pred = torch.tensor(pred)
# %%
from model.metric import WER
# %%
metric = WER()
# %%
result = metric.score(pred, ref)
# %%
result
# %%
