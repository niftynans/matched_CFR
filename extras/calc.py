import numpy as np

cfr_ate_out = np.load('../ablation/cattaneo/TRY.npy')
print(np.mean(cfr_ate_out[2:]), ' + ', np.std(cfr_ate_out[2:]))
