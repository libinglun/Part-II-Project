import numpy as np

file_name = 'fix_8states_multinomial_same_trans_diff_stick_full_bayesian_rlt_0.npz'
data = np.load(file_name)

np.set_printoptions(threshold=np.inf)

for key in data:
    print(key)

print(data['loglik'], data['hamming'])
# print(data['zt'], data['yt'])

# data.close()
