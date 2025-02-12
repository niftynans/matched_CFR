import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

alphas = [0, 1e1, 1e2, 1e3]# , 1e4]
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# ate_error1 = [0.172, 0.112, 0.099, 0.103]# , 0.113]
# ate_error2 = [0.145, 0.104, 0.101, 0.102]# , 0.111]
# ate_error3 = [0.141, 0.100, 0.100, 0.103]# , 0.113]
# ate_error4 = [0.139, 0.100, 0.101, 0.102]# , 0.111]
# ate_error5 = [0.140, 0.098, 0.101, 0.103]# , 0.109]
# ate_error6 = [0.136, 0.099, 0.100, 0.103]# , 0.110]
# ate_error7 = [0.136, 0.098, 0.101, 0.102]# , 0.110]
# ate_error8 = [0.099, 0.099, 0.101, 0.101]# , 0.111]
# ate_error9 = [0.136, 0.097, 0.102, 0.102]# , 0.111]
# ate_error10 = [0.136, 0.099, 0.102, 0.102]# , 0.111]

ate_error1  = [1.874, 1.738, 1.692, 1.688]#, 1.685]
ate_error2  = [1.859, 1.727, 1.691, 1.688]#, 1.684]
ate_error3  = [1.857, 1.722, 1.692, 1.688]#, 1.683]
ate_error4  = [1.857, 1.721, 1.692, 1.690]#, 1.684]
ate_error5  = [1.859, 1.721, 1.692, 1.689]#, 1.684]
ate_error6  = [1.855, 1.719, 1.691, 1.689]#, 1.683]
ate_error7  = [1.854, 1.717, 1.691, 1.688]#, 1.683]
ate_error8  = [1.718, 1.718, 1.693, 1.689]#, 1.683]
ate_error9  = [1.853, 1.716, 1.692, 1.689]#, 1.684]
ate_error10 = [1.852, 1.718, 1.691, 1.687]#, 1.685]

# plt.suptitle('Fish Update Hyperparameter, Ɛ vs IPM Scaling Hyperparameter, α', fontsize=20)
plt.figure(figsize=(10, 25))
plt.xscale("symlog")

plt.plot(alphas, ate_error1, linewidth = 3,
        marker='o', markersize=7, label = 'Ɛ = 0.1')
plt.plot(alphas, ate_error2, linewidth = 3,
        marker='o', markersize=7, label = 'Ɛ = 0.2')
plt.plot(alphas, ate_error3, linewidth = 3,
        marker='o', markersize=7, label = 'Ɛ = 0.3')
plt.plot(alphas, ate_error4, linewidth = 3,
        marker='o', markersize=7, label = 'Ɛ = 0.4')
plt.plot(alphas, ate_error5, linewidth = 3,
        marker='o', markersize=7, label = 'Ɛ = 0.5')
plt.plot(alphas, ate_error6, linewidth = 3,
        marker='o', markersize=7, label = 'Ɛ = 0.6')
plt.plot(alphas, ate_error7, linewidth = 3,
        marker='o', markersize=7, label = 'Ɛ = 0.7')
plt.plot(alphas, ate_error8, linewidth = 3,
        marker='o', markersize=7, label = 'Ɛ = 0.8')
plt.plot(alphas, ate_error9, linewidth = 3,
        marker='o', markersize=7, label = 'Ɛ = 0.9')
plt.plot(alphas, ate_error10, linewidth = 3,
        marker='o', markersize=7, label = 'Ɛ = 1.0')


plt.xlabel("Alpha")
plt.ylabel("PEHE Error")
plt.legend(loc='upper right', title='Legend', ncol=5)

# plt.tight_layout()
plt.show()
