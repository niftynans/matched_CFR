# ERM / FISH
# TARNet
# Confounding / Without Confounding
# Birth-Weight / head-circumference / age 

import matplotlib.pyplot as plt
import numpy as np

true_ate = [] #[4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05]
epochs = []
for i in range(1000):
    epochs.append(i)
    true_ate.append(4.016066896118338)
# epochs = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]

##############################

ate_erm = []
pehe_erm = []

for i in range(10):
    ate_erm.append(np.load('tarnet_ERM_ATE_' + str(i) + '.npy'))
    pehe_erm.append(np.load('tarnet_ERM_PEHE_' + str(i) + '.npy'))

ate_erm, pehe_erm = np.mean(np.array(ate_erm), axis = 0), np.mean(np.array(pehe_erm), axis=0)
# print(ate_erm)
# print(pehe_erm)
##############################

ate_fish = []
pehe_fish = []

for i in range(10):
    ate_fish.append(np.load('tarnet_FISH_ATE_' + str(i) + '.npy'))
    pehe_fish.append(np.load('tarnet_FISH_PEHE_' + str(i) + '.npy'))

ate_fish, pehe_fish = np.mean(np.array(ate_fish), axis = 0), np.mean(np.array(pehe_fish), axis=0)
# print(ate_fish)
# print(pehe_fish)

###############################

fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize = (20,8), sharex = False)
fig.suptitle('Model: TARNet | Dataset: IHDP (With Confounding) | Anchor Variable: Birth-Weight')

ax1.plot(epochs, ate_erm, label = 'ERM')
ax1.plot(epochs, ate_fish, label = 'FISH')
ax1.plot(epochs, true_ate, label = 'True ATE')
ax1.legend(loc = 'upper right')
ax1.title.set_text('ATE')

ax2.plot(epochs, pehe_erm , label = 'ERM' )
ax2.plot(epochs, pehe_fish , label = 'FISH')
ax2.legend(loc = 'upper right')
ax2.title.set_text('PEHE')

for ax in fig.get_axes():
    ax.label_outer()
    
plt.savefig('TARNet_CF_Birth-Weight')

################################

fig, ((ax1)) = plt.subplots(1, 1, figsize = (20,8), sharex = False)
fig.suptitle('Model: TARNet | Dataset: IHDP (With Confounding) | Anchor Variable: Birth-Weight')

ax1.plot(epochs, np.abs(ate_erm - true_ate), label = 'ERM')
ax1.plot(epochs, np.abs(ate_fish - true_ate), label = 'FISH')
ax1.legend(loc = 'upper right')
ax1.title.set_text('ATE Error')


for ax in fig.get_axes():
    ax.label_outer()
    
plt.savefig('ATE-Error_TARNet_CF_Birth-Weight')