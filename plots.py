# Models          : TARnet   vs    CFRNet
# Datasets        : IHDP     vs    Jobs
# Algorithm       : Fish     vs    ERM
# Confounding     : Yes      vs    No
# Anchor Variable : IHDP -- 'birth-weight' | Jobs -- 'age'

#__________________________________________________________________________________________________________________________________________
import matplotlib.pyplot as plt
import numpy as np

true_ate = [] #[4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05, 4.05]
epochs = []
for i in range(1000):
    epochs.append(i)
    true_ate.append(4.016066896118338)
#__________________________________________________________________________________________________________________________________________

ate_fish_cfrnet = []
pehe_fish_cfrnet = []
# for i in range(10):
ate_fish_cfrnet.append(np.load('results/CF_ATE_48_(-4)_6.npy'))
pehe_fish_cfrnet.append(np.load('results/CF_PEHE_48_(-4)_6.npy'))
ate_fish_cfrnet, pehe_fish_cfrnet = np.mean(np.array(ate_fish_cfrnet), axis = 0), np.mean(np.array(pehe_fish_cfrnet), axis=0)

ate_erm_cfrnet = []
pehe_erm_cfrnet = []
for i in range(1,10):
    ate_erm_cfrnet.append(np.load('results/CE_ATE_100_(-3)' + str(i) + '.npy'))
    pehe_erm_cfrnet.append(np.load('results/CE_PEHE_100_(-3)' + str(i) + '.npy'))
ate_erm_cfrnet, pehe_erm_cfrnet = np.mean(np.array(ate_erm_cfrnet), axis = 0), np.mean(np.array(pehe_erm_cfrnet), axis=0)


fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize = (8,8), sharex = False)
fig.suptitle('Model: CFRNet | Dataset: IHDP')

ax1.plot(epochs, np.abs(ate_erm_cfrnet - true_ate), label = 'ERM')
ax1.plot(epochs, np.abs(ate_fish_cfrnet - true_ate), label = 'FISH')
ax1.legend(loc = 'upper right')
ax1.title.set_text('ATE Error')

ax2.plot(epochs, pehe_erm_cfrnet, label = 'ERM')
ax2.plot(epochs, pehe_fish_cfrnet, label = 'FISH')
ax2.legend(loc = 'upper right')
ax2.title.set_text('PEHE')

for ax in fig.get_axes():
    ax.label_outer()
    
plt.savefig('ATE_Error__CFRNet__IHDP')

#__________________________________________________________________________________________________________________________________________


# ate_erm = []
# pehe_erm = []

# for i in range(10):
#     ate_erm.append(np.load('cfrnet_ERM_ATE_' + str(i) + '.npy'))
#     pehe_erm.append(np.load('cfrnet_ERM_PEHE_' + str(i) + '.npy'))

# ate_erm, pehe_erm = np.mean(np.array(ate_erm), axis = 0), np.mean(np.array(pehe_erm), axis=0)
# print(ate_erm)
# print(pehe_erm)

#__________________________________________________________________________________________________________________________________________

# COMPARISON OF ALL TARNET VARIANTS FOR FISH
# Learning_Rates = {10^{-3}, 10^{-4}} | Batch_Sizes = {8, 16, 32}

# ate_fish_8_3 = []
# pehe_fish_8_3 = []

# for i in range(10):
#     ate_fish_8_3.append(np.load('TF_ATE_8_(-3)_' + str(i) + '.npy'))
#     pehe_fish_8_3.append(np.load('TF_PEHE_8_(-3)_' + str(i) + '.npy'))

# ate_fish_8_3, pehe_fish_8_3 = np.mean(np.array(ate_fish_8_3), axis = 0), np.mean(np.array(pehe_fish_8_3), axis=0)

# ate_fish_16_3 = []
# pehe_fish_16_3 = []

# for i in range(10):
#     ate_fish_16_3.append(np.load('TF_ATE_16_(-3)_' + str(i) + '.npy'))
#     pehe_fish_16_3.append(np.load('TF_PEHE_16_(-3)_' + str(i) + '.npy'))

# ate_fish_16_3, pehe_fish_16_3 = np.mean(np.array(ate_fish_16_3), axis = 0), np.mean(np.array(pehe_fish_16_3), axis=0)

# ate_fish_32_3 = []
# pehe_fish_32_3 = []

# for i in range(10):
#     ate_fish_32_3.append(np.load('TF_ATE_32_(-3)_' + str(i) + '.npy'))
#     pehe_fish_32_3.append(np.load('TF_PEHE_32_(-3)_' + str(i) + '.npy'))

# ate_fish_32_3, pehe_fish_32_3 = np.mean(np.array(ate_fish_32_3), axis = 0), np.mean(np.array(pehe_fish_32_3), axis=0)

# ate_fish_8_4 = []
# pehe_fish_8_4 = []

# for i in range(10):
#     ate_fish_8_4.append(np.load('TF_ATE_8_(-4)_' + str(i) + '.npy'))
#     pehe_fish_8_4.append(np.load('TF_PEHE_8_(-4)_' + str(i) + '.npy'))

# ate_fish_8_4, pehe_fish_8_4 = np.mean(np.array(ate_fish_8_4), axis = 0), np.mean(np.array(pehe_fish_8_4), axis=0)


# ate_fish_16_4 = []
# pehe_fish_16_4 = []

# for i in range(10):
#     ate_fish_16_4.append(np.load('TF_ATE_16_(-4)_' + str(i) + '.npy'))
#     pehe_fish_16_4.append(np.load('TF_PEHE_16_(-4)_' + str(i) + '.npy'))

# ate_fish_16_4, pehe_fish_16_4 = np.mean(np.array(ate_fish_16_4), axis = 0), np.mean(np.array(pehe_fish_16_4), axis=0)

# ate_fish_32_4 = []
# pehe_fish_32_4 = []

# for i in range(10):
#     ate_fish_32_4.append(np.load('TF_ATE_32_(-4)_' + str(i) + '.npy'))
#     pehe_fish_32_4.append(np.load('TF_PEHE_32_(-4)_' + str(i) + '.npy'))

# ate_fish_32_4, pehe_fish_32_4 = np.mean(np.array(ate_fish_32_4), axis = 0), np.mean(np.array(pehe_fish_32_4), axis=0)

# fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize = (20,8), sharex = False)
# fig.suptitle('Model: TARNet for Fish | Dataset: IHDP ')

# ax1.plot(epochs, ate_fish_8_3, label = 'BS: 8 | LR: (-3)')
# ax1.plot(epochs, ate_fish_16_3, label = 'BS: 16 | LR: (-3)')
# ax1.plot(epochs, ate_fish_32_3, label = 'BS: 32 | LR: (-3)')
# ax1.plot(epochs, ate_fish_8_4, label = 'BS: 8 | LR: (-4)')
# ax1.plot(epochs, ate_fish_16_4, label = 'BS: 16 | LR: (-4)')
# ax1.plot(epochs, ate_fish_32_4, label = 'BS: 32 | LR: (-4)')
# ax1.plot(epochs, true_ate, label = 'True ATE')
# ax1.legend(loc = 'upper right')
# ax1.title.set_text('ATE')

# ax2.plot(epochs, pehe_fish_8_3, label = 'BS: 8 | LR: (-3)')
# ax2.plot(epochs, pehe_fish_16_3, label = 'BS: 16 | LR: (-3)')
# ax2.plot(epochs, pehe_fish_32_3, label = 'BS: 32 | LR: (-3)')
# ax2.plot(epochs, pehe_fish_8_4, label = 'BS: 8 | LR: (-4)')
# ax2.plot(epochs, pehe_fish_16_4, label = 'BS: 16 | LR: (-4)')
# ax2.plot(epochs, pehe_fish_32_4, label = 'BS: 32 | LR: (-4)')
# ax2.legend(loc = 'upper right')
# ax2.title.set_text('PEHE')

# for ax in fig.get_axes():
#     ax.label_outer()
    
# plt.savefig('TARNet__FISH')
#__________________________________________________________________________________________________________________________________________

# COMPARISON OF ALL TARNET VARIANTS FOR ERM
# Learning_Rates = {10^{-3}, 10^{-4}} | Batch_Sizes = {8, 16, 32}


# ate_erm_8_3 = []
# pehe_erm_8_3 = []

# for i in range(10):
#     ate_erm_8_3.append(np.load('TE_ATE_8_(-3)' + str(i) + '.npy'))
#     pehe_erm_8_3.append(np.load('TE_PEHE_8_(-3)' + str(i) + '.npy'))

# ate_erm_8_3, pehe_erm_8_3 = np.mean(np.array(ate_erm_8_3), axis = 0), np.mean(np.array(pehe_erm_8_3), axis=0)

# ate_erm_16_3 = []
# pehe_erm_16_3 = []

# for i in range(10):
#     ate_erm_16_3.append(np.load('TE_ATE_16_(-3)' + str(i) + '.npy'))
#     pehe_erm_16_3.append(np.load('TE_PEHE_16_(-3)' + str(i) + '.npy'))

# ate_erm_16_3, pehe_erm_16_3 = np.mean(np.array(ate_erm_16_3), axis = 0), np.mean(np.array(pehe_erm_16_3), axis=0)

# ate_erm_32_3 = []
# pehe_erm_32_3 = []

# for i in range(10):
#     ate_erm_32_3.append(np.load('TE_ATE_32_(-3)' + str(i) + '.npy'))
#     pehe_erm_32_3.append(np.load('TE_PEHE_32_(-3)' + str(i) + '.npy'))

# ate_erm_32_3, pehe_erm_32_3 = np.mean(np.array(ate_erm_32_3), axis = 0), np.mean(np.array(pehe_erm_32_3), axis=0)

# ate_erm_8_4 = []
# pehe_erm_8_4 = []

# for i in range(10):
#     ate_erm_8_4.append(np.load('TE_ATE_8_(-4)' + str(i) + '.npy'))
#     pehe_erm_8_4.append(np.load('TE_PEHE_8_(-4)' + str(i) + '.npy'))

# ate_erm_8_4, pehe_erm_8_4 = np.mean(np.array(ate_erm_8_4), axis = 0), np.mean(np.array(pehe_erm_8_4), axis=0)


# ate_erm_16_4 = []
# pehe_erm_16_4 = []

# for i in range(10):
#     ate_erm_16_4.append(np.load('TE_ATE_16_(-4)' + str(i) + '.npy'))
#     pehe_erm_16_4.append(np.load('TE_PEHE_16_(-4)' + str(i) + '.npy'))

# ate_erm_16_4, pehe_erm_16_4 = np.mean(np.array(ate_erm_16_4), axis = 0), np.mean(np.array(pehe_erm_16_4), axis=0)

# ate_erm_32_4 = []
# pehe_erm_32_4 = []

# for i in range(10):
#     ate_erm_32_4.append(np.load('TE_ATE_32_(-4)' + str(i) + '.npy'))
#     pehe_erm_32_4.append(np.load('TE_PEHE_32_(-4)' + str(i) + '.npy'))

# ate_erm_32_4, pehe_erm_32_4 = np.mean(np.array(ate_erm_32_4), axis = 0), np.mean(np.array(pehe_erm_32_4), axis=0)

# fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize = (20,8), sharex = False)
# fig.suptitle('Model: TARNet for erm | Dataset: IHDP ')

# ax1.plot(epochs, ate_erm_8_3, label = 'BS: 8 | LR: (-3)')
# ax1.plot(epochs, ate_erm_16_3, label = 'BS: 16 | LR: (-3)')
# ax1.plot(epochs, ate_erm_32_3, label = 'BS: 32 | LR: (-3)')
# ax1.plot(epochs, ate_erm_8_4, label = 'BS: 8 | LR: (-4)')
# ax1.plot(epochs, ate_erm_16_4, label = 'BS: 16 | LR: (-4)')
# ax1.plot(epochs, ate_erm_32_4, label = 'BS: 32 | LR: (-4)')
# ax1.plot(epochs, true_ate, label = 'True ATE')
# ax1.legend(loc = 'upper right')
# ax1.title.set_text('ATE')

# ax2.plot(epochs, pehe_erm_8_3, label = 'BS: 8 | LR: (-3)')
# ax2.plot(epochs, pehe_erm_16_3, label = 'BS: 16 | LR: (-3)')
# ax2.plot(epochs, pehe_erm_32_3, label = 'BS: 32 | LR: (-3)')
# ax2.plot(epochs, pehe_erm_8_4, label = 'BS: 8 | LR: (-4)')
# ax2.plot(epochs, pehe_erm_16_4, label = 'BS: 16 | LR: (-4)')
# ax2.plot(epochs, pehe_erm_32_4, label = 'BS: 32 | LR: (-4)')
# ax2.legend(loc = 'upper right')
# ax2.title.set_text('PEHE')

# for ax in fig.get_axes():
#     ax.label_outer()
    
# plt.savefig('TARNet__erm')

#__________________________________________________________________________________________________________________________________________

# COMPARISON OF ALL CFRNET VARIANTS FOR ERM and FISH

# ate_erm_100_3 = []
# pehe_erm_100_3 = []

# for i in range(5):
#     ate_erm_100_3.append(np.load('CE_ATE_48_(-3)' + str(i) + '.npy'))
#     pehe_erm_100_3.append(np.load('CE_PEHE_48_(-3)' + str(i) + '.npy'))

# ate_erm_100_3, pehe_erm_100_3 = np.mean(np.array(ate_erm_100_3), axis = 0), np.mean(np.array(pehe_erm_100_3), axis=0)

# ate_fish_100_4_0 = []
# pehe_fish_100_4_0 = []
# ate_fish_100_4_0.append(np.load('CF_ATE_48_(-4)_0.npy'))
# pehe_fish_100_4_0.append(np.load('CF_PEHE_48_(-4)_0.npy'))

# ate_fish_100_4_1 = []
# pehe_fish_100_4_1 = []
# ate_fish_100_4_1.append(np.load('CF_ATE_48_(-4)_1.npy'))
# pehe_fish_100_4_1.append(np.load('CF_PEHE_48_(-4)_1.npy'))

# ate_fish_100_4_2 = []
# pehe_fish_100_4_2 = []
# ate_fish_100_4_2.append(np.load('CF_ATE_48_(-4)_2.npy'))
# pehe_fish_100_4_2.append(np.load('CF_PEHE_48_(-4)_2.npy'))

# ate_fish_100_4_3 = []
# pehe_fish_100_4_3 = []
# ate_fish_100_4_3.append(np.load('CF_ATE_48_(-4)_3.npy'))
# pehe_fish_100_4_3.append(np.load('CF_PEHE_48_(-4)_3.npy'))

# ate_fish_100_4_4 = []
# pehe_fish_100_4_4 = []
# ate_fish_100_4_4.append(np.load('CF_ATE_48_(-4)_4.npy'))
# pehe_fish_100_4_4.append(np.load('CF_PEHE_48_(-4)_4.npy'))

# ate_fish_100_4_5 = []
# pehe_fish_100_4_5 = []
# ate_fish_100_4_5.append(np.load('CF_ATE_48_(-4)_5.npy'))
# pehe_fish_100_4_5.append(np.load('CF_PEHE_48_(-4)_5.npy'))

# ate_fish_100_4_6 = []
# pehe_fish_100_4_6 = []
# ate_fish_100_4_6.append(np.load('CF_ATE_48_(-4)_6.npy'))
# pehe_fish_100_4_6.append(np.load('CF_PEHE_48_(-4)_6.npy'))

# ate_fish_100_4_7 = []
# pehe_fish_100_4_7 = []
# ate_fish_100_4_7.append(np.load('CF_ATE_48_(-4)_7.npy'))
# pehe_fish_100_4_7.append(np.load('CF_PEHE_48_(-4)_7.npy'))

# ate_fish_100_4_8 = []
# pehe_fish_100_4_8 = []
# ate_fish_100_4_8.append(np.load('CF_ATE_48_(-4)_8.npy'))
# pehe_fish_100_4_8.append(np.load('CF_PEHE_48_(-4)_8.npy'))

# ate_fish_100_4_9 = []
# pehe_fish_100_4_9 = []
# ate_fish_100_4_9.append(np.load('CF_ATE_48_(-4)_9.npy'))
# pehe_fish_100_4_9.append(np.load('CF_PEHE_48_(-4)_9.npy'))

# fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize = (40,8), sharex = False)

# # ax1.plot(epochs, np.array(ate_fish_100_4_0).reshape(1000,), label = 'Iteration 0')
# # ax1.plot(epochs, np.array(ate_fish_100_4_1).reshape(1000,), label = 'Iteration 1')
# # ax1.plot(epochs, np.array(ate_fish_100_4_2).reshape(1000,), label = 'Iteration 2')
# # ax1.plot(epochs, np.array(ate_fish_100_4_3).reshape(1000,), label = 'Iteration 3')
# # ax1.plot(epochs, np.array(ate_fish_100_4_4).reshape(1000,), label = 'Iteration 4')
# # ax1.plot(epochs, np.array(ate_fish_100_4_5).reshape(1000,), label = 'Iteration 5')
# ax1.plot(epochs, np.array(ate_fish_100_4_6).reshape(1000,), label = 'Iteration 6')
# ax1.plot(epochs, np.array(ate_fish_100_4_7).reshape(1000,), label = 'Iteration 7')
# # ax1.plot(epochs, np.array(ate_fish_100_4_8).reshape(1000,), label = 'Iteration 8')
# # ax1.plot(epochs, np.array(ate_fish_100_4_9).reshape(1000,), label = 'Iteration 9')
# ax1.plot(epochs, ate_erm_100_3, label = 'ERM')

# ax1.plot(epochs, true_ate, label = 'True ATE')
# ax1.legend(loc = 'upper right')
# ax1.title.set_text('ATE')


# # ax2.plot(epochs, np.array(pehe_fish_100_4_0).reshape(1000,), label = 'Iteration 0')
# # ax2.plot(epochs, np.array(pehe_fish_100_4_1).reshape(1000,), label = 'Iteration 1)')
# # ax2.plot(epochs, np.array(pehe_fish_100_4_2).reshape(1000,), label = 'Iteration 2)')
# # ax2.plot(epochs, np.array(pehe_fish_100_4_3).reshape(1000,), label = 'Iteration 3')
# # ax2.plot(epochs, np.array(pehe_fish_100_4_4).reshape(1000,), label = 'Iteration 4)')
# # ax2.plot(epochs, np.array(pehe_fish_100_4_5).reshape(1000,), label = 'Iteration 5)')
# ax2.plot(epochs, np.array(pehe_fish_100_4_6).reshape(1000,), label = 'Iteration 6)')
# ax2.plot(epochs, np.array(pehe_fish_100_4_7).reshape(1000,), label = 'Iteration 7)')
# # ax2.plot(epochs, np.array(pehe_fish_100_4_8).reshape(1000,), label = 'Iteration 8)')
# # ax2.plot(epochs, np.array(pehe_fish_100_4_9).reshape(1000,), label = 'Iteration 9)')
# ax2.plot(epochs, pehe_erm_100_3, label = 'ERM')
# ax2.legend(loc = 'upper right')
# ax2.title.set_text('PEHE')

# for ax in fig.get_axes():
#     ax.label_outer()
    
# plt.savefig('CFRNet__Fish&ERM_BS-100')

#__________________________________________________________________________________________________________________________________________

# TARNet | Fish vs Erm | IHDP

# erm_ate = []
# erm_pehe = []
# for i in range(10):
#     erm_ate.append(np.load('TE_ATE_32_(-4)' + str(i) + '.npy'))
#     erm_pehe.append(np.load('TE_PEHE_32_(-4)' + str(i) + '.npy'))
# erm_ate, erm_pehe = np.mean(np.array(erm_ate), axis = 0), np.mean(np.array(erm_pehe), axis=0)

# fish_ate = []
# fish_pehe = []
# for i in range(10):
#     fish_ate.append(np.load('TF_ATE_8_(-4)_' + str(i) + '.npy'))
#     fish_pehe.append(np.load('TF_PEHE_8_(-4)_' + str(i) + '.npy'))
# fish_ate, fish_pehe = np.mean(np.array(fish_ate), axis = 0), np.mean(np.array(fish_pehe), axis=0)

# fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize = (20,8), sharex = False)
# fig.suptitle('TARNet | Fish vs Erm | IHDP')

# ax1.plot(epochs, erm_ate, label = 'ERM')
# ax1.plot(epochs, fish_ate, label = 'FISH')
# ax1.plot(epochs, true_ate, label = 'True ATE')
# ax1.legend(loc = 'upper right')
# ax1.title.set_text('ATE')

# ax2.plot(epochs, erm_pehe , label = 'ERM' )
# ax2.plot(epochs, fish_pehe , label = 'FISH')
# ax2.legend(loc = 'upper right')
# ax2.title.set_text('PEHE')

# for ax in fig.get_axes():
#     ax.label_outer()
    
# plt.savefig('Tarnet__Fish&Erm')
