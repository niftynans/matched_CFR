import matplotlib.pyplot as plt
import numpy as np

# # IHDP
# # __________________________________________________________________________________________________________________________________________
# epochs = []
# true_ate = []

# for i in range(0, 1000, 10):
#     epochs.append(i)
#     true_ate.append(4.016066896118338)

# ihdp_ate_in_matched_cfr = []
# ihdp_ate_out_matched_cfr = []
# ihdp_pehe_in_matched_cfr = []
# ihdp_pehe_out_matched_cfr = []
# ihdp_rmse_in_matched_cfr = []
# ihdp_rmse_out_matched_cfr = []


# ihdp_ate_in_matched_tar = []
# ihdp_ate_out_matched_tar = []
# ihdp_pehe_in_matched_tar = []
# ihdp_pehe_out_matched_tar = []
# ihdp_rmse_in_matched_tar = []
# ihdp_rmse_out_matched_tar = []

# ihdp_ate_in_cfr = []
# ihdp_ate_out_cfr = []
# ihdp_pehe_in_cfr = []
# ihdp_pehe_out_cfr = []
# ihdp_rmse_in_cfr = []
# ihdp_rmse_out_cfr = []


# ihdp_ate_in_tar = []
# ihdp_ate_out_tar = []
# ihdp_pehe_in_tar = []
# ihdp_pehe_out_tar = []
# ihdp_rmse_in_tar = []
# ihdp_rmse_out_tar = []

# for i in range(1):

#     ihdp_ate_out_cfr.append(np.load('../results/ihdp/CFRNet/CFR_ATE_OUT.npy'))
#     ihdp_pehe_out_cfr.append(np.load('../results/ihdp/CFRNet/CFR_PEHE_OUT.npy'))

#     ihdp_ate_out_tar.append(np.load('../results/ihdp/TARNet/TAR_ATE_OUT.npy'))
#     ihdp_pehe_out_tar.append(np.load('../results/ihdp/TARNet/TAR_PEHE_OUT.npy'))

#     ihdp_ate_in_cfr.append(np.load('../results/ihdp/CFRNet/CFR_ATE_IN.npy'))
#     ihdp_pehe_in_cfr.append(np.load('../results/ihdp/CFRNet/CFR_PEHE_IN.npy'))

#     ihdp_ate_in_tar.append(np.load('../results/ihdp/TARNet/TAR_ATE_IN.npy'))
#     ihdp_pehe_in_tar.append(np.load('../results/ihdp/TARNet/TAR_PEHE_IN.npy'))

#     ihdp_ate_out_matched_cfr.append(np.load('../ablation/ihdp/ATE_OUT_A1_B9.npy'))
#     ihdp_pehe_out_matched_cfr.append(np.load('../ablation/ihdp/PEHE_OUT_A1_B9.npy'))

#     ihdp_ate_out_matched_tar.append(np.load('../ablation/ihdp/ATE_OUT_A0_B9.npy'))
#     ihdp_pehe_out_matched_tar.append(np.load('../ablation/ihdp/PEHE_OUT_A0_B9.npy'))

#     ihdp_ate_in_matched_cfr.append(np.load('../ablation/ihdp/ATE_OUT_A1_B9.npy'))
#     ihdp_pehe_in_matched_cfr.append(np.load('../ablation/ihdp/PEHE_OUT_A1_B9.npy'))

#     ihdp_ate_in_matched_tar.append(np.load('../ablation/ihdp/ATE_OUT_A0_B9.npy'))
#     ihdp_pehe_in_matched_tar.append(np.load('../ablation/ihdp/PEHE_OUT_A0_B9.npy'))

# fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize = (12,4), sharex = False)

# ax1.plot(epochs, np.abs(true_ate - np.mean(ihdp_ate_out_cfr, axis = 0)), label = 'CFRNet', linewidth = 2.0)
# ax1.plot(epochs, np.abs(true_ate - np.mean(ihdp_ate_out_tar, axis = 0)), label = 'TARNet', linewidth = 2.0)
# ax1.plot(epochs, np.abs(true_ate - np.mean(ihdp_ate_out_matched_cfr, axis = 0)), label = 'Matched - CFRNet', linewidth = 2.0)
# ax1.plot(epochs, np.abs(true_ate - np.mean(ihdp_ate_out_matched_tar, axis = 0)), label = 'Matched - TARNet', linewidth = 2.0)

# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('ATE Error')
# ax1.legend(loc = 'upper right')
# ax1.title.set_text('Dataset: IHDP | Metric: Out-of-Sample ATE Error')

# ax2.plot(epochs, np.mean(ihdp_pehe_out_cfr, axis = 0), label = 'CFRNet', linewidth = 2.0)
# ax2.plot(epochs, np.mean(ihdp_pehe_out_tar, axis = 0), label = 'TARNet', linewidth = 2.0)
# ax2.plot(epochs, np.mean(ihdp_pehe_out_matched_cfr, axis = 0), label = 'Matched - CFRNet', linewidth = 2.0)
# ax2.plot(epochs, np.mean(ihdp_pehe_out_matched_tar, axis = 0), label = 'Matched - TARNet', linewidth = 2.0)
# ax2.set_xlabel('Epochs')
# ax2.set_ylabel('PEHE Error')
# ax2.legend(loc = 'upper right')
# ax2.title.set_text('Dataset: IHDP | Metric: Out-of-Sample PEHE Error')

# plt.savefig('Fig1')


# cattaneo
# __________________________________________________________________________________________________________________________________________
# epochs = []
# true_ate = []

# for i in range(0, 250, 10):
#     epochs.append(i)

# cattaneo_ate_in_cfr_erm = []
# cattaneo_ate_out_cfr_erm = []
# cattaneo_rmse_in_cfr_erm = []
# cattaneo_rmse_out_cfr_erm = []

# cattaneo_ate_in_tar_erm = []
# cattaneo_ate_out_tar_erm = []
# cattaneo_rmse_in_tar_erm = []
# cattaneo_rmse_out_tar_erm = []

# cattaneo_ate_in_cfr_fish = []
# cattaneo_ate_out_cfr_fish = []
# cattaneo_rmse_in_cfr_fish = []
# cattaneo_rmse_out_cfr_fish = []

# cattaneo_ate_in_tar_fish = []
# cattaneo_ate_out_tar_fish = []
# cattaneo_rmse_in_tar_fish = []
# cattaneo_rmse_out_tar_fish = []

# for i in range(3):
#     cattaneo_ate_out_cfr_erm.append(np.load('../results/cattaneo/CFRNet/ATE_OUT_' + str(i) + '.npy'))
#     # cattaneo_ate_out_tar_erm.append(np.load('../results/cattaneo/TARNet/TARNet_ATE_OUT_' + str(i) + '.npy'))
#     cattaneo_ate_out_cfr_fish.append(np.load('../results/cattaneo/Matched_CFRNet/ATE_OUT_' + str(i) + '.npy'))
#     # cattaneo_ate_out_tar_fish.append(np.load('../results/cattaneo/Matched_TARNet/Matched_TARNet_ATE_OUT_' + str(i) + '.npy'))

# fig, ((ax1)) = plt.subplots(1, 1, figsize = (16,8), sharex = False)
# ax1.set_yscale('symlog')
# ax1.plot(epochs, np.mean(cattaneo_ate_out_cfr_erm, axis = 0), label = 'CFRNet', linewidth = 3.0)
# # ax1.plot(epochs, np.mean(cattaneo_ate_out_tar_erm, axis = 0), label = 'TARNet')
# ax1.plot(epochs, np.mean(cattaneo_ate_out_cfr_fish, axis = 0), label = 'Matched - CFRNet', linewidth = 3.0)
# # ax1.plot(epochs, np.mean(cattaneo_ate_out_tar_fish, axis = 0), label = 'Matched - TARNet')
# ax1.axhspan(-250, -300, color='green', alpha=0.4)
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('ATE Error')
# ax1.legend(loc = 'lower right')
# ax1.title.set_text('Dataset: Cattaneo | Outof-Sample ATE Values')

# plt.savefig('Fig2')

# epochs = []
# true_ate = []


# Jobs
# __________________________________________________________________________________________________________________________________________


# epochs = []

# for i in range(0, 1000):
#     epochs.append(i)

# jobs_att_in_cfr_erm = []
# jobs_att_out_cfr_erm = []
# jobs_atc_in_cfr_erm = []
# jobs_atc_out_cfr_erm = []
# jobs_rmse_in_cfr_erm = []
# jobs_rmse_out_cfr_erm = []

# jobs_att_in_tar_erm = []
# jobs_att_out_tar_erm = []
# jobs_atc_in_tar_erm = []
# jobs_atc_out_tar_erm = []
# jobs_rmse_in_tar_erm = []
# jobs_rmse_out_tar_erm = []

# jobs_att_in_cfr_fish = []
# jobs_att_out_cfr_fish = []
# jobs_atc_in_cfr_fish = []
# jobs_atc_out_cfr_fish = []
# jobs_rmse_in_cfr_fish = []
# jobs_rmse_out_cfr_fish = []

# jobs_att_in_tar_fish = []
# jobs_att_out_tar_fish = []
# jobs_atc_in_tar_fish = []
# jobs_atc_out_tar_fish = []
# jobs_rmse_in_tar_fish = []
# jobs_rmse_out_tar_fish = []

# for i in range(3):
#     jobs_att_in_cfr_erm.append(np.load('../results/jobs/CFRNet/CFRNet_ATT_IN_' + str(i) + '.npy'))
#     jobs_att_out_cfr_erm.append(np.load('../results/jobs/CFRNet/CFRNet_ATT_OUT_' + str(i) + '.npy'))
#     jobs_atc_in_cfr_erm.append(np.load('../results/jobs/CFRNet/CFRNet_ATC_IN_' + str(i) + '.npy'))
#     jobs_atc_out_cfr_erm.append(np.load('../results/jobs/CFRNet/CFRNet_ATC_OUT_' + str(i) + '.npy'))
#     jobs_rmse_in_cfr_erm.append(np.load('../results/jobs/CFRNet/CFRNet_RMSE_IN_' + str(i) + '.npy'))
#     jobs_rmse_out_cfr_erm.append(np.load('../results/jobs/CFRNet/CFRNet_RMSE_OUT_' + str(i) + '.npy'))

#     jobs_att_in_tar_erm.append(np.load('../results/jobs/TARNet/TARNet_ATT_IN_' + str(i) + '.npy'))
#     jobs_att_out_tar_erm.append(np.load('../results/jobs/TARNet/TARNet_ATT_OUT_' + str(i) + '.npy'))
#     jobs_atc_in_tar_erm.append(np.load('../results/jobs/TARNet/TARNet_ATC_IN_' + str(i) + '.npy'))
#     jobs_atc_out_tar_erm.append(np.load('../results/jobs/TARNet/TARNet_ATC_OUT_' + str(i) + '.npy'))
#     jobs_rmse_in_tar_erm.append(np.load('../results/jobs/TARNet/TARNet_RMSE_IN_' + str(i) + '.npy'))
#     jobs_rmse_out_tar_erm.append(np.load('../results/jobs/TARNet/TARNet_RMSE_OUT_' + str(i) + '.npy'))

#     jobs_att_in_cfr_fish.append(np.load('../results/jobs/Matched_CFRNet/Matched_CFRNet_ATT_IN_' + str(i) + '.npy'))
#     jobs_att_out_cfr_fish.append(np.load('../results/jobs/Matched_CFRNet/Matched_CFRNet_ATT_OUT_' + str(i) + '.npy'))
#     jobs_atc_in_cfr_fish.append(np.load('../results/jobs/Matched_CFRNet/Matched_CFRNet_ATC_IN_' + str(i) + '.npy'))
#     jobs_atc_out_cfr_fish.append(np.load('../results/jobs/Matched_CFRNet/Matched_CFRNet_ATC_OUT_' + str(i) + '.npy'))
#     jobs_rmse_in_cfr_fish.append(np.load('../results/jobs/Matched_CFRNet/Matched_CFRNet_RMSE_IN_' + str(i) + '.npy'))
#     jobs_rmse_out_cfr_fish.append(np.load('../results/jobs/Matched_CFRNet/Matched_CFRNet_RMSE_OUT_' + str(i) + '.npy'))

#     jobs_att_in_tar_fish.append(np.load('../results/jobs/Matched_TARNet/Matched_TARNet_ATT_IN_' + str(i) + '.npy'))
#     jobs_att_out_tar_fish.append(np.load('../results/jobs/Matched_TARNet/Matched_TARNet_ATT_OUT_' + str(i) + '.npy'))
#     jobs_atc_in_tar_fish.append(np.load('../results/jobs/Matched_TARNet/Matched_TARNet_ATC_IN_' + str(i) + '.npy'))
#     jobs_atc_out_tar_fish.append(np.load('../results/jobs/Matched_TARNet/Matched_TARNet_ATC_OUT_' + str(i) + '.npy'))
#     jobs_rmse_in_tar_fish.append(np.load('../results/jobs/Matched_TARNet/Matched_TARNet_RMSE_IN_' + str(i) + '.npy'))
#     jobs_rmse_out_tar_fish.append(np.load('../results/jobs/Matched_TARNet/Matched_TARNet_RMSE_OUT_' + str(i) + '.npy'))


# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (16,16), sharex = False)

# ax1.plot(epochs, np.mean(jobs_att_in_cfr_erm, axis = 0), label = 'CFR-ERM')
# # ax1.plot(epochs, np.mean(jobs_att_in_tar_erm, axis = 0), label = 'TAR-ERM')
# ax1.plot(epochs, np.mean(jobs_att_in_cfr_fish, axis = 0), label = 'CFR-FISH')
# # ax1.plot(epochs, np.mean(jobs_att_in_tar_fish, axis = 0), label = 'TAR-FISH')
# ax1.legend(loc = 'upper right')
# ax1.title.set_text('In-Sample ATT Error')

# ax2.plot(epochs, np.mean(jobs_att_out_cfr_erm, axis = 0), label = 'CFR-ERM')
# # ax2.plot(epochs, np.mean(jobs_att_out_tar_erm, axis = 0), label = 'TAR-ERM')
# ax2.plot(epochs, np.mean(jobs_att_out_cfr_fish, axis = 0), label = 'CFR-FISH')
# # ax2.plot(epochs, np.mean(jobs_att_out_tar_fish, axis = 0), label = 'TAR-FISH')
# ax2.legend(loc = 'upper right')
# ax2.title.set_text('Outof-Sample ATT Error')

# ax3.plot(epochs, np.abs(np.mean(jobs_atc_in_cfr_erm, axis = 0)), label = 'CFR-ERM')
# # ax3.plot(epochs, np.abs(np.mean(jobs_atc_in_tar_erm, axis = 0)), label = 'TAR-ERM')
# ax3.plot(epochs, np.abs(np.mean(jobs_atc_in_cfr_fish, axis = 0)), label = 'CFR-FISH')
# # ax3.plot(epochs, np.abs(np.mean(jobs_atc_in_tar_fish, axis = 0)), label = 'TAR-FISH')
# ax3.legend(loc = 'upper right')
# ax3.title.set_text('In-Sample ATC Error')

# ax4.plot(epochs, np.abs(np.mean(jobs_atc_out_cfr_erm, axis = 0)), label = 'CFR-ERM')
# # ax4.plot(epochs, np.abs(np.mean(jobs_atc_out_tar_erm, axis = 0)), label = 'TAR-ERM')
# ax4.plot(epochs, np.abs(np.mean(jobs_atc_out_cfr_fish, axis = 0)), label = 'CFR-FISH')
# # ax4.plot(epochs, np.abs(np.mean(jobs_atc_out_tar_fish, axis = 0)), label = 'TAR-FISH')
# ax4.legend(loc = 'upper right')
# ax4.title.set_text('Outof-Sample ATC Error')

# plt.savefig('Jobs_faaltu')


alphas = [0, 10, 100, 1000, 10000]
epsilon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

ates_1 = [0.17211375, 0.11206351, 0.09924112, 0.10298735, 0.11339476]
uncertainty_ates_1 = [0.09150091, 0.084176175, 0.08518725, 0.09325687,  0.14525244]
pehe_1 = [1.8740424, 1.7386819, 1.6906167, 1.6880685, 1.6849766]
uncertainty_pehe_1 = [0.071685724,  0.10765513, 0.047431193, 0.046090085, 0.043699734]

ates_2 = [0.1453359,  0.10395002,  0.10131007, 0.10244811, 0.11138983]
uncertainty_ates_2 = [0.086784795, 0.074872, 0.0856606, 0.09168986, 0.14169955]
pehe_2 = [1.8591534, 1.726913, 1.6914967, 1.688645, 1.6836979]
uncertainty_pehe_2 = [0.069759466, 0.09455045, 0.046641152, 0.045786563, 0.043720473]

ates_3 = [0.14100975, 0.100988805, 0.10093135, 0.10285845, 0.11317648]
uncertainty_ates_3 = [0.08575837, 0.07392061, 0.08599278, 0.093849145,  0.14711322]
pehe_3 = [1.856921, 1.7216517, 1.6917369, 1.6884457, 1.6831461]
uncertainty_pehe_3 = [0.06995513, 0.092053376, 0.04684797, 0.046086673, 0.043785628]

ates_4 = [0.13916004, 0.100028366, 0.10119991, 0.10272547, 0.11163358]
uncertainty_ates_4 = [0.08601011, 0.072596915, 0.085769385, 0.093380645, 0.14439437]
pehe_4 = [1.8577327, 1.7214564, 1.6919724, 1.6901338, 1.6839702]
uncertainty_pehe_4 = [0.06910027, 0.0904277, 0.04664137, 0.04585441, 0.043714147]

ates_5 = [0.14062707, 0.09892722, 0.10128397, 0.10362613, 0.109927624]
uncertainty_ates_5 = [0.08602443, 0.0728795, 0.08610896, 0.092785016, 0.13684985]
pehe_5 = [1.8599213, 1.721391, 1.6919954, 1.6892684, 1.6842365]
uncertainty_pehe_5 = [0.06830693, 0.09044155, 0.046877444, 0.046269108, 0.043875627]

ates_6 = [0.13635665,0.09922298, 0.1004747, 0.10330915, 0.1105114]
uncertainty_ates_6 = [0.086410336, 0.07212447, 0.08628157, 0.094375856, 0.13894197]
pehe_6 = [1.8549031, 1.7196515, 1.6912246, 1.6892582, 1.683576]
uncertainty_pehe_6 = [0.06889731, 0.08816811, 0.046751168, 0.046020206, 0.043742333]

ates_7 = [0.136367, 0.09886611, 0.101644486, 0.10279118, 0.11045405]
uncertainty_ates_7 = [0.08580489,  0.0717405, 0.08577507, 0.09292251, 0.1364475]
pehe_7 = [1.8540198, 1.7171143, 1.6915071, 1.6881652, 1.6837254]
uncertainty_pehe_7 = [0.06883338, 0.08687723, 0.047095343, 0.046066508, 0.043753173]

ates_8 = [0.099376655, 0.099248655, 0.101118974, 0.10170446,  0.11195582]
uncertainty_ates_8 = [0.07432495, 0.07112495, 0.085842885, 0.09251791, 0.14846025]
pehe_8 = [1.7188551, 1.7188551, 1.6915274, 1.6885786, 1.6834478]
uncertainty_pehe_8 = [0.08658086, 0.08658086, 0.046930805, 0.045653764, 0.043493386]

ates_9 = [0.1357401, 0.09755647, 0.10123088, 0.10169852, 0.1107062]
uncertainty_ates_9 = [0.08541667, 0.06990197, 0.08625, 0.093077,  0.14303438]
pehe_9 = [1.8533297, 1.7162169, 1.6919148, 1.68933, 1.684259]
uncertainty_pehe_9 = [0.067922354, 0.08595713, 0.0464797, 0.04581729, 0.043904006]

ates_10 = [0.13583775, 0.0990029, 0.1019336, 0.10234645, 0.1110022]
uncertainty_ates_10 = [0.08570194, 0.07068435, 0.086832516, 0.09086565, 0.14701729]
pehe_10 = [1.8526565, 1.7182659, 1.6916621, 1.6873028, 1.6850771 ]
uncertainty_pehe_10 = [0.06867435, 0.08603605, 0.04660432, 0.04543149, 0.043726742]

fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize = (16,8), sharex = False)

ax1.set_xscale('symlog')
ax1.plot(alphas, np.array(ates_1), label = 'ϵ = 0.1', linewidth = 2.0, marker = 'o')
# ax1.fill_between(np.array(alphas),np.array(ates_1)-np.array(uncertainty_ates_1),np.array(ates_1)+np.array(uncertainty_ates_1), alpha = 0.1)
ax1.plot(alphas, np.array(ates_2), label = 'ϵ = 0.2', linewidth = 2.0, marker = 'o')
# ax1.fill_between(np.array(alphas),np.array(ates_2)-np.array(uncertainty_ates_2),np.array(ates_2)+np.array(uncertainty_ates_2), alpha = 0.1)
ax1.plot(alphas, np.array(ates_3), label = 'ϵ = 0.3', linewidth = 2.0, marker = 'o')
# ax1.fill_between(np.array(alphas),np.array(ates_3)-np.array(uncertainty_ates_3),np.array(ates_3)+np.array(uncertainty_ates_3), alpha = 0.1)
ax1.plot(alphas, np.array(ates_4), label = 'ϵ = 0.4', linewidth = 2.0, marker = 'o')
# ax1.fill_between(np.array(alphas),np.array(ates_4)-np.array(uncertainty_ates_4),np.array(ates_4)+np.array(uncertainty_ates_4), alpha = 0.1)
ax1.plot(alphas, np.array(ates_5), label = 'ϵ = 0.5', linewidth = 2.0, marker = 'o')
# ax1.fill_between(np.array(alphas),np.array(ates_5)-np.array(uncertainty_ates_5),np.array(ates_5)+np.array(uncertainty_ates_5), alpha = 0.1)
ax1.plot(alphas, np.array(ates_6), label = 'ϵ = 0.6', linewidth = 2.0, marker = 'o')
# ax1.fill_between(np.array(alphas),np.array(ates_6)-np.array(uncertainty_ates_6),np.array(ates_6)+np.array(uncertainty_ates_6), alpha = 0.1)
ax1.plot(alphas, np.array(ates_7), label = 'ϵ = 0.7', linewidth = 2.0, marker = 'o')
# ax1.fill_between(np.array(alphas),np.array(ates_7)-np.array(uncertainty_ates_7),np.array(ates_7)+np.array(uncertainty_ates_7), alpha = 0.1)
ax1.plot(alphas, np.array(ates_8), label = 'ϵ = 0.8', linewidth = 2.0, marker = 'o')
# ax1.fill_between(np.array(alphas),np.array(ates_8)-np.array(uncertainty_ates_8),np.array(ates_8)+np.array(uncertainty_ates_8), alpha = 0.1)
ax1.plot(alphas, np.array(ates_9), label = 'ϵ = 0.9', linewidth = 2.0, marker = 'o')
# ax1.fill_between(np.array(alphas),np.array(ates_9)-np.array(uncertainty_ates_9),np.array(ates_9)+np.array(uncertainty_ates_9), alpha = 0.1)
ax1.plot(alphas, np.array(ates_10), label = 'ϵ = 0.10', linewidth = 2.0, marker = 'o')
# ax1.fill_between(np.array(alphas),np.array(ates_10)-np.array(uncertainty_ates_10),np.array(ates_10)+np.array(uncertainty_ates_10), alpha = 0.1)

ax1.set_xlabel('α')
ax1.set_ylabel('Out-of-sample ATE Errors')
ax1.legend(loc = 'upper right')
ax1.title.set_text('Fish Update Hyperparameter, ϵ vs IPM Scaling Hyperparameter, α')

ax2.set_xscale('symlog')
ax2.plot(alphas, np.array(pehe_1), label = 'ϵ = 0.1', linewidth = 2.0, marker = 'o')
# ax2.fill_between(np.array(alphas),np.array(pehe_1)-np.array(uncertainty_pehe_1),np.array(pehe_1)+np.array(uncertainty_pehe_1), alpha = 0.1)
ax2.plot(alphas, np.array(pehe_2), label = 'ϵ = 0.2', linewidth = 2.0, marker = 'o')
# ax2.fill_between(np.array(alphas),np.array(pehe_2)-np.array(uncertainty_pehe_2),np.array(pehe_2)+np.array(uncertainty_pehe_2), alpha = 0.1)
ax2.plot(alphas, np.array(pehe_3), label = 'ϵ = 0.3', linewidth = 2.0, marker = 'o')
# ax2.fill_between(np.array(alphas),np.array(pehe_3)-np.array(uncertainty_pehe_3),np.array(pehe_3)+np.array(uncertainty_pehe_3), alpha = 0.1)
ax2.plot(alphas, np.array(pehe_4), label = 'ϵ = 0.4', linewidth = 2.0, marker = 'o')
# ax2.fill_between(np.array(alphas),np.array(pehe_4)-np.array(uncertainty_pehe_4),np.array(pehe_4)+np.array(uncertainty_pehe_4), alpha = 0.1)
ax2.plot(alphas, np.array(pehe_5), label = 'ϵ = 0.5', linewidth = 2.0, marker = 'o')
# ax2.fill_between(np.array(alphas),np.array(pehe_5)-np.array(uncertainty_pehe_5),np.array(pehe_5)+np.array(uncertainty_pehe_5), alpha = 0.1)
ax2.plot(alphas, np.array(pehe_6), label = 'ϵ = 0.6', linewidth = 2.0, marker = 'o')
# ax2.fill_between(np.array(alphas),np.array(pehe_6)-np.array(uncertainty_pehe_6),np.array(pehe_6)+np.array(uncertainty_pehe_6), alpha = 0.1)
ax2.plot(alphas, np.array(pehe_7), label = 'ϵ = 0.7', linewidth = 2.0, marker = 'o')
# ax2.fill_between(np.array(alphas),np.array(pehe_7)-np.array(uncertainty_pehe_7),np.array(pehe_7)+np.array(uncertainty_pehe_7), alpha = 0.1)
ax2.plot(alphas, np.array(pehe_8), label = 'ϵ = 0.8', linewidth = 2.0, marker = 'o')
# ax2.fill_between(np.array(alphas),np.array(pehe_8)-np.array(uncertainty_pehe_8),np.array(pehe_8)+np.array(uncertainty_pehe_8), alpha = 0.1)
ax2.plot(alphas, np.array(pehe_9), label = 'ϵ = 0.9', linewidth = 2.0, marker = 'o')
# ax2.fill_between(np.array(alphas),np.array(pehe_9)-np.array(uncertainty_pehe_9),np.array(pehe_9)+np.array(uncertainty_pehe_9), alpha = 0.1)
ax2.plot(alphas, np.array(pehe_10), label = 'ϵ = 0.10', linewidth = 2.0, marker = 'o')
# ax2.fill_between(np.array(alphas),np.array(pehe_10)-np.array(uncertainty_pehe_10),np.array(pehe_10)+np.array(uncertainty_pehe_10), alpha = 0.1)

ax2.set_xlabel('α')
ax2.set_ylabel('Out-of-sample PEHE Errors')
ax2.legend(loc = 'upper right')

plt.savefig('try')
