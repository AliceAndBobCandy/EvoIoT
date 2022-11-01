import os
import sys
# from util import generate_new_device_list

#=============================feature extraction==================================
time_flag = True # use time to split or packet number
time_dur = 20*60 # second
packet_dur = 20 # extract feature according to packet count
dataset_name = "autoRF" # 
cur_dir = os.getcwd().replace("\\","/")
sys.path.append(cur_dir)
output_dir = cur_dir + '/output'
log_dir = output_dir + '/logs'


if dataset_name == "autoRF":
    disposed_dir = output_dir + "/autoRF/time_{}".format(int(time_dur/60)) 
    device_list_file = cur_dir + '/feature_extraction/device_list_we_gather_include_lsh.csv'

else:
    pass    

if not os.path.exists(disposed_dir):
    os.makedirs(disposed_dir)

#==============================iot classification====================================
classifier_type = "rf" # rf: random_forest, dt: decision tree, mlp, cnn
n_estimators = 200

# parameters for auto generate new_device_list
new_device_list = "auto" # auto stands for generate new devices automatically
new_device_batch_num = 8 # batch number of new devices
new_device_number_range = list(range(2,7)) # device number range of each batch new devices
# other_new_devices_data = output_dir + "/autoRF/new_iot_niot_whole_idx.csv"
other_new_devices_data = None
train_ratio = 0.7
random_seed = 0
need_scaler = False
alpha = 0.7 # (gini_n-gini_n')/gini_n > alpha, then the tree should be destroyed
beta = 0.5 # if proportion of main_c in leaf node is less than beta, then destroy the tree
get_error_node_importance_type = 3 # 1 stands for entropy, 2 stands for entropy and proportion, 3 stands for proportion

gama = 0.3 # feature importance ratio of history value
feature_importance_threshold = 0.01
del_feature_ratios = [0.1,0.3,0.5] # del unimportant features

#=============================NN parameters==========================================
EPOCH = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.01

# #=============================feature importance=====================================
# importance_flag = True
# importance_common_methods = ['default','permutation']

#======================choose part of new devices for managers for verification======
show_pred_prob_flag = True # show mean,variance of prediction probs of random forest trees 
choose_part_instances_method_type = 1 # 1. cluster; 2. choose first budget devices with highest prediction variance; 3. choose top devices randomly;
# BUDGET = 20 # The most num of devices that the manager can dispose


# open source need to delete
error_interpretation_flag = False # 是否解释错误instance

#==============================plot control=============================================
plot_figures_for_a_group_flag = True # 画出一组内的图，如Fig. 6, Fig. 7, 9, 10, 11, 16
test_del_feature_and_plot_flag = False # 删除特征的测试以及画图