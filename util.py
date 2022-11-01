# util functions
import numpy as np 
import pandas as pd
from collections import Counter
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, mean_squared_error,mean_absolute_error)
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier as RF
import joblib
import datetime
import random
from config import *

from plot_util import plot_barh,plot_scatter

#===========================================log utils==========================================================================
def init_log(save_path, mode='w'):
    import logging

    parent_dir = os.path.dirname(save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    logger = logging.getLogger()  # 不加名称设置root logger
    level = logging.DEBUG
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    logger.setLevel(level)

    # 写入文件
    fh = logging.FileHandler(save_path, mode=mode)
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger,ch,fh



# random_seed = 0
#===========================================device inf=========================================================================
def get_device_name_from_idx(device_index):
    device_idx_name_dict = {}
    device_list = pd.read_csv(device_list_file)
    for name,idx,type in zip(device_list['device_name'],device_list['index'],device_list['type']):
        device_idx_name_dict[idx] = name
    
    return device_idx_name_dict[device_index]

#===========================================feature extraction=======================================================================
bin_split_port = [0,1024,49152,65535]

''' fill counter will *values, src_flag stands for src is iot devices
{'start_time':-1,'in_size':[],'out_size':[],'size':[],'ip_dst':set(),'ipv4':0,'ipv6':0,'tcp':0,'udp':0,'tcp_local_port':[],'tcp_remote_port':[],
    'udp_local_port':[],'udp_remote_port':[],'tcp_window_size':[],'domain':[],'tls':0}'''
def fill_counter(counter,*values,src_flag=True):
    values_ = []
    for item in values:
        values_.append(item)
    [size,ip_src,ip_dst,ip_proto,tcp_srcport,tcp_dstport,tcp_window_size,udp_srcport,udp_dstport,dns,dns_name,dns_response,tls_type] = values_
    # size, only record tcp, udp size
    counter['size'].append(int(size))
    if src_flag == True:
        counter['out_size'].append(int(size))
        if ip_dst != '':
            counter['ip_dst'].append(ip_dst)
    else:
        counter['in_size'].append(int(size))
        if ip_src != '':
            counter['ip_dst'].append(ip_src)
    # ip_proto
    if ip_proto!='':
        ip_proto_now = ip_proto.split(',')[0]
        if ip_proto_now == '6':
            counter['ipv4'] += 1
        elif ip_proto_now == '17':
            counter['ipv6'] += 1
    # tcp_port
    if tcp_srcport != '':
        tcp_srcport = int(float(tcp_srcport.split(',')[0]))
        tcp_dstport = int(float(tcp_dstport.split(',')[0]))
        if src_flag == True:
            counter['tcp_local_port'].append(tcp_srcport)
            counter['tcp_remote_port'].append(tcp_dstport)
        else:
            counter['tcp_local_port'].append(tcp_dstport)
            counter['tcp_remote_port'].append(tcp_srcport)
        counter['tcp'] += 1
    # udp_port
    if udp_srcport != '':
        udp_srcport = int(float(udp_srcport.split(',')[0]))
        udp_dstport = int(float(udp_dstport.split(',')[0]))
        if src_flag == True:
            counter['udp_local_port'].append(udp_srcport)
            counter['udp_remote_port'].append(udp_dstport)
        else:
            counter['udp_local_port'].append(udp_dstport)
            counter['udp_remote_port'].append(udp_srcport)
        counter['udp'] += 1
    # tcp_window_size
    if tcp_window_size != '':
        tcp_window_size = int(float(tcp_window_size.split(',')[0]))
        counter['tcp_window_size'].append(tcp_window_size)
    # dns_name
    if dns_name != '':
        counter['domain'].append(dns_name)
    # tls
    if tls_type == '16' and src_flag == True:
        counter['tls'] += 1
    return counter

''' func: extract feature from counter, return feature
{'start_time':-1,'in_size':[],'out_size':[],'size':[],'ip_dst':[],'ipv4':0,'ipv6':0,'tcp':0,'udp':0,'tcp_local_port':[],'tcp_remote_port':[],
    'udp_local_port':[],'udp_remote_port':[],'tcp_window_size':[],'domain':[],'tls':0}
'''
def get_feature(counter):
    feature = []
    # size
    feature += get_size_features(counter['size'])
    feature += get_size_features(counter['in_size'])
    feature += get_size_features(counter['out_size'])
    # protocol
    feature.append(counter['ipv4'])
    feature.append(counter['ipv6'])
    feature.append(counter['tcp'])
    feature.append(counter['udp'])
    # port
    feature += get_port_features(counter['tcp_local_port'])
    feature += get_port_features(counter['tcp_remote_port'])
    feature += get_port_features(counter['udp_local_port'])
    feature += get_port_features(counter['udp_remote_port'])
    # tcp_window_size
    if len(counter['tcp_window_size']) == 0:
        feature += [-1]*4
    else:
        feature.append(len(np.unique(counter['tcp_window_size'])))
        feature.append(min(counter['tcp_window_size']))
        feature.append(max(counter['tcp_window_size']))
        feature.append(get_entropy(counter['tcp_window_size']))
    # dns
    if len(counter['domain']) == 0:
        feature += [-1]*3
    else:
        feature.append(len(set(counter['domain'])))
        feature.append(len(counter['domain']))
        feature.append(get_entropy(counter['domain']))
    # tls
    feature.append(counter['tls'])
    # ip_dst
    if len(counter['ip_dst']) != 0:
        feature.append(len(np.unique(counter['ip_dst'])))
        feature.append(get_entropy(counter['ip_dst']))
    else:
        feature += [0,-1]

    return feature
#===========================================feature extraction components===========================================================
def get_col_names():
    # size
    size_names = ['size','packet_num','size_kind','size_mean','size_var','size_entropy']
    col_names = []
    col_names += size_names
    col_names += ['in_' + ele for ele in size_names]
    col_names += ['out_' + ele for ele in size_names]
    #
    col_names += ['ipv4','ipv6','tcp','udp']
    # port
    port_names = ['port_bin1','port_bin2','port_bin3','port_kind','port_num','port_entropy']
    local_port_names = ['local_' + ele for ele in port_names]
    remote_port_names = ['remote_' + ele for ele in port_names]
    col_names += ['tcp_' + ele for ele in local_port_names]
    col_names += ['tcp_' + ele for ele in remote_port_names]
    col_names += ['udp_' + ele for ele in local_port_names]
    col_names += ['udp_' + ele for ele in remote_port_names]
    # tcp_window_size
    col_names += ['tcp_window_size_kind','tcp_window_size_min','tcp_window_size_max','tcp_window_size_entropy']
    # dns
    col_names += ['domain_kind','domain_count','domain_entropy']
    # tls
    col_names += ['tls_count']
    # ip_dst
    col_names += ['ip_dst_kind','ip_dst_entropy']
    return col_names

def get_entropy(elements):
    if len(elements) == 0:
        return -1
    # _ = Counter(elements)
    c_nums = list(Counter(elements).values())
    whole = len(elements)
    entropy = 0
    for c_num in c_nums:
        p = float(c_num)/whole
        entropy -= p*np.log(p)
    return round(entropy,3)
'''
return 'size','packet_num','size_kind','size_mean','size_var','size_entropy'
'''
def get_size_features(sizes):
    if len(sizes) == 0:
        return [-1]*6
    feature = []
    feature.append(sum(sizes))
    feature.append(len(sizes))
    feature.append(len(np.unique(sizes)))
    feature.append(round(np.mean(sizes),3))
    feature.append(round(np.var(sizes),3))
    feature.append(get_entropy(sizes))
    return feature

def get_bin_number(ports):
    if len(ports) == 0:
        return [0]*4
    feature = []
    for idx in range(len(bin_split_port)-1):
        tmp = [x for x in ports if (x>= bin_split_port[idx] and x<bin_split_port[idx+1])]
        feature.append(len(tmp))
    return feature

'''
return 0-500, 500-1023,1024-49151,49152-65535, port_kind,port_num,entropy
'''
def get_port_features(ports):
    if len(ports) == 0:
        return [-1]*6
    feature = []
    feature += get_bin_number(ports) # 3
    feature.append(len(np.unique(ports)))
    feature.append(len(ports))
    feature.append(get_entropy(ports))
    return feature
#================================================train=====================================================

# generate new_device_list using new_device_batch_num, new_device_number_range
def generate_new_device_list(all_labels_,device_idx_type_dict,idx_instance_num_dict):
    not_satisfy = True
    while(not_satisfy):
        all_labels = set(all_labels_)
        new_device_list = []
        for i in range(new_device_batch_num):
            device_num = random.choice(new_device_number_range)
            new_devices_batch = random.sample(all_labels,device_num)
            new_device_list.append(list(new_devices_batch))
            all_labels = all_labels.difference(new_devices_batch)
        # check left. 1. left device should contain iot and non-iot; 2. instance of non-iot should reach 1000
        flag_iot = False
        flag_niot = False
        niot_num = 0
        for label in all_labels:
            if device_idx_type_dict[label] == 'iot':
                flag_iot = True
            if device_idx_type_dict[label] == 'non-iot':
                flag_niot = True
                niot_num += idx_instance_num_dict[label]
        if (flag_iot == True and flag_niot == True and niot_num>1000):
            not_satisfy = False
    return new_device_list

def store_data_into_joblib_file(data, name, path):
    f = open(path + '/{}'.format(name),'wb')
    joblib.dump(data,f)
    f.close()

def load_joblib_from_file(path):
    f = open(path,'rb')
    data = joblib.load(f)
    f.close()
    return data


def load_rf_model_i(i,path):
    f = open(path + '/rf_{}'.format(i),'rb')
    rf_model = joblib.load(f)
    f.close()
    return rf_model

'''
balance data to equal number, return resampled x and y, there are only 2 category here, y[:,2] is the label
'''
def train_dataset_balance(x,y):
    data = np.concatenate((y,x),axis=1)
    num_max = max([sum(y[:,3]==0),sum(y[:,3]==1)])
    for idx in np.unique(y[:,3]):
        num_c = sum(y[:,3]==idx)
        if num_c < num_max:
            # data_x = 
            data_c = resample(data[data[:,3]==idx], replace=True, n_samples=num_max-num_c, random_state=random_seed)
            data = np.concatenate((data,data_c),axis=0)
            print('resample {} other data of index {}'.format(num_max-num_c,idx))

    return data[:,4:], data[:,0:4]

def get_last_model_idx(path):
    model_idx = '0'
    if os.path.getsize(path + 'model_name.txt') != 0:
        f = open(path + 'model_name.txt','r')
        lines = f.readlines()
        model_idx = lines[-1].rstrip('\n').split('_')[-1]
    return int(float(model_idx))
'''
store rf model and add model name
'''
def save_model_and_name(path,model,first=False,logger=None):
    if first == True:
        with open(path+'model_name.txt','a') as f:
            f.seek(0)
            f.truncate()
            f.close()
        if os.path.exists(path+'feature_importance.csv'):
            os.remove(path+'feature_importance.csv')

    model_idx = get_last_model_idx(path) + 1
    model_name = 'rf_{}'.format(model_idx)
    f = open(path + 'model_name.txt','a')
    f.write(model_name + '\n')
    f.close()
    g = open(path + model_name,'wb')
    joblib.dump(model,g)
    g.close()
    logger.info('model {} has been saved'.format(model_name))

'''
compute feature importance from model and store it in path + 'feature_importance.csv'
'''
def compute_store_feature_importance(path,model,whole_column_names):
    model_idx = get_last_model_idx(path)
    feature_importance = model.feature_importances_
    column_names = ['model_idx']
    # column_feature_names = list(range(len(feature_importance)))
    # column_feature_names = [str(i) for i in column_feature_names]
    # column_names += column_feature_names
    column_names += list(whole_column_names[4:])
    values = [model_idx]
    values += list(feature_importance)
    values = pd.DataFrame(np.array(values).reshape(1,-1))
    values.columns = column_names
    # values = values.reshape(-1,1)
    if os.path.exists(path + 'feature_importance.csv'):
        values_before = pd.read_csv(path+'feature_importance.csv')
        values = pd.concat([values_before,values],axis = 0,ignore_index=True)

    values.to_csv(path+'feature_importance.csv',header = column_names,index=False)

def get_path(tree,X):
    nodes = []
    features = []
    node = 0
    while(tree.children_right[node]!=-1):
        nodes.append(node)
        if X[0,tree.feature[node]] <= tree.threshold[node]:
            node = tree.children_left[node]
        else:
            node = tree.children_right[node]
    return nodes

# reture gini increase and the corresponding split feature
def get_GINI_increase(tree,node):
    if tree.children_left[node] == -1:
        print('the node should not be leaf node, but the internal node')
        return
    weighted_n_node_samples = tree.weighted_n_node_samples
    N = weighted_n_node_samples[0]
    N_t = weighted_n_node_samples[node]
    left = tree.children_left[node]
    right = tree.children_right[node]
    impurity = tree.impurity
    result = N_t*impurity[node] - weighted_n_node_samples[left]*impurity[left] - weighted_n_node_samples[right]*impurity[right]
    result = result/N
    # features = tree.feature
    # print(np.unique(features))
    return result,tree.feature[node]

def compute_impurity_from_value(value):
    res = 1
    whole = sum(value)
    for v in value:
        res -= (v*v/(whole*whole))
    return res

def get_GINI_increase_from_values(values,idx,left_idx,right_idx):
    N = np.sum(values[0][0])
    N_t = np.sum(values[idx][0])
    N_t_l = np.sum(values[left_idx][0])
    N_t_r = np.sum(values[right_idx][0])
    impurity_n = compute_impurity_from_value(values[idx][0])
    impurity_n_l = compute_impurity_from_value(values[left_idx][0])
    impurity_n_r = compute_impurity_from_value(values[right_idx][0])
    result = N_t*impurity_n - N_t_l*impurity_n_l - N_t_r*impurity_n_r
    result = result/N
    return result

# compute feature importance for error sample, the features are X, the true label is y_true
def get_error_node_importance(tree,node,next_node,y_true,X):
    values_node = tree.value[next_node,0]
    # threshold = tree.threshold
    if tree.children_left[node] == -1:
        print('the node should not be leaf node, but the internal node')
        return
    weighted_n_node_samples = tree.weighted_n_node_samples
    N = weighted_n_node_samples[0]
    N_t = weighted_n_node_samples[node]
    left = tree.children_left[node]
    right = tree.children_right[node]
    impurity = tree.impurity
    # feature_used = tree.feature[node]
    fall_into_left = (left==next_node)
    child_proportion = values_node[int(y_true)]/np.sum(values_node)
    child_proportion = child_proportion/2
    if fall_into_left == True:
        result = N_t*impurity[node] - weighted_n_node_samples[left]*child_proportion - weighted_n_node_samples[right]*impurity[right]
    else:
        result = N_t*impurity[node] - weighted_n_node_samples[left]*impurity[left] - weighted_n_node_samples[right]*child_proportion
    result = result/N
    # features = tree.feature
    # print(np.unique(features))
    if result < 0:
        result = 0
    return result,tree.feature[node]

# compute feature importance for error sample through only proportion, the features are X, the true label is y_true
def get_error_node_importance_from_proportion(tree,node,next_node,y_true,X):
    if type(y_true) not in [np.float64]:
        print('something wrong')
    y_true = int(y_true)
    weighted_n_node_samples = tree.weighted_n_node_samples
    N = weighted_n_node_samples[0]
    N_t = weighted_n_node_samples[node]
    values_now = tree.value[node,0]
    values_next = tree.value[next_node,0]
    # threshold = tree.threshold
    if tree.children_left[node] == -1:
        print('the node should not be leaf node, but the internal node')
        return
    # compute the proportion change
    proportion_now = values_now[y_true]/sum(values_now)
    proportion_child = values_next[y_true]/sum(values_now)
    result = (proportion_now - proportion_child)*N_t/N
    if result < 0:
        result = 0
    return result,tree.feature[node],tree.threshold[node]


def get_interpretation_from_tree(tree,X,y_true):
    X = np.float32(X)
    X = X.reshape(1,-1)
    decision_path = tree.decision_path(X).indices
    # compute_path = get_path(tree,X) # is same as the decision_path, except no leaf node
    feature_important = [0] * X.shape[1]
    feature_threshold = {}
    for idx,node in enumerate(decision_path[:-1]): # the last if the leaf, has no feature
        if get_error_node_importance_type == 1:
            gini_increase, feature = get_GINI_increase(tree,node)
        elif get_error_node_importance_type == 2:
            gini_increase, feature = get_error_node_importance(tree,node,decision_path[idx+1],y_true,X)
        else:
            gini_increase, feature, threshold = get_error_node_importance_from_proportion(tree,node,decision_path[idx+1],y_true,X)
        feature_important[feature] += gini_increase
        if feature not in feature_threshold.keys():
            feature_threshold[feature] = [threshold]
        else:
            feature_threshold[feature].append(threshold)
    return feature_important,feature_threshold

'''
interpretate the error prediction and stored as rf_idx_interpretation.csv, need to get last_model_index
for the wrong pred, get the decision path and compute the feature importance
对每一个instance都解释
'''
def error_interpretation(path, model, test_x, test_y_4, prediction,column_names):
    model_idx = get_last_model_idx(path)
    wrong = (test_y_4[:,3]!=prediction)
    
    test_x_wrong = test_x[wrong]
    test_y_4_wrong = test_y_4[wrong]
    if len(test_x_wrong) == 0:
        print('no error instance found')
        return
    pred_wrong = prediction[wrong]
    interpretation_all = []
    interpretation_threshold_all = {}
    for X,y_4,pred in zip(test_x_wrong,test_y_4_wrong,pred_wrong): # each sample
        # only one instances may be wrong
        # if len(X) != 44:
        #     print('wrong')
        result = []
        threshold_instance = {}
        for tree in model.estimators_: # each tree
            interpretation,threshold_tree = get_interpretation_from_tree(tree.tree_,X,y_4[3])
            result.append(interpretation)
            if len(threshold_instance) == 0:
                threshold_instance = threshold_tree
            else:
                for key,value in threshold_tree.items():
                    if key not in threshold_instance.keys():
                        threshold_instance[key] = value
                    else:
                        threshold_instance[key].extend(value)
        # mean of result
        if get_error_node_importance_type != 3:
            result_mean = np.mean(result,axis=0)
        else:
            cols_num = test_x_wrong.shape[1]
            result_sum = np.sum(result,axis=0)
            non_zero_count_col = [0] * cols_num
            for res in result:
                for j in range(cols_num):
                    if res[j] != 0:
                        non_zero_count_col[j] += 1
            # non_zero_count_col = [item+1 if item==0  else item for item in non_zero_count_col] # to avoid divide zero
            result_mean = [0] * len(result_sum)
            for i in range(len(result_sum)):
                if result_sum[i] != 0:
                    result_mean[i] = result_sum[i]/non_zero_count_col[i]
                else:
                    result_mean[i] = 0
            # result_mean = result_sum/np.array(non_zero_count_col)

        result_X = result_mean/np.sum(result_mean)
        interpre_x = np.concatenate((y_4,result_X),axis=0)
        interpretation_all.append(interpre_x)
        # threshold
        interpretation_threshold_all[y_4[0]] = threshold_instance
    

    # store into csv
    if len(interpretation_all) > 0:
        interpretation_all = pd.DataFrame(np.array(interpretation_all))
        interpretation_all.to_csv(path+'interpretation_{}.csv'.format(model_idx),header=column_names,index=False)
        print('interpretation of error predictions for rf_{} has been saved'.format(model_idx))
        # store threshold
        with open(path + 'interpretation_threshold_{}'.format(model_idx),'wb') as g:
            joblib.dump(interpretation_threshold_all,g)
            g.close()

'''
judge whether to destroy the tree through comparing former_scores and new_scores, children_left is used to check whether it 
is a leaf node
'''
def destroy_tree_judge(tree,new_values):
    former_impurity = tree.impurity
    children_left = tree.children_left 
    children_right = tree.children_right
    former_values = tree.value
    node_count = tree.node_count
    leaf_node = []
    node = 0
    node_score_record = {} # node_id:x,leaf:True/False,former_score:x, current_score:x 
    new_mix_values = former_values + new_values
    for idx in range(node_count):
        if np.sum(new_values[idx,0]) > 0: # changed by new samples
            node_score_record[idx] = {}
            if children_left[idx] == -1: # leaf node
                node_score_record[idx]['leaf']=True
                former_pred_label = int(np.argmax(former_values[idx][0]))
                former_score = former_values[idx,0,former_pred_label]/np.sum(former_values[idx,0])
                current_score = new_mix_values[idx,0,former_pred_label]/np.sum(new_mix_values[idx,0])
                node_score_record[idx]['former_score'] = former_score
                node_score_record[idx]['current_score'] = current_score
            else: # internal node
                node_score_record[idx]['leaf']=False
                former_score,_ = get_GINI_increase(tree,idx)
                current_score = get_GINI_increase_from_values(new_mix_values,idx,children_left[idx],children_right[idx])
                node_score_record[idx]['former_score'] = former_score
                node_score_record[idx]['current_score'] = current_score
    # judge whether the tree should be destroyed according to alpha,beta and node_score_record
    for idx,item in node_score_record.items():
        if item['leaf'] == True:
            if item['current_score'] < beta:
                return True
        else:           
            if (item['former_score'] != 0) and (item['former_score'] - item['current_score'])/item['former_score'] > alpha:
                return True

    return False

# split data into iot and niot according to type
def split_iot_niot(data):
    data_iot = data[data['type']==1]
    data_niot = data[data['type']==0]
    return data_iot, data_niot

# mix old_data and new_data
def get_mixed_old_new_data(old_data,new_data):
    if len(new_data) == 0:
        return old_data
    old_half_num = int(len(old_data)/2)
    new_data_num = len(new_data)
    if new_data_num >= old_half_num:
        new_data_sampled = resample(new_data,replace=False,n_samples=old_half_num,random_state=random_seed)
        # old_data_indices_del = random.sample(list(range(len(old_data))),old_half_num)
        indexs = list(old_data.index)
        old_data_indices_del = random.sample(indexs,old_half_num)
        old_data = old_data.drop(old_data_indices_del)
        new_old_data = pd.concat([old_data,new_data_sampled],axis=0,ignore_index=True)
    else:
        # old_data_indices_del = random.sample(list(range(len(old_data))),new_data_num)
        indexs = list(old_data.index)
        old_data_indices_del = random.sample(indexs,old_half_num)
        old_data = old_data.drop(old_data_indices_del)
        new_old_data = pd.concat([old_data,new_data],axis=0,ignore_index=True)
    return new_old_data

'''
update model using new_data_X and test_y_4, return the updated model and store the mixed data
'''
def update_model(model, new_data_X, test_y_4, path, column_names, batch_idx, pred,logger):
    # determine which tree should be destroyed 
    destroy_tree_idx = []
    for idx,tree in enumerate(model.estimators_):
        tree = tree.tree_
        former_values = tree.value
        new_values = np.zeros(former_values.shape)
        
        for X,y_4 in zip(new_data_X,test_y_4):
            k = np.random.poisson(lam=1,size=1)
            if k > 0:
                # update new_values
                true_label = int(y_4[3])
                X = np.float32(X)
                X = X.reshape(1,-1)
                decision_path = tree.decision_path(X).indices
                for node in decision_path:
                    new_values[node,0,true_label] += int(k)
        # judge whether to destroy this tree
        if destroy_tree_judge(tree,new_values):
            destroy_tree_idx.append(idx)
    destroy_tree_num = len(destroy_tree_idx)
    logger.info('destroy {} trees'.format(destroy_tree_num))
    
    # destroy the tree
    if destroy_tree_num > 0:
        for index in reversed(destroy_tree_idx):
            model.estimators_.pop(index)
        # build new tree using the mixture of new_data_X and old_data.csv
        old_data = pd.read_csv(path + 'data_{}.csv'.format(batch_idx-1))
        old_data_iot, old_data_niot = split_iot_niot(old_data)
        new_data = pd.DataFrame(np.concatenate((test_y_4,new_data_X),axis=1))
        new_data.columns = list(old_data.columns)
        new_data_iot,new_data_niot = split_iot_niot(new_data)
        new_old_data_iot = get_mixed_old_new_data(old_data_iot,new_data_iot)
        new_old_data_niot = get_mixed_old_new_data(old_data_niot,new_data_niot)
        new_old_data = pd.concat([new_old_data_iot,new_old_data_niot],axis=0,ignore_index=True)      
        # store new_old_data
        new_old_data.to_csv(path + 'data_{}.csv'.format(batch_idx),header=list(old_data.columns),index=False)
        logger.info('the updated data_{}.csv has been saved'.format(batch_idx))
        # build new tree 
        new_old_data_X, new_old_data_y = train_dataset_balance(new_old_data.values[:,4:],new_old_data.values[:,:4])
        clf_new = RF(n_estimators=destroy_tree_num,bootstrap=True,random_state=random_seed,oob_score=True)
        clf_new.fit(new_old_data_X,new_old_data_y[:,3])
        model.estimators_.extend(clf_new.estimators_) 
        return model
    else:
        old_data = pd.read_csv(path + 'data_{}.csv'.format(batch_idx-1))
        # store old
        old_data.to_csv(path + 'data_{}.csv'.format(batch_idx),index=False)
        logger.info('the former old_data.csv has been saved')
        return model
    
#==================================observe pred_prob and path length=======================================
def get_prob_entropy(data):
    assert len(data)>0,'the len of data should be greater than 0'
    bin_count = []
    bin_split = [i/10 for i in range(10)]
    bin_split.append(1.1)
    data = np.array(data)
    for i in range(len(bin_split)-1):
        count_ = len(data[(data>=bin_split[i])&(data<bin_split[i+1])])
        bin_count.append(count_)
    entropy = 0
    assert len(data) == sum(bin_count),'The len of data should be equal to sum of bin_count'
    for i in bin_count:
        p = i*1.0/sum(bin_count)
        if p>0:
            entropy -= p*np.log(p)
    return round(entropy,4)


def show_pred_prob(model,X_all,y_3_all,pred,path,batch_idx,plot_flag=False):
    mean_prob = []
    var_prob = []
    # entropy_prob = []
    # mean_path_length = []
    # var_path_length = []
    # mean_prob_diff = []
    # var_prob_diff = []

    for X,y_3 in zip(X_all,y_3_all):
        prob_tmp = []
        # prob_diff_tmp = []
        # path_length_tmp = []

        for tree in model.estimators_:
            tree_ = tree.tree_
            X = np.float32(X)
            X = X.reshape(1,-1)
            # decision_path = tree_.decision_path(X).indices
            # path_length_tmp.append(len(decision_path))
            probs = tree.predict_proba(X) # iot prob
            prob = probs[0,1]
            # prob_diff_tmp.append(abs(probs[0,1]-probs[0,0]))
            prob_tmp.append(prob)

        # compute entropy of prob_tmp
        # entropy_prob.append(get_prob_entropy(prob_tmp))
        mean_prob.append(np.mean(prob_tmp))
        var_prob.append(np.var(prob_tmp))
        # mean_prob_diff.append(np.mean(prob_diff_tmp))
        # var_prob_diff.append(np.var(prob_diff_tmp))
        # mean_path_length.append(np.mean(path_length_tmp))
        # var_path_length.append(np.var(path_length_tmp))
    # plot
    path = path + '/prob_path_length_figs'
    if not os.path.exists(path):
        os.makedirs(path)
    right_bool = pred==y_3_all[:,3]
    label_set = np.unique(y_3_all[:,1])
    if plot_flag is True:
        from plot_util import plot_scatter_for_prob_variance
        plot_scatter_for_prob_variance(np.array(mean_prob),np.array(var_prob),y_3_all[:,1],right_bool,"The mean value of probability","The variance of probability",path+'/prob',batch_idx)
        
        
        # plot_scatter(np.array(mean_path_length),np.array(var_path_length),y_3_all[:,1],right_bool,'The mean value of path lengths','The variance of path length',path+'/path_length',batch_idx)
        # plot_scatter(np.array(mean_prob),np.array(entropy_prob),y_3_all[:,1],right_bool,'The mean value of probability of IoT','The entropy of probability of IoT',path+'/entropy',batch_idx)
        # plot_scatter(np.array(mean_prob_diff),np.array(var_prob),y_3_all[:,1],right_bool,'The mean value of absolute value of probability difference','The variance of probability of IoT',path+'/prob_diff',batch_idx)

    return var_prob

#=================================================evaluation===============================================

'''
analyze which device(index) is wrong, true is N*3 matrix(col:index,label,type), pred is one col.
if it is a two category classification, return accuracy, precison, recall, f1; else return the values of micro and macro.
also return the error rate for each device index
'''

def evaluate(true,pred,postfix='old',device_idx_name_dict=None, device_idx_type_dict=None,plot=True,path=None,logger=None):
    
    index_all = np.unique(true[:,1])
    index_error = {index:{'all':0,'error':0} for index in index_all}
    for index,true_i,pred_i in zip(true[:,1],true[:,3],pred):
        index_error[index]['all'] += 1
        if true_i != pred_i:
            index_error[index]['error'] += 1
   
    device_accuracy = {}
    device_types = [] # 1 stands for iot, 0 stands for non-iot
    device_labels = [] # labels of devices, used for plot
    for index,values in index_error.items():
        accuracy_c = round(1-float(index_error[index]['error'])/index_error[index]['all'],5)
        logger.info("index:{},all:{},wrong:{},accuracy:{}".format(index,index_error[index]['all'],index_error[index]['error'],accuracy_c))
        
        device_accuracy[index] = accuracy_c
        if plot is True:
            device_labels.append(device_idx_name_dict[index])
            device_types.append(device_idx_type_dict[index])
    if plot is True:
        plot_barh(list(device_accuracy.values()),path + '/{}_device_accuracy_{}.jpg'.format(postfix,classifier_type),device_labels,device_types)
    # compute accuracy, precision, recall, f1
    accuracy = round(accuracy_score(true[:,3],pred),5)
    precision = round(precision_score(true[:,3],pred),5)
    recall = round(recall_score(true[:,3],pred),5)
    f1 = round(f1_score(true[:,3],pred),5)
    logger.info("accuracy:{},precision:{},recall:{},f1:{}".format(accuracy,precision,recall,f1))
    
def evaluate_with_all_wrong_num(true,pred,postfix='old',device_idx_name_dict=None, device_idx_type_dict=None,plot=True,path=None,logger=None):
    
    index_all = np.unique(true[:,1])
    index_error = {index:{'all':0,'error':0} for index in index_all}
    for index,true_i,pred_i in zip(true[:,1],true[:,3],pred):
        index_error[index]['all'] += 1
        if true_i != pred_i:
            index_error[index]['error'] += 1
   
    device_accuracy = {}
    device_types = [] # 1 stands for iot, 0 stands for non-iot
    device_labels = [] # labels of devices, used for plot
    wrong_all_num = 0
    for index,values in index_error.items():
        accuracy_c = round(1-float(index_error[index]['error'])/index_error[index]['all'],5)
        logger.info("index:{},all:{},wrong:{},accuracy:{}".format(index,index_error[index]['all'],index_error[index]['error'],accuracy_c))
        wrong_all_num += index_error[index]['error']
        
        device_accuracy[index] = accuracy_c
        if plot is True:
            device_labels.append(device_idx_name_dict[index])
            device_types.append(device_idx_type_dict[index])
    if plot is True:
        plot_barh(list(device_accuracy.values()),path + '/{}_device_accuracy_{}.jpg'.format(postfix,classifier_type),device_labels,device_types)
    # compute accuracy, precision, recall, f1
    accuracy = round(accuracy_score(true[:,3],pred),5)
    precision = round(precision_score(true[:,3],pred),5)
    recall = round(recall_score(true[:,3],pred),5)
    f1 = round(f1_score(true[:,3],pred),5)
    logger.info("accuracy:{},precision:{},recall:{},f1:{}".format(accuracy,precision,recall,f1))
    return wrong_all_num


def compute_metrics(true,pred,logger,indication_sentense):
    accuracy = round(accuracy_score(true,pred),5)
    precision = round(precision_score(true,pred),5)
    recall = round(recall_score(true,pred),5)
    f1 = round(f1_score(true,pred),5)
    logger.info(indication_sentense)
    logger.info("accuracy:{},precision:{},recall:{},f1:{}".format(accuracy,precision,recall,f1))
    

def get_tp_tn_fp_fn(labels,prediction):
    # TP labels==1 and prediction==1
    bool_tp = (labels==1) & (prediction==1)
    tp = np.sum(bool_tp)
    bool_fp = (labels==0) & (prediction==1)
    tp = np.sum(bool_tp)
    bool_tp = (labels==1) & (prediction==1)
    tp = np.sum(bool_tp)
    bool_tp = (labels==1) & (prediction==1)
    tp = np.sum(bool_tp)



def get_accuracy_precision_recall(y,pred):
    accuracy = round(accuracy_score(y,pred),5)
    precision = round(precision_score(y,pred),5)
    recall = round(recall_score(y,pred),5)
    f1 = round(f1_score(y,pred),5)
    return accuracy, precision, recall, f1

#=========================================others===========================================
def get_datetime_str() :
    #datetime returns in the format: YYYY-MM-DD HH:MM:SS.millis but ':' is not supported for Windows' naming convention.
    datetime_str = str(datetime.datetime.now().strftime("%Y.%m.%d.-%H.%M.%S")) # .strftime("%Y-%m-%d-%H-%M-%S"))
    return datetime_str

#=========================================read file=========================================
# read f1 from result.txt file
def get_f1_and_accuracy_from_file(path):
    f1 = []
    accuracy = []
    f = open(path + '/result.log','r')
    line = f.readline().rstrip('\n')
    while(line):
        if line.find('f1:') != -1:
            idx_1 = line.index('f1:')
            value = line[idx_1+3:]
            value = float(value)
            f1.append(value)
            idx_acc_start = line.find('- accuracy:')
            idx_acc_end = int(line.find(','))
            accuracy_i = float(line[idx_acc_start+11:idx_acc_end])
            accuracy.append(accuracy_i)
        line = f.readline().rstrip('\n')
        if line.find('overall metrics') != -1:
            break
    f.close()
    assert len(f1)==9,'length of f1 list must be 9'
    assert len(accuracy)==9,'length of accuracy list must be 9'
    return f1,accuracy

def filtering_instances_according_labels(X,y_3,pred,labels):
    X_final = None
    y_3_final = None
    pred_final = None
    for label in np.unique(labels):
        label_indicator = y_3[:,1] == label
        X_c = X[label_indicator]
        y_c = y_3[label_indicator]
        pred_c = pred[label_indicator]
        if X_c.ndim != 2:
            print('wrong')
            X_c = X_c.reshape(1,-1)
            y_c = y_c.reshape(1,-1)
            pred_c = pred_c.reshape(1,-1)
        if X_final is None:
            X_final = X_c
            y_3_final = y_c
            pred_final = pred_c
        else:
            X_final = np.concatenate([X_final,X_c],axis=0)
            y_3_final = np.concatenate([y_3_final,y_c],axis=0)
            pred_final = np.concatenate([pred_final,pred_c],axis=0)
        # pred_final = pred_final.reshape(-1,1)
    return X_final,y_3_final,pred_final