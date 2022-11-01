
# for a certain batch, compare two models, the models idx should before the batch idx

import os 
import sys 
import numpy as np 
import pandas as pd 
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
# import config
from config import disposed_dir,get_error_node_importance_type
from util import load_joblib_from_file,load_rf_model_i,get_interpretation_from_tree
from plot_util import plot_bar_2_1,plot_thresholds,plot_thresholds_mean

NUM_TOP_FEATURES = 5
new_device_list = [[44, 45, 32, 6],[38, 5, 7],[18, 11],[10, 37, 39, 28, 1, 20],[30, 42, 47, 31, 27, 4],[25, 49, 22, 26, 14, 34],[16, 0],[19, 24]]
time_dir = disposed_dir + '/1621331189'

class ModelCompareForBatch(object):
    def __init__(self,model_idx1,model_idx2,batch_idx,device_idx,path):
        self.model_idx1 = model_idx1
        self.model_idx2 = model_idx2
        self.batch_idx = batch_idx
        self.device_idx = device_idx
        self.path = path

    def gather_delta_wrong_instances(self): # gather the increased wrong instance of self.model_idx2
        self.model_before = load_rf_model_i(self.model_idx1,time_dir)
        self.model_after = load_rf_model_i(self.model_idx2,time_dir)
        self.device_data = self.load_device_data()
        self.device_data = self.device_data.values
        y_4_all = self.device_data[:,0:4]
        X = self.device_data[:,4:]
        # filter the increased wrong instance
        pred1 = self.model_before.predict(X)
        pred2 = self.model_after.predict(X)
        wrong1 = y_4_all[:,3] != pred1
        wrong2 = y_4_all[:,3] != pred2
        delta_wrong_indicator = (wrong1==False) & (wrong2==True) # the wrong instances of model_after but not model_before
        self.delta_wrong_X = X[delta_wrong_indicator]
        self.delta_wrong_y_4 = y_4_all[delta_wrong_indicator]
        assert self.delta_wrong_X.shape[0]==self.delta_wrong_y_4.shape[0], "The num of records of self.delta_wrong_X and self.delta_wrong_y_4 should be the same"

    def find_error_imortance_features(self):
        self.interpretation_all_1, self.interpretation_threshold_1 = self.get_error_interpretation_data(self.model_after,self.model_idx2)
        self.interpretation_all_2, self.interpretation_threshold_2 = self.get_error_interpretation_data(self.model_before,self.model_idx1)
        
    def get_error_interpretation_data(self,model,model_idx):
        interpretation_all = []
        interpretation_threshold_all = {} # a dict, which record instance_idx: threshold
        for X,y_4 in zip(self.delta_wrong_X,self.delta_wrong_y_4): # each sample
            result = []
            threshold_instance = {}
            # iterate over the whole random forest
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
                cols_num = self.delta_wrong_X.shape[1]
                result_sum = np.sum(result,axis=0)
                non_zero_count_col = [0] * cols_num
                for res in result:
                    for j in range(cols_num):
                        if res[j] != 0:
                            non_zero_count_col[j] += 1
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
        
        # # store into csv
        # if len(interpretation_all) > 0:
        #     interpretation_all = pd.DataFrame(np.array(interpretation_all))
        #     interpretation_all.to_csv(self.path+'/interpretation_rf_{}_{}.csv'.format(model_idx,self.device_idx),header=column_names,index=False)
        #     print('interpretation of error predictions for rf_{} has been saved'.format(model_idx))
        #     # store threshold
        #     with open(self.path + '/interpretation_threshold_rf_{}_{}'.format(model_idx,self.device_idx),'wb') as g:
        #         joblib.dump(interpretation_threshold_all,g)
        #         g.close()
        return interpretation_all, interpretation_threshold_all

    def compare_contribution_two_model(self): 
        # for each error instances, compare after important feature and before important feature, and their their thresholds
        for interpretation1,interpretation2,X in zip(self.interpretation_all_1,self.interpretation_all_2,self.delta_wrong_X):
            features_idx_used_in_top = []
            feature_value_selected = []
            instance_idx = interpretation2[0]
            threshold2 = self.interpretation_threshold_2[instance_idx]
            threshold1 = self.interpretation_threshold_1[instance_idx]
            threshold2_selected = []
            threshold1_selected = []
            # find top important features in interpretation2
            importance = interpretation2[4:] 
            rank = [index for index,value in sorted(list(enumerate(importance)),key=lambda x:x[1],reverse=True)]
            importance_selected = []
            importance_selected_before = [] # the importance of the model_before

            for i in range(NUM_TOP_FEATURES):
                index = rank[i]
                features_idx_used_in_top.append(index)
                importance_selected.append(importance[index])
                threshold1_selected.append(threshold1[index])
                threshold2_selected.append(threshold2[index])
                feature_value_selected.append(X[index])

            # find the corresponding value in interpretation1
            for idx in features_idx_used_in_top:
                importance_selected_before.append(interpretation2[4+idx])
            xticklabels = [str(item) for item in features_idx_used_in_top]
            legends = ['RF_{}'.format(self.model_idx1),'RF_{}'.format(self.model_idx2)]
            plot_bar_2_1(importance_selected_before,importance_selected,xticklabels,legends,'Feature index','Importance',self.path,'error_feature_importance_compare_{}.jpg'.format(instance_idx))
    
            # threshold plot
            # plot_thresholds(xticklabels,threshold1_selected,threshold2_selected,feature_value_selected,features_idx_used_in_top,self.device_idx,instance_idx,self.path)
            plot_thresholds_mean(xticklabels,threshold1_selected,threshold2_selected,feature_value_selected,features_idx_used_in_top,self.device_idx,instance_idx,self.path)

    def statistical_analysis(self):
        pass

    def run(self):
        self.gather_delta_wrong_instances() # gather the increased wrong instance of self.model_idx2
        self.find_error_imortance_features()
        self.compare_contribution_two_model()
        self.statistical_analysis()

    def load_device_data(self):
        data = pd.read_csv(disposed_dir + '/data_all_iot_niot_with_instance_idx.csv')
        device_indicator_flag = data['index'] == self.device_idx
        data_device = data[device_indicator_flag]
        self.columns = list(data.columns)
        idx_feature_dict = {}
        for idx,name in enumerate(self.columns[4:]):
            idx_feature_dict[idx] = name
        self.idx_feature_dict = idx_feature_dict
        return data_device


if __name__ == '__main__':
    load_params_flag = False
    if load_params_flag == True:
        data_ = load_joblib_from_file(disposed_dir + '/cur_parameters.bin')
        time_dir, new_device_list = data_[0],data_[1]
    # defined by yourself
    model_idx1 = 3 # before model
    model_idx2 = 4 # after model
    batch_idx = 8 # the bad behavior batch of data
    device_idx = 0 # the bad behavioral device

    # create dir
    path = time_dir + '/generalization_error_analysis'
    if not os.path.exists(path):
        os.makedirs(path)

    model_compare = ModelCompareForBatch(model_idx1,model_idx2,batch_idx,device_idx,path)
    model_compare.run()


