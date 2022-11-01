# using the features to classify iot devices, new_device_list is the index of new devices
from distutils.log import error
import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import scale,StandardScaler,RobustScaler,PowerTransformer,QuantileTransformer,MinMaxScaler
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier as DT
import joblib
import time
from config import *
from util import compute_metrics, evaluate, train_dataset_balance,save_model_and_name,compute_store_feature_importance,error_interpretation,update_model,evaluate_with_all_wrong_num
from util import generate_new_device_list,store_data_into_joblib_file,load_rf_model_i
from util import show_pred_prob
from util import filtering_instances_according_labels,init_log
from manager_interaction.choose_part_instances import ChoosePartInstances
import matplotlib.pyplot as plt
from util import get_device_name_from_idx,get_f1_and_accuracy_from_file,get_tp_tn_fp_fn
from plot_util import plot_prob,plot_importance_change,plot_final_importance,plot_bar_2,plot_bar,plot_prob2,plot_prob3
import argparse

# plot control


# some flag
feature_importance_flag = False # do not compute feature importance

font={
    'family':'Arial',
    'weight':'medium',
      'size':14
}
font_bar_text={
    'family':'Arial',
    'weight':'medium',
      'size':4
}

class Classifier(object):
    def __init__(self,disposed_time_dir,del_features = None,new_device_list_former = None,BUDGET=-1,logger=None,new_device_list=None):
        self.disposed_time_dir = disposed_time_dir
        self.del_features = del_features
        self.new_device_list_former = new_device_list_former
        self.budget = BUDGET
        self.logger = logger
        self.new_device_list = new_device_list
        self.whole_data = pd.read_csv(disposed_dir + '/data_all_iot_niot_with_instance_idx.csv'.format(dataset_name))   
        if del_features is not None:
            self.whole_data = self.whole_data.drop(del_features,axis=1)
        self.column_names = list(self.whole_data.columns) 
        self.old_data = None
        self.new_data = None
        self.device_idx_name_dict = {}
        self.device_idx_type_dict = {}
    
        device_list = pd.read_csv(device_list_file)
        for name,idx,type in zip(device_list['device_name'],device_list['index'],device_list['type']):
            self.device_idx_name_dict[idx] = name
            self.device_idx_type_dict[idx] = type
        # get idx_instance_num_dict
        self.idx_instance_num_dict = {}
        for idx in np.unique(self.whole_data['index']):
            self.idx_instance_num_dict[idx] = self.whole_data[self.whole_data['index']==idx].shape[0]
        # get feature_idx_map and idx_feature_map and store into file
        self.feature_idx_map = {}
        self.idx_feature_map = {}
        feature_idx = 0
        for col_name in self.column_names[4:]:
            self.feature_idx_map[col_name] = feature_idx
            self.idx_feature_map[feature_idx] = col_name
            feature_idx += 1
        f = open(self.disposed_time_dir + '/idx_feature_map','wb')
        joblib.dump(self.idx_feature_map,f)
        f.close()
        self.test_true_labels = [] # 计算总的metrics
        self.test_predict_labels = []
        self.test_time = 0

        
    def split_old_new_data(self):
        global new_device_list
        whole_data = self.whole_data.copy(deep=True)
        all_labels = np.unique(whole_data['index'])
        if self.new_device_list_former is not None: # compare without del_features
            self.new_device_list = self.new_device_list_former
        
        if self.new_device_list == "auto":
            self.new_device_list = generate_new_device_list(all_labels,self.device_idx_type_dict,self.idx_instance_num_dict)
        # store new_device_list to file
        store_data_into_joblib_file(self.new_device_list, 'new_device_list',self.disposed_time_dir)
        # seperate new devices data from tmc data
        new_device_list_flat = [x for item in self.new_device_list for x in item]
        for idx in new_device_list_flat:
            data_c = whole_data[whole_data['index']==idx]
            if len(data_c) !=0:
                if self.new_data is None:
                    self.new_data = data_c
                else:
                    self.new_data = pd.concat([self.new_data,data_c],axis=0)
                whole_data.drop(whole_data[whole_data['index']==idx].index,inplace=True) 
        self.old_data = whole_data
        # add other new devices data
        if other_new_devices_data != None:
            df_other = pd.read_csv(other_new_devices_data)
            self.new_data = pd.concat([self.new_data,df_other],axis=0)

    # old_data train and test
    def old_data_train_test(self):
        # scaler 
        if need_scaler == True:
            self.scaler_ins = MinMaxScaler()
            self.scaler_ins.fit(self.old_data.values[:,4:].astype('float64'))
            f = open(self.disposed_time_dir + '/scaler','wb')
            joblib.dump(self.scaler_ins,f)
            f.close()
        # split self.old_data to train_data, test_data
        old_data_np = self.old_data.values
        train_x,test_x,train_y_3,test_y_3 = sklearn_train_test_split(old_data_np[:,4:],old_data_np[:,0:4],
        test_size=1-train_ratio,random_state=random_seed,shuffle=True)
        # store old data to old_data.csv
        old_data = pd.DataFrame(np.concatenate((train_y_3,train_x),axis=1))
        old_data.to_csv(self.disposed_time_dir + '/data_1.csv',header=self.column_names,index=False)
        self.logger.info('old_data.csv has been saved')
        train_x,train_y_3 = train_dataset_balance(train_x,train_y_3)
        if need_scaler == True:
            train_x = self.scaler_ins.transform(train_x)
            test_x = self.scaler_ins.transform(test_x)
        self.logger.info('dataset:{}'.format(dataset_name))
        self.logger.info('new device list:{}'.format(self.new_device_list))
        self.logger.info('classifier:{}'.format(classifier_type))
        
        if classifier_type == 'rf':           
            model = RF(n_estimators=n_estimators,bootstrap=True,random_state=random_seed,oob_score=True)
            model.fit(train_x,train_y_3[:,3])
            self.logger.info('oob score:{}'.format(model.oob_score_))
            self.model = model
            time1 = time.time()
            # predict test data of old devices
            prediction = model.predict(test_x)
            self.old_device_test_num = len(test_x)
            self.test_true_labels += list(test_y_3[:,3])
            self.test_predict_labels += list(prediction)
            evaluate(test_y_3,prediction,'old',self.device_idx_name_dict,self.device_idx_type_dict,False,self.disposed_time_dir,self.logger) # False 表示不画feature importance图
            time2 = time.time()
            self.test_time += (time2-time1)

            # # 通过通用方法得到feature importance并画图
            # if importance_flag == True and (self.del_features is None):
            #     from feature_importance.feature_importance import FeatureImportance
            #     feature_importance_ins = FeatureImportance(self.model,self.column_names,importance_common_methods,self.disposed_time_dir,
            #                                 train_x,train_y_3[:,3],test_x,test_y_3[:,3])
            #     feature_importance_ins.get_importance_common_methods()
            
            # save model and model name
            save_model_and_name(self.disposed_time_dir + '/',model,first=True,logger=self.logger)
            # compute and record feature importance
            if self.del_features is None:
                compute_store_feature_importance(self.disposed_time_dir + '/',model,self.column_names)
            # error record interpretation
            if self.del_features is None and error_interpretation_flag is True:
                error_interpretation(self.disposed_time_dir + '/', self.model, test_x, test_y_3, prediction,self.column_names)

        elif classifier_type == 'dt':
            model = DT(random_state=random_seed)
            model.fit(train_x,train_y_3[:,3])
            self.model = model
            prediction = model.predict(test_x)
            evaluate(test_y_3,prediction,'old',self.device_idx_name_dict,self.device_idx_type_dict,False,self.disposed_time_dir)

        elif classifier_type == 'mlp':
            from model.mlp import MLP
            self.model = MLP(train_x,train_y_3[:,3],test_x,test_y_3[:,3])          
            self.model.train()
            
        elif classifier_type == "cnn":
            from model.cnn import CNN
            self.model = CNN(train_x,train_y_3[:,3],test_x,test_y_3[:,3])          
            self.model.train()

    # new data test
    def new_data_test(self):
        # update model according to batch of new devices
        
        for idx,new_device_list_batch in enumerate(self.new_device_list):
            # a batch of new devices
            new_data_batch_pd = None
            for new_device_idx in new_device_list_batch:
                new_device_c_pd = self.new_data[self.new_data['index']==new_device_idx]
                if len(new_device_c_pd) != 0:
                    if new_data_batch_pd is None:
                        new_data_batch_pd = new_device_c_pd
                    else:
                        new_data_batch_pd = pd.concat([new_data_batch_pd,new_device_c_pd],axis=0)

            new_data_np = new_data_batch_pd.values
            test_y_3 = new_data_np[:,0:4]
            if need_scaler == True:
                new_data_X = self.scaler_ins.transform(new_data_np[:,4:])  
            else:
                new_data_X = new_data_np[:,4:]
            time0 = time.time()
            pred = self.model.predict(new_data_X)
            self.test_true_labels += list(test_y_3[:,3])
            self.test_predict_labels += list(pred)
            evaluate(test_y_3,pred,'new',self.device_idx_name_dict,self.device_idx_type_dict,False,self.disposed_time_dir,self.logger)
            if len(self.new_device_list) == 1:
                continue
            self.test_time += time.time()-time0
            # show_pred_prob_flag
            if show_pred_prob_flag is True:
                var_probs = show_pred_prob(self.model,new_data_X,test_y_3,pred,self.disposed_time_dir,idx+2,plot_flag=False)
            if choose_part_instances_method_type > 0 and (self.del_features is None):
                choose_part_instances = ChoosePartInstances(new_data_X,test_y_3,pred,self.disposed_time_dir,idx+2,var_probs,choose_part_instances_method_type,self.budget,plot_flag=plot_figures_for_a_group_flag,logger=self.logger)
                result_labels = choose_part_instances.result_labels
                # filter instances according to result_labels
                new_data_X, test_y_3, pred = filtering_instances_according_labels(new_data_X,test_y_3,pred,result_labels)
            
            # update model1
            self.model = update_model(self.model, new_data_X, test_y_3, self.disposed_time_dir + '/',self.column_names,idx+2,pred,self.logger)
            # save model and model name
            save_model_and_name(self.disposed_time_dir + '/',self.model,logger=self.logger)
            if self.del_features is None:
                # compute and record feature importance
                compute_store_feature_importance(self.disposed_time_dir + '/',self.model,self.column_names)
                # error record interpretation
                if error_interpretation_flag is True:
                    error_interpretation(self.disposed_time_dir + '/', self.model, new_data_X, test_y_3, pred, self.column_names)
        
        # compute the overall metrics   
        compute_metrics(self.test_true_labels,self.test_predict_labels,self.logger,'overall metrics (with old devices test):')
        compute_metrics(self.test_true_labels[self.old_device_test_num:],self.test_predict_labels[self.old_device_test_num:],self.logger,'overall metrics (only consider new devices):')
        

    # for each batch of new_devices data, evaluate the former rf model
    def generalization(self):
        
        self.logger.info("===================generalization test=========================")
        whole_data = self.whole_data 
        rf_models = []
        rf_1 = load_rf_model_i(1,self.disposed_time_dir)
        rf_models.append(rf_1)
        wrong_num_diff_all = []
        for idx,new_devices in enumerate(self.new_device_list):
            # a batch of new devices
            if idx > 0:
                wrong_num_ele = []
                self.logger.info('new_devices_batch:{}'.format(new_devices))
                data_batch = None
                for new_device in new_devices:
                    data_c = whole_data[whole_data['index']==new_device]
                    if data_batch is None:
                        data_batch = data_c
                    else:
                        data_batch = pd.concat([data_batch,data_c],axis=0,ignore_index=True)
                data_batch_np = data_batch.values
            
                # for idx i, test rf_1 to rf_i+1
                rf_model = load_rf_model_i(idx+1,self.disposed_time_dir)
                rf_models.append(rf_model)
                # test model in rf_models for data_batch
                for idx_,model in enumerate(rf_models):
                    self.logger.info('rf_:{}'.format(idx_+1))
                    pred = model.predict(data_batch_np[:,4:])
                    wrong_num = evaluate_with_all_wrong_num(data_batch_np[:,:4],pred,'new',self.device_idx_name_dict,self.device_idx_type_dict,False,self.disposed_time_dir,logger=self.logger)
                    wrong_num_ele.append(wrong_num)
                # compute wrong diff
                wrong_num_diff_i = [wrong_num_ele[i+1]-wrong_num_ele[i] for i in range(0,len(wrong_num_ele)-1)]
                wrong_num_diff_all.append(wrong_num_diff_i)

                self.logger.info('----------------------------------------------------------')
        self.logger.info(wrong_num_diff_all)

    #--------------------------------------------evaluation---------------------------------------------
   

    def feature_importance(self):
        feature_importance = []
        features = []
        all_feature_importance_recorder = []
        path_ = self.disposed_time_dir + '/{}'.format('feature_importance.csv')
        df = pd.read_csv(path_)
        column_names = list(df.columns)
        for idx,col_name in enumerate(column_names):
            feature_importance_recorder = []
            if idx > 0:
                data_col = df.loc[:,col_name]
                value = 0
                for data in data_col:
                    feature_importance_recorder.append(data)
                    if value == 0:
                        value = data
                    else:
                        value = value*gama + (1-gama)*data
                features.append(col_name)
                feature_importance.append(value)
                all_feature_importance_recorder.append(feature_importance_recorder)
        # normalize the feature importance
        feature_importance = feature_importance/np.sum(feature_importance)
        f = open(self.disposed_time_dir + '/feature_importance_normalized.txt','a')
        f.write('=====================================================\n')
        for feature,importance in zip(features,feature_importance):
            f.write('feature:{},importance:{}\n'.format(feature,importance))

        # 记录feature importance最大的五个特征及其值的变化
        feature_importance_, features_,idx = (list(x) for x in zip(*sorted(zip(feature_importance,features,list(range(len(features)))))))
        top_5_features = features_[-5:]
        top_5_idx = idx[-5:]
        used_feature_importance_recorder = []
        for idx in top_5_idx:
            used_feature_importance_recorder.append(all_feature_importance_recorder[idx])
        feature_importance_change = used_feature_importance_recorder[::-1]
        features_5 = top_5_features[::-1]
        f.write('feature change:\n')
        f.write(str(feature_importance_change) + '\n')
        f.write('features top 5\n')
        f.write(str(features_5))
        f.close()
      
        # plot the importance change of the top 5 features
        plot_importance_change(used_feature_importance_recorder[::-1],top_5_features[::-1],self.disposed_time_dir)
        # plot the final importance of the top 5 features
        feature_importance_top_5 = feature_importance_[-5:]
        plot_final_importance(feature_importance_top_5[::-1],top_5_features[::-1],self.disposed_time_dir)
        
        return feature_importance,features,all_feature_importance_recorder
        

    def evaluate_all(self):
        feature_importance,features,all_feature_importance_recorder = self.feature_importance()
        test_model_without_del_features(self.disposed_time_dir,feature_importance,features,all_feature_importance_recorder,self.new_device_list)

        # error interpretation
        # device_index = 44 # the device idx need to be interpretated
        # batch_index = 2
        # device_name = get_device_name_from_idx(device_index)
        # # error_interpretation(device_index,batch_index,device_name) # group interpretation
        # instance_idx = 25911 #25911
        # # error_interpretation_instance(instance_idx,batch_index,device_name) # instance interpretation
        
# test the model performance without del features
def test_model_without_del_features(disposed_time_dir,feature_importance,features,all_feature_importance_recorder,new_device_list_former):
    # find del features
    logger,ch,fh = init_log(disposed_time_dir + '/delete_features.log')
    del_features = []
    if del_feature_ratios is not None:
        for del_feature_ratio in del_feature_ratios:
            del_feature_num = int(len(features)*del_feature_ratio)
            feature_importance_, features_,idx = (list(x) for x in zip(*sorted(zip(feature_importance,features,list(range(len(features)))))))
            del_features = features_[:del_feature_num]
            logger.info('del_features:{}'.format(del_features))
            #-----------------------------------------------------------------------------------
            time_start = time.time()
            disposed_time_del_features_dir = disposed_time_dir + '/without_del_features_{}'.format(del_feature_ratio)
            if not os.path.exists(disposed_time_del_features_dir):
                os.makedirs(disposed_time_del_features_dir)
            logger.info('===================================================')
            logger.info('time_start:{}'.format(time_start))
            logger.info('alpha:{},beta:{}'.format(alpha,beta))
            logger.info('delete feature ratio:{}'.format(del_feature_ratio))
            
            classifier_ = Classifier(disposed_time_del_features_dir,del_features,new_device_list_former,BUDGET=20,logger=logger)
            classifier_.split_old_new_data()
            classifier_.old_data_train_test()
            classifier_.new_data_test()
            time_dur_all = time.time()-time_start
            with open(disposed_time_del_features_dir + '/result.txt','a') as g:
                g.write('time_all:{}\n'.format(time_dur_all))
                g.close()
            print('Done')

            # # 画删除特征前/后metrics变化
            # # classifier_.generalization()
            # former_f1,former_acc = get_f1_and_accuracy_from_file(disposed_time_dir)
            # after_f1,after_acc = get_f1_and_accuracy_from_file(disposed_time_del_features_dir)
            # labels = ['RF{}'.format(item+1) for item in range(9)]
            # plot_bar_2(former_f1,after_f1,labels,['Model before','Model after'],'F1 score (%)',disposed_time_del_features_dir,'f1_compare')
            # plot_bar_2(former_acc,after_acc,labels,['Model before','Model after'],'Accuracy (%)',disposed_time_del_features_dir,'acc_compare')
            # # plot_bar_2(former_f1,after_f1,labels,['Model before','Model after'],'F1 (%)',disposed_time_del_features_dir,'f1_compare.jpg')
            # # plot_bar_2(former_acc,after_acc,labels,['Model before','Model after'],u'正确率 (%)',disposed_time_del_features_dir,'acc_compare.jpg')
    else:
        for feature, importance in zip(features, feature_importance):
            if importance < feature_importance_threshold:
                del_features.append(feature)
    
    logger.removeHandler(ch)
    logger.removeHandler(fh)


def parse_str(a):
    result = []
    whole = 8
    a = a.replace(' ','')
    start_idx = 1
    while whole > 0:
        start = a.index('[',start_idx)
        end = a.index(']',start_idx)
        curr_str = a[start+1:end]
        list_now_ = curr_str.split(',')
        list_now = [int(item) for item in list_now_]
        result.append(list_now)
        start_idx = end+1
        whole -= 1
    return result

def main(alpha=0.7,beta=0.5,BUDGET=20,new_device_list='auto',new_dir=None):
    time_start = time.time()
    # disposed_time_dir = disposed_dir + '/budget_test/{}'.format(int(time_start))
    if new_dir is None:
        disposed_time_dir = disposed_dir + '/{}'.format(int(time_start))
    else:
        disposed_time_dir = new_dir + '/{}'.format(int(time_start))
    if not os.path.exists(disposed_time_dir):
        os.makedirs(disposed_time_dir)
    logger,ch,fh = init_log(disposed_time_dir + '/result.log')
    
    # print the variable parameters
    logger.info('time str:{}'.format(time_start))
    logger.info('alpha:{},beta:{},budget:{},new_device_list:{}'.format(alpha,beta,BUDGET,new_device_list))
    
    classifier = Classifier(disposed_time_dir,BUDGET=BUDGET,logger=logger,new_device_list=new_device_list) # fln to modify
    classifier.split_old_new_data()
    classifier.old_data_train_test()
    classifier.new_data_test()
    time_dur_all = time.time()-time_start
    logger.info('time_dur_all:{}'.format(time_dur_all))
    logger.info('test time:{}'.format(classifier.test_time))
    logger.info('train time:{}'.format(time_dur_all-classifier.test_time))
    
    classifier.generalization()
    logger.removeHandler(fh)
    logger.removeHandler(ch)

    logger.info('Done')
    
    
if __name__ == '__main__':
    new_dir = 'output/test'
    main(new_device_list=new_device_list,new_dir = new_dir,BUDGET=20)
    
    

