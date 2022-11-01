import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import math
import random
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import scale,StandardScaler,RobustScaler,PowerTransformer,QuantileTransformer,MinMaxScaler
from plot_util import plot_3D_scatter,plot_2D_scatter,plot_box,plot_bar_common
from plot_util import plot_line_chart

K = 20 # The maximum of cluster number

def get_optimal_k_from_slope(slopes):
    threshold = -100
    # find the first i that |slopes[i]|>100 & |slopes[i+1]|<100
    result = 0
    find_flag = False
    for i in range(len(slopes)-1):
        if slopes[i] < threshold and slopes[i+1] > threshold:
            result = i
            find_flag = True
            break
    if find_flag is True:
        return result + 2
    else:
        return 1

class ChoosePartInstances(object):
    def __init__(self,X,y_4_all,pred,path,batch_idx,var_probs,choose_part_instances_method_type,budget,plot_flag=False,logger=None):
        # the data should be first scaled
        self.X = X
        self.y_4_all = y_4_all
        self.pred = pred
        self.budget = budget
        self.path = path + '/manager_interaction'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.batch_idx = batch_idx
        self.var_probs = var_probs
        self.logger = logger
        self.plot_flag = plot_flag
        # if plot_flag is True:
        #     self.show_probs_inf_each_device()
        if choose_part_instances_method_type == 1:
            self.reduced_X = self.dimension_reduction(2)
            self.cluster_inf = self.cluster(5)
        # self.cluster(2)
        
        if choose_part_instances_method_type == 1:
            self.result_labels = self.choose_according_cluster_prob()
        elif choose_part_instances_method_type == 2: # choose top budget devices with highest prediction variance;
            self.result_labels = self.choose_highest_pred_variance()
        elif choose_part_instances_method_type == 3: # choose top budget devices with highest prediction variance;
            self.result_labels = self.choose_randomly()

    def get_scaled_data(self,data):
        scale_ins = MinMaxScaler()
        data = scale_ins.fit_transform(data.astype('float64'))
        return data
    
    # show the var probs of each device using box plot
    def show_probs_inf_each_device(self):
        labels = self.y_4_all[:,1]
        unique_labels = np.unique(labels)
        labels_str = [str(item) for item in unique_labels]
        data_devices = []
        for label in unique_labels:
            label_indicator = labels==label
            var_probs_np = np.array(self.var_probs)
            data_c = var_probs_np[label_indicator]
            data_devices.append(data_c)
        plot_box(data_devices,labels_str,'The index of devices','The variance of probabilites of IoT',self.path + '/prob_boxplot_{}.jpg'.format(self.batch_idx))
    
        

    def dimension_reduction(self,type=1):
        n_components = 2
        # pca
        if type == 1:
            if n_components == 2:
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(self.X)
                self.logger.info("dimension of reduced data:{}".format(len(reduced_data[0])))
                if self.plot_flag is True:
                    plot_2D_scatter(reduced_data,self.y_4_all[:,1],azim=48,elev=20,title="The data after dimension reduction",path=self.path+'/reduced_data_pca2_{}'.format(self.batch_idx))
                    
            elif n_components == 3:
                pca = PCA(n_components=3)
                reduced_data = pca.fit_transform(self.X)
                self.logger.info("dimension of reduced data:{}".format(len(reduced_data[0])))
                if self.plot_flag is True:
                    plot_3D_scatter(reduced_data,self.y_4_all[:,1],azim=48,elev=20,title="The data after dimension reduction",path=self.path+'/reduced_data_pca3_{}.png'.format(self.batch_idx))
            else:
                pca = PCA(n_components=n_components)
                reduced_data = pca.fit_transform(self.X)
                self.logger.info("dimension of reduced data:{}".format(len(reduced_data[0])))
        
        if type == 2:
            if n_components == 2:
                tsne = TSNE(n_components=2) #,learning_rate=100,perplexity=50
                reduced_data = tsne.fit_transform(self.X)
                if self.plot_flag is True:
                    plot_2D_scatter(reduced_data,self.y_4_all[:,1],azim=48,elev=20,title="The data after dimension reduction",path=self.path+'/reduced_data_tsne2_{}'.format(self.batch_idx))
                    
            elif n_components == 3:
                tsne = TSNE(n_components=3) #,learning_rate=100,perplexity=50
                reduced_data = tsne.fit_transform(self.X)
                if self.plot_flag is True:
                    plot_3D_scatter(reduced_data,self.y_4_all[:,1],azim=48,elev=20,title="The data after dimension reduction",path=self.path+'/reduced_data_tsne3_{}.png'.format(self.batch_idx))
            
        else:
            pass
        return reduced_data
    
    # return computed cluster centers
    def get_cluster_centers(self,labels):
        centers = []
        labels = np.array(labels)
        labels_unique = np.unique(labels)
        for label in labels_unique:
            label_indicator = labels == label
            data_c = self.reduced_X[label_indicator]
            center_c = np.mean(data_c,axis=0)
            centers.append(center_c)
        return centers


    # cluster the self.X_reduction and observe: 1.whether the instances of same label can form a cluster; 2. the prob variances of the same cluster 
    def cluster(self,type=2):
        X = range(2,K+1)
        
        if type == 1: # kmeans
            kmeans_cluster_num_method = 1
            if kmeans_cluster_num_method == 1: # elbow + kmeans
                SSE = []  # store the squared error          
                for k in range(2,K+1):
                    estimator = KMeans(n_clusters=k)  
                    estimator.fit(self.reduced_X)
                    # SSE
                    sse_value = sum(np.min(cdist(self.reduced_X,estimator.cluster_centers_,'euclidean'),axis=1))
                    # # SSE.append(estimator.inertia_)
                    SSE.append(sse_value)
                # plot figure
                if self.plot_flag is True:
                    plot_line_chart(X,SSE,xlabel='Number of clusters: k',ylabel='Minimal square error',label='SSE',path=self.path+'/sse_{}.'.format(self.batch_idx))
            elif kmeans_cluster_num_method == 1.1: # elbow contained in yellowbrick
                model = KMeans()
                visualizer = KElbowVisualizer(model,k=(1,K+1))
                visualizer.fit(self.reduced_X,timings=False)
                visualizer.show()
            
            elif kmeans_cluster_num_method == 2: # silhouette_score
                Scores = []
                for k in range(2,K+1):
                    estimator = KMeans(n_clusters=k)  
                    estimator.fit(self.reduced_X)
                    # silhouette_score
                    Scores.append(silhouette_score(self.reduced_X,estimator.labels_,metric='euclidean'))
                # plot figure
                if self.plot_flag is True:
                    plot_line_chart(X,Scores,xlabel='Number of clusters: k',ylabel='Silhouette score',label='Silhouette score',path=self.path+'/silhouette_score_{}.'.format(self.batch_idx))
                    # plot_line_chart(X,Scores,xlabel='聚类个数: k',ylabel='轮廓系数',label='Silhouette score',path=self.path+'/silhouette_score_{}.'.format(self.batch_idx))
            
            elif kmeans_cluster_num_method == 3: # BIC + kmeans
                BIC = []
                for k in range(2,K+1):
                    estimator = KMeans(n_clusters=k)
                    estimator.fit(self.reduced_X)
                    sse_value = math.sqrt(sum(np.min(cdist(self.reduced_X,estimator.cluster_centers_,'euclidean'),axis=1)))/self.reduced_X.shape[0]
                    bic = k*math.log(self.reduced_X.shape[0]) + self.reduced_X.shape[0] * math.log(math.sqrt(sse_value))
                    BIC.append(bic)
                if self.plot_flag is True:
                    plot_line_chart(X,BIC,xlabel='Number of clusters: k',ylabel='BIC',label='BIC',path=self.path+'/bic_{}.'.format(self.batch_idx))
                # delta_BIC = []
                # for i in range(len(BIC)-1):
                #     delta_BIC.append(BIC[i+1]-BIC[i])
                # return BIC.index(min(BIC)) + 2
                
            elif kmeans_cluster_num_method == 4: # AIC + Kmeans
                AIC = []
                for k in range(2,K+1):
                    estimator = KMeans(n_clusters=k)
                    estimator.fit(self.reduced_X)
                    sse_value = sum(np.min(cdist(self.reduced_X,estimator.cluster_centers_,'euclidean'),axis=1))/self.reduced_X.shape[0]
                    aic = k*2 + self.reduced_X.shape[0] * math.log(sse_value)
                    AIC.append(aic)
                if self.plot_flag is True:
                    plot_line_chart(X,AIC,xlabel='Number of clusters: k',ylabel='AIC',label='AIC',path=self.path+'/aic_{}.'.format(self.batch_idx))
                # delta_AIC = []
                # for i in range(len(AIC)-1):
                #     delta_AIC.append(AIC[i+1]-AIC[i])
                # return AIC.index(min(AIC)) + 2
        if type == 2: # optics
            from sklearn.cluster import OPTICS 
            clustering = OPTICS(min_samples=50).fit(self.reduced_X)
            cluster_labels = clustering.labels_
            num_cluster = len(set(cluster_labels))
            self.logger.info('optics num_cluster:{}'.format(num_cluster))
            if self.plot_flag is True:
                plot_2D_scatter(self.reduced_X,cluster_labels,azim=48,elev=20,title="Clusting the data after dimension reduction",path=self.path+'/clustered_data_optics_{}'.format(self.batch_idx))
        
        if type == 3: # dbscan
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(min_samples=8).fit(self.reduced_X)
            cluster_labels = clustering.labels_
            num_cluster = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            num_minus_1 = len(np.array(cluster_labels)==-1)
            self.logger.info('dbscan num_cluster:{},num_minus_1:{}'.format(num_cluster,num_minus_1))
            if self.plot_flag is True:
                plot_2D_scatter(self.reduced_X,cluster_labels,azim=48,elev=20,title="Clusting the data after dimension reduction",path=self.path+'/clustered_data_dbscan_{}'.format(self.batch_idx))

        if type == 4: # spectral clustering
            from sklearn.cluster import AgglomerativeClustering 
            clustering = AgglomerativeClustering(n_clusters=4).fit(self.reduced_X)
            cluster_labels = clustering.labels_
            num_cluster = len(set(cluster_labels))
            self.logger.info('AgglomerativeClustering num_cluster:{}'.format(num_cluster))
            if self.plot_flag is True:
                plot_2D_scatter(self.reduced_X,cluster_labels,azim=48,elev=20,title="Clusting the data after dimension reduction",path=self.path+'/clustered_data_Agglomerative_{}'.format(self.batch_idx))

        if type == 5: # hierachical clustering
            from sklearn.cluster import Birch 
            cluster_centers_set = []
            cluster_labels_set = []

            cluster_labels_for_fig = {}
            for k in range(2,K+1):
                clustering = Birch(n_clusters=k).fit(self.reduced_X)
                cluster_labels = clustering.predict(self.reduced_X)
                cluster_labels_set.append(cluster_labels)
                # cluster_centers_set.append(clustering.subcluster_centers_)
                cluster_centers_set.append(self.get_cluster_centers(cluster_labels))
                cluster_labels_for_fig[k] = cluster_labels
                        
            optimal_k, cluster_labels = self.cluster_num_observe(cluster_centers_set,cluster_labels_set,2)
            if self.reduced_X.shape[1] == 2:
                if self.plot_flag is True:
                    plot_2D_scatter(self.reduced_X,cluster_labels_for_fig[optimal_k],azim=48,elev=20,title="Clusting the data after dimension reduction",path=self.path+'/clustered_data_Birch_batch{}_({})'.format(self.batch_idx,optimal_k))

            self.cluster_labels = cluster_labels
            
            # if self.plot_flag is True:
            #     # obeserve the prob variance of different cluster
            #     cluster_prob_all = []
            #     cluster_unique_labels = np.unique(cluster_labels)
            #     cluster_labels_str = [str(item) for item in cluster_unique_labels]
            #     num_true_classes_in_cluster = []
            #     for label in cluster_unique_labels:
            #         label_indicator = cluster_labels==label
            #         num_class_c = len(np.unique(self.y_4_all[label_indicator,1]))
            #         num_true_classes_in_cluster.append(num_class_c)
            #         var_probs_np = np.array(self.var_probs)
            #         data_c = var_probs_np[label_indicator]
            #         cluster_prob_all.append(data_c)
            #     plot_box(cluster_prob_all,cluster_labels_str,'The index of the cluster','The variance of probabilites of IoT',self.path + '/cluster_prob_boxplot_{}.'.format(self.batch_idx))
            #     # observe the ground truth class num in each cluster, plot_bar
            #     xticklabels = ['']
            #     xticklabels += [str(item) for item in range(1,optimal_k+1)]
            #     plot_bar_common(num_true_classes_in_cluster,xticklabels,'The index of clusters','The number of true classes',self.path,'num_classes_of_cluster',self.batch_idx)
            

    def cluster_num_observe(self,cluster_centers_set,cluster_labels_set,type=2):
        X = range(2,K+1)
        # SSE elbow
        if type == 1:
            SSE = []  # store the squared error          
            for i in range(len(cluster_labels_set)):
                cluster_centers = cluster_centers_set[i]
                cluster_labels = cluster_labels_set[i]
                # SSE
                sse_value = sum(np.min(cdist(self.reduced_X,cluster_centers,'euclidean'),axis=1))
                # # SSE.append(estimator.inertia_)
                SSE.append(sse_value)
            # plot figure
            plot_line_chart(X,SSE,xlabel='Number of cluser: k',ylabel='Minimal square error',label='SSE',path=self.path+'/sse_{}'.format(self.batch_idx))
        
        # silhouette_score
        if type == 2:
            Scores = []
            for i in range(len(cluster_labels_set)):
                cluster_centers = cluster_centers_set[i]
                cluster_labels = cluster_labels_set[i]
                # silhouette_score
                Scores.append(silhouette_score(self.reduced_X,cluster_labels,metric='euclidean'))
            i_max = Scores.index(max(Scores))
            self.logger.info('optimal cluster num:{}'.format(i_max+2))
            if self.plot_flag is True:
                # plot_line_chart(X,Scores,xlabel='Number of clusters: k',ylabel='Silhouette score',label='Silhouette score',path=self.path+'/silhouette_score_{}'.format(self.batch_idx))
                plot_line_chart(X,Scores,xlabel='聚类个数: k',ylabel='轮廓系数',label='Silhouette score',path=self.path+'/silhouette_score_{}'.format(self.batch_idx))
            return i_max + 2, cluster_labels_set[i_max]


        if type == 3:
            # BIC + kmeans
            BIC = []
            for i in range(len(cluster_labels_set)):
                cluster_centers = cluster_centers_set[i]
                cluster_labels = cluster_labels_set[i]
                sse_value = sum(np.min(cdist(self.reduced_X,cluster_centers,'euclidean'),axis=1))/self.reduced_X.shape[0]
                bic = (i+2)*math.log(self.reduced_X.shape[0]) + self.reduced_X.shape[0] * math.log(math.sqrt(sse_value))
                BIC.append(bic)
            plot_line_chart(X,BIC,xlabel='Number of cluser: k',ylabel='BIC',label='BIC',path=self.path+'/bic_{}.'.format(self.batch_idx))
        
        # AIC + Kmeans
        if type == 4:
            AIC = []
            for i in range(len(cluster_labels_set)):
                cluster_centers = cluster_centers_set[i]
                cluster_labels = cluster_labels_set[i]
                sse_value = sum(np.min(cdist(self.reduced_X,cluster_centers,'euclidean'),axis=1))/self.reduced_X.shape[0]
                aic = (i+2)*2 + self.reduced_X.shape[0] * math.log(sse_value)
                AIC.append(aic)
            plot_line_chart(X,AIC,xlabel='Number of cluser: k',ylabel='AIC',label='AIC',path=self.path+'/aic_{}.'.format(self.batch_idx))
        

    def multi_armed_bandit_choose(self):
        pass

    # choose instance for each cluster, using self.cluster_labels,self.var_probs,self.y_4_all
    def choose_according_cluster_prob(self):
        instance_chosen_y_4 = []
        true_labels = []
        cluster_labels = self.cluster_labels
        var_probs = self.var_probs
        y_4_all = self.y_4_all
        cluster_labels_unique = np.unique(self.cluster_labels)
        for cluster_label in cluster_labels_unique[:self.budget]:
            cluster_indicator = cluster_labels == cluster_label
            var_probs = np.array(var_probs)
            var_probs_c = var_probs[cluster_indicator]
            y_4_c = y_4_all[cluster_indicator]
            i_max_c = np.argmax(var_probs_c)
            instance_chosen_y_4.append(y_4_c[i_max_c])
            instance_idx = y_4_c[i_max_c,0]
            true_labels.append(y_4_c[i_max_c,1])
            # del corresponding line for y_4_all, var_probs
            line_idx = np.where(y_4_all[:,0] == instance_idx)
            y_4_all = np.delete(y_4_all,line_idx,axis=0)
            var_probs = np.delete(var_probs,line_idx,axis=0)
            cluster_labels = np.delete(cluster_labels,line_idx,axis=0)
        left_chosen = self.budget - len(cluster_labels_unique)
        # choose left devices
        # alpha = 0.5 # the con
        # while left_chosen > 0:
        #     pass
        all_true_labels_unique = np.unique(self.y_4_all[:,1])
        missing_labels = list(set(all_true_labels_unique).difference(set(true_labels)))
        self.logger.info('batch:{}, choose {} devices, the labels are {}'.format(self.batch_idx,len(true_labels),true_labels))
        self.logger.info('get {}/{} devices of different kind, the missing devices are {}'.format(len(np.unique(true_labels)),len(all_true_labels_unique),missing_labels))
        
        return true_labels

    def choose_highest_pred_variance(self):
        instance_chosen_y_4 = []
        true_labels = []
        var_probs = self.var_probs
        y_4_all = self.y_4_all
        var_probs = np.array(var_probs)
        for i in range(self.budget):
            i_max = np.argmax(var_probs)
            instance_chosen_y_4.append(y_4_all[i_max])
            instance_idx = y_4_all[i_max,0]
            true_labels.append(y_4_all[i_max,1])
            # del corresponding line for y_4_all, var_probs
            line_idx = np.where(y_4_all[:,0] == instance_idx)
            y_4_all = np.delete(y_4_all,line_idx,axis=0)
            var_probs = np.delete(var_probs,line_idx,axis=0)

        all_true_labels_unique = np.unique(self.y_4_all[:,1])
        missing_labels = list(set(all_true_labels_unique).difference(set(true_labels)))
        print('choose {} devices, the labels are {}'.format(len(true_labels),true_labels))
        print('get {}/{} devices of different kind, the missing devices are {}'.format(len(np.unique(true_labels)),len(all_true_labels_unique),missing_labels))
        with open(self.path + '/result.txt','a') as f:
            f.write('batch:{}\n'.format(self.batch_idx))
            f.write('choose {} devices, the labels are {}\n'.format(len(true_labels),true_labels))
            f.write('get {}/{} devices of different kinds, the missing devices are {}\n'.format(len(true_labels),len(all_true_labels_unique),missing_labels))
            f.close()
        return true_labels

    def choose_randomly(self):
        instance_chosen_y_4 = []
        true_labels = []
        var_probs = self.var_probs
        y_4_all = self.y_4_all
        var_probs = np.array(var_probs)
        index_set = list(range(y_4_all.shape[0]))
        random.shuffle(index_set)
        assert len(var_probs)==len(y_4_all),'the length of var_probs and y_4_all is different'
        for i in range(self.budget):
            i_max = index_set[i]
            instance_chosen_y_4.append(y_4_all[i_max])
            instance_idx = y_4_all[i_max,0]
            true_labels.append(y_4_all[i_max,1])
            
        all_true_labels_unique = np.unique(self.y_4_all[:,1])
        missing_labels = list(set(all_true_labels_unique).difference(set(true_labels)))
        print('choose {} devices, the labels are {}'.format(len(true_labels),true_labels))
        print('get {}/{} devices of different kind, the missing devices are {}'.format(len(np.unique(true_labels)),len(all_true_labels_unique),missing_labels))
        with open(self.path + '/result.txt','a') as f:
            f.write('batch:{}\n'.format(self.batch_idx))
            f.write('choose {} devices, the labels are {}\n'.format(len(true_labels),true_labels))
            f.write('get {}/{} devices of different kinds, the missing devices are {}\n'.format(len(true_labels),len(all_true_labels_unique),missing_labels))
            f.close()
        return true_labels




    # def compute_RSS(self,cluster_labels,cluster_centers):
    #     k = len(np.unique(cluster_labels))
    #     rss = 0
    #     for c in range(k):
    #         label_c_indicator = cluster_labels == c
    #         data_c = self.reduced_X[label_c_indicator]
    #         center_c = cluster_centers[c]
    #         for data_i in data_c:
