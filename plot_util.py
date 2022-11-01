import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import colors as c
from matplotlib.font_manager import FontProperties
font_chinese = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=16)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"

SimSun = FontProperties(fname=r"/usr/share/matplotlib/mpl-data/fonts/ttf/simsun.ttf", size=24)
SimHei = FontProperties(fname=r"/usr/share/matplotlib/mpl-data/fonts/ttf/simhei.ttf",size=20)

font={
    # 'family':'Arial',
    'weight':'medium',
      'size':16
}
font_bar_text={
    # 'family':'Arial',
    'weight':'medium',
      'size':4
}
#=============================================================model result plot=================================================
'''
plot cm matrix, classes if the label of cm figure ticks
'''
def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title='cm_figure', cmap=plt.cm.Blues, save=False):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    uniqueLabel = np.unique(np.row_stack((y_true,y_pred)))
    new_class = []
    for i in range(0, uniqueLabel.shape[0]):
        new_class.append(classes[int(uniqueLabel[i])])
    classes = new_class
    del new_class
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 18)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im,fraction=0.046, pad=0.04, aspect=20)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_ylabel('True label',fontsize=18)
    ax.set_xlabel('Predicted label',fontsize=18)
    # if title is not None:
    #     ax.set_title(title, fontsize=20)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save:
        fig.savefig(title)
    return ax

colors = ['blue','red','orange','lawngreen','black']
feature_width_dict = {'size_kind':1,'ipv4':1000,'tcp_window_size_max':5000,'domain_kind':1,'domain_entropy':0.05,
'size_entropy':0.005,'in_size_entropy':0.005,'out_size_kind':1,'ip_dst_kind':1,'in_size_kind':1,'domain_count':1,
'tcp_remote_port_kind':0.5,'tcp_window_size_kind':1,'in_size':100,'in_packet_num':1000,'in_size_mean':10,
'tcp_remote_port_bin2':10,'tcp_remote_port_entropy':0.05,'tls_count':1,'out_packet_num':10,'udp_remote_port_kind':1,
'out_size_mean':1,'ip_dst_entropy':0.005,'size_mean':1,'tcp_local_port_num':1,'size':100,'packet_num':100,
'size_var':100,'in_size_var':100,'out_size_var':100,'out_size_entropy':0.005,'ipv6':1,'tcp':1,'udp':1,
'tcp_local_port_bin2':1,'tcp_local_port_bin1':1,'udp_local_port_bin2':1,'udp_local_port_bin1':1,'udp_local_port_bin3':1,
'udp_remote_port_num':1,'udp_remote_port_entropy':0.005,'udp_remote_port_bin3':10,'udp_remote_port_bin2':10,
'tcp_local_port_kind':1,'tcp_local_port_entropy':0.05,'udp_local_port_num':1,'tcp_remote_port_bin1':1}
device_list = {0:'Non-IoT',1:'IoT',2:'Other'}
feature_xlabel_dict = {'size_kind':'The number of packet length','ipv4':'IPv4 packet number',
'tcp_window_size_max':'The maximum value of TCP window size' ,'domain_kind':'The kinds of domain names',
'domain_entropy':'The entropy of domain names','size_entropy':'The entropy of packet length',
'in_size_entropy':'The entropy of packet length of incoming traffic',
'out_size_kind':'The distinct count of packet length of outgoing traffic','ip_dst_kind':'The distinct count of destination IP',
'in_size_kind':'The distinct count of packet length of incoming traffic','domain_count':'The count of domain names',
'tcp_remote_port_kind':'The distinct count of remote ports of TCP packets','tcp_window_size_kind':'The distinct count of TCP window size',
'in_size':'The bytes of incoming traffic','in_packet_num':'The packet number of incoming traffic',
'in_size_mean':'The mean value of packet lengths of incoming traffic',
'tcp_remote_port_bin2':'The number of remote ports fell into [500,1023]',
'tcp_remote_port_entropy':'The entropy of remote ports of TCP packets',
'tls_count':'The count of TLS handshake','out_packet_num':'The number of outgoing packets',
'udp_remote_port_kind':'The distinct count of remote ports of UDP packets','out_size_mean':'The mean value of packet lengths of outgoing traffic',
'ip_dst_entropy':'The entropy of destination IP','size_mean':'The mean value of packet lengths (bytes)',
'tcp_local_port_num':'The number of local ports of TCP','size':'The bytes of the traffic','packet_num':'The number of packets',
'size_var':'The variance of packet length','in_size_var':'The variance of packet length of incoming traffic',
'out_size_var':'The variance of packet length of outgoing traffic','out_size_entropy':'The entropy of packet length of outgoing traffic',
'ipv6':'IPv6 packet number','tcp':'TCP packet number','udp':'UDP packet number',
'tcp_local_port_bin2':'The number of TCP local ports fell into [500,1024)',
'tcp_local_port_bin1':'The number of TCP local ports fell into [0,500)',
'udp_local_port_bin2':'The number of UDP local ports fell into [500,1024)',
'udp_local_port_bin1':'The number of UDP local ports fell into [0,500)',
'udp_remote_port_bin2':'The number of UDP remote ports fell into [500,1024)',
'udp_remote_port_bin3':'The number of UDP remote ports fell into [1024,49152)',
'udp_remote_port_num':'The number of UDP remote ports','udp_remote_port_entropy':'The entropy of UDP remote ports',
'tcp_local_port_kind':'The distinct count of local ports of TCP packets',
'tcp_local_port_entropy':'The entropy of local ports of TCP packets',
'udp_local_port_num':'The number of local port of UDP packets',
'tcp_remote_port_bin1':'tcp_remote_port_bin1'}

# data is pandas format of data_i, index is a list(iot/non-iot), data_error is the data to be interpretated
# col_name stands for the feature, xlabel is the label of x axis
def plot_prob(data,data_error,col_name,xlabel,path,device_name=None):
    plt.clf()
    if device_name is not None:
        device_list[2] = device_name
    index = [0,1,2] # stands for iot, niot, error sample
    # change the type of data_error to 2
    data_error.loc[:,'type'] = 2
    data = pd.concat([data,data_error],axis = 0,ignore_index=True)
    plt.figure(figsize=(9, 6), dpi=400)
    for i,idx in enumerate(index):
        data_idx = data[data['type']==idx][col_name]
        data_value_count = data_idx.value_counts() # feature_value: count
        indicator = data_idx.unique() # unique feature value
#         print(indicator)
        indicator_ = []  
        prob_value = [data_value_count[i]*1.0/data_idx.shape[0] for i in indicator]
        indicator_ = indicator
#             print(sum(prob_value))
        if len(prob_value)!=0:
            plt.bar(indicator,prob_value,width=feature_width_dict[col_name],label=device_list[idx],color=colors[i],alpha=1)
#             plt.bar(indicator,prob_value,width=1,label=device_list[idx],color=colors[i])
    plt.ylabel('Probability',fontdict=font,fontsize=16) 
    plt.xlabel(feature_xlabel_dict[col_name],fontdict=font,fontsize=16)
    plt.xticks(fontsize=14,weight='semibold')
    plt.yticks(fontsize=14,weight='semibold')
    if col_name == 'tcp_window_size_max':
        plt.xlim(0,1000000)
    elif col_name == 'tcp_remote_port_kind':
        plt.xlim(0,1000)
    elif col_name == 'tcp_remote_port_bin2':
        plt.xlim(0,5000)
    elif col_name == 'udp_remote_port_bin2' or col_name=='udp_remote_port_bin3':
        plt.xlim(0,4000)
    elif col_name == 'udp_remote_port_num':
        plt.xlim(0,4000)
    elif col_name == 'in_size':
        plt.xlim(0,1000000)
    elif col_name == 'size':
        plt.xlim(0,1000000)
    elif col_name == 'size_var':
        plt.xlim(0,200000)
    elif col_name == 'packet_num':
        plt.xlim(0,100000)
    plt.tick_params(labelsize=14)
    if len(index)>1:
        plt.legend(fontsize=14,loc='best')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(0.4)
    ax.spines['left'].set_linewidth(0.4)
    ax.spines['right'].set_linewidth(0.4)
    ax.spines['top'].set_linewidth(0.4)
    plt.savefig(path + '/{}.jpg'.format(col_name))
    plt.close()

# plot error_instance (vertical line) and the data used for iot, non-iot classification
def plot_prob2(data,data_error,col_name,xlabel,path,device_name=None):
    if device_name is not None:
        device_list[2] = device_name
    index = [0,1] # stands for iot, niot, error sample
    # change the type of data_error to 2
    plt.figure(figsize=(9, 6), dpi=300)
    for i,idx in enumerate(index):
        data_idx = data[data['type']==idx][col_name]
        data_value_count = data_idx.value_counts() # feature_value: count
        indicator = data_idx.unique() # unique feature value
#         print(indicator)
        indicator_ = []  
        prob_value = [data_value_count[i]*1.0/data_idx.shape[0] for i in indicator]
        indicator_ = indicator
#             print(sum(prob_value))
        if len(prob_value)!=0:
            plt.bar(indicator,prob_value,width=feature_width_dict[col_name],label=device_list[idx],color=colors[i],alpha=1)
#             plt.bar(indicator,prob_value,width=1,label=device_list[idx],color=colors[i])
    plt.ylabel('Probability',fontdict=font,fontsize=16) 
    plt.xlabel(feature_xlabel_dict[col_name],fontdict=font,fontsize=16)
    plt.xticks(fontsize=14,weight='semibold')
    plt.yticks(fontsize=14,weight='semibold')
    if col_name == 'tcp_window_size_max':
        plt.xlim(0,1000000)
    elif col_name == 'tcp_remote_port_kind':
        plt.xlim(0,1000)
    elif col_name == 'tcp_remote_port_bin2':
        plt.xlim(0,5000)
    elif col_name == 'udp_remote_port_bin2' or col_name=='udp_remote_port_bin3':
        plt.xlim(-100,3000)
    elif col_name == 'udp_remote_port_num':
        plt.xlim(0,4000)
    elif col_name == 'in_size':
        plt.xlim(0,1000000)
    elif col_name == 'size':
        plt.xlim(0,1000000)
    elif col_name == 'size_var':
        plt.xlim(0,200000)
    elif col_name == 'packet_num':
        plt.xlim(0,100000)
    elif col_name == 'tls_count':
        plt.xlim(0,100)
    plt.tick_params(labelsize=14)
        # plt.ylim(0,0.5)
    # plot vertical line
    value_error = list(data_error.loc[:,col_name])[0]
    plt.axvline(x=value_error,ls='--',c='orange',ymin=0,ymax=0.3,label=device_name,alpha=1)


    plt.tick_params(labelsize=14)
    if len(index)>1:
        plt.legend(fontsize=14,loc='best')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(0.4)
    ax.spines['left'].set_linewidth(0.4)
    ax.spines['right'].set_linewidth(0.4)
    ax.spines['top'].set_linewidth(0.4)
    plt.savefig(path + '/{}_{}.jpg'.format(col_name,device_name))
    plt.close()

# plot error_instance (vertical line) and the data used for iot, non-iot classification, with threshold
def plot_prob3(data,data_error,col_name,xlabel,path,device_name=None,threshold_list=None):
    if device_name is not None:
        device_list[2] = device_name
    index = [0,1] # stands for iot, niot, error sample
    # change the type of data_error to 2
    plt.figure(figsize=(7, 5.5), dpi=300)
    for i,idx in enumerate(index):
        data_idx = data[data['type']==idx][col_name]
        data_value_count = data_idx.value_counts() # feature_value: count
        indicator = data_idx.unique() # unique feature value
#         print(indicator)
        indicator_ = []  
        prob_value = [data_value_count[i]*1.0/data_idx.shape[0] for i in indicator]
        indicator_ = indicator
#             print(sum(prob_value))
        if len(prob_value)!=0:
            plt.bar(indicator,prob_value,width=feature_width_dict[col_name],label=device_list[idx],color=colors[i],alpha=1)
#             plt.bar(indicator,prob_value,width=1,label=device_list[idx],color=colors[i])
    plt.ylabel('Probability',fontdict=font,fontsize=26,fontproperties=SimHei) 
    plt.xlabel(feature_xlabel_dict[col_name],fontdict=font,fontsize=26,fontproperties=SimHei)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    if col_name == 'tcp_window_size_max':
        plt.xlim(0,1000000)
    elif col_name == 'tcp_remote_port_kind':
        plt.xlim(0,1000)
    elif col_name == 'tcp_remote_port_bin2':
        plt.xlim(0,5000)
    elif col_name == 'udp_remote_port_bin2' or col_name=='udp_remote_port_bin3':
        plt.xlim(-100,3000)
    elif col_name == 'udp_remote_port_num':
        plt.xlim(0,4000)
    elif col_name == 'udp_local_port_bin2':
        plt.xlim(-100,500)
    elif col_name == 'udp_local_port_num':
        plt.xlim(-100,1500)
    elif col_name == 'in_size':
        plt.xlim(0,1000000)
    elif col_name == 'size':
        plt.xlim(0,4000000)
    elif col_name == 'size_var':
        plt.xlim(0,200000)
    elif col_name == 'packet_num':
        plt.xlim(0,100000)
    elif col_name == 'tls_count':
        plt.xlim(0,100)
    elif col_name == 'udp':
        plt.xlim(-100,5000)
    elif col_name == 'ipv6':
        plt.xlim(-100,2000)
    plt.tick_params(labelsize=24)
        # plt.ylim(0,0.5)
    # plot vertical line
    value_error = list(data_error.loc[:,col_name])[0]
    
    # plot threshold vertical line
    
    for idx,threshold in enumerate(threshold_list):
        if idx == 0:
            plt.axvline(x=threshold,ls='--',c='grey',ymin=0,ymax=0.3,alpha=1,lw=1,label='Value of split feature')
        else:
            plt.axvline(x=threshold,ls='--',c='grey',ymin=0,ymax=0.3,alpha=1,lw=1)
    plt.axvline(x=value_error,ls='--',c='darkorange',ymin=0,ymax=0.3,label=device_name[:-2],alpha=1,lw=2)
    plt.tick_params(labelsize=22)
    if len(index)>1:
        plt.legend(fontsize=16,loc='best',prop=SimHei)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(0.4)
    ax.spines['left'].set_linewidth(0.4)
    ax.spines['right'].set_linewidth(0.4)
    ax.spines['top'].set_linewidth(0.4)
    if col_name == 'tcp_local_port_kind':
        plt.xlim(0,300)
    if col_name == 'tcp_remote_port_kind':
        plt.xlim(0,150)
    plt.tight_layout()
    plt.savefig(path + '/{}_{}.png'.format(col_name,device_name))
    plt.savefig(path + '/{}_{}.pdf'.format(col_name,device_name))
    plt.close()




# iot is blue, non-iot is orange
def plot_barh(y,saved_path = None,tick_label = None, types=None):
    # construct colors, 1 is iot, use blue, 0 is non-iot, use orange
    colors = []
    for type in types:
        if type == 'iot':
            colors.append('cornflowerblue')
        else:
            colors.append('orange')
    fig,ax = plt.subplots()
    fig.set_size_inches(10, 6)
    plt.subplots_adjust(left=0.31)
    bar_width = 0.75
    x = np.arange(len(y))
    ax.barh(x,y,bar_width,color=colors)
    ax.set_yticks(x)
    ax.set_yticklabels(tick_label, minor=False,fontsize = 10)
    plt.xlim(0.2,1.1)
    for i, v in enumerate(y):
        ax.text(v+0.01, i-0.2, str(v), color='black',fontsize = 9)
    # plt.show()
    fig.savefig(saved_path,dpi=300,format='png', bbox_inches='tight')
    plt.close()

#========================================importance plot==========================================================
def plot_importance_barh(y,saved_path = None,tick_label = None):
    # construct colors, 1 is iot, use blue, 0 is non-iot, use orange
    top = -20 # 取后20个特征
    sorted_idx = y.argsort()
    y = y[sorted_idx]
    tick_label = [tick_label[idx] for idx in sorted_idx]
    y = y[top:]
    tick_label = tick_label[top:]
    y = [round(item,3) for item in y]
    fig,ax = plt.subplots()
    fig.set_size_inches(8, 6)
    plt.subplots_adjust(left=0.25)
    bar_width = 0.75
    x = np.arange(len(y))
    ax.barh(x,y,bar_width)
    ax.set_yticks(x)
    ax.set_yticklabels(tick_label, minor=False,fontsize = 10)
    plt.xlim(0,0.24)
    for i, v in enumerate(y):
        ax.text(v, i-0.25, str(v), color='black',fontsize = 9)
    # plt.show()
    fig.savefig(saved_path,dpi=300,format='png', bbox_inches='tight')
    plt.close()

# plot the importance change of top 5 features
def plot_importance_change(used_feature_importance_recorder,top_5_features,path):
    # plot top 5 features importance and the final value
    xlabels = ['RF{}'.format(item+1) for item in range(9)]
    markers = ['.','^','v','s','p','*','D','h','P']
    colors = ['grey','lightcoral', 'orange', 'blue', 'darkorange', 'cornflowerblue', 'orangered', 'saddlebrown','gold']
    # xlabels = ['']+xlabels
    # fig,ax = plt.subplots(dpi=300)
    # fig.set_size_inches(7, 5)
    ax = plt.figure(figsize=(7.5,7),dpi=300)
    for idx, batch in enumerate(used_feature_importance_recorder):
        plt.plot(batch,label = top_5_features[idx],marker=markers[idx],color=colors[idx],markersize=8)
    plt.xticks(range(len(xlabels)),xlabels)
    # ax.set(xticklabels=xlabels)
    plt.xlabel('Model',fontsize=24,fontproperties=SimHei)
    plt.ylabel('Feature importance',fontsize=24,fontproperties=SimHei)
    plt.ylim(0,0.3)
    plt.legend(bbox_to_anchor=(1.04, 1.06),borderaxespad=0,ncol=1,loc='lower right',prop=SimHei,fontsize=24)
    plt.grid(True,alpha=0.3)
    # for i in range(len(ax.get_xticklabels())):
    #     ax.get_xticklabels()[i].set_fontweight("bold")
    # for i in range(len(ax.get_yticklabels())):
    #     ax.get_yticklabels()[i].set_fontweight("bold")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.show()
    plt.tight_layout()
    plt.savefig(path + '/feature_importance_change.png')
    plt.savefig(path + '/feature_importance_change.pdf')
    plt.close()

# plot the final importance of the top 5 features
def plot_final_importance(feature_importance,top_5_features,path):
    fig,ax = plt.subplots(dpi=300)
    fig.set_size_inches(10, 7)
    top_5_features = [''] + top_5_features
    plt.bar(np.arange(5),feature_importance,width=0.5)
    ax.set(xticklabels=top_5_features)
    plt.xlabel('Features',fontdict=font,fontsize=22)
    plt.ylabel('Feature importance',fontdict=font,fontsize=22)
    for i in range(len(ax.get_xticklabels())):
        ax.get_xticklabels()[i].set_fontweight("bold")
    for i in range(len(ax.get_yticklabels())):
        ax.get_yticklabels()[i].set_fontweight("bold")
    # plt.ylim(0,0.3)
    plt.savefig(path + '/final_feature_importance.jpg')
    plt.close()


# plot bar
def plot_bar(data,xticklabels,path,filename):
    fig,ax = plt.subplots(dpi=300)
    fig.set_size_inches(11, 7) # 10,7
    xticklabels = [''] + xticklabels
    plt.bar(np.arange(5),data,width=0.5)
    ax.set(xticklabels=xticklabels)
    plt.xlabel('Features',fontdict=font,fontsize=22)
    plt.ylabel('Feature importance',fontdict=font,fontsize=22)
    for i in range(len(ax.get_xticklabels())):
        ax.get_xticklabels()[i].set_fontweight("bold")
    for i in range(len(ax.get_yticklabels())):
        ax.get_yticklabels()[i].set_fontweight("bold")
    # plt.ylim(0,0.3)
    plt.savefig(path + '/{}.jpg'.format(filename))
    plt.close()

# 
def plot_bar_common(data,xticklabels,xlabel,ylabel,path,filename,postfix):
    X = list(range(1,len(data)+1))
    fig,ax = plt.subplots(dpi=300)
    fig.set_size_inches(6, 4)
    xticklabels = [''] + xticklabels
    plt.bar(X,data)
    ax.set_xticks(X)
    # ax.set(xticklabels=xticklabels)
    plt.xlabel(xlabel,fontdict=font,fontsize=14)
    plt.ylabel(ylabel,fontdict=font,fontsize=14)
    for i in range(len(ax.get_xticklabels())):
        ax.get_xticklabels()[i].set_fontweight("bold")
    for i in range(len(ax.get_yticklabels())):
        ax.get_yticklabels()[i].set_fontweight("bold")
    plt.ylim(0,max(data)+1)
    plt.xlim(0,len(data)+1)
    plt.savefig(path + '/{}_{}.jpg'.format(filename,postfix))
    plt.close()


def plot_bar_2(data1,data2,xticklabels,labels_2,y_label,path,filename,x_label='Model'):
    font1={
    # 'family':'Arial',
    'weight':'medium',
      'size':22
    }
    data1 = [round(item*100,2) if item!=1.00 else 100 for item in data1]
    data2 = [round(item*100,2) if item!=1 else 100 for item in data2]
    
    fig,ax = plt.subplots(dpi=300)
    fig.set_size_inches(6.5, 5)
    xx = range(len(data1))
    plt.bar(xx,data1,width=0.4,label=labels_2[0],color='royalblue',alpha=0.8)
    plt.bar([i+0.4 for i in xx],data2,width=0.4,label=labels_2[1],color='#CC9966',alpha=0.7)
    plt.xticks([i+0.2 for i in xx],xticklabels)
    # ax.set(xticklabels=xticklabels)
    # plt.xlabel(x_label,fontdict=font1,fontsize=16,fontproperties=font_chinese)
    # plt.ylabel(y_label,fontdict=font1,fontsize=16,fontproperties=font_chinese)
    plt.xlabel(x_label,fontdict=font1)
    plt.ylabel(y_label,fontdict=font1)
    # for i in range(len(ax.get_xticklabels())):
    #     ax.get_xticklabels()[i].set_fontweight("bold")
    # for i in range(len(ax.get_yticklabels())):
    #     ax.get_yticklabels()[i].set_fontweight("bold")
    # for x_,y_ in zip(xx,data1):
    #     plt.text(x_,y_+0.05,'%.2f' %y_, ha='center',va='bottom',fontdict=font_bar_text)
    # for x_,y_ in zip(xx,data2):
    #     plt.text(x_+0.4,y_+0.05,'%.2f' %y_, ha='center',va='bottom',fontdict=font_bar_text)
    plt.ylim(0,140)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=16,weight='medium')
    plt.yticks(fontsize=16,weight='medium')
    plt.tight_layout()
    plt.savefig(path + '/{}.png'.format(filename))
    plt.savefig(path + '/{}.pdf'.format(filename))
    plt.close()

def plot_bar_2_1(data1,data2,xticklabels,legends,xlabel,ylabel,path,filename):
    fig,ax = plt.subplots(dpi=300)
    fig.set_size_inches(5, 4)
    xx = range(len(data1))
    plt.bar(xx,data1,width=0.4,label=legends[0],color='royalblue',alpha=0.8)
    plt.bar([i+0.4 for i in xx],data2,width=0.4,label=legends[1],color='orange',alpha=0.7)
    plt.xticks([i+0.2 for i in xx],xticklabels)
    # ax.set(xticklabels=xticklabels)
    plt.xlabel(xlabel,fontdict=font,fontsize=12)
    plt.ylabel(ylabel,fontdict=font,fontsize=12)
    # for i in range(len(ax.get_xticklabels())):
    #     ax.get_xticklabels()[i].set_fontweight("bold")
    # for i in range(len(ax.get_yticklabels())):
    #     ax.get_yticklabels()[i].set_fontweight("bold")
    for x_,y_ in zip(xx,data1):
        plt.text(x_,y_+0.05,'%.2f' %y_, ha='center',va='bottom',fontdict=font_bar_text)
    for x_,y_ in zip(xx,data2):
        plt.text(x_+0.4,y_+0.05,'%.2f' %y_, ha='center',va='bottom',fontdict=font_bar_text)
    plt.ylim(0,140)
    plt.legend(fontsize=10)
    plt.savefig(path + '/{}'.format(filename))
    plt.close()


def plot_scatter(x,y,labels,right,xlabel,ylabel,path,batch_idx):
    font11={
    # 'family':'Arial',
    'weight':'medium',
      'size':22
    }

    colors = ['grey','lightcoral','red','yellow','sienna','orange','olivedrab','yellowgreen',
    'darkolivegreen','teal','deepskyblue','slategray','cornflowerblue','navy','blue','blueviolet',
    'indigo','violet','pink','chocolate','deeppink','maroon','saddlebrown','brown']
    if not os.path.exists(path):
        os.mkdir(path)
    # plt.clf()
    plt.figure(figsize=(6,7),dpi=300)
    labels_unique = np.unique(labels)
    for i,label in enumerate(labels_unique):
        label_indicator = labels==label
        right_label_indicator = label_indicator&right
        wrong_label_indicator = label_indicator&(~right)
        if sum(right_label_indicator)>0:
            plt.scatter(x[right_label_indicator],y[right_label_indicator],s=18,color=colors[i],marker='o',label=str(int(label))+'_right')
        if sum(wrong_label_indicator)>0:
            plt.scatter(x[wrong_label_indicator],y[wrong_label_indicator],s=18,color=colors[i],marker='v',label=str(int(label))+'_wrong',edgecolors='black',linewidths=0.3)
    plt.xlabel(xlabel,fontdict=font11)
    plt.ylabel(ylabel,fontdict=font11)
    plt.xticks(fontsize=18,weight='medium')
    plt.yticks(fontsize=18,weight='medium')
    plt.legend(fontsize=18)
    # plt.show()
    plt.tight_layout()
    plt.savefig(path+'/batch_{}.png'.format(batch_idx))
    plt.savefig(path+'/batch_{}.pdf'.format(batch_idx))
    plt.close()


def plot_scatter_for_prob_variance(x,y,labels,right,xlabel,ylabel,path,batch_idx):
    font11={
    # 'family':'Arial',
    'weight':'medium',
      'size':22
    }

    colors = ['grey','lightcoral','red','deepskyblue','blueviolet','sienna','orange','olivedrab','yellowgreen',
    'darkolivegreen','teal','slategray','cornflowerblue','navy','blue',
    'indigo','violet','pink','chocolate','deeppink','maroon','saddlebrown','brown','yellow']
    if not os.path.exists(path):
        os.mkdir(path)
    # plt.clf()
    plt.figure(figsize=(7,6),dpi=300)
    labels_unique = np.unique(labels)
    for i,label in enumerate(labels_unique):
        label_indicator = labels==label
        right_label_indicator = label_indicator&right
        wrong_label_indicator = label_indicator&(~right)
        if sum(right_label_indicator)>0:
            plt.scatter(x[right_label_indicator],y[right_label_indicator],s=18,color=colors[i],marker='o',label=str(int(label))+'_right')
        if sum(wrong_label_indicator)>0:
            plt.scatter(x[wrong_label_indicator],y[wrong_label_indicator],s=18,color=colors[i],marker='v',label=str(int(label))+'_wrong',edgecolors='black',linewidths=0.3)
    plt.xlabel(xlabel,fontdict=font11)
    plt.ylabel(ylabel,fontdict=font11)
    plt.xticks(fontsize=18,weight='medium')
    plt.yticks(fontsize=18,weight='medium')
    plt.legend(fontsize=18,bbox_to_anchor=(0.9, 1.03),ncol=2,loc='lower right',borderaxespad=0)
    # plt.show()
    plt.tight_layout()
    plt.savefig(path+'/batch_{}.png'.format(batch_idx))
    plt.savefig(path+'/batch_{}.pdf'.format(batch_idx))
    plt.close()



def plot_line_chart(X,Y,xlabel,ylabel,label,path):
    font1={
    # 'family':'Arial',
    'weight':'medium',
      'size':26
    }       
    plt.clf()
    plt.figure(figsize=(9,7),dpi=300)
    plt.tick_params(labelsize=18)
    plt.xlabel(xlabel,fontdict=font1)
    plt.ylabel(ylabel,fontdict=font1)
    plt.plot(X,Y,label=label,markersize=5,marker='o')
    plt.xlim(X[0],X[-1])
    plt.grid(ls='--',alpha=0.5)       
    plt.xticks(X)
    # plt.vlines(2,0,3000,colors='black',linestyles='dashed')
    plt.savefig(path + '.png')
    plt.savefig(path + '.pdf')
    plt.tight_layout()
    plt.close()



colors = ['blue','orangered','orange','grey','gold','deepskyblue','slategray', 'cornflowerblue', 'navy','lawngreen']
def plot_3D_scatter(encoded_imgs, labels, azim, elev, title="latent_space", path=None):
    # title=None
    label_kind = len(np.unique(labels))
    labels_unique = list(np.unique(labels))
    labels_unique = [str(i) for i in labels_unique]
    cMap = c.ListedColormap(colors[:label_kind])
    ticks = np.arange(label_kind)
    fig = plt.figure(figsize=(9,6),dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], encoded_imgs[:, 2], c=labels, cmap=cMap) #plt.cm.get_cmap('jet', 25)
    # p = ax.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], encoded_imgs[:, 2], c=label, cmap=plt.cm.get_cmap('jet', 25),edgecolors='none') #plt.cm.get_cmap('jet', 25)
    ax.view_init(azim=azim,elev=elev)
    # ax.set_title(title,fontsize=20)
    cbar = fig.colorbar(p, ticks=ticks,fraction=0.046, pad=0.04, aspect=20)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels_unique)
    fig.tight_layout()
    fig.savefig(path+'png')
    fig.savefig(path+'pdf')
    plt.close()

def replace_labels(labels):
    BASE = 1000000
    labels = np.array(labels)
    label_kind = len(np.unique(labels))
    for label in np.unique(labels):
        labels[labels==label] = BASE + label
    for idx,label in enumerate(np.unique(labels)):
        labels[labels==label] = idx
    return labels

def plot_2D_scatter(encoded_imgs, labels, azim, elev, title="latent_space", path=None):
    label_kind = len(np.unique(labels))
    labels_unique = list(np.unique(labels))
    labels_unique = [str(int(i)) for i in labels_unique]
    labels = replace_labels(labels)
    cMap = c.ListedColormap(colors[:label_kind])
    ticks = np.arange(label_kind)
    fig = plt.figure(figsize=(6,5),dpi=300)
    ax = fig.add_subplot(111)
    # p = ax.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1],
    #                c=labels,cmap=cMap,alpha=0.6)#cmap=plt.cm.get_cmap('jet', 25)
    p = ax.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1],c=labels,cmap=plt.cm.get_cmap('rainbow',label_kind),alpha=0.6)
    ax.tick_params(labelsize= 18)
    # ax.set_title(title,fontsize=20)
    cbar = fig.colorbar(p, ticks=ticks,fraction=0.046, pad=0.04, aspect=20)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels_unique)
    cbar.ax.tick_params(labelsize=18)
    fig.tight_layout()
    fig.savefig(path+'.png')
    fig.savefig(path+'.pdf')
    plt.close()

def plot_box(data,labels,xlabel,ylabel,path):
    fig = plt.figure(figsize=(9,6),dpi=300)
    ax = fig.add_subplot(111)
    plt.boxplot(data,labels=labels)
    plt.xlabel(xlabel,fontdict=font)
    plt.ylabel(ylabel,fontdict=font)  
    plt.savefig(path+'png')
    plt.savefig(path+'pdf')
    plt.close()

# plot two groups of threshold
def plot_thresholds(xticklabels,threshold1_selected,threshold2_selected,feature_values,feature_idxs,device_idx,instance_idx,path):
    colors = ['blue','orangered','orange','grey','gold','deepskyblue','slategray', 'cornflowerblue', 'navy','lawngreen']
    plt.figure(figsize=(6, 4), dpi=300)
    idx = 0
    for threshold_f1,threshold_f2 in zip(threshold1_selected,threshold2_selected):
        # thresholds of each kind of feature 
        for threshold_i in threshold_f1:
            plt.axvline(x=threshold_i,ls='--',lw=0.5,c='red',ymin=0,ymax=0.3,label='rf_before_{}'.format(feature_idxs[idx]),alpha=0.8)
        for threshold_i in threshold_f2:
            plt.axvline(x=threshold_i,ls='-',lw=0.5,c='blue',ymin=0,ymax=0.3,label='rf_after_{}'.format(feature_idxs[idx]),alpha=0.8)
        plt.axvline(x=feature_values[idx],ls='-',lw=0.5,c='black',ymin=0,ymax=0.3,label='real_value_{}'.format(feature_idxs[idx]),alpha=0.8)
        # plt.legend()
        plt.savefig(path + '/thresholds_{}_{}_{}.png'.format(device_idx,instance_idx,feature_idxs[idx]))
        plt.savefig(path + '/thresholds_{}_{}_{}.pdf'.format(device_idx,instance_idx,feature_idxs[idx]))
        plt.clf()
        idx += 1
    plt.close()

def plot_thresholds_mean(xticklabels,threshold1_selected,threshold2_selected,feature_values,feature_idxs,device_idx,instance_idx,path):
    colors = ['blue','orangered','orange','grey','gold','deepskyblue','slategray', 'cornflowerblue', 'navy','lawngreen']
    plt.figure(figsize=(6, 4), dpi=300)
    idx = 0
    for threshold_f1,threshold_f2 in zip(threshold1_selected,threshold2_selected):
        # thresholds of each kind of feature 
        threshold_i = np.mean(threshold_f1)
        threshold_j = np.mean(threshold_f2)
        plt.axvline(x=threshold_i,ls='-',lw=0.5,c=colors[idx],ymin=0,ymax=0.3,label='rf_before_{}'.format(feature_idxs[idx]),alpha=1)
        plt.axvline(x=threshold_j,ls='--',lw=0.5,c=colors[idx],ymin=0,ymax=0.3,label='rf_after_{}'.format(feature_idxs[idx]),alpha=1)
        plt.axvline(x=feature_values[idx],ls='--',lw=0.5,c='black',ymin=0,ymax=0.3,label='real_value_{}'.format(feature_idxs[idx]),alpha=1)       
        plt.legend(fontsize=7)
        plt.savefig(path + '/thresholds_{}_{}_{}.png'.format(device_idx,instance_idx,feature_idxs[idx]))
        plt.savefig(path + '/thresholds_{}_{}_{}.pdf'.format(device_idx,instance_idx,feature_idxs[idx]))
        plt.clf()
        idx += 1
    plt.close()
        


if __name__ == '__main__':
    pass
    # x = [u'INFO', u'CUISINE', u'TYPE_OF_PLACE', u'DRINK', u'PLACE', u'MEAL_TIME', u'DISH', u'NEIGHBOURHOOD']
    # y = [160, 167, 137, 18, 120, 36, 155, 130]

    # fig, ax = plt.subplots()    
    # width = 0.75 # the width of the bars
    # ind = np.arange(len(y))  # the x locations for the groups
    # ax.barh(ind, y, width, color="blue")
    # ax.set_yticks(ind)
    # ax.set_yticklabels(x, minor=False)
    # plt.title('title')
    # plt.xlabel('x')
    # plt.ylabel('y')      
    # for i, v in enumerate(y):
    #     ax.text(v + 3, i-0.1, str(v), color='blue', fontweight='bold')
    # plt.show()
    # plt.savefig(os.path.join('test.png'), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
