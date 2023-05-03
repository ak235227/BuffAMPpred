#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-plot')
get_ipython().system('pip install biopython')


# In[2]:


from Bio import SeqIO
import statistics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import LeaveOneOut
import numpy as np
#x_train.shape, y_train.shape
from sklearn.metrics import accuracy_score
import pickle


# In[30]:


res_dir = 'RF_aamir'

Check_direc = os.path.isdir(res_dir)

# If folder doesn't exist, then create it.
if not Check_direc:
    os.makedirs(res_dir)
    print("created folder : ", res_dir)

else:
    print(res_dir, "folder already exists.")


# In[16]:


Xp = np.load("Positive_AMPs_len_60.npy")
Xn = np.load("Negative_AMPs_len_60.npy")
Xp.shape, Xn.shape


# In[17]:


x, y = np.vstack((Xp , Xn)), np.array([1]*Xp.shape[0] + [0]*Xn.shape[0])
print (x)
print (y)
x.shape, y.shape


# In[18]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
undersample = RandomUnderSampler(sampling_strategy='majority')


# In[19]:


#down sample val_p equal to val_n as it has more number of data compare to val_n
#oversample = SMOTE()
x, y = undersample.fit_resample(x, y)
x.shape, y.shape


# In[20]:


preprocess = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.2, random_state=42)
x_train = preprocess.fit_transform(x_train)
x_test = preprocess.fit_transform(x_test)


# In[21]:


plt.plot(x_train, 'go') # green dots
plt.plot(x_test, 'b*') # blue dots
plt.show()


# In[9]:


### k stratified validation
x.shape, y.shape
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
accuracy = []
#svm = SVC(kernel='rbf')
rf = RandomForestClassifier(n_estimators = 101,min_weight_fraction_leaf=0.1, random_state = 42)
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x_train, y_train)
for train_index, test_index in skf.split(x_train, y_train):
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
    #print(x_train.shape)
    #print(x_test.shape)


# In[10]:


accuracy = []
rf = RandomForestClassifier(n_estimators = 301)
rf.fit(x_train, y_train)
prediction = rf.predict (x_test)
score = accuracy_score (prediction, y_test)  
accuracy.append(score)
print(accuracy)


# In[22]:


idx = np.random.permutation(Xp.shape[0])

idx_train = idx[len(Xp)//5:]
idx_test = idx[:len(Xp)//5]
Xp_train, Xp_test = Xp[idx_train], Xp[idx_test]

len_dist_p = np.sum(Xp_train > 0, axis=1)
len_dist_n = np.sum(Xn > 0, axis=1)
len_dist_p

Xn_train = list()
Xn_test = list()

for i in set(len_dist_n):
    temp_n = Xn[np.where(len_dist_n==i)[0]]
    temp_p = Xp_train[np.where(len_dist_p==i)[0]]
    print(temp_n.shape, temp_p.shape)
    if temp_n.shape[0] == 30*temp_p.shape[0]:
        Xn_train.append(temp_n)
    elif temp_n.shape[0] > 30*temp_p.shape[0]:
        idx = np.random.permutation(temp_n.shape[0])
        idx_train = idx[:30*temp_p.shape[0]]
        idx_test = idx[30*temp_p.shape[0]:]
        Xn_train.append(temp_n[idx_train])
        Xn_test.append(temp_n[idx_test])
        train_n = temp_n[idx_train]
        test_n = temp_n[idx_test]
        #print(train_n.shape, test_n.shape, temp_p.shape, i, train_n.shape[0]/temp_p.shape[0])
        #print(temp_n.shape, temp_p.shape, temp_p.shape, i, temp_n.shape[0]/temp_p.shape[0])
        
Xn_train = np.vstack(Xn_train)
Xn_test = np.vstack(Xn_test)

Xp_train.shape, Xp_test.shape, Xn_train.shape, Xn_test.shape


# In[23]:


X_test, y_test = np.vstack((Xp_test, Xn_test)), np.array([1]*Xp_test.shape[0] + [0]*Xn_test.shape[0])
X_test.shape, y_test.shape


# In[27]:


import os
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, classification_report, auc, accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle


# In[28]:


def get_training_data(data_p=None, data_n=None, random_state=0):
    np.random.seed(random_state)
    seq_len_p = np.sum(data_p > 0, axis=1)
    seq_len_n = np.sum(data_n > 0, axis=1)

    List = list()
    uniq_seq_len = Counter(seq_len_p)
    for k in uniq_seq_len:
        temp = data_n[np.where(seq_len_n==k)[0]]
        #print(k, temp.shape[0], uniq_seq_len[k])
        idx = np.random.randint(temp.shape[0], size=uniq_seq_len[k])
        #print(idx, k, len(idx), temp.shape[0])
        List.append(temp[idx])
    List = np.vstack(List)
    X, y = np.vstack((data_p, List)), np.array([1]*len(data_p) + [0]*len(List))
    assert len(data_p)==len(List)

    return X, y


# In[25]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from imblearn.metrics import specificity_score, sensitivity_score


# In[15]:


x.shape, y.shape
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
accuracy = []
#svm = SVC(kernel='rbf')
rf = RandomForestClassifier(n_estimators = 101,min_weight_fraction_leaf=0.1, random_state = 42)
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x_train, y_train)
for train_index, test_index in skf.split(x_train, y_train):
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
    print(x_train.shape)
    print(x_test.shape)


# In[34]:


n_boost_round = 100
epochs = 1000
lr = 0.001
prediction_train = 0
prediction_test = 0
tol_mcc = 0.001
prev_mcc = 0
mcc_list = list()
metric_list = list()

for i in range(n_boost_round):
    print(i)

    X_tr, y_tr = get_training_data(data_p=Xp_train, data_n=Xn_train, random_state=i)
    #X_val, y_val = np.vstack((Xp_val, Xn_val)), np.array([1]*Xp_val.shape[0] + [0]*Xp_val.shape[0])

    model = RandomForestClassifier(n_estimators = 101,min_weight_fraction_leaf=0.1, random_state = 42)
    #skf = StratifiedKFold(n_splits=10)
    #skf.get_n_splits(x_train, y_train)
    #for train_index, test_index in skf.split(x_train, y_train):
    #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
    
    #w_file = os.path.join(res_dir, 'model_weights'+str(i)+'.h5')
    #csv_file = os.path.join(res_dir, 'training_logg'+str(i)+'.csv')
    
    model.fit(X_tr, y_tr)

    #model.load_weights(w_file)
    pred_test = model.predict(X_test)
    pred_train = model.predict(X_tr)
    curr_mcc = matthews_corrcoef(y_test, pred_test > 0.5)
    prediction_train = (prediction_train*i + pred_train)/(i+1)
    prediction_test = (prediction_test*i + pred_test)/(i+1)

    metric_list.append([str(i), 'ACC', accuracy_score(y_tr, pred_train > 0.5), False, "train+tune"])
    metric_list.append([str(i), 'MCC', matthews_corrcoef(y_tr, pred_train > 0.5), False, "train+tune"])
    metric_list.append([str(i), 'Specificity', specificity_score(y_tr, pred_train > 0.5), False, "train+tune"])
    metric_list.append([str(i), 'Sensitivity', sensitivity_score(y_tr, pred_train > 0.5), False, "train+tune"])
    metric_list.append([str(i), 'AUC', roc_auc_score(y_tr, pred_train), False, "train+tune"])
    metric_list.append([str(i), 'Precision', precision_score(y_tr, pred_train > 0.5), False, "train+tune"])
    metric_list.append([str(i), 'Recall', recall_score(y_tr, pred_train > 0.5), False, "train+tune"])
    metric_list.append([str(i), 'F1_Score', f1_score(y_tr, pred_train > 0.5), False, "train+tune"])

    metric_list.append([str(i), 'ACC', accuracy_score(y_tr, prediction_train > 0.5), True, "train+tune"])
    metric_list.append([str(i), 'MCC', matthews_corrcoef(y_tr, prediction_train > 0.5), True, "train+tune"])
    metric_list.append([str(i), 'Specificity', specificity_score(y_tr, prediction_train > 0.5), True, "train+tune"])
    metric_list.append([str(i), 'Sensitivity', sensitivity_score(y_tr, prediction_train > 0.5), True, "train+tune"])
    metric_list.append([str(i), 'AUC', roc_auc_score(y_tr, prediction_train), True, "train+tune"])
    metric_list.append([str(i), 'Precision', precision_score(y_tr, prediction_train > 0.5), True, "train+tune"])
    metric_list.append([str(i), 'Recall', recall_score(y_tr, prediction_train > 0.5), True, "train+tune"])
    metric_list.append([str(i), 'F1_Score', f1_score(y_tr, prediction_train > 0.5), True, "train+tune"])

    metric_list.append([str(i), 'ACC', accuracy_score(y_test, pred_test > 0.5), False, "test"])
    metric_list.append([str(i), 'MCC', matthews_corrcoef(y_test, pred_test > 0.5), False, "test"])
    metric_list.append([str(i), 'Specificity', specificity_score(y_test, pred_test > 0.5), False, "test"])
    metric_list.append([str(i), 'Sensitivity', sensitivity_score(y_test, pred_test > 0.5), False, "test"])
    metric_list.append([str(i), 'AUC', roc_auc_score(y_test, pred_test), False, "test"])
    metric_list.append([str(i), 'Precision', precision_score(y_test, pred_test > 0.5), False, "test"])
    metric_list.append([str(i), 'Recall', recall_score(y_test, pred_test > 0.5), False, "test"])
    metric_list.append([str(i), 'F1_Score', f1_score(y_test, pred_test > 0.5), False, "test"])

    metric_list.append([str(i), 'ACC', accuracy_score(y_test, prediction_test > 0.5), True, "test"])
    metric_list.append([str(i), 'MCC', matthews_corrcoef(y_test, prediction_test > 0.5), True, "test"])
    metric_list.append([str(i), 'Specificity', specificity_score(y_test, prediction_test > 0.5), True, "test"])
    metric_list.append([str(i), 'Sensitivity', sensitivity_score(y_test, prediction_test > 0.5), True, "test"])
    metric_list.append([str(i), 'AUC', roc_auc_score(y_test, prediction_test), True, "test"])
    metric_list.append([str(i), 'Precision', precision_score(y_test, prediction_test > 0.5), True, "test"])
    metric_list.append([str(i), 'Recall', recall_score(y_test, prediction_test > 0.5), True, "test"])
    metric_list.append([str(i), 'F1_Score', f1_score(y_test, prediction_test > 0.5), True, "test"])
#    mcc_list.append(matthews_corrcoef(y_test, pred_test > 0.5))
#    if (i > 10) and np.abs(np.mean(mcc_list) - np.mean(mcc_list[:-1]) < tol_mcc):
#        break
#    prev_mcc = curr_mcc
    df_metric = pd.DataFrame(metric_list, columns=['Model_No.', 'Metric', 'Score', 'Ensemble', 'Dataset'])
    df_metric.to_csv(os.path.join(res_dir,'RF_results_score.csv'))
metric_list


# In[35]:


df_metric


# In[37]:


from scipy import stats
grp = df_metric.loc[(df_metric['Ensemble']==True) & (df_metric['Dataset']=="test")].groupby("Metric")
df1 = grp['Score'].agg([np.sum, np.mean, np.std, stats.sem])
df1.to_csv(os.path.join(res_dir,'RF-Ensemble-T_Sum-Mean-Std-SE_score.csv'))

grp1 = df_metric.loc[(df_metric['Ensemble']==False) & (df_metric['Dataset']=="test")].groupby("Metric")
df2 = grp['Score'].agg([np.sum, np.mean, np.std, stats.sem])
df2.to_csv(os.path.join(res_dir,'RF-Ensemble-F_Sum-Mean-Std-SE_score.csv'))


# In[40]:


df = pd.read_csv('RF_aamir/RF_results_score.csv', index_col=0)
df


# In[47]:


df_results = dict()
df_results_summary = dict()

for e in df["Ensemble"].unique():
    for d in df["Dataset"].unique():
        if e==True:
            df_temp = df.loc[(df["Ensemble"]==e) & (df["Dataset"]==d) & (df['Model_No.']==df['Model_No.'].max()),:].reset_index(drop=True)
            df_results[str(e)+'_'+str(d)]=df_temp
            grp = df_temp.groupby("Metric")
            temp = grp['Score'].agg([np.mean])
            #temp = temp.T
            temp = temp.round(4)
        else:
            df_temp = df.loc[(df["Ensemble"]==e) & (df["Dataset"]==d),:].reset_index(drop=True)
            df_results[str(e)+'_'+str(d)]=df_temp
            grp = df_temp.groupby("Metric")
            temp = grp['Score'].agg([np.mean, stats.sem])
            #temp = temp.T
            temp = temp.round(4)
            tt = [str(a) + ' ' + '(±' + str(b) +')' for (a,b) in zip(temp['mean'], temp['sem'])]
            temp['mean_score  (± se)'] = tt
        df_results_summary[str(e)+'_'+str(d)] = temp.T


# In[48]:


df_results_summary.keys()


# In[49]:


df_results_summary["False_test"]


# In[50]:


df_results_summary["False_train+tune"]


# In[51]:


df_results_summary["True_test"]


# In[52]:


df_results_summary["True_train+tune"]


# In[ ]:


CM_Ensemble_False = confusion_matrix(y_te, pred_test > 0.5)
CM_Ensemble_True = confusion_matrix(y_te, prediction_test > 0.5)
CM_Ensemble_False, CM_Ensemble_True, CM_Ensemble_False.dtype, CM_Ensemble_False.shape, CM_Ensemble_False.ndim


# In[53]:


filename = 'random_forest_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[ ]:




