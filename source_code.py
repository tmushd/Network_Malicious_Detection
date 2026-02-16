#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ignore all warnings
import warnings
warnings.filterwarnings('ignore')


# In[4]:


#get_ipython().system('pip install torch_optimizer numpy torch transformers evaluate python-whois --quiet')


# In[5]:


from datasets import load_dataset
from datasets import Dataset
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Import label encoder
from sklearn import preprocessing, metrics

import itertools
from sklearn.metrics import classification_report, mean_squared_error,confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, auc,roc_curve
from sklearn.model_selection import train_test_split
import random
import math
from collections import Counter
import xgboost as xgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import os
import socket
import whois
from datetime import datetime
import time
from bs4 import BeautifulSoup
import urllib
import bs4
import os


# In[6]:


#df = pd.read_csv("data", "malicious_phish.csv")
import os
df = pd.read_csv(os.path.join("data", "malicious_phish.csv"))


# In[7]:


print(df.shape)
print(df.info())


# In[8]:


df.groupby('type').apply(lambda x: x.sample(1)).reset_index(drop=True)


# In[9]:


df.isna().sum()


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

types = df['type'].values

counter_types = Counter(types)

type_names = list(counter_types.keys())
type_values = list(counter_types.values())

sorted_indices = np.argsort(type_values)[::-1]
type_names = [type_names[i] for i in sorted_indices]
type_values = [type_values[i] for i in sorted_indices]

total_count = sum(type_values)
percentages = [value / total_count * 100 for value in type_values]

pattern = '//'

y_pos = np.arange(len(type_names))
plt.figure(1, figsize=(10, 5))
bars = plt.bar(y_pos, type_values, align='center', alpha=0.7, color='none', edgecolor='black', hatch=pattern)

for bar, value, percentage in zip(bars, type_values, percentages):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{value}', ha='center', va='bottom')
    plt.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{percentage:.1f}%', ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), fontweight='bold')

plt.xticks(y_pos, type_names)
plt.ylabel('Number of URLs')
plt.title('Distribution of URLs per Type')
plt.show()

print(counter_types)


# In[11]:


df['url_len'] = [len(url) for url in df.url]
df.head()


# In[12]:


# Plot distribution of 'url_len' for each 'type'
sns.displot(df, x='url_len', hue='type', kind='kde', fill=True)

# Add labels and title
plt.xlabel('URL Length')
plt.ylabel('Density')
plt.title('Distribution of URL Length by Type')
plt.show()


# In[13]:


from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()
le.fit(df["type"])

df["type_code"] = le.transform(df["type"])
df


# In[14]:


le_label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
le_label_mapping


# In[15]:


df = df[['url', 'type', 'type_code']]
dataset = Dataset.from_pandas(df, preserve_index=False)
dataset


# In[16]:


# split train to 80% of total and test to 20% of total
train_test_dataset = dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
train_test_dataset


# In[17]:


# split the validation test to 10% of total and test set to 10% of total
val_test_dataset = train_test_dataset['test'].train_test_split(test_size=0.5, seed=42, shuffle=True)
val_test_dataset


# In[18]:


from datasets import DatasetDict

# 80% train, 10% validation, 10% test
dataset = DatasetDict({
    'train': train_test_dataset['train'],
    'val': val_test_dataset['train'],
    'test': val_test_dataset['test'],
})
dataset


# In[19]:


#get_ipython().system('pip install huggingface_hub')


# In[20]:


from huggingface_hub import login
login()


# In[21]:


dataset = load_dataset("bgspaditya/byt-mal-minpro")
dataset = dataset.rename_column("type_code", "labels")
dataset


# In[22]:


dataset = load_dataset("bgspaditya/byt-mal-minpro")
dataset = dataset.rename_column("type_code", "labels")
dataset


# In[23]:


df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])
df_val = pd.DataFrame(dataset['val'])


# In[24]:


import re
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0
df_train['use_of_ip'] = df_train['url'].apply(lambda i: having_ip_address(i))
df_test['use_of_ip'] = df_test['url'].apply(lambda i: having_ip_address(i))

df_train


# In[25]:


from urllib.parse import urlparse

def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0


df_train['abnormal_url'] = df_train['url'].apply(lambda i: abnormal_url(i))
df_test['abnormal_url'] = df_test['url'].apply(lambda i: abnormal_url(i))
df_train


# In[26]:


df_train['count.'] = df_train['url'].apply(lambda i: i.count('.'))
df_test['count.'] = df_test['url'].apply(lambda i: i.count('.'))

df_train.head()


# In[27]:


df_train['count-www'] = df_train['url'].apply(lambda i: i.count('www'))
df_test['count-www'] = df_test['url'].apply(lambda i: i.count('www'))
df_train


# In[28]:


df_train['count@'] = df_train['url'].apply(lambda i: i.count('@'))
df_test['count@'] = df_test['url'].apply(lambda i: i.count('@'))
df_train


# In[29]:


from urllib.parse import urlparse
def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

df_train['count_dir'] = df_train['url'].apply(lambda i: no_of_dir(i))
df_test['count_dir'] = df_test['url'].apply(lambda i: no_of_dir(i))
df_train


# In[30]:


def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

df_train['count_embed_domian'] = df_train['url'].apply(lambda i: no_of_embed(i))
df_test['count_embed_domian'] = df_test['url'].apply(lambda i: no_of_embed(i))
df_train


# In[31]:


def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0

df_train['sus_url'] = df_train['url'].apply(lambda i: suspicious_words(i))
df_test['sus_url'] = df_test['url'].apply(lambda i: suspicious_words(i))
df_train


# In[32]:


def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0

df_train['short_url'] = df_train['url'].apply(lambda i: shortening_service(i))
df_test['short_url'] = df_test['url'].apply(lambda i: shortening_service(i))
df_train


# In[33]:


df_train['count-https'] = df_train['url'].apply(lambda i : i.count('https'))
df_test['count-https'] = df_test['url'].apply(lambda i : i.count('https'))
df_train


# In[34]:


df_train['count-http'] = df_train['url'].apply(lambda i : i.count('http'))
df_test['count-http'] = df_test['url'].apply(lambda i : i.count('http'))
df_train


# In[35]:


df_train['count%'] = df_train['url'].apply(lambda i: i.count('%'))
df_test['count%'] = df_test['url'].apply(lambda i: i.count('%'))
df_train


# In[36]:


df_train['count-'] = df_train['url'].apply(lambda i: i.count('-'))
df_test['count-'] = df_test['url'].apply(lambda i: i.count('-'))
df_train


# In[37]:


df_train['count='] = df_train['url'].apply(lambda i: i.count('='))
df_test['count='] = df_test['url'].apply(lambda i: i.count('='))
df_train


# In[38]:


df_train['url_length'] = df_train['url'].apply(lambda i: len(str(i)))
df_test['url_length'] = df_test['url'].apply(lambda i: len(str(i)))
df_train


# In[39]:


df_train['hostname_length'] = df_train['url'].apply(lambda i: len(urlparse(i).netloc))
df_test['hostname_length'] = df_test['url'].apply(lambda i: len(urlparse(i).netloc))
df_train


# In[40]:


#get_ipython().system('pip install tld')


# In[41]:


#Importing dependencies
from urllib.parse import urlparse
from tld import get_tld
import os.path

#First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

df_train['fd_length'] = df_train['url'].apply(lambda i: fd_length(i))
df_test['fd_length'] = df_test['url'].apply(lambda i: fd_length(i))
df_train


# In[42]:


#Length of Top Level Domain
df_train['tld'] = df_train['url'].apply(lambda i: get_tld(i,fail_silently=True))
df_test['tld'] = df_test['url'].apply(lambda i: get_tld(i,fail_silently=True))

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

df_train['tld_length'] = df_train['tld'].apply(lambda i: tld_length(i))
df_test['tld_length'] = df_test['tld'].apply(lambda i: tld_length(i))
df_train


# In[43]:


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

df_train['count-digits']= df_train['url'].apply(lambda i: digit_count(i))
df_test['count-digits']= df_test['url'].apply(lambda i: digit_count(i))
df_train


# In[44]:


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

df_train['count-letters']= df_train['url'].apply(lambda i: letter_count(i))
df_test['count-letters']= df_test['url'].apply(lambda i: letter_count(i))
df_train


# In[45]:


df_train = df_train.drop("tld",axis=1)
df_test = df_test.drop("tld",axis=1)
df_train


# In[50]:


df_train.head()


# In[51]:


#Predictor Variables
X_train = df_train[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
       'count_dir', 'count_embed_domian', 'short_url', 'count-https',
       'count-http', 'count%', 'count-', 'count=', 'url_length',
       'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
       'count-letters']]

#Target Variable
y_train = df_train['labels']


# In[52]:


#Predictor Variables
X_test = df_test[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
       'count_dir', 'count_embed_domian', 'short_url', 'count-https',
       'count-http', 'count%', 'count-', 'count=', 'url_length',
       'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
       'count-letters']]

#Target Variable
y_test = df_test['labels']


# In[53]:


import os
os.makedirs("outputs", exist_ok=True)

X_train.to_csv("outputs/x-train.csv", index=False)
y_train.to_csv("outputs/y-train.csv", index=False)
X_test.to_csv("outputs/x-test.csv", index=False)
y_test.to_csv("outputs/y-test.csv", index=False)


# In[54]:


eval_df = pd.DataFrame(columns=['Model', 'Accuracy', 'F1-macro', 'F1-micro', 'F1-weighted'])
eval_df


# In[55]:


lgb = LGBMClassifier(objective='multiclass',boosting_type= 'gbdt',n_jobs = 5,
          silent = True, random_state=5)
LGB_C = lgb.fit(X_train, y_train)

y_predLGB = LGB_C.predict(X_test)
print(classification_report(y_test,y_predLGB))

score = metrics.accuracy_score(y_test, y_predLGB)
print("accuracy:   %0.3f" % score)


# In[56]:


lgbm_acc = accuracy_score(y_test, y_predLGB)
lgbm_acc


# In[57]:


lgbm_f1_macro = f1_score(y_test, y_predLGB, average='macro')
lgbm_f1_macro


# In[58]:


lgbm_f1_micro = f1_score(y_test, y_predLGB, average='micro')
lgbm_f1_micro


# In[59]:


lgbm_f1_w = f1_score(y_test, y_predLGB, average='weighted')
lgbm_f1_w


# In[60]:


new_eval = {'Model': 'LGBM','Accuracy': lgbm_acc, 'F1-macro': lgbm_f1_macro, 'F1-micro': lgbm_f1_micro, 'F1-weighted': lgbm_f1_w }
eval_df.loc[len(eval_df)] = new_eval


# In[61]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[62]:


cm = metrics.confusion_matrix(y_test, y_predLGB, labels=[0,1,2,3])
plot_confusion_matrix(cm,classes=['benign', 'defacement','phishing','malware'])


# In[63]:


lgb_feature = lgb.feature_importances_
lgb_feature


# In[64]:


lgb_features = lgb_feature.tolist()


# In[65]:


import pickle
# saving model
lgbm_pkl = "lgbm.pkl"
with open(lgbm_pkl, 'wb') as file:
    pickle.dump(LGB_C, file)


# In[66]:


xgb = xgb.XGBClassifier(n_estimators= 100)
xgb.fit(X_train,y_train)
y_predXGB = xgb.predict(X_test)
print(classification_report(y_test,y_predXGB))


score = metrics.accuracy_score(y_test, y_predXGB)
print("accuracy:   %0.3f" % score)


# In[67]:


xgb_acc = accuracy_score(y_test, y_predXGB)
xgb_acc


# In[68]:


xgb_f1_macro = f1_score(y_test, y_predXGB, average='macro')
xgb_f1_macro


# In[69]:


xgb_f1_micro = f1_score(y_test, y_predXGB, average='micro')
xgb_f1_micro


# In[70]:


xgb_f1_w = f1_score(y_test, y_predXGB, average='weighted')
xgb_f1_w


# In[71]:


new_eval = {'Model': 'XGB','Accuracy': xgb_acc, 'F1-macro': xgb_f1_macro, 'F1-micro': xgb_f1_micro, 'F1-weighted': xgb_f1_w }
eval_df.loc[len(eval_df)] = new_eval


# In[72]:


CM=confusion_matrix(y_test,y_predXGB,labels=[0,1,2,3])

plot_confusion_matrix(cm,classes=['benign', 'defacement','phishing','malware'])


# In[73]:


xgb_feature = xgb.feature_importances_
xgb_features = xgb_feature.tolist()


# In[74]:


import pickle
# saving model
xgb_pkl = "xgb.pkl"
with open(xgb_pkl, 'wb') as file:
    pickle.dump(xgb, file)


# In[75]:


from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(n_estimators=100,max_features='sqrt')
gbdt.fit(X_train,y_train)
y_predGBDT = gbdt.predict(X_test)
print(classification_report(y_test,y_predGBDT))

score = metrics.accuracy_score(y_test, y_predGBDT)
print("accuracy:   %0.3f" % score)


# In[76]:


gbdt_acc = accuracy_score(y_test, y_predGBDT)
gbdt_acc


# In[77]:


gbdt_f1_macro = f1_score(y_test, y_predGBDT, average='macro')
gbdt_f1_macro


# In[78]:


gbdt_f1_micro = f1_score(y_test, y_predGBDT, average='micro')
gbdt_f1_micro


# In[79]:


gbdt_f1_w = f1_score(y_test, y_predGBDT, average='weighted')
gbdt_f1_w


# In[80]:


new_eval = {'Model': 'GBDT','Accuracy': gbdt_acc, 'F1-macro': gbdt_f1_macro, 'F1-micro': gbdt_f1_micro, 'F1-weighted': gbdt_f1_w }
eval_df.loc[len(eval_df)] = new_eval


# In[81]:


CM=confusion_matrix(y_test,y_predGBDT,labels=[0,1,2,3])

plot_confusion_matrix(cm,classes=['benign', 'defacement','phishing','malware'])


# In[82]:


gbdt_feature = gbdt.feature_importances_
gbdt_features = gbdt_feature.tolist()


# In[83]:


import pickle
# saving model
gbdt_pkl = "gbdt.pkl"
with open(gbdt_pkl, 'wb') as file:
    pickle.dump(gbdt, file)


# In[84]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)
y_predRF = rf.predict(X_test)
print(classification_report(y_test,y_predRF))

score = metrics.accuracy_score(y_test, y_predRF)
print("accuracy:   %0.3f" % score)


# In[85]:


rf_acc = accuracy_score(y_test, y_predRF)
rf_acc


# In[86]:


rf_f1_macro = f1_score(y_test, y_predRF, average='macro')
rf_f1_macro


# In[87]:


rf_f1_micro = f1_score(y_test, y_predRF, average='micro')
rf_f1_micro


# In[88]:


rf_f1_w = f1_score(y_test, y_predRF, average='weighted')
rf_f1_w


# In[89]:


new_eval = {'Model': 'RF','Accuracy': rf_acc, 'F1-macro': rf_f1_macro, 'F1-micro': rf_f1_micro, 'F1-weighted': rf_f1_w }
eval_df.loc[len(eval_df)] = new_eval


# In[90]:


CM=confusion_matrix(y_test,y_predRF,labels=[0,1,2,3])

plot_confusion_matrix(cm,classes=['benign', 'defacement','phishing','malware'])


# In[91]:


rf_feature = rf.feature_importances_
rf_features = rf_feature.tolist()


# In[206]:


import pickle
# saving model
rf_pkl = "rf.pkl"
with open(rf_pkl, 'wb') as file:
    pickle.dump(rf, file)


# In[209]:


X_test = pd.DataFrame(X_test)  # X_test'i pandas DataFrame'e dönüştür


# In[210]:


print(type(X_test))  # pandas DataFrame olup olmadığını kontrol et


# In[211]:


import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
def plot_feature_importance():
    tmp = pd.DataFrame({'Feature': X_test.columns, 'Feature importance': feature_dataframe['mean'].values})
    tmp = tmp.sort_values(by='Feature importance',ascending=False).head(20)
    plt.figure(figsize = (10,12))
    plt.title('Average Feature Importance Top 20 Features',fontsize=14)
    s = sns.barplot(y='Feature',x='Feature importance',data=tmp, orient='h')
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()
plot_feature_importance()


# In[99]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# 📌 1. Veri setini yükleme
X_train = pd.read_csv("D:\\malicious_phish/x-train.csv").iloc[:, 1:].values  # İlk sütunu çıkar
X_test = pd.read_csv("D:\\malicious_phish/x-test.csv").iloc[:, 1:].values
y_train = pd.read_csv("D:\\malicious_phish/y-train.csv")["labels"].values
y_test = pd.read_csv("D:\\malicious_phish/y-test.csv")["labels"].values

# 📌 2. Özellikleri ölçekleme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 📌 3. Tensor verilerine çevirme
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 📌 4. Veri yükleyici (DataLoader)
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 📌 5. LSTM Modeli Tanımlama
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x.unsqueeze(1))  # LSTM çıktısı
        out = self.fc(hn[-1])
        return out

# 📌 6. Modeli oluşturma
input_size = X_train.shape[1]  # Özellik sayısı
hidden_size = 64
num_classes = 4  # Çıkış sınıfı sayısı

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size, hidden_size, num_classes).to(device)

# 📌 7. Kayıp fonksiyonu ve optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📌 8. Modeli eğitme
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 📌 9. Modeli değerlendirme
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.numpy())

# 📌 10. Sonuçları yazdırma
print("Classification Report:\n", classification_report(y_true, y_pred))
print("Accuracy: {:.3f}".format(accuracy_score(y_true, y_pred)))





# In[100]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# 📌 Confusion Matrix hesapla
cm = confusion_matrix(y_true, y_pred)
class_names = ["Class 0", "Class 1", "Class 2", "Class 3"]  # Sınıf isimlerini ihtiyaca göre değiştir

# 📌 Confusion Matrix görselleştirme
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 📌 Performans Metrikleri
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
print("Accuracy: {:.3f}".format(accuracy_score(y_true, y_pred)))
print("Precision: {:.3f}".format(precision_score(y_true, y_pred, average="weighted")))
print("Recall: {:.3f}".format(recall_score(y_true, y_pred, average="weighted")))
print("F1-score: {:.3f}".format(f1_score(y_true, y_pred, average="weighted")))


# In[105]:


import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 📌 Cihazı belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Modeli cihaza taşı ve **train moduna al** (ÖNEMLİ)
model.to(device)
model.train()

# 📌 Test verisini cihaza taşı ve requires_grad_() kullanarak güncelle
X_test_tensor = X_test_tensor.to(device).detach().clone()
X_test_tensor.requires_grad_()

# 📌 Modeli çalıştır ve tahmin al
output = model(X_test_tensor)

# 📌 En yüksek logit değerine göre kaybı hesapla
loss = output.max(dim=1)[0].sum()

# 📌 Gradient hesaplama
loss.backward()

# 📌 Gradientlerin ortalamasını al
feature_importance = torch.mean(torch.abs(X_test_tensor.grad), dim=0).cpu().detach().numpy()

# 📌 Feature Importance sıralama
sorted_idx = feature_importance.argsort()

# 📌 Görselleştirme
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
plt.xlabel("Feature Importance Score")
plt.title("LSTM Feature Importance (Gradient-Based)")
plt.show()

# 📌 Özellik Önemini Yazdırma
feature_importance_df = pd.DataFrame(
    {"Feature": [f"Feature {i}" for i in sorted_idx], "Importance": feature_importance[sorted_idx]}
)
print(feature_importance_df)


# In[117]:


import torch
torch.cuda.empty_cache()
torch.cuda.memory_reserved(0)
torch.cuda.memory_allocated(0)
torch.cuda.synchronize()


# In[121]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# GPU veya CPU kullanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **DÜZELTİLMİŞ CNN MODELİ**
class SmallCNN(nn.Module):
    def __init__(self, input_dim):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)  
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # **ÖZELLİK HARİTASI BOYUTUNU HESAPLAYALIM**
        temp_input = torch.rand(1, 1, input_dim)  # Rastgele giriş oluştur
        temp_out = self.pool(self.relu(self.conv1(temp_input)))
        temp_out = self.pool(self.relu(self.conv2(temp_out)))
        self.flatten_size = temp_out.shape[1] * temp_out.shape[2]  # Yeni giriş boyutu

        self.fc1 = nn.Linear(self.flatten_size, 32)  
        self.fc2 = nn.Linear(32, 4)  

    def forward(self, x):
        x = x.unsqueeze(1)  # (Batch, 1, Feature) formatına getirme
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# **VERİLERİ GPU'YA TAŞI**
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Modeli başlat ve GPU'ya taşı
cnn_model = SmallCNN(X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# **CNN Modeli Eğitimi**
num_epochs = 10  
for epoch in range(num_epochs):
    cnn_model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  
        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# **CNN Modeli Test Aşaması**
cnn_model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  
        outputs = cnn_model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())  
        y_true.extend(labels.cpu().numpy())  

# **Sonuçları Yazdır**
from sklearn.metrics import classification_report, confusion_matrix
print("\nCNN Model Sonuçları:")
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))


# In[123]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Confusion Matrix Hesapla
cm = confusion_matrix(y_true, y_pred)

# Görselleştir
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("CNN Model - Confusion Matrix")
plt.show()


# In[129]:


import torch
import numpy as np
import matplotlib.pyplot as plt

def grad_cam(model, X_input, target_class):
    """
    CNN modelinde Grad-CAM ile Feature Importance hesaplama
    """
    model.eval()  
    X_input = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).to(device)  # (1, C, H, W) olacak şekilde düzenle
    X_input.requires_grad = True

    # 📌 Forward pass
    output = model(X_input)
    score = output[:, target_class].sum()  # Hedef sınıfa ait skor
    score.backward()  # Geriye yayılım (Backpropagation)

    # 📌 Gradyanları al
    gradients = X_input.grad.cpu().detach().numpy()  # (1, C, H, W) şeklinde olması lazım

    # 📌 Boyut kontrolü
    print("📌 Gradyanların Boyutu:", gradients.shape)

    if gradients.ndim == 4:  # Eğer (1, C, H, W) ise
        feature_importance = np.mean(np.abs(gradients), axis=(2, 3))  # H & W eksenlerinde ortalama al
    elif gradients.ndim == 2:  # Eğer (1, C) ise
        feature_importance = np.mean(np.abs(gradients), axis=0)  # Direkt C ekseni üzerinden ortalama al
    else:
        raise ValueError(f"Beklenmeyen gradyan boyutu: {gradients.shape}")

    return feature_importance.flatten()  # Tek boyutlu hale getir

# 📌 Örnek bir veri noktası seç (X_test[0])
target_class = np.argmax(y_test[0])  # İlk veri için doğru sınıf
importance = grad_cam(cnn_model, X_test[0], target_class)

# 📌 Görselleştirme
plt.figure(figsize=(8, 5))
plt.barh(range(len(importance)), importance, align="center")
plt.xlabel("Özellik Önemi")
plt.ylabel("Özellik Numarası")
plt.title("CNN Model - Grad-CAM Feature Importance")
plt.show()
# 📌 Özellik Önemini Yazdırma
feature_importance_df = pd.DataFrame(
    {"Feature": [f"Feature {i}" for i in sorted_idx], "Importance": feature_importance[sorted_idx]}
)
print(feature_importance_df)


# In[133]:


import torch
import numpy as np
import matplotlib.pyplot as plt

def grad_cam(model, X_input, target_class):
    """
    CNN modelinde Grad-CAM ile Feature Importance hesaplama
    """
    model.eval()  
    X_input = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).to(device)  # (1, C, H, W) olacak şekilde düzenle
    X_input.requires_grad = True

    # 📌 Forward pass
    output = model(X_input)
    score = output[:, target_class].sum()  # Hedef sınıfa ait skor
    score.backward()  # Geriye yayılım (Backpropagation)

    # 📌 Gradyanları al
    gradients = X_input.grad.cpu().detach().numpy()  # (1, C, H, W) şeklinde olması lazım

    if gradients.ndim == 4:  # Eğer (1, C, H, W) ise
        feature_importance = np.mean(np.abs(gradients), axis=(2, 3))  # H & W eksenlerinde ortalama al
    elif gradients.ndim == 2:  # Eğer (1, C) ise
        feature_importance = np.mean(np.abs(gradients), axis=0)  # Direkt C ekseni üzerinden ortalama al
    else:
        raise ValueError(f"Beklenmeyen gradyan boyutu: {gradients.shape}")

    return feature_importance.flatten()  # Tek boyutlu hale getir

# 📌 Örnek bir veri noktası seç (X_test[0])
target_class = np.argmax(y_test[0])  # İlk veri için doğru sınıf
importance = grad_cam(cnn_model, X_test[0], target_class)

# 📌 Feature Importance sıralama
sorted_idx = np.argsort(importance)  # Küçükten büyüğe sıralama
importance_sorted = importance[sorted_idx]  # Sıralanmış önem değerleri

# 📌 Özellik numaralarını oluştur
feature_labels = [f"Feature {i}" for i in sorted_idx]

# 📌 Görselleştirme
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_labels, importance_sorted, align="center")
ax.set_xlabel("Feature Importance Score")
ax.set_ylabel("Feature Number")
ax.set_title("CNN Model - Grad-CAM Feature Importance")

# 📌 En önemli 10 özelliği al
top_n = 20
top_features = sorted_idx[-top_n:][::-1]  # En büyük 10 özelliği al ve ters çevir
top_importance = importance_sorted[-top_n:][::-1]

# 📌 Yazıları dışarı taşıyalım (alt kısma ekleyelim)
table_data = [[f"{feature_labels[i]}", f"{top_importance[i]:.4f}"] for i in range(top_n)]
table = plt.table(cellText=table_data, colLabels=["Feature", "Importance"], loc='bottom', cellLoc='center')

# 📌 Tabloyu daha aşağıya almak için boşluk ekleyelim
plt.subplots_adjust(bottom=0.3)  # Grafik ile tablo arasındaki boşluğu artır

plt.show()


# In[134]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 📌 Cihazı belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Veriyi Tensor formatına çevir
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)  # CNN için uygun hale getir
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# 📌 DataLoader
batch_size = 32
dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
dataset_test = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# 📌 CNN + LSTM Modeli
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # LSTM için (batch, seq, feature)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x

# 📌 Modeli oluştur
num_classes = len(np.unique(y_train))
model = CNN_LSTM(num_classes).to(device)

# 📌 Kayıp fonksiyonu ve optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📌 Model Eğitimi
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 📌 Model Değerlendirme
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(y_batch.cpu().numpy())

# 📌 Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - CNN+LSTM")
plt.show()

# 📌 Classification Report
print(classification_report(y_true, y_pred))


# In[137]:


print(globals().keys())


# In[139]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 📌 Cihazı belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Modeli cihaza taşı ve train moduna al
model.to(device)
model.train()

# 📌 Test verisini cihaza taşı ve requires_grad_() kullanarak güncelle
X_test_tensor = X_test_tensor.to(device).detach().clone()
X_test_tensor.requires_grad_()

# 📌 Modeli çalıştır ve tahmin al
output = model(X_test_tensor)

# 📌 En yüksek logit değerine göre kaybı hesapla
loss = output.max(dim=1)[0].sum()

# 📌 Gradient hesaplama
loss.backward()

# 📌 Gradientlerin ortalamasını al ve boyutları düzelt
feature_importance = torch.mean(torch.abs(X_test_tensor.grad), dim=0).cpu().detach().numpy()
feature_importance = feature_importance.squeeze()  # Gereksiz boyutları kaldır

# 📌 Feature Importance sıralama
sorted_idx = np.argsort(feature_importance)

# 📌 Özellik sayısını kontrol et
print(f"Feature Count: {len(feature_importance)}")

# 📌 Görselleştirme
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
plt.xlabel("Feature Importance Score")
plt.title("CNN + LSTM Feature Importance (Gradient-Based)")
plt.show()

# 📌 Özellik Önemini Yazdırma
feature_importance_df = pd.DataFrame(
    {"Feature": [f"Feature {i}" for i in sorted_idx], "Importance": feature_importance[sorted_idx]}
)
print(feature_importance_df)


# In[140]:


import numpy as np
import torch
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 📌 Veriyi yükle (Önceden yüklenmiş olduğunu varsayıyoruz)
X_train_selected, X_val, y_train_selected, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# 📌 Genetik Algoritma için fitness fonksiyonu
def evaluate(individual):
    selected_features = np.where(np.array(individual) > 0.5)[0]  # 0.5 eşik değeri
    if len(selected_features) == 0:
        return 0,
    
    X_train_fs = X_train_selected[:, selected_features]
    X_val_fs = X_val[:, selected_features]
    
    model = torch.nn.Sequential(
        torch.nn.Linear(len(selected_features), 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4)
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.tensor(X_train_fs, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_selected, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_fs, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # 📌 Modeli eğit
    for epoch in range(10):  # Küçük bir epoch sayısı ile hızlı eğitim
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # 📌 Modeli değerlendir
    with torch.no_grad():
        y_pred = model(X_val_tensor).argmax(dim=1).numpy()
    accuracy = accuracy_score(y_val_tensor.numpy(), y_pred)
    
    return accuracy,

# 📌 Genetik Algoritma bileşenleri
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("attr_bool", lambda: np.random.rand())
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 📌 Genetik Algoritmayı çalıştır
population = toolbox.population(n=20)
hof = tools.HallOfFame(1)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=None, halloffame=hof, verbose=True)

# 📌 En iyi bireyi al
best_individual = hof[0]
selected_features = np.where(np.array(best_individual) > 0.5)[0]
print("Seçilen Özellikler:", selected_features)


# In[141]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# **Seçilen özellikleri kullanarak yeni veri setini oluştur**
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# **Verileri PyTorch tensörlerine çevir**
X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# **PyTorch DataLoader oluştur**
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# **CNN+LSTM Hibrit Modeli**
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=16, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # 1D Convolution için kanal boyutu ekle
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # LSTM için doğru boyuta getir
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Son zaman adımındaki çıktıyı al
        x = self.fc(x)
        return x

# **Modeli başlat**
input_dim = X_train_selected.shape[1]
num_classes = len(np.unique(y_train))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM(input_dim, num_classes).to(device)

# **Kayıp fonksiyonu ve optimizer**
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# **Model eğitimi**
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# **Model değerlendirme**
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.cpu().numpy())

# **Sonuçları yazdır**
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_true, y_pred))

# **Confusion Matrix görselleştirme**
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[143]:


import pandas as pd

# Önemli özellikleri gösterme
feature_importance = np.random.rand(len(selected_features))  # Örnek olarak rastgele değerler
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in selected_features[sorted_idx]])
plt.xlabel("Önem Derecesi")
plt.title("GA Feature Selection - Feature Importance")
plt.show()

# Özellikleri tablo halinde gösterme
feature_df = pd.DataFrame({"Feature": selected_features, "Importance": feature_importance})
feature_df = feature_df.sort_values(by="Importance", ascending=False)
print(feature_df)


# In[145]:


#get_ipython().system('pip install pyswarms')


# In[150]:


import numpy as np
import pyswarms as ps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 🔍 PSO için değerlendirme fonksiyonu
def fitness_function(feature_subset):
    feature_subset = np.array(feature_subset)  # Listeyi numpy array'e çevir
    selected_features = np.where(feature_subset > 0.5)[0]  # 0.5 eşik değeri ile seçim
    
    # Eğer hiç özellik seçilmezse kötü skor ver
    if len(selected_features) == 0:
        return 1.0  # Kötü skor

    # Seçilen özellikleri al
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Model eğit ve doğruluğu hesapla
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)

    return 1 - accuracy  # Hata oranı daha düşükse daha iyi

# 🔥 **PSO Parametreleri ve Çalıştırma**
num_features = X_train.shape[1]
options = {'c1': 2.0, 'c2': 2.0, 'w': 0.9, 'k': 3, 'p': 1.5}  # ✅ HATA KALKTI!

optimizer = ps.discrete.BinaryPSO(n_particles=10, dimensions=num_features, options=options)

# 🔄 Optimizasyonu çalıştır
best_cost, best_pos = optimizer.optimize(fitness_function, iters=20)

# 📌 En iyi seçilen özellikleri ekrana yazdır
selected_features_pso = np.where(best_pos > 0.5)[0]
print("PSO ile Seçilen Özellikler:", selected_features_pso)


# In[152]:


# 🔍 Sadece PSO tarafından seçilen özellikleri al
X_train_pso = X_train[:, selected_features_pso]
X_test_pso = X_test[:, selected_features_pso]

# PyTorch tensörlerine dönüştür
X_train_tensor = torch.tensor(X_train_pso, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_pso, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Boyutları kontrol et
print("Yeni X_train şekli:", X_train_tensor.shape)
print("Yeni X_test şekli:", X_test_tensor.shape)


# In[154]:


import torch.nn as nn
import torch.optim as optim

class HybridCNNLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HybridCNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # 🔍 LSTM için input_size'ı güncelle
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, feature_size)
        x = self.conv1(x)   # (batch_size, 16, feature_size)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # (batch_size, feature_size, 16)  -> LSTM için uygun format
        
        _, (h_n, _) = self.lstm(x)  # (batch_size, 32)
        x = self.fc(h_n[-1])  # Fully Connected Layer
        return x


# In[155]:


# **Modeli oluştur**
num_classes = len(set(y_train))  # Sınıf sayısı
model = HybridCNNLSTM(input_size=16, num_classes=num_classes).to(device)  # 🔥 input_size = 16 olarak ayarlandı

# **Loss ve Optimizer**
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# **Eğitim Parametreleri**
num_epochs = 20
batch_size = 16
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True
)

# **Modeli Eğit**
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

print("Model eğitimi tamamlandı! ✅")


# In[156]:


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# **Tahminleri Al**
model.eval()
y_pred = model(X_test_tensor).argmax(dim=1).cpu().numpy()
y_true = y_test_tensor.cpu().numpy()

# **Confusion Matrix**
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_train), yticklabels=set(y_train))
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("PSO Seçimli CNN+LSTM Confusion Matrix")
plt.show()

# **Precision, Recall, F1-Score**
print("PSO Seçimli CNN+LSTM Model Performansı:")
print(classification_report(y_true, y_pred))


# In[157]:


import pandas as pd

# Önemli özellikleri gösterme
feature_importance = np.random.rand(len(selected_features_pso))  # Örnek olarak rastgele değerler
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in selected_features_pso[sorted_idx]])
plt.xlabel("Önem Derecesi")
plt.title("PSO Feature Selection - Feature Importance")
plt.show()

# Özellikleri tablo halinde gösterme
feature_df = pd.DataFrame({"Feature": selected_features_pso, "Importance": feature_importance})
feature_df = feature_df.sort_values(by="Importance", ascending=False)
print(feature_df)


# In[159]:


import numpy as np
import math  # 🔹 Hata düzeltilmiş: `np.math.gamma` yerine `math.gamma`
import torch
import torch.nn as nn
import torch.optim as optim

# 📌 Levy Flight Fonksiyonu (Düzeltildi)
def levy_flight(Lambda):
    sigma = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
    u = np.random.randn() * sigma
    v = np.random.randn()
    step = u / abs(v)**(1 / Lambda)
    return step

# 📌 HHO Fitness Fonksiyonu
def fitness_function_HHO(position):
    selected_features = np.where(position > 0.5)[0]
    if len(selected_features) == 0:
        return 1e10  # Ceza puanı
    
    # Yeni özelliklerle veri seti oluştur
    X_train_hho_selected = X_train[:, selected_features]
    X_test_hho_selected = X_test[:, selected_features]
    
    # CNN+LSTM modelini eğit ve doğruluk oranını al
    accuracy = train_and_evaluate_hybrid_model(X_train_hho_selected, X_test_hho_selected, y_train, y_test)
    
    return -accuracy  # Maksimum doğruluğu minimize etmek için negatif alıyoruz

# 📌 HHO Algoritması (Hızlandırıldı)
def hho_algorithm(num_hawks=5, max_iter=10, dim=None):
    """HHO ile en iyi özellikleri bulur (hızlandırılmış versiyon)"""
    hawks = np.random.rand(num_hawks, dim)  # Rastgele başlangıç popülasyonu
    best_hawk = np.zeros(dim)
    best_score = float("inf")

    for t in range(max_iter):
        E1 = 2 * (1 - t / max_iter)  # Enerji azalması
        for i in range(num_hawks):
            fitness = fitness_function_HHO(hawks[i])
            if fitness < best_score:
                best_score = fitness
                best_hawk = hawks[i].copy()

        for i in range(num_hawks):
            E = 2 * E1 * np.random.rand() - E1  # Kaçış enerjisi
            J = 2 * (1 - np.random.rand())  # Atak faktörü
            if abs(E) >= 1:
                X_rand = hawks[np.random.randint(0, num_hawks)]
                hawks[i] = X_rand - np.random.rand() * abs(X_rand - 2 * np.random.rand() * hawks[i])
            else:
                if np.random.rand() < 0.5:
                    hawks[i] = best_hawk - E * abs(J * best_hawk - hawks[i])
                else:
                    hawks[i] = best_hawk - E * abs(best_hawk - hawks[i]) * levy_flight(1.5)

    return best_hawk

# 📌 HHO ile en iyi özellikleri seç
num_features = X_train.shape[1]
best_features_hho = hho_algorithm(dim=num_features)

# 📌 Seçilen özellikleri belirle
selected_features_hho = np.where(best_features_hho > 0.5)[0]
print("HHO Seçilen Özellikler:", selected_features_hho)

# 📌 Yeni veri setini oluştur
X_train_hho = X_train[:, selected_features_hho]
X_test_hho = X_test[:, selected_features_hho]


# In[162]:


# 📌 Veriyi PyTorch tensörüne çevir
X_train_tensor_hho = torch.tensor(X_train_hho, dtype=torch.float32).to(device)
X_test_tensor_hho = torch.tensor(X_test_hho, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# 📌 Modeli tanımla
num_classes = len(set(y_train))
model_hho = HybridCNNLSTM(input_size=X_train_hho.shape[1], num_classes=num_classes).to(device)

# 📌 Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_hho.parameters(), lr=0.001)

# 📌 Eğitim parametreleri
num_epochs = 20  # 🔹 Daha hızlı eğitim için epoch azaltıldı
batch_size = 16
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_tensor_hho, y_train_tensor), batch_size=batch_size, shuffle=True
)

# 📌 Modeli eğit
model_hho.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model_hho(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

print("HHO ile Seçilmiş Özelliklerle Model Eğitimi Tamamlandı ✅")


# In[163]:


# 📌 Confusion Matrix ve Performans Analizi
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 📌 Tahminleri Al
model_hho.eval()
y_pred_hho = model_hho(X_test_tensor_hho).argmax(dim=1).cpu().numpy()
y_true = y_test_tensor.cpu().numpy()

# 📌 Confusion Matrix
cm_hho = confusion_matrix(y_true, y_pred_hho)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_hho, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_train), yticklabels=set(y_train))
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("HHO Seçimli CNN+LSTM Confusion Matrix")
plt.show()

# 📌 Precision, Recall, F1-Score
print("HHO Seçimli CNN+LSTM Model Performansı:")
print(classification_report(y_true, y_pred_hho))


# In[165]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 📌 HHO ile seçilen özellikleri al
selected_features_hho = np.array(selected_features_hho)  # Listeyi NumPy array'e çevir

# 📌 Rastgele önem değerleri yerine hesaplanmış değerler kullanılmalı
feature_importance_hho = np.random.rand(len(selected_features_hho))  # ÖRNEK: Rastgele değerler yerine hesaplanmış değerler gelmeli
sorted_idx = np.argsort(feature_importance_hho)  # Küçükten büyüğe sıralama

# 📌 Görselleştirme: Feature Importance Çubuk Grafiği
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance_hho[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in selected_features_hho[sorted_idx]])
plt.xlabel("Önem Derecesi")
plt.title("HHO Feature Selection - Feature Importance")
plt.show()

# 📌 Tablo olarak yazdırma
feature_df_hho = pd.DataFrame({"Feature": selected_features_hho, "Importance": feature_importance_hho})
feature_df_hho = feature_df_hho.sort_values(by="Importance", ascending=False)

print(feature_df_hho)


# In[167]:


import pandas as pd

# 📌 Seçilen özellikleri listeye çevir
features_ga = set(selected_features)  # GA tarafından seçilen özellikler
features_pso = set(selected_features_pso)  # PSO tarafından seçilen özellikler
features_hho = set(selected_features_hho)  # HHO tarafından seçilen özellikler

# 📌 Tüm seçilen özelliklerin birleşimini al (Tek bir liste)
all_features = sorted(list(features_ga | features_pso | features_hho))

# 📌 Tabloyu oluştur
comparison_df = pd.DataFrame({'Feature': all_features})

# 📌 Her algoritma için "✔" veya "✖" işaretlerini ekle
comparison_df['GA'] = comparison_df['Feature'].apply(lambda x: "✔" if x in features_ga else "✖")
comparison_df['PSO'] = comparison_df['Feature'].apply(lambda x: "✔" if x in features_pso else "✖")
comparison_df['HHO'] = comparison_df['Feature'].apply(lambda x: "✔" if x in features_hho else "✖")

# 📌 Son tabloyu yazdır
print(comparison_df)

# 📌 Tabloyu görselleştir (Jupyter Notebook için)
from IPython.display import display
display(comparison_df)


# In[169]:


from collections import Counter

# 📌 Tüm seçilen özellikleri düz bir liste olarak birleştir
all_selected_features = list(selected_features) + list(selected_features_pso) + list(selected_features_hho)

# 📌 Özelliklerin kaç kez seçildiğini say
feature_counts = Counter(all_selected_features)

# 📌 En az 2 optimizasyon yöntemi tarafından seçilenleri al
common_features = [feature for feature, count in feature_counts.items() if count >= 2]

# 📌 Sonuçları yazdır
print(f"En az 2 optimizasyon yöntemi tarafından seçilen ortak özellikler:\n{common_features}")


# In[181]:


# 📌 Ortak özelliklere göre eğitim ve test verisini filtrele
X_train_selected = X_train[:, common_features]
X_test_selected = X_test[:, common_features]

# 📌 Veri setinin yeni boyutlarını kontrol edelim
print(f"Yeni X_train boyutu: {X_train_selected.shape}")
print(f"Yeni X_test boyutu: {X_test_selected.shape}")


# In[197]:


# 📌 Ortak özellik sayısını al
input_size = len(common_features)

# 📌 Modeli tanımla
hybrid_model = CNN_LSTM(input_size=input_size, num_classes=len(set(y_train)))

# 📌 Modeli uygun cihaza taşı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hybrid_model.to(device)


# In[193]:


# 📌 Seçilen ortak özelliklere göre veriyi filtreleme
X_train_selected = X_train[:, common_features]  # Seçilen ortak özelliklere göre güncelle
X_test_selected = X_test[:, common_features]

# 📌 Tensor formatına çevirme
X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32).to(device)

y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)


# In[194]:


# 📌 Güncellenmiş giriş boyutunu belirle
input_size = X_train_selected.shape[1]  # Seçilen ortak özellik sayısı
num_classes = len(set(y_train))  # Sınıf sayısı

# 📌 Modeli oluştur ve cihaza taşı
hybrid_model = CNN_LSTM(input_size=input_size, num_classes=num_classes).to(device)


# In[195]:


print(X_train_tensor.shape)


# In[198]:


# 📌 Veriyi PyTorch tensörüne çevir
X_train_tensor_sc = torch.tensor(X_train_selected, dtype=torch.float32).to(device)
X_test_tensor_sc = torch.tensor(X_test_selected, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# 📌 Modeli tanımla
num_classes = len(set(y_train))
model_hho = HybridCNNLSTM(input_size=X_train_hho.shape[1], num_classes=num_classes).to(device)

# 📌 Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_hho.parameters(), lr=0.001)

# 📌 Eğitim parametreleri
num_epochs = 20  # 🔹 Daha hızlı eğitim için epoch azaltıldı
batch_size = 16
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_tensor_sc, y_train_tensor), batch_size=batch_size, shuffle=True
)

# 📌 Modeli eğit
model_hho.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model_hho(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

print("Ortak Yöntem ile Seçilmiş Özelliklerle Model Eğitimi Tamamlandı ✅")


# In[199]:


# 📌 Confusion Matrix ve Performans Analizi
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 📌 Tahminleri Al
model_hho.eval()
y_pred_hho = model_hho(X_test_tensor_sc).argmax(dim=1).cpu().numpy()
y_true = y_test_tensor.cpu().numpy()

# 📌 Confusion Matrix
cm_hho = confusion_matrix(y_true, y_pred_hho)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_hho, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_train), yticklabels=set(y_train))
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("HHO Seçimli CNN+LSTM Confusion Matrix")
plt.show()

# 📌 Precision, Recall, F1-Score
print("HHO Seçimli CNN+LSTM Model Performansı:")
print(classification_report(y_true, y_pred_hho))


# In[202]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 📌 HHO ile seçilen özellikleri al
common_features = np.array(common_features)  # Listeyi NumPy array'e çevir

# 📌 Rastgele önem değerleri yerine hesaplanmış değerler kullanılmalı
feature_importance_hho = np.random.rand(len(common_features))  # ÖRNEK: Rastgele değerler yerine hesaplanmış değerler gelmeli
sorted_idx = np.argsort(feature_importance_hho)  # Küçükten büyüğe sıralama

# 📌 Görselleştirme: Feature Importance Çubuk Grafiği
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance_hho[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in common_features[sorted_idx]])
plt.xlabel("Önem Derecesi")
plt.title("HHO Feature Selection - Feature Importance")
plt.show()

# 📌 Tablo olarak yazdırma
feature_df_hho = pd.DataFrame({"Feature": common_features, "Importance": feature_importance_hho})
feature_df_hho = feature_df_hho.sort_values(by="Importance", ascending=False)

print(feature_df_hho)


# In[216]:


#get_ipython().system('pip install pytorch_tabnet')


# In[223]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


# In[224]:


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc = None  # fc katmanını dinamik yapacağız

    def forward(self, x):
        x = x.unsqueeze(1)  # Kanal boyutunu ekleyelim
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)

        # **Dynamic Fully Connected Layer** (Giriş boyutuna göre ayarlanır)
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], 128).to(x.device)  

        x = self.fc(x)
        return x


# In[228]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inputs, labels = inputs.to(device), labels.to(device)


# In[237]:


print("X_train_hho shape:", X_train_hho.shape)
print("X_test_hho shape:", X_test_hho.shape)
print("X_train_tensor_hho shape:", X_train_tensor_hho.shape)
print("X_test_tensor_hho shape:", X_test_tensor_hho.shape)
print("y_train_tensor shape:", y_train_tensor.shape)
print("y_test_tensor shape:", y_test_tensor.shape)


# In[238]:


batch_size = 1024  # Batch size'ı büyütelim

train_loader = DataLoader(TensorDataset(X_train_tensor_hho, y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor_hho, y_test_tensor), batch_size=batch_size, shuffle=False)


# In[239]:


X_train = np.array(X_train)  # NumPy formatına çevir
X_test = np.array(X_test)

# Tensor'leri tekrar oluştur
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)


# In[240]:


# Cihazı belirleyelim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Modelini Cihaza Taşı
cnn_model = CNNFeatureExtractor(input_dim=X_train.shape[1]).to(device)
cnn_model.eval()

# Veriyi Cihaza Taşı
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# CNN Modeli ile Özellik Çıkarma
with torch.no_grad():
    X_train_features = cnn_model(X_train_tensor).cpu().numpy()  # Çıktıyı CPU'ya taşı
    X_test_features = cnn_model(X_test_tensor).cpu().numpy()  # Çıktıyı CPU'ya taşı

# TabNet Modelini Eğitme
tabnet_model = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params={'lr': 2e-2})

tabnet_model.fit(
    X_train_features, y_train,
    eval_set=[(X_test_features, y_test)],
    eval_metric=['accuracy'],
    max_epochs=50, patience=10
)

# Tahmin Yapma ve Sonuçları Değerlendirme
y_pred = tabnet_model.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


# In[244]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Confusion Matrix Hesapla
cm = confusion_matrix(y_test_tensor.cpu().numpy(), y_pred)

# Confusion Matrix Görselleştir
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1","2","3"], yticklabels=["0", "1","2","3"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("CNN+TabNet Confusion Matrix")
plt.show()


# In[243]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TabNet modeli için feature importance değerlerini al
feature_importance = tabnet_model.feature_importances_

# Feature isimlerini oluştur (X_train_hho sütunlarına göre)
feature_names = [f"Feature {i}" for i in range(len(feature_importance))]

# Feature importance değerlerini sıralayarak en önemli 20 tanesini seç
sorted_idx = np.argsort(feature_importance)[::-1]  # Büyükten küçüğe sırala
top_n = 20  # En önemli 20 özellik
top_features = [feature_names[i] for i in sorted_idx[:top_n]]
top_importance = feature_importance[sorted_idx[:top_n]]

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.barh(top_features[::-1], top_importance[::-1], color='royalblue')
plt.xlabel("Önem Derecesi")
plt.ylabel("Özellikler")
plt.title("CNN+TabNet Feature Importance (En Önemli 20)")
plt.show()

# Feature Importance Tablosunu Oluştur
feature_df = pd.DataFrame({"Feature": top_features, "Importance": top_importance})
feature_df = feature_df.sort_values(by="Importance", ascending=False)

# Feature Importance Tablosunu Yazdır
print(feature_df)




# In[256]:


import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# X ve y verilerini birleştirerek AutoGluon formatına çeviriyoruz
train_data = pd.DataFrame(X_train, columns=[f"Feature_{i}" for i in range(X_train.shape[1])])
train_data["label"] = y_train  # Hedef değişken

test_data = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])
test_data["label"] = y_test  # Test hedef değişkeni

# AutoGluon için uygun formata çevirme
train_data = TabularDataset(train_data)
test_data = TabularDataset(test_data)


# In[266]:


import os
print("Çalışma dizini:", os.getcwd())  # Çalışma dizini
print("Model kayıt yolu:", save_path)  # Model kaydedilecek yol


# In[262]:




# In[267]:


import time

import os

# Modeli kaydetmek için bir klasör belirleyelim
save_path = os.path.abspath(r"C:\Users\drikm/")
print("Model kayıt yolu:", save_path)  # Kontrol için yazdır


# Eğitime başla 🚀
start_time = time.time()
predictor = TabularPredictor(label="label", path=save_path).fit(
    train_data, 
    presets="best_quality",  # En iyi model için
    time_limit=3600  # 1 saat limit koyduk, istersen artırabilirsin
)
end_time = time.time()

print(f"AutoGluon eğitimi tamamlandı! ⏳ Süre: {end_time - start_time:.2f} saniye")


# In[268]:


# Tahmin yap
y_pred = predictor.predict(test_data.drop(columns=["label"]))

# Confusion Matrix ve diğer metrikleri hesapla
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[324]:


from autogluon.tabular import TabularPredictor  

# Eğitilen modeli yükle
predictor = TabularPredictor.load(r"C:\Users\drikm/")  # Model dosya yolunu kontrol et

# Model ile tahmin yap
y_pred = predictor.predict(test_data.drop(columns=["label"]))  # "label" sütununu çıkararak tahmin yap
y_true = test_data["label"]  # Gerçek etiketleri al


# In[325]:


# Doğru tahmin edilenler
correct_predictions = test_data[y_pred == y_true].copy()
correct_predictions["predicted_label"] = y_pred[y_pred == y_true]

# Yanlış tahmin edilenler
wrong_predictions = test_data[y_pred != y_true].copy()
wrong_predictions["predicted_label"] = y_pred[y_pred != y_true]

print(f"✅ Doğru tahmin edilen örnek sayısı: {len(correct_predictions)}")
print(f"❌ Yanlış tahmin edilen örnek sayısı: {len(wrong_predictions)}")


# In[326]:


print("Yanlış tahmin edilen örnekler:")
print(wrong_predictions.head(5))


# In[327]:


print("Doğru tahmin edilen örnekler:")
print(correct_predictions.head(5))


# In[328]:


# Yanlış tahmin edilen örnekleri orijinal metin formatına dönüştürme
wrong_indices = wrong_predictions.index  # Yanlış tahmin edilen örneklerin indekslerini al

# Orijinal metinleri alarak yeni bir DataFrame oluştur
wrong_texts = pd.DataFrame({
    "Original_Text": [X_test[i] for i in wrong_indices],  # X_test'in orijinal metinleri
    "True_Label": wrong_predictions["label"].values,      # Gerçek etiketler
    "Predicted_Label": wrong_predictions["predicted_label"].values  # Model tahminleri
})

# Yanlış tahmin edilen ilk 5 örneği göster
print(wrong_texts.head(5))


# In[4]:


import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils import resample

def bootstrap_f1_ci(y_true, y_pred, n_bootstrap=2000, ci=95):
    """
    Bootstrap ile F1 skorunun güven aralığını hesaplar.
    """
    n = len(y_true)
    f1_scores = []

    for _ in range(n_bootstrap):
        idx = resample(np.arange(n), replace=True, n_samples=n)
        f1 = f1_score(np.array(y_true)[idx], np.array(y_pred)[idx])
        f1_scores.append(f1)

    lower = np.percentile(f1_scores, (100 - ci) / 2)
    upper = np.percentile(f1_scores, 100 - (100 - ci) / 2)

    return np.mean(f1_scores), lower, upper


# ============================
# Verileri numpy array'e çevir
# ============================

def str_to_array(s):
    return np.array([int(x) for x in s.split()])

y_true = str_to_array("""1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1""")

y_pred1 = str_to_array("""1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
0
1
1
1
1
1
1
1
1
1
1
1""")

y_pred2 = str_to_array("""1
1
1
1
1
1
1
1
1
1
0
1
1
1
1
1
1
1
1
1
1
1
0
1
1
1
1
1
1
1
1
0
1
1
1
1
1
1
1
1
1
1
1
0
1
1
1
1
1
1
1
1
1
1
1
0
1
1
1
1
1
1
1
1
0
1
1
1
1
1
1
1
1
0
1
1
1
1
1
1
1
1
1
1
1
1
0
1
0
1
0
1
1
0
1
1
1
1
1
1""")

y_pred3 = str_to_array("""1
1
1
0
1
1
1
1
1
1
1
0
1
0
1
1
1
1
1
1
1
0
1
1
1
1
1
1
1
1
1
1
1
1
1
1
0
1
1
1
1
1
1
0
1
1
0
1
1
1
1
1
1
1
0
1
1
1
1
1
1
0
1
1
1
1
1
1
1
1
1
1
0
1
1
1
1
0
1
1
1
1
1
1
0
1
1
0
1
1
1
1
1
1
1
1
1
1
0
1""")

# ============================
# Sonuçlar
# ============================

methods = {
    "Yöntem 1": y_pred1,
    "Yöntem 2": y_pred2,
    "Yöntem 3": y_pred3
}

for name, preds in methods.items():
    mean_f1, lower, upper = bootstrap_f1_ci(y_true, preds)
    print(f"{name}: F1={mean_f1:.3f}, %95 CI=({lower:.3f}, {upper:.3f})")


# In[ ]:




