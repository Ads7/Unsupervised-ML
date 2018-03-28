import pandas  as pd
import sns as sns
from scipy.linalg.tests.test_fblas import accuracy
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
data_train = pd.read_csv('/Users/aman/workspace/DataMining/d17428d0-e-Criminal/criminal_train.csv')
data_test = pd.read_csv('/Users/aman/workspace/DataMining/d17428d0-e-Criminal/criminal_test.csv')

data_train = data_train.dropna()

ids = data_train.pop('PERID')
test_ids = data_test.pop('PERID')
labels = data_train.pop('Criminal')

# normalisation
data_train = StandardScaler().fit_transform(data_train)

# X_train,y_train = data_train,labels
X_train, X_test, y_train, y_test = train_test_split(data_train, labels, test_size=0.4, random_state=0)

# X_test=data_test
# feature reduction

pca = PCA(n_components=30)
pca.fit(X_train, y_train)
clf =  LogisticRegression()
# data_train = pca.transform(X_train)
# training
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
# check
resp = {'Criminal':pred,'PERID':test_ids}

# df = pd.DataFrame(data=resp)
# df = df[['PERID','Criminal']]
# df.to_csv('/Users/aman/workspace/DataMining/d17428d0-e-Criminal/criminal_test_resp.csv',index=False)
print(matthews_corrcoef(y_test, pred))
print(accuracy_score(y_test, pred))
