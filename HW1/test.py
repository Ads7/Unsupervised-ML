
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_data = pd.read_csv('/Users/aman/Workspace/DataMining/file_data.csv', header = None, names=["index","title","author","date", "veneue" ],engine='python')
ref_data = pd.read_csv('/Users/aman/Workspace/DataMining/AP_ref.txt', header = None, names=["index","ref"],engine='python')


impact_dict={}
df1 = ref_data.groupby('ref').count().reset_index()
df1.columns = ['index','ref']
grouped = file_data.groupby('veneue')
impact_list = dict()
for key in grouped.groups.keys()[1:]:
    indexes = grouped.get_group(key)['index'].tolist()
    impact_list[key] =  df1[df1['index'].isin(indexes)]['ref'].sum()/len(indexes)

import json

with open('data1.json', 'w') as fp:
    json.dump(impact_list, fp)
print "index selected"
# for i in file_data['veneue'].unique()[1:]:
#     index = file_data[file_data['veneue']==i]['index']
#     index_list.append(index.tolist())
# print "index selected"
# for i in index_list:
#     len_ = len(i)
#     if len_:
#         cite = len(ref_data[ref_data['ref'].isin(i)])/len_
#         if cite:
#             impact.append(cite)