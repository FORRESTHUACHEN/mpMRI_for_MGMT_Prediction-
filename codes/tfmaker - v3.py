

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 14:41:38 2023

@author: chenj
"""

import os 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from itertools import chain
import numpy as np
import pandas as pd


label=pd.read_csv('D:/DatasetFromTCIA/MGMTDataset/train_labels.csv')

#Radiomics_feature=pd.read_excel('F:/DataforExtract/Radiomics_summary_normalized.xlsx')

deep_feature=pd.read_csv('D:/DatasetFromTCIA/MGMTDataset/GBM_kaggle_r2plus_deep_feature_normalized_summary.csv')
# n10l_Deep_feature=pd.read_excel('F:/DataforExtract/Deep_feature_summary_normalized.xlsx')
# ircsn_Deep_feature=pd.read_csv('F:/DataforExtract/RADCURE_ircsn_normal_deep_feature_summary.csv')
# Radiomics_feature_group=Radiomics_feature.groupby(['Patient_ID'])

# Deep_feature_group=Deep_feature.groupby(['Patient_ID'])

# Radiomics_label_unique=list(Radiomics_feature.loc[:,'Patient_ID'].unique())
#manu_label=pd.read_excel('F:/DataforExtract/Manufacturer_information_v3.xlsx')

label_list=label.iloc[:,0]

#manu_list=list(manu_label.iloc[:,0])
#HPV_list=list(label.iloc[:,0])
#label_list=set(manu_list) & set(HPV_list)
label_available=label_list

selected_deep_feature=deep_feature.iloc[:4,:]
selected_label=label[label['Patient_ID']==0]
for patient_num in range(1,1251):#len(label_available)):
    if int(deep_feature.iloc[patient_num*4,0][10:15]) in list(label_available.iloc[:]):
        selected_pa_id=int(deep_feature.iloc[patient_num*4,0][10:15])
        selected_deep_feature=pd.concat([selected_deep_feature,deep_feature.iloc[patient_num*4:patient_num*4+4,:]])
        selected_label=pd.concat([selected_label,label[label['Patient_ID']==selected_pa_id]])
        print(deep_feature.iloc[patient_num*4,0][:-6])
    


writer_root_path="D:/DatasetFromTCIA/MGMTDataset/Tf_Record_Kaggle/"

 

index=list(range(len(selected_label)))
np.random.shuffle(index)

count1=0
count2=0


# # # Radiomics_feature.to_csv('D:/Datasets/Data Recovered/DataSets/ACRIN-6698/Classifier-LSTM/combatcorrectedRadiomics.csv')
# # # Deep_feature.to_csv('D:/Datasets/Data Recovered/DataSets/ACRIN-6698/Classifier-LSTM/combatcorrectedDeepfeatures.csv')
for fold_number in range(5):
    writer1=tf.python_io.TFRecordWriter(writer_root_path+"Trainr2plusrunning3_"+str(fold_number)+".tfrecords")
    writer2=tf.python_io.TFRecordWriter(writer_root_path+"Testr2plusrunning3_"+str(fold_number)+".tfrecords")
    for i in range(len(selected_label)):
        df_value=np.array(selected_deep_feature.iloc[i*4:i*4+4,1:],dtype=np.float32)
        df_flair=df_value[0,:].tobytes()
        df_t1=df_value[1,:].tobytes()
        df_t1ce=df_value[2,:].tobytes()
        df_t2=df_value[3,:].tobytes()

        label_value=np.array(selected_label.iloc[i,1],dtype=np.int32).tobytes()

    
        example = tf.train.Example(features=tf.train.Features(feature={
           
            'DF_FLAIR': tf.train.Feature(bytes_list=tf.train.BytesList(value=[df_flair])),
            'DF_T1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[df_t1])),
            'DF_T1CE':tf.train.Feature(bytes_list=tf.train.BytesList(value=[df_t1ce])),
            'DF_T2':tf.train.Feature(bytes_list=tf.train.BytesList(value=[df_t2])),
            'Label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_value]))
            }))
    

        # writer1=tf.python_io.TFRecordWriter(writer_root_path+"Traincombat"+str(fold_num)+".tfrecords")
        # writer2=tf.python_io.TFRecordWriter(writer_root_path+"Testcombat"+str(fold_num)+".tfrecords")
        if i in index[115*fold_number:115*fold_number+115]:
            count1=count1+1
            writer2.write(example.SerializeToString())  
        #print(Radiomics_features_value.shape)
        #print(Deep_features_value.shape)
            print(label_value)
        else:
            count2=count2+1
            writer1.write(example.SerializeToString())  
            print(label_value)
    writer1.close()
    writer2.close()