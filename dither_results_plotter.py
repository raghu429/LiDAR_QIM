#!/usr/bin/env python
import sys
import numpy as np
import ntpath
import shutil
import re
import os
import glob
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from tabulate import tabulate


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data_directory = './QIM_data/decode_results/'

match_bs2_dr2_dist = re.compile(r'sigma_0-0_uniform_clean_bs[0-9]*_dr4_[0-9]*_BER.npy')

#variables: dither range, block size, sigma, tampering 
# goal is to plot the correlation and BER for clean (no noise or tampering added) point cloud for a given delta, dither range, sigma 

#the x axis would be the block sizes, y -axis would be the BER values for multiple files


data_dict = dict()
count = 0
key_count  = 1

for filename in os.listdir(data_directory):
    # print('file:',filename)
    if(match_bs2_dr2_dist.match(filename)):
        print('*******************')
        print('matched:',filename)
        
        split_file_name = filename.split('_')
        # print('split file name', split_file_name)
        
        temp_data = np.load(os.path.join(data_directory, filename))
        
        data_key = split_file_name[4]

        if(data_key in data_dict):
            data_dict[data_key].append(temp_data)
        else:
            data_dict[data_key] = [temp_data]
        # temp_data = np.load(os.path.join(data_directory, filename))[:1]
        
        print('key and value', data_key, temp_data)
        print('--------------------------')
        count += 1
            # print('count', count)

        
# print('data dict length', len(data_dict))
df = pd.DataFrame(data=data_dict)

# print(data_dict)

pdtabulate=lambda df:tabulate(df,headers='keys',tablefmt='psql')

df.to_csv(index=False)
# print(df)
print(pdtabulate(df))
# plt.figure()

# for i in data_dict:
#     print('values', i, data_dict[i])

# sns.boxplot(x="Gaussian Noise: $\sigma$ (m)", y="Bounding box distortion (m)", data= pd.melt(df, var_name = 'Gaussian Noise: $\sigma$ (m)', value_name = 'Bounding box distortion (m)'))

# sns.boxplot(x="Block size in samples: $\$ (m)", y="Bounding box distortion (m)", data= pd.melt(df, var_name = 'Uniform Noise: $\sigma$ (m)', value_name = 'Bounding box distortion (m)'))


# sns.boxplot(x="Gaussian Noise: $\sigma$ (m)", y="Bit Error Rate (%)", data= pd.melt(df, var_name = 'Gaussian Noise: $\sigma$ (m)', value_name = 'Bit Error Rate (%)'))

# sns.boxplot(x="Uniform Noise: $\sigma$ (m)", y="Bit Error Rate (%)", data= pd.melt(df, var_name = 'Uniform Noise: $\sigma$ (m)', value_name = 'Bit Error Rate (%)'))

# plt.show()





        # if(data_key == 'sigma_0'):
        #     data_key = '0.00'
        # elif(data_key == 'sigma_001'):
        #     data_key = '0.001'
        # elif(data_key == 'sigma_002'):
        #     data_key = '0.002'
        # elif(data_key == 'sigma_0025'):
        #     data_key = '0.0025'
        # elif(data_key == 'sigma_003'):
        #     data_key = '0.003'
        # elif(data_key == 'sigma_0035'):
        #     data_key = '0.0035'
        # elif(data_key == 'sigma_004'):
        #     data_key = '0.004'
        # elif(data_key == 'sigma_0045'):
        #     data_key = '0.0045'
        # elif(data_key == 'sigma_005'):
        #     data_key = '0.005'
        # elif(data_key == 'sigma_006'):
        #     data_key = '0.006'
        # elif(data_key == 'sigma_007'):
        #     data_key = '0.007'
        # elif(data_key == 'sigma_0075'):
        #     data_key = '0.0075'
        # elif(data_key == 'sigma_008'):
        #     data_key = '0.008'
        # elif(data_key == 'sigma_009'):
        #     data_key = '0.009'
        # elif(data_key == 'sigma_01'):
        #     data_key = '0.01'
        # elif(data_key == 'sigma_015'):
        #     data_key = '0.015'
        # elif(data_key == 'sigma_02'):
        #     data_key = '0.02'
        # elif(data_key == 'sigma_03'):
        #     data_key = '0.03'

        # if(data_key == '0.00' or data_key == '0.005' or data_key == '0.01' or data_key == '0.015' or data_key == '0.02' or data_key == '0.03'):
        # if(data_key == '0.001' or data_key == '0.002' or data_key == '0.0025' or data_key == '0.003' or data_key == '0.0035' or data_key == '0.004' or data_key == '0.007'):

        # if(data_key == '0.00' or data_key == '0.001' or data_key == '0.002' or data_key == '0.0025'):



# filename = 'sigma_12-1_3_uniform_addition_CORR.npy'

# if(match_UA_corr.match(filename)):
#     print('matched:',filename)
        
# # 

# sns.boxplot(x="Uniform Noise Upper-bound", y="Bit Error Rate", data= pd.melt(df, var_name = 'Uniform Noise Upper-bound', value_name = 'Bit Error Rate'))

# sns.boxplot(x="Noise Variance", y="BER %", data= pd.melt(df, var_name = 'Noise Variance', value_name = 'BER %'))


# sns.boxplot(x="Noise Variance", y="Correlation", data= pd.melt(df, var_name = 'Noise Variance', value_name = 'Correlation'))

# sns.boxplot(x="Noise Variance", y="", data= pd.melt(df, var_name = 'Noise Variance', value_name = 'Correlation'))


# data_key = find_sigma.findall(filename)[0]
# print('data_key', data_key)
# sub_string = data_key.split('_')[1]
# print('sub_string', sub_string)
# number_string = sub_string.replace('-', '.')
# print('number_string', number_string)