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

# data_directory = './QIM_data/decode_results_0-05/'
# data_directory = './QIM_data/decode_results_0-1/'
# data_directory = './QIM_data/decode_results_0-2/'
# data_directory = './QIM_data/decode_results_0-3/'
# data_directory = './QIM_data/decode_results_0-35/'
# data_directory = './QIM_data/decode_results_0-4/'

# data_directory = './QIM_data/decode_results/'

# data_directory = './QIM_data/decode_results_1_0-1/'
data_directory = './QIM_data/decode_results_2_0-1/'


# match_GA_ber = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_addition_BER.npy')
# match_GA_corr = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_addition_CORR.npy')
# match_GA_vol = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_addition_VOL.npy')
# match_GA_confm = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_addition_CONFM.npy')
# match_GA_dist = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_addition_DIST.npy')

# match_UA_ber = re.compile(r'sigma_[0-9_-]*_[3]_uniform_addition_BER.npy')
# match_UA_corr = re.compile(r'sigma_[0-9_-]*_[3]_uniform_addition_CORR.npy')
# match_UA_vol = re.compile(r'sigma_[0-9_-]*_[3]_uniform_addition_VOL.npy')
# match_UA_confm = re.compile(r'sigma_[0-9_-]*_[3]_uniform_addition_CONFM.npy')
# match_UA_dist = re.compile(r'sigma_[0-9_-]*_[3]_uniform_addition_DIST.npy')

# match_GD_ber = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_deletion_BER.npy')
# match_GD_corr = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_deletion_CORR.npy')
# match_GD_vol = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_deletion_VOL.npy')
# match_GD_confm = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_deletion_CONFM.npy')
# match_GD_dist = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_deletion_DIST.npy')

# match_UC_ber = re.compile(r'sigma_[0-9_-]*_[3]_uniform_clean_BER.npy')
# match_UC_corr = re.compile(r'sigma_[0-9_-]*_[3]_uniform_clean_CORR.npy')
# match_UC_vol = re.compile(r'sigma_[0-9_-]*_[3]_uniform_clean_VOL.npy')
# match_UC_confm = re.compile(r'sigma_[0-9_-]*_[3]_uniform_clean_CONFM.npy')
# match_UC_dist = re.compile(r'sigma_[0-9_-]*_[3]_uniform_clean_DIST.npy')

# match_GC_ber = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_clean_BER.npy')
# match_GC_corr = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_clean_CORR.npy')
# match_GC_vol = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_clean_VOL.npy')
# match_GC_confm = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_clean_CONFM.npy')
# match_GC_dist = re.compile(r'sigma_[0-9_-]*_[3]_gaussian_clean_DIST.npy')

# match_UD_ber = re.compile(r'sigma_[0-9_-]*_[3]_uniform_deletion_BER.npy')
# match_UD_corr = re.compile(r'sigma_[0-9_-]*_[3]_uniform_deletion_CORR.npy')
# match_UD_vol = re.compile(r'sigma_[0-9_-]*_[3]_uniform_deletion_VOL.npy')
# match_UD_confm = re.compile(r'sigma_[0-9_-]*_[3]_uniform_deletion_CONFM.npy')
# match_UD_dist = re.compile(r'sigma_[0-9_-]*_[3]_uniform_deletion_DIST.npy')



match_GA_ber = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_addition_BER.npy')
match_GA_corr = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_addition_CORR.npy')
match_GA_vol = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_addition_VOL.npy')
match_GA_confm = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_addition_CONFM.npy')
match_GA_dist = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_addition_DIST.npy')

match_UA_ber = re.compile(r'sigma_[0-9_-]*_[2]_uniform_addition_BER.npy')
match_UA_corr = re.compile(r'sigma_[0-9_-]*_[2]_uniform_addition_CORR.npy')
match_UA_vol = re.compile(r'sigma_[0-9_-]*_[2]_uniform_addition_VOL.npy')
match_UA_confm = re.compile(r'sigma_[0-9_-]*_[2]_uniform_addition_CONFM.npy')
match_UA_dist = re.compile(r'sigma_[0-9_-]*_[2]_uniform_addition_DIST.npy')

match_GD_ber = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_deletion_BER.npy')
match_GD_corr = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_deletion_CORR.npy')
match_GD_vol = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_deletion_VOL.npy')
match_GD_confm = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_deletion_CONFM.npy')
match_GD_dist = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_deletion_DIST.npy')

match_UC_ber = re.compile(r'sigma_[0-9_-]*_[2]_uniform_clean_BER.npy')
match_UC_corr = re.compile(r'sigma_[0-9_-]*_[2]_uniform_clean_CORR.npy')
match_UC_vol = re.compile(r'sigma_[0-9_-]*_[2]_uniform_clean_VOL.npy')
match_UC_confm = re.compile(r'sigma_[0-9_-]*_[2]_uniform_clean_CONFM.npy')
match_UC_dist = re.compile(r'sigma_[0-9_-]*_[2]_uniform_clean_DIST.npy')

match_GC_ber = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_clean_BER.npy')
match_GC_corr = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_clean_CORR.npy')
match_GC_vol = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_clean_VOL.npy')
match_GC_confm = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_clean_CONFM.npy')
match_GC_dist = re.compile(r'sigma_[0-9_-]*_[2]_gaussian_clean_DIST.npy')

match_UD_ber = re.compile(r'sigma_[0-9_-]*_[2]_uniform_deletion_BER.npy')
match_UD_corr = re.compile(r'sigma_[0-9_-]*_[2]_uniform_deletion_CORR.npy')
match_UD_vol = re.compile(r'sigma_[0-9_-]*_[2]_uniform_deletion_VOL.npy')
match_UD_confm = re.compile(r'sigma_[0-9_-]*_[2]_uniform_deletion_CONFM.npy')
match_UD_dist = re.compile(r'sigma_[0-9_-]*_[2]_uniform_deletion_DIST.npy')





# match_GA_ber = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_addition_BER.npy')
# match_GA_corr = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_addition_CORR.npy')
# match_GA_vol = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_addition_VOL.npy')
# match_GA_confm = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_addition_CONFM.npy')
# match_GA_dist = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_addition_DIST.npy')

# match_UA_ber = re.compile(r'sigma_[0-9_-]*_[1]_uniform_addition_BER.npy')
# match_UA_corr = re.compile(r'sigma_[0-9_-]*_[1]_uniform_addition_CORR.npy')
# match_UA_vol = re.compile(r'sigma_[0-9_-]*_[1]_uniform_addition_VOL.npy')
# match_UA_confm = re.compile(r'sigma_[0-9_-]*_[1]_uniform_addition_CONFM.npy')
# match_UA_dist = re.compile(r'sigma_[0-9_-]*_[1]_uniform_addition_DIST.npy')

# match_GD_ber = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_deletion_BER.npy')
# match_GD_corr = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_deletion_CORR.npy')
# match_GD_vol = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_deletion_VOL.npy')
# match_GD_confm = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_deletion_CONFM.npy')
# match_GD_dist = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_deletion_DIST.npy')

# match_UC_ber = re.compile(r'sigma_[0-9_-]*_[1]_uniform_clean_BER.npy')
# match_UC_corr = re.compile(r'sigma_[0-9_-]*_[1]_uniform_clean_CORR.npy')
# match_UC_vol = re.compile(r'sigma_[0-9_-]*_[1]_uniform_clean_VOL.npy')
# match_UC_confm = re.compile(r'sigma_[0-9_-]*_[1]_uniform_clean_CONFM.npy')
# match_UC_dist = re.compile(r'sigma_[0-9_-]*_[1]_uniform_clean_DIST.npy')

# match_GC_ber = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_clean_BER.npy')
# match_GC_corr = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_clean_CORR.npy')
# match_GC_vol = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_clean_VOL.npy')
# match_GC_confm = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_clean_CONFM.npy')
# match_GC_dist = re.compile(r'sigma_[0-9_-]*_[1]_gaussian_clean_DIST.npy')

# match_UD_ber = re.compile(r'sigma_[0-9_-]*_[1]_uniform_deletion_BER.npy')
# match_UD_corr = re.compile(r'sigma_[0-9_-]*_[1]_uniform_deletion_CORR.npy')
# match_UD_vol = re.compile(r'sigma_[0-9_-]*_[1]_uniform_deletion_VOL.npy')
# match_UD_confm = re.compile(r'sigma_[0-9_-]*_[1]_uniform_deletion_CONFM.npy')
# match_UD_dist = re.compile(r'sigma_[0-9_-]*_[1]_uniform_deletion_DIST.npy')




find_sigma =  re.compile(r'sigma_[0-9_-]*')
# find_hue =  re.compile(r'sigma_[0-9]*_[1 2 3]')

data_dict = {}
count = 0
key_count  = 1

for filename in os.listdir(data_directory):
    if(match_UA_dist.match(filename)):
        # print('matched:',filename)
        
        key_string = find_sigma.findall(filename)[0]
        # print('key_string', key_string)
        sub_string = key_string.split('_')[1]
        # print('sub_string', sub_string)
        data_key = sub_string.replace('-', '.')
        # print('data_key', data_key)
        # print('sigma value', data_key)
        # print('hue value', hue_value)

        # temp_data = np.load(os.path.join(data_directory, filename))[:1]
        temp_data = np.load(os.path.join(data_directory, filename))[:]
        # print('temp data shape and size', temp_data.shape)
            
        # if(data_key == '0.0' or data_key == '0.018' or data_key == '0.024' or data_key == '0.007'): #or data_key == '0.014' ):

        # print(temp_data.reshape(-1))
        data_dict[data_key] = temp_data.reshape(-1)
        count += 1
            # print('count', count)

        
# print('data dict length', len(data_dict))
df = pd.DataFrame(data=data_dict)

print(df)

# plt.figure()

# for i in data_dict:
#     print('values', i, data_dict[i])

# sns.boxplot(x="Gaussian Noise: $\sigma$ (m)", y="Bounding box distortion (m)", data= pd.melt(df, var_name = 'Gaussian Noise: $\sigma$ (m)', value_name = 'Bounding box distortion (m)'))

sns.boxplot(x="Uniform Noise: $\sigma$ (m)", y="Bounding box distortion (m)", data= pd.melt(df, var_name = 'Uniform Noise: $\sigma$ (m)', value_name = 'Bounding box distortion (m)'))


# sns.boxplot(x="Gaussian Noise: $\sigma$ (m)", y="Bit Error Rate (%)", data= pd.melt(df, var_name = 'Gaussian Noise: $\sigma$ (m)', value_name = 'Bit Error Rate (%)'))

# sns.boxplot(x="Uniform Noise: $\sigma$ (m)", y="Bit Error Rate (%)", data= pd.melt(df, var_name = 'Uniform Noise: $\sigma$ (m)', value_name = 'Bit Error Rate (%)'))

plt.show()





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