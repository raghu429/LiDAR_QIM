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

# data_directory = './QIM_data/decode_results_full_Jan5/delta_5_100_full'
data_directory = './QIM_data/decode_results/delta_5_100_half_results/'

delta = 0.05
# I = pd.Index([delta/2.0, delta/3.0, delta/4.0, delta/8.0], name="rows")

file_name  =  data_directory + '/BER_DRall_BSall_00144Sig_delta005.csv'

dataframe = pd.read_csv(file_name)
print(dataframe)

# df = pd.DataFrame(dataframe, index=I)
# dataframe.set_index('DR')
ax = dataframe.plot(x='Dither Range', grid=1, kind='bar', legend = 1, fontsize = 20)
ax.set_ylabel("Bit Error Rate", fontsize = 20)
# ax.set_ylabel("False Negatives", fontsize = 20)
# ax.set_ylabel("Distortion", fontsize = 20)

ax.set_xlabel("Dither Range", fontsize = 20)
# ax.legend(loc='upper center')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

plt.show()














































# # match_string = re.compile(r'sigma_0-0_uniform_clean_bs[0-9]*_dr2_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-0_uniform_clean_bs[0-9]*_dr3_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-0_uniform_clean_bs[0-9]*_dr4_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-0_uniform_clean_bs[0-9]*_dr8_[0-9]*_BER.npy')


# # match_string = re.compile(r'sigma_0-0072_uniform_add_bs[0-9]*_dr3_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-1_uniform_clean_bs[0-9]*_dr3_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-1_uniform_clean_bs[0-9]*_dr4_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-1_uniform_clean_bs[0-9]*_dr8_[0-9]*_BER.npy')

# # match_string = re.compile(r'sigma_0-05_uniform_clean_bs[0-9]*_dr2_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-05_uniform_clean_bs[0-9]*_dr3_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-05_uniform_clean_bs[0-9]*_dr4_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-05_uniform_clean_bs[0-9]*_dr8_[0-9]*_BER.npy')

# # match_string = re.compile(r'sigma_0-025_uniform_clean_bs[0-9]*_dr2_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-025_uniform_clean_bs[0-9]*_dr3_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-025_uniform_clean_bs[0-9]*_dr4_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-025_uniform_clean_bs[0-9]*_dr8_[0-9]*_BER.npy')

# # match_string = re.compile(r'sigma_0-0125_uniform_clean_bs[0-9]*_dr2_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-0125_uniform_clean_bs[0-9]*_dr3_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-0125_uniform_clean_bs[0-9]*_dr4_[0-9]*_BER.npy')
# # match_string = re.compile(r'sigma_0-0125_uniform_clean_bs[0-9]*_dr8_[0-9]*_BER.npy')



# ##variables: dither range, block size, sigma, tampering 
# ## goal is to plot the correlation and BER for clean (no noise or tampering added) point cloud for a given delta, dither range, sigma 

# ## the x axis would be the block sizes, y -axis would be the BER values for multiple files

# #### This line selects the files from the directory that satisfy the criteria mentioend below
# ####
# ###
# # DR = 2, all BS, sigma = 0072, delta = 0.05, BER 
# match_string = re.compile(r'sigma_0-0072_uniform_add_bs[0-9]*_dr8_[0-9]*_FN.npy')

# data_dict = dict()
# count = 0
# key_count  = 1
# key_list=[]
# split_file_name =[]

# for filename in os.listdir(data_directory):
#     # print('file:',filename)
#     if(match_string.match(filename)):
#         # print('*******************')
#         # print('matched:',filename)
        
#         split_file_name = filename.split('_')
#         # print('split file name', split_file_name)
        
#         temp_data = np.load(os.path.join(data_directory, filename))
        
#         data_key = split_file_name[4]

#         if(data_key in data_dict):
#             data_dict[data_key].append(temp_data)
#         else:
#             data_dict[data_key] = [temp_data]
#             key_list.append(data_key)
#         # temp_data = np.load(os.path.join(data_directory, filename))[:1]
        
#         # print('key and value', data_key, temp_data)
#         # print('--------------------------')
#         count += 1
#             # print('count', count)

        
# # print('data dict length', len(data_dict))
# df = pd.DataFrame(data=data_dict)
# columnsTitles = ['bs2', 'bs4', 'bs8', 'bs16', 'bs32', 'bs64', 'bs128', 'bs256', 'bs512', 'bs1024']
# # df = df.reindex(columns=columnsTitles)

# ## if you want to ignore all the headers and indices  
# # df.to_csv('my_csv.csv', mode='a', header=False, index=False)

# print(split_file_name)
# # df[columnsTitles].astype(float).describe().loc()[['mean']].to_csv('my_csv.csv', mode='a', header=False, index=False)

# # df[key_list].astype(float).describe().to_csv('my_csv.csv', mode='a', head=False, index=False)

# # print(df[key_list].astype(float).describe().loc()[['mean']])

# ## since this file gets appended, its rows represent the DR values starting from higher value or delta/2 and end ing with delta/8 .. each file should have five rowsthe values are [2,3,4,6,8]. Change the file name once you are done running this script for five times with all the DR values.
# results_file = './QIM_data/decode_results/delta_5_100_half_results/CORR_DRall_BSall_0072Sig_delta005.csv' 


# print(df[columnsTitles].astype(float).describe().loc()[['mean']])

# ###This line prints the mean value to the csv file name and location mentioned in the result file
# ###

# df[columnsTitles].astype(float).describe().loc()[['mean']].to_csv(results_file, mode='a', header=False, index=False)


# ##visualization of the values in nicve tabular form
# pdtabulate=lambda df:tabulate(df,headers='keys',tablefmt='psql')

# # print(pdtabulate(df))
