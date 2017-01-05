# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 09:03:49 2016

@author: Eric-He
"""
# Import Package Area
##################################################
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from arch.univariate import arch_model

# Variable Setting Area
###################################################
n = 20 #ARCH(m) the max lag in finding the min AIC
       #n為ARCH(m)中m要檢測的最大值
column_1 = 'vwretd'

df=pd.read_csv('D:/data/spreturn.csv')

# Creat Data Area
##################################################
#date = pd.DataFrame({'year':df['year'],'month':df['month'],'day':df['day']})
date = pd.DataFrame(df[['year','month','day']])
date = pd.to_datetime(date)
value = df[column_1].values
data = pd.DataFrame(value, index = date, columns = [column_1])

# Find the lag(m) of the min AIC
##################################################
m = 0 
aic=0
#aa=np.zeros(n+1)
for i in range(1,n):
    am = arch_model(data,vol='ARCH', p=i)
    result = am.fit(update_freq=5)
    a = result.aic
#    aa[i]= result.aic
    if a < aic:
        m = i
        aic = result.aic 
#        aa[i]= result.aic
#print (aa)

# ARCH（1）
##################################################
r_arch = arch_model(data[column_1],vol='ARCH',p=m)
result_arch = r_arch.fit(update_freq=5)
#print (result_arch.summary())        
        
# GARCH(1,1)  ==> prediction of conditional vilatility
##################################################
r_garch = arch_model(data[column_1],p=1,q=1) #vol默認值為'GARCH'
result_garch = r_garch.fit(update_freq=5)
#print (result_garch.summary())     

#畫圖
##################################################
#result_garch.plot()

# Output
##################################################
def pp(pr_n):
    for i in range(pr_n):
        print ()
    print ('*'*80)

pp(30)
print ('         Min AIC: ',aic)
print ('The m of Min AIC: ',m)
pp(5)
print (result_arch.summary())
pp(5)
print (result_garch.summary())     
pp(5)
result_garch.plot()











