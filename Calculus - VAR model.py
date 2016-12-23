# -*- coding: utf-8 -*-
"""
@author: Eric-He
整理后的代码
"""
import pandas as pd
import statsmodels.tsa.api as sm
import matplotlib.pyplot as plt

#参数设定区域
date_begin = '2002-07-08'
date_end = '2013-06-06'
column_1 = 'EUR'
column_2 = 'GBP'
column_3 = 'JPY'
lag = 10  # The Maximum Lag(P) of AIC and BIC
t = 10    # IRF 和 FEVD 的期數

#讀取Excel資料,建構日期數列,調整已讀取之數據
#合成以日期為Index的DataFrame  ==> data
df=pd.read_excel('D:/data/currencynew.xls')
Date=pd.date_range(date_begin,date_end)
df = df.drop(0)
df = df.drop('Date',axis = 1)
df = df.drop(column_1,axis = 1)
df = df.drop(column_2,axis = 1)
df = df.drop(column_3,axis = 1)
df = df.values
data=pd.DataFrame(df,index = Date,columns=[column_1,column_2,column_3])

# 定义函数輸出Dickey-Fuller Test 單根檢定 結果
def print_adfuller(string):
    result = sm.stattools.adfuller(data[string])
    print (string)
    print ('adf(Test statistic)', result[0])
    print ('p-value            ', result[1])
    print ('Num of lags used   ', result[2])
    print ('Num of Obs         ', result[3])
    print ('Critical Value     ', result[4])
    print ('*'*105)

# 进行Dickey-Fuller Test 單根檢定 
print ('Dickey-Fuller Test 單根檢定')
print_adfuller(column_1)
print_adfuller(column_2)
print_adfuller(column_3)

'''
#不使用函數的形式
result = sm.stattools.adfuller(data[column_1])
print (column_1)
print ('adf(Test statistic)', result[0])
print ('p-value            ', result[1])
print ('Num of lags used   ', result[2])
print ('Num of Obs         ', result[3])
print ('Critical Value     ', result[4])
print ('*'*50)
'''

#尋找最小的AIC和BIC，即最合適的P
model = sm.VAR(data)
P=model.select_order(lag)
print ('*'*60)

#輸出基於AIC之最適P的VAR結果
result = model.fit(P['aic'])
print ()
print ('The Result of Vector AutoRegression ( p =',P['aic'],')')
print (result.summary())

#Granger causality test

result.test_causality(column_1,column_2,kind='wald')
result.test_causality(column_1,column_3,kind='wald')
result.test_causality(column_1,[column_2,column_3],kind='wald')

result.test_causality(column_2,column_1,kind='wald')
result.test_causality(column_2,column_3,kind='wald')
result.test_causality(column_2,[column_1,column_3],kind='wald')

result.test_causality(column_3,column_1,kind='wald')
result.test_causality(column_3,column_2,kind='wald')
result.test_causality(column_3,[column_1,column_2],kind='wald')

#IRF - Impulse Response analysis
irf = result.irf(t)
irf.plot(orth = True)
plt.show()

#FEVD - Forecast Error Variance Decompositions
fevd = result.fevd(t)
fevd.plot()
plt.show()
print (fevd.summary())
