'''
Created on 11 de set de 2017

@author: alisson
'''

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from sklearn import preprocessing
#from statsmodels.tsa.x13 import x13_arima_select_order


def load_files(file_name):

    print("Load File " + file_name)
    serie = np.loadtxt('./data/'+file_name+'.txt')
    dtf_serie = pd.DataFrame(serie)
    df = dtf_serie.values
    
    return df

def Save(real, pred):
    
    #plt.plot(real, "Blue")
    #plt.plot(pred, "Green")
    #plt.show()
    
    matrix = np.zeros((len(real), 2))
    
    for i in range(len(real)):        
        matrix[i] = [real[i], pred[i]]
        
    return matrix
    
def train_test_arima(df, file_name):
    
    erros = np.zeros((1,4))
    len_serie = len(df)
    scaler = preprocessing.MinMaxScaler()
    df = scaler.fit_transform(df)
    
    train_len = round(len_serie * 0.6)
    test_len = round(len_serie * 0.4)
    
    train =  df[0:train_len]
    test = df[train_len+1:len(df)]
            
    #order = x13_arima_select_order(train, maxorder=(2, 1), maxdiff=(2, 1))
    order = (0,0,1)
    
    model = ARIMA(train, order=order)
    results_AR = model.fit(disp=-1)
    pred_train = results_AR.fittedvalues
        
    df_train = pd.DataFrame(data=Save(train, pred_train), columns=["Train", "Pred"])
    train_name = file_name+'-train.csv'
    df_train.to_csv('data/'+train_name)

    mse_train = mean_squared_error(train, pred_train)
    rmse_train = np.sqrt(mse_train)

    model = ARIMA(test, order=order)
    results_AR = model.fit(disp=-1)
    pred_test = results_AR.fittedvalues
    
    df_test = pd.DataFrame(data=Save(test, pred_test), columns=["Test", "Pred"])
    test_name = file_name+'-test.csv'
    df_test.to_csv('data/'+test_name)

    mse_test = mean_squared_error(test, pred_test)
    rmse_test = np.sqrt(mse_test)
    
    erros[0] = [mse_train, rmse_train, mse_test, rmse_test]
    
    
    return erros
    
files = ["NN3-001","NN3-002", "NN3-003", "NN3-004", "NN3-005", "NN3-006", "NN3-007", "NN3-008", "NN3-009", "NN3-010",
          "NN3-011", "NN3-012", "NN3-013", "NN3-014", "NN3-015", "NN3-016", "NN3-017", "NN3-018", "NN3-019", "NN3-020"]
          
for i in files:
    df = load_files(i)
    res = train_test_arima(df, i)
    print(res)
    df_erros = pd.DataFrame(data = res, columns=['mse_train', 'rmse_train', 'mse_test', 'rmse_test'])
    #df_erros = pd.DataFrame(data = res, columns=["mse_train"])
    name = i+"-errors.csv"
    df_erros.to_csv('data/'+name)
    