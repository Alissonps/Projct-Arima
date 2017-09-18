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

def Plot(real, pred):
    
    plt.plot(real, "Blue")
    plt.plot(pred, "Green")
    plt.show()

def Save(real, pred):
    
    #plt.plot(real, "Blue")
    #plt.plot(pred, "Green")
    #plt.show()
    print(len(real))
    print(len(pred))
    
    matrix = np.zeros((len(pred), 2))
    
    
    for i in range(len(pred)):        
        matrix[i] = [real[i], pred[i]]
        
    return matrix

def Load_order(file_name):
    
    dtf_serie = pd.read_csv('./ARIMA/ordens/'+file_name, usecols=[0,1,2])
    order = dtf_serie.values
    order = order.astype('int')
    order[0][0] = 1
    print("Ordem do modelo: " , order)
    
    return order
    
def train_test_arima(df, file_name):
    
    modelo = np.zeros((1, 3))
    erros = np.zeros((1,6))
    len_serie = len(df)
    scaler = preprocessing.MinMaxScaler()
    df = scaler.fit_transform(df)
    
    train_len = round(len_serie * 0.6)
    val_len = round(len_serie * 0.2)
    test_len = round(len_serie * 0.2)
    
    train =  df[0:train_len]
    val = df[train_len+1:len(df)-test_len]
    test = df[train_len+val_len+1:len(df)]
            
    ordem = Load_order(file_name)
    order = (ordem[0][0], ordem[0][1], ordem[0][2])
    #order = evaluate_models(train, val, p_values, d_values, q_values)
    
    modelo[0] = [order[0], order[1], order[2]]

    df_modelo = pd.DataFrame(data = modelo, columns=['p', 'd', 'q'])
    df_modelo.to_csv('ARIMA/ordens/'+file_name, header=True, index= False)


    lags = 5
    #-----------------Treinamento----------------------------------
    pred_train = []

    model = ARIMA(train, order=order)
    model_fit = model.fit(disp=-1)
    pred_train = model_fit.fittedvalues
        
    df_train = pd.DataFrame(data=Save(train, pred_train))
    df_train.to_csv('ARIMA/treinamento/'+file_name, header=False, index= False)

    mse_train = mean_squared_error(train, pred_train)
    rmse_train = np.sqrt(mse_train)

    #-----------------Validação----------------------------------
    pred_val = []
    
    pred_val = model_fit.predict(end=val_len-2, exog=val) 

    #Plot(val, pred_val)
           
    df_val = pd.DataFrame(data=Save(val, pred_val))
    df_val.to_csv('ARIMA/validação/'+file_name, header=False, index= False)

    mse_val = mean_squared_error(val, pred_val)
    rmse_val = np.sqrt(mse_val)
    
    #-----------------Teste----------------------------------
    pred_test = []

    pred_test = model_fit.predict(end=test_len-2, exog=test)
    
    #Plot(test, pred_test)
    
    df_test = pd.DataFrame(data=Save(test, pred_test))
    df_test.to_csv('ARIMA/teste/'+file_name, header=False, index= False)

    mse_test = mean_squared_error(test, pred_test)
    rmse_test = np.sqrt(mse_test)

    erros[0] = [mse_train, rmse_train, mse_val, rmse_val, mse_test, rmse_test]
    
    return erros

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(train_x, val_x, arima_order):
    # prepare training dataset
    train_size = len(train_x)
    train = train_x
    test = val_x
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=-1)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(train_x, val_x, p_values, d_values, q_values):

    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(train_x, val_x, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

    return best_cfg
    
#files = ["NN3-001", "NN3-007", "NN3-008", "NN3-009", "NN3-010", "NN3-012", "NN3-013", "NN3-018", "NN3-019", "NN3-020"]          

files = ["NN3-002", "NN3-004", "NN3-005", "NN3-006", "NN3-011", "NN3-014", "NN3-015", "NN3-016", "NN3-017"]          


          
for i in files:
    df = load_files(i)
    file = i +'.csv'
    res = train_test_arima(df, file)
    print(res)
    df_erros = pd.DataFrame(data = res, columns=['MSE_train', 'RMSE_train', 'MSE_val', 'RMSE_val', 'MSE_test', 'RMSE_test'])
    #df_erros = pd.DataFrame(data = res, columns=["mse_train"])
    df_erros.to_csv('ARIMA/erros/'+file, header=True, index= False)
    
