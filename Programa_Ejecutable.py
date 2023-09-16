#!/usr/bin/env python
# coding: utf-8

# ## **Carga de Datos de la Locacion y Limpieza preliminar de Datos**

# In[1]:


import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import math
import random
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)
import gc
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import *
import missingno as msno
import sklearn.neighbors._base
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from missingpy import MissForest

#Datos de Ingreso para el programa
#Input1= 'Estacion01'
#Input2= 2019
#Input3= 2020
print("")
print("Buenos dias, por favor ingrese de los siguientes datos para la prediccion")
print("Asegurese de que los registros de monitoreo estes guardados en la carpeta local")
print("")

NombreEstaciones=("Estacion01","Estacion02","Estacion03","Estacion04","Estacion05","Estacion06","Estacion07","Estacion08",
                  "Estacion09","Estacion10","Estacion11","Estacion12","Estacion13")

YearMonitoreo=(2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020)

while1 = 1
while2 = 1
print("")
print("1.Por favor ingrese los siguientes datos:")
print("")
while while1 != -1:
    print ("-Ingrese el nombre de la estacion")
    Input1 = (input()) #Pedimos numero al usuario
    print("")
    if Input1 in NombreEstaciones:
        while1=-1
    else:
        print ("Por favor ingrese el nombre correcto")
        print("")
        while1=0

while while2 != -1:
    print ("-Ingrese el año inicial de las mediciones")
    Input2 = int(input()) #Pedimos numero al usuario
    print("")
    print("-Ingrese el año final de las mediciones")
    Input3 = int(input())  # Pedimos numero al usuario

    if Input2>Input3:
        while2 = 1
        print("Por favor ingrese correctamente los datos")
        print("")
    elif (Input2 in YearMonitoreo) and (Input3 in YearMonitoreo):
        print("")
        while2=-1
    else:
        while2=1
        print("Por favor ingrese correctamente los datos")

####Registro de eventos
while3=1
lsteventos = []
while while3!= -1:

    print("2.Registre los eventos historicos de deslizamientos")
    print("Deberá cumplir con la sintaxis YYYY-MM-DD")
    print("")
    n = int(input("-Ingrese la cantidad de eventos : "))
    print("")
    if n==0:
        print("No se registraron eventos")
        print("Iniciando procesamiento de datos")
        print(" ")
        while3 = -1
    elif n>0:
        for i in range(0, n):
            print("-Ingrese el registro N°",i+1 )
            print("")
            ele = input()
            print("")
            # adding the element
            lsteventos.append(ele)
        print("Las fechas de eventos ingresados son:",lsteventos)
        print("")
        print("Iniciando procesamiento de datos")
        print("--------------------------------")
        while3 = -1
    elif n<0:
        print("")
        print("Digite vuelva a digitar correctamente la infomormacion")


def DataEntry(locacion,ano_inicio,ano_fin):
    #Concatenacion de datos
    Locacion=str(locacion)
    x=ano_inicio
    y=ano_fin
    location_data=pd.DataFrame()
    #path='/Location Data/' (Path en caso se use google collab)
    path='C:/Programa Prediccion/Universidad internacional de Valencia/Location Data/'
    new_columns=['Fecha', 'Hora', 'Temperatura', 'Velocidad', 'Direccion',
          'Direccion_rosa', 'Presion', 'Humedad', 'Precipitacion',
          'Precipitacion_Real', 'Radiacion', 'Evapotranspiracion',
          'Evapo_real', 'Puerta_Abierta', 'Bateria_Baja',
          'Solar_Energia', 'Observaciones']
    for i in range(x,y+1):
        # Con la variable C se coloca el nombre del dataframe
        input1=Locacion+'_'
        c = input1 + str(i)
        # La siguiente linea genera la ruta del csv
        input2 = path+Locacion+'/'
        input3 = str(i) + '_' + Locacion + '.csv'
        v1=input2+input3
        # Generamos los dataframes panda
        data = pd.read_csv(v1,encoding="utf-8", header=2,sep=';')
        old_columns=data.columns
        data.rename(columns=dict(zip(old_columns, new_columns)), inplace=True)
        data.columns=data.columns.str.strip()
        location_data=location_data.append(data,ignore_index=True)

    #Inicio de Limpieza de Datos
    location_data = location_data.replace("-", np.nan)
    location_data = location_data.replace(',','', regex=True)
    location_data.Precipitacion_Real = location_data.Precipitacion_Real.replace('aquí', 0)
    location_data.Precipitacion_Real = location_data.Precipitacion_Real.replace(' Precipita', 0)
    location_data=location_data.replace(".", np.nan)

    #Adecuamos el formato de los datos
    location_data["Fecha"] = pd.to_datetime(location_data["Fecha"])
    ##location_data["Hora"] =  pd.to_datetime(location_data["Hora"]).dt.time
    ##location_data["Direccion"] = location_data["Direccion"].fillna(-91)
    ##location_data["Direccion"] = location_data["Direccion"].astype(int)
    ##location_data["Humedad"] = location_data["Humedad"].fillna(-91)
    ##location_data["Humedad"] = location_data["Humedad"].astype(int)
    ##location_data["Radiacion"] = location_data["Radiacion"].fillna(-91)
    location_data["Radiacion"] = location_data["Radiacion"].astype(float)
    location_data["Precipitacion"] = location_data["Precipitacion"].astype(float)
    location_data["Precipitacion_Real"] = location_data["Precipitacion_Real"].astype(float)
    location_data["Temperatura"] =  location_data["Temperatura"].astype(float)
    location_data["Velocidad"] =  location_data["Velocidad"].astype(float)
    location_data["Presion"] =  location_data["Presion"].astype(float)
    location_data["Humedad"] =  location_data["Humedad"].astype(float)

    #Agregamos el nombre de la Estacion en una columna
    location_data["Estacion"] = locacion

    return location_data

test= DataEntry(Input1,Input2,Input3)

# ## **Tratamiento de valores atipicos - Variable Precipitacion**

def Outliers_Precipitacion(df,opinion_experto,limite,input_valor):

  df["Precipitacion"] = round(df["Precipitacion"], 1)

 #Para la variable precipitacion es necesaria tener la opinion del experto
  #Si el experto determina valores altos que deban de ser reemplazado por la media o algun valor en especifico
  if opinion_experto=='si':
    df["Precipitacion"] = np.where((df.Precipitacion >limite ), input_valor, df["Precipitacion"])

  return df

Outliers_Precipitacion(test,'si',300,200)

# ## **Tratamiento de valores atipicos - Variable Velocidad**

print("3. Procesando la carga de datos")
print("")
#Funcion para pasar de fecha a string
def funcion_fecha_str(a):
    year=str(a.year)
    month=str(a.month)
    day=str(a.day)
    if len(month)<=1:
      month="0"+month
    if len(day)<=1:
      day="0"+day
    date=year+"-"+month+"-"+day
    return date

#Funcion para pasar de string a fecha
def funcion_str_fecha(a):
    date=datetime.strptime(a,'%Y-%m-%d').date()
    return date

#Funcion que coloca valores random en un intervalo
def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start

print("4. Procesando los datos anomalos y outliers")
print("")

def Outliers_Velocidad(df):
    n = 32.6
    st = 1
    step = 0.4
    datasets=[df]

    #Seleccion de Fecha Inicio
    year_ini=str(df.Fecha[1].year)
    month_ini=str(df.Fecha[1].month)
    day_ini=str(df.Fecha[1].day)
    if len(month_ini)<=1:
      month_ini="0"+month_ini
    if len(day_ini)<=1:
      day_ini="0"+day_ini
    ini=year_ini+"-"+month_ini+"-"+day_ini

    #Seleccion de Fecha Fin
    x=len(df.Fecha)-1
    year_fin=str(df.Fecha[x].year)
    month_fin=str(df.Fecha[x].month)
    day_fin=str(df.Fecha[x].day)
    if len(month_fin)<=1:
      month_fin="0"+month_fin
    if len(day_fin)<=1:
      day_fin="0"+day_fin
    fin=year_fin+"-"+month_fin+"-"+day_fin

    #Cantidad de periodos de 90 dias
    a=datetime.strptime(ini, '%Y-%m-%d').date()
    b=datetime.strptime(fin, '%Y-%m-%d').date()
    c=b-a
    periodos=c.days/90
    parte_decimal, parte_entera = math.modf(periodos)
    parte_entera=int(parte_entera)

    #Cantidad de periodos de 90 dias
    for i in range(1,parte_entera+1):
        stop=funcion_str_fecha(ini)+relativedelta(days=+90)
        stop_t=funcion_fecha_str(stop)
        for i in datasets:
            e = len(i)
            promedio = i[(i.Fecha > ini) & (i.Fecha < stop_t) & (i.Velocidad <= n)].Velocidad.mean()
            if math.isnan(promedio) == False :
               if promedio != 0:
                  random.seed(42)
                  promedio = [randrange_float(promedio-st, promedio +st, step) for p in range(e)]
                  promedio = np.round(promedio,1)
                  promedio = pd.DataFrame(promedio, columns=["valor"])
                  i["Velocidad"] = np.where((i.Fecha >= ini) & (i.Fecha <= stop_t) & (i.Velocidad > n), promedio["valor"], i["Velocidad"])
        #start_t=stop_t
        ini=funcion_fecha_str(stop)

    if parte_decimal>0:
       for i in datasets:
            e = len(i)
            promedio = i[(i.Fecha > ini) & (i.Fecha < fin) & (i.Velocidad <= n)].Velocidad.mean()
            if math.isnan(promedio) == False :
               if promedio != 0:
                  random.seed(42)
                  promedio = [randrange_float(promedio-st, promedio +st, step) for p in range(e)]
                  promedio = np.round(promedio,1)
                  promedio = pd.DataFrame(promedio, columns=["valor"])
                  i["Velocidad"] = np.where((i.Fecha >= ini) & (i.Fecha <= fin) & (i.Velocidad > n), promedio["valor"], i["Velocidad"])
    return df

Outliers_Velocidad(test)

# ## **Tratamiento de valores atipicos - Variable Temperatura**

#Codigo para Valores Atipicos de valores minimos
def Outliers_Temp_min(df):
    n = 15 #Temperatura minima del lugar
    s = 1
    step = 0.8
    datasets=[df]
    for i in datasets:
        e = len(i)
        np.random.seed(42)
        temp = [randrange_float(n, n + s, step) for p in range(e)]
        temp = np.round(temp,2)
        #temp = round(np.random.uniform(7,7+1),2)
        i["Temperatura"] = np.where((i.Temperatura < n), temp, i["Temperatura"])

    return df

#Codigo para Valores Atipicos de valores maximos
def Outliers_Temp_max(df):
    n = 32 #Temperatura maxima del lugar
    st = 1
    step = 0.4
    datasets=[df]

    #Seleccion de Fecha Inicio
    year_ini=str(df.Fecha[1].year)
    month_ini=str(df.Fecha[1].month)
    day_ini=str(df.Fecha[1].day)
    if len(month_ini)<=1:
      month_ini="0"+month_ini
    if len(day_ini)<=1:
      day_ini="0"+day_ini
    ini=year_ini+"-"+month_ini+"-"+day_ini

    #Seleccion de Fecha Fin
    x=len(df.Fecha)-1
    year_fin=str(df.Fecha[x].year)
    month_fin=str(df.Fecha[x].month)
    day_fin=str(df.Fecha[x].day)
    if len(month_fin)<=1:
      month_fin="0"+month_fin
    if len(day_fin)<=1:
      day_fin="0"+day_fin
    fin=year_fin+"-"+month_fin+"-"+day_fin

    #Cantidad de periodos de 120 dias
    a=datetime.strptime(ini, '%Y-%m-%d').date()
    b=datetime.strptime(fin, '%Y-%m-%d').date()
    c=b-a
    periodos=c.days/90
    parte_decimal, parte_entera = math.modf(periodos)
    parte_entera=int(parte_entera)

    #Cantidad de periodos de 120 dias
    for i in range(1,parte_entera+1):
        stop=funcion_str_fecha(ini)+relativedelta(days=+90)
        stop_t=funcion_fecha_str(stop)
        for i in datasets:
            e = len(i)
            promedio = i[(i.Fecha > ini) & (i.Fecha < stop_t) & (i.Temperatura <= n)].Temperatura.mean()
            if math.isnan(promedio) == False :
               if promedio != 0:
                  random.seed(42)
                  promedio = [randrange_float(promedio-st, promedio +st, step) for p in range(e)]
                  promedio = np.round(promedio,1)
                  promedio = pd.DataFrame(promedio, columns=["valor"])
                  i["Temperatura"] = np.where((i.Fecha >= ini) & (i.Fecha <= stop_t) & (i.Temperatura > n), promedio["valor"], i["Temperatura"])
        #start_t=stop_t
        ini=funcion_fecha_str(stop)

    if parte_decimal>0:
       for i in datasets:
            e = len(i)
            promedio = i[(i.Fecha > ini) & (i.Fecha < fin) & (i.Velocidad <= n)].Velocidad.mean()
            if math.isnan(promedio) == False :
               if promedio != 0:
                  random.seed(42)
                  promedio = [randrange_float(promedio-st, promedio +st, step) for p in range(e)]
                  promedio = np.round(promedio,1)
                  promedio = pd.DataFrame(promedio, columns=["valor"])
                  i["Temperatura"] = np.where((i.Fecha >= ini) & (i.Fecha <= fin) & (i.Temperatura > n), promedio["valor"], i["Temperatura"])
    return df

Outliers_Temp_min(test)
Outliers_Temp_max(test)


# ## **Tratamiento de valores atipicos - Variable Presion Barometrica**
#Codigo para Valores de Presiones iguales a cero
def Outliers_Presion(df):
    n = 0
    datasets=[df]
    for i in datasets:
        promedio = i[(i.Presion != n)].Presion.mean()
        i["Presion"] = np.where((i.Presion == n), promedio, i["Presion"])

    return df

Outliers_Presion(test)

# ## **Tratamiento de valores atipicos - Variable Humedad**
#Codigo para Valores Atipicos de valores mayores a 100%
def Outliers_Humedad(df):
    n = 100
    st = 1
    step = 0.4
    datasets=[df]

    #Seleccion de Fecha Inicio
    year_ini=str(df.Fecha[1].year)
    month_ini=str(df.Fecha[1].month)
    day_ini=str(df.Fecha[1].day)
    if len(month_ini)<=1:
      month_ini="0"+month_ini
    if len(day_ini)<=1:
      day_ini="0"+day_ini
    ini=year_ini+"-"+month_ini+"-"+day_ini

    #Seleccion de Fecha Fin
    x=len(df.Fecha)-1
    year_fin=str(df.Fecha[x].year)
    month_fin=str(df.Fecha[x].month)
    day_fin=str(df.Fecha[x].day)
    if len(month_fin)<=1:
      month_fin="0"+month_fin
    if len(day_fin)<=1:
      day_fin="0"+day_fin
    fin=year_fin+"-"+month_fin+"-"+day_fin

    #Cantidad de periodos de 120 dias
    a=datetime.strptime(ini, '%Y-%m-%d').date()
    b=datetime.strptime(fin, '%Y-%m-%d').date()
    c=b-a
    periodos=c.days/90
    parte_decimal, parte_entera = math.modf(periodos)
    parte_entera=int(parte_entera)

    #Cantidad de periodos de 120 dias
    for i in range(1,parte_entera+1):
        stop=funcion_str_fecha(ini)+relativedelta(days=+90)
        stop_t=funcion_fecha_str(stop)
        for i in datasets:
            e = len(i)
            promedio = i[(i.Fecha > ini) & (i.Fecha < stop_t) & (i.Humedad <= n)].Humedad.mean()
            if math.isnan(promedio) == False :
               if promedio != 0:
                  random.seed(42)
                  promedio = [randrange_float(promedio-st, promedio +st, step) for p in range(e)]
                  promedio = np.round(promedio,1)
                  promedio = pd.DataFrame(promedio, columns=["valor"])
                  i["Humedad"] = np.where((i.Fecha >= ini) & (i.Fecha <= stop_t) & (i.Humedad > n), promedio["valor"], i["Humedad"])
        #start_t=stop_t
        ini=funcion_fecha_str(stop)

    if parte_decimal>0:
       for i in datasets:
            e = len(i)
            promedio = i[(i.Fecha > ini) & (i.Fecha < fin) & (i.Humedad <= n)].Humedad.mean()
            if math.isnan(promedio) == False :
               if promedio != 0:
                  random.seed(42)
                  promedio = [randrange_float(promedio-st, promedio +st, step) for p in range(e)]
                  promedio = np.round(promedio,1)
                  promedio = pd.DataFrame(promedio, columns=["valor"])
                  i["Humedad"] = np.where((i.Fecha >= ini) & (i.Fecha <= fin) & (i.Humedad > n), promedio["valor"], i["Humedad"])
    return df

Outliers_Humedad(test)

# ## **Tratamiento de valores atipicos - Variable Radiacion**
#Codigo para Valores Atipicos de valores maximos
def Outliers_Radiacion(df):
    n = 1500
    st = 100
    step = 0.4
    datasets=[df]

    #Seleccion de Fecha Inicio
    year_ini=str(df.Fecha[1].year)
    month_ini=str(df.Fecha[1].month)
    day_ini=str(df.Fecha[1].day)
    if len(month_ini)<=1:
      month_ini="0"+month_ini
    if len(day_ini)<=1:
      day_ini="0"+day_ini
    ini=year_ini+"-"+month_ini+"-"+day_ini

    #Seleccion de Fecha Fin
    x=len(df.Fecha)-1
    year_fin=str(df.Fecha[x].year)
    month_fin=str(df.Fecha[x].month)
    day_fin=str(df.Fecha[x].day)
    if len(month_fin)<=1:
      month_fin="0"+month_fin
    if len(day_fin)<=1:
      day_fin="0"+day_fin
    fin=year_fin+"-"+month_fin+"-"+day_fin

    #Cantidad de periodos de 90 dias
    a=datetime.strptime(ini, '%Y-%m-%d').date()
    b=datetime.strptime(fin, '%Y-%m-%d').date()
    c=b-a
    periodos=c.days/90
    parte_decimal, parte_entera = math.modf(periodos)
    parte_entera=int(parte_entera)

    #Cantidad de periodos de 120 dias
    for i in range(1,parte_entera+1):
        stop=funcion_str_fecha(ini)+relativedelta(days=+90)
        stop_t=funcion_fecha_str(stop)
        for i in datasets:
            e = len(i)
            promedio = i[(i.Fecha > ini) & (i.Fecha < stop_t) & (i.Radiacion <= n)].Radiacion.mean()
            if math.isnan(promedio) == False :
               if promedio != 0:
                  random.seed(42)
                  promedio = [randrange_float(promedio-st, promedio +st, step) for p in range(e)]
                  promedio = np.round(promedio,1)
                  promedio = pd.DataFrame(promedio, columns=["valor"])
                  i["Radiacion"] = np.where((i.Fecha >= ini) & (i.Fecha <= stop_t) & (i.Radiacion > n), promedio["valor"], i["Radiacion"])
        #start_t=stop_t
        ini=funcion_fecha_str(stop)

    if parte_decimal>0:
       for i in datasets:
            e = len(i)
            promedio = i[(i.Fecha > ini) & (i.Fecha < fin) & (i.Radiacion <= n)].Radiacion.mean()
            if math.isnan(promedio) == False :
               if promedio != 0:
                  random.seed(42)
                  promedio = [randrange_float(promedio-st, promedio +st, step) for p in range(e)]
                  promedio = np.round(promedio,1)
                  promedio = pd.DataFrame(promedio, columns=["valor"])
                  i["Radiacion"] = np.where((i.Fecha >= ini) & (i.Fecha <= fin) & (i.Radiacion > n), promedio["valor"], i["Radiacion"])
    return df

Outliers_Radiacion(test)

# ## **Imputacion de Datos Nulos**
print("5. Procesando la imputación de Datos Nulos - Algoritmo Miss Forest")
print("")

#Funcion para imputacion de Datos Nulos
def Imputacion_Nulos(df):
    data = pd.DataFrame()
    data = df[["Fecha","Hora","Temperatura","Velocidad","Presion","Humedad","Precipitacion","Radiacion"]]
    data["Fecha"] = pd.to_datetime(data["Fecha"])
    data["year"] = df["Fecha"].dt.year
    data["month"] = df["Fecha"].dt.month
    data["day"] = df["Fecha"].dt.day
    data = data.drop(columns={"Fecha"}, axis = 1)
    return data

datas = Imputacion_Nulos(test)
data1=datas[["Temperatura","Velocidad","Presion","Humedad","Precipitacion","Radiacion","year","month","day"]]
data2=datas[["Hora"]]
data3=test[["Fecha"]]
imputer = MissForest()
data_imputed = imputer.fit_transform(data1)
df_imputed = pd.DataFrame(data_imputed)
col_list = tuple(data1.columns.values)
old_col_names = tuple(df_imputed.columns.values)

df_imputed.rename(
    columns={i:j for i,j in zip(old_col_names,col_list)}, inplace=True
) 

df_imputed = df_imputed.astype({"year": int,"month": int,"day": int})

#Result es la data final ordenada, sin outliers y datos nulos
result = pd.concat([data2 , df_imputed], axis=1)

#Se agrega la columna estacion, el cual es un dato inicial en el codigo
result["Estacion"]=Input1

#Se agrega la columna Fecha en formato datetime para utilizarse en el modelo
df=result
df["Fecha-str"]=df["year"].astype(str)+"-"+df["month"].astype(str)+"-"+df["day"].astype(str)+" "+df["Hora"].astype(str)
df["Fecha-str"]=df["Fecha-str"].astype('string') 
df["Fecha"] = pd.to_datetime(df["Fecha-str"])
df = df.drop(columns={"Fecha-str"}, axis = 1)


# ## **Implementacion de Modelo para Series de Tiempo**

tipos = df.Estacion.unique()

print("6. Realizando el cálculo de Predicción")
print("")
#Se pueden definir 
holidays = pd.DataFrame({
                'holiday': 'eventossignificativos', 
                'ds': pd.to_datetime(lsteventos),
                'lower_window': 0,
                'upper_window': 1,
            })
holidays


## Modelo Prophet Multivariado - 5 variables

def group(data, column_name, frec):

        """
        Esta funcion agrupa por serie de tiempo en dias para obtener la media los datos
        """
        data = data.groupby([pd.Grouper(key=column_name, freq=frec)]).mean() 
        data = data.reset_index()
        data[column_name] = pd.to_datetime(data[column_name]) 
        data[column_name] = data[column_name].sort_values(ascending=False)
        data = data.set_index(column_name)

        return data

def RMSE_score_prophet(y_true, y_pred):

                y1 = mean_squared_error(y_true[['y']].iloc[-7], y_pred[['yhat']].iloc[-14])
                y2 = mean_squared_error(y_true[['y']].iloc[-6], y_pred[['yhat']].iloc[-13])
                y3 = mean_squared_error(y_true[['y']].iloc[-5], y_pred[['yhat']].iloc[-12])
                y4 = mean_squared_error(y_true[['y']].iloc[-4], y_pred[['yhat']].iloc[-11])
                y5 = mean_squared_error(y_true[['y']].iloc[-3], y_pred[['yhat']].iloc[-10])
                y6 = mean_squared_error(y_true[['y']].iloc[-2], y_pred[['yhat']].iloc[-9])
                y7 = mean_squared_error(y_true[['y']].iloc[-1], y_pred[['yhat']].iloc[-8])

                mean = sqrt((y1 + y2 + y3 + y4 + y5 + y6 + y7)/7)
                return mean 

def MSE_score_prophet(y_true, y_pred):

                y1 = mean_squared_error(y_true[['y']].iloc[-7], y_pred[['yhat']].iloc[-14])
                y2 = mean_squared_error(y_true[['y']].iloc[-6], y_pred[['yhat']].iloc[-13])
                y3 = mean_squared_error(y_true[['y']].iloc[-5], y_pred[['yhat']].iloc[-12])
                y4 = mean_squared_error(y_true[['y']].iloc[-4], y_pred[['yhat']].iloc[-11])
                y5 = mean_squared_error(y_true[['y']].iloc[-3], y_pred[['yhat']].iloc[-10])
                y6 = mean_squared_error(y_true[['y']].iloc[-2], y_pred[['yhat']].iloc[-9])
                y7 = mean_squared_error(y_true[['y']].iloc[-1], y_pred[['yhat']].iloc[-8])

                mean = (y1 + y2 + y3 + y4 + y5 + y6 + y7)/7 
                return mean 

def MAE_score_prophet(y_true, y_pred):

                y1 = mean_absolute_error(y_true[['y']].iloc[-7], y_pred[['yhat']].iloc[-14])
                y2 = mean_absolute_error(y_true[['y']].iloc[-6], y_pred[['yhat']].iloc[-13])
                y3 = mean_absolute_error(y_true[['y']].iloc[-5], y_pred[['yhat']].iloc[-12])
                y4 = mean_absolute_error(y_true[['y']].iloc[-4], y_pred[['yhat']].iloc[-11])
                y5 = mean_absolute_error(y_true[['y']].iloc[-3], y_pred[['yhat']].iloc[-10])
                y6 = mean_absolute_error(y_true[['y']].iloc[-2], y_pred[['yhat']].iloc[-9])
                y7 = mean_absolute_error(y_true[['y']].iloc[-1], y_pred[['yhat']].iloc[-8])

                mean = (y1 + y2 + y3 + y4 + y5 + y6 + y7)/7 
                return mean 


# ## **Resultados**


print("7. A continuación las métricas obtenidas para el modelo Propeth")
print("")
#Propeth Multivariable - 4 Variables

import datetime as datetime
guardar = []

for i in tipos:
    print("Estacion :", i)
    df_1 = df[df.Estacion == i] 
    df_1 = group(df_1,"Fecha", frec="1D")
    df_1["Estacion"] = i
    #Rellenamos los valores nulls con la media
    df_1 = df_1.fillna(df_1.mean()) 

    df2 = df_1[["Precipitacion",'Presion','Temperatura','Humedad','Radiacion']]  
    df2.reset_index(drop = False, inplace = True)
    df2 = df2.rename(columns={"Precipitacion":"y", "Fecha": "ds"})
    
    
    m = Prophet(interval_width=0.95, daily_seasonality=True, holidays = holidays) 
    m.add_regressor('Presion') 
    m.add_regressor('Temperatura')
    m.add_regressor('Humedad')
    m.add_regressor('Radiacion')
    m.fit(df2) 
    
    futur = m.make_future_dataframe(periods=7, freq="1D") 
    future_train = futur.iloc[:-14] 
    future  = futur.iloc[-31:]   
    
    future_train.reset_index(inplace = True, drop = True)
    future.reset_index(inplace = True, drop = True)
    
    future_train = pd.merge(future_train ,df2[['Presion','Temperatura','Humedad','Radiacion',"ds"]].iloc[:-7], how = "left", on = ['ds'])
    future_train = future_train.fillna(df_1.mean())
    
    
    future = pd.merge(future,df2[['Presion','Temperatura','Humedad','Radiacion',"ds"]].iloc[-24:], how = "left", on = ['ds'])
    future = future.fillna(df_1.mean()) #
    
    
    forecast_train = m.predict(future_train) 
    forecast = m.predict(future)
    
    #Fecha de Inicio para el plot
    y1=df.iloc[1][7]
    m1=df.iloc[1][8]
    d1=df.iloc[1][9]
    #Fecha de Fin para el plot
    x=len(df.Fecha)-1
    y2=int(df.Fecha[x].year)
    m2=int(df.Fecha[x].month)
    d2=int(df.Fecha[x].day)
      
    rmse_train = sqrt(mean_squared_error(df2[:-7].y.values, forecast_train.yhat.values)) 
    mse_train = mean_squared_error(df2[:-7].y.values, forecast_train.yhat.values)
    mae_train = mean_absolute_error(df2[:-7].y.values, forecast_train.yhat.values) 
          

    rmse = RMSE_score_prophet(df2, forecast)
    mse = MSE_score_prophet(df2, forecast)
    mae = MAE_score_prophet(df2, forecast)
    print(" ")
    print("El error RMSE obtenido para la predicción = ", rmse)
    print("El error MSE  obtenido para la predicción = ", mse)
    print("El error MAE  obtenido para la predicción = ", mae)
    print("\n")
    
    forecast2 = forecast.copy()
    forecast2.rename(columns={'yhat':'y'},inplace = True) 
    frames = [df2, forecast2[['ds','y','Presion','Temperatura','Humedad','Radiacion']].iloc[-7:]] 
    final = pd.concat(frames).reset_index(drop = True)
    riesgo = final.y.tail(25).sum() 
    guardar.append([riesgo,i])
    
# ## **Codigo para Alertas de Precipitaciones**
a=df.Fecha[x]-datetime.timedelta(days=18)
b=df.Fecha[x]+datetime.timedelta(days=7)
print("8. Calculando la predicción para la estación:", Input1)
print("")

table = pd.DataFrame(guardar, )   
table.rename(columns={0:'Precipitacion acumulada', 1:'Estacion'}, inplace=True)   
table['Riesgo'] = np.NaN 
table.sort_values(by = 'Precipitacion acumulada', inplace=True)
table.reset_index(drop=True, inplace=True) 

for i in range(len(table)): 
    if table['Precipitacion acumulada'][i] < 200: 
        table['Riesgo'].iloc[i] = "No Hay Peligro" 

    if (table['Precipitacion acumulada'][i]  >= 200) & (table['Precipitacion acumulada'][i] < 300):  
        table['Riesgo'].iloc[i] = "Alerta Amarilla" 

    elif (table['Precipitacion acumulada'][i] < 400) & (table['Precipitacion acumulada'][i] >= 300):
        table['Riesgo'].iloc[i] = "Alerta Naranja"

    elif table['Precipitacion acumulada'][i] >=400:   
        table['Riesgo'].iloc[i] = "Alerta Roja" 

print("La predicción para la precipitación acumulada desde:", a ,"hasta: ", b)
print("es:",table['Precipitacion acumulada'][0],"mm.")
print(" ")
print("9. Imprimiendo la predicción para la estación:", Input1)

plt.figure(figsize=(30,2))  
sns.set(font_scale = 2)

palette = {'No Hay Peligro':'blue', 'Alerta Amarilla': 'yellow', 'Alerta Naranja': 'orange', 'Alerta Roja': 'red'}        

sns.barplot(x = 'Precipitacion acumulada', y = table.Estacion, data = table, hue = 'Riesgo', palette = palette,dodge=False )      

print("6. Imprimiendo Precipitación A25 para la estación", table.iloc[0][1])
print("Guarde la predicción obtenida y presione Enter")
print("")
plt.title('Precipitación Acumulada (25 dias) - A25', fontsize=30)      
plt.legend(['Alerta Roja'], prop={'size': 13})  
plt.xlabel('Precipitación (mm)', fontsize = 30)    
plt.ylabel("Estacion",fontsize = 30)   
plt.legend(loc = 1,prop={'size': 18}) 
#ax.set_xlim(1,50)
plt.show()  

import smtplib
from email.message import EmailMessage 

if table['Precipitacion acumulada'][0]>=400: 
    email_subject = "Alerta Roja - Precipitacion mayor a 400mm en estacion "+table['Estacion'][0] 
elif table['Precipitacion acumulada'][0]>=300 and table['Precipitacion acumulada'][0]<400:
    email_subject = "Alerta Naranja- Precipitacion mayor a 300mm en estacion "+table['Estacion'][0]
elif table['Precipitacion acumulada'][0]>=200 and table['Precipitacion acumulada'][0]<300:
    email_subject = "Alerta Amarilla- Precipitacion mayor a 200mm en estacion "+table['Estacion'][0]
elif table['Precipitacion acumulada'][0]<200:
    email_subject = "Precipitacion menor a 200mm en estacion "+table['Estacion'][0] +" sin riesgos"

sender_email_address = "mallccacocordova@gmail.com" 
receiver_email_address = "mallccacocordova@gmail.com"
email_smtp = "smtp.gmail.com" 
email_password = "fsahxnlclevjwmus"

# Create an email message object 
message = EmailMessage() 

# Configure email headers 
message['Subject'] = email_subject 
message['From'] = sender_email_address 
message['To'] = receiver_email_address 

# Set email body text 
message.set_content(email_subject) 

# Set smtp server and port 
server = smtplib.SMTP(email_smtp, '587') 

# Identify this client to the SMTP server 
server.ehlo() 

# Secure the SMTP connection 
server.starttls() 

# Login to email account 
server.login(sender_email_address, email_password) 

# Send email 
server.send_message(message) 

# Close connection to server 
server.quit()

print("10. Correo de Alerta enviado con éxito al remitente:",receiver_email_address)
print("")