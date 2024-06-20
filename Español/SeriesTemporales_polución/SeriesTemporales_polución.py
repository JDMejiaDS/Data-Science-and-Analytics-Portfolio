import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import datetime as dt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
import warnings

warnings.filterwarnings("ignore")

#Esta función está construida de manera que puede aplicarse modelos univariados
#Esta función reforma el arreglo de 2 dimensiones a 3 dimensiones
def ventana_paquetes_datos(array, tamaño_paquete_entrada, tamaño_paquete_salida):
    X, Y = [], []
    shape = array.shape

    if len(shape) == 1:
        filas, cols = array.shape[0], 1
    else:
        filas, cols = array.shape

    for i in range(filas - tamaño_paquete_entrada - tamaño_paquete_salida):
        X.append(array[i:i+tamaño_paquete_entrada, 0:cols])
        #En la linea siguiente se selecciona la variable a predecir, ser cuidadoso al seleccionar su index de columna
        Y.append(array[i + tamaño_paquete_entrada: i + tamaño_paquete_entrada + tamaño_paquete_salida, 0].reshape(tamaño_paquete_salida,1))

    x = np.array(X)
    y = np.array(Y)
    return x, y

#Elaboración de una función de predicción y escalado inverso para comparación con los datos reales
def predecir(x, modelo, escalador):
    y_pred = modelo.predict(x)
    y_pred = y_pred.reshape(-1, 1)
    y_pr = escalador.inverse_transform(y_pred)
    y_pr = y_pr.flatten()
    return y_pr

#Elaboración de la función de escalamiento
def escalar_dataset(data_input, col_ref):
    num_características = data_input['x_tr'].shape[2]
    col_ref = df.columns.get_loc(col_ref)
    # Generar listado con "scalers"
    scalers = [MinMaxScaler(feature_range=(-1,1)) for i in range(num_características)]
    # Arreglos que contendrán los datasets escalados
    x_tr_s = np.zeros(data_input['x_tr'].shape)
    x_vl_s = np.zeros(data_input['x_vl'].shape)
    x_ts_s = np.zeros(data_input['x_ts'].shape)
    y_tr_s = np.zeros(data_input['y_tr'].shape)
    y_vl_s = np.zeros(data_input['y_vl'].shape)
    y_ts_s = np.zeros(data_input['y_ts'].shape)
    # Escalamiento: se usarán los min/max del set de entrenamiento para la totalidad de los datasets
    # Escalamiento Xs
    for i in range(num_características):
        x_tr_s[:,:,i] = scalers[i].fit_transform(x_tr[:,:,i])
        x_vl_s[:,:,i] = scalers[i].transform(x_val[:,:,i])
        x_ts_s[:,:,i] = scalers[i].transform(x_ts[:,:,i])
    # Escalamiento Ys
    y_tr_s[:,:,0] = scalers[col_ref].fit_transform(y_tr[:,:,0])
    y_vl_s[:,:,0] = scalers[col_ref].transform(y_val[:,:,0])
    y_ts_s[:,:,0] = scalers[col_ref].transform(y_ts[:,:,0])
    # Conformar ` de salida
    data_scal = {
        'x_tr_s': x_tr_s, 'y_tr_s': y_tr_s,
        'x_vl_s': x_vl_s, 'y_vl_s': y_vl_s,
        'x_ts_s': x_ts_s, 'y_ts_s': y_ts_s,
    }
    return data_scal, scalers[col_ref]

#Elaboración de la función para visualización de importancia de características
def pfi_mod(modelo, x, y, cols):
    N_vars=x.shape[2]
    pfi_results=[]

    rmse= mod.evaluate(x, y, verbose=0)
    for i in range(N_vars):
        print(f"Permutando y calculando PFI para la variable {i+1} de {N_vars}")
        col_original = x[:,:,i].copy()
        np.random.shuffle(x[:,:,i])
        rmse_perm = mod.evaluate(x, y, verbose=0)
        pfi_results.append({"Variable":cols[i],"PFI":rmse_perm/rmse})
        x[:,:,i] = col_original

    PFI_df= pd.DataFrame(pfi_results).sort_values(by="PFI", ascending=False)
    return PFI_df

raw_df = pd.read_csv("LSTM-Multivariate_pollution.csv")
raw_df.head()

#Ajustando la variable "date" a tipo de dato datetime y llevandola a su respectivo formato
raw_df["date"] = pd.to_datetime(raw_df["date"], format="%Y-%m-%d %H:%M:%S")

#Conviertiendo los indices del conjunto de datos a su tiempo de medición respectiva
raw_df.set_index(["date"], inplace= True)
raw_df.sort_index()

#verificación de la equitatividad entre la distancia de todas las mediciones realizadas y copia de data original
poll_df = raw_df.copy()
dt_dif = poll_df.index.to_series().diff().dt.total_seconds()

#Visualización de la serie temporal de la variable polución
plt.figure(figsize=(10, 6))
plt.plot(raw_df.index, raw_df['pollution'], label='Pollution', color = "teal")
plt.title('Niveles de Polución (2010-2015)')
plt.xlabel('Fecha')
plt.ylabel('Polución')
plt.show()

#Visualización de las covariables
raw_df[['dew', 'temp', 'press', 'wnd_dir', 'snow', 'rain']].plot(subplots=True, figsize=(12, 18))
plt.show()

#Descomposición de la serie temporal en tendencia y  estacionalidad con visualización
df = raw_df.copy()
df.pop("wnd_dir")
result = seasonal_decompose(df['pollution'], model='additive', period=365)
plt.figure(figsize=(15, 6))
plt.title("Tendencia")
plt.minorticks_on()
result.trend.plot()
plt.show()
plt.figure(figsize=(15, 6))
plt.title("Estacionalidad")
plt.minorticks_on()
result.seasonal["2012"].plot(color="green")
plt.show()

#Visualización de autocorrelación y autocorrelación relativa
plot_acf(df['pollution'], lags=60, color="purple", alpha=0.01)
plt.grid()
plt.minorticks_on()
plt.show()
plot_pacf(df['pollution'], lags=60, color="purple", alpha =0.01)
plt.grid()
plt.minorticks_on()
plt.show()

#Comporando la estacionalidad de los datos a diferentes niveles de significancia (1%, 5% y 10%)
p_estacionalidad = sm.tsa.adfuller(raw_df['pollution'], maxlag=30, regression='c', autolag='AIC')
print('ADF Statc', p_estacionalidad[0])
print('p-valor:', p_estacionalidad[1])
print('Valor crítico:', p_estacionalidad[4])

#División del conjunto de datos
tr_len = int(len(df)*0.8)
tst_len = int(len(df)*0.1)
val_len = df.shape[0] - tr_len - tst_len
print(tr_len, tst_len, val_len)
tr_df = df[0:tr_len]
val_df =df[tr_len: tr_len+val_len]
tst_df = df[tr_len+val_len:]
print(f"el tamaño del set del entrenamiento es :{tr_df.shape}")
print(f"el tamaño del set de validación es :{val_df.shape}")
print(f"el tamaño del set testeo es :{tst_df.shape}")

paquete_ent = 24
paquete_sal = 1

x_tr, y_tr = ventana_paquetes_datos(tr_df.values, paquete_ent, paquete_sal)
x_val, y_val = ventana_paquetes_datos(val_df.values, paquete_ent, paquete_sal)
x_ts, y_ts = ventana_paquetes_datos(tst_df.values, paquete_ent, paquete_sal)

#Verificación de la dimensionalidad del código
print(f"subconjunto de entrenamiento - x_tr: {x_tr.shape}, y_tr: {y_tr.shape}")
print(f"subconjunto de validación - x_vl: {x_val.shape}, y_vl: {y_val.shape}")
print(f"subconjunto de test - x_ts: {x_ts.shape}, y_ts: {y_ts.shape}")

data_in = {
    'x_tr': x_tr, 'y_tr': y_tr,
    'x_vl': x_val, 'y_vl': y_val,
    'x_ts': x_ts, 'y_ts': y_ts,
}
#Ejecucuón de la función de escalamiento
data_s, scaler = escalar_dataset(data_in, col_ref="pollution")

#creando datasets de entrenamiento, validación y prueba
x_tr_s, y_tr_s = data_s['x_tr_s'], data_s['y_tr_s']
x_vl_s, y_vl_s = data_s['x_vl_s'], data_s['y_vl_s']
x_ts_s, y_ts_s = data_s['x_ts_s'], data_s['y_ts_s']

#Visualización de las distribuciones de las variables
labels = ['pollution','dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
positions = np.arange(len(labels))

fig, ax = plt.subplots(figsize = (12 ,5))
for i in range(data_in['x_tr'].shape[2]):
    ax.violinplot(x_tr_s[:,:,i].flatten(), positions = [i])
    ax.violinplot(x_vl_s[:,:,i].flatten(), positions = [i])
    ax.violinplot(x_ts_s[:,:,i].flatten(), positions = [i])

ax.set_xticks(positions)
plt.title("Distribución de los datos para cada partición y variable")
ax.set_xticklabels(labels)
ax.autoscale()
plt.show()

#Visualización de las distribuciones de las variables
labels = ['pollution','dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
positions = np.arange(len(labels))

fig, ax = plt.subplots(figsize = (12 ,5))
for i in range(data_in['x_tr'].shape[2]):
    ax.violinplot(x_tr_s[:,:,i].flatten(), positions = [i])
    ax.violinplot(x_vl_s[:,:,i].flatten(), positions = [i])
    ax.violinplot(x_ts_s[:,:,i].flatten(), positions = [i])

ax.set_xticks(positions)
plt.title("Distribución de los datos para cada partición y variable")
ax.set_xticklabels(labels)
ax.autoscale()
plt.show()

labels = ['pollution','dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
positions = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(12, 5))

for i in range(len(labels)):
    parts_tr = ax.violinplot(x_tr_s[:, :, i].flatten(), positions=[positions[i] - 0.3], showmeans=False, showmedians=True)
    parts_vl = ax.violinplot(x_vl_s[:, :, i].flatten(), positions=[positions[i]], showmeans=False, showmedians=True)
    parts_ts = ax.violinplot(x_ts_s[:, :, i].flatten(), positions=[positions[i] + 0.3], showmeans=False, showmedians=True)

ax.set_xticks(positions)
ax.set_xticklabels(labels)
legend_labels = ['Train', 'Validation', 'Test']
legend_colors = [parts_tr['bodies'][0].get_facecolor().flatten(),
                parts_vl['bodies'][0].get_facecolor().flatten(),
                parts_ts['bodies'][0].get_facecolor().flatten()]

custom_lines = [plt.Line2D([0], [0], color=color) for color in legend_colors]
ax.legend(custom_lines, legend_labels)
plt.title("Distribución de los datos para cada partición y variable (sin sobreposición)")
plt.show()

#visualización de la variable respuesta
fig, ax = plt.subplots(figsize = (7 ,4))
ax.violinplot(y_tr_s[:,:,0].flatten())
ax.violinplot(y_vl_s[:,:,0].flatten())
ax.violinplot(y_ts_s[:,:,0].flatten())

ax.autoscale()
plt.title("Distribución de la variable a predecir")
plt.show()

#Generando pseudoaleatoriedad para reproductibilidad
tf.random.set_seed(12896)
np.random.seed(12896)
tf.config.experimental.enable_op_determinism()

#Generación del modelo
unds = 140
input = (x_tr_s.shape[1],x_tr_s.shape[2])
mod = Sequential()
mod.add(LSTM(units=unds, input_shape=input))
mod.add(Dense(paquete_sal, activation="linear"))
optim = RMSprop(learning_rate=0.00005)
mod.compile(optimizer=optim, loss=tf.keras.losses.MeanSquaredError())
mod.summary()
EPOCHS = 80
BATCH_SIZE = 256
historia = mod.fit(
    x = x_tr_s,
    y = y_tr_s,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = (x_vl_s, y_vl_s),
    verbose=1
)

#Curva de pérdida para verificar que el modelo no se haya sobreajustado
plt.plot(historia.history['loss'], label = "RMSE de entrenamiento", color="teal")
plt.plot(historia.history['val_loss'], label = "RMSE de validación", color="purple")
plt.title("Curva de pérdida")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend()
plt.show()

#Métricas de evaluación sobre las 3 particiones de datos realizadas
RMSE_tr = mod.evaluate(x_tr_s, y_tr_s, verbose=0)
RMSE_vl = mod.evaluate(x_vl_s, y_vl_s, verbose=0)
RME_ts = mod.evaluate(x_ts_s, y_ts_s, verbose=0)
print(f"RMSE de entrenamiento: {RMSE_tr:.4f}")
print(f"RMSE de validación: {RMSE_vl:.4f}")
print(f"RMSE de test: {RME_ts:.4f}")

#Realización de predicciones
y_ts_pred = predecir(x_ts_s, mod, scaler)
y_ts_pred.shape

#Visualización de los errores
y_ts_flat = y_ts.flatten()
error_pred =  y_ts_pred - y_ts_flat
error_pred.shape
errores = pd.DataFrame(data={"y_true":y_ts_flat, "y_pred":y_ts_pred})
plt.plot(error_pred, label = "Error de predicción", color = "red")
plt.title("Error de predicción")
plt.xlabel("Observaciones")
plt.ylabel("Error de predicción")
plt.legend()
plt.grid()
plt.show()
#visualización de la serie predicha y la real
plt.figure(figsize=(10, 6))
plt.plot(errores["y_true"], label="Comportamiento real", color ="red", ls = ":")
plt.plot(errores["y_pred"], label= "Comportamiento predicho", color = "teal", ls = "--", alpha = 0.6)
plt.title("Comportamiento real vs predicción")
plt.xlabel("Tiempo")
plt.ylabel("Polución")
plt.legend()
plt.grid()
plt.show()
#Realizando análisis de PFI y visualización de relevancia
PFI_df = pfi_mod(mod, x_ts_s, y_ts_s, df.columns)
ax = sns.barplot(x="PFI", y="Variable", data = PFI_df , palette="turbo")
ax.set_xlabel("Importancia relativa")
plt.title("Importancia relativa de las variables")
plt.grid()
plt.show()
