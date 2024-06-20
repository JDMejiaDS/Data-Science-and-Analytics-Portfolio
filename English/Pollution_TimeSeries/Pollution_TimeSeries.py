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
raw_df = pd.read_csv("LSTM-Multivariate_pollution.csv")
raw_df.head()

#This function is designed to apply univariate models
#Transforming time series creating a supervised learning problem
def ventana_paquetes_datos(array, tamaño_paquete_entrada, tamaño_paquete_salida):
    X, Y = [], []
    shape = array.shape

    if len(shape) == 1:
        filas, cols = array.shape[0], 1
    else:
        filas, cols = array.shape

    for i in range(filas - tamaño_paquete_entrada - tamaño_paquete_salida):
        X.append(array[i:i+tamaño_paquete_entrada, 0:cols])
        #The following line selects the variable to predict. Be careful when selecting its column index.
        Y.append(array[i + tamaño_paquete_entrada: i + tamaño_paquete_entrada + tamaño_paquete_salida, 0].reshape(tamaño_paquete_salida,1))

    x = np.array(X)
    y = np.array(Y)
    return x, y

#Development of the scaling function
def escalar_dataset(data_input, col_ref):
    num_características = data_input['x_tr'].shape[2]
    col_ref = df.columns.get_loc(col_ref)

    # Generate a list of "scalers"
    scalers = [MinMaxScaler(feature_range=(-1,1)) for i in range(num_características)]
    # Arrays that will contain the scaled datasets.
    x_tr_s = np.zeros(data_input['x_tr'].shape)
    x_vl_s = np.zeros(data_input['x_vl'].shape)
    x_ts_s = np.zeros(data_input['x_ts'].shape)
    y_tr_s = np.zeros(data_input['y_tr'].shape)
    y_vl_s = np.zeros(data_input['y_vl'].shape)
    y_ts_s = np.zeros(data_input['y_ts'].shape)
    # Scaling: the min/max of the training set will be used to scale the entire dataset
    # scaling Xs
    for i in range(num_características):
        x_tr_s[:,:,i] = scalers[i].fit_transform(x_tr[:,:,i])
        x_vl_s[:,:,i] = scalers[i].transform(x_val[:,:,i])
        x_ts_s[:,:,i] = scalers[i].transform(x_ts[:,:,i])

    # scaling Ys
    y_tr_s[:,:,0] = scalers[col_ref].fit_transform(y_tr[:,:,0])
    y_vl_s[:,:,0] = scalers[col_ref].transform(y_val[:,:,0])
    y_ts_s[:,:,0] = scalers[col_ref].transform(y_ts[:,:,0])

    # Developing axit arrays
    data_scal = {
        'x_tr_s': x_tr_s, 'y_tr_s': y_tr_s,
        'x_vl_s': x_vl_s, 'y_vl_s': y_vl_s,
        'x_ts_s': x_ts_s, 'y_ts_s': y_ts_s,
    }

    return data_scal, scalers[col_ref]

def predecir(x, modelo, escalador):
    y_pred = modelo.predict(x)
    y_pred = y_pred.reshape(-1, 1)
    y_pr = escalador.inverse_transform(y_pred)
    y_pr = y_pr.flatten()
    return y_pr

#Development of the function for visualizing feature importance.
def pfi_mod(modelo, x, y, cols):
  N_vars=x.shape[2]
  pfi_results=[]

  rmse= mod.evaluate(x, y, verbose=0)
  for i in range(N_vars):
    print(f"Permuting and calculating PFI for feature {i+1} of {N_vars}")
    col_original = x[:,:,i].copy()
    np.random.shuffle(x[:,:,i])
    rmse_perm = mod.evaluate(x, y, verbose=0)
    pfi_results.append({"Variable":cols[i],"PFI":rmse_perm/rmse})
    x[:,:,i] = col_original

  PFI_df= pd.DataFrame(pfi_results).sort_values(by="PFI", ascending=False)
  return PFI_df

#Adjusting the "date" variable to datetime data type and formatting it accordingly.
raw_df["date"] = pd.to_datetime(raw_df["date"], format="%Y-%m-%d %H:%M:%S")

#Looking for missing information
raw_df.isna().sum()

#Converting dataset indices to their respective measurement times.
raw_df.set_index(["date"], inplace= True)
raw_df.sort_index()

#Verification of equity among the distance of all measurements taken and copy of original data.
poll_df = raw_df.copy()
dt_dif = poll_df.index.to_series().diff().dt.total_seconds()
print(dt_dif.value_counts())

#Target feature (pollution) time series and covariates visualization
plt.figure(figsize=(10, 6))
plt.plot(raw_df.index, raw_df['pollution'], label='Pollution', color = "teal")
plt.title('Pollution metrics (2010-2015)')
plt.xlabel('Fecha')
plt.ylabel('Polución')
plt.show()
raw_df[['dew', 'temp', 'press', 'wnd_dir', 'snow', 'rain']].plot(subplots=True, figsize=(12, 18))
plt.show()

#Time series decomposing
df = raw_df.copy()
df.pop("wnd_dir")
result = seasonal_decompose(df['pollution'], model='additive', period=365)
plt.figure(figsize=(15, 6))
plt.title("Trend")
plt.minorticks_on()
result.trend.plot()
plt.show()
plt.figure(figsize=(15, 6))
plt.title("Estacionality")
plt.minorticks_on()
result.seasonal["2012"].plot(color="green")
plt.show()

#Auto correlation and Partial autocorrelation visualizations
plot_acf(df['pollution'], lags=60, color="purple", alpha=0.01)
plt.grid()
plt.minorticks_on()
plt.show()
plot_pacf(df['pollution'], lags=60, color="purple", alpha =0.01)
plt.grid()
plt.minorticks_on()
plt.show()

#looking for seasonality of the data at different significance levels (1%, 5%, and 10%).
p_estacionalidad = sm.tsa.adfuller(raw_df['pollution'], maxlag=30, regression='c', autolag='AIC')
print('ADF Statc', p_estacionalidad[0])
print('p-valor:', p_estacionalidad[1])
print('Valor crítico:', p_estacionalidad[4])

#Data Splitting
tr_len = int(len(df)*0.8)
tst_len = int(len(df)*0.1)
val_len = df.shape[0] - tr_len - tst_len
print(tr_len, tst_len, val_len)
tr_df = df[0:tr_len]
val_df =df[tr_len: tr_len+val_len]
tst_df = df[tr_len+val_len:]
print(f"traning dataset shape:{tr_df.shape}")
print(f"validation dataset shape:{val_df.shape}")
print(f"testing dataset shape:{tst_df.shape}")

#partitions visualization
fig, ax = plt.subplots(figsize = (12 ,5))
ax.plot(tr_df["pollution"], label = "traning", color= "teal")
ax.plot(val_df["pollution"], label = "Validation", color= "green")
ax.plot(tst_df["pollution"], label = "Test", color="red")
plt.title("Dataset partitions")
plt.legend()
plt.grid()
plt.show()

#Batches creation for modeling
paquete_ent = 24
paquete_sal = 1

x_tr, y_tr = ventana_paquetes_datos(tr_df.values, paquete_ent, paquete_sal)
x_val, y_val = ventana_paquetes_datos(val_df.values, paquete_ent, paquete_sal)
x_ts, y_ts = ventana_paquetes_datos(tst_df.values, paquete_ent, paquete_sal)

#Verification of the dataset's dimensionality
print(f"traning subset- x_tr: {x_tr.shape}, y_tr: {y_tr.shape}")
print(f"validation subset - x_vl: {x_val.shape}, y_vl: {y_val.shape}")
print(f"test subset - x_ts: {x_ts.shape}, y_ts: {y_ts.shape}")

data_in = {
    'x_tr': x_tr, 'y_tr': y_tr,
    'x_vl': x_val, 'y_vl': y_val,
    'x_ts': x_ts, 'y_ts': y_ts,
}

#Executing scaling function
data_s, scaler = escalar_dataset(data_in, col_ref="pollution")
x_tr_s, y_tr_s = data_s['x_tr_s'], data_s['y_tr_s']
x_vl_s, y_vl_s = data_s['x_vl_s'], data_s['y_vl_s']
x_ts_s, y_ts_s = data_s['x_ts_s'], data_s['y_ts_s']

#Visualization of the distributions of the variables
labels = ['pollution','dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
positions = np.arange(len(labels))

fig, ax = plt.subplots(figsize = (12 ,5))
for i in range(data_in['x_tr'].shape[2]):
    ax.violinplot(x_tr_s[:,:,i].flatten(), positions = [i])
    ax.violinplot(x_vl_s[:,:,i].flatten(), positions = [i])
    ax.violinplot(x_ts_s[:,:,i].flatten(), positions = [i])

ax.set_xticks(positions)
plt.title("Data distribution per each variable and partition")
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
plt.title("Data distribution per each variable and partition (No overlapping)")
plt.show()

#Targe feature visualization
fig, ax = plt.subplots(figsize = (7 ,4))
ax.violinplot(y_tr_s[:,:,0].flatten())
ax.violinplot(y_vl_s[:,:,0].flatten())
ax.violinplot(y_ts_s[:,:,0].flatten())

ax.autoscale()
plt.title("Target variable distribution")
plt.show()

#Seed and determinism for reproductibility
tf.random.set_seed(12896)
np.random.seed(12896)
tf.config.experimental.enable_op_determinism()

#Generating model
unds = 140
input = (x_tr_s.shape[1],x_tr_s.shape[2])
mod = Sequential()
mod.add(LSTM(units=unds, input_shape=input))
mod.add(Dense(paquete_sal, activation="linear"))

optim = RMSprop(learning_rate=0.00005)
mod.compile(optimizer=optim, loss=tf.keras.losses.MeanSquaredError())
mod.summary()

#Model training
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

#Loss curve visualization for tracking overfit
plt.plot(historia.history['loss'], label = "traning RMSE", color="teal")
plt.plot(historia.history['val_loss'], label = "Validation RMSE", color="purple")
plt.title("Loss curve")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend()
plt.show()

#Model´s evaluation metrics 
RMSE_tr = mod.evaluate(x_tr_s, y_tr_s, verbose=0)
RMSE_vl = mod.evaluate(x_vl_s, y_vl_s, verbose=0)
RME_ts = mod.evaluate(x_ts_s, y_ts_s, verbose=0)
print(f"RMSE de entrenamiento: {RMSE_tr:.4f}")
print(f"RMSE de validación: {RMSE_vl:.4f}")
print(f"RMSE de test: {RME_ts:.4f}")

#Making predictions
y_ts_pred = predecir(x_ts_s, mod, scaler)
y_ts_pred.shape

#Assessing error rate and Plotting the difference between predicted and actual values as a metric of error
y_ts_flat = y_ts.flatten()
error_pred =  y_ts_pred - y_ts_flat
errores = pd.DataFrame(data={"y_true":y_ts_flat, "y_pred":y_ts_pred})
plt.plot(error_pred, label = "Error predictions", color = "red")
plt.title("Error predictions")
plt.xlabel("Samples")
plt.ylabel("Error")
plt.legend()
plt.grid()
plt.show()

#Visualization of the predicted and actual series.
plt.figure(figsize=(10, 6))
plt.plot(errores["y_true"], label="Real values", color ="red", ls = ":")
plt.plot(errores["y_pred"], label= "Predicted values", color = "teal", ls = "--", alpha = 0.6)
plt.title("Real values vs predictions")
plt.xlabel("Time")
plt.ylabel("Pollution")
plt.legend()
plt.grid()
plt.show()

#Feature importance visualization
PFI_df = pfi_mod(mod, x_ts_s, y_ts_s, df.columns)
ax = sns.barplot(x="PFI", y="Variable", data = PFI_df , palette="turbo")
ax.set_xlabel("Relative importance")
plt.title("Relative importance per each feature")
plt.grid()
plt.show()

#Model saving
mod.save("Pollution:TS_model.h5")