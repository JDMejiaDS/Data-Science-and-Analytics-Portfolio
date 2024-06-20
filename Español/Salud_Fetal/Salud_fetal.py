#imporatndo dependencias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
import random

def encode_inverso(df):
    return df.idxmax(axis=1)


raw_df = pd.read_csv("fetal_health.csv")
random.seed(28)

#mapeando la variable respuesta para mayor comodidad de manipulación
raw_df["fetal_health"] = raw_df["fetal_health"].apply(lambda x : int(x))
raw_df["fetal_health"] = raw_df["fetal_health"].map({1:0,2:1,3:2})

#visualización de la frecuencia de cada clase
sns.countplot(x= raw_df["fetal_health"],hue =raw_df["fetal_health"], legend=False)
plt.xticks(ticks=plt.xticks()[0], labels = ["sanos", "sospechosos", "patológicos"])

#computación de la correlación de Kendall-Tau y Pearson
sns.heatmap(raw_df.corr(method="kendall"), cmap="icefire")
sns.heatmap(raw_df.corr(), cmap="icefire")

#Seleccionando variables
df_kendall = raw_df[["accelerations","uterine_contractions","prolongued_decelerations","abnormal_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability","mean_value_of_long_term_variability", "fetal_health"
    ]]
df_pearson = raw_df[["accelerations", "uterine_contractions","prolongued_decelerations","abnormal_short_term_variability",
                    "percentage_of_time_with_abnormal_long_term_variability","mean_value_of_long_term_variability","histogram_mode",
                    "histogram_mean", "histogram_median","fetal_health"
                    ]]

#aplicando SMOTE y generando particiones en los conjuntos de datos
X_k = df_kendall.drop("fetal_health", axis = 1)
X_p = df_pearson.drop("fetal_health", axis = 1)
y_k= df_kendall["fetal_health"]
y_p= df_pearson["fetal_health"]
sampling = {0:1655, 1:1100, 2:1100}
X_resamp_k, y_resamp_k = SMOTE(sampling_strategy=sampling).fit_resample(X_k, y_k)
X_resamp_p, y_resamp_p = SMOTE(sampling_strategy=sampling).fit_resample(X_p, y_p)

#estandarización de los datos
scaler = StandardScaler()
X_res_k = scaler.fit_transform(X_resamp_k)
X_res_p = scaler.fit_transform(X_resamp_p)

#Visualización de clases blalanceadas
yk_df = pd.DataFrame(y_resamp_k)
yp_df = pd.DataFrame(y_resamp_p)
sns.countplot(x = yp_df["fetal_health"], hue = yp_df["fetal_health"], palette="viridis", legend = False)
plt.xticks(ticks=plt.xticks()[0], labels = ["sanos", "sospechosos", "paológicos"])
plt.show()

#Primero se crean las divisiones de entrenamiento y test
X_trainK, X_testK, y_trainK, y_testK = train_test_split(X_res_k, y_resamp_k, test_size=0.25)
X_trainp, X_testp, y_trainp, y_testp = train_test_split(X_res_p, y_resamp_p, test_size=0.25)

#Posteriormente se extrae un conjunto de de datos de validación para cada conjunto de datos
X_trainK= X_trainK[:int(len(X_trainK)*0.8)]
XVal_K = X_trainK[int(len(X_trainK)*0.8):]
y_trainK= y_trainK[:int(len(y_trainK)*0.8)]
yVal_K = y_trainK[int(len(y_trainK)*0.8):]
X_trainp= X_trainp[:int(len(X_trainp)*0.8)]
XVal_p = X_trainp[int(len(X_trainp)*0.8):]
y_trainp= y_trainp[:int(len(y_trainp)*0.8)]
yVal_p = y_trainp[int(len(y_trainp)*0.8):]

#Creación de modelos de clasificación SVM para ambos conjuntos de datos
svm_classifier= SVC(kernel = "poly", random_state=1996)
svm_k = svm_classifier.fit(X_trainK, y_trainK)
SVMK_pred = svm_k.predict(X_testK)
svm_p = svm_classifier.fit(X_trainp, y_trainp)
SVMP_pred = svm_p.predict(X_testp)

#Métricas de desempeño para las predicciones sobre ambos conjuntos de datos
print(classification_report(y_testK, SVMK_pred))
print(classification_report(y_testp, SVMP_pred))

#Creación de modelos de clasificación de Random forest para ambos conjuntos de datos
rfor = RandomForestClassifier(n_estimators= 150, random_state= 28)
Rand_fK = rfor.fit(X_trainK, y_trainK)
RFK_pred = Rand_fK.predict(X_testK)
Rand_fP = rfor.fit(X_trainp, y_trainp)
RFP_pred = Rand_fP.predict(X_testp)

#Visualización de métricas de rendimiento 
print(classification_report(y_testK, RFK_pred))
print(classification_report(y_testp, RFP_pred))

#Visualización de matrices de confusion para ambos modelos
plt.title("Matriz de confusión para bosque aleatorio cor. de Kendall")
sns.heatmap(confusion_matrix(y_testK, RFK_pred), cmap="icefire", annot = True)
plt.xticks(ticks= plt.xticks()[0], labels = ["Sano", "Sospechoso", "Patologico"])
plt.yticks(ticks = plt.yticks()[0], labels= ["Pred_sano","Pred_sospe", "Pred_Patologico"], rotation = 45)
plt.show()
plt.title("Matriz de confusión para bosque aleatorio cor. de Pearson")
sns.heatmap(confusion_matrix(y_testp, RFP_pred), cmap="icefire", annot = True)
plt.xticks(ticks= plt.xticks()[0], labels = ["Sano", "Sospechoso", "Patologico"])
plt.yticks(ticks = plt.yticks()[0], labels= ["Pred_sano","Pred_sospe", "Pred_Patologico"], rotation = 45)
plt.show()

#Elaboración de modelos de machine learning

modK = Sequential()
modK.add(Dense(100, activation = "relu", input_shape = (6,)))
modK.add(Dropout(0.3))
modK.add(Dense(50, activation="relu"))
modK.add(Dropout(0.2))                
modK.add(Dense(3, activation="softmax"))
modK.compile(optimizer = "adam", loss='sparse_categorical_crossentropy', metrics = ["accuracy"])
hist_k=modK.fit(X_trainK, y_trainK, epochs=100, batch_size = 30, verbose=0,validation_data=(XVal_K,yVal_K))

modP = Sequential()
modP.add(Dense(100, activation = "relu", input_shape = (9,)))
modP.add(Dropout(0.3))
modP.add(Dense(50, activation="relu"))
modP.add(Dropout(0.2))                
modP.add(Dense(3, activation="softmax"))
modP.compile(optimizer = "adam", loss='sparse_categorical_crossentropy', metrics = ["accuracy"])
hist_p = modP.fit(X_trainp, y_trainp, epochs=100, batch_size = 30, verbose=0, validation_data=(XVal_p, yVal_p))

#Visualización de curvas de pérdida
fig = plt.figure()
plt.plot(hist_k.history['loss'], color='purple', label='loss', ls = "--")
plt.plot(hist_k.history['val_loss'], color='teal', label='val_loss', ls = ":")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Curva de aprendizaje para Kendall')
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(hist_p.history['loss'], color='red', label='loss', ls = "--")
plt.plot(hist_p.history['val_loss'], color='teal', label='val_loss', ls = ":")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Curva de aprendizaje para Pearson')
plt.legend()
plt.show()

#generando predicciones
K_pred = modK.predict(X_testK)
P_pred = modP.predict(X_testp)
K_pred=pd.DataFrame(K_pred, columns=["1","2", "3"])
P_pred=pd.DataFrame(P_pred, columns=["1","2", "3"])
K_pred = K_pred.apply(lambda x : round(x))
P_pred = P_pred.apply(lambda x : round(x))
K_prNN = encode_inverso(K_pred)
P_prNN = encode_inverso(P_pred)

#remapeando valores de variable respuesta a sus cifras originales para evitar confusión en el reporte de clasificación
y_testp= pd.DataFrame(y_testp,columns=["fetal_health"])
y_testK= pd.DataFrame(y_testK,columns=["fetal_health"])
y_testK["fetal_health"] = y_testK["fetal_health"].map({0:1, 1:2, 2:3})
y_testp["fetal_health"] = y_testp["fetal_health"].map({0:1, 1:2, 2:3})

#Visualización de métricas de rendimiento
print(classification_report(y_testK, K_prNN))
print(classification_report(y_testp, P_prNN))

#Visualización de matrices de confusión para cada conjunto de datos
plt.title("Matriz de confusión para deep learning cor. de Kendall")
sns.heatmap(confusion_matrix(y_testK, K_prNN), cmap="turbo", annot = True)
plt.xticks(ticks= plt.xticks()[0], labels = ["Sano", "Sospechoso", "Patologico"])
plt.yticks(ticks = plt.yticks()[0], labels= ["Pred_sano","Pred_sospe", "Pred_Patologico"], rotation = 45)
plt.show()
plt.title("Matriz de confusión para deep learning cor. de Kendall")
sns.heatmap(confusion_matrix(y_testp, P_prNN), cmap="turbo", annot = True)
plt.xticks(ticks= plt.xticks()[0], labels = ["Sano", "Sospechoso", "Patologico"])
plt.yticks(ticks = plt.yticks()[0], labels= ["Pred_sano","Pred_sospe", "Pred_Patologico"], rotation = 45)
plt.show()

modP.save("Predictor_salud_entrenado.h5")




