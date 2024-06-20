import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import warnings
from scipy.stats import kstest, levene, mannwhitneyu, shapiro
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
tf.random.set_seed(96)
warnings.filterwarnings('ignore')

def imput_marital(data):
    if pd.isna(data):
        data = np.random.choice(class_prob.index, p=class_prob.values)
    return data

raw_df = pd.read_csv("Metabolic Syndrome.csv", index_col=0)
raw_df.head()
df = raw_df.copy()
df.describe()

#visualización de la frecuencias de las clases en MetabolicSyndrome(MS)
sns.countplot(x = "MetabolicSyndrome", data = df, palette = "winter")
plt.grid(True)
plt.xticks(ticks =plt.xticks()[0], labels = ["Sanos", "Enfermos"])
plt.show()

#Visualización bivariada de MS/age
sns.boxplot(x = "MetabolicSyndrome", y="Age", data = df , palette ="coolwarm")
plt.grid(True)
plt.xticks(ticks =plt.xticks()[0], labels = ["Sanos", "Enfermos"])
plt.show()
sns.displot(x= "Age", data = df, hue = "MetabolicSyndrome", rug = True, kind = "kde", palette = "coolwarm", legend = False)
plt.title("Distribución edad por estado de salud")
plt.legend(labels = ["Enfermos","Sanos"])
plt.show()

#Determinando diferencias entre grupos em función de su edad
sanos_edad = df[df["MetabolicSyndrome"]==0]["Age"]
enfermos_edad = df[df["MetabolicSyndrome"]==1]["Age"]
estat_sanos, sanos_p_value = kstest(sanos_edad, "norm")
estat_enfermos, enfermos_p_value = kstest(enfermos_edad, "norm")
print(f"el valor p para la prueba de Kolmogorov en los individuos sanos es: {sanos_p_value}")
print(f"el valor p para la prueba de Kolmogorov en los individuos enfermos es: {enfermos_p_value}")
stat_levene, p_valor_levene = levene(sanos_edad, enfermos_edad)
print(f'Levene test p-value es :{p_valor_levene}')

MWU_stati, p_valor_age = mannwhitneyu(enfermos_edad, sanos_edad)
print(f'El p-value asociado a la U de Mann - Whitney es :{p_valor_age}')

#Visualización bivariada de MS/Tryglicerides
sns.boxplot(x = "MetabolicSyndrome", y="Triglycerides", data = df, palette="inferno")
plt.xticks(ticks =plt.xticks()[0], labels = ["Sanos", "Enfermos"])
plt.grid(True)
sns.displot(x= "Triglycerides", data = df, hue = "MetabolicSyndrome", rug = True, legend=False, palette="icefire")
plt.title("Distribución triglicéridos por estado de salud")
plt.legend(["Sanos","Enfermos"])
plt.show()

#Determinando diferencias entre grupos en función de sus trigliceridos
T_saludable = df[df["MetabolicSyndrome"]==0]["Triglycerides"]
T_enfermo = df[df["MetabolicSyndrome"]==1]["Triglycerides"]
stat1, T_saludable_p_value = shapiro(T_saludable)
stat2, T_enfermo_p_value = shapiro(T_enfermo)
print(f"el valor p para la prueba de Shapiro-Wilk en los individuos sanos es: {T_saludable_p_value}")
print(f"el valor p para la prueba de Shapiro-Wilk en los individuos enfermos es: {T_enfermo_p_value}")
stat_levene2, T_p_value_levene = levene(T_saludable, T_enfermo)
print(f'Levene test p-value es :{T_p_value_levene}')

MWU_stat2, p_value_Tglcs = mannwhitneyu(T_saludable, T_enfermo)
print(f'El p-value para la U de Mann - Whitney es :{p_value_Tglcs}')

#Visualización bivariada de MS/BMI
sns.boxplot(x = "MetabolicSyndrome", y="BMI", data = df, palette = "viridis")
plt.xticks(ticks =plt.xticks()[0], labels = ["Sanos", "Enfermos"])
plt.grid(True)

plt.figure(figsize=(12,6))
sns.displot(x= "BMI", data = df, hue = "MetabolicSyndrome", rug = True, palette = "plasma", legend=False)
plt.title("Distribución BMI en función del estado de salud")
plt.legend(labels = ["Sanos","Enfermos"])
plt.show()

#Visualización bivariada de MS/HDL
sns.boxplot(x = "MetabolicSyndrome", y="HDL", data = df , palette = "magma")
plt.xticks(ticks =plt.xticks()[0], labels = ["Sanos", "Enfermos"])
plt.grid(True)

sns.displot(x= "HDL", data = df, hue = "MetabolicSyndrome", rug = True, palette ="viridis", legend=False)
plt.title("Distribución HDL en función del estado de salud")
plt.legend(labels = ["Enfermos","Sanos"])
plt.show()

#Visualización bivariada de MS/BloodGlucose
sns.boxplot(x = "MetabolicSyndrome", y="BloodGlucose", data = df , palette = "turbo")
plt.xticks(ticks =plt.xticks()[0], labels = ["Sanos", "Enfermos"])
plt.grid(True)

sns.displot(x= "BloodGlucose", data = df, hue = "MetabolicSyndrome", rug = True, kind = "kde", palette ="turbo",legend = False)
plt.title("Distribución Glucosa en función del estado de salud")
plt.legend(labels = ["Enfermos","Sanos"])
plt.show()

#verificación de datos faltantes
df[["WaistCirc","BMI", "Marital", "Income"]].isna().sum()

#Imputación de datos faltantes numéricos
imput = KNNImputer(n_neighbors= 6)
df_ip = df.copy()
faltantes_num = df[["Income", "WaistCirc", "BMI"]]
imputed = imput.fit_transform(faltantes_num)
imputed_val = pd.DataFrame(imputed, columns=["Income", "WaistCirc", "BMI"])
df_ip = df_ip.drop(["Income","WaistCirc","BMI"], axis = 1)
df_ip.reset_index(inplace=True)
df_clas = pd.concat([df_ip,imputed_val], axis=1)

#imputaciónde datos faltantes de variable categórica
class_prob = df_clas["Marital"].value_counts(normalize=True)
df_clas["Marital"] = df_clas["Marital"].apply(lambda x: imput_marital(x))

#Escalamiento de datos para modelamiento
scaler = MinMaxScaler()
df_clas["Sex"] = df_clas["Sex"].map({"Male":1,"Female":0})
df_No_sex = df_clas.drop(["Sex","MetabolicSyndrome"], axis = 1)
df = scaler.fit_transform(df_No_sex)
df = pd.DataFrame(df, columns=['Age', 'UrAlbCr', 'UricAcid',
        'BloodGlucose', 'HDL', 'Triglycerides', 'Income',
        'WaistCirc', 'BMI'])
df_NN = pd.concat([df,df_clas[["Sex","MetabolicSyndrome"]]], axis = 1)

#Visualización de la matriz de correlación
plt.figure(figsize=(9,6))
sns.heatmap(df_NN.corr(), cmap = "coolwarm", annot = True)

#Visualización de las variables con correlación significativa
df_clas.corr()["MetabolicSyndrome"][:-1].plot(kind="bar", color="purple")
plt.title("Correlación con la variable MetabolicSyndrome")
plt.xticks(rotation=45)
plt.axhline(y = 0.3, color='r', linestyle='--', label='Línea Horizontal')
plt.axhline(y = -0.3, color='r', linestyle='--', label='Línea Horizontal')
plt.show()

#Aplicación de PCA para reducción dimensional
pca = PCA(n_components=2)
meta_synd_PCA = pca.fit_transform(df_NN)
meta_synd = pd.DataFrame(meta_synd_PCA)
plt.figure(figsize=(8,6))
sns.scatterplot(x = meta_synd[0], y =meta_synd[1], hue =df_NN['MetabolicSyndrome'], palette="viridis")
plt.xlabel('Primer componente principal')
plt.xlabel('Primer componente principal')
plt.ylabel('Segundo componente principal')
plt.title("PCA Síndrome metabólico")
plt.legend(labels = ["Sanos","Enfermos"])
plt.show()

#Aplícación del método del codo para selección del número de clusters
WCSS = []
for i in range(1,16):
    kmeans = KMeans(i, n_init="auto")
    kmeans.fit(df_NN)
    WCSS_inertia= kmeans.inertia_
    WCSS.append(WCSS_inertia)

num = range(1,16)
plt.plot(num,WCSS)
plt.title("Codo para formación de clusters")
plt.ylabel("Error intra-clusters")
plt.xlabel("Número de clusters")
plt.show()

kmeans = KMeans(4, n_init="auto")
kmeans.fit(df_NN)
df_clas["Cluster"] = kmeans.labels_

#Visualización de los clusters
sns.scatterplot(x = meta_synd[0], y =meta_synd[1], data = meta_synd, hue =df_clas['Cluster'], palette="turbo")

#Aporte de cada variable a al total de varianza 
Componentes = pd.DataFrame(pca.explained_variance_ratio_*100, columns=["%_var_explicado"])

#Visualización del porcentaje de varianza explicado por cada variable en cada componente 
plt.figure(figsize=(10,4))
plt.title("Contribución de varianza a cada componente")
sns.heatmap(pca.components_, annot=True, cmap="icefire")
plt.yticks(ticks= plt.yticks()[0], labels =["1ra componente","2da componente"], rotation = 45)
plt.xticks(ticks=plt.xticks()[0], labels=pca.feature_names_in_, rotation = 45)
plt.show()

#Aplicación de SMOTE
X = df_NN.drop("MetabolicSyndrome", axis = 1)
y = df_NN["MetabolicSyndrome"]
X_resamp, y_resamp = SMOTE().fit_resample(X,y)

#Partición del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X_resamp, y_resamp, test_size=0.15, shuffle=True)
tr_len = int(len(X_train)*0.85)
val_len = int(len(X_train)*0.15)
print(tr_len, val_len)
X_tr = X_train[:tr_len]
X_val = X_train[tr_len:]
y_tr = y_train[:tr_len]
y_val = y_train[tr_len:]
print(X_tr.shape, X_val.shape, y_tr.shape,y_val.shape)
print(X_test.shape, y_test.shape)

#Elaboración del modelo de deep learning
modelo = Sequential([
    Dense(64, activation='relu', input_shape=(10,)), Dropout(0.4),
    Dense(32, activation='relu'), Dropout(0.3),
    Dense(1, activation='sigmoid')
])
modelo.compile(optimizer="adam" , loss='binary_crossentropy', metrics=['accuracy'])
modelo.fit(X_tr, y_tr,epochs=100, batch_size=32, verbose = 0, validation_data=(X_val, y_val))

#Visualización de la curva de pérdida 
plt.title("Pérdida por ciclo de entrenamiento")
plt.plot(modelo.history.history["loss"], label ="Pérdida", linestyle = "dashed", color = "red" )
plt.plot(modelo.history.history["val_loss"], label ="Pérdida validación", linestyle = "dashed", color = "teal" )
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

#Generando predicciones 
Predicciones = pd.DataFrame(modelo.predict(X_test))
Predicciones = Predicciones.apply(lambda x : round(x))

#Métricas de rendimiento del modelo
print(classification_report(y_test, Predicciones))

#Visualización de las matrices 
plt.title("Matriz de confusión")
sns.heatmap(confusion_matrix(y_test, Predicciones), annot=True, cmap = "icefire")
plt.yticks(ticks= plt.yticks()[0], labels =["Sanos","Enfermos"], rotation = 45)
plt.xticks(ticks = plt.xticks()[0], labels = ["Pred. Sanos", "Pred. Enfermos"], rotation = 45)
plt.show()






