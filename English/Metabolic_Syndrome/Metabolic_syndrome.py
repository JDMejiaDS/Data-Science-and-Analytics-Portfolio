import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, levene
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
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
import warnings
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")

def imput_marital(data):
    if pd.isna(data):
        data = np.random.choice(class_prob.index, p=class_prob.values)
    return data

raw_df = pd.read_csv("Metabolic_Syndrome.csv", index_col=0)
raw_df.head()
df = raw_df.copy()

#Class count visualization for Metabolic syndrome, our target feature (MS)
sns.countplot(x = "MetabolicSyndrome", data = df, palette = "winter")
plt.xticks(ticks =plt.xticks()[0], labels = ["Healthy", "Sick"])
plt.show()

#Bivariate distribution visualizations MS/age
sns.boxplot(x = "MetabolicSyndrome", y="Age", data = df , palette ="coolwarm")
plt.grid(True)
plt.xticks(ticks =plt.xticks()[0], labels = ["Healthy", "Sick"])
plt.show()
sns.displot(x= "Age", data = df, hue = "MetabolicSyndrome", rug = True, kind = "kde", palette = "coolwarm", legend = False)
plt.title("Age distribution by age given health status")
plt.legend(labels = ["Healthy","Sick"])
plt.show()

#statistics computation MS/age
H_age = df[df["MetabolicSyndrome"]==0]["Age"]
S_age = df[df["MetabolicSyndrome"]==1]["Age"]
stat_h, healthy_p_value = kstest(H_age, "norm")
stat_s, sick_p_value = kstest(S_age, "norm")
print(f"The p value for Kolmogorov test for healthy individuals is: {healthy_p_value}")
print(f"The p value for Kolmogorov test for sick individuals is:  {sick_p_value}")
stat_levene, p_value_levene = levene(H_age, S_age)
print(f'Levene test p-value is :{p_value_levene}')
MWU_stati, p_value_age = mannwhitneyu(S_age, H_age)
print(f'The p-value asociated to U de Mann - Whitney test is :{p_value_age}')

#Bivariate distribution visualizations MS/Triglycerides 
sns.boxplot(x = "MetabolicSyndrome", y="Triglycerides", data = df, palette="inferno")
plt.xticks(ticks =plt.xticks()[0], labels = ["Healthy", "Sick"])
plt.grid(True)
plt.figure(figsize=(12,6))
sns.displot(x= "Triglycerides", data = df, hue = "MetabolicSyndrome", rug = True, legend=False, palette="icefire")
plt.title("Tryglicerides distribution given health status")
plt.legend(["Healthy","Sick"])
plt.show()

#statistics computation MS/Triglycerides
T_healthy = df[df["MetabolicSyndrome"]==0]["Triglycerides"]
T_sick = df[df["MetabolicSyndrome"]==1]["Triglycerides"]
stat1, T_healthy_p_value = shapiro(T_healthy)
stat2, T_sick_p_value = shapiro(T_sick)
print(f"The p value for Shapiro-Wilk test in healthy individuals is: {T_healthy_p_value}")
print(f"The p value for Shapiro-Wilk test in sick individuals is: {T_sick_p_value}")
stat_levene2, T_p_value_levene = levene(T_healthy, T_sick)
print(f'Levene test p-value es :{T_p_value_levene}')
MWU_stat2, p_value_Tglcs = mannwhitneyu(T_healthy, T_sick)
print(f'The p-value for U de Mann - Whitney test is :{p_value_Tglcs}')

#Bivariate distribution visualizations MS/BMI
sns.boxplot(x = "MetabolicSyndrome", y="BMI", data = df, palette = "viridis")
plt.xticks(ticks =plt.xticks()[0], labels = ["Healthy", "Sick"])
plt.grid(True)
plt.figure(figsize=(12,6))
sns.displot(x= "BMI", data = df, hue = "MetabolicSyndrome", rug = True, palette = "plasma", legend=False)
plt.title("BMI distribution given health status")
plt.legend(labels = ["Healthy","sick"])
plt.show()

#Bivariate distribution visualizations MS/HDL
sns.boxplot(x = "MetabolicSyndrome", y="HDL", data = df , palette = "magma")
plt.xticks(ticks =plt.xticks()[0], labels = ["Healthy", "Sick"])
plt.grid(True)
plt.figure(figsize=(12,6))
sns.displot(x= "HDL", data = df, hue = "MetabolicSyndrome", rug = True, palette ="icefire", legend=False)
plt.title("HDL distribution given health status")
plt.legend(labels = ["Healthy", "Sick"])
plt.show()

#Bivariate distribution visualizations MS/BloodGlucose
sns.boxplot(x = "MetabolicSyndrome", y="BloodGlucose", data = df , palette = "turbo")
plt.xticks(ticks =plt.xticks()[0], labels = ["Healthy", "Sick"])
plt.grid(True)
sns.displot(x= "BloodGlucose", data = df, hue = "MetabolicSyndrome", rug = True, kind = "kde", palette ="turbo",legend = False)
plt.title("Blood glucose distribution given health status")
plt.legend(labels = ["Healthy", "Sick"])
plt.show()

#Numeric missing data processing
imput = KNNImputer(n_neighbors= 6)
df_ip = df.copy()
faltantes_num = df[["Income", "WaistCirc", "BMI"]]
imputed = imput.fit_transform(faltantes_num)
imputed_val = pd.DataFrame(imputed, columns=["Income", "WaistCirc", "BMI"])
df_ip = df_ip.drop(["Income","WaistCirc","BMI"], axis = 1)
df_ip.reset_index(inplace=True)
df_clas = pd.concat([df_ip,imputed_val], axis=1)

#Categorical missing data processing
class_prob = df_clas["Marital"].value_counts(normalize=True)
df_clas["Marital"] = df_clas["Marital"].apply(lambda x: imput_marital(x))
df_clas= df_clas[['Age', 'Sex', 'UrAlbCr', 'UricAcid',
        'BloodGlucose', 'HDL', 'Triglycerides', 'Income',
        'WaistCirc', 'BMI', 'MetabolicSyndrome'
        ]]

#Data scaling for modeling
scaler = MinMaxScaler()
df_clas["Sex"] = df_clas["Sex"].map({"Male":1,"Female":0})
df_No_sex = df_clas.drop(["Sex","MetabolicSyndrome"], axis = 1)
df = scaler.fit_transform(df_No_sex)
df = pd.DataFrame(df, columns=['Age', 'UrAlbCr', 'UricAcid',
    'BloodGlucose', 'HDL', 'Triglycerides', 'Income',
    'WaistCirc', 'BMI'
    ])
df_NN = pd.concat([df,df_clas[["Sex","MetabolicSyndrome"]]], axis = 1)

#Correlation visualization
plt.figure(figsize=(9,6))
sns.heatmap(df_NN.corr(), cmap = "coolwarm", annot = True)

#Selected feature visualization
df_clas.corr()["MetabolicSyndrome"][:-1].plot(kind="bar", color="purple")
plt.title("Correlation with MetabolicSyndrome variable")
plt.xticks(rotation=45)
plt.axhline(y = 0.3, color='r', linestyle='--', label='Horizontal sup')
plt.axhline(y = -0.3, color='r', linestyle='--', label='Horizontal inf')
plt.show()

#PCA for dimentional reduction and visualization
pca = PCA(n_components=2)
meta_synd_PCA = pca.fit_transform(df_NN)
meta_synd = pd.DataFrame(meta_synd_PCA)
plt.figure(figsize=(8,6))
sns.scatterplot(x = meta_synd[0], y =meta_synd[1], hue =df_NN['MetabolicSyndrome'], palette="viridis")
plt.xlabel('1st PC')
plt.ylabel('2nd PC')
plt.title("PCA Metabolic syndrome")
plt.legend(labels = ["Healthy","Sick"])
plt.show()

#Number of clusters selection
WCSS = []
for i in range(1,16):
    kmeans = KMeans(i, n_init="auto")
    kmeans.fit(df_NN)
    WCSS_inertia= kmeans.inertia_
    WCSS.append(WCSS_inertia)
num = range(1,16)
plt.plot(num,WCSS)
plt.title("Elbow for number of clusters selection")
plt.ylabel("Intra-clusters error")
plt.xlabel("Num of clusters")
plt.show()

#Kmeans aplication for Clustering 
kmeans = KMeans(4, n_init="auto")
kmeans.fit(df_NN)
df_clas["Cluster"] = kmeans.labels_
df_clas.head()

#Clusters visualizations
kmeans = KMeans(4, n_init="auto")
kmeans.fit(df_NN)
df_clas["Cluster"] = kmeans.labels_
df_clas.head()

#Variance explained by component visualization
Comps = pd.DataFrame(pca.explained_variance_ratio_*100, columns=["%_var_explicated"])
plt.figure(figsize=(10,4))
plt.title("Contribution of each variable to total variance explained by each component")
sns.heatmap(pca.components_, annot=True, cmap="icefire")
plt.yticks(ticks= plt.yticks()[0], labels =["1st comp","2nd comp"], rotation = 45)
plt.xticks(ticks=plt.xticks()[0], labels=pca.feature_names_in_, rotation = 45)
plt.show()

#Handling class imbalance
X = df_NN.drop("MetabolicSyndrome", axis = 1)
y = df_NN["MetabolicSyndrome"]
X_resamp, y_resamp = SMOTE().fit_resample(X,y)

#Creating partitions on the dataset
X_train, X_test, y_train, y_test = train_test_split(X_resamp, y_resamp, test_size=0.15, shuffle=True)
tr_len = int(len(X_train)*0.85)
val_len = int(len(X_train)*0.15)
print(tr_len, val_len)
X_tr = X_train[:tr_len]
X_val = X_train[tr_len:]
y_tr = y_train[:tr_len]
y_val = y_train[tr_len:]
print(X_tr.shape, X_val.shape, y_tr.shape,y_val.shape)

#Model creation and traning
modelo = Sequential([
    Dense(64, activation='relu', input_shape=(10,)), Dropout(0.4),
    Dense(32, activation='relu'), Dropout(0.3),
    Dense(1, activation='sigmoid')
])
modelo.compile(optimizer="adam" , loss='binary_crossentropy', metrics=['accuracy'])
modelo.fit(X_tr, y_tr,epochs=100, batch_size=32, verbose = 0, validation_data=(X_val, y_val))

#Loss curve visualization
plt.title("Loss vs Epochs")
plt.plot(modelo.history.history["loss"], label ="Loss", linestyle = "dashed", color = "red" )
plt.plot(modelo.history.history["val_loss"], label ="Validation_loss", linestyle = "dashed", color = "teal" )
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

#Predictions 
Predicts = pd.DataFrame(modelo.predict(X_test))
Predicts = Predicts.apply(lambda x : round(x))

#Classification Matrix
print(classification_report(y_test, Predicts))

#Confusion MAtrix visualization
plt.title("Confusion matrix")
sns.heatmap(confusion_matrix(y_test, Predicts), annot=True, cmap = "icefire")
plt.yticks(ticks= plt.yticks()[0], labels =["Healthy","Sick"], rotation = 45)
plt.xticks(ticks = plt.xticks()[0], labels = ["Pred. Healthy","Pred. Sick"], rotation = 45)
plt.show()

#Model save
modelo.save('MS_trained_model.h5')