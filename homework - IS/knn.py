import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.neighbors import *
from KNearestNeighbors import *
import datetime as time


# Podesavanje prikaza
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', None)

path = 'knn_img/'

if not os.path.exists(path):
    os.makedirs(path)

# Citanje iz fajla
data = pd.read_csv('datasets/car_state.csv')

#################################################
# Prikaz prvih i poslednjih 5 redova tabele
#################################################
print("\nPrvih 5 redova:")
print(data.head(), end='\n\n')  # default n=5
print("Poslednjih 5 redova:")
print(data.tail(n=5), end='\n\n')
print('-------------------------------------------------------------------------------------\n')

#################################################
# Prikaz dodatnih informacija o tabeli
#################################################
print("Osnovne informacije o tabeli:\n")
data.info(memory_usage='deep')
print('\n Informacije o atributima:\n', data.describe(include=np.object_), end='\n\n')
print('-------------------------------------------------------------------------------------')

#################################################
# Graficka predstava zavisnosti izlaznog atributa od ulaznih
#################################################
plt.figure(constrained_layout=True)
sb.displot(data, x='buying_price', hue='status', multiple='fill')
plt.savefig(path+'price.png')
plt.figure(constrained_layout=True)
sb.displot(data, x='maintenance', hue='status', multiple='fill')
plt.savefig(path+'maintenance.png')
plt.figure(constrained_layout=True)
sb.displot(data, x='doors', hue='status', multiple='fill')
plt.savefig(path+'doors.png')
plt.figure(constrained_layout=True)
sb.displot(data, x='seats', hue='status', multiple='fill')
plt.savefig(path+'seats.png')
plt.figure(constrained_layout=True)
sb.displot(data, x='trunk_size', hue='status', multiple='fill')
plt.savefig(path+'trunck.png')
plt.figure(constrained_layout=True)
sb.displot(data, x='safety', hue='status', multiple='fill')
plt.savefig(path+'safety.png')
plt.close('all')

#################################################
# Korelacija atributa
#################################################
plt.figure(constrained_layout=True)
corr_data = data.copy()
le = LabelEncoder()
corr_data.buying_price = le.fit_transform(data.buying_price)
corr_data.maintenance = le.fit_transform(data.maintenance)
corr_data.doors = le.fit_transform(data.doors)
corr_data.seats = le.fit_transform(data.seats)
corr_data.trunk_size = le.fit_transform(data.trunk_size)
corr_data.safety = le.fit_transform(data.safety)
corr_data.status = le.fit_transform(data.status)
sb.heatmap(corr_data.corr(), annot=True, linewidths=1, fmt='.2f', square=True, vmin=-1, vmax=1)
plt.savefig(path+'correlation.png')
plt.close('all')

#################################################
# Izbor atributa za treniranje
#################################################
data_train1 = corr_data.loc[:, ['buying_price', 'maintenance', 'doors', 'seats', 'trunk_size', 'safety']]
label1 = corr_data['status']

print("\nAtributi koji se koriste za treniranje ML:\n", data_train1)
print('\nIzlazni atribut: status\n', label1)
print('\n-------------------------------------------------------------------------------------\n')
#################################################
# Treniranje modela
#################################################
X_train, X_test, y_train, y_test = train_test_split(data_train1, label1, train_size=0.7, random_state=123, shuffle=False)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# ODREDITI BTOJ SUSEDA
#######################
num = 21
#######################


# KNN iz sklearn modula
knn = KNeighborsClassifier(n_neighbors=num)
knn.fit(X_train, y_train)
y_prediction = pd.DataFrame(data=knn.predict(X_test), columns=['knn_prediction'])
knn1_output = X_test.join(y_prediction)
knn1_output = knn1_output.join(y_test)
print("\nKNN bez prosirivanja kolona:\n", knn1_output)
rez1 = knn.score(X_test, y_test)

#################################################
# KNN iz sklearn modula SA PROSIRIVANJEM KOLONA
#################################################
ohe = OneHotEncoder(dtype=int, sparse=False)

buying_price = ohe.fit_transform(data.buying_price.to_numpy().reshape(-1, 1))
data.drop(columns='buying_price', inplace=True)
data = data.join(pd.DataFrame(data=buying_price, columns=ohe.get_feature_names_out(['buying_price'])))

maintenance = ohe.fit_transform(data.maintenance.to_numpy().reshape(-1, 1))
data.drop(columns='maintenance', inplace=True)
data = data.join(pd.DataFrame(data=maintenance, columns=ohe.get_feature_names_out(['maintenance'])))

doors = ohe.fit_transform(data.doors.to_numpy().reshape(-1, 1))
data.drop(columns='doors', inplace=True)
data = data.join(pd.DataFrame(data=doors, columns=ohe.get_feature_names_out(['doors'])))

seats = ohe.fit_transform(data.seats.to_numpy().reshape(-1, 1))
data.drop(columns='seats', inplace=True)
data = data.join(pd.DataFrame(data=seats, columns=ohe.get_feature_names_out(['seats'])))

trunk_size = ohe.fit_transform(data.trunk_size.to_numpy().reshape(-1, 1))
data.drop(columns='trunk_size', inplace=True)
data = data.join(pd.DataFrame(data=trunk_size, columns=ohe.get_feature_names_out(['trunk_size'])))

safety = ohe.fit_transform(data.safety.to_numpy().reshape(-1, 1))
data.drop(columns='safety', inplace=True)
data = data.join(pd.DataFrame(data=safety, columns=ohe.get_feature_names_out(['safety'])))
data.status = le.fit_transform(data.status)

data_train2 = data.loc[:, ['buying_price_high', 'buying_price_low', 'buying_price_medium', 'buying_price_very high',
                          'maintenance_high', 'maintenance_low', 'maintenance_medium', 'maintenance_very high',
                          'doors_2', 'doors_3', 'doors_4', 'doors_5 or more', 'seats_2', 'seats_4', 'seats_5 or more',
                          'trunk_size_big', 'trunk_size_medium', 'trunk_size_small',
                          'safety_high', 'safety_low', 'safety_medium']]
label2 = data['status']

X_train, X_test, y_train, y_test = train_test_split(data_train2, label2, train_size=0.7, random_state=123, shuffle=False)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

knn.fit(X_train, y_train)
y_prediction = pd.DataFrame(data=knn.predict(X_test), columns=['knn_prediction'])
knn2_output = X_test.join(y_prediction)
knn2_output = knn2_output.join(y_test)
print("\nKNN bez prosirivanja kolona:\n", knn2_output)
rez2 = knn.score(X_test, y_test)

#################################################
# KNN implementacija SA PROSIRIVANJEM KOLONA
#################################################
knn_impl = KNearestNeighbors(n_neighbors=num)
knn_impl.get_data(X_train, y_train)
y_prediction, rez3 = knn_impl.predict_and_score(X_test, y_test)

y_prediction = pd.DataFrame(data=y_prediction, columns=['knn_prediction'])
knn3_output = X_test.join(y_prediction)
knn3_output = knn3_output.join(y_test)
print("\nKNN implementacija sa prosirivanjem kolona:\n", knn3_output)

print("\nAko se posmatra ", num, ' najblizih suseda, preciznisti su sledece:\n')
print("KNN bez prosirenja tabele, preciznost:", rez1)
print("KNN sa prosirenjem tabele, preciznost:", rez2)
print("KNN implementaacija sa prosirenjem tabele, preciznost:", rez3)

#######################################################
# Performanse algoritama
#######################################################
rez_1 = []
rez_2 = []
rez_3 = []
n = []
x11, x12, y11, y12 = train_test_split(data_train1, label1, train_size=0.7, random_state=123, shuffle=False)
x21, x22, y21, y22 = train_test_split(data_train2, label2, train_size=0.7, random_state=123, shuffle=False)
s = time.datetime.now()
print("Performance Simulation:")
for j in range(1, 52, 5):
    knn_1 = KNeighborsClassifier(n_neighbors=j)
    knn_2 = KNeighborsClassifier(n_neighbors=j)
    knn_3 = KNearestNeighbors(n_neighbors=j)
    knn_1.fit(x11, y11)
    knn_2.fit(x21, y21)
    knn_3.get_data(x21, y21)
    x, r = knn_3.predict_and_score(x22, y22)
    rez_1.append(knn_1.score(x12, y12))
    rez_2.append(knn_2.score(x22, y22))
    rez_3.append(r)
    n.append(j)
    e = time.datetime.now()
    print(e-s, ' j = ', j)
plt.figure(30, constrained_layout=True)
plt.plot(n, rez_1, label='KNN without column expansion')
plt.plot(n, rez_2, label='KNN with column expansion')
plt.plot(n, rez_3, label='KNN implementation with column expansion')
plt.axvline(x=np.floor(np.sqrt(len(y_train))), color='r', linestyle='dotted', label='k = sqrt(N)')
plt.xlabel('K nearest neighbors')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.savefig(path+'KNN_performance.png')
plt.close('all')
