import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import median_absolute_error

from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.neural_network import *
from xgboost import *


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def plot_corr(data, name):
    data_corr = data.corr()
    mask = np.triu(np.ones_like(data_corr, dtype=bool))[1:, :-1]
    plt.figure(figsize=(10, 8))
    sns.heatmap(data=data_corr.iloc[1:, :-1], mask=mask, annot=True, fmt=".2f",
                vmin=-1, vmax=1, linecolor='white', linewidths=0.5, center=0)
    plt.tight_layout()
    plt.savefig(f'./{name}.png', dpi=300)
    plt.close()


data = pd.read_csv("./train.csv")
data = data.drop(columns=['id'])
print(data)
print(data.info())
print(data.describe())

#########################################################################
#                          DATA ANALYSIS                                #
#########################################################################
'''
plot_corr(data, 'Corr')

k = 1
plt.figure(figsize=(15, 15))
for col in data.columns:
    if col == 'Hardness':
        continue
    x = data[col].to_numpy()
    y = data['Hardness'].to_numpy()
    plt.subplot(4, 3, k)
    k += 1
    plt.scatter(x=x, y=y, c='grey', marker='.')
    plt.xlabel(col)
    plt.ylabel('Hardness')
    plt.tight_layout()
plt.savefig(f'./vsHardness/Hardness.png', dpi=300)
plt.close()

for col in data.columns:
    if col == 'Hardness':
        continue
    x = data[col].to_numpy()
    y = data['Hardness'].to_numpy()
    plt.scatter(x=x, y=y, c='grey', marker='.')
    plt.xlabel(col)
    plt.ylabel('Hardness')
    plt.tight_layout()
    plt.savefig(f'./vsHardness/{col}.png', dpi=300)
    plt.close()

pairs = [
    ('allelectrons_Average', 'atomicweight_Average'),
    ('R_vdw_element_Average', 'R_cov_element_Average'),
    ('allelectrons_Average', 'density_Average'),
    ('atomicweight_Average', 'density_Average'),
    ('ionenergy_Average', 'el_neg_chi_Average'),
    ('allelectrons_Average', 'R_cov_element_Average'),
]
plt.figure(figsize=(20, 10))
k = 1
for p in pairs:
    x = data[p[0]].to_numpy()
    y = data[p[1]].to_numpy()
    plt.subplot(2, 3, k)
    k += 1
    plt.scatter(x=x, y=y, c='grey', marker='.')
    plt.xlabel(p[0])
    plt.ylabel(p[1])
    plt.tight_layout()
plt.savefig(f'./HighCorr/HighCorr.png', dpi=300)
plt.close()

for p in pairs:
    x = data[p[0]].to_numpy()
    y = data[p[1]].to_numpy()
    plt.scatter(x=x, y=y, c='grey', marker='.')
    plt.xlabel(p[0])
    plt.ylabel(p[1])
    plt.tight_layout()
    plt.savefig(f'./HighCorr/{p[0]}_vs_{p[1]}.png', dpi=300)
    plt.close()
'''

#########################################################################
#                          DATA FILTERING                               #
#########################################################################

filter_data = data.copy(deep=True)
filter_data = filter_data.drop(index=filter_data.loc[filter_data['allelectrons_Total'] > 750, 'allelectrons_Total'].index)
filter_data = filter_data.drop(index=filter_data.loc[filter_data['atomicweight_Average'] > 150, 'atomicweight_Average'].index)
filter_data = filter_data.drop(index=filter_data.loc[filter_data['atomicweight_Average'] < 8, 'atomicweight_Average'].index)
filter_data = filter_data.drop(index=filter_data.loc[filter_data['density_Total'] > 50, 'density_Total'].index)
filter_data = filter_data.drop(index=filter_data.loc[filter_data['val_e_Average'] < 2.5, 'val_e_Average'].index)
filter_data = filter_data.drop(index=filter_data.loc[filter_data['el_neg_chi_Average'] < 1.5, 'el_neg_chi_Average'].index)
filter_data = filter_data.drop(index=filter_data.loc[filter_data['el_neg_chi_Average'] > 3.05, 'el_neg_chi_Average'].index)
filter_data = filter_data.drop(index=filter_data.loc[filter_data['ionenergy_Average'] < 8, 'ionenergy_Average'].index)
filter_data = filter_data.drop(index=filter_data.loc[filter_data['R_cov_element_Average'] < 0.5, 'R_cov_element_Average'].index)
filter_data = filter_data.drop(index=filter_data.loc[filter_data['R_vdw_element_Average'] < 1.3, 'R_vdw_element_Average'].index)
filter_data = filter_data.drop(index=filter_data.loc[filter_data['R_vdw_element_Average'] > 2.2, 'R_vdw_element_Average'].index)
filter_data.insert(len(filter_data.columns) - 1, 'R_Average', filter_data['R_cov_element_Average'] * filter_data['R_vdw_element_Average'])
filter_data = filter_data.drop(columns=['R_cov_element_Average', 'R_vdw_element_Average'])
filter_data = filter_data.drop(index=filter_data.loc[filter_data['density_Average'] > 8, 'density_Average'].index)
filter_data = filter_data.drop(index=filter_data.loc[filter_data['zaratio_Average'] < 0.4, 'zaratio_Average'].index)
filter_data = filter_data.drop(index=filter_data.loc[(filter_data['allelectrons_Total'] > 300) & (filter_data['density_Total'] < 15), 'allelectrons_Total'].index)
filter_data = filter_data.drop(columns=['allelectrons_Average'])


plot_corr(filter_data, 'Corr2')
'''
for col in filter_data.columns:
    if col == 'Hardness':
        continue
    x = filter_data[col].to_numpy()
    y = filter_data['Hardness'].to_numpy()
    plt.scatter(x=x, y=y, c='grey', marker='.')
    plt.xlabel(col)
    plt.ylabel('Hardness')
    plt.tight_layout()
    plt.savefig(f'./vsHardness_Filtered/{col}.png', dpi=300)
    plt.close()
'''
#########################################################################
#                          MODEL SELECTION                              #
#########################################################################


y = filter_data["Hardness"]
X = filter_data.drop(columns=['Hardness'])

st = StandardScaler()
X = st.fit_transform(X)

lof = LocalOutlierFactor(n_neighbors=20)
outliers = lof.fit_predict(X)
mask = outliers != -1
X, y = X[mask], y[mask]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=123, shuffle=False)

kf = KFold(n_splits=5, shuffle=True, random_state=123)

models = [
    ('Linear Regression', LinearRegression(), {}),
    ('Ridge Regression', Ridge(), {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
    }),
    ('Lasso Regression', Lasso(random_state=123), {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
    }),
    ('ElasticNet Regression', ElasticNet(random_state=123), {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
    }),
    ('SGD Regression', SGDRegressor(random_state=123), {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
    }),
    ('DecisionTree Regressor', DecisionTreeRegressor(random_state=123), {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 10, 20]
    }),
    ('RandomForest Regressor', RandomForestRegressor(
        random_state=123,
        n_estimators=100,
        max_depth=10,
        min_samples_split=2
    ), {}),
    ('Gradient Boosting Regressor', GradientBoostingRegressor(
        random_state=123,
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1
    ), {}),
    ('HistGradientBoosting Regression', HistGradientBoostingRegressor(
        random_state=123,
        max_depth=10,
        max_iter=300
    ), {
        'learning_rate': [0.01, 0.1, 0.2]
    }),
    ('Voting Regressor', VotingRegressor(estimators=[
        ('dt', DecisionTreeRegressor(random_state=123, max_depth=10, min_samples_split=2)),
        ('hgb', HistGradientBoostingRegressor(random_state=123, max_depth=10, max_iter=300)),
        ('gb', GradientBoostingRegressor(random_state=123, n_estimators=100, max_depth=7))
    ]), {}),
    ('XGBoost Regressor', XGBRegressor(
        objective='reg:squarederror',
        random_state=123,
        max_depth=7
    ), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }),
    ('MLP Regressor', MLPRegressor(
        random_state=123,
        max_iter=500,
        hidden_layer_sizes=(50, 50),
        activation='relu'
    ), {}),
]

for name, model, params in models:
    t = time.time()
    grid = GridSearchCV(model, params, cv=kf, scoring='neg_median_absolute_error')
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_
    pred = best_model.predict(x_test)
    cv_results = cross_val_score(best_model, x_train, y_train, cv=kf, scoring='neg_median_absolute_error')
    t = time.time() - t

    print(f"\n\n{name}:")
    if len(params) != 0:
        print(f"Best Hyperparameters: {grid.best_params_}")
    print(f"Training Score: {best_model.score(x_train, y_train):.3f}")
    print(f"Median Absolute Error on Test Set: {median_absolute_error(y_test, pred):.3f}")
    print(f"Cross-Validation Median Absolute Error: {-cv_results.mean():.3f} ± {cv_results.std():.3f}")
    print(f'Time required: {t:.3f}s')

print("\n\nK Nearest Neighbors Regression:")
knn_dict = {
    "K": [],
    "Score": [],
    "MedAE": [],
    "CV_MedAE": []
}

k = int(np.sqrt(len(x_train)))
k += 1 - k % 2

for n in range(5, k+1, 6):
    t1 = time.time()

    model = KNeighborsRegressor(n_neighbors=n)
    cv_scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='neg_median_absolute_error')
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    knn_dict["K"].append(n)
    knn_dict["Score"].append(f"{model.score(x_train, y_train):.3f}")
    knn_dict["MedAE"].append(f"{median_absolute_error(y_test, pred):.3f}")
    knn_dict["CV_MedAE"].append(f"{-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    t2 = time.time() - t1
    print(f"Training model for K = {n}, t = {t2:.3f}s")

print("\nResults:")
knn_results = pd.DataFrame(data=knn_dict)
print(knn_results)
