import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from LinearRegressionGradientDescent import LinearRegressionGradientDescent


# Podesavanje prikaza
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', None)

path = 'lin_reg_img/'

if not os.path.exists(path):
    os.makedirs(path)

# Citanje iz fajla
data = pd.read_csv('datasets/car_purchase.csv')

# Prikaz prvih i poslednjih 5 redova tabele
print("\nPrvih 5 redova:")
print(data.head(), end='\n\n')  # default n=5
print("Poslednjih 5 redova:")
print(data.tail(n=5), end='\n\n')
print('-------------------------------------------------------------------------------------\n')

# Prikaz dodatnih informacija o tabeli
print("Osnovne informacije o tabeli:\n")
data.info(memory_usage='deep')
print('\n Informacije o atributima numerickog tipa:\n', data.describe())
print('\n Informacije o ostalim atributima:\n', data.describe(include=np.object_), end='\n\n')
print('-------------------------------------------------------------------------------------')

y = data.max_purchase_amount.to_numpy()

x = data.age.to_numpy()
plt.figure(1)
plt.scatter(x, y, marker='1', c=x)
plt.grid()
plt.xlabel('Age')
plt.ylabel('Max purchase amount')
plt.title("Purchase amount vs. Age")
plt.savefig(path+"vsAge.png")
plt.close()

x = data.annual_salary.to_numpy()
plt.figure(2)
plt.scatter(x, y, marker='h', c=y)
plt.grid()
plt.xlabel('Annual salary')
plt.ylabel('Max purchase amount')
plt.title("Purchase amount vs. Annual salary")
plt.savefig(path+"vsSalary.png")
plt.close()

x = data.credit_card_debt.to_numpy()
plt.figure(3)
plt.scatter(x, y, marker='^')
plt.grid()
plt.xlabel('Credit card debt')
plt.ylabel('Max purchase amount')
plt.title("Purchase amount vs. Credit card debt")
plt.savefig(path+"vsDebt.png")
plt.close()

x = data.net_worth.to_numpy()
plt.figure(4)
plt.scatter(x, y, marker='.', c=y)
plt.ticklabel_format(useOffset=0, style='plain')
plt.grid()
plt.xlabel('Net worth')
plt.ylabel('Max purchase amount')
plt.title("Purchase amount vs. Net worth")
plt.savefig(path+"vsNet.png")
plt.close()

plt.figure(5)
y = data.loc[:, ['gender', 'max_purchase_amount']].to_numpy()
m = []
f = []
for e in y:
    if e[0] == "F":
        f.append(e[1])
    else:
        m.append(e[1])
m.sort()
f.sort()
x = []
for j in range(len(m)):
    x.append(-1)
plt.scatter(x, m, marker="*", c="red")
x = []
for j in range(len(f)):
    x.append(1)
plt.scatter(x, f, marker='3', c='blue')
plt.xlim(-3, 3)
plt.xticks([])
plt.grid()
plt.xlabel('Gender')
plt.ylabel('Max purchase amount')
plt.legend(["Male", "Female"])
plt.title("Purchase amount vs. Gender")
plt.savefig(path+"vsGender.png")
plt.close()

plt.figure(constrained_layout=True)
le = LabelEncoder()
data.gender = le.fit_transform(data.gender)
sb.heatmap(data.corr(), annot=True, linewidths=1, fmt='.2f', square=True, vmax=1, vmin=-1)
plt.savefig(path+"Correlation.png")
plt.close()


# Na osnovu heatmap, dobru korelaciju imaju age, annual_salary i net_worth sa labelom
data_train = data.loc[:, ["age", "annual_salary", "net_worth"]]
label = data.loc[:, 'max_purchase_amount']

print("\nAtributi koji se koriste za treniranje ML:\n", data_train)
print('\nIzlazni atribut: max_purchase_amount\n', label)

# Podesavanje atributa da vrednosti budu priblizne
data_train["annual_salary"] = data_train["annual_salary"]/1000.0
data_train["net_worth"] = data_train["net_worth"]/10000.0
label = label/1000.0

print("\nAtributi nakon skaliranja:\n")
print(data_train)
print("\nmax_purchase_amount\n", label)
print('\n-------------------------------------------------------------------------------------\n')


# Treniranje modela
# train_size = 0.7 -> 0.7*400 = 280 redova se koriste za treniranje
# -> 0.3*400 = 120 redova se koriste za proveru ML
X_train, X_test, y_train, y_test = train_test_split(data_train, label, train_size=0.7, random_state=123, shuffle=False)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# Linearna regresija metodom gradijentnog sputa
############################
iter = 200          # BROJ ITERACIJA!!!!
############################
lrgd = LinearRegressionGradientDescent()
lrgd.fit(X_train, y_train)
learning_rates = np.array([[1], [0.0001], [0.0001], [0.0001]])
res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rates, iter) # preciznost 93%
#
#   Najbolji rezultat ako se stavi:
#   1, 0.0001, 0.0001, 0.0001, 5000 -> poklapanje koeficijenata LR i LRGD
#   1, 0.0001, 0.0001, 0.0001, 500  -> preciznost ~100%
#   1, 0.0001, 0.0001, 0.0001, 200  -> preciznost 93%

y_prediction = pd.DataFrame(data=lrgd.predict(X_test), columns=['LRGD_prediction'])
lrgd_output = X_test.join(y_prediction)
lrgd_output = lrgd_output.join(y_test)
print("Predikcije linearne regresije metodom gradijentnig spusta:\n", lrgd_output)

# Lienarna regresija koristeci ugradjenu funkciju iz skliear modula
lr = LinearRegression()
lr.fit(X_train, y_train)

y_prediction = pd.DataFrame(data=lr.predict(X_test), columns=['LR_prediction'])
lr_output = X_test.join(y_prediction)
lr_output = lr_output.join(y_test)
print("\nPredikcije linearne regresije iz 'sklearn' modula:\n", lr_output, end='\n\n')

# Parametri oba modela predikcije
s = ['w0', 'w1', 'w2', 'w3']
lr_coeff = [lr.intercept_]
for n in lr.coef_:
    lr_coeff.append(n)
coeff = pd.DataFrame(data=s, columns=[' '])
coeff = coeff.join(pd.DataFrame(data=res_coeff, columns=['LRGD_coeff']))
coeff = coeff.join(pd.DataFrame(data=lr_coeff, columns=['LR_coeff']))
print("\nKoeficijenti dobijeni obema metodama:\n", coeff)

plt.figure('MS Error')
plt.plot(np.arange(0, len(mse_history), 1), mse_history)
plt.xlabel('Iteration', fontsize=13)
plt.ylabel('MS error value', fontsize=13)
plt.xticks(np.arange(0, len(mse_history), iter/10))
plt.title('Mean-square error function')
plt.tight_layout()
plt.legend(['MS Error'])
plt.savefig(path+'MSE.png')
plt.close()

# Vrednost MSE funkcije za oba modela
lrgd.set_coefficients(res_coeff)    # Koeficijenti LRGD
lrgd_mse_value = lrgd.cost()

lrgd.set_coefficients(lr_coeff)         # Koeficijenti LR
lr_mse_value = lrgd.cost()

lrgd.set_coefficients(res_coeff)    # Restauracija koeficijenata

print(f'\nLinear Regression MSE rezultat: {lr_mse_value:.2f}')
print(f'Linear Regression Gradient Descent MSE rezultat: {lrgd_mse_value:.2f}')

# Izracunavanje preciznosti oba modela
lr_coef_ = lr.coef_                 # Cuvanje vrednosti LR u privremene promenljive
lr_int_ = lr.intercept_

lr.coef_ = lrgd.coeff.flatten()[1:] # Postavljanje koeficijenata od LRGD za izracunavanje score()
lr.intercept_ = lrgd.coeff.flatten()[0]
rez1 = lr.score(X_test, y_test)

lr.coef_ = lr_coef_                 # Restauriraju se koeficijenti LR modela i izracunavanje score()
lr.intercept_ = lr_int_
rez2 = lr.score(X_test, y_test)

print(f'Linear Regression preciznost: {rez2:.2f}')
print(f'Linear Regression Gradient Descent preciznost: {rez1:.2f}')

plt.figure(10, constrained_layout=True)
plt.subplot(211)
plt.scatter(lrgd_output['age'], lrgd_output['max_purchase_amount'], label='True value')
plt.scatter(lrgd_output['age'], lrgd_output['LRGD_prediction'], label='LRGD Prediction')
plt.xlabel('Age'); plt.ylabel('max_purchase_amount'); plt.legend()
plt.subplot(212)
plt.scatter(lr_output['age'], lr_output['max_purchase_amount'], label='True value')
plt.scatter(lr_output['age'], lr_output['LR_prediction'], label='LR Prediction')
plt.xlabel('Age'); plt.ylabel('max_purchase_amount'); plt.legend()
plt.savefig(path+'predAge.png')

plt.figure(11, constrained_layout=True)
plt.subplot(211)
plt.scatter(lrgd_output['annual_salary'], lrgd_output['max_purchase_amount'], label='True value')
plt.scatter(lrgd_output['annual_salary'], lrgd_output['LRGD_prediction'], label='LRGD Prediction')
plt.xlabel('Annual salary'); plt.ylabel('max_purchase_amount'); plt.legend()
plt.subplot(212)
plt.scatter(lr_output['annual_salary'], lr_output['max_purchase_amount'], label='True value')
plt.scatter(lr_output['annual_salary'], lr_output['LR_prediction'], label='LR Prediction')
plt.xlabel('Annual salary'); plt.ylabel('max_purchase_amount'); plt.legend()
plt.savefig(path+'predSalary.png')

plt.figure(12, constrained_layout=True)
plt.subplot(211)
plt.scatter(lrgd_output['net_worth'], lrgd_output['max_purchase_amount'], label='True value')
plt.scatter(lrgd_output['net_worth'], lrgd_output['LRGD_prediction'], label='LRGD Prediction')
plt.xlabel('Net worth'); plt.ylabel('max_purchase_amount'); plt.legend()
plt.subplot(212)
plt.scatter(lr_output['net_worth'], lr_output['max_purchase_amount'], label='True value')
plt.scatter(lr_output['net_worth'], lr_output['LR_prediction'], label='LR Prediction')
plt.xlabel('Net worth'); plt.ylabel('max_purchase_amount'); plt.legend()
plt.savefig(path+'predNet.png')

