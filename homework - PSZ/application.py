import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from machine_learning import LinearRegressionGradientDescent
from machine_learning import LogisticRegressionOneVsOne
from machine_learning import LogisticRegressionMultinomial
from machine_learning import split_dataset, KFoldBayesianTargetEncoding

from gui import create_app

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 10000)


def surface(fmt: str):
    fmt = [float(v) for v in fmt.split('x')]
    return fmt[0] * fmt[1]


def main():
    data = pd.read_csv('./preprocesirane_knjige.csv')

    path = './LR_imgs'
    cat_path = './LR_imgs/Kategorija/'

    if not os.path.exists(path):
        os.mkdir(path=path)

    if not os.path.exists(cat_path):
        os.mkdir(path=cat_path)

    if not os.path.exists(cat_path + 'PagesVsPrice/'):
        os.mkdir(path=(cat_path + 'PagesVsPrice/'))

    ################################################################################
    # REMOVE SPECIFIC 
    # data = data.drop(index=data.loc[(data['izdavac'] == 'studio bečkerek'), 'izdavac'].index).reset_index(drop=True)
    # data = data.drop(index=data.loc[(data['izdavac'] == 'happy print'), 'izdavac'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[data['naziv'].str.contains('komplet'), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[data['kategorija'] == 'gift knjige', 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[data['kategorija'] == 'od 0 do 2 godine', 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'udžbenici i priručnici') & (data['broj_strana'] > 300), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'strip') & (data['tip_poveza'] == 'Broš'), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'od 10 do 12 godina') & (data['tip_poveza'] == 'Tvrd'), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'strani jezici') & (data['cena'] > 2000) & (data['broj_strana'] < 400), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'sociologija') & (data['cena'] > 1500) & (data['broj_strana'] < 150), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'sociologija') & (data['cena'] < 3000) & (data['broj_strana'] > 600), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'razonoda') & (data['cena'] > 1500) & (data['broj_strana'] < 300), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'putopisi') & (data['cena'] > 1900) & (data['broj_strana'] < 400), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'psihologija') & (data['cena'] > 3000) & (data['broj_strana'] < 300), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'pripovetke') & (data['cena'] > 3000), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'porodica') & (data['cena'] > 3000) & (data['broj_strana'] < 100), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'esejistika i publicistika') & (data['cena'] > 3000), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'domaći klasici') & (data['cena'] > 4000), 'naziv'].index).reset_index(drop=True)
    data = data.drop(index=data.loc[(data['kategorija'] == 'domaće priče i pripovetke') & (data['cena'] > 4000), 'naziv'].index).reset_index(drop=True)

    ################################################################################

    # FILTER OUTLIERS
    max_price = 5000
    max_surface = 900
    min_publish = 20
    min_year = 2005

    data = data.drop(index=data.loc[(data['cena'] > max_price), 'cena'].index).reset_index(drop=True)

    data['povrsina'] = data['format'].apply(surface)
    data = data.drop(index=data.loc[(data['povrsina'] >= max_surface), 'povrsina'].index).reset_index(drop=True)

    data = data.drop(index=data.loc[(data['godina_izdavanja'] < min_year), 'godina_izdavanja'].index).reset_index(drop=True)

    d = data['izdavac'].value_counts()
    remove = [idx for idx in d.index if d[idx] < min_publish]
    data = data.drop(index=data.loc[data['izdavac'].isin(remove), 'izdavac'].index).reset_index(drop=True)

    #######################################################################

    # folder = './analysis/'
    # if not os.path.exists(folder):
    #     os.mkdir(path=folder)
    #
    # f = open(folder + "publisher_avg_price.txt", 'w', encoding='utf-8')
    # d = data[['izdavac', 'cena']].groupby(by='izdavac').mean()
    # d['kolicina'] = 0
    # for izd in d.index:
    #     d.loc[izd, 'kolicina'] = data.loc[data['izdavac'] == izd, 'izdavac'].count()
    # f.write(d.sort_values(by='cena', ascending=False).to_string())
    # f.close()
    #
    # f = open(folder + "publisher_category.txt", 'w', encoding='utf-8')
    # d = data[['izdavac', 'kategorija', 'cena']].sort_values(by=['izdavac', 'kategorija', 'cena'])
    # f.write(d.to_string())
    # f.close()

    #######################################################################

    # d = sorted(data['kategorija'].unique())
    # for cat in d:

    #     tmp = data.loc[data['kategorija'] == cat, ['broj_strana', 'tip_poveza', 'cena']]

    #     y = tmp['cena'].to_numpy()
    #     x = tmp['broj_strana'].to_numpy().reshape(-1, 1)
    #     c = []
    #     for t in tmp['tip_poveza']:
    #         c.append('b' if t == 'Tvrd' else 'r')
    #     plt.scatter(x, y, c=c)
    #     plt.scatter([], [], c='b', label='Tvrd')
    #     plt.scatter([], [], c='r', label='Broš')
    #     plt.xlabel("Broj strana")
    #     plt.ylabel("Cena")
    #     plt.title(f"Kategorija {cat}")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(cat_path + f'PagesVsPrice/{cat}.png', dpi=300)
    #     plt.close()
    #     print(f'\n\n{cat}')
    #     print(tmp.corr(numeric_only=True))

    #######################################################################

    data = data.drop(columns=['_id', 'opis', 'naziv'])
    print(data.head())
    print(data.describe())
    data.to_csv('./filtrirane_knjige.csv', index=False)
    original_data = data.copy(deep=True)

    #######################################################################

    # y = data['cena'].to_numpy()
    #
    # x = data['broj_strana'].to_numpy().reshape(-1, 1)
    # plt.figure(1)
    # plt.scatter(x=x, y=y, c=x, marker='1')
    # plt.tight_layout()
    # plt.savefig(path+"/vsPages.png", dpi=300)
    # plt.close()
    #
    #
    # x = data['godina_izdavanja'].to_numpy().reshape(-1, 1)
    # plt.figure(2)
    # plt.scatter(x=x, y=y, c=x, marker='1')
    # plt.tight_layout()
    # plt.savefig(path+"/vsYear.png", dpi=300)
    # plt.close()
    #
    #
    # x = data['povrsina'].to_numpy().reshape(-1, 1)
    # plt.figure(3)
    # plt.scatter(x=x, y=y, c=y, marker='1')
    # plt.tight_layout()
    # plt.savefig(path+"/vsFormat.png", dpi=300)
    # plt.close()
    #
    # plt.rcParams.update({'font.size':8.0})
    # x = data[['kategorija', 'cena']].groupby(by='kategorija').mean().sort_values(by='cena')
    # plt.figure(4, figsize=(10, 10))
    # plt.scatter(y=x.index, x=x['cena'], marker='1')
    # plt.ylim(-1, len(x.index))
    # plt.tight_layout()
    # plt.savefig(path+"/vsCategory.png", dpi=300)
    # plt.close()
    # plt.rcParams.update({'font.size':10.0})
    #
    # plt.rcParams.update({'font.size':8.0})
    # x = data[['izdavac', 'cena']].groupby(by='izdavac').mean().sort_values(by='cena')
    # plt.figure(5, figsize=(10, 10))
    # plt.scatter(y=x.index, x=x['cena'], marker='1')
    # plt.ylim(-1, len(x.index))
    # plt.tight_layout()
    # plt.savefig(path+"/vsPublisher.png", dpi=300)
    # plt.close()
    # plt.rcParams.update({'font.size':10.0})
    #
    # x = data[['godina_izdavanja', 'cena']].groupby(by='godina_izdavanja').mean().sort_values(by='cena')
    # plt.figure(6, figsize=(10, 10))
    # plt.scatter(x=(x.index % 100), y=x['cena'], marker='1')
    # plt.xticks(range(min(x.index % 100), 1 + max(x.index % 100), 1))
    # plt.tight_layout()
    # plt.savefig(path+"/avgYear.png", dpi=300)
    # plt.close()

    ##########################################################################

    data['broj_strana'] = data['broj_strana'] / 100

    # data['broj_strana'] = MinMaxScaler().fit_transform(data['broj_strana'].to_numpy().reshape(-1, 1))

    # data['velicina'] = data['broj_strana'] * data['povrsina'] / 10000
    # data = data.drop(columns=['broj_strana', 'povrsina'])

    ##############################

    # data['godina_izdavanja'] = MinMaxScaler().fit_transform(data['godina_izdavanja'].to_numpy().reshape(-1, 1))

    data['godina_izdavanja'] = data['godina_izdavanja'] % 100

    ##############################
    min_surface = min(data['povrsina'].to_numpy())
    max_surface = max(data['povrsina'].to_numpy())
    data['povrsina'] = (data['povrsina'] - min_surface) / (max_surface - min_surface)

    # d = data.loc[:, ['format', 'cena']].groupby('format').mean()
    # data['format'] = data['format'].apply(lambda x: d['cena'][x])
    # data['format'] = MinMaxScaler().fit_transform(data['format'].to_numpy().reshape(-1, 1))

    # data[['sirina', 'visina']] = data['format'].str.split('x', expand=True).astype(float)
    # data = data.drop(columns='format')

    ##############################

    # ohe = OneHotEncoder(sparse_output=False, dtype=int)
    # binds = ohe.fit_transform(data['tip_poveza'].to_numpy().reshape(-1, 1))
    # data = data.drop(columns='tip_poveza')
    # data = data.join(pd.DataFrame(data=binds, columns=ohe.get_feature_names_out(['povez'])))

    data['tip_poveza'] = data['tip_poveza'].apply(lambda x: 1 if x == "Tvrd" else 0)

    ##############################

    # d = data.loc[:, ['kategorija', 'cena']].groupby('kategorija').mean().sort_values(by='cena').index.to_list()
    # for k in range(len(d)):
    #     data.loc[data['kategorija'] == d[k], 'kategorija'] = k

    data, cat_enc_dict= KFoldBayesianTargetEncoding(data, 'kategorija', 'cena', k=10, alpha=2)
    data['kategorija'] = data['kategorija'] / 100

    # d = data.loc[:, ['kategorija', 'cena']].groupby('kategorija').mean()
    # data['kategorija'] = data['kategorija'].apply(lambda x: d['cena'][x] / 100)

    # d = data.loc[:, ['kategorija', 'cena']].groupby('kategorija').mean()
    # data['kategorija'] = data['kategorija'].apply(lambda x: d['cena'][x])
    # data['kategorija'] = MinMaxScaler().fit_transform(data['kategorija'].to_numpy().reshape(-1, 1))

    # ohe = OneHotEncoder(sparse_output=False, dtype=int)
    # binds = ohe.fit_transform(data['kategorija'].to_numpy().reshape(-1, 1))
    # data = data.drop(columns='kategorija')
    # data = data.join(pd.DataFrame(data=binds, columns=ohe.get_feature_names_out(['K'])))

    ##############################

    # d = data.loc[:, ['izdavac', 'cena']].groupby('izdavac').mean().sort_values(by='cena').index.to_list()
    # for k in range(len(d)):
    #     data.loc[data['izdavac'] == d[k], 'izdavac'] = k

    data, pub_enc_dict = KFoldBayesianTargetEncoding(data, 'izdavac', 'cena', k=10, alpha=5)
    data['izdavac'] = data['izdavac'] / 100

    # d = data.loc[:, ['izdavac', 'cena']].groupby('izdavac').mean()
    # data['izdavac'] = data['izdavac'].apply(lambda x: d['cena'][x] / 100)

    # d = data.loc[:, ['izdavac', 'cena']].groupby('izdavac').mean()
    # data['izdavac'] = data['izdavac'].apply(lambda x: d['cena'][x])
    # data['izdavac'] = MinMaxScaler().fit_transform(data['izdavac'].to_numpy().reshape(-1, 1))

    ##############################

    def author_number(authors: str):
        return len(authors.split(','))

    data['broj_autora'] = 0
    data.loc[data['autor'] != 'grupa autora', 'broj_autora'] = data.loc[data['autor'] != 'grupa autora', 'autor'].apply(author_number)
    data.loc[data['autor'] == 'grupa autora', 'broj_autora'] = 5

    # f = open("./auth_num.txt", 'w', encoding='utf-8')
    # f.write(data['broj_autora'].value_counts().to_string())
    # f.write('\n')
    # f.write(data.loc[:, ['autor', 'broj_autora']].value_counts().to_string())
    # f.close()

    # data = data.drop(columns='autor')

    ##############################

    data = data.loc[:, [
                           'cena',
                           'broj_strana',
                           'godina_izdavanja',
                           'povrsina',
                           # 'format',
                           'tip_poveza',
                           'kategorija',
                           'izdavac',
                           # 'autor'
                       ]]

    ##############################

    print(data)

    # plt.figure(constrained_layout=True)
    # sns.heatmap(data.corr(numeric_only=True), annot=True, linewidths=1, fmt='.2f', square=True, vmax=1, vmin=-1)
    # plt.savefig(path + "/Correlation.png", dpi=300)
    # plt.close()

    #######################################################################

    y = data['cena'] / 100
    x = data.drop(columns=['cena'])

    x_train, x_test, y_train, y_test = split_dataset(x, y, train_size=0.75, random_state=123)

    learning_rate = [
        [1],        # w0
        [0.001],    # broj_strana
        [0.0001],   # godina_izdavanja
        [0.1],      # povrsina
        [0.01],     # tip_poveza
        [0.001],    # kategorija
        [0.0001]    # izdavac
    ]
    iterations = 5000

    linreg_gd = LinearRegressionGradientDescent()
    linreg_gd.fit(x_train, y_train, learning_rate=learning_rate, iter=iterations)
    linreg_gd_out = linreg_gd.predict(x_test)

    print("\n\nLinear Regression with Gradient Descent performance:")
    print(f"Mean squared error: {linreg_gd.error_function(x_train, y_train):.3f}")
    print(f"R^2 Score: {linreg_gd.score(linreg_gd_out, y_test.to_numpy()):.3f}\n\n")

    #######################################################################

    ranges = [0, 500, 1000, 1500, 3000, 5000]
    price_cat_dict = {
        0: "do 500 din",
        1: "od 500 do 1000 din",
        2: "od 1000 do 1500 din",
        3: "od 1500 do 3000 din",
        4: "od 3000 do 5000 din"
    }
    for k in range(len(ranges) - 1):
        idx = (data['cena'] > ranges[k]) & (data['cena'] <= ranges[k + 1])
        data.loc[idx, 'cena'] = k
    
    print(data)

    y = data['cena']
    x = data.drop(columns=['cena'])

    x_train, x_test, y_train, y_test = split_dataset(x, y, train_size=0.75, random_state=123)

    learning_rate = [
        [1],        # w0
        [0.1],      # broj_strana
        [0.01],     # godina_izdavanja
        [0.1],      # povrsina
        [0.01],     # tip_poveza
        [0.01],     # kategorija
        [0.01]      # izdavac
    ]
    iterations = 10000

    logreg_ovo = LogisticRegressionOneVsOne(alpha=0.1)
    logreg_ovo.fit(x_train, y_train, learning_rate=learning_rate, iter=iterations)
    logreg_ovo_out = logreg_ovo.predict(x_test)

    learning_rate = [
        [1],        # w0
        [0.01],     # broj_strana
        [0.01],     # godina_izdavanja
        [0.1],      # povrsina
        [0.1],      # tip_poveza
        [0.01],     # kategorija
        [0.01]      # izdavac
    ]
    iterations = 10000

    logreg_multi = LogisticRegressionMultinomial()
    logreg_multi.fit(x_train, y_train, learning_rate=learning_rate, iter=iterations)
    logreg_multi_out = logreg_multi.predict(x_test)

    print(f"F1 Score One vs One: {logreg_ovo.score(logreg_ovo_out, y_test):.3f}")
    print(f"F1 Score Multinomial: {logreg_multi.score(logreg_multi_out, y_test):.3f}")

    create_app(
        original_data,
        cat_enc_dict,
        pub_enc_dict,
        linreg_gd,
        logreg_ovo,
        logreg_multi,
        price_cat_dict
    )


if __name__ == "__main__":
    main()
