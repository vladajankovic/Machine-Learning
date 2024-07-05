import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def pretty_pie_chart(dict_data: dict, colors: list, explode: list, startangle: int, title: str, save_path: str):

    base_d = sum(list(dict_data.values()))
    final_data = {k+f': ({m})':m/base_d*100 for k,m in dict_data.items()}

    fig, ax = plt.subplots(figsize=(12, 5), subplot_kw=dict(aspect="equal"))
    recipe = list(final_data.keys())
    dict_data = list(final_data.values())
    perc = [str(round(e, 2)) + '%' for e in dict_data]
    plt.title(title)
    wedges, texts = ax.pie(dict_data, wedgeprops=dict(width=0.5),explode=explode, colors=colors, startangle=startangle)
    kw = dict(arrowprops=dict(arrowstyle="-"), zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        kw["arrowprops"].update({"connectionstyle": f"angle,angleA=0,angleB={ang}"})
        ax.annotate(recipe[i] + ' ' + perc[i], xy=(x, y), xytext=(1.4*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)

    plt.savefig(save_path, dpi=300)
    plt.close()



def main():

    if not os.path.exists('./charts'):
        os.mkdir('./charts')

    data = pd.read_csv("preprocesirane_knjige.csv")

    top_10_publishers = data.loc[:, 'izdavac'].value_counts().sort_values(ascending=False).head(n=10).to_dict()
    plt.figure(1, figsize=(12, 7))
    plt.title("Top 10 izdavaca sa najvecim brojem knjiga u ponudi")
    plt.barh(top_10_publishers.keys(), top_10_publishers.values(), height=0.6)
    plt.xlabel("Broj knjiga")
    plt.xticks(ticks=range(0, max(top_10_publishers.values())+250, 250))
    plt.grid()
    plt.tight_layout()
    plt.savefig("./charts/a.top10Publishers.png", dpi=300)
    plt.close()

    ###########################################################################################################################
    ###########################################################################################################################    

    categories = data['kategorija'].apply(str.lower).value_counts().sort_values(ascending=False).to_dict()
    plt.rcParams.update({'font.size':8.0})
    plt.figure(2, figsize=(15, 10))
    plt.title("Broj knjiga po kategorijama")
    plt.barh(categories.keys(), categories.values(), height=0.5)
    plt.grid()
    plt.ylim(-1, len(categories.keys()))
    plt.xticks(ticks=range(0, max(categories.values()), 100))
    plt.xlabel("Broj knjiga")
    plt.tight_layout()
    plt.savefig("./charts/b.categories.png", dpi=300)
    plt.rcParams.update({'font.size':10.0})
    plt.close()

    ###########################################################################################################################
    ###########################################################################################################################

    years = {
        '1961-1970': data.loc[(data['godina_izdavanja'] > 1960) & (data['godina_izdavanja'] <= 1970), 'godina_izdavanja'].count(),
        '1971-1980': data.loc[(data['godina_izdavanja'] > 1970) & (data['godina_izdavanja'] <= 1980), 'godina_izdavanja'].count(),
        '1981-1990': data.loc[(data['godina_izdavanja'] > 1980) & (data['godina_izdavanja'] <= 1990), 'godina_izdavanja'].count(),
        '1991-2000': data.loc[(data['godina_izdavanja'] > 1990) & (data['godina_izdavanja'] <= 2000), 'godina_izdavanja'].count(),
        '2001-2010': data.loc[(data['godina_izdavanja'] > 2000) & (data['godina_izdavanja'] <= 2010), 'godina_izdavanja'].count(),
        '2011-2020': data.loc[(data['godina_izdavanja'] > 2010) & (data['godina_izdavanja'] <= 2020), 'godina_izdavanja'].count(),
        '2021-sada': data.loc[data['godina_izdavanja'] > 2020, 'godina_izdavanja'].count()
    }

    plt.figure(3)
    plt.title("Broj knjiga po dekadama")
    plt.barh(years.keys(), years.values())
    plt.xlabel("Broj knjiga")
    plt.tight_layout()
    plt.savefig('./charts/c.booksByDecades.png')
    plt.close()

    # years = {
    #     '1961-2000': data.loc[data['godina_izdavanja'] <= 2000, 'godina_izdavanja'].count(),
    #     '2001-2005': data.loc[(data['godina_izdavanja'] > 2000) & (data['godina_izdavanja'] <= 2005), 'godina_izdavanja'].count(),
    #     '2006-2010': data.loc[(data['godina_izdavanja'] > 2005) & (data['godina_izdavanja'] <= 2010), 'godina_izdavanja'].count(),
    #     '2011-2015': data.loc[(data['godina_izdavanja'] > 2010) & (data['godina_izdavanja'] <= 2015), 'godina_izdavanja'].count(),
    #     '2016-2020': data.loc[(data['godina_izdavanja'] > 2015) & (data['godina_izdavanja'] <= 2020), 'godina_izdavanja'].count(),
    #     '2021-sada': data.loc[data['godina_izdavanja'] > 2020, 'godina_izdavanja'].count()
    # }

    years = {
        '1961-2000': data.loc[data['godina_izdavanja'] <= 2000, 'godina_izdavanja'].count(),
        '2001-2004': data.loc[(data['godina_izdavanja'] > 2000) & (data['godina_izdavanja'] <= 2004), 'godina_izdavanja'].count(),
        '2005-2008': data.loc[(data['godina_izdavanja'] > 2004) & (data['godina_izdavanja'] <= 2008), 'godina_izdavanja'].count(),
        '2009-2012': data.loc[(data['godina_izdavanja'] > 2008) & (data['godina_izdavanja'] <= 2012), 'godina_izdavanja'].count(),
        '2013-2016': data.loc[(data['godina_izdavanja'] > 2012) & (data['godina_izdavanja'] <= 2016), 'godina_izdavanja'].count(),
        '2017-2020': data.loc[(data['godina_izdavanja'] > 2016) & (data['godina_izdavanja'] <= 2020), 'godina_izdavanja'].count(),
        '2021-sada': data.loc[data['godina_izdavanja'] > 2020, 'godina_izdavanja'].count()
    }

    plt.figure(4, figsize=(10, 5))
    plt.title("Broj knjiga na po 4 godina")
    plt.bar(years.keys(), years.values(), width=0.5)
    plt.plot(range(len(years)), years.values(), 'ro', ls='-')
    plt.grid()
    plt.ylabel("Broj knjiga")
    plt.tight_layout()
    plt.savefig('./charts/c.booksBy4Years.png', dpi=300)
    plt.close()

    ###########################################################################################################################
    ###########################################################################################################################
    
    top_publishers = data.loc[:, 'izdavac'].value_counts().sort_values(ascending=False).to_dict()
    keys = list(top_publishers.keys())
    tmp = {}
    for k in range(5):
        tmp[keys[k]] = top_publishers[keys[k]]
    tmp['Ostali izdavaci'] = 0
    for k in range(5, len(top_publishers)):
        tmp['Ostali izdavaci'] += top_publishers[keys[k]]

    colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]

    pretty_pie_chart(dict_data=tmp, colors=colors, explode=[0.02, 0.03, 0.04, 0.05, 0.05, 0], 
                     startangle=-80, title="Procenat knjiga za prodaju po izdavackim kucama",
                     save_path="./charts/d.publishersPieChart.png")

    ###########################################################################################################################
    ###########################################################################################################################

    prices = {
        'do 500 din': data.loc[data['cena'] <= 500, 'cena'].count(),
        'od 10001 do 15000 din': data.loc[(data['cena'] > 10000) & (data['cena'] <= 15000), 'cena'].count(),
        'od 501 do 1500 din': data.loc[(data['cena'] > 500) & (data['cena'] <= 1500), 'cena'].count(),
        'od 15001 din i vise': data.loc[data['cena'] > 15000, 'cena'].count(),
        'od 1501 do 3000 din': data.loc[(data['cena'] > 1500) & (data['cena'] <= 3000), 'cena'].count(),
        'od 3001 do 5000 din': data.loc[(data['cena'] > 3000) & (data['cena'] <= 5000), 'cena'].count(),
        'od 5001 do 10000 din': data.loc[(data['cena'] > 5000) & (data['cena'] <= 10000), 'cena'].count(),
    }
    colors = ["#264653", "#2a9d8f", "#8ab17d", "#000000", "#f4a261", "#e76f51", "#daab25"]

    pretty_pie_chart(dict_data=prices, colors=colors, explode=[0.01, 0.5, 0, 0.5, 0, 0.03, 0.05], 
                     startangle=0, title="Procenat knjiga za prodaju po opsezima cena",
                     save_path="./charts/e.pricesPieChart.png")

    

    prices = {
        'do 500 din': data.loc[data['cena'] <= 500, 'cena'].count(),
        'od 501 do 1000 din': data.loc[(data['cena'] > 500) & (data['cena'] <= 1000), 'cena'].count(),
        'od 1001 do 1500 din': data.loc[(data['cena'] > 1000) & (data['cena'] <= 1500), 'cena'].count(),
        'od 10001 din i vise': data.loc[data['cena'] > 10000, 'cena'].count(),
        'od 1501 do 3000 din': data.loc[(data['cena'] > 1500) & (data['cena'] <= 3000), 'cena'].count(),
        'od 3001 do 5000 din': data.loc[(data['cena'] > 3000) & (data['cena'] <= 5000), 'cena'].count(),
        'od 5001 do 10000 din': data.loc[(data['cena'] > 5000) & (data['cena'] <= 10000), 'cena'].count(),
    }
    colors = ["#264653", "#2a9d8f", "#8ab17d", "#000000", "#f4a261", "#e76f51", "#daab25"]

    pretty_pie_chart(dict_data=prices, colors=colors, explode=[0, 0, 0, 0.2, 0.01, 0.03, 0.1], 
                     startangle=30, title="Procenat knjiga za prodaju po opsezima cena",
                     save_path="./charts/e.prices2PieChart.png")

    ###########################################################################################################################
    ###########################################################################################################################

    def func(pct, allvals):
        absolute = int(np.round(pct * allvals / 100.0))
        return f"({absolute})\n{pct:.2f}%"

    last_3_years = data.loc[data['godina_izdavanja'].isin([2023, 2022, 2021]), 'tip_poveza'].value_counts().to_dict()
    total = sum(last_3_years.values())
    plt.figure(6)
    plt.title("Procenat knjiga za prodaju sa Tvrdim povezom, u poslednje 3 godine")
    plt.pie(last_3_years.values(),labels=last_3_years.keys(), colors=["#E3461C", "#E3AA1C"],
            autopct=lambda pct: func(pct, total))
    plt.tight_layout()
    plt.savefig("./charts/f.bindPieChart.png", dpi=300)
    plt.close()
    



if __name__ == "__main__":
    main()