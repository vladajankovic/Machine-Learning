import pandas as pd
from math import inf
from numpy import isnan
from pprint import pprint
import re
import os

pd.set_option('display.max_columns', 11)

folder = './analysis/'
if not os.path.exists(folder):
    os.mkdir(path=folder)


def count_missing(data: pd.DataFrame):
    print("Broj podataka koji nedostaje za svaku kolonu:")
    print(data.isnull().sum(), end='\n\n\n')
    count = [0, 0, 0, 0, 0, 0, 0, 0]
    for id in data.index:
        nuls = data.iloc[id].isnull().sum()
        if (nuls > 0) :
            count[nuls-1] += 1
    print(f"Broj redova u kojima fali jedan  podatak: {count[0]}\n",
          f"Broj redova u kojima fali dva    podatka: {count[1]}\n",
          f"Broj redova u kojima fali tri    podatka: {count[2]}\n",
          f"Broj redova u kojima fali cetiri podatka: {count[3]}\n",
          f"Broj redova u kojima fali pet    podatka: {count[4]}\n",
          f"Broj redova u kojima fali sest   podatka: {count[5]}\n",
          f"Broj redova u kojima fali sedam  podatka: {count[6]}\n",
          f"Broj redova u kojima fali osam   podatka: {count[7]}\n\n", sep='')


def filter_description(data: pd.DataFrame):
    print("U slucaju da knjiga nema opis, u kolonu 'opis' se upisuje 'Knjiga nema opis...'\n")
    data['opis'] = data['opis'].fillna("Knjiga nema opis...")
    return data


def filter_name(data: pd.DataFrame):
    data['naziv'] = data['naziv'].apply(str.lower)
    return data


def filter_pages(data: pd.DataFrame):
    print("\nU slucaju da knjiga nema broj strana, uklanja se")
    last_count = data['_id'].count()
    data = data.drop(index=data.loc[data['broj_strana'].isnull(), 'broj_strana'].index)
    data = data.reset_index(drop=True)
    next_count = data['_id'].count()
    print(f"Uklonjeno je {last_count - next_count} redova u tabeli BEZ broja strana", end='\n\n')

    print("Moguce da su veliki broj strana netacni podaci pa ako knjiga ima preko 1400 strana, uklanja se'")
    last_count = data['_id'].count()
    data = data.drop(index=data.loc[data['broj_strana'] > 1400, 'broj_strana'].index)
    data = data.drop(index=data.loc[data['broj_strana'] < 6, 'broj_strana'].index)
    data = data.reset_index(drop=True)
    next_count = data['_id'].count()
    print(f"Uklonjeno je {last_count - next_count} redova u tabeli BEZ broja strana", end='\n\n')

    data['broj_strana'] = data['broj_strana'].apply(int)

    return data


def filter_year(data: pd.DataFrame):
    print("Ispravljaju se vrednosti za godinu izdavanja knjige...")
    fix_years = {
        15: 2015, 17: 2017, 19:2019, 20: 2020, 100: 2020, 178: 1997, 200: 2010, 201: 2011, 202: 2019, 
        204: 2015, 206: 2006, 207: 2007, 208: 2008, 209: 2009, 214: 2014, 215: 2015, 219: 2019, 220: 2020, 
        240: 2019, 280: 2008, 288: 2010, 344: 2006, 2088: 2008, 2105: 2015, 2115: 2015, 2507: 2007, 2996: 1996, 
        20011: 2011, 20085: 2008, 20108: 2019, 20158: 2018, 20210: 2021, 21016: 2016
    }
    to_swap = [382, 114, 192, 430]
    for idx in data.index:
        v = data.loc[idx, 'godina_izdavanja']
        if not isnan(v):
            v = int(v)
            if v in to_swap:
                data.loc[idx, 'godina_izdavanja'], data.loc[idx, 'broj_strana'] = data.loc[idx, 'broj_strana'], data.loc[idx, 'godina_izdavanja']
            elif v in fix_years:
                data.loc[idx, 'godina_izdavanja'] = fix_years[v]

    print("Godina izdavanja se ciklicno popunjava kod knjiga kojim fali.")
    year_fill = [x for x in range(2023, 2006, -1)]
    for idx1, idx2 in enumerate(data.loc[data['godina_izdavanja'].isnull(), 'godina_izdavanja'].index):
        data.loc[idx2, 'godina_izdavanja'] = year_fill[idx1 % len(year_fill)]

    print("Zbog mogucih gresaka u podacima za godinu izdavanja, uzimaju se u obzir samo godine od 1960. do 2024.")
    last_count = data['_id'].count()
    data = data.drop(index=data.loc[data['godina_izdavanja'] < 1960, 'godina_izdavanja'].index)
    data = data.drop(index=data.loc[data['godina_izdavanja'] > 2024, 'godina_izdavanja'].index)
    data = data.reset_index(drop=True)
    next_count = data['_id'].count()
    print(f"Uklonjeno je {last_count - next_count} redova u tabeli", end='\n\n')

    data['godina_izdavanja'] = data['godina_izdavanja'].apply(int)

    return data


def filter_bind(data: pd.DataFrame):
    print("\nBroj razlicitih tipova poveza:")
    print(data.loc[data['tip_poveza'].notnull(), 'tip_poveza'].value_counts())
    print("\nPostoji 3x vise poveza tipa 'Broš' nego 'Tvrd' a postoji i jedan 'Tvrd povez'.")
    print("'Tvrd povez' se pretvara u 'Tvrd'")
    data.loc[data['tip_poveza'] == 'Tvrd povez', 'tip_poveza'] = "Tvrd"
    print(data.loc[data['tip_poveza'].notnull(), 'tip_poveza'].value_counts())
    print("\nPosto ima vise Bros od Tvrdog poveza, knjige za koje nije pronadjen povez ce biti Tvrd\n", 
          "kako bi se broj tvrdih malo priblizio broju Bros poveza")
    data.loc[data['tip_poveza'].isnull(), 'tip_poveza'] = 'Tvrd'
    print(data.loc[data['tip_poveza'].notnull(), 'tip_poveza'].value_counts(), end='\n\n')

    return data


def filter_category(data: pd.DataFrame):

    print("\nKategoriju knjige odredjuje knjizara, pa ako takvi podaci nedostaju, bice uklonjeni")
    last_count = data['_id'].count()
    data = data.drop(index=data.loc[data['kategorija'].isnull(), 'kategorija'].index)
    data = data.reset_index(drop=True)
    next_count = data['_id'].count()
    print(f"Uklonjeno je {last_count - next_count} redova u tabeli BEZ kategorije", end='\n\n')

    print("\nKategorija 'MAPE I KARTE' ne predstavljaju knjige pa se zato uklanjaju svi redovi sa ovom kategorijom...")
    last_count = data['_id'].count()
    data = data.drop(index=data.loc[data['kategorija'] == 'MAPE I KARTE', 'kategorija'].index)
    data = data.reset_index(drop=True)
    next_count = data['_id'].count()
    print(f"Uklonjeno je {last_count - next_count} redova u tabeli sa kategorijom 'MAPE I KARTE'", end='\n\n')


    print("Podkategorije za medicinske knjige imaju malo knjiga pa se mogu grupisati u vecu kategoriju 'MEDICINA'")
    subcategory = ['AKUŠERSTVO I GINEKOLOGIJA', 'ALERGOLOGIJA I IMUNOLOGIJA', 'ANATOMIJA I FIZIOLOGIJA',
                   'DERMATOLOGIJA I VENEROLOGIJA', 'GASTROENTEROLOGIJA', 'GENETIKA', 'ISHRANA I DIJETE STRUČNA', 
                   'KARDIOLOGIJA', 'MEDICINA (OPŠTA)', 'NARKOMANIJA I ALKOHOLIZAM', 'NEUROLOGIJA', 'ONKOLOGIJA', 
                   'PEDIJATRIJA', 'PSIHIJATRIJA', 'RADIOLOGIJA I NUKLEARNA MEDICINA', 'STOMATOLOGIJA', 
                   'UROLOGIJA I NEFROLOGIJA']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "MEDICINA"

    print("Podkategorije za strane jezike imaju malo knjiga pa se mogu grupisati u vecu kategoriju 'STRANI JEZICI'")
    subcategory = ['ARAPSKI JEZIK', 'DANSKI JEZIK', 'ENGLESKI JEZIK', 'ENGLESKI JEZIK - PRIRUČNICI I KURSEVI', 
                   'ENGLESKI JEZIK/GRAMATIKA', 'FRANCUSKI JEZIK', 'GRČKI JEZIK', 'HEBREJSKI JEZIK', 'HOLANDSKI JEZIK',
                   'ITALIJANSKI JEZIK', 'JAPANSKI JEZIK', 'KINESKI JEZIK', 'LATINSKI', 'NEMAČKI JEZIK', 'NORVEŠKI JEZIK',
                   'NORVEŠKI JEZIK', 'OSTALI JEZICI', 'POLJSKI JEZIK', 'PORTUGALSKI JEZIK', 'REČNICI ENGLESKOG JEZIKA',
                   'RUSKI JEZIK', 'TURSKI JEZIK', 'ČEŠKI JEZIK', 'ŠPANSKI JEZIK', 'ŠVEDSKI JEZIK']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = 'STRANI JEZICI'

    redundant = ['ASERTIVNOST, MOTIVACIJA I SAMOPOŠTOVANJE (domaći autori)', 'EZOTERIJA (domaći autori)',
                 'KOMUNIKACIJA, EMOCIJA I ODNOSI SA DRUGIMA (domaći autori)', 
                 'NLP - NEURO LINGVISTIČKO PROGRAMIRANJE (domaći autori)',
                 'POBOLJŠANJE PAMĆENJA I TEHNIKE RAZMIŠLJANJA (domaći autori)',
                 'SAVETI ZA KARIJERU I POSTIZANJE USPEHA (domaći autori)']
    
    for red in redundant:
        data.loc[data['kategorija'] == red, 'kategorija'] = red.split(' (')[0]

    print("Grupisanje knjiga za decu od 1. do 8. razreda")
    grades = {
        '1': '1.', '1.': '1.', 'i'   : '1.', 'prvi'    : '1.', 'prvo'  : '1.', 'prvii': '1.',
        '2': '2.', '2.': '2.', 'ii'  : '2.', 'drugi'   : '2.',
        '3': '3.', '3.': '3.', 'iii' : '3.', 'treći'   : '3.',
        '4': '4.', '4.': '4.', 'iv'  : '4.', 'četvrti' : '4.',
        '5': '5.', '5.': '5.', 'v'   : '5.', 'peti'    : '5.',
        '6': '6.', '6.': '6.', 'vi'  : '6.', 'šesti'   : '6.',
        '7': '7.', '7.': '7.', 'vii' : '7.', 'sedmi'   : '7.',
        '8': '8.', '8.': '8.', 'viii': '8.', 'osmi'    : '8.', 'više': '8.'
    }

    for idx in data.index:
        if 'razred' in data.loc[idx, 'naziv'].lower():
            data.loc[idx, 'kategorija'] = 'osnovna skola'
            # v = grades.get(data.loc[idx, 'naziv'].lower().split('razred')[0].strip().split()[-1])
            # data.loc[idx, 'kategorija'] = f"{v} RAZRED"
    
    print("Grupisanje knjiga za decu 0-2, 3-6, 7-9, 10-12 godina")
    subcategory = ['UZRAST OD 0 DO 2 GODINE', 'UZRAST OD 10 DO 12 GODINA', 'UZRAST OD 3 DO 6 GODINA',
                   'UZRAST OD 7 DO 9 GODINA', 'ROMANI I PRIČE ZA DECU OD 7 DO 9 GODINA',
                   'ROMANI I PRIČE ZA DECU OD 10 DO 12 GODINA']
    for cat in subcategory:
        data.loc[data['kategorija'] == cat, 'kategorija'] = "OD " + cat.split('OD ')[1]
    
    print("Podkategorije za tehnicke nauke se mogu grupisati u vecu kategoriju 'TEHNIČKE NAUKE I MATEMATIKA'")
    subcategory = ['ASTRONOMIJA', 'ELEKTROTEHNIKA', 'FIZIKA', 'GRAĐEVINARSTVO', 'KOMPJUTERSKA LITERATURA',
                   'MATEMATIKA', 'MAŠINSTVO', 'SAOBRAĆAJ', 'TEHNOLOGIJA (OPŠTA)']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "TEHNIČKE NAUKE I MATEMATIKA"

    print("Podkategorije za prirodne nauke se mogu grupisati u vecu kategoriju 'PRIRODNE NAUKE'")
    subcategory = ['GEOGRAFIJA', 'GEOLOGIJA', 'HEMIJA', 'EKOLOGIJA', 'POLJOPRIVREDA', 'POPULARNA NAUKA',
                   'MOLEKULARNA BIOLOGIJA/GENETIKA', 'BIOHEMIJA/BIOFIZIKA', 'BIOLOGIJA']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "PRIRODNE NAUKE"

    print("Podkategorije za biljke i zivotinje se mogu grupisati u vecu kategoriju 'BILJKE I ŽIVOTINJE'")
    subcategory = ['BOTANIKA', 'DIVLJE ŽIVOTINJE', 'DOMAĆE ŽIVOTINJE', 'INSEKTI', 'PTICE', 'REPTILI', 'VODENE ŽIVOTINJE']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "BILJKE I ŽIVOTINJE"

    print("Podkategorije za sport se mogu grupisati u vecu kategoriju 'SPORT'")
    subcategory = ['AUTOMOBILI I MOTOCIKLI', 'FUDBAL', 'KOŠARKA', 'SPORT', 'LOV I RIBOLOV']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "SPORT"

    print("Podkategorije za razomodu se mogu grupisati u vecu kategoriju 'RAZONODA'")
    subcategory = ['ASTROLOGIJA', 'HOBI', 'HUMOR', 'RAZONODA', 'VOJSKA I AVIJACIJA', 'ZANATI']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "RAZONODA"

    print("Podkategorije za ekonomiju se mogu grupisati u vecu kategoriju 'EKONOMIJA'")
    subcategory = ['BIZNIS', 'EKONOMIJA', 'EKONOMSKA SITUACIJA I TEORIJA', 'FINANSIJE', 'MARKETING', 
                   'MENADŽMENT', 'RAČUNOVODSTVO I POREZI', 'TRGOVINA']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "EKONOMIJA"

    print("Podkategorije za turizam se mogu grupisati u vecu kategoriju 'TURIZAM'")
    subcategory = ['ATLASI I VODIČI', 'TURIZAM']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "TURIZAM"

    print("Podkategorije za licni razvoj se mogu grupisati u vecu kategoriju 'LIČNI RAZVOJ'")
    subcategory = ['ASERTIVNOST, MOTIVACIJA I SAMOPOŠTOVANJE', 'KOMUNIKACIJA, EMOCIJA I ODNOSI SA DRUGIMA',
                   'NLP - NEURO LINGVISTIČKO PROGRAMIRANJE', 'POBOLJŠANJE PAMĆENJA I TEHNIKE RAZMIŠLJANJA',
                   'SAVETI ZA KARIJERU I POSTIZANJE USPEHA']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "LIČNI RAZVOJ"

    print("Podkategorije za lingvistiku i fililogiju se mogu grupisati u vecu kategoriju 'LINGVISTIKA/FILOLOGIJA'")
    subcategory = ['LINGVISTIKA', 'SRPSKI JEZIK', 'FILOLOGIJA']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "LINGVISTIKA/FILOLOGIJA"

    print("Podkategorije za umetnosti se mogu grupisati u vecu kategoriju 'UMETNOST'")
    subcategory = ['ISTORIJA UMETNOSTI', 'OPŠTA UMETNOST', 'MUZIKA', 'ENTERIJER I UNUTRAŠNJI DEKOR',
                   'MODA', 'GRAFIČKI DIZAJN', 'FOTOGRAFIJA', 'DIZAJN', ]
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "UMETNOST"

    print("Podkategorije za primenjene umetnosti se mogu grupisati u vecu kategoriju 'FILM/POZORIŠTE'")
    subcategory = ['FILM', 'POZORIŠTE']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "FILM/POZORIŠTE"

    print("Podkategorije za bastu i ljubimce se mogu grupisati u vecu kategoriju 'DOMAĆINSTVO'")
    subcategory = ['DOMAĆINSTVO', 'VRT I BAŠTA', 'KUĆNI LJUBIMCI']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "DOMAĆINSTVO"

    print("Podkategorije za zdravlje se mogu grupisati u vecu kategoriju 'ZDRAVLJE'")
    subcategory = ['ZDRAV ŽIVOT', 'ZDRAVA ISHRANA I TRAVARSTVO', 'ALTERNATIVNA MEDICINA', 'UM TELO I DUH']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "ZDRAVLJE"

    print("Podkategorije za porodicu, trudnocu i vaspitanje se mogu grupisati u vecu kategoriju 'PORODICA'")
    subcategory = ['VASPITANJE I PSIHOLOGIJA', 'TRUDNOĆA I NEGA DECE']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "PORODICA"

    print("Podkategorije za Romane se mogu grupisati u neke postojece kategorije")
    subcategory = ['CHICK LIT - ROMANTIČNA KOMEDIJA', 'EROTSKI ROMAN']
    data.loc[data['kategorija'].isin(subcategory), 'kategorija'] = "LJUBAVNI ROMAN"
    data.loc[data['kategorija'] == 'RATNI ROMAN', 'kategorija'] = 'ISTORIJSKI ROMAN'
    data.loc[data['kategorija'] == 'NAUČNA FANTASTIKA', 'kategorija'] = 'FANTASTIKA'
    data.loc[data['kategorija'] == 'AKCIJA/AVANTURA', 'kategorija'] = 'ROMAN'

    data.loc[data['kategorija'] == 'DRUŠTVENE NAUKE', 'kategorija'] = "PEDAGOGIJA"
    data.loc[data['naziv'] == 'AKADEMSKO PISANJE I TEHNIKA NAUČNOISTRAŽIVAČKOG RADA ', 'kategorija'] = "PEDAGOGIJA"

    data.loc[data['kategorija'] == 'DOMAĆI LJUBAVNI ROMAN', 'kategorija'] = "DOMAĆI ROMAN"

    data.loc[data['kategorija'] == 'HRANA I PIĆE', 'kategorija'] = "KULINARSTVO"

    data.loc[data['kategorija'] == 'TINEJDŽ I YA PRIRUČNICI', 'kategorija'] = "TINEJDŽ I YA"
    data.loc[data['kategorija'] == 'TINEJDŽ I YA ROMAN', 'kategorija'] = "TINEJDŽ I YA"

    data.loc[data['kategorija'] == 'MITOLOGIJA', 'kategorija'] = "ISTORIJA"
    data.loc[data['kategorija'] == 'DEČJE KNJIGE ZA UČENJE STRANIH JEZIKA', 'kategorija'] = "STRANI JEZICI"
    
    data['kategorija'] = data['kategorija'].apply(str.lower)
    data['kategorija'] = data['kategorija'].apply(lambda x: x.replace('/', '-'))

    return data


def filter_authors(data: pd.DataFrame):
    print("Ako nije poznat autor knjige, uklanja se")
    last_count = data['_id'].count()
    data = data.drop(index=data.loc[data['autor'].isnull(), 'autor'].index)
    data = data.reset_index(drop=True)
    next_count = data['_id'].count()
    print(f"Uklonjeno je {last_count - next_count} redova u tabeli BEZ autora", end='\n\n')

    data['autor'] = data['autor'].apply(str.lower)
    
    data.loc[data['autor'] == 'nolwena monnier. eve grosset', 'autor'] = 'nolwena monnier, eve grosset'
    data.loc[data['autor'] == 'izbor, prevod i predgovor tatjana simonović', 'autor'] = 'tatjana simonović'
    data.loc[data['autor'] == 'izbor i prevod nikola b. cvetković', 'autor'] = 'nikola b. cvetković'
    data.loc[data['autor'] == 'učim i igram se', 'autor'] = 'učim - igram se'
    data.loc[data['autor'] == 'bojanka i vežbanka', 'autor'] = 'bojanka - vežbanka'
    data.loc[data['autor'] == 'skupio i preveo albin vilhar', 'autor'] = 'albin vilhar'
    data.loc[data['autor'] == 'redaktor i urednik protođakon radomir rakić', 'autor'] = 'radomir rakić'
    data.loc[data['autor'] == 'priredio i preveo želidrag nikčević', 'autor'] = 'želidrag nikčević'
    data.loc[data['autor'] == 'priredio i predgovor napisao\\nboško mijatović', 'autor'] = 'boško mijatović'
    data.loc[data['autor'] == 'popuni i pokloni', 'autor'] = 'popuni - pokloni'
    data.loc[data['autor'] == 'priredila i prevela bojana kovačević petrović', 'autor'] = 'bojana kovačević petrović'
    data.loc[data['autor'] == 'sastavi modeli i igraj se', 'autor'] = 'sastavi modeli - igraj se'
    data.loc[data['autor'] == 'scenario ž. b. đian i olivije legran; crtež i kolo', 'autor'] = 'ž. b. đian, olivije legran'
    data.loc[data['autor'] == 'gordana živković glavni i odgovorni urednik', 'autor'] = 'gordana živković'

    data['autor'] = data['autor'].apply(lambda x: re.sub(r',.*:', ',', x))
    data['autor'] = data['autor'].apply(lambda x: re.sub(r'^.*:', '', x))
    data['autor'] = data['autor'].apply(lambda x: re.sub(r'[&;]', ',', x))
    data['autor'] = data['autor'].apply(lambda x: re.sub(r' i ', ',', x))

    return data


def filter_publishers(data: pd.DataFrame):
    print("Ako nije poznat izdavac knjige, popunjava se najfrekventnijim izdavacem.")
    data.loc[data['izdavac'].isnull(), 'izdavac'] = data['izdavac'].value_counts().index[0]

    data = data.drop(index=data.loc[data['izdavac'] == 'MAGIC MAP D.O.O.', "izdavac"].index).reset_index(drop=True)

    data.loc[data['izdavac'] == "ALGORITAM MEDIA", 'izdavac'] = 'ALGORITAM'
    data.loc[data['izdavac'] == "BEGEN BOOKS", 'izdavac'] = 'BEGEN COMERC'
    data.loc[data['izdavac'] == "BEL MEDIA", 'izdavac'] = 'BELMEDIA'
    data.loc[data['izdavac'] == "BIBLIONER/PLATO", 'izdavac'] = 'PLATO'
    data.loc[data['izdavac'] == "BOOK", 'izdavac'] = 'BOOK MARSO'
    data.loc[data['izdavac'] == "DRASLAR TANESI", 'izdavac'] = 'DRASLAR'
    data.loc[data['izdavac'] == "FUTURA", 'izdavac'] = 'FUTURA PUBLIKACIJE'
    data.loc[data['izdavac'] == "HERAedu i Institut za noviju istoriju Srbije", 'izdavac'] = 'HERAEDU'
    data.loc[data['izdavac'] == "IA NOVA POETIKA ARGUS BOOKS&MAGAZINES", 'izdavac'] = 'NOVA POETIKA'
    data.loc[data['izdavac'] == "ARGUS BOOKS&MAGAZINES", 'izdavac'] = 'NOVA POETIKA'
    data.loc[data['izdavac'] == "NOVA POETIKA ARGUS BOOKS MAGAZINES", 'izdavac'] = 'NOVA POETIKA'
    data.loc[data['izdavac'] == "ID LEO COMMERCE D.O.O./ - 45%", 'izdavac'] = 'LEO COMMERCE'
    data.loc[data['izdavac'] == "LEO COMERC RIJEKA", 'izdavac'] = 'LEO COMMERCE'
    data.loc[data['izdavac'] == "LEO COMMERCE ZRAK", 'izdavac'] = 'LEO COMMERCE'
    data.loc[data['izdavac'] == "IK STRAHOR", 'izdavac'] = 'STRAHOR'
    data.loc[data['izdavac'] == "IMPRIMATUR  Banja Luka", 'izdavac'] = 'IMPRIMATUR'
    data.loc[data['izdavac'] == "IMPRIMATUR PUBLISHING", 'izdavac'] = 'IMPRIMATUR'
    data.loc[data['izdavac'] == "INSITUT ZA FILOZOFIJU I DRUŠTVENU TEORIJU", 'izdavac'] = 'INSTITUT DRUŠTVENIH NAUKA'
    data.loc[data['izdavac'] == "INSTITUT ZA EKONOMIKU I FINANSIJE", 'izdavac'] = 'INSTITUT DRUŠTVENIH NAUKA'
    data.loc[data['izdavac'] == "INSTITUT ZA POLITIČKO UMREŽAVANJE", 'izdavac'] = 'INSTITUT DRUŠTVENIH NAUKA'
    data.loc[data['izdavac'] == "IP ŽARKO ALBULJ", 'izdavac'] = 'ZARKO ALBULJ'
    data.loc[data['izdavac'] == "IPS", 'izdavac'] = 'IPC MEDIA'
    data.loc[data['izdavac'] == "IPS MEDIA", 'izdavac'] = 'IPC MEDIA'
    data.loc[data['izdavac'] == "KIŠA I ARTKULT", 'izdavac'] = 'KIŠA'
    data.loc[data['izdavac'] == "KIŠA IZDAVAČKA KUĆA", 'izdavac'] = 'KIŠA'
    data.loc[data['izdavac'] == "KOKORO LIBER", 'izdavac'] = 'KOKORO'
    data.loc[data['izdavac'] == "KONTRAST IZDAVAŠTVO/DERETA", 'izdavac'] = 'KONTRAST'
    data.loc[data['izdavac'] == "KOSMOS BG NOVA KNJIGA PG", 'izdavac'] = 'KOSMOS IZDAVAŠTVO'
    data.loc[data['izdavac'] == "KREATIVNI CENTAR D.O.O.  UDŽBENICI", 'izdavac'] = 'KREATIVNI CENTAR'
    data.loc[data['izdavac'] == "CENTAR ZA IZUČAVANJE TRADICIJE UKRONIJA", 'izdavac'] = 'UKRONIJA'
    data.loc[data['izdavac'] == "LOGOS ART", 'izdavac'] = 'LOGOS'
    data.loc[data['izdavac'] == "LOGOS I SLUŽBENI GLASNIK", 'izdavac'] = 'LOGOS'
    data.loc[data['izdavac'] == "LOGOS I UKRONIJA", 'izdavac'] = 'LOGOS'
    data.loc[data['izdavac'] == "LUKA BABIĆ", 'izdavac'] = 'LUKA BOOKS'
    data.loc[data['izdavac'] == "LUKA BOOKS / INSTITUT ZA DEČIJU KNJIŽEVNOST", 'izdavac'] = 'LUKA BOOKS'
    data.loc[data['izdavac'] == "MAKART/ADMIRAL", 'izdavac'] = 'MAKART'
    data.loc[data['izdavac'] == "MARSO", 'izdavac'] = 'BOOK MARSO'
    data.loc[data['izdavac'] == "MAKART/ADMIRAL", 'izdavac'] = 'MAKART'
    data.loc[data['izdavac'] == "MASCOM EC BOOKING", 'izdavac'] = 'MASCOM'
    data.loc[data['izdavac'] == "MATICA SRPSKA I SANU", 'izdavac'] = 'MATICA SRPSKA'
    data.loc[data['izdavac'] == "SIA MATIĆ", 'izdavac'] = 'AGENCIJA MATIĆ'
    data.loc[data['izdavac'] == "MATIĆ", 'izdavac'] = 'AGENCIJA MATIĆ'
    data.loc[data['izdavac'] == "MIBA BOOKS & DN CENTAR", 'izdavac'] = 'MIBA BOOKS'
    data.loc[data['izdavac'] == "MIBA BOOKS NARODNA KNJIGA", 'izdavac'] = 'MIBA BOOKS'
    data.loc[data['izdavac'] == "MLADINSKA KNJIGA SARAJEVO", 'izdavac'] = 'MLADINSKA KNJIGA'
    data.loc[data['izdavac'] == "MODESTY STRIPOVI / KOMIKO", 'izdavac'] = 'MODESTY STRIPOVI'
    data.loc[data['izdavac'] == "OMNIBUS/MODESTY STRIPOVI", 'izdavac'] = 'MODESTY STRIPOVI'
    data.loc[data['izdavac'] == "SYSTEM COMICS", 'izdavac'] = 'MORO DOO'
    data.loc[data['izdavac'] == "MORO DOO SYSTEM COMICS", 'izdavac'] = 'MORO DOO'
    data.loc[data['izdavac'] == "NARODNA KNJIGA ALFA", 'izdavac'] = 'NARODNA KNJIGA'
    data.loc[data['izdavac'] == "NARODNA KNJIGA PODGORICA", 'izdavac'] = 'NARODNA KNJIGA'
    data.loc[data['izdavac'] == 'NARODNO DELO BEOGRAD', 'izdavac'] = 'NARODNO DELO'
    data.loc[data['izdavac'] == 'NARODNO DELO DOO HRAM SVETOG SAVE', 'izdavac'] = 'NARODNO DELO'
    data.loc[data['izdavac'] == 'NARODNO DELO DOO PROMETEJ', 'izdavac'] = 'NARODNO DELO'
    data.loc[data['izdavac'] == 'NOVA KNJIGA BG NOVA KNJIGA PG', 'izdavac'] = 'NOVA KNJIGA'
    data.loc[data['izdavac'] == 'NOVA KNJIGA PG', 'izdavac'] = 'NOVA KNJIGA'
    data.loc[data['izdavac'] == 'NOVOSTI  IK Metella Ksenia', 'izdavac'] = 'NOVOSTI'
    data.loc[data['izdavac'] == 'NOVOSTI I GLAS CRKVE', 'izdavac'] = 'NOVOSTI'
    data.loc[data['izdavac'] == 'ODYSSEUS', 'izdavac'] = 'ODISEJA'
    data.loc[data['izdavac'] == 'ORION ART BOOKS', 'izdavac'] = 'ORION ART'
    data.loc[data['izdavac'] == 'PARTIZANSKA KNJIGA KRR', 'izdavac'] = 'PARTIZANSKA KNJIGA'
    data.loc[data['izdavac'] == 'POETA', 'izdavac'] = 'POETIKUM'
    data.loc[data['izdavac'] == 'PORTAL DOO', 'izdavac'] = 'PORTALIBRIS'
    data.loc[data['izdavac'] == 'PRINCIP', 'izdavac'] = 'PRINCIP PRES'
    data.loc[data['izdavac'] == 'PROMETEJ RTS', 'izdavac'] = 'PROMETEJ'
    data.loc[data['izdavac'] == 'PROPOLIS PLUS', 'izdavac'] = 'PROPOLIS BOOKS'
    data.loc[data['izdavac'] == 'ROSVETA  MALI PRINC', 'izdavac'] = 'PROSVETA'
    data.loc[data['izdavac'] == 'ROMANOV BARDFIN', 'izdavac'] = 'ROMANOV DOO'
    data.loc[data['izdavac'] == 'RTS', 'izdavac'] = 'RTS'
    data.loc[data['izdavac'] == 'RTS I Fondacija Sreten Stojanovic', 'izdavac'] = 'RTS'
    data.loc[data['izdavac'] == 'RTS I INSTITUT ZA DEČJU KNJIŽEVNOST', 'izdavac'] = 'RTS'
    data.loc[data['izdavac'] == 'RTS I INSTITUT ZA UMETNOST I KNJIŽEVNOST', 'izdavac'] = 'RTS'
    data.loc[data['izdavac'] == 'RTS I MUZEJ VOJVODINE', 'izdavac'] = 'RTS'
    data.loc[data['izdavac'] == 'RTS I MUZIKOLOŠKO DRUŠTVO SRBIJE', 'izdavac'] = 'RTS'
    data.loc[data['izdavac'] == 'RTS I ORION ART', 'izdavac'] = 'RTS'
    data.loc[data['izdavac'] == 'RTS I PROMETEJ', 'izdavac'] = 'RTS'
    data.loc[data['izdavac'] == 'RTS I RTV', 'izdavac'] = 'RTS'
    data.loc[data['izdavac'] == 'RTS I Signature/Larus', 'izdavac'] = 'RTS'
    data.loc[data['izdavac'] == 'RTS I THIS AND THAT PRODUCTIONS', 'izdavac'] = 'RTS'
    data.loc[data['izdavac'] == 'RTS I ZADUŽBINA VLADA PETRIĆ', 'izdavac'] = 'RTS'
    data.loc[data['izdavac'] == 'SMART DEVELOPMENT', 'izdavac'] = 'SMART'
    data.loc[data['izdavac'] == 'SMART PRODUCTION', 'izdavac'] = 'SMART'
    data.loc[data['izdavac'] == 'SMART TARGET', 'izdavac'] = 'SMART'
    data.loc[data['izdavac'] == 'TREĆI TRG/SREBRNO DRVO', 'izdavac'] = 'TREĆI TRG'
    data.loc[data['izdavac'] == 'UDRUŽENJE STRIPO-TETKE/BESNA KOBILA', 'izdavac'] = 'BESNA KOBILA'
    data.loc[data['izdavac'] == 'UDUS', 'izdavac'] = 'DRUŽENJE DRAMSKIH UMETNIKA SRBIJE'
    data.loc[data['izdavac'] == 'URBAN READS / MAKONDO', 'izdavac'] = 'URBAN READS'
    data.loc[data['izdavac'] == 'VULKAN IZDAVAŠTVO - ALNARI', 'izdavac'] = 'VULKAN IZDAVAŠTVO'
    data.loc[data['izdavac'] == 'VULKAN IZDAVAŠTVO - MONO I MANJANA', 'izdavac'] = 'VULKAN IZDAVAŠTVO'
    data.loc[data['izdavac'] == 'VULKAN ZNANJE', 'izdavac'] = 'VULKAN IZDAVAŠTVO'
    data.loc[data['izdavac'] == 'ZEBRA', 'izdavac'] = 'ZEBRA PUBLISHING'
    data.loc[data['izdavac'] == 'ZMAJ  SEZAM BUK', 'izdavac'] = 'ZMAJ'

    data['izdavac'] = data['izdavac'].apply(str.lower)

    return data


def filter_format(data: pd.DataFrame):
    print("\nKnjige koje nemaju format kao podatak ce biti obrisane")
    last_count = data['_id'].count()
    data = data.drop(index=data.loc[data['format'].isnull(), 'format'].index)
    data = data.reset_index(drop=True)
    next_count = data['_id'].count()
    print(f"Uklonjeno je {last_count - next_count} redova u tabeli BEZ formata", end='\n\n')

    data['povrsina'] = 0.0

    formats = [f'{n}x{n}' for n in range(6, 41)]
    formats += ['10.5x14.8','14.8x21.0','21.0x29.7','29.7x42.0','42.0x59.4',
                '12.5x17.6','17.6x25.0','25.0x35.3','35.3x50.0','50.0x70.7',
                '11.4x16.2','16.2x22.9','22.9x32.4','32.4x45.8','45.8x64.8']
    f = open(folder + 'formating_errors.txt', 'w')
    print("Knjige u bazi imaju ogroman broj formata, jer postoje slicni formati sa malim odstupanjem\n",
          "Formati se filtriraju tako sto se grupisu u jedan od standardnih formata: ",
          "A6-A2, B6-B2, C6-C2 \nili kvadratni format NxN (N=6, 7, 8, ..., 40)", end='\n\n')
    print("Filtriranje i grupisanje formata knjiga")
    for id in data.index:
        if id % 1000 == 0:
            print(f"Odradjeno {data.last_valid_index()}/{id}")
        s: str = data.loc[id, 'format']
        if type(s) == str:
            s = s.replace(' ', '').replace(',', '.').replace('u00a0', '').replace('u00d7', 'x')
            s = s.replace('B5', '17.6x25').replace('tvrd', '25x25').replace('13020', '13x20')
            s = s.replace('brou0161', '17.6x25').replace('A5', '14.8x21')
            s = re.sub(r'(X|\?)', 'x', s)
            s = re.sub(r'x+', 'x', s)
            s = re.sub(r'(CM|V|cm)', '', s)
            s = re.sub(r'\.+', '.', s)
            s = s.split('x')
            s[0] = float(s[0])
            s[0] = s[0] / 10 if s[0] > 70 else s[0]
            s.append(s[0])
            s[1] = float(s[1])
            s[1] = s[1] / 10 if s[1] > 70 else s[1]
            if s[1] < s[0]:
                s[0], s[1] = s[1], s[0]
            while s[1] > 65:
                s[1] = s[1] / 10

            min_val = inf
            closest_format = ""
            for fmt in formats:
               tmp = [float(x) for x in fmt.split('x')]
               dist = ((tmp[0] - s[0])**2 + (tmp[1] - s[1])**2)**0.5
               if dist < min_val:
                   min_val = dist
                   closest_format = fmt
            
            f.write('{:.2f} -> {}x{} -> {}\n'.format(min_val, s[0], s[1], closest_format))
            data.loc[id, 'format'] = closest_format
            tmp = closest_format.split('x')
            data.loc[id, 'povrsina'] = float(tmp[0]) * float(tmp[1])
            # data.loc[id, 'format'] = str(s[0])+'x'+str(s[1])
            # data.loc[id, 'povrsina'] = s[0] * s[1]
            
    f.close()
    print(f"Odradjeno {data.last_valid_index()}/{data.last_valid_index()}")
    print("\nFiltritanje i grupisanje formata zavrseno. Izlazne informacije sacuvane u formating_errors.txt i format_count.txt")
    
    return data


def log_data(data: pd.DataFrame):

    f = open(folder + 'format_count.txt', 'w')
    pprint(data.loc[:, 'format'].value_counts().to_dict(), stream=f)
    f.close()

    f = open(folder + "category_list.txt", "w", encoding='utf-8')
    d = data.loc[data['kategorija'].notnull(), 'kategorija'].value_counts().to_dict()
    for key in d:
        f.write('{: <5}\t{}\n'.format(d[key], key))
    f.close()

    f = open(folder + "publisher_list.txt", 'w', encoding='utf-8')
    d = data.loc[data['izdavac'].notnull(), 'izdavac'].value_counts().to_dict()
    for key in d:
        f.write('{: <5}\t{}\n'.format(d[key], key))
    f.close()
    
    f = open(folder + "author_list.txt", "w", encoding='utf-8')
    f.write(data.loc[:, ['autor']].value_counts().to_string())
    f.close()

    #count_missing(data)

    print(data.info(), end='\n\n\n')
    print(data.describe(), end='\n\n\n')

    f = open(folder +"analysis_output.txt", 'w', encoding='utf-8')
    f.write("\nRezultati analize prikupljenih i preciscenih podataka\n")

    f.write("\nBroj knjiga za prodaju po kategorijama knjige:\n")
    f.write("broj knjiga\t\tkategorija\n")
    d = data.loc[data['kategorija'].notnull(), 'kategorija'].value_counts().to_dict()
    for key in d:
        f.write('{: <5}      \t\t{}\n'.format(d[key], key))

    f.write("\n########################################################################\n")
    f.write("\nBroj knjiga za prodaju od strane svakog izdavaca:\n")
    f.write("broj knjiga\t\tizdavac\n")
    d = data.loc[data['izdavac'].notnull(), 'izdavac'].value_counts().to_dict()
    for key in d:
        f.write('{: <5}      \t\t{}\n'.format(d[key], key))

    love_count = 0
    for idx in data.index:
        s = data.loc[idx, 'opis']
        if 'ljubav' in s or 'Ljubav' in s:
            love_count += 1

    f.write("\n########################################################################\n")
    f.write(f"\nBroj knjiga koje u opisu imaju rec 'LJUBAV' je {love_count}\n")
    
    f.write("\n########################################################################\n")
    f.write("\nBroj knjiga izdato po godinama, poslednjih 7 godina.\n")
    last_7_years = [2023, 2022, 2021, 2020, 2019, 2018, 2017]
    d = data.loc[:, 'godina_izdavanja'].value_counts().to_dict()
    for year in last_7_years:
        f.write('{}. godina: {} knjiga\n'.format(year, d[year]))

    f.write("\n########################################################################\n")
    f.write('\nRang lista 30 najskupljih knjiga u prodaji:\n')
    d = data.sort_values(by='cena', ascending=False).head(n=30).reset_index(drop=True).drop(columns=['opis', '_id', 'povrsina'])
    d = pd.DataFrame(d['cena']).join(d.drop(columns='cena'))
    d.index += 1
    f.write(d.to_string())

    f.write("\n\n########################################################################\n")
    f.write(f'\nU 2024. godini (do datuma 3.6.2024.) je objavljeno {data.loc[data['godina_izdavanja'] == 2024, 'godina_izdavanja'].count()} knjiga\n')
    f.write("Rang lista svih knjiga objavljenih u 2024. godini je:\n")
    d = data.sort_values(by='cena').reset_index(drop=True).drop(columns=['opis', '_id', 'povrsina'])
    d = pd.DataFrame(d['cena']).join(d.drop(columns='cena'))
    d = d.loc[d['godina_izdavanja'] == 2024, :].reset_index(drop=True)
    d.index += 1
    f.write(d.to_string())
    
    f.write("\n\n########################################################################\n")
    f.write('\nRang lista 30 knjiga sa najvecim brojem strana:\n')
    d = data.sort_values(by='broj_strana', ascending=False).head(n=30).reset_index(drop=True).drop(columns=['opis', '_id', 'povrsina'])
    d = pd.DataFrame(d['broj_strana']).join(d.drop(columns='broj_strana'))
    d.index += 1
    f.write(d.to_string())

    f.write('\n\nRang lista 30 knjiga sa najvecim cenama:\n')
    d = data.sort_values(by='cena', ascending=False).head(n=30).reset_index(drop=True).drop(columns=['opis', '_id', 'povrsina'])
    d = pd.DataFrame(d['cena']).join(d.drop(columns='cena'))
    d.index += 1
    f.write(d.to_string())

    f.write('\n\nRang lista 30 knjiga sa najvecim formatom:\n')
    d = data.sort_values(by='povrsina', ascending=False).head(n=30).reset_index(drop=True).drop(columns=['opis', '_id', 'povrsina'])
    d = pd.DataFrame(d['format']).join(d.drop(columns='format'))
    d.index += 1
    f.write(d.to_string())

    f.close()



def main():
    data = pd.read_csv("./knjigeDB.knjige.csv")
    print(data, end='\n\n\n')
    print(data.info(), end='\n\n\n')
    print(data.describe(), end='\n\n\n')
    data = filter_description(data)
    data = filter_name(data)
    data = filter_authors(data)
    data = filter_pages(data)
    data = filter_year(data)
    data = filter_category(data)
    data = filter_bind(data)
    data = filter_format(data)
    data = filter_publishers(data)
    log_data(data)
    data.drop(columns='povrsina').to_csv("preprocesirane_knjige.csv", index=False)
    
    

if __name__ == "__main__":
    main()