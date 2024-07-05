# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class BookspiderItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass

class BookItem(scrapy.Item):
    naziv = scrapy.Field()
    autor = scrapy.Field()
    kategorija = scrapy.Field()
    izdavac = scrapy.Field()
    godina_izdavanja = scrapy.Field()
    broj_strana = scrapy.Field()
    tip_poveza = scrapy.Field()
    format = scrapy.Field()
    opis = scrapy.Field()
    cena = scrapy.Field()