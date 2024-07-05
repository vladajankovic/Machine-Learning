# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface

from itemadapter import ItemAdapter

class BookspiderPipeline:

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        if adapter['cena'] is not None:
            adapter['cena'] = int(adapter['cena'].split(',')[0].replace('.', ''))
        if adapter['broj_strana'] is not None:
            adapter['broj_strana'] = int(adapter['broj_strana'])
        if adapter['godina_izdavanja'] is not None:
            adapter['godina_izdavanja'] = int(adapter['godina_izdavanja'])
        return item

from pymongo import MongoClient

class DatabasePipeline:

    def __init__(self) -> None:
        self.connection = MongoClient("mongodb://localhost:27017")
        self.db = self.connection["knjigeDB"]
        if "knjige" in self.db.list_collection_names():
            self.db.drop_collection("knjige")
        self.coll = self.db['knjige']

    def process_item(self, item, spider):
        self.coll.insert_one(dict(item))
        return item
    
    def close_spider(self, spider):
        self.connection.close()