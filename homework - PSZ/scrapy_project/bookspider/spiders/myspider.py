import scrapy
import re

from bookspider.items import BookItem


class MyspiderSpider(scrapy.Spider):
    name = "myspider"
    allowed_domains = ["knjizare-vulkan.rs"]
    start_urls = ["https://www.knjizare-vulkan.rs/domace-knjige"]

    custom_settings = {
        'FEEDS': {
            'knjige.json': {'format': 'json'},
        }
    }

    def parse(self, response):
        book_links = response.css('div.product-listing-items div.text-wrapper div.title a::attr(href)').getall()
        for link in book_links:
            yield response.follow(url=link, callback=self.parse_book)
        
        next_page = response.css('li.next.first-last a::attr(href)').get()
        if next_page is not None:
            next_page = next_page.split('(')[1].split(')')[0]
            next_page_url = 'https://www.knjizare-vulkan.rs/domace-knjige/page-' + next_page
            yield response.follow(url=next_page_url, callback=self.parse)


    def parse_book(self, response):

        text = response.css('#tab_product_description::text').getall()
        text = [re.sub("\r\n", "", t).strip() for t in text]
        text = [t for t in text if t != ""]
        text = " ".join(text)

        table_rows = response.css('table.table tbody tr *::text').getall()
        table_rows = [re.sub("\r\n", "", t).strip() for t in table_rows]
        table_rows = [t for t in table_rows if t != ""]
        table_dict = {}
        for k in range(0, len(table_rows), 2):
            table_dict[table_rows[k]] = table_rows[k+1]

        book_item = BookItem()
        
        book_item['naziv'] = response.css('div.title h1 span::text').get()
        book_item['autor'] = table_dict.get('Autor')
        book_item['kategorija'] = table_dict.get('Kategorija')
        book_item['izdavac'] = table_dict.get('Izdavač')
        book_item['godina_izdavanja'] = table_dict.get('Godina')
        book_item['broj_strana'] = table_dict.get('Strana')
        book_item['tip_poveza'] = table_dict.get('Povez')
        book_item['format'] = table_dict.get('Format')
        book_item['opis'] = text
        book_item['cena'] = response.css('.product-price-without-discount-value::text').get()
        
        yield book_item