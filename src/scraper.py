import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
# import time

url = r'https://oilprice.com/'

# Request
r1 = requests.get(url)
r1.status_code

# We'll save in coverpage the cover page content
coverpage = r1.content

# Soup creation
soup1 = BeautifulSoup(coverpage, 'html5lib')

# This doesn't work with oilprice.com!
coverpage_news = soup1.find_all('h2', class_='newsBlock__title')
len(coverpage_news)

coverpage_news[0].get_text()
coverpage_news[-1].get_text()

for data in soup1.find_all('h2'):
    print(data)

almsot_data = soup1.find_all('div', class_='newsBlock')
type(almsot_data) # == bs4.element.ResultSet
len(almsot_data )
first_half_almost_data = almsot_data[0]

data  = first_half_almost_data.find_all('a', class_='newsBlock__article')
data[0]['href']
data[0].get_text()