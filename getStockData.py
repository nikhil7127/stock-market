import requests
from bs4 import BeautifulSoup
import time as t
import pandas as p
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

class Stock_info():

    def __init__(self,from_date="11/12/2020",to_date="12/14/2020"):
        self.driver = webdriver.Chrome(executable_path="c:/selenium_python/chromedriver.exe")
        self.driver.maximize_window()
        self.driver.get("https://in.investing.com/equities/itc-historical-data")
        self.driver.implicitly_wait(10)
        self.driver.find_element_by_class_name("js-dropdown-display").click()
        t.sleep(2)
        self.driver.implicitly_wait(10)
        fromDate = self.driver.find_element_by_class_name("js-date-from")
        fromDate.send_keys(Keys.CONTROL+"a",Keys.DELETE)
        t.sleep(2)
        fromDate.send_keys(from_date)
        t.sleep(2)
        toDate = self.driver.find_element_by_class_name("js-date-to")
        toDate.send_keys(Keys.CONTROL+"a",Keys.DELETE)
        t.sleep(2)
        toDate.send_keys(to_date)
        t.sleep(2)
        self.driver.find_element_by_class_name("js-apply-button").click()
        t.sleep(2)

    def get_table(self,save=False,save_as="ITC_stock.csv"):
        HEADERS = ({'User-Agent': 'Nikhil\'s_request'})
        response = requests.get(self.driver.current_url, headers=HEADERS)
        soup = BeautifulSoup(response.text,"lxml")
        frame = p.DataFrame()
        frame["Date"] = [k.text.strip() for k in soup.find_all("td",attrs={"class":"col-rowDate"})]
        frame["Price"] = [k.text.strip() for k in soup.find_all("td",attrs={"class":"col-last_close"})]
        frame["Open"] = [k.text.strip() for k in soup.find_all("td",attrs={"class":"col-last_open"})]
        frame["High"] = [k.text.strip() for k in soup.find_all("td",attrs={"class":"col-last_max"})]
        frame["Low"] = [k.text.strip() for k in soup.find_all("td",attrs={"class":"col-last_min"})]
        frame["Volume"] = [k.text.strip() for ind,k in enumerate(soup.find_all("td",attrs={"class":"col-volume"})) if ind<len(frame["Low"])]
        frame["% Change"] = [k.text.strip() for k in soup.find_all("td",attrs={"class":"col-change_percent"})]
        frame = frame.iloc[::-1]
        if save:
            frame.to_csv(save_as,index=False)
        else:
            return frame

a = Stock_info(from_date="12/30/2019",to_date="12/30/2020")
a.get_table(save=True)
