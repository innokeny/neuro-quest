import hashlib
from typing import Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
import time
import joblib
from pathlib import Path
import time
from treelib import Tree
from loguru import logger

class Page:
    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver
    
    @property
    def soup(self) -> BeautifulSoup:
        return BeautifulSoup(self.driver.page_source, 'html.parser')

    @property
    def hash(self) -> str:
        try:
            content = self.driver.find_element(By.CSS_SELECTOR, 'div[style*="padding:0px 30px 50px 30px"]').text
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.error(e)
            return ""

short_fantasy_quests = ''

class ChooseYourStoryParser:
    def __init__(self, timeout: Optional[int] = 60*10):
        self.driver = webdriver.Chrome()
        self.wait = WebDriverWait(self.driver, 10)
        self.page = Page(self.driver)
        self.tree = Tree()
        self.start: Optional[float] = None
        self.timeout = timeout

    def div(self):
        return self.page.soup.find('div', style=lambda x: x and 'padding:0px 30px 50px 30px' in x) # type: ignore
    
    def text(self, div) -> str:
        text = ' '.join([p.get_text(strip=True) for p in div.find_all('p') if p.get_text(strip=True)]) # type: ignore
        return text
    
    def options(self, div) -> list[str]:
        options = div.find_all('a', onclick=lambda x: x and 'PostBack' in x) # type: ignore
        options_texts: list[str] = [option.get_text(strip=True) for option in options]
        return [o for o in options_texts if o.lower().find('go back') == -1 or o.lower().find('exit') == -1 or o.lower().find('reset') == -1]
    
    def __wait(self, hsh: str):
        time.sleep(.5)
        self.wait.until(lambda d: hash(self.page) != hsh)
    
    def __back(self, root: bool):
        hsh = self.page.hash
        if not root:
            self.driver.back()
            self.__wait(hsh)
    
    def __check_timeout(self):
        if self.start is None:
            self.start = time.time()
        delta = time.time() - self.start
        if self.timeout and delta > self.timeout:
            raise TimeoutError(f'Timeout reached: {delta}')


    def _parse(self, parent: Optional[str], tag: str):
        self.__check_timeout()
        hsh = self.page.hash
        if self.tree.contains(hsh):
            logger.info(f"Node {hsh} already exists")
            self.__back(parent is None)
            self.__wait(hsh)
            return


        try:
            div = self.div()
        except Exception as e:
            logger.error(e)
            self.__back(parent is None)
            return
        
        try:
            text = self.text(div)
        except Exception as e:
            logger.error(e)
            self.__back(parent is None)
            return
        
        try:
            options = self.options(div)
        except Exception as e:
            logger.error(e)
            self.__back(parent is None)
            return
        
        self.tree.create_node(
            tag=tag,
            identifier=hsh,
            parent=parent,
            data=dict(text=text)
        )
        logger.success(f"Added node {hsh}")

        for option in options:
            try:
                self.driver.find_element(By.LINK_TEXT, option).click()
                self.__wait(hsh)
                self._parse(hsh, option)
            except Exception as e:
                logger.error(e)
        
        self.__back(parent is None)
    
    def save(self, path: Path):
        joblib.dump(self.tree, path)

    
    def parse(self, url: str, directory: Path = Path(__file__).parent):
        self.driver.get(url)
        self.start = time.time()
        try:
            self._parse(None, 'root')
            logger.success(f"Done with {url}")
        except Exception as e:
            logger.error(e)
        self.driver.quit()
        idx = parse_qs(urlparse(url).query).get('StoryId', hashlib.md5(url.encode()).hexdigest())[0]
        self.save(directory / f"{idx}.tree")