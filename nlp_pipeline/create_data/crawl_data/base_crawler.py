from abc import ABC, abstractmethod

import undetected_chromedriver as uc

from mongodb.mongo_client import connect_mongodb


class Crawler(ABC):
    def __init__(
        self,
        domain: str,
        task: str,
        headless: bool,
        return_html: bool,
        database_config_path: str,
    ):
        self.browser = None
        self.domain = domain
        self.task = task
        self.headless = headless
        self.return_html = return_html
        self.database_config_path = database_config_path
        self._initialize()

    def _initialize(self):
        self.database = connect_mongodb(
            self.database_config_path, "crawl_collection_name"
        )

    def create_browser(self):
        options = uc.ChromeOptions()
        options.headless = self.headless

        self.browser = uc.Chrome(options=options)

    def close_browser(self):
        self.browser.close()
        self.browser = None

    @abstractmethod
    def prepare_saving_item(self, data_input: str, data_output: str) -> dict:
        pass

    @abstractmethod
    def crawl_data(self):
        pass
