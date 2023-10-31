import time
from typing import List, Optional

from tqdm import tqdm

from questgen.create_data.crawl_data import Crawler
from questgen.utils.file_utils import get_time, logger


class VietjackCrawler(Crawler):
    def __init__(
        self,
        domain: str,
        task: str,
        url: str,
        headless: bool,
        return_html: bool,
        database_config_path: str,
    ):
        """
        VietjackCrawler class.
        Args:
            domain (str): Domain of crawling data.
            task (str): Task of crawling data.
            url (str): Url of vietjack website.
            headless (bool): Show chrome browser window or not.
            return_html (bool): Reurn html raw of crawled data or not.
            database_config_path (str): Path of logging database config.
        """
        super().__init__(domain, task, headless, return_html, database_config_path)
        self.url = url

    def __get_chapter_url(self) -> List[str]:
        """
        Return chapter urls of current page.
        Returns:
            List[str]: A list of chapter urls.
        """
        return self.browser.execute_script(
            script="""
            return Array.from(document.querySelectorAll('.middle-col ul.list a')).map(function (e) {
                return location.origin + e.getAttribute('href').split("..")[1];
            })
        """
        )

    def __clean_ads(self):
        self.browser.execute_script(
            script="""
            function remove_els(class_name) {
                els = document.querySelectorAll(class_name);
                while (els.length > 0) {
                    els[0].remove();
                    els = document.querySelectorAll(class_name);
                }
            }
            remove_els('.ads_ads');
            remove_els('.ads_txt');
        """
        )

    def __get_content(self) -> str:
        """
        Get crawled content of current page.
        Returns:
            str: Crawled data.
        """
        if self.return_html:
            script = "return document.querySelector('.middle-col').innerHTML.trim();"
        else:
            script = "return document.querySelector('.middle-col').innerText.trim();"

        return self.browser.execute_script(script=script)

    def prepare_saving_item(self, data_input: str, data_output: Optional[str]) -> dict:
        """
        Prepare the logging database dictionary of crawled information.
        Args:
            data_input (str): Input crawling text.
            data_output (str): Response of ChatGPT.

        Returns:
            dict: Output dictionary of crawled information.
        """
        return {
            "domain": self.domain,
            "task": self.task,
            "source": "vietjack",
            "data": {"context": data_input, "crawl": data_output},
            "time": get_time(),
        }

    def crawl_data(self):
        """
        Crawling data.
        """
        self.create_browser()
        self.browser.get(self.url)
        time.sleep(1)
        chapter_url_lst = self.__get_chapter_url()
        chapter_url_lst = [
            url for url in chapter_url_lst if url != "https://vietjack.comundefined"
        ]

        success, fail = 0, 0
        for url in tqdm(chapter_url_lst, desc="Crawling data from pages: "):
            self.browser.get(url)
            try:
                time.sleep(1)
                self.__clean_ads()
                self.database.insert_one(
                    self.prepare_saving_item(
                        data_input=url, data_output=self.__get_content()
                    )
                )
                success += 1
            except Exception as e:
                logger.error(e)
                fail += 1
                self.database.insert_one(
                    self.prepare_saving_item(data_input=url, data_output=None)
                )
            finally:
                time.sleep(1)

        logger.info(f"# Success: {success} --- # Fail: {fail}")
        self.close_browser()
