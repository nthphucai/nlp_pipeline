# -*- coding: utf-8 -*-
# Le Tran Bao Minh Code License

# Permission is hereby granted, free of charge, to any person obtaining a copy of this code and associated documentation files, authored by Le Tran Bao Minh, to use, copy, modify, and distribute the Code for any purpose, subject to the following conditions:
# 1. Redistributions of the Code must retain the above copyright notice, this list of conditions, and the following disclaimer.
# 2. The Code is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall Le Tran Bao Minh or the copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the Code or the use or other dealings in the Code.
# For inquiries or more information, please contact: letbaominh@gmail.com
""" ChatGPT Crawling """


import os
import time
from typing import Optional

from loguru import logger
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm
from undetected_chromedriver.webelement import By

from questgen.create_data.crawl_data import Crawler
from questgen.create_data.modules.mc_chatgpt_preprocessor import ChatGPTPreprocessor
from questgen.utils.file_utils import get_time, load_json_file


class ChatGPTCrawler(Crawler):
    def __init__(
        self,
        domain: str,
        task: str,
        headless: bool,
        return_html: bool,
        n_loop: int,
        data_path: str,
        accounts_path: str,
        database_config_path: str,
        is_preprocessing: bool,
    ):
        """
        ChatGPTCrawler class.
        Args:
            domain (str): Domain of crawling data.
            task (str): Task of crawling data.
            headless (bool): Show chrome browser window or not.
            return_html (bool): Return raw html crawled data or not.
            n_loop (int): Number of iterations during crawling one example.
            data_path (str): Path of input data.
            accounts_path (str): Path of accounts data.
            database_config_path (str): Path of logging database config.
            is_preprocessing (bool): Preprocessing crawled data or not.
        """
        super().__init__(domain, task, headless, return_html, database_config_path)
        self.idx = 0
        self.n_loop = n_loop
        self.data_path = data_path
        self.accounts_path = accounts_path
        self.is_preprocessing = is_preprocessing
        self.__import_data()

    def __import_data(self):
        if not os.path.exists(self.data_path):
            logger.error(f"data_path: `{self.data_path}` is not exist")
            self.data = None
        else:
            self.data = load_json_file(self.data_path)

        if not os.path.exists(self.accounts_path):
            logger.error(f"accounts_path: `{self.accounts_path}` is not exist")
            self.accounts_path = None
        else:
            self.accounts = load_json_file(self.accounts_path)

    @staticmethod
    def __normalize_text(text: str) -> str:
        """
        Normalize the input text for crawling data.
        Args:
            text (str): Input data for crawling data.

        Returns:
            str: Normalized data.
        """
        return text.replace("\n", "\\n").replace('"', '\\"').strip()

    def __check_login(self) -> bool:
        """
        Check current page is the login page or not.
        Returns:
            bool: True if current page is the login page or vice versa.
        """
        return self.browser.execute_script(
            script="""
                els = document.querySelectorAll(".text-center");
                if (els.length > 0) {
                    for (let i = 0; i < els.length; i ++) {
                        if (els[i].innerText.trim().toLowerCase().indexOf("welcome") >= 0) {
                            return true;
                        } else if (els[i].innerText.trim().toLowerCase().indexOf("get started") >= 0) {
                            return true;
                        }
                    }
                }
                return false
            """
        )

    def __click_log_in_button(self):
        self.browser.execute_script(
            script="""
                els = document.querySelectorAll("button");
                for (let i = 0; i < els.length; i++) {
                    if (els[i].innerText.trim().toLowerCase().indexOf("log in") >= 0) {
                        els[i].click();
                        break;
                    }
                }
            """
        )

    def __wait_elements(self, element: str) -> bool:
        """
        Check the element has been showed up or not.
        Args:
            element (str): ID of element.

        Returns:
            bool: True if the element has been showed up or vice versa.
        """
        return self.browser.execute_script(
            script="""
                els = document.querySelectorAll(\"#"""
            + element
            + """\")
                if (els.length > 0)
                    return true;
                return false;
            """
        )

    def __input_information(self, element: str, information: str):
        """
        Input the information into current page by the element id.
        Args:
            element (str): ID of input element.
            information (str): Text of input information.
        """
        while not self.__wait_elements(element):
            time.sleep(1)

        self.browser.execute_script(
            script=f'document.querySelector("#{element}").value = "{self.__normalize_text(information)}"'
        )

    def __check_notification(self) -> bool:
        """
        Check the pop up notification is showed up or not.
        Returns:
            bool: True if the pop up notification is showed up or vice versa.
        """
        return self.browser.execute_script(
            script="""
                els = document.querySelectorAll("button.btn.ml-auto");
                if (els.length > 0)
                    return true;
                return false;
            """
        )

    def __escape_notification(self):
        while self.__check_notification():
            self.browser.execute_script(
                script='document.querySelector("button.btn.ml-auto").click();'
            )
            time.sleep(1)

    def __login(self) -> bool:
        """
        Login ChatGPT.
        Args:
            account (dict): Account dictionary (it has 2 keys is `username` and `password`).

        Returns:
            bool: True if login successfully or vice versa.
        """
        self.browser.get("https://chat.openai.com/")
        time.sleep(2)

        # Check log in page
        count = 0
        while not self.__check_login():
            self.browser.execute_script(script="location.reload()")
            time.sleep(5)
            count += 1
            if count == 10:
                return False

        self.__click_log_in_button()
        time.sleep(1)

        if self.idx == len(self.accounts):
            self.idx = 0

        # Input username
        self.__input_information("username", self.accounts[self.idx]["username"])
        self.browser.execute_script(
            script='document.querySelector("._button-login-id").click();'
        )
        time.sleep(1)

        # Input password
        self.__input_information("password", self.accounts[self.idx]["password"])
        self.browser.execute_script(
            script='document.querySelector("._button-login-password").click();'
        )
        time.sleep(1)

        self.idx += 1

        time.sleep(10)
        self.__escape_notification()
        return True

    def __wait_generating(self) -> bool:
        """
        Check if ChatGPT finished generating or not.
        Returns:
            bool: True if ChatGPT finished or vice versa.
        """
        self.browser.execute_script(
            script="""
                let els = document.querySelectorAll("div.flex.items-center.justify-center.gap-2");
                for (let i = 0; i < els.length; i++) {
                    if (els[i].innerText.trim() == "Continue generating") {
                        els[i].click();
                    }
                }
            """
        )
        return self.browser.execute_script(
            script="""
            let els = document.querySelectorAll("div.flex.items-center.justify-center.gap-2");
            for (let i = 0; i < els.length; i++)
                if (els[i].innerText.trim() == "Regenerate") return true;
            return false;
        """
        )

    def __get_data(self, text: str) -> Optional[str]:
        """
        Get chatting data from ChatGPT.
        Args:
            text (str): Input text crawling data.

        Returns:
            Optional[str]: Response of ChatGPT.
        """
        time.sleep(2)

        self.browser.execute_script(
            script="""
                document.querySelector("textarea").value = \""""
            + self.__normalize_text(text)
            + """\";
        """
        )
        time.sleep(1)
        button = self.browser.find_element(By.TAG_NAME, "textarea")
        button.send_keys(Keys.ENTER)
        time.sleep(1)
        self.browser.execute_script(
            script="""
                els = document.querySelectorAll(".absolute.p-1.rounded-md");
                if (els.length >= 0) els[0].click();
            """
        )
        time.sleep(5)

        count = 0
        while not self.__wait_generating():
            time.sleep(1)
            count += 1
            if count == 600:
                return None

        if self.return_html:
            return self.browser.execute_script(
                script="""
                    els = document.querySelectorAll(".markdown");
                    return els[els.length - 1].innerHTML.trim();
                """
            )
        return self.browser.execute_script(
            script="""
                els = document.querySelectorAll(".markdown");
                return els[els.length - 1].innerText.trim();
            """
        )

    def __new_turn(self) -> bool:
        """
        Begin a new turn of crawling data from ChatGPT.
        Returns:
            bool: True if create browser and login successfully or vice versa.
        """
        if self.browser is not None:
            self.close_browser()
            time.sleep(300)
        self.create_browser()
        count = 0
        while not self.__login():
            self.close_browser()
            self.create_browser()
            count += 1
            if count == len(self.accounts):
                return False
        return True

    def prepare_saving_item(self, data_input: str, data_output: str) -> dict:
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
            "source": "chatgpt",
            "data": {
                "context": data_input,
                "crawl": data_output,
            },
            "time": get_time(),
        }

    def crawl_data(self):
        """
        Crawling data.
        """
        logged = self.__new_turn()
        if not logged:
            logger.error("Something wrong with crawling pipeline")
            return

        data = self.data

        loop = tqdm(data)
        loop.desc = "Crawling"
        loop.set_postfix(account=self.accounts[self.idx - 1]["username"].split("@")[0])
        for item in loop:
            for _ in range(self.n_loop):
                try:
                    response = self.__get_data(item)
                except Exception as e:
                    logger.warning(e)
                    response = None
                while response is None:
                    logger.info("Changing to another account")
                    logged = self.__new_turn()
                    if not logged:
                        logger.error("Something wrong with crawling pipeline")
                        return
                    loop.set_postfix(
                        account=self.accounts[self.idx - 1]["username"].split("@")[0]
                    )
                    try:
                        response = self.__get_data(item)
                    except Exception as e:
                        logger.warning(e)
                        response = None
                self.database.insert_one(
                    self.prepare_saving_item(data_input=item, data_output=response)
                )
        self.close_browser()

        if self.is_preprocessing:
            preprocessor = ChatGPTPreprocessor(
                task=self.task,
                domain=self.domain,
                database_config_path=self.database_config_path,
            )
            preprocessor.preprocessing_data()
