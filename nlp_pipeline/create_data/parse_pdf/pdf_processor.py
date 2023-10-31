import math
import os
import shutil
from typing import List

import requests
from tqdm.auto import tqdm, trange

import fitz
from questgen.utils.file_utils import logger, read_yaml_file


NUM_PAGES = 5


class PDFProcessor:
    def __init__(self, config: str) -> None:
        self.config = read_yaml_file(config)
        self.endpoint = self.config["ocr_endpoint"]

    def _sort_files(self, files: List[str]) -> List[str]:
        """
        Sort split pdf files by name.
        Args:
            files (List[str]): List of input files name.

        Returns:
            List[str]: List of sorted files name.
        """
        file_dict = {}
        for file in files:
            file_dict[file] = int(file.split(".")[0].split("-")[-1])
        file_dict = sorted(file_dict.items(), key=lambda x: x[1])
        return [item[0] for item in file_dict]

    def _split_pdf_file(self, file_path: str):
        """
        Split a pdf file to many smaller pdf files.
        Args:
            file_path (str): Path for pdf file.
        """
        if not os.path.exists(file_path):
            logger.error(f"File path is not existed at {file_path}")
        elif ".pdf" not in file_path:
            logger.error("Only support with pdf file")
        else:
            splitted_dir = os.path.join("/".join(file_path.split("/")[:-1]), "splitted")
            if not os.path.exists(splitted_dir):
                os.mkdir(splitted_dir)
            splitted_dir = splitted_dir

            input_pdf = fitz.open(file_path)
            for i in trange(
                math.ceil(len(input_pdf) / NUM_PAGES), desc="Splitting pdf file: "
            ):
                output_pdf = fitz.open()
                if i == math.ceil(len(input_pdf) / NUM_PAGES):
                    max_pages = len(input_pdf)
                else:
                    max_pages = (i + 1) * NUM_PAGES - 1
                output_pdf.insert_pdf(
                    input_pdf, from_page=i * NUM_PAGES, to_page=max_pages
                )
                output_pdf.save(
                    os.path.join(
                        splitted_dir,
                        file_path.split("/")[-1].split(".")[0]
                        + f"-{i*NUM_PAGES}-{max_pages}.pdf",
                    )
                )
            del input_pdf
            del output_pdf

    def _parse_with_ocr(self, input_path: str) -> List[str]:
        """
        Parsing text from pdf file by using ocr tool.
        Args:
            input_path (str): Path for input pdf file.
            output_path (str): Path for output text file.
        """
        cwd = os.getcwd()
        context = []
        splitted_dir = os.path.join("/".join(input_path.split("/")[:-1]), "splitted")
        if not os.path.join(splitted_dir):
            logger.error("Something wrong with splitted pdf folder")
        else:
            os.chdir(splitted_dir)
            splitted_dir = os.getcwd()
            pdf_files = os.listdir()
            pdf_files = self._sort_files(pdf_files)
            for pdf_file in tqdm(pdf_files, desc="Parsing text from pdf files: "):
                with open(os.path.join(splitted_dir, pdf_file), "rb") as files:
                    output = requests.post(
                        self.endpoint,
                        headers={
                            "Access-Control-Allow-Origin": "*",
                            "Accept-Encoding": "gzip, deflate, br",
                        },
                        files=[("files", (pdf_file, files, "application/pdf"))],
                        data={},
                        timeout=5,
                    )
                if output.status_code == 200:
                    context.extend(output.json()["questions"])
                else:
                    logger.error("Something wrong with ocr parsing server")
                    break
        os.chdir(cwd)
        shutil.rmtree(splitted_dir)
        return context

    def parse_text_from_pdf(self, input_path: str) -> List[str]:
        """
        Parsing text from pdf file function.
        Args:
            input_path (str): Path for input pdf file.
            output_path (str): Path for output text file.
        """
        logger.info(f"Parse text from {input_path}")

        logger.info("Splitting pdf file into many smaller pdf files")
        self._split_pdf_file(input_path)

        logger.info("Parsing text from many pdf files")
        return self._parse_with_ocr(input_path)
