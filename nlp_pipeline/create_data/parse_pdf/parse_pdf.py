import argparse

from pdf_processor import PDFProcessor
from questgen.create_data.modules.preprocessor import Preprocessor
from questgen.utils.file_utils import logger, write_text_file


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="Path for input pdf file.")
parser.add_argument("--output_path", type=str, help="Path for output text file.")
parser.add_argument(
    "--parse_pdf_config_path",
    type=str,
    default="configs/parse_pdf_config.yml",
    help="Path for parse pdf config file.",
)


def main():
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    config_path = args.parse_pdf_config_path

    context = None

    logger.info("Create PDF Processor")
    try:
        processor = PDFProcessor(config_path)
        logger.info("Processor working")
        context = processor.parse_text_from_pdf(input_path)
    except Exception as e:
        logger.error("Something wrong with the config file!!!")
        logger.error(e)

    if context is None:
        logger.error("No context to preprocess")
    else:
        try:
            preprocessor = Preprocessor(config_path)
            context = preprocessor.preprocess_pdf_context(context)
            write_text_file(context, output_path)
            logger.info(f"Output at {output_path}")
        except Exception as e:
            logger.error("Something wrong with the config file!!!")
            logger.error(e)


if __name__ == "__main__":
    main()
