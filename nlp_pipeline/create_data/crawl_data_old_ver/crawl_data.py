import argparse
import os

from questgen.create_data.crawl_data import ChatGPTCrawler, VietjackCrawler
from questgen.utils.file_utils import format_arg_str, get_time, logger


def prepare_arguments():
    parser = argparse.ArgumentParser(description="Data Crawling")
    parser.add_argument("--domain", type=str, help="Domain of crawling data")
    parser.add_argument("--task", type=str, help="Task of crawling data")
    parser.add_argument(
        "--source", type=str, help="Source of crawling data (`vietjack` or `chatgpt`)"
    )
    parser.add_argument(
        "--return_html",
        default=False,
        type=bool,
        help="Data output is in html format or not",
    )
    parser.add_argument(
        "--url", default=None, type=str, help="Url for vietjack crawling data"
    )
    parser.add_argument(
        "--n_loop",
        default=1,
        type=int,
        help="Number of loop iteraction when crawling an item with ChatGPT",
    )
    parser.add_argument(
        "--headless", type=bool, default=False, help="Show web browser window or not"
    )
    parser.add_argument(
        "--data_path", default=None, type=str, help="Path of ChatGPT crawling data"
    )
    parser.add_argument(
        "--accounts_path",
        default=None,
        type=str,
        help="Path of ChatGPT accountf for crawling data",
    )
    parser.add_argument(
        "--is_preprocessing",
        default=False,
        type=bool,
        help="Preprocessing crawled data from ChatGPT or not",
    )
    parser.add_argument(
        "--database_config_path", type=str, help="Config of database connection"
    )
    return parser.parse_args()


def main():
    args = prepare_arguments()
    logger.info(format_arg_str(args))

    if not os.path.exists(args.database_config_path):
        logger.error(f"{args.database_config_path} is not exist")
        return

    if args.source.lower() == "vietjack":
        crawler = VietjackCrawler(
            domain=args.domain,
            task=args.task,
            headless=args.headless,
            return_html=args.return_html,
            url=args.url,
            database_config_path=args.database_config_path,
        )
    elif args.source.lower() == "chatgpt":
        crawler = ChatGPTCrawler(
            domain=args.domain,
            task=args.task,
            headless=args.headless,
            return_html=args.return_html,
            n_loop=args.n_loop,
            data_path=args.data_path,
            accounts_path=args.accounts_path,
            database_config_path=args.database_config_path,
            is_preprocessing=args.is_preprocessing,
        )
        if crawler.data is None or crawler.accounts is None:
            return
    else:
        logger.warning("Only support source from `vietjack` or `chatgpt`")
        return

    # TODO:
    crawler.crawl_data()

    # if crawler.database is not None:
    #     logger.info("Data crawling has been starting...")
    #     crawler.crawl_data()
    # else:
    #     logger.error("Something wrong with database connection!!!\nPleases check it!!!")


if __name__ == "__main__":
    logger.info("-" * 10 + " BEGIN: " + get_time() + " " + "-" * 10)
    main()
    logger.info("-" * 10 + " END: " + get_time() + " " + "-" * 10)
