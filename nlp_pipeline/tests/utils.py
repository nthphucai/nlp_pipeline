import os
import re
import time
from itertools import accumulate
from typing import List

import psutil


def get_cpu_usage():
    # Getting loadover15 minutes
    load1, load5, load15 = psutil.getloadavg()
    cpu_usage = (load5 / os.cpu_count()) * 100
    return cpu_usage


def cpu_usage(func):
    def wrapper(*args, **kwargs):
        mem_before = get_cpu_usage()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_cpu_usage()
        print(
            "{}: cpu before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
                func.__name__,
                mem_before,
                mem_after,
                mem_after - mem_before,
                elapsed_time,
            )
        )
        return result

    return wrapper


def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def memory_usage(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        print(
            "{}: memory before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
                func.__name__,
                mem_before,
                mem_after,
                mem_after - mem_before,
                elapsed_time,
            )
        )
        return result

    return wrapper


def split_lists(inputs: List[str], lengths_to_split: List[int]) -> List[str]:
    # split a list into sublist of given lengths
    output = [
        inputs[x - y : x]
        for x, y in zip(accumulate(lengths_to_split), lengths_to_split)
    ]
    return output


def extract_answer_bt_hl_token(context: str):
    # find pattern {hl_token} string {hl_token}
    extract_answer = list(re.finditer("<hl>", context))

    num_length = len(extract_answer) // 2
    input_lst = split_lists(extract_answer, [2] * num_length)

    for idc in range(len(input_lst)):
        start_idc = input_lst[idc][0].start()
        end_idc = input_lst[idc][1].end()
        out = context[start_idc:end_idc]
        yield out
