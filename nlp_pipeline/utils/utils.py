import os

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm


def get_progress(iterable=None, total=None, desc=None, disable=False):
    """
    get progress bar
    :param iterable: target to be iterated
    :param total: total length of the progress bar
    :param desc: description of the progress bar
    :return: progress bar
    """
    return tqdm(iterable=iterable, total=total, desc=desc, disable=disable)


"""
Ex: calculate_dice(idx, df, axis, return_df=False)
mutiprocess(calculate_dice, range(len(pairs)), df=pairs, return_df=True)
"""


def multiprocess(
    iter_func,
    iter_args,
    workers=8,
    batch_size=32,
    use_index=False,
    disable=False,
    desc: str = None,
    **kwargs
):
    """
    parallel iterate array
    :param arr: array to be iterated
    :param iter_func: function to be called for each data, signature (idx, arg) or arg
    :param workers: number of worker to run
    :param batch_size: chunk size
    :param use_index: whether to add index to each call of iter func
    :return list of result if not all is None
    """
    n = len(iter_args)
    n_chunk = n // batch_size
    if n_chunk * batch_size != n:
        n_chunk += 1

    pool = mp.Pool(workers)
    chunks = np.array_split(iter_args, n_chunk)
    offset = 0
    final_results = []

    with get_progress(total=n, disable=disable, desc=desc) as pbar:
        for c in chunks:
            jobs = [
                pool.apply_async(
                    iter_func,
                    args=(offset + i, arg) if use_index else (arg,),
                    kwds=kwargs,
                )
                for i, arg in enumerate(c)
            ]

            results = []
            for j in jobs:
                results.append(j.get())
                pbar.update()

            final_results = final_results + results
            offset += len(c)

    pool.close()
    pool.join()

    if not all([r is None for r in final_results]):
        return final_results


"""
set gpu_ids for training, ids start from 0
"""


def set_gpu(*gpu_ids):
    assert len(gpu_ids) > 0, "require at least 1 is=d"

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in gpu_ids])

    # print(','.join([str(id) for id in gpu_ids]))


def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
