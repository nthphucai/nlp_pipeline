import random
import unittest

import torch

import pytest
from utils import extract_answer_bt_hl_token


class TestDataset(unittest.TestCase):
    """
    Test Dataset for Multitask
    """

    dataset = torch.load("output/data/history/multitask/train_data_hl_t5.pt")
    source_text = dataset["source_text"]
    target_text = dataset["target_text"]
    task = dataset["task"]

    start = random.randrange(len(dataset))
    end = start + 10
    for idc in range(start, end):
        if task[idc] == "answer_ext":
            answer_ext_idc = idc
        elif task[idc] == "qg":
            qg_task_idc = idc
        elif task[idc] == "qa":
            qa_task_idc = idc
        elif task[idc] == "mc":
            mc_task_idc = idc

    answer_ext_task = task[answer_ext_idc]
    answer_ext_src = source_text[answer_ext_idc]
    answer_ext_tget = target_text[answer_ext_idc]

    qa_task = task[qa_task_idc]
    qa_src = source_text[qa_task_idc]
    qa_tget = target_text[qa_task_idc]

    qg_task = task[qg_task_idc]
    qg_src = source_text[qg_task_idc]
    qg_tget = target_text[qg_task_idc]

    mc_task = task[mc_task_idc]
    mc_src = source_text[mc_task_idc]
    mc_tget = target_text[mc_task_idc]

    print(
        "task:", answer_ext_task, "\nsrc:", answer_ext_src, "\ntarget:", answer_ext_tget
    )
    print("\ntask:", qg_task, "\nsrc:", qg_src, "\ntarget:", qg_tget)
    print("\ntask:", qa_task, "\nsrc:", qa_src, "\ntarget:", qa_tget)
    print("\ntask:", mc_task, "\nsrc:", mc_src, "\ntarget:", mc_tget)

    answer = qa_tget.split("</s>")[0].strip()
    _, context = qa_src.split("context:")

    # Setting up for the test
    def setUp(self):
        pass

    # Cleaning up after the test
    def tearDown(self):
        pass

    def test_answer_ext_task(self):
        # answer extraction task
        answer_ext = self.answer_ext_src.split(":")
        prefix = answer_ext[0].strip()
        extract_text = answer_ext[1].strip()

        extract_answer = list(extract_answer_bt_hl_token(extract_text))[0]

        self.assertEqual(prefix, "extract_answer")
        self.assertIn(extract_answer, extract_text)

    def test_qg_task(self):
        # question answering task
        qg_src = self.qg_src.split(":")
        prefix = qg_src[0]
        self.assertEqual(prefix, "generate question")

    def test_qa_task(self):
        # qg task
        qa_src = self.qa_src.split(":")
        prefix = qa_src[0]
        self.assertEqual(prefix, "question")


if __name__ == "__main__":
    pytest.main()
