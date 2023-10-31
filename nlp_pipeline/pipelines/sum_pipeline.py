from itertools import chain
from typing import Union

import pandas as pd

from questgen.pipelines.modules.summarize.summary_text import TextSummarize
from questgen.pipelines.qg_pipeline import QAPipeline
from questgen.utils.utils import multiprocess


class SummaryPipeline(QAPipeline):
    def __call__(self, context: Union[dict, list], rate_cluster: float):
        df = pd.DataFrame.from_dict(context)
        df["summary"] = df["context"].apply(self._prepare_inputs_for_summary)
        df["summary"] = multiprocess(
            self._generate_summary, df["summary"], workers=4, rate_cluster=0.6
        )
        update_dict = [
            df.iloc[[idc]].to_dict(orient="records") for idc in range(df.shape[0])
        ]
        result = list(chain(*update_dict))

        return result

    def _generate_summary(self, sentences, rate_cluster) -> str:
        return TextSummarize(sentences, rate_cluster)()
