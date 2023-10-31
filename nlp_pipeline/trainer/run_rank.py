import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture
from transformers import HfArgumentParser

from questgen.ranking.qa_utils import save_sklearn_model
from questgen.utils.file_utils import load_json_file
from questgen.utils.utils import seed_everything as set_seed


# Setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_path: Optional[str] = field(metadata={"help": "Path for valid dataset"})
    eval_data_path: Optional[str] = field(metadata={"help": "Path for valid dataset"})
    test_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path for test dataset"}
    )

    gaussian_mixture_path: str = field(
        default=None, metadata={"help": "Path for gaussian_mixture_path model"}
    )

    isotonic_regressor_path: str = field(
        default=None, metadata={"help": "Path for isotonic_regressor_path model"}
    )

    features_qa_path: str = field(
        default="data/qa_ranking/npy", metadata={"help": "Path for test dataset"}
    )


def main():
    # Set seed
    set_seed(42)

    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    eval_data = load_json_file(data_args.eval_data_path)

    feature_train = np.load(Path(f"{data_args.features_qa_path}", "feature_train.npy"))
    print(f"the number of train data: {feature_train.shape}")

    feature_eval = np.load(Path(f"{data_args.features_qa_path}", "feature_eval.npy"))
    print(f"the number of eval data: {feature_eval.shape}")
    target_eval = [eval_data[idc]["targets"] for idc in range(len(eval_data))]
    print(f"target_eval: {target_eval[:10]}")

    if data_args.test_data_path is not None:
        test_data = load_json_file(data_args.test_data_path)
        feature_test = np.load("qa_ranking/data/npy/feature_test.npy")
        target_test = [test_data[idc]["targets"] for idc in range(len(test_data))]
        print(f"the number of test data: {feature_test.shape[0]}")
        print(f"target_test: {target_test[:10]}")

    # Apply Gaussian Mixture
    gaussian_mixture = GaussianMixture(
        covariance_type="spherical", n_components=18, max_iter=int(1e7), verbose=1
    )

    gaussian_mixture.fit(feature_train)

    log_probs_val = gaussian_mixture.score_samples(feature_eval)

    # Apply isotonic-regressor to score samples
    isotonic_regressor = IsotonicRegression(out_of_bounds="clip")
    isotonic_regressor.fit(log_probs_val, target_eval)

    save_sklearn_model(Path(data_args.gaussian_mixture_path), gaussian_mixture)
    save_sklearn_model(Path(data_args.isotonic_regressor_path), isotonic_regressor)

    logger.info(
        f"save gaussian_mixture_path at %s {Path(data_args.gaussian_mixture_path)}"
    )
    logger.info(
        f"save isotonic_regressor_path at %s {Path(data_args.isotonic_regressor_path)}"
    )


if __name__ == "__main__":
    main()
