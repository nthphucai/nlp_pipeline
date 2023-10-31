import torch


class BestRatioMixedFineTune:
    ratio_range = torch.arange(0.1, 1.0, 0.05)

    def __init__(
        self,
        len_pretrained_train_ds: int,
        len_new_train_ds: int,
        len_total_train_ds: int,
    ):
        """This class aims to find the best ratio b/t Dataset
            in every single Batch for Mixed Fine-Tune Method

        Args:
            len_pretrained_train_ds (int): The length of pretrained train dataset
            len_new_train_ds (int): The length of new train dataset
            len_total_train_ds (int): The length of total train dataset
        """
        self.len_pretrained_train_ds = len_pretrained_train_ds
        self.len_new_train_ds = len_new_train_ds
        self.len_total_train_ds = len_total_train_ds

        self.rate_bt_dataset = self.len_pretrained_train_ds / self.len_new_train_ds

    @classmethod
    def from_dataset(
        cls,
        len_pretrained_train_ds: int,
        len_new_train_ds: int,
        len_total_train_ds: int,
    ):
        return cls(len_pretrained_train_ds, len_new_train_ds, len_total_train_ds)

    def get_best_ratio(self) -> torch.Tensor:
        rate = abs(
            torch.tensor(list(map(self._calculate_rate_one_item, self.ratio_range)))
            - self.rate_bt_dataset
        )
        index = torch.argmin(rate)
        best_ratio = self.ratio_range[index]
        best_ratio = torch.round(best_ratio, decimals=2)
        return best_ratio

    def _calculate_rate_one_item(self, ratio: float) -> float:
        weights = torch.tensor(
            [
                ratio / self.len_pretrained_train_ds
                if i < self.len_new_train_ds
                else (1 - ratio) / self.len_new_train_ds
                for i in range(self.len_total_train_ds)
            ]
        )
        rate = weights[-1] / weights[0]
        return rate
