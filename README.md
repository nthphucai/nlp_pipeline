<div align="center">
    <br>
    <h1>Question Generation</h1>
    <p>
    A reasonably standardized project structure for NLP.
    </p>
    <hr/>
</div>
<p align="center">
    <a href="https://gitlab.ftech.ai/nlp/research/pizza-nlp/-/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/license-Apache_2.0-blue.svg">
    </a>
    <br/>
</p>

## Requirements

- Python 3.8+
- All libraries/packages can be installed by using `requirements_dev.txt` file:

```bash
$ pip install -r requirements_dev.txt
```

## Usage

### Training pipeline

- Before begining the training model step, you have to prepare the training and validation dataset. The preparing dataset step has 3 dataset formats to use for 3 tasks:
  - `simple`: Use for simple question generation task.
  - `mc`: Use for multiple-choice question generation task.
  - `multitask`: Use for simple and multitple-choice question generation task.
- You can change config which fit your dataset and model, then using file `prepare_dataset.sh` (it is located at `scripts/trainer/prepare_dataset.sh`) or below scripts:

```
export MODEL_PATH="vietai/vit5-base"
export TRAIN_DATA_PATH="data/mc/mcqg_visquad_train_data_v2.0.1.json"
export VALID_DATA_PATH="data/mc/mcqg_visquad_test_data_v2.0.1.json"

python questgen/dataset/build_dataset.py \
    --task multitask \
    --model_type t5 \
    --model_name_or_path $MODEL_PATH \
    --pretrained_tokenizer_name_or_path $MODEL_PATH \
    --customized_tokenizer_save_path t5_qg_tokenizer \
    --output_dir output/data/history/multitask/ \
    --valid_for_qg_only \
    --qg_format highlight_qg_format \
    --dataset_train_path $TRAIN_DATA_PATH \
    --dataset_valid_path $VALID_DATA_PATH \
    --dataset_test_path $VALID_DATA_PATH \
    --max_source_length 768 \
    --max_target_length 128 \
    --train_file_name train_data_hl_t5.pt \
    --valid_file_name valid_data_hl_t5.pt \
    --test_file_name test_data_hl_t5.pt
```

- After preparing data, you now can run the script to training your  question generation model. The example script is located at `scripts/trainer/train_aqg.sh` or you can use the script below:

```
export MODEL_PATH="vietai/vit5-base"
export TRAIN_DATA_PATH="output/data/history/multitask/train_data_hl_t5.pt
export VALID_DATA_PATH="output/data/history/multitask/valid_data_hl_t5.pt

python questgen/trainer/run_aqg.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path t5_qg_tokenizer \
    --model_type t5 \
    --project_name question_generation_multitask \
    --output_dir output/model/history/multitask \
    --train_file_path $TRAIN_DATA_PATH \
    --valid_file_path $VALID_DATA_PATH \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --weight_decay 1e-6 \
    --num_train_epochs 10 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --seed 42 \
    --do_train \
    --do_eval \
    --remove_unused_columns False \
    --logging_steps 2 \
    --report_to wandb \
    --evaluation_strategy epoch \
    --logging_first_step True \
    --save_strategy epoch \
    --save_total_limit 2 \
    --load_best_model_at_end False \
    --greater_is_better False \
    --metric_for_best_model eval_loss \
    --auto_find_batch_size True \
    --overwrite_output_dir True
```
