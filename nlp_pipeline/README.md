<div align="center">
    <h1>AI-powered Question Generation (AQG)</h1>
    <hr/>
</div>
<p align="center">
    <a href="https://gitlab.ftech.ai/nlp/research/aqg/-/blob/develop/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/license-Apache_2.0-blue.svg">
    </a>
    <br/>
</p>

## Introduction

**Automatic Question Generation (AQG)** - aka **QuestGen** - is an important task in Natural Language Processing (NLP)
that involves generating questions automatically when given a context paragraph. The most straight-forward way for this
is answer-aware question generation. In answer-aware question generation the model is presented with the answer and the
paragraph and asked to generate a question for that answer by considering the context paragraph. AQG can improve the
training of Question Answering (QA), help chatbots to start or continue a conversation with humans, and provide
assessment materials for educational purposes.

This project is aimed as an open source research on question generation with pre-trained transformers using
straight-forward end-to-end (e2e) methods without much complicated pipelines. The goal is to provide simplified data
processing, training methods and easy to use pipelines for inference.

## Requirements

This repository is tested on Python 3.7+ and with all [requirements.txt](requirements.txt) dependencies. We recommend to
run QuestGen in a [virtual environment](https://docs.python.org/3/library/venv.html).

1. Download and install Anaconda. Please see details at [here](https://docs.anaconda.com/anaconda/install/).

2. Create an Anaconda environment with Python 3.7+ (3.8 or greater would work as well) and activate it:

```bash
$ conda create --name aqg python=3.7.2
$ conda activate aqg
```

## Installing

To install our package, you can use following commandline.

```shell
$ pip install http://minio.dev.ftech.ai/questgen-package-v0.1-7f030021/questgen-0.1.0-py3-none-any.whl
```


Because `AQG` using `Punkt Sentence Tokenizer` of `NLTK` as a Tokenizer for preprocessing, you must download this module after install our package by following command:

```shell
$ python
>> import nltk
>> nltk.download('punkt')
```
For manual installation of NLTK please read more at [NLTK documentation](https://www.nltk.org/data.html).

## Model & Dataset
You can download our models & dataset sample from following URL.

##### 1. Multitask Model (Simple/Single QA):
+ _Vietnamese Model (Updating)_
+ [English Model](http://minio.dev.ftech.ai/fschool-english-simple-questgen-v1.1-21e842d1/fschool_english_simple_questgen_v1.1.zip) 

##### 2. MC Model (Multiple choice QA):
+ Vietnamese Model (Updating)
+ [English Model](http://minio.dev.ftech.ai/fschool-english-multiple-choice-questgen-v1.0-ae736037/fschool_english_multiple_questgen_v1.0.zip)

##### 3. QA Data Sample:
+ _Vietnamese Data (Updating)_
+ [English Data](http://minio.dev.ftech.ai/mcqg-english-v1.0.0-8ac19529/MCQG_English_Data_v1.0.0.json)
## Usage

### A. Inference

#### 1. Import

```python
from questgen.inference import Inference
```

#### 2. Init instance

To user our QuestGen inference, you must download our models first. To download our model for the first time to default
folder **/temp/**:

```python
inference = Inference(multitask_model_name_or_path="path_to_multitask_model",
                      mc_model_name_or_path="path_to_multiple_choices_model",
                      # mc_model_name_or_path is needed if task="mc"
                      config_aqg_path="path_to_configs",
                      download_model=None,
                      # Set download_model = "history" or "english" to download our trained model,
                      )
```

#### 3. Generate Q&A

**Predict on python List**

```python
qa = inference.create(task="multitask", context=["list", "of", "context"])
print(qa)

```
**Predict on file**

```python
qa = inference.create(task="multitask", data_path="path_to_context_file")
print(qa)
```
**Save inference result**
Pass save_path_mc/save_path_multitask args into create function to save inference result
```python
qa = inference.create(task="multitask", 
                      data_path="path_to_context_file",
                      save_path_mc="path to save mc result", 
                      save_path_multitask="path to save simple question result")
print(qa)
```
### B. Training

#### 1. Prepare training data

We provide a create_training_data pipeline to create training data from raw text.

```python
from questgen.dataset import build_dataset

build_dataset(
    task="mc or multitask",
    dataset_train_path="path_to_train_data",
    dataset_valid_path="path_to_valid_data",
    dataset_test_path="path_to_test_data",
    pretrained_tokenizer_name_or_path="path_to_tokenizer_or_tokenizer_name",
    customized_tokenizer_save_path="path_to_save_customized_tokenizer",
    output_dir="path_to_store_encoded_data",
    train_file_name="encoded_train_file_name",
    valid_file_name="encoded_valid_file_name",
    test_file_name="encoded_test_file_name"
)
```

#### 2. Train model

Download training config
file [here](http://minio.dev.ftech.ai/questgen-package-v0.1-7f030021/training_configs.yaml). If your browser doesn't download above link, please use following command:
```bash
wget http://minio.dev.ftech.ai/questgen-package-v0.1-7f030021/training_configs.yaml
```
You can change the training configuration by replace the value in the configuration file.

```python
from questgen import QuestGenTrainer

trainer = QuestGenTrainer(
    config_path="Path to config file"
)

trainer.train()
```

You can pass training parameters into `QuestGenTrainer` directly instead using a configuration file. Note that these
parameter must be defined in our sample configuration file.

```python
from questgen import QuestGenTrainer

trainer = QuestGenTrainer(
    model_name_or_path="Path to config file",
    train_file_path="Path to training data",
    valid_file_path="Path to valid data",
    output_dir="Path to output directory",
    model_type="'t5' or 'bart' or 't5-copy-enhance'",
)

trainer.train()
```

However, if configuration file and individual parameter were passed at the same time, `QuestGenTrainer` will prioritize
the configuration file.

#### 3. ONNX Training support:

We also support ONNX training with CUDA DEVICE, you can enable this by set `onnx_mode=True` in training configuration. 
However, your training environment must be setup for [onnx](https://github.com/microsoft/onnxruntime) and [optimum](https://github.com/huggingface/optimum), you can read more about this at [Huggingface document](https://huggingface.co/docs/transformers/installation)
or use our [docker image](https://gitlab.ftech.ai/nlp/research/questgen/-/blob/24-onnx-setup/docker/onnx-env/Dockerfile) 
(support `CUDA_VERSION 11.6`) which was tested and run successfully.

### C. Evaluate

QuestGen support evaluate output using various unsupervised automated metrics for NLG. For more information of these
metrics, please read more about [nlg-eval](https://github.com/Maluuba/nlg-eval).

#### 1. Prepare evaluate data

QuestGen provide automatic pipeline to create data for evaluation from your own data and model.

```python
from questgen import QuestGenEvaluator

evaluator = QuestGenEvaluator(
    task="Task to create evaluate data: 'multitask' or'multiplechoice",
    model_name_or_path="Path to pretrained model or model identifier from huggingface.co/models",
    valid_file_path="Path for input dataset",
    tokenizer_name_or_path="Pretrained tokenizer name or path",
    reference_path="Whether save the ground truth reference text strings, default value is 'references.txt', if None value were passed, reference data won't be created",
    output_path="Path to output directory"
)

evaluator.generate()
```

#### 2. Evaluate

QuestGen using `nlg-eval`to support evaluate output by various unsupervised automated metrics for NLG. Please use
following command to set up `nlg-eval`.

```bash
nlg-eval --setup
```

If you are using macOS High Sierra or higher, then run this to allow multithreading:

```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

For more information about nlg-eval installation and setup, please go
to [nlg-eval repositories](https://github.com/Maluuba/nlg-eval).

Then you can use QuestGen to evaluate your model output.

```python
evaluator.compute_metrics(result_save_path="path_to_save_eval_scores")
```