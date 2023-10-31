from torch.multiprocessing import set_start_method
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from questgen.pipelines.mc_pipeline import MCPipeline
from questgen.pipelines.multitask_pipeline import MultiTaskPipeline
from questgen.tests.utils import cpu_usage, memory_usage
from questgen.utils.file_utils import load_json_file, read_yaml_file


class PerformanceMeasureMC(MCPipeline):
    def __init__(self, task, examples, **kwargs):
        super().__init__(**kwargs)

        self.examples = examples
        self.task = task

    def __call__(self, module_type: str):
        examples = self.examples
        if module_type == "preprocess":
            examples = self.measure_preprocess(self.examples)

        elif module_type == "generator":
            examples = self.preprocess_input_mc(examples)
            examples = self.measure_generator(examples)

        elif module_type == "postprocess":
            examples = self.preprocess_input_mc(self.examples)
            distractors = self.generate(examples)
            examples = self.measure_postprocess(examples, distractors)
        else:
            examples = self.measure_preprocess(examples)
            distractors = self.measure_generator(examples)
            examples = self.measure_postprocess(examples, distractors)

    @cpu_usage
    @memory_usage
    def measure_preprocess(self, examples):
        examples = self.preprocess_input_mc(examples)
        return examples

    @cpu_usage
    @memory_usage
    def measure_generator(self, examples):
        output = self.generate(examples)
        return output

    @cpu_usage
    @memory_usage
    def measure_postprocess(self, examples, distractors):
        output = self.postprocess_output_mc(examples, distractors)
        return output


class PerformanceMeasureMULTITASK(MultiTaskPipeline):
    def __init__(self, task, examples, use_summary, **kwargs):
        super().__init__(**kwargs)

        self.examples = examples
        self.task = task
        self.use_summary = use_summary

    def __call__(self, module_type: str):
        examples = self.examples
        if module_type == "preprocess":
            examples = self.measure_preprocess(self.examples)

        elif module_type == "generator":
            examples = self.preprocess_input_multitask(examples)
            examples = self.measure_generator(examples)

        elif module_type == "postprocess":
            examples = self.preprocess_input_multitask(examples)
            examples = self.generate_batch_qa_pairs(examples)
            examples = self.measure_postprocess(examples)
        else:
            examples = self.measure_preprocess(examples)
            examples = self.measure_generator(examples)
            examples = self.measure_postprocess(examples)

    @cpu_usage
    @memory_usage
    def measure_preprocess(self, examples):
        examples = self.preprocess_input_multitask(examples)
        return examples

    @cpu_usage
    @memory_usage
    def measure_generator(self, examples):
        output = self.generate_batch_qa_pairs(examples)
        return output

    @cpu_usage
    @memory_usage
    def measure_postprocess(self, examples):
        output = self.postprocess_output_multitask(examples)
        return output


def measure_performance(
    task: str,
    module_type: str,
    use_summary: bool,
    use_multiprocess: bool,
    data_path: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str = None,
    config_path: str = None,
):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    if tokenizer_name_or_path is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    config = read_yaml_file(config_path)
    examples = load_json_file(data_path)["data"]
    print("The number of data:", len(examples))

    if task == "mc":
        performance = PerformanceMeasureMC(
            task,
            examples,
            use_multiprocess=use_multiprocess,
            num_options=4,
            fillmask_model_path=None,
            model=model,
            tokenizer=tokenizer,
            **config["generate_distractors"]
        )
    elif task == "multitask":
        performance = PerformanceMeasureMULTITASK(
            task,
            examples,
            use_summary=use_summary,
            use_multiprocess=use_multiprocess,
            model=model,
            tokenizer=tokenizer,
            **config["generate_qa"]
        )
    return performance(module_type=module_type)


if __name__ == "__main__":
    set_start_method("spawn")
    measure_performance(
        task="multitask",
        module_type="postprocess",
        use_summary=True,
        use_multiprocess=True,
        data_path="output/multitask/mc_infer_model_v1_0.json",
        model_name_or_path="output/models/simple-question/v1.2",
        config_path="configs/faqg_pipeline_t5_vi_base_hl.yaml",
    )
