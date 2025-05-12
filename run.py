import os
from . import learner as learner_module
from .dataset import get_splited_array_table


def run_fit_task(config):
    dataset_config, task_config, fitter_config = config['dataset'], config['task'], config['fitter']
    fitter_config.update({"outdir": task_config["outdir"], "expt_name": config["fitting_name"]})
    datasets = get_splited_array_table(dataset_config, task_config, set_names=["train", "valid"])
    if not config.get("canopus_inference", False):  # 如果用canopus_inference则无需配test dataset
        if "inference_dataset" in config:  # test set的dataset不一样
            inference_dataset_config = config["inference_dataset"]
            dataset_test = get_splited_array_table(inference_dataset_config, task_config, set_names=["test"])
        else:
            dataset_test = get_splited_array_table(dataset_config, task_config, set_names=["test"])
        datasets.update(dataset_test)
    learner = getattr(learner_module, fitter_config["class"])(fitter_config)
    learner.run(datasets)
    
def run_inference_task(config):
    inference_dataset_config, task_config, fitter_config = config['inference_dataset'], config['task'], config['fitter']
    fitter_config.update({"outdir": task_config["outdir"]})
    datasets = get_splited_array_table(inference_dataset_config, task_config, set_names=["period"])
    learner = getattr(learner_module, fitter_config["class"])(fitter_config)
    if "inference" in task_config["task"]:
        learner.run_inference({task_config["set_name"]: datasets["period"]})

#
#source /data/nfshome/ruud/.bash_profile_py11 && /data/gfsy3/apps/mambaforge/envs/dev11/bin/python /data/nfshome/ruud/git/rug/qrfitter3/qrfitter3/apps/run_fit_task.py /data/nfshome/ruud/git/rug/qrfitter3/qrfitter3/mlfitter/test/fit_task_cfg.json