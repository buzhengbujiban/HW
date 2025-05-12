import pickle, os
from canopus_py.dump_shm import *
from canopus_py.ca_infer_utils import canopus_inference_allinone_helper
from qrfitter3.rnodes.qrfitter3_node import *

Y = "/data/beef3/data/CA.v24.1220.test_release/CNEQ/Panels/Y/Y_V5/terms.YYYYMMDD.sdiv"
W = "/data/bees3/data/CAPQ.v24.1220.test_release/CNEQ/Panels/I/fq5m/BB/terms.YYYYMMDD.sdiv"
ST = "/data/bees3/data/CAPQ.v24.1220.test_release/CNEQ/Panels/D/eod/sid_st/st_YYYYMMDD.sdiv"

DEFAULT_CONFIG = dict(
    Y_pattern = {"Y": {"pattern": Y, "columns": [], "calc_f": get_overnight_intraday_ret}},
    W_pattern = {"W": {"pattern": W, "columns": ['is.active'], "calc_f": wgt_mul_index}},
    univ = "G2548",
    is_start=20180101, is_end=20221231, os_start=20230101, os_end=20241227,
    stats_vd_start=20220101, fit_vd_start=20220701,
    exclude_start_date=20200201, exclude_end_date=20200231,
    is_output_ii=list(range(1,49,2)), os_output_ii=list(range(1,49,1)),
)


def main(
    fitting_name, X_pattern: Dict[str, dict|List[dict]], default_config=DEFAULT_CONFIG, fit_y="mret24h", fit_w="is.active",
    model="lgbnew", sampling_w=None, rolling:dict=None, load_para_spec=80, dump_config_only_list=[],
    extra_tasks_config:dict=None, grid_tasks_config:dict=None, fit_para_spec=1,
    rnode_root=f"/data/beer2/{os.environ['USER']}/equity_fitting",
    canopus_inference=True, canopus_inference_periods=["valid", "test"],
):
    Y_pattern = default_config["Y_pattern"]
    W_pattern = default_config["W_pattern"]
    univ = default_config["univ"]
    data = copy.deepcopy(X_pattern)
    data.update(Y_pattern | W_pattern)
    assert len(X_pattern) == 1, "tree can only have one X file."
    Xname, Yname, Wname = list(X_pattern.keys())[0], list(Y_pattern.keys())[0], list(W_pattern.keys())[0]
    is_end = default_config["os_end"] if isinstance(rolling, dict) and rolling.get("roll_type", None)=="vanilla" else default_config["is_end"]
    configs = [{
        "univ": univ, "start_date": default_config["is_start"], "end_date": is_end, "output_ii": default_config["is_output_ii"],
        "exclude_start_date": default_config["exclude_start_date"], "exclude_end_date": default_config["exclude_end_date"], 
    }]
    if canopus_inference:
        start_ends = prepare_shm(data=data, configs=configs, d_concat=True, dump_config_only_list=dump_config_only_list, para_spec=load_para_spec, rnode_root=rnode_root, chain_name=fitting_name)
        train_start_end = start_ends[0]
    else:
        configs.append({
            "univ": univ, "start_date": default_config["os_start"], "end_date": default_config["os_end"], "output_ii": default_config["os_output_ii"],
        })
        start_ends = prepare_shm(data=data, configs=configs, d_concat=True, dump_config_only_list=dump_config_only_list, para_spec=load_para_spec, rnode_root=rnode_root, chain_name=fitting_name)
        train_start_end, test_start_end = start_ends
        inference_dataset_config = QRFitter3Node.base_concat_dataset_config(start_end=test_start_end, Xname=Xname, Yname=Yname, Wname=Wname)
        inference_dataset_config["fit_w"] = fit_w
        inference_dataset_config["fit_y"] = fit_y                                   
    dataset_config = QRFitter3Node.base_concat_dataset_config(start_end=train_start_end, Xname=Xname, Yname=Yname, Wname=Wname)
    fitter_config = QRFitter3Node.base_fitter_config(model=model)  # lgbnew DF, lgb for all fs due to low memory usage
    task_config = {
        "train": {"start_date": str(default_config["is_start"]), "end_date": str(default_config["fit_vd_start"]-1)},
        "valid": {"start_date": str(default_config["fit_vd_start"]), "end_date": str(default_config["is_end"])},
        "test": {"start_date": str(default_config["os_start"]), "end_date": str(default_config["os_end"])},
    }
    dataset_config["fit_w"] = fit_w
    dataset_config["fit_y"] = fit_y
    if sampling_w is not None:
        dataset_config["sampling_w"] = sampling_w

    qf_config = {
        "canopus_inference": canopus_inference,
        "fitting_name": fitting_name,
        "dataset": dataset_config,
        "fitter": fitter_config,
        "task": task_config,
    }
    if extra_tasks_config is not None:
        qf_config["extra_tasks"] = extra_tasks_config
    if grid_tasks_config is not None:
        qf_config["grid_tasks"] = grid_tasks_config

    if canopus_inference:
        if "extra_tasks" in qf_config:
            for task in qf_config["extra_tasks"]:
                if "inference_dataset" in task:
                    del task["inference_dataset"]
    else:
        qf_config["inference_dataset"] = inference_dataset_config
    if rolling is not None:
        roller_confg = QRFitter3Node.base_roller_config(**rolling)
        qf_config["roller"] = roller_confg  # 如果指定了roller，就算task存在也会优先用roller，主要是因为croll inference要用task中的test period
    qf3 = QRFitter3Node(qf_config, label="expt_folder", clean_dir_name=True, rnode_root=rnode_root, chain_name=fitting_name)
    pkl_path = os.path.join(qf3.stage_dir(), "X_pattern.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(X_pattern, f)
    os.chmod(pkl_path, 0o700)
    qf3.gen_tasks()
    qf3.run(para_spec="1")
    prepare_shm(data=data, clear_data=True, d_concat=True)

    if canopus_inference:
        canopus_inference_allinone_helper(expt_folder=str(qf3.stage_dir()), inference_periods=canopus_inference_periods, univ=univ, slurm_cfg={"smem": 10000, "retries": 2, "timeout": "01:00:00"})


if __name__ == "__main__":
    pattern_list = [
        '/data/beer1/data/CAPQ.v24.1220.test_release/CNEQ/Panels/T/fq1m/PQ_bk2/terms.YYYYMMDD.sdiv',
        '/data/beer1/data/CAPQ.v24.1220.test_release/CNEQ/Panels/T/fq1m/PQ_pas/terms.YYYYMMDD.sdiv',
    ]
    fitting_name = "tree_CAPQ_1m_tesing"
    Xname = f"X.{fitting_name}"
    X_pattern = {Xname: {"pattern": pattern_list,},}
    main(fitting_name, X_pattern, fit_y="mret1h")