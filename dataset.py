import numpy as np
from qrfitter3.nnfitter.torch_dataset_sdiv import get_data_d_concat


def get_attrs(dim, Ds, Is, Ss):
    if dim == ('S', 'D', 'I', 'V'):
        S_array = np.repeat(Ss, len(Ds)*len(Is))
        D_array = np.tile(np.repeat(Ds, len(Is)), len(Ss))
        I_array = np.tile(Is, len(Ss)*len(Ds))
    elif dim == ('D', 'I', 'S', 'V'):
        D_array = np.repeat(Ds, len(Ss)*len(Is))
        I_array = np.tile(np.repeat(Is, len(Ss)), len(Ds))
        S_array = np.tile(Ss, len(Ds)*len(Is))
    else:
        raise NotImplementedError(f"dim={dim} not supported.")
    return S_array, D_array, I_array

def get_splited_array_table(dataset_config, task_config, set_names=["train", "valid", "test"]):
    x_xarr, y_xarr, w_xarr, x_names, y_names, w_names, dim = get_data_d_concat(dataset_config)
    datasets = {}
    for set_name in set_names:
        if set_name not in task_config.keys():  # task_config中可以不指定valid
            continue
        dates = (task_config[set_name]['start_date'],  task_config[set_name]['end_date'])
        if "use_x_names" in dataset_config:
            x_names = np.array(dataset_config["use_x_names"])  # 选择部分features
            x_xarr = x_xarr.sel(V=x_names)  # 这里应该会造成拷贝
        x_, y_, w_ =  x_xarr.sel(D=slice(*dates)).values.reshape(-1, len(x_names)).squeeze(),  y_xarr.sel(D=slice(*dates)).values.reshape(-1, len(y_names)),  w_xarr.sel(D=slice(*dates), V=dataset_config['fit_w']).values.reshape(-1).squeeze()
        S, D, I = get_attrs(dim, Ds=y_xarr.sel(D=slice(*dates)).D.values, Is=y_xarr.sel(D=slice(*dates)).I.values, Ss=y_xarr.sel(D=slice(*dates)).S.values)
        if "sampling_w" in dataset_config:
            sampling_w_ = w_xarr.sel(D=slice(*dates), V=dataset_config["sampling_w"]).values.reshape(-1).squeeze()
            x_, y_, w_ = x_[sampling_w_==1], y_[sampling_w_==1], w_[sampling_w_==1]  # 注意这里会产生copy，如果sampling_w_.sum()比较大的话慎用
            S, D, I = S[sampling_w_==1], D[sampling_w_==1], I[sampling_w_==1]
        assert x_.ndim == 2
        assert y_.ndim == 2
        assert w_.ndim == 1
        assert len(x_) == len(y_)
        assert len(x_) == len(w_)
        datasets[set_name] = {'X': x_, 'Y': y_, 'w': w_, 'S': S, 'D': D, 'I': I, 'x_names': x_names.tolist(), 'y_names': y_names.tolist(), 'fit_y': dataset_config['fit_y']} 
    return datasets