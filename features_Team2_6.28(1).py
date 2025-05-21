
# List of features:
####### feature 1: tsz_ret1
ret1 = data['load_simvar_hhmm']('ret1')                    
tsz_ret1 = op.at_nan2zero(op.ts_zscore(ret1, 21))          
df['tsz_ret1'] = tsz_ret1[flatten_mask]                    

####### feature 2: tsz_ret1_delay1
tsz_ret1_d1 = op.at_nan2zero(op.ts_zscore(op.ts_delay(ret1,1), 21))
df['tsz_ret1_d1'] = tsz_ret1_d1[flatten_mask]

####### feature 3: linkup_job_active_count
linkup_job_active_count = data['load_simvar_hhmm']('linkup_job_active_count')
linkup_job_active_count = op.at_nan2zero(op.ts_zscore(op.ts_fill(linkup_job_active_count), 21))
df['linkup_job_active_count'] = linkup_job_active_count[flatten_mask]

####### Your features:

####### inplementation
def CRO_vec(data, win_len=63):
    nT = data.shape[1]
    cro = np.full(data.shape, np.nan, dtype=np.float32)
    for i in tqdm(range(nT)):
        data_window = data[:, max(i - win_len + 1, 0):i + 1]
        win_len_real = data_window.shape[1]
        day_labels = np.arange(win_len_real, 0, -1)

        # Normalize A and b by subtracting their respective means
        A_mean = np.nanmean(data_window, axis=1, keepdims=True)
        b_mean = np.nanmean(day_labels)

        A_centered = data_window - A_mean
        b_centered = day_labels - b_mean

        # Compute standard deviations
        A_std = np.nanstd(A_centered, axis=1, keepdims=True)
        b_std = np.nanstd(b_centered)

        # Standardize A and b
        A_standardized = A_centered / A_std
        b_standardized = b_centered / b_std

        # Calculate correlations using dot product
        cro[:, i] = np.dot(A_standardized, b_standardized) / win_len_real

    return cro

def VSS(data, win_len=63):
    nT = data.shape[1]
    nS = data.shape[0]
    vss = np.full(data.shape, np.nan, dtype=np.float32)
    for i in tqdm(range(nT)):
        data_window = data[:, max(i - win_len + 1, 0):i + 1]
        data_window_ = data_window - data_window[:, 0].reshape(-1, 1)
        win_len_real = data_window_.shape[1]

        data_linear = np.tile(np.linspace(0.0, 1.0, win_len_real), (nS, 1)) * data_window_[:, -1][:, np.newaxis]
        delta_cum = data_window_ - data_linear

        Hp = np.sum(np.where(delta_cum > 0.0, delta_cum, 0.0), axis=1)
        Hm = np.sum(np.where(delta_cum <= 0.0, -delta_cum, 0.0), axis=1)

        vss[:, i] = Hp / (Hp + Hm)

    return vss

import scipy
def asymetric(datay, tails=1, lookback=25):
    asymetric = np.full(datay.shape, np.nan, dtype=np.float32)
    for i in tqdm(range(lookback, datay.shape[1])):
        past_21_days = datay[:, i-lookback:i+1]
        z_scores = scipy.stats.zscore(past_21_days, axis=1)
        z_score_diff = (z_scores > tails).sum(axis=1) - (z_scores < -tails).sum(axis=1)
        asymetric[:, i] = z_score_diff
    return asymetric








####### feature 1: cro
close = data['load_simvar_hhmm']("close")
shsout = data['load_simvar_hhmm']('shsout')
uprice = shsout * close
cro = CRO_vec(uprice, 252) # 0.105, 0.095 (no clip)

####### feature 2: vss_uclose
close = data['load_simvar_hhmm']('close')
uclose_shs = close * shsout
vss_uclose = VSS(uclose_shs, 63)
vss_uclose_csz = op.cs_zscore(vss_uclose) # 0.099, 0.093
df['vss_uclose'] = op.at_nan2zero(vss_uclose_csz)[flatten_mask]

####### feature 3: asymetric_ret1
industry = data["load_simvar_hhmm"]('industry')
ret1 = data["load_simvar_hhmm"]('ret1')
asymetric_ret1 = op.at_nan2zero((op.cs_indneut(op.cs_zscore(asymetric(ret1, 1, 42)),  industry)))
df['asymetric_ret1'] = asymetric_ret1[flatten_mask]  # 0.090 0.089  £ 0.106  0.092   £ 0.105  0.093

