import numpy as np
import operators as op
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import time
from xgboost import XGBRegressor
import xgboost as xgb
from numpy import exp, log

def ts_days_since_max(matrix, lookback):
    matrix = op.ts_fill(matrix)
    num_stocks, num_dates = matrix.shape
    argmax_matrix = np.zeros((num_stocks, num_dates))
    for t in range(num_dates):
        start = max(0, t - lookback + 1)
        end = t + 1
        current_window = matrix[:, start:end]
        max_indices = np.argmax(current_window, axis=1)
        relative_indices = t - (start + max_indices)
        argmax_matrix[:, t] = relative_indices
        nan_mask = np.isnan(current_window)
        all_nan_rows = nan_mask.all(axis=1)
        argmax_matrix[all_nan_rows, t] = int(lookback / 2)
    return argmax_matrix

### slope function
def slope_ols(inp):
    out = np.full(inp[0].shape, np.nan, order='F', dtype=np.float32)
    for di in range(out.shape[1]):
        yi = inp[:, :, di].T
        xi = np.tile(np.arange(0, inp.shape[0]), (out.shape[0], 1))
        ymean = np.nanmean(yi, axis=1, keepdims=True)
        xmean = np.nanmean(xi, axis=1, keepdims=True)
        a = (xi-xmean) * (yi-ymean)
        b = (xi-xmean) * (xi-xmean)
        k = np.nansum(a, axis=1, keepdims=True) / np.nansum(b, axis=1, keepdims=True)
        out[:,di] = k.flatten()
    return out

def get_earnings_mask(data):
    tdays = data['load_simvar_hhmm']('trading_days_til_next_ann')
    valid_mask = 1*(tdays == 1) + 2*(op.ts_delay(tdays,1) == 1)
    fs_next_ann_time = data['load_simvar_hhmm']('fs_next_ann_time')
    fs_next_ann_time_delay = op.ts_delay(1.0*fs_next_ann_time, 1)

    # based on trading_days_til_next_ann and fs_next_ann_time
    # delay one day if earnings is announced after we trade
    valid_effect_mask = np.zeros_like(valid_mask, dtype=np.float32)
    for si, di in zip(*np.where((valid_mask >= 1) & (data['valids']))):
        if di + data['delay'] >= data['numdates']:
            continue

        time = fs_next_ann_time[si, di]
        time_delay = fs_next_ann_time_delay[si, di]

        if time <= 1500 and (valid_mask[si,di]==1):
            valid_effect_mask[si, di] = 1

        elif time_delay > 1500  and (valid_mask[si,di]==2):
            valid_effect_mask[si, di] = 1

    bet_days_mask = valid_effect_mask==1
    bet_days_mask = op.at_nan2zero(bet_days_mask*1.0)
    flatten_mask = (bet_days_mask==1.0)

    ret_excess = data['load_simvar_hhmm']('ret1_excess')
    fwd_ret = op.ts_delay((ret_excess),-1)

    x_axis = np.where(flatten_mask[:,-1] == True)[0]
    flatten_mask[np.isnan(fwd_ret)] = False
    if len(x_axis)>0:
        flatten_mask[x_axis,-1] = True

    return flatten_mask

def get_basic_df(data, flatten_mask):
    ret1 = data['ret1'].copy()
    ret_excess = data['load_simvar']('ret1_excess')
    fwd_ret = op.ts_delay((ret_excess),-1)

    target = fwd_ret[flatten_mask]

    df = pd.DataFrame()
    df['target'] = (target)
    si_list = [list(np.arange(ret1.shape[0]))]*ret1.shape[1]
    si_list = np.array(si_list).transpose()
    di_list = [list(np.arange(ret1.shape[1]))]*ret1.shape[0]
    di_list = np.array(di_list)
    df['si'] = si_list[flatten_mask]
    df['di'] = di_list[flatten_mask]
    dates_mat = np.tile(data['dates'],(np.shape(ret1)[0],1))
    df['dates'] = dates_mat[flatten_mask]

    return df

############################################# FEATURE CHANGES - make your changes in this function #############################################
def get_df_feat(data, flatten_mask):
    load_simvar_hhmm = data['load_simvar_hhmm']
    load_simvar      = data['load_simvar']
    df = pd.DataFrame()

    ####### feature 1: tsz_ret1
    ret1 = load_simvar_hhmm('ret1')
    tsz_ret1 = op.at_nan2zero(op.ts_zscore(ret1, 21))
    df['tsz_ret1'] = tsz_ret1[flatten_mask]

    ####### feature 2: tsz_ret1_delay1
    tsz_ret1_d1 = op.at_nan2zero(op.ts_zscore(op.ts_delay(ret1,1), 21))
    df['tsz_ret1_d1'] = tsz_ret1_d1[flatten_mask]

    ####### feature 3: linkup_job_active_count
    linkup_job_active_count = op.at_zero2nan(load_simvar_hhmm('linkup_job_active_count'))
    linkup_job_active_count = op.at_nan2zero(op.ts_zscore(op.ts_fill(linkup_job_active_count), 21))
    df['linkup_job_active_count'] = linkup_job_active_count[flatten_mask]

    ####### feature 6: team-16's sentiment feature:
    neu = op.ts_fill(load_simvar_hhmm('thefly_h_neu'))
    neu = op.at_nan2zero(op.ts_mean(neu, 63))
    df['thefly_h_neu'] = neu[flatten_mask]


    ####### feature 7: team-12's options feature:
    num_days_30 = 10
    call_itm_vol_30 = op.ts_fill(load_simvar_hhmm('ivy_ITM_call_30_volume_vol_wt_mean'))
    call_atm_vol_30 = op.ts_fill(load_simvar_hhmm('ivy_ATM_call_30_volume_vol_wt_mean'))
    call_otm_vol_30 = op.ts_fill(load_simvar_hhmm('ivy_OTM_call_30_volume_vol_wt_mean'))
    var_load = op.ts_mean(call_otm_vol_30,num_days_30)/(op.ts_mean(call_itm_vol_30,num_days_30) + op.ts_mean(call_atm_vol_30,num_days_30))
    var_load = op.at_nan2zero(var_load)
    df['option_Sachin'] = var_load[flatten_mask]

    ####### feature 10: ret1_spx
    var_load = op.ts_fill(load_simvar_hhmm('ret1_SPX'))
    var_load = op.at_nan2zero(op.ts_rank(var_load,63))
    df['ret1_SPX_rank'] = var_load[flatten_mask]


    ####### feature 11: Sanchit & Richard - analyst
    def analyst_graph_connection(return_var, graph_var, valids):
        return_var = op.ts_fill(op.at_nan2zero(return_var))
        var = np.full(return_var.shape, np.nan)
        for di in range(len(graph_var)):
            graph_mat = graph_var[di]
            ret = return_var[:, di]
            valid = np.ones(ret.shape) * valids[:, di] * ~np.isnan(ret)
            w_unnormalized = graph_mat.T.dot(ret)
            norm_factor = graph_mat.T.dot(valid)
            di_graph_momentum = op.at_nan2zero(w_unnormalized / norm_factor)
            var[:, di] = di_graph_momentum
        var_avg = op.ts_fill(op.at_nan2zero(var))
        signal = var_avg - return_var
        return signal
    def expand_to_uncompressed(compressed_data, valid_stock_mask):
        uncompressed_data = np.zeros((len(valid_stock_mask), compressed_data.shape[1]), dtype=compressed_data.dtype)
        uncompressed_index = np.where(valid_stock_mask)[0]
        uncompressed_data[uncompressed_index, :] = compressed_data
        return uncompressed_data
    def compress_from_uncompressed(uncompressed_data, valid_stock_mask):
        compressed_data = uncompressed_data[valid_stock_mask]
        return compressed_data
    # uncompressing the valids
    valid_stock_mask = data['valid_stock_mask']
    valids_uncompressed = expand_to_uncompressed(data['valids'],valid_stock_mask)
    # uncompressing the return_var
    return_var_compressed = op.ts_fill(load_simvar_hhmm('ibes_pdet_pred_ann_ret_down_quarter'))
    return_var = expand_to_uncompressed(return_var_compressed,valid_stock_mask)
    # making the uncompressed signal
    relationship = load_simvar_hhmm('fs_analyst_relation_20days')
    signal_uncompressed = analyst_graph_connection(return_var, relationship, valids_uncompressed)
    # re-compressing the signal
    signal_compressed = np.zeros_like(return_var_compressed)
    signal_compressed = compress_from_uncompressed(signal_uncompressed,valid_stock_mask)
    df['analyst_alpha'] = signal_compressed[flatten_mask]

    ###### feature 12: bbganalyst
    var = op.ts_fill(load_simvar_hhmm('bbganalyst'))
    tsz_bbganalyst = op.ts_zscore(var,21)
    df['tsz_bbganalyst'] = op.at_nan2zero(tsz_bbganalyst[flatten_mask])

    ###### feature 13: spillover effect
    var_industry = load_simvar_hhmm('industry')
    def fill_nans_by_industry_average(array, industry):
        unique_industries = np.unique(industry)
        filled_array = np.copy(array)

        for industry_value in unique_industries:
            industry_mask = (industry == industry_value)
            for col in range(array.shape[1]):
                column_values = array[:, col]
                industry_column_values = column_values[industry_mask[:, col]]
                mean_value = np.nanmean(industry_column_values)
                filled_array[industry_mask[:, col], col] = np.where(np.isnan(filled_array[industry_mask[:, col], col]), mean_value, filled_array[industry_mask[:, col], col])

        return filled_array
    v_ret1 = load_simvar('ret1')
    v_ret1 = np.where(abs(op.ts_zscore(v_ret1,5))>1.0, op.ts_zscore(v_ret1,5), np.nan)
    v_ret1 = v_ret1 * (op.ts_delay(1.0*flatten_mask,1))
    v_ret1 = op.ts_sum((v_ret1),5)
    # Apply the function
    filled_array = fill_nans_by_industry_average(op.at_zero2nan(v_ret1), var_industry)
    avg_ret1_recent = filled_array
    df['avg_ret1_recent'] = avg_ret1_recent[flatten_mask]


    ####### Yingjie's feature
    price_std_allday = data['load_simvar_hhmm']('price_std_allday')
    upret_price_std_allday = data['load_simvar_hhmm']('upret_price_std_allday')
    df['upret_price_std_allday_ratio_mean63'] = (((op.ts_mean(upret_price_std_allday / price_std_allday, 63))))[flatten_mask]

    ####### Kinjal's feature
    featname = "recency_ratio_with_dailybeta"
    var = data["load_simvar_hhmm"]('beta_daily63')
    rel_change = (var-op.ts_delay(var, 1))/var
    alpha = 1 - ts_days_since_max(rel_change, 63)/63
    alpha = op.at_nan2zero(alpha)
    df[featname] = alpha[flatten_mask]

    featname = "recency_ratio_with_target_price_mean"
    var = data["load_simvar_hhmm"]('fs_price_tgt_cons_mean')
    rel_change = (var-op.ts_delay(var, 1))/var
    alpha = 1 - ts_days_since_max(rel_change, 63)/63
    alpha = op.at_nan2zero(alpha)
    df[featname] = alpha[flatten_mask]

    ####### Peicheng's team feature-2 and 3
    fast_pre_cube_q_same_analyst_change_yield = op.ts_sum(load_simvar_hhmm('fast_pre_cube_q_same_analyst_change_yield'), 63)
    df['fast_pre_cube_q_same_analyst_change_yield'] = fast_pre_cube_q_same_analyst_change_yield[flatten_mask]

    rkd_deii_ntp_same_analyst_change_yield = op.ts_sum(load_simvar_hhmm('rkd_deii_ntp_same_analyst_change_yield'), 63)
    df['rkd_deii_ntp_same_analyst_change_yield'] = rkd_deii_ntp_same_analyst_change_yield[flatten_mask]

    ####### dpv corr
    dpv_corr = op.ts_mean_exp(load_simvar_hhmm('dpv_rank_corr_allday'), 5, 2/5)
    df['dpv_corr'] = dpv_corr[flatten_mask]

    ibes_sumk_ope_median_next_q1 = load_simvar_hhmm('ibes_sumk_ope_median_next_q1')
    ibes_sumk_ope_median_next_q2 = load_simvar_hhmm('ibes_sumk_ope_median_next_q2')
    ibes_sumk_ope_median_next_q3 = load_simvar_hhmm('ibes_sumk_ope_median_next_q3')
    analyst_stru1 = op.ts_zscore(slope_ols(np.stack((ibes_sumk_ope_median_next_q1, ibes_sumk_ope_median_next_q2, ibes_sumk_ope_median_next_q3), axis=0)), 126)
    df['analyst_stru1'] = analyst_stru1[flatten_mask]

    ibes_sum_prr_median_next_q1 = load_simvar_hhmm('ibes_sum_prr_median_next_q1')
    ibes_sum_prr_median_next_q2 = load_simvar_hhmm('ibes_sum_prr_median_next_q2')
    ibes_sum_prr_median_next_q3 = load_simvar_hhmm('ibes_sum_prr_median_next_q3')
    analyst_stru2 = op.ts_zscore(slope_ols(np.stack((ibes_sum_prr_median_next_q1, ibes_sum_prr_median_next_q2, ibes_sum_prr_median_next_q3), axis=0)), 126)
    df['analyst_stru2'] = analyst_stru2[flatten_mask]

    ibes_sum_eps_low_next_q1 = load_simvar_hhmm('ibes_sum_eps_low_next_q1')
    ibes_sum_eps_low_next_q2 = load_simvar_hhmm('ibes_sum_eps_low_next_q2')
    ibes_sum_eps_low_next_q3 = load_simvar_hhmm('ibes_sum_eps_low_next_q3')
    analyst_stru3 = op.ts_zscore(slope_ols(np.stack((ibes_sum_eps_low_next_q1, ibes_sum_eps_low_next_q2, ibes_sum_eps_low_next_q3), axis=0)), 126)
    df['analyst_stru3'] = analyst_stru3[flatten_mask]

    ####### team 6
    nanex_ITM_call_30_volsurf_premium_mean = load_simvar_hhmm("nanex_ITM_call_30_volsurf_premium_mean")
    nanex_ITM_put_60_volsurf_premium_mean = load_simvar_hhmm("nanex_ITM_put_60_volsurf_premium_mean")
    alpha = op.cs_zscore((op.ts_fill(nanex_ITM_call_30_volsurf_premium_mean)-op.ts_fill(nanex_ITM_put_60_volsurf_premium_mean))/op.ts_fill(nanex_ITM_put_60_volsurf_premium_mean))
    df['feat'] = alpha[flatten_mask]

    nanex_ITM_call_30_volsurf_premium_mean = load_simvar_hhmm("nanex_ITM_call_30_volsurf_premium_mean")
    nanex_ITM_put_30_volsurf_premium_mean = load_simvar_hhmm("nanex_ITM_put_30_volsurf_premium_mean")
    alpha = op.ts_zscore((op.ts_fill(nanex_ITM_call_30_volsurf_premium_mean)-op.ts_fill(nanex_ITM_put_30_volsurf_premium_mean))/op.ts_fill(nanex_ITM_put_30_volsurf_premium_mean),20)
    df['feat2'] = alpha[flatten_mask]

    nanex_High_ITM_put_60_volsurf_vola_mean = load_simvar_hhmm("nanex_High_ITM_put_60_volsurf_vola_mean")
    nanex_ATM_call_60_volsurf_vola_mean = load_simvar_hhmm("nanex_ATM_call_60_volsurf_vola_mean")
    alpha = (op.ts_fill(nanex_High_ITM_put_60_volsurf_vola_mean)-op.ts_fill(nanex_ATM_call_60_volsurf_vola_mean))/op.ts_fill(nanex_ATM_call_60_volsurf_vola_mean)
    df['feat3'] = alpha[flatten_mask]

    ####### feature 82: Analyst eps forecast dispersion
    analyst_std=op.ts_fill(data['load_simvar_hhmm']('zacks_ce_eps_stddev_next_q1'))
    analyst_mean=op.ts_fill(data['load_simvar_hhmm']('zacks_ce_eps_mean_next_q1'))
    analyst_eps=op.at_nan2zero(analyst_std/np.abs(analyst_mean))
    analyst_eps = op.at_zero2nan(analyst_eps)
    analyst_eps = (analyst_eps + 4)/8
    df['analyst_eps']=analyst_eps[flatten_mask]


    ########## final df #############
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0) #here we fill the missing values with zeros, you may change this if required

    return df



def squared_softplusplus(y_pred: np.ndarray,
                y_true: xgb.DMatrix):
    y_true = y_true.get_label()
    grad = (-0.3*y_true + 1.4*(y_pred - y_true)*(exp(y_pred*y_true) + 1))/(exp(y_pred*y_true) + 1)
    hess = (0.3*y_true**2*exp(y_pred*y_true) + 1.4*exp(2*y_pred*y_true) + 2.8*exp(y_pred*y_true) + 1.4)/(1.0*exp(2*y_pred*y_true) + 2.0*exp(y_pred*y_true) + 1.0)
    return grad, hess




############################################## MODEL CHANGES - make your changes in this function ###############################################
def fit_model(feat_X, Y):
    # XGB as a new benchmark

    bst = xgb.train(
        {
            'objective':'reg:squarederror',
            'eta': 0.1,
            'max_depth': 1,
            'seed': 42,
        },
        xgb.DMatrix(feat_X, label=Y),
        num_boost_round=1400,
        obj=squared_softplusplus,
    )

    return bst


############################################################ END DOING YOUR CHANGES #############################################################


old_settings = np.seterr(divide='ignore')
lookback = 252 * 12


class fit_function:
    def __init__(self):
        self.prod_alpha_postA_length = 10  ## Either 1 or 10

    def fit(self, data, filter_matrix):
        assert (np.all(np.isfinite(filter_matrix)))

        alpha_list = data['alpha_list']
        dates = data['dates']
        delay = data['delay']
        rebalance_dates_mask = data['rebalance_dates_mask']
        fixed_di = np.where(dates >= 20191210)[0][0]

        flatten_mask = get_earnings_mask(data)
        basic_df = get_basic_df(data, flatten_mask)
        feat_df = get_df_feat(data, flatten_mask)

        model_dict = {}
        for di, date in enumerate(dates):
            if di <= fixed_di: continue
            if rebalance_dates_mask[di]:
                print(f'⏩⏩ Fitting on: {dates[di]}')

                selected_idx_di = np.where(filter_matrix[:, di - delay] == True)[0]
                aid = alpha_list[selected_idx_di][0]

                train_enddate = dates[di]
                if di - lookback < 0:
                    train_startdate = dates[0]
                else:
                    train_startdate = dates[di - lookback]

                basic_df_IS = basic_df[(basic_df['dates'] <= train_enddate) & (basic_df['dates'] >= train_startdate)]
                feat_df_IS = feat_df[(basic_df['dates'] <= train_enddate) & (basic_df['dates'] >= train_startdate)]

                feat_X = op.at_nan2zero(feat_df_IS.to_numpy())
                Y = op.at_nan2zero(basic_df_IS['target'].to_numpy())

                model = fit_model(feat_X, Y)

                m_dict = {'alpha': aid, 'model': model}
                model_dict[date] = pickle.dumps(m_dict)

        alpha_attribution_weights = np.copy(filter_matrix)

        return model_dict, alpha_attribution_weights

    def construct_preA(self, data, model_dict, mode):
        print(f'contruct_preA mode: {mode}')
        for k in model_dict:
            model_dict[k] = pickle.loads(model_dict[k])

        numdates = data['numdates']
        numstocks = data['numstocks']
        delay = data['delay']
        dates = data['dates']
        region = data['region']

        flatten_mask = get_earnings_mask(data)
        basic_df = get_basic_df(data, flatten_mask)
        feat_df = get_df_feat(data, flatten_mask)

        if mode == 'last':
            preA_ld = np.zeros(shape=numstocks, dtype=np.float32, order='F')

            date = dates[-1]

            basic_df_OS = basic_df[(basic_df['dates'] == date)]
            feat_df_OS = feat_df[(basic_df['dates'] == date)]
            feat_X = feat_df_OS.to_numpy()

            fitted_model_dict = (list(model_dict.values())[0])
            alpha_load = data['load_alpha'](fitted_model_dict['alpha'])
            alpha_load = op.at_nan2zero(alpha_load)

            si_, di_ = basic_df_OS['si'].values, basic_df_OS['di'].values
            if len(si_) == 0: return preA_ld

            model = fitted_model_dict['model']
            dfeat_X = xgb.DMatrix(feat_X)
            y_pred = model.predict(dfeat_X)

            preA_ld[si_] = y_pred

            return preA_ld

        if mode == 'full':
            refit_dates = np.array(sorted(model_dict.keys()))
            preA = np.zeros(shape=(numstocks, numdates), dtype=np.float32, order='F')

            for di, date in enumerate(dates):
                if date in model_dict:
                    print('refit day = ', date)

                    refit_idx = np.where(refit_dates == date)[0][0]
                    if refit_idx == refit_dates.size - 1:
                        idx_start, idx_end = di, numdates
                    else:
                        idx_start, idx_end = di, np.where(dates == refit_dates[refit_idx + 1])[0][0]

                    test_enddate = dates[idx_end - 1]
                    test_startdate = dates[idx_start]

                    basic_df_OS = basic_df[(basic_df['dates'] <= test_enddate) & (basic_df['dates'] >= test_startdate)]
                    feat_df_OS = feat_df[(basic_df['dates'] <= test_enddate) & (basic_df['dates'] >= test_startdate)]
                    feat_X = feat_df_OS.to_numpy()

                    si_, di_ = basic_df_OS['si'].values, basic_df_OS['di'].values

                    model = model_dict[date]['model']
                    dfeat_X = xgb.DMatrix(feat_X)
                    y_pred = model.predict(dfeat_X)

                    alpha_load = data['load_alpha'](model_dict[date]['alpha'])
                    alpha_load = op.at_nan2zero(alpha_load)

                    preA[si_, di_] = y_pred

            return preA
        else:
            raise Exception('Unrecognized mode: %s' % mode)
