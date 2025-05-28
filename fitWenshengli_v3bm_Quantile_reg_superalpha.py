import numpy as np
import operators as op
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import pickle
import time
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler


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

    def groupSum(data, group):
        gMean = np.full(data.shape, np.nan, order="F", dtype=np.float32)
        for di in range(gMean.shape[1]):
            df = pd.DataFrame({"data": data[:, di], "group": group[:, di]})
            gMean[:, di] = df.groupby("group")["data"].transform("sum").values
        return gMean

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


    ########## final df #############
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0) #here we fill the missing values with zeros, you may change this if required

    return df


############################################## MODEL CHANGES - make your changes in this function ###############################################
def fit_model_combo(feat_X, Y):


    X_train = feat_X
    Y_train = Y
    gbr75 = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=1, random_state=42,
                                      loss='quantile', alpha=0.75)
    gbr75.fit(X_train, Y_train)

    xgbr05 = XGBRegressor(n_estimators=400, learning_rate=0.1, max_depth=1, random_state=42, tree_method='exact')
    xgbr05.fit(X_train, Y_train)

    gbr25 = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=1, random_state=42,
                                      loss='quantile', alpha=0.25)
    gbr25.fit(X_train, Y_train)
    YPRED_train_dist = gbr75.predict(X_train)
    YPRED_train_dist2 = xgbr05.predict(X_train)
    YPRED_train_distc = gbr25.predict(X_train)
    X_train_new = np.column_stack((
        X_train,
        YPRED_train_dist,
        YPRED_train_dist2,
        YPRED_train_distc,
    ))


    xgbr = XGBRegressor(
        n_estimators=450,
        learning_rate=0.1, max_depth=1, random_state=42,
        tree_method='exact',
    )
    xgbr.fit(X_train_new, Y_train)

    return gbr75, xgbr05, gbr25, xgbr
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
        fixed_di = fixed_di = np.where(dates>=20131210)[0][0]

        flatten_mask = get_earnings_mask(data)
        basic_df = get_basic_df(data, flatten_mask)
        feat_df = get_df_feat(data, flatten_mask)

        model_dict = {}
        for di, date in enumerate(dates):
            if di <= fixed_di: continue
            if rebalance_dates_mask[di]:
                print(f'⏩⏩ Fitting on: {dates[di]}')
                
                selected_idx_di = np.where(filter_matrix[:, di - delay] == True)[0]
                aid=alpha_list[selected_idx_di][0]

                train_enddate = dates[di]
                if di-lookback<0:
                    train_startdate = dates[0]
                else:    
                    train_startdate = dates[di-lookback]

                basic_df_IS = basic_df[(basic_df['dates']<=train_enddate)&(basic_df['dates']>=train_startdate)]
                feat_df_IS = feat_df[(basic_df['dates']<=train_enddate)&(basic_df['dates']>=train_startdate)]

                feat_X = op.at_nan2zero(feat_df_IS.to_numpy())
                Y = op.at_nan2zero(basic_df_IS['target'].to_numpy())

                gbr75, xgbr05, gbr25, xgbr  = fit_model_combo(feat_X, Y)

                m_dict = {'alpha': aid, 'model1': gbr75, 'model2': xgbr05, 'model3': gbr25, 'model4':xgbr}
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
            
            basic_df_OS = basic_df[(basic_df['dates']==date)]
            feat_df_OS = feat_df[(basic_df['dates']==date)]
            feat_X = feat_df_OS.to_numpy()
            
            fitted_model_dict = (list(model_dict.values())[0])
            alpha_load = data['load_alpha'](fitted_model_dict['alpha'])
            alpha_load = op.at_nan2zero(alpha_load)

            si_, di_ = basic_df_OS['si'].values, basic_df_OS['di'].values
            if len(si_)==0: return preA_ld

            model1 = fitted_model_dict['model1']
            model2 = fitted_model_dict['model2']
            model3 = fitted_model_dict['model3']
            model4 = fitted_model_dict['model4']

            y1_pred = model1.predict(feat_X)
            y2_pred = model2.predict(feat_X)
            y3_pred = model3.predict(feat_X)

            feat_X_new = np.column_stack((
                feat_X,
                y1_pred,
                y2_pred,
                y3_pred,
            ))

            y4_pred = model4.predict(feat_X_new)


            y_pred = (op.at_nan2zero(y4_pred))
            
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
                    
                    test_enddate = dates[idx_end-1]
                    test_startdate = dates[idx_start]

                    basic_df_OS = basic_df[(basic_df['dates']<=test_enddate)&(basic_df['dates']>=test_startdate)]
                    feat_df_OS = feat_df[(basic_df['dates']<=test_enddate)&(basic_df['dates']>=test_startdate)]
                    feat_X = feat_df_OS.to_numpy()

                    si_, di_ = basic_df_OS['si'].values, basic_df_OS['di'].values


                    model1 = model_dict[date]['model1']
                    model2 = model_dict[date]['model2']
                    model3 = model_dict[date]['model3']
                    model4 = model_dict[date]['model4']


                    y1_pred = model1.predict(feat_X)
                    y2_pred = model2.predict(feat_X)
                    y3_pred = model3.predict(feat_X)

                    feat_X_new = np.column_stack((
                        feat_X,
                        y1_pred,
                        y2_pred,
                        y3_pred,
                    ))

                    y4_pred = model4.predict(feat_X_new)

                    y_pred = (op.at_nan2zero(y4_pred))

                    alpha_load = data['load_alpha'](model_dict[date]['alpha'])
                    alpha_load = op.at_nan2zero(alpha_load)

                    preA[si_, di_] = y_pred

            return preA
        else:
            raise Exception('Unrecognized mode: %s' % mode)
