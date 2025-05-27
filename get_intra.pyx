

def get_intra900002_1(begDate, endDate){
    upret_ret_1d = SELECT date, minTime, instrument,
    CASE 
        WHEN percentChange(close, 5) > 0 THEN percentChange(close, 5)
        ELSE 0
    END AS upret_ret,
    percentChange(close, 5) AS ret_ret
    FROM loadTable('dfs://yj_bar_equity', 'YJ_Minute1') //数据库表默认 date,instrument,minTime有序，实际使用时需要确认一下
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    //instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] // 测试的时候选少一些股票，验证因子逻辑有效性和debug
    context by date, instrument
    
    factor_1min_df = select date, minTime, instrument, -mstd(upret_ret, 20) as upret_ret_std,
    -mstd(ret_ret, 20) as ret_std
    from upret_ret_1d
    
    
    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000] // 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}




def OP_wensheng_upretpricemovstd_001(close, winspan=5, winstd=15, winpct=5){
    // emw_upret_std
    ret1m_ma5 = ewmMean(percentChange(close, winpct), span=winspan) 
    upret = iif(percentChange(close, winpct)>0, ret1m_ma5, NULL) 
    return -mstd(upret, winstd, int(winstd/3))
}

def upret_pricemov_std(close){
    ret1m_ma5 = ewmMean(percentChange(close, 5), span=5) //  ewmMean(percentChange(close, 5), span=5, minPeriods=3)  # 
    upret = iif(percentChange(close, 5)>0, ret1m_ma5, NULL) 
    return -mstd(upret, 15, 5)
}


def get_intra900004(begDate, endDate){

    upret_ret_1d = SELECT date, minTime, instrument, upret_pricemov_std(close) as upret5_std_30
    FROM loadTable('dfs://yj_bar_equity', 'YJ_Minute1') //数据库表默认 date,instrument,minTime有序，实际使用时需要确认一下
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] // 测试的时候选少一些股票，验证因子逻辑有效性和debug
    context by date, instrument
    
    factor_sample_df = select * 
    from upret_ret_1d
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000] // 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}





def get_intra900005_1(begDate, endDate){
    
    mrankv_p_2 = SELECT date, minTime, instrument, mrank(volume, true, 14) as mrankv, prev(mrank(close, true, 14)) as mrankp
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') //数据库表默认 date,instrument,minTime有序，实际使用时需要确认一下
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument // 每天每个股票，日内进行rolling计算因子，使用context by开窗的时候一定要确认没有用到未来数据
    

    
    dpv_rank_corr = SELECT date, minTime, instrument, -mcorr(mrankv, mrankp, 15, 5) as dpv_rank_corrs
    FROM mrankv_p_2 context by date, instrument

    factor_sample_df = select * 
    from dpv_rank_corr
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000] // 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}




def get_intra900007_1(begDate, endDate){
    smartnTop_vol = select date, minTime, instrument, -mavgTopN(percentChange(close, 5), (turnover), 20, int(0.3*20), false) as smart_vol_bm
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument


    factor_sample_df = select * 
    from smartnTop_vol
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}




def OP_wensheng_smartnToppct_002(close_1d, vol_1d, WIN1 = 20, wind = 5){
    vol_5d = mavg(vol_1d, wind, 3)  
    pct_int = mavg(log(percentChange(close_1d, wind)+1), wind, 3)
    smart_vol_pm = mavgTopN(power(pct_int, 2), prev(vol_5d), WIN1, int(0.3*WIN1), false) / mstd(pct_int, WIN1, int(0.5*WIN1))
    return smart_vol_pm
}

def smartnToppct(close_1d, vol_1d){
    WIN1 = 20
    vol_5d = mavg(vol_1d, 5, 3)  
    pct_int = mavg(log(percentChange(close_1d, 5)+1), 5, 3)
    smart_vol_pm = mavgTopN(power(pct_int, 2), prev(vol_5d), WIN1, int(0.3*WIN1), false) / mstd(pct_int, WIN1)
    return smart_vol_pm
}


def get_intra900007_2(begDate, endDate){
    smartnTop_vol = select date, minTime, instrument, -smartnToppct(close, turnover) as smart_vol_p
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument
    smartnTop_vol1 = update!(smartnTop_vol, 'smart_vol_p', null, 0..14)

    factor_sample_df = select * 
    from smartnTop_vol
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}





def Talyer(close_1d, vol_1d){
    WIN1 = 15
    vol_5d = mavg(vol_1d, 5, 3)  // tvr
    pct_int = mavg(log(percentChange(close_1d, 5)+1), 5, 3)
    talyer = exp(pct_int) - 1 - pct_int - 0.5 * power(pct_int, 2)
    return mavg(talyer, WIN1)  // mavgTopN(talyer, talyer, WIN1, int(0.3*WIN1), false)
}

def get_intra900008(begDate, endDate){
    Talyer_pct = select date, minTime, instrument, -Talyer(close, turnover) as talyer_mean
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument
    Talyer_pct1 = update!(Talyer_pct, 'talyer_mean', null, 0..14)

    factor_sample_df = select * 
    from Talyer_pct1
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}







def asy_lws(ret){
    zscores = (ret - mavg(ret, 15, 10)) / mstd(ret, 15, 10)
    return msum(float(zscores>0.8) - float(zscores<-0.8), 15, 15)
}


def get_intra900009(begDate, endDate){
    Talyer_pct = select date, minTime, instrument, -asy_lws(percentChange(close, 5)) as asy_cps
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument

    factor_sample_df = select * 
    from Talyer_pct
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}





def asy_lws_vol(ret, vol_1d){
    zscores = (ret - mavg(ret, 15, 10)) / mstd(ret, 15, 10)
    return msum(iif(zscores>0.8, prev(vol_1d), 0.0), 15, 15) / msum(vol_1d, 15, 15)
}


def get_intra900009_1(begDate, endDate){
    Talyer_pct = select date, minTime, instrument, -asy_lws_vol(percentChange(close, 5), volume) as asy_cps
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument

    factor_sample_df = select * 
    from Talyer_pct
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}












defg cvx_lws(vol){
    // vol = mavg(vol_1d, 5, 3)  // tvr
    cumsum_vol = vol //cumsum(vol)
    se_mean = avg(cumsum_vol)
    cumbffill = bfill(ffill(cumsum_vol))
    av_mean = (first(cumbffill) + last(cumbffill)) / 2.0
    return (av_mean - se_mean) / se_mean
}


def get_intra900010(begDate, endDate){

    cvx_pct = select date, minTime, instrument, -moving(cvx_lws, close, 20, 20) as cvx_vol
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument
    
    factor_sample_df = select * 
    from cvx_pct
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}



def get_intra900011(begDate, endDate){

    cvx_pct = select date, minTime, instrument, mstd(mstd(percentChange(close, 5), 5, 5), 15, 15) as std_std_pct
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument
    
    factor_sample_df = select * 
    from cvx_pct
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}










