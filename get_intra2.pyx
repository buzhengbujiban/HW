
def get_intra900012(begDate, endDate){
    ambiguou_df = select date, minTime, instrument, mstd(mstd(percentChange(close, 5), 5, 5), 5, 5) as ambiguous, close
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument
    
    order_money_df = select date, minTime, instrument, order_money
    from loadTable('dfs://yj_bar_equity', 'min_order_price_volume_stats') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument
    
    joined_df = SELECT date, minTime, instrument, order_money, ambiguous, close
    FROM order_money_df
    JOIN ambiguou_df
    ON order_money_df.date=ambiguou_df.date and order_money_df.minTime=ambiguou_df.minTime and order_money_df.instrument=ambiguou_df.instrument order by date, instrument
    
    corr_ambi_money_df = SELECT date, minTime, instrument, mavgTopN(ambiguous, order_money, 15, 7, false) as corr_ambi_money
    from joined_df context by date, instrument

    factor_sample_df = select * 
    from corr_ambi_money_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}







def get_intra900013(begDate, endDate){
    factor_1min_df = select date, minTime, instrument, -mcorr(close, float(1..size(close)), 20, 20) as cro 
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument 

    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}





def get_intra900014(begDate, endDate){
    factor_1min_df = select date, minTime, instrument, mcorr(cumsum(volume), float(1..size(volume)), 45,45) as cro 
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument 

    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}



def linspaces(start, ends, num){
    step = float(ends - start) / float(num - 1)
    return float(0..(num-1)) * step + start
}


defg vss(prc){
    prct = cumsum(prc)
    firstone_seq = bfill(ffill(prct))
    firstone_seq = firstone_seq - first(firstone_seq)
    seq = linspaces(0, last(firstone_seq), size(prct))
    diff_area = firstone_seq - seq
    uparea = sum(iif(diff_area>0, diff_area, 0.0))
    downarea = sum(iif(diff_area<0, diff_area, 0.0))
    return (uparea / (uparea + downarea))
}







def get_intra900016(begDate, endDate){
    factor_1min_df = select date, minTime, instrument, mmax(mmin(low, 5, 5), 15) as lowuppos_scale 
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument 

    factor_1min_df2 = select date, minTime, instrument,  -mstd(lowuppos_scale, 10)  as zscore_lowuppos
    from factor_1min_df context by date, instrument 
    factor_sample_df = select * 
    from factor_1min_df2
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}
// (lowuppos_scale - mavg(lowuppos_scale, 10)) /






def get_intra900017(begDate, endDate){
    factor_1min_df = select date, minTime, instrument, 
    float(mavg(active_money_buy - active_money_sell, 5, 5)) / float( mstd(active_money_buy - active_money_sell, 5, 5)) as active_net_buy_tense5, 
    float(mavg(active_money_buy - active_money_sell, 15, 15)) / float(mstd(active_money_buy - active_money_sell, 15, 15)) as active_net_buy_tense15
    from loadTable('dfs://yj_bar_equity', 'min_trade_price_volume_stats') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument 


    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}










def get_intra900017_1(begDate, endDate){

    factor_1min_df = select date, minTime, instrument,  active_money_buy, active_money_sell
    from loadTable('dfs://yj_bar_equity', 'min_trade_price_volume_stats') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument 
    
    dret_df = select date, minTime, instrument, prev(percentChange(close, 5)) as dret, volume
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument 
    
    
    
    corr_dret5_netactmflb = SELECT date, minTime, instrument, dret, active_money_buy, active_money_sell, volume
    FROM dret_df
    JOIN factor_1min_df
    ON dret_df.date=factor_1min_df.date and 
    dret_df.minTime=factor_1min_df.minTime and 
    dret_df.instrument=factor_1min_df.instrument order by date, instrument
    
    
    
    corr_dret5_netactmflb = SELECT date, minTime, instrument, mavgTopN(dret, active_money_buy + active_money_sell, 20, 5, false) as dpactmflb4
    from corr_dret5_netactmflb context by date, instrument
    

    factor_sample_df = select * 
    from corr_dret5_netactmflb
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}













def get_intra900017_2(begDate, endDate){


    dret_df = select date, minTime, instrument, mavgTopN(prev(percentChange(close, 5)), volume, 20, 5, false) as dret_nTop
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument 
    

    factor_sample_df = select * 
    from dret_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}











def soft_ts_fill(ret, vol){
    thed = mpercentile(ret, 80, 20)
    return ema(iif(ret>thed, vol, 0), 15)
}





def get_intra900018(begDate, endDate){ 


    factor_1min_df = select date, minTime, instrument,  active_money_buy, active_money_sell, money_buy, great_numbers_buy, great_numbers_sell
    from loadTable('dfs://yj_bar_equity', 'min_trade_price_volume_stats') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument 
    
    dret_df = select date, minTime, instrument, close, volume
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument 
    
    
    
    corr_dret5_netactmflb = SELECT date, minTime, instrument, active_money_buy, active_money_sell, volume, close, money_buy, great_numbers_buy, great_numbers_sell
    FROM dret_df
    JOIN factor_1min_df
    ON dret_df.date=factor_1min_df.date and 
    dret_df.minTime=factor_1min_df.minTime and 
    dret_df.instrument=factor_1min_df.instrument order by date, instrument
    


    dret_df = select date, minTime, instrument, soft_ts_fill(active_money_buy, percentChange(close, 3))  as soft_ts_fill_lpl, soft_ts_fill(active_money_buy / money_buy, (active_money_buy - active_money_sell) / money_buy)  as soft_ts_fill_lambs1, mavgTopN(percentChange(close, 3), mstd(money_buy, 20, 10), 15, 15)  as bm_lpl, soft_ts_fill(mstd(percentChange(close, 3), 5), percentChange(close, 3))  as soft_ts_fill_lambs4, soft_ts_fill(prev(large_numbers_buy + great_numbers_buy), large_numbers_buy + great_numbers_buy - large_numbers_sell - great_numbers_sell)  as soft_ts_fill_lgreat
    from corr_dret5_netactmflb context by date, instrument 


    factor_sample_df = select * 
    from dret_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}





defg semi_waves(vol, prc){
    imaxx = imax(prev(vol))
    return (prc[imaxx] - last(prc)) / prc[imaxx] 
}







def get_intra900019(begDate, endDate){ 

    updated_YJ_minute = select date, minTime, instrument, percentChange(close, 3) as rev3, percentChange(close, 5) as rev5, mavg(percentChange(close, 3), 20) as mavg_rev3, mavg(percentChange(close, 3), 20) \ mstd(percentChange(close, 3), 20) as stable_rev3avg20, (percentChange(close, 3) - mavg(percentChange(close, 3), 20)) \ mstd(percentChange(close, 3), 20)  as rev3_norm
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by instrument 
    
    
    factor_sample_df = select * 
    from updated_YJ_minute
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}






def get_intra900020(begDate, endDate){ 


    updated_YJ_minute = select date, minTime, instrument, (mmax(high, 5) - mmin(low, 5)) / prev(mavg(close, 5)) as vowel
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument  
    
    
    
    
    
    
    OneminDF = select date, minTime, instrument, mavgTopN(vowel, exp(vowel) - 1 - vowel - 0.5 * power(vowel, 2), 20, 5, false) as taTopN, vowel
    from updated_YJ_minute
    context by date, instrument  
        
    factor_sample_df = select * 
    from OneminDF
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}



defg rvi(close, open, high, low){
    return ( last(close) - first(open) ) /  (max(high) - min(low))
}

def get_intra900021(begDate, endDate){ 


    updated_YJ_minute = select date, minTime, instrument, -moving(rvi, (close, open, high, low), 15, 15)  as rvi
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument  
    
    factor_sample_df = select * 
    from updated_YJ_minute
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}







def ATR(high, low, close_1){
    ATR1 = ( (high-low) / close_1)
    ATR2 = (abs(high-close_1)/close_1)
    ATR3 = (abs(low-close_1)/close_1)
    risks1 = iif(ATR1>ATR2,ATR1,ATR2)
    risks = iif(ATR3>risks1, ATR3, risks1)
    return risks
}


defg prevs(seq){
    return first(seq)
}  


def get_intra900022(begDate, endDate){ 


    updated_YJ_minute = select date, minTime, instrument, percentChange(close, 5) as rev, ATR(mmax(high, 5), mmin(low, 5), moving(prevs, close, 5, 5)) as risks
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument  
    
    
    updated_YJ_minute1 = select date, minTime, instrument, -float(rev) / float(risks) as rev_risks, -risks
    from updated_YJ_minute
    context by date, instrument  
    
    factor_sample_df = select * 
    from updated_YJ_minute1
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}



def get_intra900022_fd(begDate, endDate){ 


    updated_YJ_minute = select date, minTime, instrument, percentChange(close, 5) as rev, ATR(mmax(high, 5), mmin(low, 5), moving(prevs, close, 5, 5)) as risks
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between 2024.06.03: 2024.06.05, minTime between 09:15:00.000: 13:30:01.000, 
    // instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument  
    
    
    updated_YJ_minute1 = select date, minTime, instrument, -float(rev) / float(risks) as rev_risks, -risks
    from updated_YJ_minute
    context by date, instrument  
    
    factor_sample_df = select * 
    from updated_YJ_minute1
    where minTime in [13:30:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}



def avg_out_order2(ret, money, win=20){

    money_down = msum(iif(ret<0.0, money, 0.0), win, win)
    money_up = msum(iif(ret>0.0, money, 0.0), win, win)
    
    denominator = (money_down - money_up) / msum(money, win, win)
    return denominator
}



def avg_out_order3(money, win=20){

    denominator = -mstd(money, win, win)  / msum(money, win, win)
    return denominator
}



 // order_money_buy - order_money_sell
def get_intra900023(begDate, endDate){ 

    updated_YJ_minute = select date, minTime, instrument, avg_out_order2(percentChange(order_close, 5), order_money, 20) as avg_out_orders
    from loadTable('dfs://yj_bar_equity', 'min_order_price_volume_stats') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument  
    
        
    factor_sample_df = select * 
    from updated_YJ_minute
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df /// 
}



def get_intra900023_2(begDate, endDate){ 

    updated_YJ_minute = select date, minTime, instrument, avg_out_order3(order_money, 25) as avg_out_orders
    from loadTable('dfs://yj_bar_equity', 'min_order_price_volume_stats') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument  
    
        
    factor_sample_df = select * 
    from updated_YJ_minute
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df 
}







//  float(order_volume_buy - order_volume_sell) / float(order_volume_buy + order_volume_sell) * 

def get_intra900024(begDate, endDate){ 

    factor_1min_df = select date, minTime, instrument,  mavg(float(order_volume_buy - order_volume_sell), 5, 5) / mavg(float(order_volume), 5, 5) as imbalance, mavg( float(order_close_sell -  order_close_buy), 15, 15) as spread, order_money 
    from loadTable('dfs://yj_bar_equity', 'min_order_price_volume_stats') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument 



    factor_1min_df2 = select date, minTime, instrument, imbalance, -mavgTopN(imbalance, order_money, 20, 5, false) as topN_vol_imbalance
    from factor_1min_df
    context by date, instrument 
    
    factor_sample_df = select * 
    from factor_1min_df2
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df /// 
}







defg peak(vol){
    vol_zsc = (vol - avg(vol)) / std(vol)
    return sum((vol_zsc>0.8) * vol_zsc) 

}

def asy_mM(vol){
    vol2 = vol
    //vol2 = iif(vol>mavg(vol, 10), vol, NULL)
    return (mavg(vol2, 20) - mmed(vol2, 20)) / mavg(vol2, 20)
}









def get_intra900026(begDate, endDate){ 


    
    factor_1min_df = select date, minTime, instrument, asy_mM(volume) as asy_mMvol, moving(peak, percentChange(close, 5), 20, 20) as vol_up_peaknum
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1')  // order_money_buy - order_money_sell
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument 
    
    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df 

}






def rsi_up(seq_raw, win=15){
    seq = float(nullFill(seq_raw, 0))
    return ema(iif(seq>0.0, seq, 0.0), win) / ema(abs(seq), win)
}




def get_intra900027(begDate, endDate){ 


        
    factor_1min_df = select date, minTime, instrument, -ema((nullFill(large_money_buy, 0) + nullFill(great_money_buy, 0) - nullFill(large_money_sell, 0) - nullFill(great_money_sell, 0)) / (money_buy), 15) as vol_totbid10_tot10, mcorr(ema((nullFill(large_money_buy, 0) + nullFill(great_money_buy, 0) - nullFill(large_money_sell, 0) - nullFill(great_money_sell, 0)) / (money_buy), 5), money_buy, 25, 20) as coremaorimb,  -rsi_up((  nullFill(large_money_buy, 0) + nullFill(median_money_buy, 0) + nullFill(great_money_buy, 0) - nullFill(large_money_sell, 0) - nullFill(median_money_sell, 0) - nullFill(great_money_sell, 0) ) / (money_buy), 20) as rsi_large_order_inflow
    from loadTable('dfs://yj_bar_equity', 'min_trade_price_volume_stats')  
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument 
    
    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df 

}


def rsi_u(seq_raw, win=15){
    seq = float(nullFill(seq_raw, 0))
    return msum(iif(seq>0.0, seq, 0.0), win) / msum(abs(seq), win)
}




defg path_rati(seqraw){
    seq = float(seqraw)
    path = sum(abs(eachPre(-, seq)))
    cumbffill = bfill(ffill(seq))
    return abs(last(cumbffill) - first(cumbffill)) / path
}


def path_ratio(seqraw, win=20){
    seqfilled = (ffill(float(seqraw)))
    return abs(seqfilled - mfirst(seqfilled, win)) / msum(abs(deltas(seqfilled)), win, win)
}



def get_intra900029(begDate, endDate){ 

    factor_1min_df = select date, minTime, instrument, -path_ratio(close, 20) as pathratio, path_ratio(close, 20) * percentChange(close, 5) as path2ratio, percentChange(close, 5) as rev, -path_ratio(cumsum(volume), 20) as pathratiocumvol
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1')  
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument  
    
    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df 

}






def get_intra900030(begDate, endDate){ 

    factor_1min_df = select date, minTime, instrument, mcorr(order_close, order_money, 20) as mcorbm, mcorr(prev(order_low_sell), order_money_buy, 20) as order_mone_imbalance2
    from loadTable('dfs://yj_bar_equity', 'min_order_price_volume_stats')  
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument  
    
    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df 

}











def get_intra900031(begDate, endDate){ 

    factor_1min_df = select date, minTime, instrument, linearTimeTrend(close, 20)[1] as slope_close
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1')  
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
     // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument
    
    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df 

}









def get_intra900031_1(begDate, endDate){ 

    factor_1min_df = select date, minTime, instrument, mstd((order_low_buy * order_money_buy + order_high_sell * order_money_sell) /(order_money_buy + order_money_sell) - order_close, 20, 20 ) as slope_all_order
    from loadTable('dfs://yj_bar_equity', 'min_order_price_volume_stats')  
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
     // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument
    
    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df 

}







