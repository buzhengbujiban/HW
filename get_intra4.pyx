
def get_intra900000(begDate, endDate){ 

    factor_1min_df = select date, minTime, instrument,  active_money_buy, active_money_sell, money_buy, money_sell
    from loadTable('dfs://yj_bar_equity', 'min_trade_price_volume_stats') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument 
    
    dret_df = select date, minTime, instrument, close, volume, turnover
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725']
    context by date, instrument 
    
    
    corr_dret5_netactmflb = SELECT date, minTime, instrument, volume, close, turnover, active_money_buy, active_money_sell, money_buy, money_sell
    FROM dret_df
    JOIN factor_1min_df
    ON dret_df.date=factor_1min_df.date and 
    dret_df.minTime=factor_1min_df.minTime and 
    dret_df.instrument=factor_1min_df.instrument order by date, instrument

    
    factor_barra_df = select date, minTime, instrument, 
    percentChange(close, 3) as rev,
    ratio(msumTopN(turnover, turnover, 30, 3), msum(turnover, 30)) as m30_top01_money_pct, 
    ratio(msumTopN(percentChange(close), turnover, 30, 3), msum(percentChange(close), 30)) as m30_top01_ret_pct_rank_by_money,
    mimax(pow(iif(ratios(close) - 1 < 0, mstd(ratios(close) - 1, 20),close), 2.0), 5) as alpha001,
    mavgTopN(percentChange(close, 3), mstd(money_buy, 20, 10), 15, 15)  as bm_lpl
    from corr_dret5_netactmflb context by date, instrument 


    
    factor_sample_df = select * 
    from factor_barra_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df 
}



defg runsup01(seqraw){
    seq = iif(eachPre(-, seqraw)>=0.0, 1.0, 0.0)
    currentLength = 0
    maxlength = 0
    for(num in seq){
        if(num == 1.0)
            currentLength += 1
        else
            currentLength = 0
        if(maxlength < currentLength)
            maxlength = currentLength
    }
    return maxlength
}






defg runsup01_vowel(seqraw){
    seq = iif(eachPre(-, seqraw)>=0.0, 1.0, 0.0)
    currentLength = 0
    maxlength = 0
    for(num in seq){
        if(num == 1.0)
            currentLength += 1.0
        else
            currentLength = 0.0
        if(maxlength < currentLength)
            maxlength = currentLength
    }
    return maxlength
}






def get_intra900033(begDate, endDate){
    factor_1min_df = select date, minTime, instrument,
    moving(runsup01, close, 25, 15) as runs01_close1, 
    moving(runsup01, mavg(close, 5, 5), 25, 15) as runs01_close5
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1')
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] // 测试的时候选少一些股票，验证因子逻辑有效性和debug
    context by date, instrument 

    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}



def dead(seq){
    seq_01 = iif(eachPre(-, seq)<0.0, 1.0, 0.0)
    firstqt = iif(eachPre(-, seq_01)==1.0, cumsum(seq_01), NULL)
    upseq = iif(seq_01==1.0, cumsum(seq_01) - ffill(firstqt)+1, 0.0)
    return ema(mmax(upseq, 5) * seq, 20)  \ mavg(abs(seq), 20)
}


def dead2(seq){
    seq_01 = iif(eachPre(-, seq)<0.0, 1.0, 0.0)
    firstqt = iif(eachPre(-, seq_01)==1.0, cumsum(seq_01), NULL)
    upseq = iif(seq_01==1.0, cumsum(seq_01) - ffill(firstqt)+1, 0.0)
    return ema(mmax(upseq, 5) * eachPre(-, seq, 5) \ abs(mfirst(seq, 5)), 20) // \ mavg(abs(seq), 20)
}


def deadbm(seq){
    seq_01 = iif(eachPre(-, seq)<0.0, 1.0, 0.0)
    firstqt = iif(eachPre(-, seq_01)==1.0, cumsum(seq_01), NULL)
    upseq = iif(seq_01==1.0, cumsum(seq_01) - ffill(firstqt)+1, 0.0)
    return ema(mmax(upseq, 5), 20)
}



def dead4(seq){
    seq_01 = iif(eachPre(-, seq)<0.0, 1.0, 0.0)
    firstqt = iif(eachPre(-, seq_01)==1.0, cumsum(seq_01), NULL)
    upseq = iif(seq_01==1.0, cumsum(seq_01) - ffill(firstqt)+1, 0.0)
    return mavgTopN(mmax(upseq, 5) * eachPre(-, seq, 5) \ abs(mfirst(seq, 5)), upseq, 20, 7, false)
}






def largeorder_emaabsbm(seq){
    return ema(seq, 20)  \ mavg(abs(seq), 20)
}

def upsemaall(seq){
    return ema(iif(seq>0.0, seq, NULL), 20)  \ mavg(abs(seq), 20)
}

def downsemaall(seq){
    return ema(iif(seq>0.0, seq, NULL), 20) - ema(iif(seq<0.0, seq, NULL), 20)  \ mavg(abs(seq), 20)
}



def get_intra900034(begDate, endDate){
    factor_1min_df = select date, minTime, instrument,
    largeorder_emaabsbm(cumsum(nullFill(large_money_buy, 0) + nullFill(great_money_buy, 0) - nullFill(large_money_sell, 0) - nullFill(great_money_sell, 0))) as largeorder_emaabsbm,
    largeorder_emaabsbm(cumsum(mavg(nullFill(large_money_buy, 0) + nullFill(great_money_buy, 0) - nullFill(large_money_sell, 0) - nullFill(great_money_sell, 0), 5, 5))) as largeorder_emaabbm5, 
    upsemaall((nullFill(large_money_buy, 0) + nullFill(great_money_buy, 0) - nullFill(large_money_sell, 0) - nullFill(great_money_sell, 0))) as upsemaall1,
    downsemaall((nullFill(large_money_buy, 0) + nullFill(great_money_buy, 0) - nullFill(large_money_sell, 0) - nullFill(great_money_sell, 0))) as downsemaall1,
    largeorder_emaabsbm((nullFill(large_money_buy, 0) + nullFill(great_money_buy, 0) - nullFill(large_money_sell, 0) - nullFill(great_money_sell, 0))) as semaall1,
    

    dead(cumsum(nullFill(large_money_buy, 0) + nullFill(great_money_buy, 0) - nullFill(large_money_sell, 0) - nullFill(great_money_sell, 0))) as daedds,
    dead2(cumsum(nullFill(large_money_buy, 0) + nullFill(great_money_buy, 0) - nullFill(large_money_sell, 0) - nullFill(great_money_sell, 0))) as daedds2,
    deadbm(cumsum(nullFill(large_money_buy, 0) + nullFill(great_money_buy, 0) - nullFill(large_money_sell, 0) - nullFill(great_money_sell, 0))) as daeddsbm,
    dead4(cumsum(nullFill(large_money_buy, 0) + nullFill(great_money_buy, 0) - nullFill(large_money_sell, 0) - nullFill(great_money_sell, 0))) as daedds4

    from loadTable('dfs://yj_bar_equity', 'min_trade_price_volume_stats')
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] // 测试的时候选少一些股票，验证因子逻辑有效性和debug
    context by date, instrument 

    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}






















def shift_to_numdiff_ratio(ret5, win=20){
    return (msum(iif(ret5>0.0, abs(ret5), 0.0), win) \ msum(iif(ret5<=0.0, abs(ret5), 0.0), win)) - (msum(iif(ret5>0.0, 1.0, 0.0), win) \ msum(iif(ret5<=0., 1.0, 0.0), win))   
}







def get_intra900036(begDate, endDate){
    win = 20
    factor_1min_df = select date, minTime, instrument,
    shift_to_numdiff_ratio(percentChange(close, 3)) as shift_to_num_ratclose
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1')  
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] // 测试的时候选少一些股票，验证因子逻辑有效性和debug
    context by date, instrument 

    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df
}









def get_intra900037(begDate, endDate){
    factor_1min_df = select date, minTime, instrument,
    mavgTopN(turnover, percentChange(close, 5), 20, 5, false) - mavgTopN(turnover, percentChange(close, 5), 20, 5) as tvupdown_bm,
    mavgTopN(turnover/abs(percentChange(close, 5)), percentChange(close, 5), 20, 5, false) - mavgTopN(turnover/abs(percentChange(close, 5)), percentChange(close, 5), 20, 5) as tvupdown_bm2,
    mavgTopN(turnover*abs(percentChange(close, 5)), percentChange(close, 5), 20, 5, false) - mavgTopN(turnover*abs(percentChange(close, 5)), percentChange(close, 5), 20, 5) as tvupdown_bm3
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














// new
//inner
def wavgSOIR(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 10 9 8 7 6 5 4 3 2 1)
    return avg(imbalance)
}
def wstdSOIR(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 10 9 8 7 6 5 4 3 2 1)
    return stdp(imbalance)
}
def wskewSOIR(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 10 9 8 7 6 5 4 3 2 1)
    return skew(imbalance)
}
def wkuisSOIR(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 10 9 8 7 6 5 4 3 2 1)
    return kurtosis(imbalance)
}
def whighSOIR(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 10 9 8 7 6 5 4 3 2 1)
    return max(imbalance)
}
def wlowSOIR(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 10 9 8 7 6 5 4 3 2 1)
    return min(imbalance)
}
def wpathSOIR(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 10 9 8 7 6 5 4 3 2 1)
    return sum(imbalance) \ sum(abs(imbalance))
}

def wavgSOIR5(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 5 4 3 2 1)
    return avg(imbalance)
}
def wstdSOIR5(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 5 4 3 2 1)
    return stdp(imbalance)
}
def wskewSOIR5(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 5 4 3 2 1)
    return skew(imbalance)
}
def wkuisSOIR5(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 5 4 3 2 1)
    return kurtosis(imbalance)
}
def whighSOIR5(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 5 4 3 2 1)
    return max(imbalance)
}
def wlowSOIR5(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 5 4 3 2 1)
    return min(imbalance)
}
def wpathSOIR5(bidQty, askQty){
    imbalance = rowWavg((bidQty - askQty)\(bidQty + askQty), 5 4 3 2 1)
    return sum(imbalance) \ sum(abs(imbalance))
}

//inner end


def feat_dos_001(){
    begDate = 2024.01.01
    endDate = 2024.07.01    
    
    
    feature1m_SOIR_stats = 
    select wavgSOIR(fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4, bid_sz_5, bid_sz_6, bid_sz_7, bid_sz_8, bid_sz_9), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4, ask_sz_5, ask_sz_6, ask_sz_7, ask_sz_8, ask_sz_9)) as wavgSolr,
    wstdSOIR(fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4, bid_sz_5, bid_sz_6, bid_sz_7, bid_sz_8, bid_sz_9), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4, ask_sz_5, ask_sz_6, ask_sz_7, ask_sz_8, ask_sz_9)) as wstdSolr,
    wskewSOIR(fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4, bid_sz_5, bid_sz_6, bid_sz_7, bid_sz_8, bid_sz_9), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4, ask_sz_5, ask_sz_6, ask_sz_7, ask_sz_8, ask_sz_9)) as wskewSolr,
    wkuisSOIR(fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4, bid_sz_5, bid_sz_6, bid_sz_7, bid_sz_8, bid_sz_9), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4, ask_sz_5, ask_sz_6, ask_sz_7, ask_sz_8, ask_sz_9)) as wkuisSolr,
    whighSOIR(fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4, bid_sz_5, bid_sz_6, bid_sz_7, bid_sz_8, bid_sz_9), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4, ask_sz_5, ask_sz_6, ask_sz_7, ask_sz_8, ask_sz_9)) as whighSolr,
    wlowSOIR(fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4, bid_sz_5, bid_sz_6, bid_sz_7, bid_sz_8, bid_sz_9), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4, ask_sz_5, ask_sz_6, ask_sz_7, ask_sz_8, ask_sz_9)) as wlowSolr,
    
    
    
    wavgSOIR5(fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4)) as wavgSolr5,
    wstdSOIR5(fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4)) as wstdSolr5,
    wskewSOIR5(fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4)) as wskewSolr5,
    wkuisSOIR5(fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4)) as wkuisSolr5,
    whighSOIR5(fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4)) as whighSolr5,
    wlowSOIR5(fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4)) as wlowSolr5
    
    
    from loadTable('dfs://stock_rawdata_sample', 'QuoteL2')
    where settlement_date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6'),
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'],
    (exch_time_ms between 09:15:00.000: 11:30:00.000 )
    or (exch_time_ms between 13:00:00.000: 15:00:00.000)
    group by settlement_date as date,instrument,bar(exch_time_ms,1m) as minTime 
    
    
    
    share(table=streamTable(feature1m_SOIR_stats), sharedName=`feature1m_SOIR_stat_001)
    dropStreamTable(tableName='feature1m_SOIR_stat_001')
    
    
}

def get_intra900040(begDate, endDate){
    
    factor_1min_df = select date, minTime, instrument, 
    iif(mstd(prev(wavgSolr), 15, 15)>0.000001, (wavgSolr - mavg(prev(wavgSolr), 15, 15)) \ mstd(prev(wavgSolr), 15, 15), NULL) as ts_norm_wavgSolr, 
    mavgTopN((wavgSolr5), wstdSolr5, 20, 5) as topSolrstd5_avg
    from feature1m_SOIR_stat_001
    context by date, instrument
    
    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df

}








def feat_dos_002(){
    begDate = 2024.01.01
    endDate = 2024.07.01    
    
    feature1m_MPC_stats = 
    select last((ask_px_0 * ask_sz_0 + bid_px_0 * bid_sz_0) \ (ask_sz_0 + bid_sz_0)) \ (std(ask_px_0 + bid_px_0) + 1.0) as prs_stable, std((ask_px_0 * ask_sz_0 + bid_px_0 * bid_sz_0) \ (ask_sz_0 + bid_sz_0)) as prs_std, last((ask_px_0 * ask_sz_0 + bid_px_0 * bid_sz_0) \ (ask_sz_0 + bid_sz_0)) as closeprs
    from loadTable('dfs://stock_rawdata_sample', 'QuoteL2')
    where settlement_date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6'),
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'],
    (exch_time_ms between 09:15:00.000: 11:30:00.000 )
    or (exch_time_ms between 13:00:00.000: 15:00:00.000)
    group by settlement_date as date,instrument,bar(exch_time_ms,1m) as minTime 
    
        
    
    share(table=streamTable(feature1m_MPC_stats), sharedName=`feature1m_MPC_stats_002)
    dropStreamTable(tableName='feature1m_MPC_stats_002')

    
}





def get_intra900041(begDate, endDate){
    
    factor_1min_df = select date, minTime, instrument, percentChange(closeprs, 3)
    from feature1m_MPC_stats_002
    context by date, instrument
        
    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df

}




def feat_dos_003(){
    begDate = 2024.01.01
    endDate = 2024.07.01    
    
    feature1m_MPC_stats = 
    select last((ask_px_0 * ask_sz_0 + bid_px_0 * bid_sz_0) \ (ask_sz_0 + bid_sz_0)) \ (std(ask_px_0 + bid_px_0) + 1.0) as prs_stable, std((ask_px_0 * ask_sz_0 + bid_px_0 * bid_sz_0) \ (ask_sz_0 + bid_sz_0)) as prs_std, last((ask_px_0 * ask_sz_0 + bid_px_0 * bid_sz_0) \ (ask_sz_0 + bid_sz_0)) as closeprs
    from loadTable('dfs://stock_rawdata_sample', 'QuoteL2')
    where settlement_date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6'),
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'],
    (exch_time_ms between 09:15:00.000: 11:30:00.000 )
    or (exch_time_ms between 13:00:00.000: 15:00:00.000)
    group by settlement_date as date,instrument,bar(exch_time_ms,1m) as minTime 
    
        
    
    share(table=streamTable(feature1m_MPC_stats), sharedName=`feature1m_MPC_stats_002)
    dropStreamTable(tableName='feature1m_MPC_stats_002')

    
}

def OFLimbalance(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0,
                 bid_px_1, ask_px_1, bid_sz_1, ask_sz_1,
                 bid_px_2, ask_px_2, bid_sz_2, ask_sz_2,
                 bid_px_3, ask_px_3, bid_sz_3, ask_sz_3,
                 bid_px_4, ask_px_4, bid_sz_4, ask_sz_4){
    delta_bid_v_0 = iif(bid_px_0 > prev(bid_px_0 ), bid_sz_0 , iif(bid_px_0 ==prev(bid_px_0 ), bid_sz_0-prev(bid_sz_0), -prev(bid_sz_0) )) - iif(ask_px_0 > prev(ask_px_0 ), ask_sz_0 , iif(ask_px_0 ==prev(ask_px_0 ), ask_sz_0-prev(ask_sz_0), -prev(ask_sz_0) ))
    delta_bid_v_1 = iif(bid_px_1 > prev(bid_px_1 ), bid_sz_1 , iif(bid_px_1 ==prev(bid_px_1 ), bid_sz_1-prev(bid_sz_1), -prev(bid_sz_1) )) - iif(ask_px_1 > prev(ask_px_1 ), ask_sz_1 , iif(ask_px_1 ==prev(ask_px_1 ), ask_sz_1-prev(ask_sz_1), -prev(ask_sz_1) ))
    delta_bid_v_2 = iif(bid_px_2 > prev(bid_px_2 ), bid_sz_2 , iif(bid_px_2 ==prev(bid_px_2 ), bid_sz_2-prev(bid_sz_2), -prev(bid_sz_2) )) - iif(ask_px_2 > prev(ask_px_2 ), ask_sz_2 , iif(ask_px_2 ==prev(ask_px_2 ), ask_sz_2-prev(ask_sz_2), -prev(ask_sz_2) ))
    delta_bid_v_3 = iif(bid_px_3 > prev(bid_px_3 ), bid_sz_3 , iif(bid_px_3 ==prev(bid_px_3 ), bid_sz_3-prev(bid_sz_3), -prev(bid_sz_3) )) - iif(ask_px_3 > prev(ask_px_3 ), ask_sz_3 , iif(ask_px_3 ==prev(ask_px_3 ), ask_sz_3-prev(ask_sz_3), -prev(ask_sz_3) ))
    delta_bid_v_4 = iif(bid_px_4 > prev(bid_px_4 ), bid_sz_4 , iif(bid_px_4 ==prev(bid_px_4 ), bid_sz_4-prev(bid_sz_4), -prev(bid_sz_4) )) - iif(ask_px_4 > prev(ask_px_4 ), ask_sz_4 , iif(ask_px_4 ==prev(ask_px_4 ), ask_sz_4-prev(ask_sz_4), -prev(ask_sz_4) ))
    return avg(rowWavg(fixedLengthArrayVector(delta_bid_v_0,delta_bid_v_1,delta_bid_v_2,delta_bid_v_3,delta_bid_v_4), 1 2 3 4 5))
}



def OFLimbalance0(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0){
    delta_bid_v_0 = iif(bid_px_0 > prev(bid_px_0 ), bid_sz_0 , iif(bid_px_0 ==prev(bid_px_0 ), bid_sz_0-prev(bid_sz_0), -prev(bid_sz_0) )) - iif(ask_px_0 > prev(ask_px_0 ), ask_sz_0 , iif(ask_px_0 ==prev(ask_px_0 ), ask_sz_0-prev(ask_sz_0), -prev(ask_sz_0) ))
    return avg(delta_bid_v_0)
}

def OFLimbalance4(bid_px_4, ask_px_4, bid_sz_4, ask_sz_4){
    delta_bid_v_4 = iif(bid_px_4 > prev(bid_px_4 ), bid_sz_4 , iif(bid_px_4 ==prev(bid_px_4 ), bid_sz_4-prev(bid_sz_4), -prev(bid_sz_4) )) - iif(ask_px_4 > prev(ask_px_4 ), ask_sz_4 , iif(ask_px_4 ==prev(ask_px_4 ), ask_sz_4-prev(ask_sz_4), -prev(ask_sz_4) ))
    return avg(delta_bid_v_4)
}

def OFLimbalance_std4(bid_px_4, ask_px_4, bid_sz_4, ask_sz_4){
    delta_bid_v_4 = iif(bid_px_4 > prev(bid_px_4 ), bid_sz_4 , iif(bid_px_4 ==prev(bid_px_4 ), bid_sz_4-prev(bid_sz_4), -prev(bid_sz_4) )) - iif(ask_px_4 > prev(ask_px_4 ), ask_sz_4 , iif(ask_px_4 ==prev(ask_px_4 ), ask_sz_4-prev(ask_sz_4), -prev(ask_sz_4) ))
    return std(delta_bid_v_4)
}



def OFLimbalance_std(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0,
                 bid_px_1, ask_px_1, bid_sz_1, ask_sz_1,
                 bid_px_2, ask_px_2, bid_sz_2, ask_sz_2,
                 bid_px_3, ask_px_3, bid_sz_3, ask_sz_3,
                 bid_px_4, ask_px_4, bid_sz_4, ask_sz_4){
    delta_bid_v_0 = iif(bid_px_0 > prev(bid_px_0 ), bid_sz_0 , iif(bid_px_0 ==prev(bid_px_0 ), bid_sz_0-prev(bid_sz_0), -prev(bid_sz_0) )) - iif(ask_px_0 > prev(ask_px_0 ), ask_sz_0 , iif(ask_px_0 ==prev(ask_px_0 ), ask_sz_0-prev(ask_sz_0), -prev(ask_sz_0) ))
    delta_bid_v_1 = iif(bid_px_1 > prev(bid_px_1 ), bid_sz_1 , iif(bid_px_1 ==prev(bid_px_1 ), bid_sz_1-prev(bid_sz_1), -prev(bid_sz_1) )) - iif(ask_px_1 > prev(ask_px_1 ), ask_sz_1 , iif(ask_px_1 ==prev(ask_px_1 ), ask_sz_1-prev(ask_sz_1), -prev(ask_sz_1) ))
    delta_bid_v_2 = iif(bid_px_2 > prev(bid_px_2 ), bid_sz_2 , iif(bid_px_2 ==prev(bid_px_2 ), bid_sz_2-prev(bid_sz_2), -prev(bid_sz_2) )) - iif(ask_px_2 > prev(ask_px_2 ), ask_sz_2 , iif(ask_px_2 ==prev(ask_px_2 ), ask_sz_2-prev(ask_sz_2), -prev(ask_sz_2) ))
    delta_bid_v_3 = iif(bid_px_3 > prev(bid_px_3 ), bid_sz_3 , iif(bid_px_3 ==prev(bid_px_3 ), bid_sz_3-prev(bid_sz_3), -prev(bid_sz_3) )) - iif(ask_px_3 > prev(ask_px_3 ), ask_sz_3 , iif(ask_px_3 ==prev(ask_px_3 ), ask_sz_3-prev(ask_sz_3), -prev(ask_sz_3) ))
    delta_bid_v_4 = iif(bid_px_4 > prev(bid_px_4 ), bid_sz_4 , iif(bid_px_4 ==prev(bid_px_4 ), bid_sz_4-prev(bid_sz_4), -prev(bid_sz_4) )) - iif(ask_px_4 > prev(ask_px_4 ), ask_sz_4 , iif(ask_px_4 ==prev(ask_px_4 ), ask_sz_4-prev(ask_sz_4), -prev(ask_sz_4) ))
    return std(rowWavg(fixedLengthArrayVector(delta_bid_v_0,delta_bid_v_1,delta_bid_v_2,delta_bid_v_3,delta_bid_v_4), 1 2 3 4 5))
}


def OFLimbalance_high(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0,
                 bid_px_1, ask_px_1, bid_sz_1, ask_sz_1,
                 bid_px_2, ask_px_2, bid_sz_2, ask_sz_2,
                 bid_px_3, ask_px_3, bid_sz_3, ask_sz_3,
                 bid_px_4, ask_px_4, bid_sz_4, ask_sz_4){
    delta_bid_v_0 = iif(bid_px_0 > prev(bid_px_0 ), bid_sz_0 , iif(bid_px_0 ==prev(bid_px_0 ), bid_sz_0-prev(bid_sz_0), -prev(bid_sz_0) )) - iif(ask_px_0 > prev(ask_px_0 ), ask_sz_0 , iif(ask_px_0 ==prev(ask_px_0 ), ask_sz_0-prev(ask_sz_0), -prev(ask_sz_0) ))
    delta_bid_v_1 = iif(bid_px_1 > prev(bid_px_1 ), bid_sz_1 , iif(bid_px_1 ==prev(bid_px_1 ), bid_sz_1-prev(bid_sz_1), -prev(bid_sz_1) )) - iif(ask_px_1 > prev(ask_px_1 ), ask_sz_1 , iif(ask_px_1 ==prev(ask_px_1 ), ask_sz_1-prev(ask_sz_1), -prev(ask_sz_1) ))
    delta_bid_v_2 = iif(bid_px_2 > prev(bid_px_2 ), bid_sz_2 , iif(bid_px_2 ==prev(bid_px_2 ), bid_sz_2-prev(bid_sz_2), -prev(bid_sz_2) )) - iif(ask_px_2 > prev(ask_px_2 ), ask_sz_2 , iif(ask_px_2 ==prev(ask_px_2 ), ask_sz_2-prev(ask_sz_2), -prev(ask_sz_2) ))
    delta_bid_v_3 = iif(bid_px_3 > prev(bid_px_3 ), bid_sz_3 , iif(bid_px_3 ==prev(bid_px_3 ), bid_sz_3-prev(bid_sz_3), -prev(bid_sz_3) )) - iif(ask_px_3 > prev(ask_px_3 ), ask_sz_3 , iif(ask_px_3 ==prev(ask_px_3 ), ask_sz_3-prev(ask_sz_3), -prev(ask_sz_3) ))
    delta_bid_v_4 = iif(bid_px_4 > prev(bid_px_4 ), bid_sz_4 , iif(bid_px_4 ==prev(bid_px_4 ), bid_sz_4-prev(bid_sz_4), -prev(bid_sz_4) )) - iif(ask_px_4 > prev(ask_px_4 ), ask_sz_4 , iif(ask_px_4 ==prev(ask_px_4 ), ask_sz_4-prev(ask_sz_4), -prev(ask_sz_4) ))
    return max(rowWavg(fixedLengthArrayVector(delta_bid_v_0,delta_bid_v_1,delta_bid_v_2,delta_bid_v_3,delta_bid_v_4), 1 2 3 4 5))
}


def OFLimbalance_low(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0,
                 bid_px_1, ask_px_1, bid_sz_1, ask_sz_1,
                 bid_px_2, ask_px_2, bid_sz_2, ask_sz_2,
                 bid_px_3, ask_px_3, bid_sz_3, ask_sz_3,
                 bid_px_4, ask_px_4, bid_sz_4, ask_sz_4){
    delta_bid_v_0 = iif(bid_px_0 > prev(bid_px_0 ), bid_sz_0 , iif(bid_px_0 ==prev(bid_px_0 ), bid_sz_0-prev(bid_sz_0), -prev(bid_sz_0) )) - iif(ask_px_0 > prev(ask_px_0 ), ask_sz_0 , iif(ask_px_0 ==prev(ask_px_0 ), ask_sz_0-prev(ask_sz_0), -prev(ask_sz_0) ))
    delta_bid_v_1 = iif(bid_px_1 > prev(bid_px_1 ), bid_sz_1 , iif(bid_px_1 ==prev(bid_px_1 ), bid_sz_1-prev(bid_sz_1), -prev(bid_sz_1) )) - iif(ask_px_1 > prev(ask_px_1 ), ask_sz_1 , iif(ask_px_1 ==prev(ask_px_1 ), ask_sz_1-prev(ask_sz_1), -prev(ask_sz_1) ))
    delta_bid_v_2 = iif(bid_px_2 > prev(bid_px_2 ), bid_sz_2 , iif(bid_px_2 ==prev(bid_px_2 ), bid_sz_2-prev(bid_sz_2), -prev(bid_sz_2) )) - iif(ask_px_2 > prev(ask_px_2 ), ask_sz_2 , iif(ask_px_2 ==prev(ask_px_2 ), ask_sz_2-prev(ask_sz_2), -prev(ask_sz_2) ))
    delta_bid_v_3 = iif(bid_px_3 > prev(bid_px_3 ), bid_sz_3 , iif(bid_px_3 ==prev(bid_px_3 ), bid_sz_3-prev(bid_sz_3), -prev(bid_sz_3) )) - iif(ask_px_3 > prev(ask_px_3 ), ask_sz_3 , iif(ask_px_3 ==prev(ask_px_3 ), ask_sz_3-prev(ask_sz_3), -prev(ask_sz_3) ))
    delta_bid_v_4 = iif(bid_px_4 > prev(bid_px_4 ), bid_sz_4 , iif(bid_px_4 ==prev(bid_px_4 ), bid_sz_4-prev(bid_sz_4), -prev(bid_sz_4) )) - iif(ask_px_4 > prev(ask_px_4 ), ask_sz_4 , iif(ask_px_4 ==prev(ask_px_4 ), ask_sz_4-prev(ask_sz_4), -prev(ask_sz_4) ))
    return min(rowWavg(fixedLengthArrayVector(delta_bid_v_0,delta_bid_v_1,delta_bid_v_2,delta_bid_v_3,delta_bid_v_4), 1 2 3 4 5))
}


def OFLimbalance_up(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0,
                 bid_px_1, ask_px_1, bid_sz_1, ask_sz_1,
                 bid_px_2, ask_px_2, bid_sz_2, ask_sz_2,
                 bid_px_3, ask_px_3, bid_sz_3, ask_sz_3,
                 bid_px_4, ask_px_4, bid_sz_4, ask_sz_4){
    delta_bid_v_0 = iif(bid_px_0 > prev(bid_px_0 ), bid_sz_0 , iif(bid_px_0 ==prev(bid_px_0 ), bid_sz_0-prev(bid_sz_0), -prev(bid_sz_0) )) - iif(ask_px_0 > prev(ask_px_0 ), ask_sz_0 , iif(ask_px_0 ==prev(ask_px_0 ), ask_sz_0-prev(ask_sz_0), -prev(ask_sz_0) ))
    delta_bid_v_1 = iif(bid_px_1 > prev(bid_px_1 ), bid_sz_1 , iif(bid_px_1 ==prev(bid_px_1 ), bid_sz_1-prev(bid_sz_1), -prev(bid_sz_1) )) - iif(ask_px_1 > prev(ask_px_1 ), ask_sz_1 , iif(ask_px_1 ==prev(ask_px_1 ), ask_sz_1-prev(ask_sz_1), -prev(ask_sz_1) ))
    delta_bid_v_2 = iif(bid_px_2 > prev(bid_px_2 ), bid_sz_2 , iif(bid_px_2 ==prev(bid_px_2 ), bid_sz_2-prev(bid_sz_2), -prev(bid_sz_2) )) - iif(ask_px_2 > prev(ask_px_2 ), ask_sz_2 , iif(ask_px_2 ==prev(ask_px_2 ), ask_sz_2-prev(ask_sz_2), -prev(ask_sz_2) ))
    delta_bid_v_3 = iif(bid_px_3 > prev(bid_px_3 ), bid_sz_3 , iif(bid_px_3 ==prev(bid_px_3 ), bid_sz_3-prev(bid_sz_3), -prev(bid_sz_3) )) - iif(ask_px_3 > prev(ask_px_3 ), ask_sz_3 , iif(ask_px_3 ==prev(ask_px_3 ), ask_sz_3-prev(ask_sz_3), -prev(ask_sz_3) ))
    delta_bid_v_4 = iif(bid_px_4 > prev(bid_px_4 ), bid_sz_4 , iif(bid_px_4 ==prev(bid_px_4 ), bid_sz_4-prev(bid_sz_4), -prev(bid_sz_4) )) - iif(ask_px_4 > prev(ask_px_4 ), ask_sz_4 , iif(ask_px_4 ==prev(ask_px_4 ), ask_sz_4-prev(ask_sz_4), -prev(ask_sz_4) ))
    avv = (rowWavg(fixedLengthArrayVector(delta_bid_v_0,delta_bid_v_1,delta_bid_v_2,delta_bid_v_3,delta_bid_v_4), 1 2 3 4 5))
    return avg(iif(avv>0.0, avv, NULL))
}



def feat_dos_004(){




feature1m_OFL_stats = 
select OFLimbalance(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0,
                 bid_px_1, ask_px_1, bid_sz_1, ask_sz_1,
                 bid_px_2, ask_px_2, bid_sz_2, ask_sz_2,
                 bid_px_3, ask_px_3, bid_sz_3, ask_sz_3,
                 bid_px_4, ask_px_4, bid_sz_4, ask_sz_4) as OFLimbalance5all, 
        OFLimbalance_up(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0,
                 bid_px_1, ask_px_1, bid_sz_1, ask_sz_1,
                 bid_px_2, ask_px_2, bid_sz_2, ask_sz_2,
                 bid_px_3, ask_px_3, bid_sz_3, ask_sz_3,
                 bid_px_4, ask_px_4, bid_sz_4, ask_sz_4) as OFLimbalance5_up, 
        OFLimbalance_low(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0,
                 bid_px_1, ask_px_1, bid_sz_1, ask_sz_1,
                 bid_px_2, ask_px_2, bid_sz_2, ask_sz_2,
                 bid_px_3, ask_px_3, bid_sz_3, ask_sz_3,
                 bid_px_4, ask_px_4, bid_sz_4, ask_sz_4) as OFLimbalance5_low,     
        OFLimbalance_high(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0,
                 bid_px_1, ask_px_1, bid_sz_1, ask_sz_1,
                 bid_px_2, ask_px_2, bid_sz_2, ask_sz_2,
                 bid_px_3, ask_px_3, bid_sz_3, ask_sz_3,
                 bid_px_4, ask_px_4, bid_sz_4, ask_sz_4) as OFLimbalance5_high,     
       
        OFLimbalance_std(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0,
                 bid_px_1, ask_px_1, bid_sz_1, ask_sz_1,
                 bid_px_2, ask_px_2, bid_sz_2, ask_sz_2,
                 bid_px_3, ask_px_3, bid_sz_3, ask_sz_3,
                 bid_px_4, ask_px_4, bid_sz_4, ask_sz_4) as OFLimbalance5_std,         
       OFLimbalance0(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0) as OFLimbalance0avg,
       OFLimbalance4(bid_px_4, ask_px_4, bid_sz_4, ask_sz_4) as OFLimbalance4avg,
       OFLimbalance_std4(bid_px_4, ask_px_4, bid_sz_4, ask_sz_4) as OFLimbalancestd4           
from loadTable('dfs://stock_rawdata_sample', 'QuoteL2')
where settlement_date between begDate: endDate,
instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6'),
// instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'],
(exch_time_ms between 09:15:00.000: 11:30:00.000 )
or (exch_time_ms between 13:00:00.000: 15:00:00.000)
group by settlement_date as date,instrument,bar(exch_time_ms,1m) as minTime 

}


def get_intra900043(begDate, endDate){
    
    factor_1min_df = select date, minTime, instrument, (OFLimbalance5all - mavg(OFLimbalance5all, 15)) \ mstd(OFLimbalance5all, 15) as OFLimbalance5all_norm
    from feature1m_OFL_stats_003
    context by date, instrument


    
    
    
    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df

}


def get_intra900044(begDate, endDate){


    ambiguou_df = select date, minTime, instrument, close
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 

    order_money_df = select date, minTime, instrument, slope_pxsz_avg, slope_pxsz_std, slope_pxsz_last
    from feature1m_SLOPE_stats_004
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')


    
    joined_df = SELECT date, minTime, instrument, close, slope_pxsz_avg, slope_pxsz_std, slope_pxsz_last
    FROM order_money_df
    JOIN ambiguou_df
    ON order_money_df.date=ambiguou_df.date and order_money_df.minTime=ambiguou_df.minTime and order_money_df.instrument=ambiguou_df.instrument order by date, instrument
    
    
    corr_ambi_money_df = SELECT date, minTime, instrument, mstd(slope_pxsz_avg, 20, 20), mstd(slope_pxsz_std, 20, 20), mavg(slope_pxsz_std, 20, 20), mcorr(prev(percentChange(slope_pxsz_last, 5)), close / first(close), 25) as corr_shiftslopliq_relativeprxm,
    mcorr(mfirst(slope_pxsz_avg, 2), percentChange(close, 5), 25) as corr_slopliq_relativeprxm 
    from joined_df context by date, instrument

    
    factor_sample_df = select * 
    from corr_ambi_money_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000]// 日内每一分钟都有因子生成，采样数据回测快一些
    return factor_sample_df

}






def w10slope_avgall(bid_px_0, ask_px_0, bid_px_10, ask_px_10, bidQ9_1, askQ9_1){
    slope_bid_v = abs((bid_px_10 - bid_px_0) \ rowSum(bidQ9_1))
    slope_ask_v = abs((ask_px_10 - ask_px_0) \ rowSum(askQ9_1))
    return avg((slope_bid_v - slope_ask_v) / (slope_bid_v + slope_ask_v))
}

def w10slope_avgonly(bid_px_0, ask_px_0, bid_px_10, ask_px_10, bidQ9_1, askQ9_1){
    slope_bid_v = abs((bid_px_10 - bid_px_0) \ rowSum(bidQ9_1))
    slope_ask_v = abs((ask_px_10 - ask_px_0) \ rowSum(askQ9_1))
    return avg((slope_bid_v - slope_ask_v))
}

def w10slope_std(bid_px_0, ask_px_0, bid_px_10, ask_px_10, bidQ9_1, askQ9_1){
    slope_bid_v = abs((bid_px_10 - bid_px_0) \ rowSum(bidQ9_1))
    slope_ask_v = abs((ask_px_10 - ask_px_0) \ rowSum(askQ9_1))
    return std((slope_bid_v - slope_ask_v) / (slope_bid_v + slope_ask_v))
}

def w10slope_skew(bid_px_0, ask_px_0, bid_px_10, ask_px_10, bidQ9_1, askQ9_1){
    slope_bid_v = abs((bid_px_10 - bid_px_0) \ rowSum(bidQ9_1))
    slope_ask_v = abs((ask_px_10 - ask_px_0) \ rowSum(askQ9_1))
    return skew((slope_bid_v - slope_ask_v) / (slope_bid_v + slope_ask_v))
}

def w10slope_high(bid_px_0, ask_px_0, bid_px_10, ask_px_10, bidQ9_1, askQ9_1){
    slope_bid_v = abs((bid_px_10 - bid_px_0) \ rowSum(bidQ9_1))
    slope_ask_v = abs((ask_px_10 - ask_px_0) \ rowSum(askQ9_1))
    return max((slope_bid_v - slope_ask_v) / (slope_bid_v + slope_ask_v))
}
def w10slope_low(bid_px_0, ask_px_0, bid_px_10, ask_px_10, bidQ9_1, askQ9_1){
    slope_bid_v = abs((bid_px_10 - bid_px_0) \ rowSum(bidQ9_1))
    slope_ask_v = abs((ask_px_10 - ask_px_0) \ rowSum(askQ9_1))
    return min((slope_bid_v - slope_ask_v) / (slope_bid_v + slope_ask_v))
}
def w10slope_path(bid_px_0, ask_px_0, bid_px_10, ask_px_10, bidQ9_1, askQ9_1){
    slope_bid_v = abs((bid_px_10 - bid_px_0) \ rowSum(bidQ9_1))
    slope_ask_v = abs((ask_px_10 - ask_px_0) \ rowSum(askQ9_1))
    return sum(slope_bid_v - slope_ask_v) \ sum(abs(slope_bid_v - slope_ask_v))
}
def w10slope_last(bid_px_0, ask_px_0, bid_px_10, ask_px_10, bidQ9_1, askQ9_1){
    slope_bid_v = abs((bid_px_10 - bid_px_0) \ rowSum(bidQ9_1))
    slope_ask_v = abs((ask_px_10 - ask_px_0) \ rowSum(askQ9_1))
    return last((slope_bid_v - slope_ask_v) / (slope_bid_v + slope_ask_v))
}


def w5diffslope_avg(bid_px_0, ask_px_0, bid_px_10, ask_px_10, bidQ9_1, askQ9_1){

    return avg(abs((bid_px_0 - ask_px_0) \ (rowSum(bidQ9_1) +  rowSum(askQ9_1))))
}
def w0slope_avg(bid_px_0, ask_px_0, bidQ9_1, askQ9_1){

    return avg(abs((bid_px_0 - ask_px_0) \ (bidQ9_1 +  askQ9_1)))
}


def feat_dos_005(){


    
    



feature1m_CVXINNERSLOP_stats = 


    select w10slope_avgall(bid_px_0, ask_px_0, bid_px_9, ask_px_9, fixedLengthArrayVector(bid_sz_1,bid_sz_2,bid_sz_3,bid_sz_4,bid_sz_5,bid_sz_6,bid_sz_7,bid_sz_8,bid_sz_9), fixedLengthArrayVector(ask_sz_1,ask_sz_2,ask_sz_3,ask_sz_4,ask_sz_5,ask_sz_6,ask_sz_7,ask_sz_8,ask_sz_9)) as w10slope_avgall,
    w10slope_avgall(bid_px_0, ask_px_0, bid_px_5, ask_px_5, fixedLengthArrayVector(bid_sz_1,bid_sz_2,bid_sz_3,bid_sz_4,bid_sz_5), fixedLengthArrayVector(ask_sz_1,ask_sz_2,ask_sz_3,ask_sz_4,ask_sz_5)) as w5slope_avg,
    w10slope_avgonly(bid_px_0, ask_px_0, bid_px_5, ask_px_5, fixedLengthArrayVector(bid_sz_1,bid_sz_2,bid_sz_3,bid_sz_4,bid_sz_5), fixedLengthArrayVector(ask_sz_1,ask_sz_2,ask_sz_3,ask_sz_4,ask_sz_5)) as w5slope_avgonly,
    w10slope_avgall(bid_px_5, ask_px_5, bid_px_9, ask_px_9, fixedLengthArrayVector(bid_sz_6,bid_sz_7,bid_sz_8,bid_sz_9), fixedLengthArrayVector(ask_sz_6,ask_sz_7,ask_sz_8,ask_sz_9)) as w510slope_avg,
    w5diffslope_avg(bid_px_0, ask_px_0, bid_px_5, ask_px_5, fixedLengthArrayVector(bid_sz_1,bid_sz_2,bid_sz_3,bid_sz_4,bid_sz_5), fixedLengthArrayVector(ask_sz_1,ask_sz_2,ask_sz_3,ask_sz_4,ask_sz_5)) as w5diffslope_avg,
    w0slope_avg(bid_px_0, ask_px_0, bid_sz_0, ask_sz_0) as w0slope_avg,
    w10slope_std(bid_px_0, ask_px_0, bid_px_5, ask_px_5, fixedLengthArrayVector(bid_sz_1,bid_sz_2,bid_sz_3,bid_sz_4,bid_sz_5), fixedLengthArrayVector(ask_sz_1,ask_sz_2,ask_sz_3,ask_sz_4,ask_sz_5)) as w5slope_std,
    w10slope_skew(bid_px_0, ask_px_0, bid_px_5, ask_px_5, fixedLengthArrayVector(bid_sz_1,bid_sz_2,bid_sz_3,bid_sz_4,bid_sz_5), fixedLengthArrayVector(ask_sz_1,ask_sz_2,ask_sz_3,ask_sz_4,ask_sz_5)) as w5slope_skew,
    w10slope_high(bid_px_0, ask_px_0, bid_px_5, ask_px_5, fixedLengthArrayVector(bid_sz_1,bid_sz_2,bid_sz_3,bid_sz_4,bid_sz_5), fixedLengthArrayVector(ask_sz_1,ask_sz_2,ask_sz_3,ask_sz_4,ask_sz_5)) as w5slope_high,
    w10slope_last(bid_px_0, ask_px_0, bid_px_5, ask_px_5, fixedLengthArrayVector(bid_sz_1,bid_sz_2,bid_sz_3,bid_sz_4,bid_sz_5), fixedLengthArrayVector(ask_sz_1,ask_sz_2,ask_sz_3,ask_sz_4,ask_sz_5)) as w5slope_last,
    w10slope_path(bid_px_0, ask_px_0, bid_px_5, ask_px_5, fixedLengthArrayVector(bid_sz_1,bid_sz_2,bid_sz_3,bid_sz_4,bid_sz_5), fixedLengthArrayVector(ask_sz_1,ask_sz_2,ask_sz_3,ask_sz_4,ask_sz_5)) as w5slope_path
    from loadTable('dfs://stock_rawdata_sample', 'QuoteL2')
    where settlement_date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6'),
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'],
    (exch_time_ms between 09:15:00.000: 11:30:00.000 )
    or (exch_time_ms between 13:00:00.000: 15:00:00.000)
    group by settlement_date as date,instrument,bar(exch_time_ms,1m) as minTime 
    



}





def get_intra900045(begDate, endDate){


  
    factor_1min_df = select date, minTime, instrument, (w5slope_avg - mavg(w5slope_avg, 20)) \ mstd(w5slope_avg, 20) as w5slope_avg_norm, -w5slope_avg
    from feature1m_CVXINNERSLOP_stats_005
    context by date, instrument


    
    
    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000] 
    return factor_sample_df

}


def traPriceWeightedNetBuyQuoteVolumeRatio(bid,bidQty,ask,askQty,TotalValTrd,TotalVolTrd){
	prevbid = prev(bid)
	prevbidQty = prev(bidQty)
	prevask = prev(ask)
	prevaskQty = prev(askQty)
	bidchg = iif(round(bid-prevbid,2)>0, bidQty, iif(round(bid-prevbid,2)<0, -prevbidQty, bidQty-prevbidQty))
	offerchg = iif(iif(ask==0,iif(prevask>0,1,0), ask-prevask)>0, -prevaskQty, iif(iif(prevask==0,
		iif(ask>0,-1,0), iif(ask>0,ask-prevask,1))<0, askQty, askQty-prevaskQty))
	avgprice = deltas(TotalValTrd)\deltas(TotalVolTrd)
	factorValue = (bidchg-offerchg)\(abs(bidchg)+abs(offerchg))*avgprice
	return sum(factorValue) \ sum(avgprice)
}
def level10_Diff(price, qty, buy){
        prevPrice = price.prev()
        left, right = rowAlign(price, prevPrice, how=iif(buy, "bid", "ask"))
        qtyDiff = (qty.rowAt(left).nullFill(0) - qty.prev().rowAt(right).nullFill(0)) 
        amtDiff = rowSum(nullFill(price.rowAt(left), prevPrice.rowAt(right)) * qtyDiff)
        return sum(amtDiff) 
}
def level10_Diff_bid_unit(price, qty, buy){
        prevPrice = price.prev()
        left, right = rowAlign(price, prevPrice, how=iif(buy, "bid", "ask"))
        qtyDiff = (qty.rowAt(left).nullFill(0) - qty.prev().rowAt(right).nullFill(0)) 
        amtDiff = rowSum(nullFill(price.rowAt(left), prevPrice.rowAt(right)) * qtyDiff)
        return sum(amtDiff) \  sum(rowSum(qty))
}



def level10_Diff_bidaskstd(price, qty, price2, qty2){
        prevPrice = price.prev()
        left, right = rowAlign(price, prevPrice, "bid")
        qtyDiff = (qty.rowAt(left).nullFill(0) - qty.prev().rowAt(right).nullFill(0)) 
        amtDiff = rowSum(nullFill(price.rowAt(left), prevPrice.rowAt(right)) * qtyDiff)
        
        prevPrice2 = price2.prev()
        left2, right2 = rowAlign(price2, prevPrice2, "ask")
        qtyDiff2 = (qty2.rowAt(left2).nullFill(0) - qty2.prev().rowAt(right2).nullFill(0)) 
        amtDiff2 = rowSum(nullFill(price2.rowAt(left2), prevPrice2.rowAt(right2)) * qtyDiff2)
        return (std(amtDiff - amtDiff2))  \ sum((rowSum(qty) + rowSum(qty2)))
}
def level10_Diff_bidaskavg(price, qty, price2, qty2){
        prevPrice = price.prev()
        left, right = rowAlign(price, prevPrice, "bid")
        qtyDiff = (qty.rowAt(left).nullFill(0) - qty.prev().rowAt(right).nullFill(0)) 
        amtDiff = rowSum(nullFill(price.rowAt(left), prevPrice.rowAt(right)) * qtyDiff)
        
        prevPrice2 = price2.prev()
        left2, right2 = rowAlign(price2, prevPrice2, "ask")
        qtyDiff2 = (qty2.rowAt(left2).nullFill(0) - qty2.prev().rowAt(right2).nullFill(0)) 
        amtDiff2 = rowSum(nullFill(price2.rowAt(left2), prevPrice2.rowAt(right2)) * qtyDiff2)
        return (sum(amtDiff - amtDiff2))  \ sum(rowSum(qty) + rowSum(qty2))
}




def feat_dos_006(){

    feature1m_PWB_stats = 
    select traPriceWeightedNetBuyQuoteVolumeRatio(bid_px_0, bid_sz_0, ask_px_0, ask_sz_0, turnover, volume) as PNBprice_netbid0_share,
    level10_Diff(fixedLengthArrayVector(bid_px_0, bid_px_1, bid_px_2, bid_px_3, bid_px_4), fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), true) as level05_bidQtDiff,
    level10_Diff_bid_unit(fixedLengthArrayVector(bid_px_0, bid_px_1, bid_px_2, bid_px_3, bid_px_4), fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), true) as level05_bidQtDiff_unit,
    level10_Diff(fixedLengthArrayVector(ask_px_0, ask_px_1, ask_px_2, ask_px_3, ask_px_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4), false) as level05_askQtDiff,
    level10_Diff_bidaskstd(fixedLengthArrayVector(bid_px_0, bid_px_1, bid_px_2, bid_px_3, bid_px_4), fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4),fixedLengthArrayVector(ask_px_0, ask_px_1, ask_px_2, ask_px_3, ask_px_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4) ) as level05_bidaskQtDiffstd,
    level10_Diff_bidaskavg(fixedLengthArrayVector(bid_px_0, bid_px_1, bid_px_2, bid_px_3, bid_px_4), fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4),fixedLengthArrayVector(ask_px_0, ask_px_1, ask_px_2, ask_px_3, ask_px_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4) ) as level05_bidaskQtDiffavg
    from loadTable('dfs://stock_rawdata_sample', 'QuoteL2')
    where settlement_date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6'),
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'],
    (exch_time_ms between 09:15:00.000: 11:30:00.000 )
    or (exch_time_ms between 13:00:00.000: 15:00:00.000)
    group by settlement_date as date,instrument,bar(exch_time_ms,1m) as minTime 


}





def get_intra900046(begDate, endDate){


  
    factor_1min_df = select date, minTime, instrument, PNBprice_netbid0_share, (PNBprice_netbid0_share - mavg(PNBprice_netbid0_share, 25)) \ mstd(PNBprice_netbid0_share, 25) as PNBprice_netbid0_share_norm,  mavg(PNBprice_netbid0_share, 15) \ mstd(PNBprice_netbid0_share, 15), level05_bidQtDiff_unit, (level05_bidaskQtDiffavg - mavg(level05_bidaskQtDiffavg, 25)) \ mstd(level05_bidaskQtDiffavg, 25) as level05_bidaskQtDiffavg_norm
    from feature1m_PWB_stats_006
    context by date, instrument


    
    
    factor_sample_df = select * 
    from factor_1min_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000] 
    return factor_sample_df

}

def get_intra900047(begDate, endDate){

    ambiguou_df = select date, minTime, instrument, close, volume
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    
    order_money_df = select date, minTime, instrument, PNBprice_netbid0_share, level05_bidQtDiff, level05_askQtDiff
    from feature1m_PWB_stats_006
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 

    
    joined_df = SELECT date, minTime, instrument, PNBprice_netbid0_share, level05_bidQtDiff, level05_askQtDiff, close, volume
    FROM order_money_df
    JOIN ambiguou_df
    ON order_money_df.date=ambiguou_df.date and order_money_df.minTime=ambiguou_df.minTime and order_money_df.instrument=ambiguou_df.instrument order by date, instrument

    corr_ambi_money_df = SELECT date, minTime, instrument, mavgTopN(percentChange(close, 5), level05_bidQtDiff, 25, 7, false) as level05_bidQtDiff_ext_pct
    from joined_df context by date, instrument

    
    
    factor_sample_df = select * 
    from corr_ambi_money_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000] 
    return factor_sample_df

}












def trendcorrlevel5(bid,bidQty,ask,askQty){
    avgPrice5 =(rowSum(bid*bidQty)+rowSum(ask*askQty))\(rowSum(bidQty)+rowSum(askQty))
    return corr(avgPrice5, 1..size(avgPrice5))
}


def trendslopelevel5(bid,bidQty,ask,askQty){
    avgPrice5 =(rowSum(bid*bidQty)+rowSum(ask*askQty))\(rowSum(bidQty)+rowSum(askQty))
    return beta(avgPrice5, 1..size(avgPrice5)) \ first(avgPrice5)
}


def inner_outer_diff_prs(bid,bidQty,ask,askQty, bidout,bidQtyout,askout,askQtyout){
    avgPrice5 =(rowSum(bid*bidQty)+rowSum(ask*askQty))\(rowSum(bidQty)+rowSum(askQty))
    avgPrice5_out =(rowSum(bidout*bidQtyout)+rowSum(askout*askQtyout))\(rowSum(bidQtyout)+rowSum(askQtyout))
    return avg((avgPrice5 - avgPrice5_out) \ (avgPrice5 + avgPrice5_out))
}


def inner_outer_diff_prsstd(bid,bidQty,ask,askQty, bidout,bidQtyout,askout,askQtyout){
    avgPrice5 =(rowSum(bid*bidQty)+rowSum(ask*askQty))\(rowSum(bidQty)+rowSum(askQty))
    avgPrice5_out =(rowSum(bidout*bidQtyout)+rowSum(askout*askQtyout))\(rowSum(bidQtyout)+rowSum(askQtyout))
    return std((avgPrice5 - avgPrice5_out) \ (avgPrice5 + avgPrice5_out))
}


def feat_dos_007(begDate, endDate){

  
    
    feature1m_TRENDslopavg_stats = 
    select trendcorrlevel5(fixedLengthArrayVector(bid_px_0, bid_px_1, bid_px_2, bid_px_3, bid_px_4), fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), fixedLengthArrayVector(ask_px_0, ask_px_1, ask_px_2, ask_px_3, ask_px_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4)) as trendcorrlevel5, 
    select trendslopelevel5(fixedLengthArrayVector(bid_px_0, bid_px_1, bid_px_2, bid_px_3, bid_px_4), fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), fixedLengthArrayVector(ask_px_0, ask_px_1, ask_px_2, ask_px_3, ask_px_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4)) as trendslopelevel5, 
    inner_outer_diff_prs(fixedLengthArrayVector(bid_px_0, bid_px_1, bid_px_2, bid_px_3, bid_px_4), fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), fixedLengthArrayVector(ask_px_0, ask_px_1, ask_px_2, ask_px_3, ask_px_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4),
    fixedLengthArrayVector(bid_px_5, bid_px_6, bid_px_7, bid_px_8, bid_px_9), fixedLengthArrayVector(bid_sz_5, bid_sz_6, bid_sz_7, bid_sz_8, bid_sz_9), fixedLengthArrayVector(ask_px_5, ask_px_6, ask_px_7, ask_px_8, ask_px_9), fixedLengthArrayVector(ask_sz_5, ask_sz_6, ask_sz_7, ask_sz_8, ask_sz_9)) as inner_outer_diff_prsavg, 
    inner_outer_diff_prsstd(fixedLengthArrayVector(bid_px_0, bid_px_1, bid_px_2, bid_px_3, bid_px_4), fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), fixedLengthArrayVector(ask_px_0, ask_px_1, ask_px_2, ask_px_3, ask_px_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4),
    fixedLengthArrayVector(bid_px_5, bid_px_6, bid_px_7, bid_px_8, bid_px_9), fixedLengthArrayVector(bid_sz_5, bid_sz_6, bid_sz_7, bid_sz_8, bid_sz_9), fixedLengthArrayVector(ask_px_5, ask_px_6, ask_px_7, ask_px_8, ask_px_9), fixedLengthArrayVector(ask_sz_5, ask_sz_6, ask_sz_7, ask_sz_8, ask_sz_9)) as inner_outer_diff_prsstd
    from loadTable('dfs://stock_rawdata_sample', 'QuoteL2')
    where settlement_date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6'),
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'],
    (exch_time_ms between 09:15:00.000: 11:30:00.000)
    or (exch_time_ms between 13:00:00.000: 15:00:00.000)
    group by settlement_date as date,instrument,bar(exch_time_ms,1m) as minTime 
    

}









def get_intra900048(begDate, endDate){


    order_money_df = select date, minTime, instrument, trendslopelevel5, mstd(trendslopelevel5, 20), mstd(inner_outer_diff_prsavg, 20), mstd(inner_outer_diff_prsstd, 20), mstd(inner_outer_diff_prsavg, 25) \ mavg(inner_outer_diff_prsavg, 25), mavg(inner_outer_diff_prsavg \ inner_outer_diff_prsstd, 20)
    from feature1m_TRENDslopavg_stats_007
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument

    factor_sample_df = select * 
    from order_money_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000] 
    return factor_sample_df

}


def marketurgency(bid_px_0, ask_px_0, bidQty, askQty){
    return avg(rowWavg((bidQty - askQty)\(bidQty + askQty), 5 4 3 2 1) * (ask_px_0 - bid_px_0))
}

def marketurgency_std(bid_px_0, ask_px_0, bidQty, askQty){
    return std(rowWavg((bidQty - askQty)\(bidQty + askQty), 5 4 3 2 1) * (ask_px_0 - bid_px_0))
}



def feat_dos_008(begDate, endDate){

  feature1m_marketurgencyfeat_stats = 
select marketurgency(bid_px_0, ask_px_0, fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3,ask_sz_4)) as marketurgencym, marketurgency_std(bid_px_0, ask_px_0, fixedLengthArrayVector(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4), fixedLengthArrayVector(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3,ask_sz_4)) as marketurgency_std
from loadTable('dfs://stock_rawdata_sample', 'QuoteL2')
where settlement_date between begDate: endDate,
instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6'),
// instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'],
(exch_time_ms between 09:15:00.000: 11:30:00.000 )
or (exch_time_ms between 13:00:00.000: 15:00:00.000)
group by settlement_date as date,instrument,bar(exch_time_ms,1m) as minTime     

}






def get_intra900049(begDate, endDate){


    order_money_df = select date, minTime, instrument, (marketurgencym - mavg(marketurgencym, 20)) \ mstd(marketurgencym, 20) as marketurgencym_zsc,  marketurgencym
    
    from feature1m_marketurgencyfeat_stats_008
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument

    factor_sample_df = select * 
    from order_money_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000] 
    return factor_sample_df

}

def Press(bidPrice, bidQty, offerPrice, offerQty){
	wap = (bidPrice[0]*offerQty[0] + offerPrice[0]*bidQty[0])\(bidQty[0]+offerQty[0])
	bidw=(1.0\(bidPrice-wap))
	bidw=bidw\(bidw.rowSum())
	offerw=(1.0\(offerPrice-wap))
	offerw=offerw\(offerw.rowSum())
	press = log((bidQty*bidw).rowSum())-log((offerQty*offerw).rowSum())
	return avg(press)
}

def feat_dos_009(begDate, endDate){

    feature1m_Press_stats = 
    select Press(matrix(bid_px_0, bid_px_1, bid_px_2, bid_px_3, bid_px_4),matrix(bid_sz_0, bid_sz_1, bid_sz_2, bid_sz_3, bid_sz_4),matrix(ask_px_0, ask_px_1, ask_px_2, ask_px_3, ask_px_4),matrix(ask_sz_0, ask_sz_1, ask_sz_2, ask_sz_3, ask_sz_4)) as Press
    from loadTable('dfs://stock_rawdata_sample', 'QuoteL2')
    where settlement_date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6'),
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'],
    (exch_time_ms between 09:15:00.000: 11:30:00.000 )
    or (exch_time_ms between 13:00:00.000: 15:00:00.000)
    group by settlement_date as date,instrument,bar(exch_time_ms,1m) as minTime 

}


def get_intra900050(begDate, endDate){


    order_money_df = select date, minTime, instrument
    from feature1m_Press_stats_009
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument

    factor_sample_df = select * 
    from order_money_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000] 
    return factor_sample_df

}




def get_intra900051(begDate, endDate){


    order_money_df = select date, minTime, instrument, wavgSpread5_adj_avg, (wavgSpread5_adj_avg - mavg(wavgSpread5_adj_avg, 20)) \ mstd(wavgSpread5_adj_avg, 20) as wavgSpread5_adj_avg_norm, 
          mstd(wavgSpread5_adj_avg, 20), mavg(wavgSpread5_adj_avg, 15), volSpread5_adj_avg, wavgSpread5_pr60_avg,
           (wavgSpread5_pr60_avg - mavg(wavgSpread5_pr60_avg, 20)) \ mstd(wavgSpread5_pr60_avg, 20) as wavgSpread5_pr60_avg_norm
    from feature1m_wavgSpread_stats_010
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument

    factor_sample_df = select * 
    from order_money_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000] 
    return factor_sample_df

}


def get_intra900052(begDate, endDate){


    ambiguou_df = select date, minTime, instrument, close, volume
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 

    order_money_df = select date, minTime, instrument, wavgSpread5_adj_avg, wavgSpread5_pr60_avg, 
          wavgSpread_adj_avg, volSpread5_adj_avg
    from feature1m_wavgSpread_stats_010
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 

    
    joined_df = SELECT date, minTime, instrument, close, volume, wavgSpread5_adj_avg, wavgSpread5_pr60_avg, 
          wavgSpread_adj_avg, volSpread5_adj_avg
    FROM order_money_df
    JOIN ambiguou_df
    ON order_money_df.date=ambiguou_df.date and order_money_df.minTime=ambiguou_df.minTime and order_money_df.instrument=ambiguou_df.instrument order by date, instrument
    
    
    corr_ambi_money_df = SELECT date, minTime, instrument, mavgTopN(wavgSpread5_adj_avg, abs(percentChange(close, 5)), 25, 7, false) as TopNspread5_pct, 
          mavgTopN(wavgSpread5_adj_avg, volume, 25, 12) as TopNspread5_lowvol, mavgTopN(volSpread5_adj_avg, volume, 25, 12) as TopNvolSpread5_lowvol, wavgSpread_adj_avg,
          mavgTopN(wavgSpread_adj_avg, volume, 25, 12) as TopNspread_lowvol
    from joined_df context by date, instrument


}






def get_intra900051(begDate, endDate){


    order_money_df = select date, minTime, instrument, wavgSpread5_adj_avg, (wavgSpread5_adj_avg - mavg(wavgSpread5_adj_avg, 20)) \ mstd(wavgSpread5_adj_avg, 20) as wavgSpread5_adj_avg_norm, 
          mstd(wavgSpread5_adj_avg, 20), mavg(wavgSpread5_adj_avg, 15), volSpread5_adj_avg, wavgSpread5_pr60_avg,
           (wavgSpread5_pr60_avg - mavg(wavgSpread5_pr60_avg, 20)) \ mstd(wavgSpread5_pr60_avg, 20) as wavgSpread5_pr60_avg_norm
    from feature1m_wavgSpread_stats_010
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 
    context by date, instrument

    factor_sample_df = select * 
    from order_money_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000] 
    return factor_sample_df

}





def get_intra900052(begDate, endDate){


    ambiguou_df = select date, minTime, instrument, close, volume
    from loadTable('dfs://yj_bar_equity', 'YJ_Minute1') 
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 

    order_money_df = select date, minTime, instrument, wavgSpread5_adj_avg, 
          wavgSpread_adj_avg, volSpread5_adj_avg
    from feature1m_wavgSpread_stats_010
    where date between begDate: endDate,
    instrument.startsWith('EQT_SZSE_0') || instrument.startsWith('EQT_SZSE_3') || instrument.startsWith('EQT_SHSE_6')
    // instrument in ['EQT_SHSE_600000', 'EQT_SZSE_000001', 'EQT_SHSE_600519', 'EQT_SHSE_000725'] 

    
    joined_df = SELECT date, minTime, instrument, close, volume, wavgSpread5_adj_avg, 
          wavgSpread_adj_avg, volSpread5_adj_avg
    FROM order_money_df
    JOIN ambiguou_df
    ON order_money_df.date=ambiguou_df.date and order_money_df.minTime=ambiguou_df.minTime and order_money_df.instrument=ambiguou_df.instrument order by date, instrument
    
    
    corr_ambi_money_df = SELECT date, minTime, instrument, mavgTopN(wavgSpread5_adj_avg, abs(percentChange(close, 5)), 25, 7, false) as TopNspread5_pct, 
          mavgTopN(wavgSpread5_adj_avg, volume, 25, 12) as TopNspread5_lowvol, mavgTopN(volSpread5_adj_avg, volume, 25, 12) as TopNvolSpread5_lowvol,
          mavgTopN(wavgSpread_adj_avg, volume, 25, 12) as TopNspread_lowvol
    from joined_df context by date, instrument



    factor_sample_df = select * 
    from corr_ambi_money_df
    where minTime in [10:00:00.000, 10:30:00.000, 13:30:00.000, 14:00:00.000] 
    return factor_sample_df
    

}





