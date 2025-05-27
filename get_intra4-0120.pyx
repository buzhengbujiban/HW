
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



