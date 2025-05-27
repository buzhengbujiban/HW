import panel
import pandas as pd
import numpy as np
import warnings
import os
from busdates import PortableBusDates
from tqdm import tqdm
from joblib import Parallel, delayed
pd.set_option('display.max_rows', 200)
warnings.filterwarnings("ignore")
pbd = PortableBusDates()

class FeatureGenerator_daily:
    def __init__(self, begDate=None, endDate=None, n_jobs=100):
        self.begDate = begDate if begDate is not None else "20170101"
        self.endDate = endDate if endDate is not None else "20250103"
        self.n_jobs = n_jobs
        self.tradingdays = pbd.get_range(self.begDate, self.endDate)
        self.preface_tradingdays = pbd.get_range(pbd.prev_date(self.begDate, 122), pbd.prev_date(self.begDate, 1)) + self.tradingdays

    # def saving_daily_sdiv(self, dailyadjust_df_univers, save_dir, save_name, daily_I="00:00:00"):
    #     dailyadjust_df_univers = dailyadjust_df_univers.rename(columns={'sid':'S','date':'D'})
    #     dailyadjust_df_univers['I'] = daily_I
    #     for sampleDate, grouped  in dailyadjust_df_univers.groupby('D'):
    #         panel.write(os.path.join(save_dir, save_name) + f"{sampleDate}.sdiv",
    #                     panel.df2sdiv(grouped)
    #                     )

    def saving_daily_sdiv(self, dailyadjust_df_univers, save_dir, save_name, daily_I="00:00:00"):
        dailyadjust_df_univers = dailyadjust_df_univers.rename(columns={'sid':'S','date':'D'})
        dailyadjust_df_univers['I'] = daily_I
        def process_write_daily(sampleDate, grouped):
            panel.write(os.path.join(save_dir, save_name) + f"{sampleDate}.sdiv",
                        panel.df2sdiv(grouped)
                        )
        Parallel(n_jobs=self.n_jobs, verbose=10)(delayed(process_write_daily)(sampleDate, grouped) \
                                    for sampleDate, grouped in tqdm(dailyadjust_df_univers.groupby('D')))

    def load_panel_data_002(self, sampleDate, root_dir='/data/nfs0/chinaEquityData/panel/'):
        """加载每日的面板数据"""
        univers = list(panel.sdiv2df(panel.read(f"/data/beer2/wensheng/univers_nost/morning/daily/terms.{sampleDate}.sdiv")).set_index('sid').index )
        daily_date = panel.sdiv2df(panel.read(f"{root_dir}daily/daily-{sampleDate}.sdiv"))
        dailyadjust_date = panel.sdiv2df(panel.read(f"{root_dir}daily-adjusted-2007/morning/daily-adjusted-{sampleDate}.sdiv"))
        barra_date = panel.sdiv2df(panel.read(f"{root_dir}barra/stats-{sampleDate}.sdiv"))
        cap_date = panel.sdiv2df(panel.read(f"{root_dir}cap/cap-{sampleDate}.sdiv"))
        return daily_date[daily_date['sid'].isin(univers)], dailyadjust_date[dailyadjust_date['sid'].isin(univers)], barra_date[barra_date['sid'].isin(univers)], cap_date[cap_date['sid'].isin(univers)]

    def daily_feature_002(self, pretradingdays=None, fd=False):
        """并行处理每日数据并生成特征"""
        ### do not modify for fd check
        preface_tradingdays = pretradingdays if pretradingdays is not None else self.preface_tradingdays
        if fd:
            preface_tradingdays = [date for date in preface_tradingdays if date<='20170105']
            root_dir            = '/data/beer2/wensheng/fd_check/'
        else:
            root_dir            = '/data/nfs0/chinaEquityData/panel/'

        # 加载面板数据
        panel_results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.load_panel_data_002)(sampleDate, root_dir) for sampleDate in tqdm(preface_tradingdays)
        )

        ### write your expression here 
        all_daily_data_df   = pd.concat([result[0] for result in panel_results], ignore_index=True).set_index(['sid', 'date'])
        dailyadjust_data_df = pd.concat([result[1] for result in panel_results], ignore_index=True).set_index(['sid', 'date'])
        allbarra_data_df    = pd.concat([result[2] for result in panel_results], ignore_index=True).set_index(['sid', 'date'])
        capall_data_df      = pd.concat([result[3] for result in panel_results], ignore_index=True).set_index(['sid', 'date'])

        # 计算特征
        all_daily_data_df['close']         = all_daily_data_df['close'] * dailyadjust_data_df['adj.factor.locf']
        all_daily_data_df['dclose']        = all_daily_data_df.groupby('sid', group_keys=False)['close'].shift()
        all_daily_data_df[all_daily_data_df['dclose']<1e-7]['dclose'] = np.nan
        all_daily_data_df['pct_df']        = all_daily_data_df['close'] / all_daily_data_df['dclose'] - 1.

        close_mean                         = all_daily_data_df.groupby('sid', group_keys=False)['close'].rolling(20).mean().shift().droplevel(level=0)
        close_std                          = all_daily_data_df.groupby('sid', group_keys=False)['close'].rolling(20).std().shift().droplevel(level=0)

        ret_mean                           = all_daily_data_df.groupby('sid', group_keys=False)['pct_df'].rolling(20).mean().shift().droplevel(level=0)
        ret_std                            = all_daily_data_df.groupby('sid', group_keys=False)['pct_df'].rolling(20).std().shift().droplevel(level=0)
        
        prs_zsc   = (all_daily_data_df.groupby('sid', group_keys=False)['close'].shift() - close_mean) / close_std
        ret10_zsc = (all_daily_data_df.groupby('sid', group_keys=False)['pct_df'].shift() - ret_mean) / ret_std


        result_df = pd.DataFrame({
            'ret10_zsc': ret10_zsc,
            'prs_zsc': prs_zsc,
            }
        )
        # 对特征进行标准化
        
        result_df_rank = result_df.groupby('date', group_keys=False).apply(lambda seq: seq.rank() / pd.notnull(seq).sum()).reset_index()
        return result_df_rank[result_df_rank['date']>=self.begDate].set_index(['sid','date'])
     

    def load_panel_data_003(self, sampleDate, root_dir='/data/nfs0/chinaEquityData/panel/'):
        """加载每日的面板数据"""
        univers = list(panel.sdiv2df(panel.read(f"/data/beer2/wensheng/univers_nost/morning/daily/terms.{sampleDate}.sdiv")).set_index('sid').index )
        daily_rollstat = panel.sdiv2df(panel.read(f"{root_dir}daily-rollstats/daily-rollstats-{sampleDate}.sdiv"))

        return daily_rollstat[daily_rollstat['sid'].isin(univers)]
    
    def daily_feature_003(self, pretradingdays=None, fd=False):
        """并行处理每日数据并生成特征"""
        ### do not modify for fd check
        preface_tradingdays = pretradingdays if pretradingdays is not None else self.preface_tradingdays
        if fd:
            preface_tradingdays = [date for date in preface_tradingdays if date<='20170105']
            root_dir            = '/data/beer2/wensheng/fd_check/'
        else:
            root_dir            = '/data/nfs0/chinaEquityData/panel/'

        # 加载面板数据
        panel_results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.load_panel_data_003)(sampleDate, root_dir) for sampleDate in tqdm(preface_tradingdays)
        )

        ### write your expression here 
        all_daily_rollstat   = pd.concat(panel_results, ignore_index=True).set_index(['sid', 'date'])
        # dailyadjust_data_df = pd.concat([result[1] for result in panel_results], ignore_index=True).set_index(['sid', 'date'])
        # allbarra_data_df    = pd.concat([result[2] for result in panel_results], ignore_index=True).set_index(['sid', 'date'])
        # capall_data_df      = pd.concat([result[3] for result in panel_results], ignore_index=True).set_index(['sid', 'date'])

        # 计算特征
        # all_daily_data_df['close']         = all_daily_data_df['close'] * dailyadjust_data_df['adj.factor.locf']
        # all_daily_data_df['dclose']        = all_daily_data_df.groupby('sid', group_keys=False)['close'].shift()
        # all_daily_data_df[all_daily_data_df['dclose']<1e-7]['dclose'] = np.nan
        # all_daily_data_df['pct_df']        = all_daily_data_df['close'] / all_daily_data_df['dclose'] - 1.
        

        # result_df = pd.DataFrame({
        #     'ret10_zsc': ret10_zsc,
        #     'prs_zsc': prs_zsc,
        #     }
        # )


        # 对特征进行标准化
        result_df = all_daily_rollstat[['avg_spread.rw22', 'avg_spread_bps.rw22']].groupby('date', group_keys=False).shift()
        
        result_df_rank = result_df.groupby('date', group_keys=False).apply(lambda seq: seq.rank() / pd.notnull(seq).sum()).reset_index()
        return result_df_rank[result_df_rank['date']>=self.begDate].set_index(['sid','date'])
     

