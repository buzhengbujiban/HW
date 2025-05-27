import panel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from busdates import PortableBusDates
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
pd.set_option('display.max_rows', 200)
warnings.filterwarnings("ignore")
pbd = PortableBusDates()

class FeatureGenerator_1min:
    def __init__(self, begDate=None, endData=None, n_jobs=None, fd_check_dir=None):
        self.begDate    =  begDate if begDate is not None else "20170101"
        self.endData    =  endData if endData is not None else "20250103"
        self.tradingdays           = pbd.get_range(self.begDate, self.endData)
        self.preface_tradingdays   = pbd.get_range(pbd.prev_date(self.begDate, 122), pbd.prev_date(self.begDate, 1)) + self.tradingdays
        self.n_jobs = n_jobs if n_jobs is not None else 100
        self.fd_check_dir = fd_check_dir if fd_check_dir is not None else '/data/beer2/wensheng/fd_check/'


    def plot_result_df_cumclose(self, results_df):
        # 确保 date 是 datetime 类型
        results_df['date'] = pd.to_datetime(results_df['date'], format='%Y%m%d')
        results_df['month'] = results_df['date'].dt.to_period('M')

        # 计算统计数据
        monthly_stats = results_df.groupby('month').apply(lambda df: pd.Series({
            'precise_close': df.loc[df['condition'], 'return_close'].gt(0).mean(),
            'precise_5min_1dcont': df.loc[df['condition'], 'return_5min_1dcont'].gt(0).mean(),
            'recall_close': df.loc[df['condition'] & df['return_close'].gt(0)].shape[0] / max(df['return_close'].gt(0).sum(), 1),
            'recall_5min_1dcont': df.loc[df['condition'] & df['return_5min_1dcont'].gt(0)].shape[0] / max(df['return_5min_1dcont'].gt(0).sum(), 1),
            'pnlret': df.loc[df['condition'], 'return_close'].mean()
        })).reset_index()

        # 转换 month 为字符串，便于绘图
        monthly_stats['month'] = monthly_stats['month'].astype(str)
        monthly_stats.set_index('month', inplace=True)

        # 计算 cumulative pnlret
        monthly_stats['cumulative_pnlret'] = monthly_stats['pnlret'].cumsum()

        # 绘图
        fig, ax1 = plt.subplots(figsize=(12,6))

        # 设置 bar 宽度
        bar_width = 0.05
        x = np.arange(len(monthly_stats))

        # 左轴：precise 和 recall（柱状图，调整宽度）
        ax1.bar(x - bar_width/2, monthly_stats['precise_close'], width=bar_width, color='blue', alpha=0.6, label='Precise close')
        ax1.bar(x + bar_width/2, monthly_stats['precise_5min_1dcont'], width=bar_width, color='green', alpha=0.6, label='Precise 5min_1dcont')

        ax1.plot(x, monthly_stats['recall_close'], color='blue', marker='o', linestyle='dashed', label='Recall close')
        ax1.plot(x, monthly_stats['recall_5min_1dcont'], color='green', marker='o', linestyle='dashed', label='Recall 5min_1dcont')

        ax1.set_ylabel('Precise / Recall')
        ax1.set_xlabel('Month')
        ax1.set_xticks(x)
        ax1.set_xticklabels(monthly_stats.index, rotation=45)
        ax1.legend(loc='upper left')

        # 右轴：累计 pnlret（折线图）
        ax2 = ax1.twinx()
        ax2.plot(x, monthly_stats['cumulative_pnlret'], color='red', marker='o', linestyle='solid', label='Cumulative PnlRet')
        ax2.set_ylabel('Cumulative PnlRet')
        ax2.legend(loc='upper right')

        plt.title("Monthly Precise, Recall and Cumulative PnlRet")
        plt.grid()
        plt.show()


    def plot_result_df_1dcont(self, results_df, save_name=None):
        # 确保 date 是 datetime 类型
        results_df['date'] = pd.to_datetime(results_df['date'], format='%Y%m%d')
        results_df['month'] = results_df['date'].dt.to_period('M')

        # 计算统计数据
        monthly_stats = results_df.groupby('month').apply(lambda df: pd.Series({
            'precise_close': df.loc[df['condition'], 'return_close'].gt(0).mean(),
            'precise_5min_1dcont': df.loc[df['condition'], 'return_5min_1dcont'].gt(0).mean(),
            'recall_close': df.loc[df['condition'] & df['return_close'].gt(0)].shape[0] / max(df['return_close'].gt(0).sum(), 1),
            'recall_5min_1dcont': df.loc[df['condition'] & df['return_5min_1dcont'].gt(0)].shape[0] / max(df['return_5min_1dcont'].gt(0).sum(), 1),
            'pnlret': df.loc[df['condition'], 'return_5min_1dcont'].mean()
        })).reset_index()

        # 转换 month 为字符串，便于绘图
        monthly_stats['month'] = monthly_stats['month'].astype(str)
        monthly_stats.set_index('month', inplace=True)

        # 计算 cumulative pnlret
        monthly_stats['cumulative_pnlret'] = monthly_stats['pnlret'].cumsum()

        # 绘图
        fig, ax1 = plt.subplots(figsize=(12,6))

        # 设置 bar 宽度
        bar_width = 0.05
        x = np.arange(len(monthly_stats))

        # 左轴：precise 和 recall（柱状图，调整宽度）
        ax1.bar(x - bar_width/2, monthly_stats['precise_close'], width=bar_width, color='blue', alpha=0.6, label='Precise close')
        ax1.bar(x + bar_width/2, monthly_stats['precise_5min_1dcont'], width=bar_width, color='green', alpha=0.6, label='Precise 5min_1dcont')

        ax1.plot(x, monthly_stats['recall_close'], color='blue', marker='o', linestyle='dashed', label='Recall close')
        ax1.plot(x, monthly_stats['recall_5min_1dcont'], color='green', marker='o', linestyle='dashed', label='Recall 5min_1dcont')

        ax1.set_ylabel('Precise / Recall')
        ax1.set_xlabel('Month')
        ax1.set_xticks(x)
        ax1.set_xticklabels(monthly_stats.index, rotation=45)
        ax1.legend(loc='upper left')

        # 右轴：累计 pnlret（折线图）
        ax2 = ax1.twinx()
        ax2.plot(x, monthly_stats['cumulative_pnlret'], color='red', marker='o', linestyle='solid', label='Cumulative PnlRet')
        ax2.set_ylabel('Cumulative PnlRet')
        ax2.legend(loc='upper right')

        plt.title("Monthly Precise, Recall and Cumulative PnlRet")
        plt.grid()
        if save_name is not None:
            plt.savefig(f'/data/beer2/wensheng/plot/{save_name}.png')
        plt.show()



    def saving_1min_sdiv(self, feature_1min_df, save_dir, save_name):
        feature_1min_df                = feature_1min_df.rename(columns={'sid':'S','date':'D','time':'I'})
        def process_write(sampleDate, grouped):
            panel.write(os.path.join(save_dir, save_name) + f"{sampleDate}.sdiv",
                        panel.df2sdiv(grouped)
                        )
        Parallel(n_jobs=self.n_jobs, verbose=10)(delayed(process_write)(sampleDate, grouped) for sampleDate, grouped in tqdm(feature_1min_df.groupby('D')))

    def Omin_feature_013(self, sampleDate, root_dir = '/data/beer1/data/chinaEquityData/panel/', univers_filter=None):
        sample_1min_df = panel.sdiv2df(
            panel.read(f"{root_dir}1min-lts/1min-{sampleDate}.sdiv").sel(V=slice("close","auctionVolume"), I = slice("09:24:00", None))
        )

        if univers_filter is not None:
            univers        = univers_filter.set_index('date').loc[sampleDate]['sid']
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]
        else:
            univers        = panel.sdiv2df(panel.read(f"/data/beer2/wensheng/univers_nost/morning/daily/terms.{sampleDate}.sdiv")).set_index('sid').index 
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]
        
        sample_1min_df['close'] = np.where(sample_1min_df['close'].isna() & ((sample_1min_df['time']<"09:31:00") | (sample_1min_df['time']>"14:57:00")),
                                            sample_1min_df['auctionPrice'], sample_1min_df['close'])
        sample_1min_df['close'] = sample_1min_df['close'].replace(0.0, np.nan)
        sample_1min_df.set_index(['date','sid','time'], inplace=True)
        sample_1min_df['close'] = sample_1min_df[['close']].groupby('sid', group_keys=False).ffill()

        sample_1min_df['dclose']       = sample_1min_df[['close']].groupby('sid', group_keys=False).shift()
        sample_1min_df[sample_1min_df['dclose']<=1e-7]['dclose'] = np.nan
        sample_1min_df['pct_df']       = sample_1min_df['close'] / sample_1min_df['dclose'] - 1.     
        sample_1min_df['pct_df_down']  = np.where(sample_1min_df['pct_df'] < 0, sample_1min_df['pct_df'], np.nan)

        sample_1min_df['cumpctmean']  = sample_1min_df[['pct_df']].groupby('sid', group_keys=False).rolling(2500, min_periods=1).mean().droplevel(level=0)
        sample_1min_df['cumpctstd']  = sample_1min_df[['pct_df']].groupby('sid', group_keys=False).rolling(2500, min_periods=1).std().droplevel(level=0)
        sample_1min_df[sample_1min_df['cumpctstd']<=1e-7] = np.nan
        sample_1min_df['cumpctzsc']  = (sample_1min_df['pct_df'] - sample_1min_df['cumpctmean']) / sample_1min_df['cumpctstd'] 


        sample_1min_df['cumdown'] = sample_1min_df.groupby('sid', group_keys=False)['pct_df_down'].rolling(2500, min_periods=1).sum().droplevel(level=0)
        sample_1min_df['cummean'] = sample_1min_df[['dclose']].groupby('sid', group_keys=False).rolling(2500, min_periods=1).mean().droplevel(level=0)
        sample_1min_df['cummax'] = sample_1min_df[['dclose']].groupby('sid', group_keys=False).rolling(2500, min_periods=1).max().droplevel(level=0)
        sample_1min_df['dclose_from_open_meanratio'] = sample_1min_df['dclose'] / sample_1min_df['cummean']
        sample_1min_df['dclose_from_open_maxratio'] = sample_1min_df['dclose'] / sample_1min_df['cummax']
        sample_1min_df = sample_1min_df.groupby('sid', group_keys=False).shift()
        # feature_1min_df.append(sample_1min_df.reset_index()[['time', 'sid', 'date', 'dclose_from_open_meanratio', 'dclose_from_open_maxratio', 'dvol_from_open_meanratio', 
        #                                                      'dvol_from_open_maxratio',
        #                                      'cum_volzsc', 'cumpctzsc', 'cumdown','cumstd_vol']])
        return sample_1min_df.reset_index()[['dclose_from_open_meanratio', 'dclose_from_open_maxratio', 
                                            'cumpctzsc', 'cumdown','cumpctstd', 'time', 'sid', 'date']]


    def OminProcess_013(self, date_range=None, univers_filter=None, fd = False, root_dir='/data/beer1/data/chinaEquityData/panel/'):
        date_range = date_range if date_range is not None else self.tradingdays
        if fd:
            root_dir = self.fd_check_dir 
            date_range = [date for date in date_range if date <='20170105']
        Process_1min_df = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.Omin_feature_013)(sampleDate, root_dir, univers_filter) for sampleDate in tqdm(date_range)
        )
        feature_1min_df = pd.concat(Process_1min_df, ignore_index=True)
        feature_1min_df.set_index(['sid','date', 'time'], inplace=True)
        return feature_1min_df

    def agg1min_feature_001(self, sampleDate, root_dir='/data/beer1/data/chinaEquityData/panel/', univers_filter=None):
        # 读取数据
        ## intraday_features
        if sampleDate<self.begDate:
            return None
        sample_1min_df = panel.sdiv2df(
            panel.read(f"{root_dir}1min-lts/1min-{pbd.prev_date(sampleDate)}.sdiv")
        )
        if univers_filter is not None:
            univers        = univers_filter.set_index('date').loc[sampleDate]['sid']
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]
        else:
            univers        = panel.sdiv2df(panel.read(f"/data/beer2/wensheng/univers_nost/morning/daily/terms.{sampleDate}.sdiv")).set_index('sid').index 
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]
        

        sample_1min_df.set_index(['sid','time'], inplace=True)
        sample_1min_df['pct_speed1min']  = sample_1min_df.groupby('sid', group_keys=False)['close'].pct_change(periods=1, fill_method=None) * 100  
        sample_1min_df['num_none0_pct1'] = sample_1min_df['pct_speed1min'] != 0.0

        sample_1min_df['spread'] = sample_1min_df['close.ask.avg'] - sample_1min_df['close.bid.avg']
        sample_1min_df['spread'][sample_1min_df['spread'].abs() > 5.0] = np.nan
        
        agg_sample1min = sample_1min_df.groupby('sid').agg(
            num_none0_pct1=('num_none0_pct1', 'sum'),
            spread_avg=('spread', 'mean')
        ).reset_index()
        
        agg_sample1min['spread_avg'] = agg_sample1min['spread_avg'].rank() / pd.notnull(agg_sample1min['spread_avg']).sum()
        agg_sample1min['num_none0_pct1'] = agg_sample1min['num_none0_pct1'].rank() / pd.notnull(agg_sample1min['num_none0_pct1']).sum()
        agg_sample1min['date'] = sampleDate
        return agg_sample1min

 
    def agg1minProcess_001(self, date_range=None, univers_filter=None, fd = False, root_dir='/data/beer1/data/chinaEquityData/panel/'):
        date_range = date_range if date_range is not None else self.tradingdays
        if fd:
            root_dir = self.fd_check_dir 
            date_range = [date for date in date_range if date <='20170105']
        Process_1min_df = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.agg1min_feature_001)(sampleDate, root_dir, univers_filter) for sampleDate in tqdm(date_range)
        )
        feature_1min_df = pd.concat(Process_1min_df, ignore_index=True)
        feature_1min_df.set_index(['sid','date'], inplace=True)
        return feature_1min_df

    def calculate_corr1(self, group):
        return group['buyQty'].rolling(window=5, min_periods=4).corr(group['time_map'])

    def calculate_corr2(self, group):
        return group['close'].rolling(window=5, min_periods=4).corr(group['time_map'])

    def calculate_corr3(self, group):
        return group['sellQty'].rolling(window=5, min_periods=4).corr(group['time_map'])

    def Omin_feature_023(self, sampleDate, root_dir = '/data/beer1/data/chinaEquityData/panel/', univers_filter=None):
        sample_1min_df = panel.sdiv2df(
            panel.read(f"{root_dir}1min-lts/1min-{sampleDate}.sdiv").sel(V=slice("close","close.ask.avg"), I = slice("09:31:00", None))
        )
        sample_1min_df_prev = panel.sdiv2df(
            panel.read(f"{root_dir}1min-lts/1min-{pbd.prev_date(sampleDate)}.sdiv").sel(V=slice("close","close.ask.avg"), I = slice("14:00:00", "14:57:00"))
        )
        sample_1min_df_prev['time'] = '!' + sample_1min_df_prev['time']
        sample_1min_df_prev['date'] = sampleDate
        sample_1min_df = pd.concat((sample_1min_df_prev, sample_1min_df), ignore_index=True)
        if univers_filter is not None:
            univers        = univers_filter.set_index('date').loc[sampleDate]['sid']
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]
        else:
            univers        = panel.sdiv2df(panel.read(f"/data/beer2/wensheng/univers_nost/morning/daily/terms.{sampleDate}.sdiv")).set_index('sid').index 
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]
        


        sample_1min_df['close'] = np.where(sample_1min_df['close'].isna() & ((sample_1min_df['time']<"09:31:00") | (sample_1min_df['time']>"14:57:00")),
                                            sample_1min_df['auctionPrice'], sample_1min_df['close'])
        sample_1min_df['close'] = sample_1min_df['close'].replace(0.0, np.nan)

        t_slice = sample_1min_df[sample_1min_df['sid']==sample_1min_df['sid'].iloc[0]]
        t_slice.index = pd.RangeIndex(start=10000, stop=len(t_slice) + 10000)
        time_map = dict(t_slice.reset_index().set_index(['time'])['index'])
        sample_1min_df['time_map'] = sample_1min_df['time'].map(lambda x:time_map[x])
        sample_1min_df = sample_1min_df.set_index(['date','sid','time']).sort_index()

        sample_1min_df['close'] = sample_1min_df[['close']].groupby('sid', group_keys=False).ffill()
        sample_1min_df['cro_buyQty'] = sample_1min_df[['buyQty','time_map']].groupby('sid', group_keys=False).apply(self.calculate_corr1)
        sample_1min_df['cro_close']  = sample_1min_df[['close','time_map']].groupby('sid', group_keys=False).apply(self.calculate_corr2).replace(np.inf, np.nan).replace(-np.inf, np.nan)
        sample_1min_df['cro_sellQty']= sample_1min_df[['sellQty','time_map']].groupby('sid', group_keys=False).apply(self.calculate_corr3)

        sample_1min_df = sample_1min_df.groupby('sid', group_keys=False).shift()
        sample_1min_df_return = sample_1min_df.reset_index()[['cro_sellQty', 'cro_close', 
                                            'cro_buyQty', 'time', 'sid', 'date']]
        return sample_1min_df_return[sample_1min_df_return['time']>='09:31:00']

    def OminProcess_023(self, date_range=None, univers_filter=None, fd = False, root_dir='/data/beer1/data/chinaEquityData/panel/'):
        date_range = date_range if date_range is not None else self.tradingdays
        if fd:
            root_dir = self.fd_check_dir 
            date_range = [date for date in date_range if date <='20170105']
        Process_1min_df = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.Omin_feature_023)(sampleDate, root_dir, univers_filter) for sampleDate in tqdm(date_range)
        )
        feature_1min_df = pd.concat(Process_1min_df, ignore_index=True)
        feature_1min_df.set_index(['sid','date', 'time'], inplace=True)
        return feature_1min_df




    def Omin_feature_024(self, sampleDate, root_dir = '/data/beer2/data/CA.LTS/CNEQ/Panels/I/fq1m/', univers_filter=None):
        sample_1min_df_act = panel.sdiv2df(
            panel.read(f"{root_dir}add_act/terms.{sampleDate}.sdiv")
        ).set_index(['sid','date','time'])
        sample_1min_df_ppq2 = panel.sdiv2df(
            panel.read(f"{root_dir}ppq2/terms.{sampleDate}.sdiv")
        ).set_index(['sid','date','time'])
        # sample_1min_df_prev = panel.sdiv2df(
        #     panel.read(f"{root_dir}1min-lts/1min-{pbd.prev_date(sampleDate)}.sdiv").sel(V=slice("close","close.ask.avg"), I = slice("14:00:00", "14:57:00"))
        # )
        # sample_1min_df_prev['time'] = '!' + sample_1min_df_prev['time']
        # sample_1min_df_prev['date'] = sampleDate
        sample_1min_df = pd.concat((sample_1min_df_act, sample_1min_df_ppq2), axis=1).reset_index()
        if univers_filter is not None:
            univers        = univers_filter.set_index('date').loc[sampleDate]['sid']
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]
        else:
            univers        = panel.sdiv2df(panel.read(f"/data/beer2/wensheng/univers_nost/morning/daily/terms.{sampleDate}.sdiv")).set_index('sid').index 
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]

        sample_1min_df.set_index(['sid','date','time'], inplace=True)
        sample_1min_df['addact_BSall_pq']    = sample_1min_df[['act.B.big0.pq.p1', 'act.S.big0.pq.p1', 'add.B.big0.pq.p1', 'add.S.big0.pq.p1',
                                                                'act.B.med0.pq.p1', 'act.S.med0.pq.p1', 'add.B.med0.pq.p1', 'add.S.med0.pq.p1',
                                                                'act.B.sml0.pq.p1', 'act.S.sml0.pq.p1', 'add.B.sml0.pq.p1', 'add.S.sml0.pq.p1']].sum(axis=1)

        sample_1min_df['act.B.big0.pq.ratio'] = sample_1min_df['act.B.big0.pq.p1'] / sample_1min_df['addact_BSall_pq']
        sample_1min_df['act.S.big0.pq.ratio'] = sample_1min_df['act.S.big0.pq.p1'] / sample_1min_df['addact_BSall_pq']

        sample_1min_df['act.S.pq.ratio']      = sample_1min_df[['act.S.big0.pq.p1', 'act.S.med0.pq.p1', 'act.S.sml0.pq.p1']].sum(axis=1) / sample_1min_df['addact_BSall_pq']
        sample_1min_df['act.B.pq.ratio']      = sample_1min_df[['act.B.big0.pq.p1', 'act.B.med0.pq.p1', 'act.B.sml0.pq.p1']].sum(axis=1) / sample_1min_df['addact_BSall_pq']

        sample_1min_df['ppq2_BStrd_pq']       = sample_1min_df[['ppq2.B.trd_big0.pq.p1','ppq2.S.trd_big0.pq.p1','ppq2.B.trd_med0.pq.p1','ppq2.S.trd_med0.pq.p1','ppq2.B.trd_sml0.pq.p1','ppq2.S.trd_sml0.pq.p1']].sum(axis=1)
        sample_1min_df['ppq2.B.trd_big0.pq.ratio'] = (sample_1min_df['ppq2.B.trd_big0.pq.p1']) / sample_1min_df['ppq2_BStrd_pq']
        sample_1min_df['ppq2.S.trd_big0.pq.ratio'] = (sample_1min_df['ppq2.S.trd_big0.pq.p1']) / sample_1min_df['ppq2_BStrd_pq']
        
        sample_1min_df['ppq2_BSadd_pq']                  = sample_1min_df[['ppq2.B.add_big0.pq.p1','ppq2.S.add_big0.pq.p1','ppq2.B.add_med0.pq.p1','ppq2.S.add_med0.pq.p1','ppq2.B.add_sml0.pq.p1','ppq2.S.add_sml0.pq.p1']].sum(axis=1)
        sample_1min_df['ppq2.trd_add.ratio']             = sample_1min_df['ppq2_BStrd_pq'] /  sample_1min_df['ppq2_BSadd_pq']
        
        

        sample_1min_df['ppq2.B.trd_bigmed0.pq.ratio'] = (sample_1min_df['ppq2.B.trd_big0.pq.p1'] + sample_1min_df['ppq2.B.trd_med0.pq.p1']) / sample_1min_df['ppq2_BStrd_pq']
        sample_1min_df['ppq2.S.trd_bigmed0.pq.ratio'] = (sample_1min_df['ppq2.S.trd_big0.pq.p1'] + sample_1min_df['ppq2.S.trd_med0.pq.p1']) / sample_1min_df['ppq2_BStrd_pq']
        sample_1min_df['ppq2.trd_BbigdifsmlS0.pq.ratio'] = (sample_1min_df['ppq2.B.trd_big0.pq.p1'] - sample_1min_df['ppq2.S.trd_sml0.pq.p1']) / sample_1min_df['ppq2_BStrd_pq']
        sample_1min_df['ppq2.trd_SbigdifsmlB0.pq.ratio'] = (sample_1min_df['ppq2.S.trd_big0.pq.p1'] - sample_1min_df['ppq2.B.trd_sml0.pq.p1']) / sample_1min_df['ppq2_BStrd_pq']

        sample_1min_df = sample_1min_df.groupby('sid', group_keys=False).shift()
        sample_1min_df_return = sample_1min_df.reset_index()[['act.B.big0.pq.ratio', 'act.S.big0.pq.ratio', 'act.S.pq.ratio', 'act.B.pq.ratio', 'ppq2.B.trd_big0.pq.ratio', 'ppq2.S.trd_big0.pq.ratio', \
                                                              'ppq2.trd_add.ratio', 'ppq2.B.trd_bigmed0.pq.ratio', 'ppq2.S.trd_bigmed0.pq.ratio', 'ppq2.trd_BbigdifsmlS0.pq.ratio', 'ppq2.trd_SbigdifsmlB0.pq.ratio', 'time', 'sid', 'date']]
        return sample_1min_df_return

    def OminProcess_024(self, date_range=None, univers_filter=None, fd = False, root_dir='/data/beer2/data/CA.LTS/CNEQ/Panels/I/fq1m/'):
        date_range = date_range if date_range is not None else self.tradingdays
        if fd:
            root_dir = self.fd_check_dir 
            date_range = [date for date in date_range if date <='20170105']
        Process_1min_df = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.Omin_feature_024)(sampleDate, root_dir, univers_filter) for sampleDate in tqdm(date_range)
        )
        feature_1min_df = pd.concat(Process_1min_df, ignore_index=True)
        feature_1min_df.set_index(['sid','date', 'time'], inplace=True)
        return feature_1min_df

    def custom_diff(self, x):
        result = x.diff()
        result.iloc[0] = x.iloc[0]  # 第一行保留原值
        return result

    def Omin_feature_025(self, sampleDate, root_dir = '/data/beer2/data/CA.LTS/CNEQ/Panels/I/fq1m/', univers_filter=None):
        sample_1min_df_act = panel.sdiv2df(
            panel.read(f"{root_dir}add_act/terms.{sampleDate}.sdiv")
        ).set_index(['sid','date','time'])
        sample_1min_df_ppq2 = panel.sdiv2df(
            panel.read(f"{root_dir}ppq2/terms.{sampleDate}.sdiv")
        ).set_index(['sid','date','time'])
        # sample_1min_df_prev = panel.sdiv2df(
        #     panel.read(f"{root_dir}1min-lts/1min-{pbd.prev_date(sampleDate)}.sdiv").sel(V=slice("close","close.ask.avg"), I = slice("14:00:00", "14:57:00"))
        # )
        # sample_1min_df_prev['time'] = '!' + sample_1min_df_prev['time']
        # sample_1min_df_prev['date'] = sampleDate
        sample_1min_df = pd.concat((sample_1min_df_act, sample_1min_df_ppq2), axis=1).reset_index()
        if univers_filter is not None:
            univers        = univers_filter.set_index('date').loc[sampleDate]['sid']
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]
        else:
            univers        = panel.sdiv2df(panel.read(f"/data/beer2/wensheng/univers_nost/morning/daily/terms.{sampleDate}.sdiv")).set_index('sid').index 
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]

        sample_1min_df.set_index(['sid','date','time'], inplace=True)
        sample_1min_df['addact_BSall_pq']    = sample_1min_df[['act.B.big0.pq.p1', 'act.S.big0.pq.p1', 'add.B.big0.pq.p1', 'add.S.big0.pq.p1',
                                                                'act.B.med0.pq.p1', 'act.S.med0.pq.p1', 'add.B.med0.pq.p1', 'add.S.med0.pq.p1',
                                                                'act.B.sml0.pq.p1', 'act.S.sml0.pq.p1', 'add.B.sml0.pq.p1', 'add.S.sml0.pq.p1']].sum(axis=1)

        sample_1min_df['act.B.big0.pq.ratio'] = sample_1min_df['act.B.big0.pq.p1'] / sample_1min_df['addact_BSall_pq']
        sample_1min_df['act.S.big0.pq.ratio'] = sample_1min_df['act.S.big0.pq.p1'] / sample_1min_df['addact_BSall_pq']

        sample_1min_df['act.S.pq.ratio']      = sample_1min_df[['act.S.big0.pq.p1', 'act.S.med0.pq.p1', 'act.S.sml0.pq.p1']].sum(axis=1) / sample_1min_df['addact_BSall_pq']
        sample_1min_df['act.B.pq.ratio']      = sample_1min_df[['act.B.big0.pq.p1', 'act.B.med0.pq.p1', 'act.B.sml0.pq.p1']].sum(axis=1) / sample_1min_df['addact_BSall_pq']

        sample_1min_df['ppq2_BStrd_pq']       = sample_1min_df[['ppq2.B.trd_big0.pq.p1','ppq2.S.trd_big0.pq.p1','ppq2.B.trd_med0.pq.p1','ppq2.S.trd_med0.pq.p1','ppq2.B.trd_sml0.pq.p1','ppq2.S.trd_sml0.pq.p1']].sum(axis=1)
        sample_1min_df['ppq2.B.trd_big0.pq.ratio'] = (sample_1min_df['ppq2.B.trd_big0.pq.p1']) / sample_1min_df['ppq2_BStrd_pq']
        sample_1min_df['ppq2.S.trd_big0.pq.ratio'] = (sample_1min_df['ppq2.S.trd_big0.pq.p1']) / sample_1min_df['ppq2_BStrd_pq']
        
        sample_1min_df['ppq2_BSadd_pq']                  = sample_1min_df[['ppq2.B.add_big0.pq.p1','ppq2.S.add_big0.pq.p1','ppq2.B.add_med0.pq.p1','ppq2.S.add_med0.pq.p1','ppq2.B.add_sml0.pq.p1','ppq2.S.add_sml0.pq.p1']].sum(axis=1)
        sample_1min_df['ppq2.trd_add.ratio']             = sample_1min_df['ppq2_BStrd_pq'] /  sample_1min_df['ppq2_BSadd_pq']
        

        sample_1min_df = sample_1min_df.groupby('sid', group_keys=False).shift().replace(np.inf, np.nan).replace(-np.inf, np.nan)
        sample_1min_df_return = sample_1min_df.reset_index()[['act.B.big0.pq.ratio', 'act.S.big0.pq.ratio', 'act.S.pq.ratio', 'act.B.pq.ratio', 'ppq2.B.trd_big0.pq.ratio', 'ppq2.S.trd_big0.pq.ratio', \
                                                              'ppq2.trd_add.ratio', 'time', 'sid', 'date']]
        return sample_1min_df_return

    def OminProcess_025(self, date_range=None, univers_filter=None, fd = False, root_dir='/data/beer2/data/CA.LTS/CNEQ/Panels/I/fq1m/'):
        date_range = date_range if date_range is not None else self.tradingdays
        if fd:
            root_dir = self.fd_check_dir 
            date_range = [date for date in date_range if date <='20170105']
        Process_1min_df = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.Omin_feature_025)(sampleDate, root_dir, univers_filter) for sampleDate in tqdm(date_range)
        )
        feature_1min_df = pd.concat(Process_1min_df, ignore_index=True)
        feature_1min_df.set_index(['sid','date', 'time'], inplace=True)
        return feature_1min_df


    def Omin_feature_026(self, sampleDate, root_dir = '/data/beer1/data/chinaEquityData/panel/', univers_filter=None):

        sample_1min_df = panel.sdiv2df(
            panel.read(f"{root_dir}1min-lts/1min-{sampleDate}.sdiv").sel(V=slice("close","close.ask.avg"), I = slice("09:31:00", None))
        )
        sample_1min_df_prev = panel.sdiv2df(
            panel.read(f"{root_dir}1min-lts/1min-{pbd.prev_date(sampleDate)}.sdiv").sel(V=slice("close","close.ask.avg"), I = slice("14:00:00", "14:57:00"))
        )
        
        sample_1min_df['close'] = sample_1min_df['close'].replace(0.0, np.nan)
        sample_1min_df.set_index(['date','sid','time'], inplace=True)
        sample_1min_df['close'] = sample_1min_df[['close']].groupby('sid', group_keys=False).ffill()
        sample_1min_df['volume'] = sample_1min_df[['volumeTotal']].groupby('sid', group_keys=False).apply(self.custom_diff)
        
        sample_1min_df_prev['close'] = sample_1min_df_prev['close'].replace(0.0, np.nan)
        sample_1min_df_prev.set_index(['date','sid','time'], inplace=True)
        sample_1min_df_prev['close'] = sample_1min_df_prev[['close']].groupby('sid', group_keys=False).ffill()
        sample_1min_df_prev['volume'] = sample_1min_df_prev[['volumeTotal']].groupby('sid', group_keys=False).diff()

        sample_1min_df_prev.reset_index(inplace=True)
        sample_1min_df.reset_index(inplace=True)
        sample_1min_df_prev['time'] = '!' + sample_1min_df_prev['time']
        sample_1min_df_prev['date'] = sampleDate
        sample_1min_df = pd.concat((sample_1min_df, sample_1min_df_prev), ignore_index=True)


        

        if univers_filter is not None:
            univers        = univers_filter.set_index('date').loc[sampleDate]['sid']
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]
        else:
            univers        = panel.sdiv2df(panel.read(f"/data/beer2/wensheng/univers_nost/morning/daily/terms.{sampleDate}.sdiv")).set_index('sid').index 
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]

        
        sample_1min_df = sample_1min_df.set_index(['date','sid','time']).sort_index()
        sample_1min_df['dvolume'] = sample_1min_df[['volume']].groupby('sid', group_keys=False).shift()
        sample_1min_df['dclose']  = sample_1min_df[['close']].groupby('sid', group_keys=False).shift()
        sample_1min_df[sample_1min_df['dclose']<1e-7]['dclose']  = np.nan
        sample_1min_df['pct']         = sample_1min_df['close'] / sample_1min_df['dclose'] - 1.
        sample_1min_df['vol']         = sample_1min_df['pct'] * sample_1min_df['pct'] #sample_1min_df[['pct']].groupby('sid', group_keys=False).rolling(window=5, min_periods=5).std()
        sample_1min_df['dvol']        = sample_1min_df[['vol']].groupby('sid', group_keys=False).shift()
    
        sample_1min_df['autocorr_vol']    = sample_1min_df[['vol','dvol']].groupby('sid', group_keys=False).apply(lambda group: group['vol'].rolling(window=5, min_periods=5).corr(group['dvol']))
        sample_1min_df['autocorr_volume'] = sample_1min_df[['volume','dvolume']].groupby('sid', group_keys=False).apply(lambda group: group['volume'].rolling(window=5, min_periods=5).corr(group['dvolume']))
        sample_1min_df['dpv_corr']        = sample_1min_df[['volume','dclose']].groupby('sid', group_keys=False).apply(lambda group: group['dclose'].rolling(window=5, min_periods=5).corr(group['volume']))

        sample_1min_df['autocorr_vol_15']    = sample_1min_df[['vol','dvol']].groupby('sid', group_keys=False).apply(lambda group: group['vol'].rolling(window=15, min_periods=15).corr(group['dvol']))
        sample_1min_df['autocorr_volume_15'] = sample_1min_df[['volume','dvolume']].groupby('sid', group_keys=False).apply(lambda group: group['volume'].rolling(window=15, min_periods=15).corr(group['dvolume']))
        sample_1min_df['dpv_corr_15']        = sample_1min_df[['volume','dclose']].groupby('sid', group_keys=False).apply(lambda group: group['dclose'].rolling(window=15, min_periods=15).corr(group['volume']))


        sample_1min_df['volume_power']     = sample_1min_df['volume'] * sample_1min_df['volume']
        sample_1min_df['volume_power_mean5'] = sample_1min_df[['volume_power']].groupby('sid', group_keys=False).rolling(window=5, min_periods=5).mean().droplevel(level=0)
        sample_1min_df['volume_mean5']       = sample_1min_df[['volume']].groupby('sid', group_keys=False).rolling(window=5, min_periods=5).mean().droplevel(level=0)
        sample_1min_df['peak_cluster']       = sample_1min_df['volume_power_mean5']  / sample_1min_df['volume_mean5'] / sample_1min_df['volume_mean5']

        sample_1min_df['vol_power'] = sample_1min_df['vol'] * sample_1min_df['vol']
        sample_1min_df['vol_power_mean5'] = sample_1min_df[['vol_power']].groupby('sid', group_keys=False).rolling(window=5, min_periods=5).mean().droplevel(level=0)
        sample_1min_df['vol_mean5']       = sample_1min_df[['vol']].groupby('sid', group_keys=False).rolling(window=5, min_periods=5).mean().droplevel(level=0)
        sample_1min_df['peak_cluster_vol']= sample_1min_df['vol_power_mean5']  / sample_1min_df['vol_mean5'] / sample_1min_df['vol_mean5']


        sample_1min_df = sample_1min_df.groupby('sid', group_keys=False).shift().replace(np.inf, np.nan).replace(-np.inf, np.nan)
        sample_1min_df_return = sample_1min_df.reset_index()[['autocorr_vol', 'autocorr_volume', 'dpv_corr', 'autocorr_vol_15', 'autocorr_volume_15', 'dpv_corr_15', 
                                                              'peak_cluster', 'peak_cluster_vol', 'time', 'sid', 'date']]
        return sample_1min_df_return[sample_1min_df_return['time']>='09:31:00']

    def OminProcess_026(self, date_range=None, univers_filter=None, fd = False, root_dir='/data/beer1/data/chinaEquityData/panel/'):
        date_range = date_range if date_range is not None else self.tradingdays
        if fd:
            root_dir = self.fd_check_dir 
            date_range = [date for date in date_range if date <='20170105']
        Process_1min_df = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.Omin_feature_026)(sampleDate, root_dir, univers_filter) for sampleDate in tqdm(date_range)
        )
        feature_1min_df = pd.concat(Process_1min_df, ignore_index=True)
        feature_1min_df.set_index(['sid','date', 'time'], inplace=True)
        return feature_1min_df




    def Omin_feature_027(self, sampleDate, root_dir = ['/data/beer1/data/chinaEquityData/panel/','/data/beer2/data/CA.LTS/CNEQ/Panels/I/fq1m/'], univers_filter=None):     ## 细节检查

        sample_1min_df = panel.sdiv2df(
            panel.read(f"{root_dir[0]}1min-lts/1min-{sampleDate}.sdiv").sel(V=slice("close","close.ask.avg"), I = slice("09:31:00", None))
        )
        sample_1min_df_prev = panel.sdiv2df(
            panel.read(f"{root_dir[0]}1min-lts/1min-{pbd.prev_date(sampleDate)}.sdiv").sel(V=slice("close","close.ask.avg"), I = slice("14:00:00", "14:57:00"))
        )
        
   
        sample_1min_df['close'] = sample_1min_df['close'].replace(0.0, np.nan)
        sample_1min_df.set_index(['date','sid','time'], inplace=True)
        sample_1min_df['close'] = sample_1min_df[['close']].groupby('sid', group_keys=False).ffill()
        sample_1min_df['volume'] = sample_1min_df[['volumeTotal']].groupby('sid', group_keys=False).apply(self.custom_diff)
        
        sample_1min_df_prev['close'] = sample_1min_df_prev['close'].replace(0.0, np.nan)
        sample_1min_df_prev.set_index(['date','sid','time'], inplace=True)
        sample_1min_df_prev['close'] = sample_1min_df_prev[['close']].groupby('sid', group_keys=False).ffill()
        sample_1min_df_prev['volume'] = sample_1min_df_prev[['volumeTotal']].groupby('sid', group_keys=False).diff()

        sample_1min_df_prev.reset_index(inplace=True)
        sample_1min_df.reset_index(inplace=True)
        sample_1min_df_prev['time'] = '!' + sample_1min_df_prev['time']
        sample_1min_df_prev['date'] = sampleDate
        sample_1min_df = pd.concat((sample_1min_df, sample_1min_df_prev), ignore_index=True)


        sample_1min_df_ppq2 = panel.sdiv2df(
            panel.read(f"{root_dir[1]}ppq2/terms.{sampleDate}.sdiv")
        ).set_index(['sid','date','time'])
        sample_1min_df_ppq2_prev = panel.sdiv2df(
            panel.read(f"{root_dir[1]}ppq2/terms.{pbd.prev_date(sampleDate)}.sdiv")
        ).set_index(['sid','date','time'])
        # sample_1min_df_ppq2      = sample_1min_df_ppq2.groupby('sid', group_keys=False)['ppq2.B.trd_big0.q.p1'].apply(self.custom_diff).ewm(alpha=0.8).mean()  
        sample_1min_df_ppq2      = sample_1min_df_ppq2.groupby('sid', group_keys=False).apply(self.custom_diff).ewm(alpha=0.8).mean()
        sample_1min_df_ppq2_prev = sample_1min_df_ppq2_prev.groupby('sid', group_keys=False).diff().ewm(alpha=0.8).mean()
        sample_1min_df_ppq2_prev.reset_index(inplace=True)
        sample_1min_df_ppq2.reset_index(inplace=True)
        sample_1min_df_ppq2_prev['time'] = '!' + sample_1min_df_ppq2_prev['time']
        sample_1min_df_ppq2_prev['date'] = sampleDate
        sample_1min_df_ppq2 = pd.concat((sample_1min_df_ppq2, sample_1min_df_ppq2_prev), ignore_index=True)
    

        if univers_filter is not None:
            univers             = univers_filter.set_index('date').loc[sampleDate]['sid']
            sample_1min_df      = sample_1min_df[sample_1min_df['sid'].isin(univers)]
            sample_1min_df_ppq2 = sample_1min_df_ppq2[sample_1min_df_ppq2['sid'].isin(univers)]
        else:
            univers        = panel.sdiv2df(panel.read(f"/data/beer2/wensheng/univers_nost/morning/daily/terms.{sampleDate}.sdiv")).set_index('sid').index 
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]
            sample_1min_df_ppq2 = sample_1min_df_ppq2[sample_1min_df_ppq2['sid'].isin(univers)]

        

        sample_1min_df_ppq2 = sample_1min_df_ppq2.set_index(['date','sid','time']).sort_index()
        sample_1min_df      = sample_1min_df.set_index(['date','sid','time']).sort_index()


        sample_1min_df = sample_1min_df.merge(sample_1min_df_ppq2, left_index=True, right_index=True, how='left')
        
        sample_1min_df['dppq2.B.trd_big0.q.p1']    = sample_1min_df['ppq2.B.trd_big0.q.p1'].groupby('sid', group_keys=False).shift()
        sample_1min_df['dppq2.S.trd_big0.q.p1']    = sample_1min_df['ppq2.S.trd_big0.q.p1'].groupby('sid', group_keys=False).shift()  ## corr 算出了超过1的部分？？
        sample_1min_df['pdBlarge_corr_15']         = sample_1min_df[['dppq2.B.trd_big0.q.p1','close']].groupby('sid', group_keys=False).apply(lambda group: group['close'].rolling(window=15, min_periods=12).corr(group['dppq2.B.trd_big0.q.p1']))
        sample_1min_df['pdSlarge_corr_15']         = sample_1min_df[['dppq2.S.trd_big0.q.p1','close']].groupby('sid', group_keys=False).apply(lambda group: group['close'].rolling(window=15, min_periods=12).corr(group['dppq2.S.trd_big0.q.p1']))
        
        sample_1min_df.loc[sample_1min_df['pdBlarge_corr_15'].abs()>1.0,'pdBlarge_corr_15'] = np.nan
        sample_1min_df.loc[sample_1min_df['pdSlarge_corr_15'].abs()>1.0,'pdSlarge_corr_15'] = np.nan
        sample_1min_df['ppq2_BStrd_pq']       = sample_1min_df[['ppq2.B.trd_big0.pq.p1','ppq2.S.trd_big0.pq.p1','ppq2.B.trd_med0.pq.p1','ppq2.S.trd_med0.pq.p1','ppq2.B.trd_sml0.pq.p1','ppq2.S.trd_sml0.pq.p1']].sum(axis=1)
        sample_1min_df['ppq2.S.trd_big0.pq.ratio'] = (sample_1min_df['ppq2.S.trd_big0.pq.p1'] + sample_1min_df['ppq2.S.trd_med0.pq.p1']) / sample_1min_df['ppq2_BStrd_pq']

        sample_1min_df = sample_1min_df.groupby('sid', group_keys=False).shift()
        sample_1min_df_return = sample_1min_df.reset_index()[['pdBlarge_corr_15', 'pdSlarge_corr_15', 'ppq2.S.trd_big0.pq.ratio', 'time', 'sid', 'date']]
        return sample_1min_df_return[sample_1min_df_return['time']>='09:31:00'] 

    def OminProcess_027(self, date_range=None, univers_filter=None, fd = False, root_dir=['/data/beer2/wensheng/allu_data_with_auction_os/','/data/beer2/data/CA.LTS/CNEQ/Panels/I/fq1m/']):
        date_range = date_range if date_range is not None else self.tradingdays
        if fd:
            root_dir = self.fd_check_dir if len(root_dir)==1 else [self.fd_check_dir] * len(root_dir)
            date_range = [date for date in date_range if date <='20170105']
        Process_1min_df = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.Omin_feature_027)(sampleDate, root_dir, univers_filter) for sampleDate in tqdm(date_range)
        )
        feature_1min_df = pd.concat(Process_1min_df, ignore_index=True)
        # feature_1min_df.set_index(['sid','date', 'time'], inplace=True)
        return feature_1min_df


    def load_3s_df(self, sampleDate, root_path):
        # 获取指定目录下的所有文件路径
        folder_path = os.path.join(root_path, sampleDate)
        file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)] # if os.path.isfile(os.path.join(folder_path, file))
        with ThreadPoolExecutor(max_workers=5) as executor:
            df_list = list(executor.map(lambda file_path: pd.read_csv(file_path), file_paths))
        return pd.concat(df_list, ignore_index=True)

    def WavgPriceany_created(self, sample_3s_df, bid_cols, bidQty_cols, ask_cols, askQty_cols):
        # 1. 计算每行 snapshot 的五档加权均价
        sample_3s_df = sample_3s_df.copy()
        bid_price_vol = sample_3s_df[bid_cols].values * sample_3s_df[bidQty_cols].values
        ask_price_vol = sample_3s_df[ask_cols].values * sample_3s_df[askQty_cols].values
        total_vol = sample_3s_df[bidQty_cols].sum(axis=1) + sample_3s_df[askQty_cols].sum(axis=1)
        total_price_vol = bid_price_vol.sum(axis=1) + ask_price_vol.sum(axis=1)
        return total_price_vol/ total_vol



    def snap_agg1minbar_028(self, sampleDate, root_dir = '/data/beer2/wensheng/snapshot_date/', univers_filter=None):    ##  细节检查

        sample_3s_df            = self.load_3s_df(sampleDate, root_dir)
        sample_3s_df['timeHMS'] = pd.to_datetime(sample_3s_df['ourEpochHMS'], format="%H:%M:%S.%f").dt.strftime("%H:%M:%S") 
        sample_3s_df['sid']     = sample_3s_df['sid'].astype(str)
        sample_3s_df.set_index(['sid','timeHMS'], inplace=True)
        duplicates   = sample_3s_df.index.duplicated(keep='first')
        sample_3s_df = sample_3s_df[~duplicates].reset_index()

        if univers_filter is not None:
            univers             = univers_filter.set_index('date').loc[sampleDate]['sid']
            sample_3s_df        = sample_3s_df[sample_3s_df['sid'].isin(univers)]
        else:
            univers        = panel.sdiv2df(panel.read(f"/data/beer2/wensheng/univers_nost/morning/daily/terms.{sampleDate}.sdiv")).set_index('sid').index 
            sample_3s_df        = sample_3s_df[sample_3s_df['sid'].isin(univers)]
        
        time_strs = pd.date_range(start=pd.to_datetime('09:15:00', format='%H:%M:%S'),\
                                end=pd.to_datetime('15:00:00', format='%H:%M:%S'), freq='1S').strftime('%H:%M:%S')
        time_map_dict = {t: i+1 for i, t in enumerate(time_strs)}  # 1..5000
        sample_3s_df['time_map'] = sample_3s_df['timeHMS'].map(time_map_dict)
        sample_3s_df['time']  = pd.to_datetime(sample_3s_df['timeHMS'], format='%H:%M:%S').dt.ceil('1min')
        sample_3s_df['time']  = sample_3s_df['time'].dt.strftime('%H:%M:%S')
        sample_3s_df.set_index(['sid','time','timeHMS'], inplace=True)

 
        for i in range(1, 6):
            # sample_3s_df[f'MidPrice_bdis{i}'] = 1/(sample_3s_df['MidPrice'] - sample_3s_df[f'bid1{i}'] + 0.01)
            # sample_3s_df[f'MidPrice_adis{i}'] = 1/(sample_3s_df[f'ask1{i}'] - sample_3s_df['MidPrice'] + 0.01) 
            sample_3s_df[f'1_bsize{i}']   = 1/(sample_3s_df[f'bsize{i}'] + 100)    
            sample_3s_df[f'1_asize{i}']   = 1/(sample_3s_df[f'asize{i}'] + 100)    

        bid_cols = ['bid1', 'bid2', 'bid3', 'bid4', 'bid5']
        ask_cols = ['ask1', 'ask2', 'ask3', 'ask4', 'ask5']
        bidQty_cols = ['bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5'] 
        askQty_cols = ['asize1', 'asize2', 'asize3', 'asize4', 'asize5']
        O_bidQty_cols = ['1_bsize1', '1_bsize2', '1_bsize3', '1_bsize4', '1_bsize5']    
        O_askQty_cols = ['1_asize1', '1_asize2', '1_asize3', '1_asize4', '1_asize5']
        # MidPrice_bdis = ['MidPrice_bdis1', 'MidPrice_bdis2', 'MidPrice_bdis3', 'MidPrice_bdis4', 'MidPrice_bdis5']
        # MidPrice_adis = ['MidPrice_adis1', 'MidPrice_adis2', 'MidPrice_adis3', 'MidPrice_adis4', 'MidPrice_adis5']

        sample_3s_df['bidQ5'] = sample_3s_df[bidQty_cols].sum(axis=1) 
        sample_3s_df['askQ5'] = sample_3s_df[askQty_cols].sum(axis=1) 
        sample_3s_df['PressW'] = sample_3s_df.eval('(bidQ5 - askQ5) / (bidQ5 + askQ5)')

        sample_3s_df['MidPrice']   = (sample_3s_df['ask1'] + sample_3s_df['bid1']) / 2
        sample_3s_df['WavgPrice5'] = self.WavgPriceany_created(sample_3s_df, bid_cols, bidQty_cols, ask_cols, askQty_cols)
        sample_3s_df['PressPrice'] = self.WavgPriceany_created(sample_3s_df, bid_cols, O_bidQty_cols, ask_cols, O_askQty_cols)
        sample_3s_df['DMidAvgPrice'] = (sample_3s_df['MidPrice'] - sample_3s_df['WavgPrice5']) / (sample_3s_df['MidPrice'] + sample_3s_df['WavgPrice5'])
        
        sample_3s_df['bosize'] = sample_3s_df[['bsize5', 'bsize6', 'bsize7', 'bsize8']].sum(axis=1)
        sample_3s_df['aosize'] = sample_3s_df[['asize5', 'asize6', 'asize7', 'asize8']].sum(axis=1)
        sample_3s_df['bisize'] = sample_3s_df[['bsize1', 'bsize2', 'bsize3', 'bsize4']].sum(axis=1)
        sample_3s_df['aisize'] = sample_3s_df[['asize1', 'asize2', 'asize3', 'asize4']].sum(axis=1)
        sample_3s_df['sizeBSd_outer'] =  sample_3s_df.eval('(bosize - aosize) / (bosize + aosize)')
        sample_3s_df['sizeBSd_inner'] =  sample_3s_df.eval('(bisize - aisize) / (bisize + aisize)')

        
        sample_1min_df_aggfrom3s = sample_3s_df.groupby(['sid','time'], group_keys=False).agg(
            Mean_DMidAvgPrice=('DMidAvgPrice', 'mean'),
            sizeBSd_outer_std=('sizeBSd_outer', 'std'),
            sizeBSd_inner_std=('sizeBSd_inner', 'std'),
        )

        sample_1min_df_aggfrom3s['Trendcorr_WavgPrice5']   = sample_3s_df[['WavgPrice5','time_map']].groupby(['sid','time']).apply(lambda group: group['WavgPrice5'].corr(group['time_map']))
        sample_1min_df_aggfrom3s['Trendcorr_PressPrice']   = sample_3s_df[['PressPrice','time_map']].groupby(['sid','time']).apply(lambda group: group['PressPrice'].corr(group['time_map']))
        sample_1min_df_aggfrom3s['Trendcorr_MidPrice']     = sample_3s_df[['MidPrice','time_map']].groupby(['sid','time']).apply(lambda group: group['MidPrice'].corr(group['time_map']))
        sample_1min_df_aggfrom3s['Trendcorr_DMidAvgPrice'] = sample_3s_df[['DMidAvgPrice','time_map']].groupby(['sid','time']).apply(lambda group: group['DMidAvgPrice'].corr(group['time_map']))
        sample_1min_df_aggfrom3s['consensus_PressW_Price'] = sample_3s_df[['PressW', 'MidPrice']].groupby(['sid','time']).apply(lambda group: group['PressW'].corr(group['MidPrice']))
        sample_1min_df_aggfrom3s['stddiff_inner_outer'] = sample_1min_df_aggfrom3s['sizeBSd_inner_std'] - sample_1min_df_aggfrom3s['sizeBSd_outer_std']
        sample_1min_df_aggfrom3s['date'] = sampleDate
        sample_1min_df_aggfrom3s = sample_1min_df_aggfrom3s.reset_index().set_index(['sid','date','time'])

        sample_1min_df        = sample_1min_df_aggfrom3s.groupby('sid', group_keys=False).shift()
        sample_1min_df_return = sample_1min_df.reset_index()[['Trendcorr_WavgPrice5', 'Trendcorr_PressPrice', 'Trendcorr_MidPrice',
                                                              'Trendcorr_DMidAvgPrice', 'Mean_DMidAvgPrice', 'consensus_PressW_Price',
                                                              'stddiff_inner_outer','time', 'sid', 'date']]
        return sample_1min_df_return


    def snapProcess_028(self, date_range=None, univers_filter=None, fd = False, root_dir= '/data/beer2/wensheng/snapshot_date/'):
        date_range = date_range if date_range is not None else self.tradingdays
        if fd:
            root_dir = self.fd_check_dir 
            date_range = [date for date in date_range if date <='20170105']
        Process_1min_df = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.snap_agg1minbar_028)(sampleDate, root_dir, univers_filter) for sampleDate in tqdm(date_range)
        )
        feature_1min_df = pd.concat(Process_1min_df, ignore_index=True)
        feature_1min_df.set_index(['sid','date', 'time'], inplace=True)
        return feature_1min_df

    


    def machine_028(self, sampleDate, root_dir = '/data/beer2/mike/pshared/terms_mc/', univers_filter=None):    ##  细节检查


        sample_1min_df = panel.sdiv2df(
            panel.read(f"{root_dir}/terms.{sampleDate}.sdiv")
        )

        if univers_filter is not None:
            univers        = univers_filter.set_index('date').loc[sampleDate]['sid']
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]
        else:
            univers        = panel.sdiv2df(panel.read(f"/data/beer2/wensheng/univers_nost/morning/daily/terms.{sampleDate}.sdiv")).set_index('sid').index 
            sample_1min_df = sample_1min_df[sample_1min_df['sid'].isin(univers)]
        sample_1min_df.set_index(['sid','date','time'], inplace=True)

        sample_1min_df        = sample_1min_df.groupby('sid', group_keys=False).shift()
        sample_1min_df_return = sample_1min_df.reset_index()

        return sample_1min_df_return


    def machineProcess_028(self, date_range=None, univers_filter=None, fd = False, root_dir= '/data/beer2/mike/pshared/terms_mc/'):
        date_range = date_range if date_range is not None else self.tradingdays
        if fd:
            root_dir = self.fd_check_dir 
            date_range = [date for date in date_range if date <='20170105']
        Process_1min_df = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.machine_028)(sampleDate, root_dir, univers_filter) for sampleDate in tqdm(date_range)
        )
        feature_1min_df = pd.concat(Process_1min_df, ignore_index=True)
        feature_1min_df.set_index(['sid','date', 'time'], inplace=True)
        return feature_1min_df