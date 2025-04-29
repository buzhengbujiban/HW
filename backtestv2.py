# -*- coding: utf-8 -*-
# 作者: Li Tianle
# 出自:  北京
# 创建时间: 19:09
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, returns_df, benchmark_returns=None, compound = False):
        self.returns_df = returns_df
        self.compound = compound
        if isinstance(benchmark_returns, str):
            if benchmark_returns == "mean":
                self.benchmark_returns = returns_df.mean(axis=1)
            else:
                self.benchmark_returns = benchmark_returns

        else:
            self.benchmark_returns = benchmark_returns


    @staticmethod
    def monthly_to_daily_weight(monthly_weight_df, daily_ret):
        inter_col = monthly_weight_df.columns.intersection(daily_ret.columns)
        monthly_weight_df, daily_ret = monthly_weight_df.loc[:, inter_col], daily_ret.loc[:, inter_col]

        cum_daily_ret = (1 + daily_ret).cumprod()
        last_trading_days = cum_daily_ret.groupby(cum_daily_ret.index.to_period('M')).apply(lambda x: x.index.max())
        cum_daily_ret = cum_daily_ret / cum_daily_ret.loc[last_trading_days, :].reindex(cum_daily_ret.index).ffill()
        daily_weight_df = monthly_weight_df.reindex(cum_daily_ret.index).ffill()
        daily_weight_df = daily_weight_df * cum_daily_ret
        daily_weight_df = daily_weight_df.div(daily_weight_df.sum(axis=1), axis=0).dropna(how='all')
        return daily_weight_df

    def backtest(self, weights_df, transaction_cost = 0.0015):
        # 对齐数据
        weights_df, returns_df = weights_df.align(self.returns_df, join='inner')
        if self.benchmark_returns is not None:
            weights_df, benchmark_returns = weights_df.align(self.benchmark_returns, join='inner', axis=0)
            weights_df, returns_df = weights_df.align(self.returns_df, join='inner')

        else:
            benchmark_returns = None

        weights_df = weights_df.fillna(0)

        Numstk_long = (weights_df>0).sum(axis=1)
        Numstk_short = (weights_df<0).sum(axis=1)

        # 计算每日的组合收益率
        portfolio_returns = (weights_df * returns_df).sum(axis=1)

        # 计算每天结束时的股票价值和投资组合总价值
        stock_values = weights_df.shift() * (1 + returns_df)
        total_values = stock_values.sum(axis=1)

        # 计算每天结束时的实际权重
        actual_weights = stock_values.div(total_values, axis=0)

        # 计算换手率
        turnover_rate = (weights_df - actual_weights.shift()).abs().sum(axis=1)

        # 计算每天的交易费用
        daily_transaction_cost = turnover_rate * transaction_cost

        # 计算费后的每日收益
        portfolio_returns_after_cost = portfolio_returns - daily_transaction_cost

        if self.compound:
            # 计算累计收益率（考虑交易费用和不考虑交易费用）
            cumulative_returns_after_cost = (1 + portfolio_returns_after_cost).cumprod() - 1
            cumulative_returns = (1 + portfolio_returns).cumprod() - 1

            # 计算指数的累计收益率（如果有）
            if benchmark_returns is not None:
                benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1
        else:
            # 计算累计收益率（考虑交易费用和不考虑交易费用）
            cumulative_returns_after_cost = portfolio_returns_after_cost.cumsum() + 1
            cumulative_returns = portfolio_returns.cumsum() + 1

            # 计算指数的累计收益率（如果有）
            if benchmark_returns is not None:
                benchmark_cumulative_returns = benchmark_returns.cumsum() + 1

        # 绘制累计收益率图
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(cumulative_returns, label='Without transaction cost')
        ax1.plot(cumulative_returns_after_cost, label='With transaction cost')
        if benchmark_returns is not None:
            ax1.plot(benchmark_cumulative_returns, label='Benchmark')
        ax1.set_title('Cumulative Returns of Portfolio and Turnover Rate')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Returns', color='blue')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # 在右侧另一坐标轴上绘制换手率
        ax2 = ax1.twinx()
        ax2.plot(turnover_rate, color='red', label='Turnover Rate')
        ax2.set_ylabel('Turnover Rate', color='red')
        ax2.legend(loc='upper right')

        fig, ax3 = plt.subplots(figsize=(12, 2))
        ax3.plot(Numstk_long, label='Numstk_long')
        ax3.plot(Numstk_short, label='Numstk_short')
        ax3.set_title('Numstk')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Num stocks', color='blue')
        ax3.legend(loc='upper left')
        ax3.grid(True)

        return cumulative_returns, cumulative_returns_after_cost, turnover_rate

    def calculate_metrics(self, cumulative_returns, turnover_rate):
        # 计算投资期限（以年为单位）


        # 计算年化波动率
        if self.compound:
            daily_returns = cumulative_returns.pct_change().dropna()
            years = len(cumulative_returns) / 252

            # 计算年化收益率
            total_return = cumulative_returns[-1]
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            daily_returns = cumulative_returns.diff().dropna()
            annual_return = daily_returns.mean() * 252
        annual_volatility = daily_returns.std() * np.sqrt(252)


        # 计算平均换手率
        average_turnover_rate = turnover_rate.mean()

        # 计算夏普比率
        sharpe_ratio = annual_return / annual_volatility

        # 计算最大回撤
        drawdowns = 1 - cumulative_returns / cumulative_returns.cummax()
        max_drawdown = drawdowns.max()

        # 计算卡尔曼比率
        calmar_ratio = annual_return / max_drawdown

        # 返回所有指标
        return {
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Average Turnover Rate': average_turnover_rate,
        }

    def calculate_annual_metrics(self, cumulative_returns, turnover_rate):
        metrics_per_year = {}
        for year in range(int(cumulative_returns.index.year.min()),int(cumulative_returns.index.year.max() + 1)):
            annual_returns = cumulative_returns[cumulative_returns.index.year == year]
            annual_returns = annual_returns / annual_returns.iloc[0]
            annual_turnover_rate = turnover_rate[turnover_rate.index.year == year]
            if len(annual_returns) > 0:
                metrics_per_year[year] = self.calculate_metrics(annual_returns, annual_turnover_rate)

        return pd.DataFrame(metrics_per_year).T

    def run_backtest(self, weights_df, transaction_cost = 0.0015):
        cumulative_returns, cumulative_returns_after_cost, turnover_rate = self.backtest(weights_df, transaction_cost)
        metrics_without_cost = pd.Series(self.calculate_metrics(cumulative_returns, turnover_rate))
        metrics_with_cost = pd.Series(self.calculate_metrics(cumulative_returns_after_cost, turnover_rate))

        annual_metrics_without_cost = self.calculate_annual_metrics(cumulative_returns, turnover_rate)
        annual_metrics_with_cost = self.calculate_annual_metrics(cumulative_returns_after_cost, turnover_rate)

        return (pd.concat([metrics_without_cost, metrics_with_cost], keys=['Without Transaction Cost', 'With Transaction Cost']),
                pd.concat([annual_metrics_without_cost, annual_metrics_with_cost], keys=['Without Transaction Cost', 'With Transaction Cost']))


def zscore(factor):
    factor =  ((factor.T - np.nanmean(factor, axis=1)) / np.nanstd(factor, axis=1))
    factor = factor / np.nansum(np.abs(factor), axis=0) * 2
    return factor.T


def get_trade_days(start_date, end_date):
    trading_days = pd.read_pickle('trading_days.pkl')
    return trading_days[pd.to_datetime(start_date): pd.to_datetime(end_date)].index


class FactorBacktestmachine():
    def __init__(self, ret):
        self.ret = ret

    def backtest(self, factor, start_date, end_date, group_num=10, plot=True, cost=0.0025):
        # 日期序列
        date_list_in_use = get_trade_days(start_date, end_date)
        date_list_in_use = date_list_in_use.intersection(factor.index)
        date_list_in_use = date_list_in_use.intersection(self.ret.index)

        # 原始收益
        ret = self.ret.loc[date_list_in_use, :].replace([np.inf, -np.inf], 0)
        backtest_df = factor.loc[date_list_in_use, :]

        # 回测因子值
        demean_backtest_df = backtest_df.sub(backtest_df.mean(axis=1), axis=0)
        std_backtest_df = demean_backtest_df.div((demean_backtest_df.abs().sum(axis=1) / 2), axis=0)

        # 算ic和rankic
        ic_list = factor.loc[date_list_in_use, :].corrwith(ret, axis=1)
        ic = ic_list.mean()
        rank_factor_df = factor.loc[date_list_in_use, :].rank(axis=1)
        rank_ret_df = ret.rank(axis=1)
        rankic_list = rank_factor_df.corrwith(rank_ret_df, axis=1)
        rankic = rankic_list.mean()

        # 算ic decay
        ic_decay_list = []
        rankic_decay_list = []
        for i in range(10):
            ic_decay_list.append(
                factor.loc[date_list_in_use, :].corrwith(ret, axis=1).shift(-i).mean())
            rankic_decay_list.append(factor.loc[date_list_in_use, :].rank(axis=1).corrwith(rank_ret_df.shift(-i), axis=1).mean())


        # 生成多空信号值
        long_signal_df = backtest_df.copy()
        long_signal_df.iloc[:, :] = np.where(std_backtest_df >= 0, std_backtest_df, 0)
        long_cost_df = np.abs(cost * long_signal_df.diff(1, axis=0)) / 2

        short_signal_df = backtest_df.copy()
        short_signal_df.iloc[:, :] = np.where(std_backtest_df <= 0, -1 * std_backtest_df, 0)
        short_cost_df = np.abs(cost * short_signal_df.diff(1, axis=0)) / 2

        factor_neutral = backtest_df.sub(backtest_df.mean(axis=1), axis=0).div(backtest_df.std(axis=1), axis=0)
        factor_neutral = factor_neutral.div(factor_neutral.abs().sum(axis=1), axis=0)
        factor_ret_no_cost = (factor_neutral * ret).sum(axis=1)

        # 生成多空头pnl
        long_ret_no_cost = (long_signal_df * ret).sum(axis=1) / long_signal_df.sum(axis=1)
        short_ret_no_cost = (short_signal_df * ret).sum(axis=1) / short_signal_df.sum(axis=1)

        long_ret_after_cost = long_ret_no_cost - long_cost_df.sum(axis=1) / long_signal_df.sum(axis=1)
        short_ret_after_cost = short_ret_no_cost - short_cost_df.sum(axis=1) / short_signal_df.sum(axis=1)

        # 判断Long/Short的方向
        if long_ret_no_cost.sum() < short_ret_no_cost.sum():
            long_ret_no_cost, short_ret_no_cost = short_ret_no_cost, long_ret_no_cost
            long_ret_after_cost, short_ret_after_cost = short_ret_after_cost, long_ret_after_cost
            long_signal_df, short_signal_df = short_signal_df, long_signal_df


        # 换手序列
        turnover_series = long_signal_df.diff().abs().sum(axis=1).fillna(0).replace(np.infty, 0)

        # 基准收益
        index_ret = ret.mean(axis=1)
        # 年化指数
        annual_coef = 52 / len(long_ret_after_cost.T)

        def cal_maxdd(array):
            drawdowns = []
            max_so_far = array[0]
            for i in range(len(array)):
                if array[i] > max_so_far:
                    drawdown = 0
                    drawdowns.append(drawdown)
                    max_so_far = array[i]
                else:
                    drawdown = max_so_far - array[i]
                    drawdowns.append(drawdown)
            return max(drawdowns)

        group_signal_df_list = list(np.arange(group_num))
        stock_num_base = backtest_df.rank(axis=1).max(axis=1) / group_num
        # print('股票分组个数：{}'.format(stock_num_base))

        uprank_df = backtest_df.rank(axis=1, ascending=False)
        downrank_df = backtest_df.rank(axis=1, ascending=True)

        for i in range(group_num):
            group_signal_df_list[i] = backtest_df.copy()
            group_signal_df_list[i].iloc[:, :] = np.where(
                (uprank_df.T <= (i + 1) * stock_num_base).T & (uprank_df.T > i * stock_num_base).T, 1, 0)

        group_ret_series_list_no_cost = list(np.arange(group_num))
        group_ret_series_list_after_cost = list(np.arange(group_num))
        group_ret_list_no_cost = list(np.arange(group_num))
        group_ret_list_after_cost = list(np.arange(group_num))
        group_cost_df_list = list(np.arange(group_num))
        group_tov_list = list(np.arange(group_num))

        for i in range(group_num):
            group_signal_df = group_signal_df_list[i]

            group_ret_series_list_no_cost[i] = (group_signal_df * ret).sum(axis=1) / group_signal_df.sum(axis=1)
            group_cost_df_list[i] = np.abs(cost * group_signal_df.diff(1, axis=0)) / 2
            group_ret_series_list_after_cost[i] = group_ret_series_list_no_cost[i] - group_cost_df_list[i].sum(axis=1) / group_signal_df_list[i].sum(axis=1)

            group_ret_list_no_cost[i] = group_ret_series_list_no_cost[i].cumsum().values[-1]
            group_ret_list_after_cost[i] = group_ret_series_list_after_cost[i].cumsum().values[-1]

            weight_df = group_signal_df.div(group_signal_df.sum(axis=1).fillna(0).replace(np.inf, 0), axis=0)
            turnover = np.abs(weight_df - weight_df.shift(1, axis=0)).sum(axis=1)
            group_tov_list[i] = turnover.fillna(0).replace(np.infty, 0)

        if group_ret_list_no_cost[0] > group_ret_list_no_cost[-1]:
            top_group_ret_series_no_cost = group_ret_series_list_no_cost[0]
            top_group_ret_series_after_cost = group_ret_series_list_after_cost[0]
        else:
            top_group_ret_series_no_cost = group_ret_series_list_no_cost[-1]
            top_group_ret_series_after_cost = group_ret_series_list_after_cost[-1]


        # 回测指标
        data_dict = {}
        data_dict['IC'] = ic
        data_dict['rankIC'] = rankic
        data_dict['IR'] = ic_list.mean() / ic_list.std()
        data_dict['TurnOver'] = turnover_series.mean()
        data_dict['AlphaRet'] = (long_ret_after_cost - index_ret).cumsum().dropna().values[-1] * annual_coef
        data_dict['AlphaRetNC'] = (long_ret_no_cost - index_ret).cumsum().dropna().values[-1] * annual_coef
        data_dict['AlphaSharpe'] = (long_ret_after_cost - index_ret).mean() / (
                long_ret_after_cost - index_ret).std() * np.sqrt(52)
        data_dict['AlphaSharpeNC'] = (long_ret_no_cost - index_ret).mean() / (
                long_ret_no_cost - index_ret).std() * np.sqrt(52)
        data_dict['AlphaDrawdown'] = cal_maxdd((long_ret_after_cost - index_ret).cumsum().dropna().values)
        data_dict['AlphaDrawdownNC'] = cal_maxdd((long_ret_no_cost - index_ret).cumsum().dropna().values)
        data_dict['DrawdownRatio'] = data_dict['AlphaDrawdownNC'] / data_dict['AlphaRetNC']
        data_dict['TopGroupAlphaRet'] = (top_group_ret_series_after_cost - index_ret).cumsum().dropna().values[-1] * annual_coef
        data_dict['TopGroupAlphaRetNC'] = (top_group_ret_series_no_cost - index_ret).cumsum().dropna().values[-1] * annual_coef
        data_dict['TopGroupAlphaSharpe'] = (top_group_ret_series_after_cost - index_ret).mean() / (
                top_group_ret_series_after_cost - index_ret).std() * np.sqrt(252)
        data_dict['TopGroupAlphaSharpeNC'] = (top_group_ret_series_no_cost - index_ret).mean() / (
                top_group_ret_series_no_cost - index_ret).std() * np.sqrt(252)
        data_dict['TopGroupAlphaDrawdown'] = cal_maxdd((top_group_ret_series_after_cost - index_ret).cumsum().dropna().values)
        data_dict['TopGroupAlphaDrawdownNC'] = cal_maxdd((top_group_ret_series_no_cost - index_ret).cumsum().dropna().values)
        data_dict['TopGroupDrawdownRatio'] = data_dict['TopGroupAlphaDrawdownNC'] / data_dict['TopGroupAlphaRetNC']



        data_dict = pd.Series(data_dict)


        if plot:
            fig = plt.figure(figsize=(20, 30), dpi=200)

            ax1 = fig.add_subplot(5, 2, 1)
            long_ret_no_cost.cumsum().plot(ax=ax1, color='darkorange')
            short_ret_no_cost.cumsum().plot(ax=ax1, color='limegreen')
            index_ret.cumsum().plot(ax=ax1, color='indianred')
            ax1.legend(['long', 'short', 'index'])
            ax1.grid(axis='y')
            ax1.set_title('Long Short Absolute No Cost Return')

            ax2 = fig.add_subplot(5, 2, 2)
            (long_ret_no_cost - index_ret).cumsum().plot(ax=ax2, color='darkorange')
            (short_ret_no_cost - index_ret).cumsum().plot(ax=ax2, color='limegreen')
            ax2.legend(['long', 'short'])
            ax2.grid(axis='y')
            ax2.set_title('Long Short Excess No Cost Return')

            ax3 = fig.add_subplot(5, 2, 3)
            long_ret_after_cost.cumsum().plot(ax=ax3, color='darkorange')
            short_ret_after_cost.cumsum().plot(ax=ax3, color='limegreen')
            index_ret.cumsum().plot(ax=ax3, color='indianred')
            ax3.legend(['long', 'short', 'index'])
            ax3.grid(axis='y')
            ax3.set_title('Long Short Absolute After Cost Return')

            ax4 = fig.add_subplot(5, 2, 4)
            (long_ret_after_cost - index_ret).cumsum().plot(ax=ax4, color='darkorange')
            (short_ret_after_cost - index_ret).cumsum().plot(ax=ax4, color='limegreen')
            ax4.legend(['long', 'short'])
            ax4.grid(axis='y')
            ax4.set_title('Long Short Excess After Cost Return')

            ax5 = fig.add_subplot(5, 2, 5)
            for i in range((group_num)):
                (group_ret_series_list_no_cost[i].cumsum() - index_ret.cumsum()).plot(ax=ax5)
            group_legend_names = list((np.arange(group_num)))
            group_legend_names.append('index')
            ax5.legend(group_legend_names, fontsize=8)
            ax5.grid(axis='y')
            ax5.set_title('Group Excess Return No Cost')

            ax6 = fig.add_subplot(5, 2, 6)
            total_ret = [np.nansum(ret) for ret in group_ret_series_list_no_cost]
            ax6.bar(range(len(total_ret)), total_ret)
            ax6.hlines(np.mean(total_ret), xmin=0, xmax=len(total_ret) - 1, color='r')
            ax6.set_xticks(range(group_num))
            ax6.set_title('Group Return No Cost Bar')

            ax7 = fig.add_subplot(5, 2, 7)
            for i in range(group_num):
                (group_ret_series_list_after_cost[i] - index_ret).cumsum().plot(ax=ax7)
            ax7.legend(group_legend_names, fontsize=8)
            ax7.grid(axis='y')
            ax7.set_title('Group Excess Return After Cost')

            ax8 = fig.add_subplot(5, 2, 8)
            total_ret = [np.nansum(ret) for ret in group_ret_series_list_after_cost]
            ax8.bar(range(len(total_ret)), total_ret)
            ax8.hlines(np.mean(total_ret), xmin=0, xmax=len(total_ret) - 1, color='r')
            ax8.set_xticks(range(group_num))
            ax8.set_title('Group Return After Cost Bar')

            ax9 = fig.add_subplot(5, 2, 9)
            width = 0.4
            ax9.bar(np.arange(len(ic_decay_list)) - width / 2, ic_decay_list, width)
            ax9.bar(np.arange(len(rankic_decay_list)) + width / 2, rankic_decay_list, width)
            ax9.set_xticks(range(len(ic_decay_list)))
            ax9.legend(labels=['IC', 'rankIC'])
            # ax9.grid(b=True, axis='y')
            ax9.set_title('IC and rankIC decay')

            ax10 = fig.add_subplot(5, 2, 10)
            ic_list.cumsum().plot(ax=ax10)
            ax10.set_xticks(list(ic_list.index)[::int(len(list(ic_list.index)) / 6)])
            ax10.grid(axis='y')
            ax10.set_title('Cumulated IC_IR: {}'.format(round(ic_list.mean() / ic_list.std(), 3)))

        return data_dict


def remove_outlier(factor, threshold=5):
    """ this is for dataframe with index date and columns ticker"""
    me = factor.median(axis=1)
    abs_me = abs(factor.sub(me, axis=0)).median(axis=1)

    lb, ub = (-threshold * abs_me).add(me, axis=0), (threshold * abs_me).add(me, axis=0)
    tmp = factor.copy(deep=True)
    tmp[tmp.ge(ub, axis=0)] = np.nan
    tmp[tmp.le(lb, axis=0)] = np.nan

    return tmp


def MAD(factor, threshold=3):
    """ this is for dataframe with index date and columns ticker"""
    me = factor.median(axis=1)
    abs_me = abs(factor.sub(me, axis=0)).median(axis=1)
    lb, ub = (-threshold * abs_me).add(me, axis=0), (threshold * abs_me).add(me, axis=0)

    tmp = factor.copy(deep=True)

    return tmp.clip(lower=lb, upper=ub, axis=0)


if __name__ == '__main__':
    # 创建随机数据
    # np.random.seed(0)
    # dates = pd.date_range(start='1/1/2020', periods=500)
    # tickers = ['stock' + str(i) for i in range(1, 6)]
    #
    # weights_df = pd.DataFrame(np.random.dirichlet(np.ones(len(tickers)), len(dates)), index=dates, columns=tickers)
    # returns_df = pd.DataFrame(np.random.normal(0, 0.01, size=(len(dates), len(tickers))), index=dates, columns=tickers)
    #
    # # 创建一个稍短的benchmark_returns，用来测试数据不完全对齐的情况
    # benchmark_returns = pd.Series(np.random.normal(0, 0.01, size=len(dates) - 50), index=dates[:-50])

    # 导入数据
    month_weight_df = pd.read_pickle('daily_weight_15_0.1_0.05.pkl')
    open_df = pd.read_pickle("./database/etf_daily_data/open.pkl")
    close_df = pd.read_pickle('./database/etf_daily_data/close.pkl')
    backtest_return = open_df.pct_change().shift(-2)
    benchmark_return = backtest_return.loc[:, '510300.XSHG']


    # factor_backtest = FactorBacktestmachine(backtest_return)
    # start_date, end_date = '2016-01-04', '2021-12-31'
    # factor = close_df.pct_change().rolling(20, min_periods=10).mean() / close_df.pct_change().rolling(10,
    #                                                                                                   min_periods=10).std()
    # factor_backtest.backtest(factor.rank(axis=1, pct=True), start_date, end_date, group_num=5)
    # plt.show()


    backtest_machine = Backtester(backtest_return, benchmark_returns=benchmark_return)
    # weights_df = backtest_machine.monthly_to_daily_weight(month_weight_df, backtest_return)
    res, year_res = backtest_machine.run_backtest(month_weight_df.loc['2020-01-01':, :])
    plt.show()
