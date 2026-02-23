import ccxt.pro
import asyncio
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pprint import pprint
from decimal import Decimal

# ===================== 核心配置（新增PCT因子相关配置）=====================
SYMBOL = 'CC-USDT-SWAP'
SYMBOL2 = 'CC/USDT:USDT'
TICK_SIZE = 7
VOL_PERIOD = 25
PUSH_INTERVAL = 1
DEPTH = 3

# AS模型核心参数
k2 = 0.0008
k = 50
INVENT_K = 1e-7
gamma = 9600
max_position = 400
order_amount = 10
inventory_theta = 0.00001

# 撤单核心规则参数
order_refresh_tolerance_pct = 0.001
filled_order_delay = PUSH_INTERVAL/50
SCHEDULED_CANCEL_INTERVAL = 40*PUSH_INTERVAL
CXL_RULES_INTERVAL = 0.1
KLINE_INTERVAL = 0.5

# MACD配置
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_ADJUST_COEFF = 0.00012
adjust_delta = 0.000055

# 止盈止损配置
TAKE_PROFIT_PCT = 0.005
STOP_LOSS_PCT = 0.001

# ========== 新增PCT因子配置 ==========
PCT_LAG = 10  # 因子衰减滞后参数
PCT_RESET = True  # 价格变动是否重置衰减
PCT_RELATIVE = True  # 是否相对化因子
PCT_ON_FULL = False  # 相对化分母计算方式
PCT_QTY_ADJ = True  # 是否对数调整量
# PCT因子阈值（超过/低于该值触发调整）
PCT_UPPER_THRESHOLD = 0.5  # 因子上限阈值
PCT_LOWER_THRESHOLD = -0.5  # 因子下限阈值
PCT_ADJUST_COEFF = 0.0001  # PCT因子调整系数

# 存储结构（新增PCT因子相关字段）
data_store = {
    'klines': {},         
    'order_books': {},    
    'order_books_df': {}, # 新增：订单簿DataFrame（用于因子计算）
    'trades': {},         
    'positions': {},      
    'order_history': {},  
    'open_orders': {},    
    'open_orders_all': {}, 
    'volatility': {},     
    'last_quote_mid': {}, 
    'filled_flag': {},    
    'last_filled_time': {},
    'order_id_map': {},   
    'current_macd_state': {},
    'current_pct_state': {},  # 新增：PCT因子状态 {'over_upper'/'under_lower'/'normal'}
    'tp_sl_orders': {},   
}

# 模拟Exchange_configs（替换为你的真实配置）
exchange_configs = {
    'okx': {
        'apiKey': 'your-api-key',
        'secret': 'your-api-secret',
        'password': 'your-api-passphrase',
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    }
}

# ===================== 工具函数（新增PCT因子计算+调整逻辑）=====================
def net_pos(recs):
    net = Decimal('0')
    for r in recs:
        net += Decimal(str(r['contracts'])) * (1 if r['side']=='long' else -1)
    return float(net)

def calculate_volatility(klines):
    if len(klines) < VOL_PERIOD:
        return 0.2
    closes = [float(k['close']) for k in klines[-VOL_PERIOD:]]
    log_returns = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
    daily_vol = np.std(log_returns) * math.sqrt(1440)
    annual_vol = daily_vol * math.sqrt(365)
    return annual_vol

def align_to_tick(price, tick_size):
    return round(price , TICK_SIZE)

def calculate_as_quote(S_t, sigma, q_t, tau, k, gamma):
    tau_annual = tau
    spread_base = (1/gamma) * math.log(1 + gamma/k) * S_t + (k2 * sigma**2 * tau_annual / 2)* S_t
    inventory_adjust = (k * sigma * tau_annual / 2) * abs(q_t) * INVENT_K

    if q_t>0:
        Bid_star = S_t - spread_base  - inventory_adjust* S_t
        Ask_star = S_t + spread_base
    else:
        Bid_star = S_t - spread_base
        Ask_star = S_t + spread_base  + inventory_adjust* S_t
    Bid_star = align_to_tick(Bid_star, TICK_SIZE)
    Ask_star = align_to_tick(Ask_star, TICK_SIZE)
    return Bid_star, Ask_star

# MACD相关函数（原有逻辑）
def calculate_simple_macd(klines, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    if len(klines) < slow + signal:
        return None, None
    closes = np.array([float(k['close']) for k in klines])
    ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
    dif = ema_fast - ema_slow
    dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().values
    return dif[-1], dea[-1]

def judge_macd_continuous_state(klines):
    dif, dea = calculate_simple_macd(klines)
    if dif is None or dea is None:
        return 'none'
    if dif > dea:
        return 'golden'
    elif dif < dea:
        return 'dead'
    else:
        return 'none'

# ========== 新增：PCT因子计算函数 ==========
def calc_PCT_1_I(df, start_b, end_b, qty_adj=PCT_QTY_ADJ, lag=PCT_LAG, 
                 reset=PCT_RESET, relative=PCT_RELATIVE, on_full=PCT_ON_FULL):
    # Step 1. 基础数据
    bp, ap = df["BP01"].to_numpy(), df["AP01"].to_numpy()
    bs, a_s = df["BS01"].to_numpy(), df["AS01"].to_numpy()
    # 模拟交易数据（若无真实交易数据，暂时设为0）
    tp = np.zeros_like(bp)
    tv = np.zeros_like(bp)
    mid = (bp + ap) / 2.0
    t = df["TIME"].to_numpy().astype(float)
    dt = np.diff(t, prepend=t[0])
    dt[0] = np.median(dt[1:]) if len(dt) > 2 else 1.0

    n = len(df)
    buys = np.zeros(n)
    sells = np.zeros(n)

    # Step 2. 检测交易方向（无真实交易数据时，这部分为0）
    buy_trades = np.zeros(n)
    buy_trades[(tp >= ap)] = tv[tp >= ap]
    sell_trades = np.zeros(n)
    sell_trades[(tp <= bp)] = tv[tp <= bp]

    # Step 3. 前一tick的窗口
    pbp, pap = np.roll(bp, 1), np.roll(ap, 1)
    pbs, pas = np.roll(bs, 1), np.roll(a_s, 1)
    pbp[0], pap[0], pbs[0], pas[0] = bp[0], ap[0], bs[0], a_s[0]

    # Step 4. 根据订单变化分类
    pbs[pbp < bp] = 0.0
    pas[pbp > ap] = 0.0
    buys = bs - pbs 
    sells = a_s - pas 
    buys[pbp > bp] = pbs[pbp > bp] 
    sells[pbp < ap] = pas[pbp < ap] 

    # Step 5. 百分比条件
    pbs1 = np.roll(bs, 1); pas1 = np.roll(a_s, 1)
    pbs1[0], pas1[0] = bs[0], a_s[0]
    buys_pct = np.divide(buys, pbs1, out=np.zeros_like(buys), where=pbs1 != 0)
    sells_pct = np.divide(sells, pas1, out=np.zeros_like(sells), where=pas1 != 0)
    buys_pct = np.nan_to_num(buys_pct, nan=0.0, posinf=100.0, neginf=-100.0)
    sells_pct = np.nan_to_num(sells_pct, nan=0.0, posinf=100.0, neginf=-100.0)

    cond_buy = ((buys != 0) & (pbs1 != 0) & (buys_pct >= start_b) & (buys_pct <= end_b)).astype(float)
    cond_sell = ((sells != 0) & (pas1 != 0) & (sells_pct >= start_b) & (sells_pct <= end_b)).astype(float)

    # Step 6. 应用条件
    if qty_adj:
        buys = np.sign(buys) * np.log(np.abs(buys) + 1)
        sells = np.sign(sells) * np.log(np.abs(sells) + 1)
    else:
        buys = np.sign(buys)
        sells = np.sign(sells)

    buys *= cond_buy
    sells *= cond_sell

    # Step 7. 衰减计算
    decay_buys = np.zeros(n)
    decay_sells = np.zeros(n)
    decay_factor = np.exp(-dt / max(lag, 1e-6))

    for i in range(1, n):
        if reset and (abs(mid[i] - mid[i-1]) > 1e-5):
            decay_buys[i] = buys[i]
            decay_sells[i] = sells[i]
        else:
            decay_buys[i] = decay_buys[i-1] * decay_factor[i] + buys[i]
            decay_sells[i] = decay_sells[i-1] * decay_factor[i] + sells[i]

    raw = np.nan_to_num(decay_buys - decay_sells, nan=0.0)

    # Step 8. 相对化
    if not relative:
        values = raw
    else:
        t_factor = np.clip((t - t[0]) / lag, 0.0, 1.0)
        if not on_full:
            denom = np.zeros(n)
            for i in range(1, n):
                denom[i] = denom[i-1] * np.exp(-dt[i] / (lag * 2)) + abs(buys[i] + sells[i])
        else:
            denom = np.zeros(n)
            for i in range(1, n):
                denom[i] = denom[i-1] * np.exp(-dt[i] / lag) + 1.0
        denom = np.where(denom == 0, 1e-9, denom)
        values = np.clip(t_factor * (raw / denom), -1.0, 1.0)

    return pd.DataFrame({f'PCT_{qty_adj}_{start_b}_{end_b}_I': values}, index=pd.Index(t, name='TIME'))

# ========== 新增：PCT因子状态判断 ==========
def judge_pct_state(exchange_id):
    """判断PCT因子状态：over_upper/under_lower/normal"""
    if exchange_id not in data_store['order_books_df'] or len(data_store['order_books_df'][exchange_id]) < 10:
        return 'normal'
    
    df = data_store['order_books_df'][exchange_id]
    # 计算多区间PCT因子（取最后一个值）
    pct_results = [
        calc_PCT_1_I(df, -1e6, -0.99),
        calc_PCT_1_I(df, -0.99, -0.5),
        calc_PCT_1_I(df, -0.5, 0),
        calc_PCT_1_I(df, 0, 0.25),
        calc_PCT_1_I(df, 0.25, 0.5),
        calc_PCT_1_I(df, 0.5, 1),
        calc_PCT_1_I(df, 1, 1e6)
    ]
    # 取所有因子的均值作为最终PCT值
    pct_values = [res.iloc[-1, 0] for res in pct_results]
    pct_final = np.mean(pct_values)
    
    # 判断状态
    if pct_final > PCT_UPPER_THRESHOLD:
        return 'over_upper'
    elif pct_final < PCT_LOWER_THRESHOLD:
        return 'under_lower'
    else:
        return 'normal'

# 修改adjust_quote_by_market：新增PCT因子调整逻辑
def adjust_quote_by_market(Bid_star, Ask_star, order_book, exchange_id, current_mid):
    # 原有盘口深度调整逻辑
    bid_depth = sum(float(b[1]) for b in order_book['bids'][:DEPTH])
    ask_depth = sum(float(a[1]) for a in order_book['asks'][:DEPTH])
    if ask_depth == 0 or bid_depth == 0:
        return Bid_star, Ask_star
    if bid_depth / ask_depth > 1.2:
        Ask_star = align_to_tick(Ask_star * (1 + adjust_delta), TICK_SIZE)
        print(f"[imb adjust] 买压力大, 多挂卖{Ask_star * (adjust_delta):.6f}")
    if ask_depth / bid_depth > 1.2:
        Bid_star = align_to_tick(Bid_star * (1 - adjust_delta), TICK_SIZE)
        print(f"[imb adjust] 卖压力大, 少挂买{Bid_star * (adjust_delta):.6f}")
    
    # MACD调整逻辑（原有）
    macd_state = data_store.get('current_macd_state', {}).get(exchange_id, 'none')
    if macd_state != 'none':
        adjust_amt = current_mid * MACD_ADJUST_COEFF
        if macd_state == 'golden':
            Ask_star = align_to_tick(Ask_star + adjust_amt, TICK_SIZE)
            print(f"[MACD持续金叉] 空单额外远离mid：{adjust_amt:.6f}，调整后Ask_star：{Ask_star:.6f}")
        else:
            Bid_star = align_to_tick(Bid_star - adjust_amt, TICK_SIZE)
            print(f"[MACD持续死叉] 多单额外远离mid：{adjust_amt:.6f}，调整后Bid_star：{Bid_star:.6f}")
    
    # ========== 新增：PCT因子调整逻辑 ==========
    pct_state = data_store.get('current_pct_state', {}).get(exchange_id, 'normal')
    if pct_state != 'normal':
        adjust_amt = current_mid * PCT_ADJUST_COEFF
        if pct_state == 'over_upper':
            # 因子超过上限：买单远离mid，卖单靠近mid
            Bid_star = align_to_tick(Bid_star - adjust_amt, TICK_SIZE)
            print(f"[PCT因子超上限] 买单额外远离mid：{adjust_amt:.6f}，调整后Bid_star：{Bid_star:.6f}")
        elif pct_state == 'under_lower':
            # 因子低于下限：卖单远离mid，买单靠近mid
            Ask_star = align_to_tick(Ask_star + adjust_amt, TICK_SIZE)
            print(f"[PCT因子低于下限] 卖单额外远离mid：{adjust_amt:.6f}，调整后Ask_star：{Ask_star:.6f}")
    
    return Bid_star, Ask_star

def adjust_quote_by_inventory(Bid_star, Ask_star, q_t, sigma):
    if q_t > 0:
        Bid_star = align_to_tick(Bid_star * (1 - inventory_theta * sigma), TICK_SIZE)
        Ask_star = align_to_tick(Ask_star * (1 + inventory_theta * sigma), TICK_SIZE)
    elif q_t < 0:
        Bid_star = align_to_tick(Bid_star * (1 + inventory_theta * sigma), TICK_SIZE)
        Ask_star = align_to_tick(Ask_star * (1 - inventory_theta * sigma), TICK_SIZE)
    return Bid_star, Ask_star

def calculate_tp_sl_price(fill_side, fill_price):
    fill_price = float(fill_price)
    if fill_side == 'buy':
        tp_price = align_to_tick(fill_price * (1 + TAKE_PROFIT_PCT), TICK_SIZE)
        sl_price = align_to_tick(fill_price * (1 - STOP_LOSS_PCT), TICK_SIZE)
    else:
        tp_price = align_to_tick(fill_price * (1 - TAKE_PROFIT_PCT), TICK_SIZE)
        sl_price = align_to_tick(fill_price * (1 + STOP_LOSS_PCT), TICK_SIZE)
    print(f"[止盈止损计算] {fill_side}单 | 成交价：{fill_price:.6f} | 止盈价：{tp_price:.6f} | 止损价：{sl_price:.6f}")
    return tp_price, sl_price

def check_duplicate_order(exchange_id, side, target_price):
    if exchange_id not in data_store['open_orders']:
        data_store['open_orders'][exchange_id] = {'BuyOpenOrder': {}, 'SellOpenOrder': {}}
    target_price_aligned = align_to_tick(float(target_price), TICK_SIZE)
    side_dict_key = 'BuyOpenOrder' if side == 'buy' else 'SellOpenOrder'
    side_orders = data_store['open_orders'][exchange_id][side_dict_key]
    if target_price_aligned in side_orders:
        order_id = side_orders[target_price_aligned]
        print(f"[{datetime.now(timezone.utc)}] 重复挂单检查：{side} | 价格{target_price_aligned} 已存在未成交挂单（orderId={order_id}），跳过挂单")
        return (True, order_id)
    return (False, None)

# ===================== 数据更新函数（新增PCT因子状态更新）=====================
async def handle_kline_update(exchange_id, kline_data):
    if exchange_id not in data_store['klines']:
        data_store['klines'][exchange_id] = []
    processed_kline = {
        'timestart': kline_data.get('timestart'),
        'open': float(kline_data.get('open')),
        'high': float(kline_data.get('high')),
        'low': float(kline_data.get('low')),
        'close': float(kline_data.get('close')),
        'volume': float(kline_data.get('volume')),
        'TIME': time.time()
    }
    data_store['klines'][exchange_id].append(processed_kline)
    if len(data_store['klines'][exchange_id]) > VOL_PERIOD * 10:
        data_store['klines'][exchange_id] = data_store['klines'][exchange_id][-VOL_PERIOD * 10:]
    data_store['volatility'][exchange_id] = calculate_volatility(data_store['klines'][exchange_id])
    data_store['current_macd_state'][exchange_id] = judge_macd_continuous_state(data_store['klines'][exchange_id])

async def handle_order_book_update(exchange_id, order_book):
    # 原有订单簿存储
    if exchange_id not in data_store['order_books']:
        data_store['order_books'][exchange_id] = []
    processed_ob = {
        'bids': order_book.get('bids', []),
        'asks': order_book.get('asks', []),
        'timestamp': order_book.get('timestamp', time.time() * 1000),
        'TIME': time.time()
    }
    data_store['order_books'][exchange_id].append(processed_ob)
    if len(data_store['order_books'][exchange_id]) > 100:
        data_store['order_books'][exchange_id] = data_store['order_books'][exchange_id][-100:]
    
    # ========== 新增：构建订单簿DataFrame（用于PCT因子计算） ==========
    if exchange_id not in data_store['order_books_df']:
        data_store['order_books_df'][exchange_id] = pd.DataFrame(columns=['BP01', 'BS01', 'AP01', 'AS01', 'TIME'])
    # 提取买一/卖一数据
    bp01 = float(processed_ob['bids'][0][0]) if processed_ob['bids'] else 0
    bs01 = float(processed_ob['bids'][0][1]) if processed_ob['bids'] else 0
    ap01 = float(processed_ob['asks'][0][0]) if processed_ob['asks'] else 0
    as01 = float(processed_ob['asks'][0][1]) if processed_ob['asks'] else 0
    # 追加到DataFrame
    new_row = pd.DataFrame({
        'BP01': [bp01],
        'BS01': [bs01],
        'AP01': [ap01],
        'AS01': [as01],
        'TIME': [processed_ob['TIME']]
    })
    data_store['order_books_df'][exchange_id] = pd.concat([data_store['order_books_df'][exchange_id], new_row], ignore_index=True)
    # 保留最近100条数据
    if len(data_store['order_books_df'][exchange_id]) > 100:
        data_store['order_books_df'][exchange_id] = data_store['order_books_df'][exchange_id].iloc[-100:]
    
    # ========== 新增：更新PCT因子状态 ==========
    data_store['current_pct_state'][exchange_id] = judge_pct_state(exchange_id)

async def fetch_positions(exchange, exchange_id):
    while True:
        try:
            positions = await exchange.fetch_positions(symbols=[SYMBOL])
            current_pos = [p for p in positions if p['symbol']==SYMBOL2] 
            q_t = net_pos(current_pos)
            print(f"当前净仓位: {q_t:.4f} (多头正/空头负)")
            data_store['positions'][exchange_id] = {
                'net_qt': q_t, 'abs_qt': abs(q_t), 'pos_side': 'long' if q_t > 0 else 'short' if q_t < 0 else 'flat'
            }
            await asyncio.sleep(PUSH_INTERVAL)
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] 库存更新错误: {e}")
            await asyncio.sleep(1)

# ===================== 撤单/下单/做市核心逻辑（无改动）=====================
async def cancel_orders_by_rules(exchange, exchange_id, current_mid):
    is_canceled = False
    if exchange_id not in data_store['open_orders'] or len(data_store['open_orders'][exchange_id]) == 0:
        return is_canceled
    if exchange_id not in data_store['last_quote_mid']:
        data_store['last_quote_mid'][exchange_id] = current_mid
    if exchange_id not in data_store['filled_flag']:
        data_store['filled_flag'][exchange_id] = False
    if exchange_id not in data_store['last_filled_time']:
        data_store['last_filled_time'][exchange_id] = 0

    last_mid = data_store['last_quote_mid'][exchange_id]
    filled_flag = data_store['filled_flag'][exchange_id]

    if filled_flag:
        print(f"[{datetime.now(timezone.utc)}] 检测到订单成交 → 执行成交即撤，撤销所有未成交挂单")
        is_canceled = True
    elif abs(current_mid - last_mid) / last_mid > order_refresh_tolerance_pct:
        print(f"[{datetime.now(timezone.utc)}] 价格偏离触发撤单 | 上次挂单mid:{last_mid:.4f} | 当前mid:{current_mid:.4f} | 偏离幅度:{abs(current_mid-last_mid)/last_mid*100:.4f}% > {order_refresh_tolerance_pct*100}%")
        is_canceled = True

    if is_canceled:
        open_orders = data_store['open_orders_all'][exchange_id].copy()
        for order_id in open_orders:
            try:
                await exchange.cancel_order(order_id, SYMBOL)
                print(f"[{datetime.now(timezone.utc)}] 定时撤单成功: {order_id}")
                if order_id in data_store['open_orders_all'][exchange_id]:
                    data_store['open_orders_all'][exchange_id].remove(order_id)
            except Exception as e:
                print(f"[{datetime.now(timezone.utc)}] 定时撤单失败 {order_id}: {e}")
        
        tp_sl_orders = data_store.get('tp_sl_orders', {}).get(exchange_id, {})
        for filled_order_id, (tp_order_id, sl_order_id) in tp_sl_orders.items():
            for order_id in [tp_order_id, sl_order_id]:
                try:
                    await exchange.cancel_order(order_id, SYMBOL)
                    print(f"[{datetime.now(timezone.utc)}] 止盈止损单撤单成功: {order_id}")
                except Exception as e:
                    print(f"[{datetime.now(timezone.utc)}] 止盈止损单撤单失败 {order_id}: {e}")
        data_store['tp_sl_orders'][exchange_id] = {}
        data_store['filled_flag'][exchange_id] = False
        data_store['last_filled_time'][exchange_id] = time.time()

    return is_canceled

async def cancel_old_orders(exchange, exchange_id):
    if exchange_id not in data_store['open_orders']:
        return
    open_orders = data_store['open_orders_all'][exchange_id].copy()
    for order_id in open_orders:
        try:
            await exchange.cancel_order(order_id, SYMBOL)
            print(f"[{datetime.now(timezone.utc)}] 定时撤单成功: {order_id}")
            if order_id in data_store['open_orders_all'][exchange_id]:
                data_store['open_orders_all'][exchange_id].remove(order_id)
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] 定时撤单失败 {order_id}: {e}")
    
    tp_sl_orders = data_store.get('tp_sl_orders', {}).get(exchange_id, {})
    for filled_order_id, (tp_order_id, sl_order_id) in tp_sl_orders.items():
        for order_id in [tp_order_id, sl_order_id]:
            try:
                await exchange.cancel_order(order_id, SYMBOL)
                print(f"[{datetime.now(timezone.utc)}] 定时撤单（TP/SL）成功: {order_id}")
            except Exception as e:
                print(f"[{datetime.now(timezone.utc)}] 定时撤单（TP/SL）失败 {order_id}: {e}")
    data_store['tp_sl_orders'][exchange_id] = {}

async def scheduled_cancel_loop(exchange, exchange_id):
    print(f"[{datetime.now(timezone.utc)}] 启动定时撤单协程（每隔{SCHEDULED_CANCEL_INTERVAL}秒执行一次）")
    while True:
        try:
            await cancel_old_orders(exchange, exchange_id)
            await asyncio.sleep(SCHEDULED_CANCEL_INTERVAL)
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] 定时撤单协程异常: {e}")
            await asyncio.sleep(5)

async def place_tp_sl_orders(exchange, exchange_id, fill_order_id, fill_side, fill_price, fill_amount):
    if fill_order_id in data_store['tp_sl_orders'][exchange_id]:
        print(f"[{datetime.now(timezone.utc)}] 成交订单{fill_order_id}已挂止盈止损单，跳过重复挂单")
        return
    try:
        tp_price, sl_price = calculate_tp_sl_price(fill_side, fill_price)
        tp_side = 'sell' if fill_side == 'buy' else 'buy'
        sl_side = 'sell' if fill_side == 'buy' else 'buy'
        common_params = {'tdMode': 'cross', 'reduceOnly': True}
        
        tp_order = await exchange.create_order(SYMBOL, 'limit', tp_side, fill_amount, tp_price, params=common_params)
        sl_order = await exchange.create_order(SYMBOL, 'limit', sl_side, fill_amount, sl_price, params=common_params)
        
        data_store['tp_sl_orders'][exchange_id][fill_order_id] = (tp_order['id'], sl_order['id'])
        print(f"[{datetime.now(timezone.utc)}] 止盈止损单挂单成功 → TP(orderId={tp_order['id']}) | SL(orderId={sl_order['id']})")
    except Exception as e:
        print(f"[{datetime.now(timezone.utc)}] 止盈止损单挂单失败: {e}")

async def monitor_filled_orders(exchange, exchange_id):
    print(f"[{datetime.now(timezone.utc)}] 启动成交监控协程 → 实时检测订单成交状态")
    if exchange_id not in data_store['filled_flag']:
        data_store['filled_flag'][exchange_id] = False
    if exchange_id not in data_store['order_history']:
        data_store['order_history'][exchange_id] = []
    if exchange_id not in data_store['tp_sl_orders']:
        data_store['tp_sl_orders'][exchange_id] = {}

    while True:
        try:
            current_orders = await exchange.fetchOpenOrders(SYMBOL, limit=20)
            filled_orders = [o for o in current_orders if o['status'] == 'filled' and o['symbol'] == SYMBOL]
            if filled_orders and not data_store['filled_flag'][exchange_id]:
                data_store['filled_flag'][exchange_id] = True
                for fill_detail in filled_orders:
                    fill_order_id = fill_detail['id']
                    fill_side = fill_detail['side']
                    fill_price = fill_detail['price']
                    fill_amount = fill_detail['amount']
                    print(f"[{datetime.now(timezone.utc)}] 监控到成交订单 → side:{fill_side} | price:{fill_price} | amount:{fill_amount} | orderId:{fill_order_id}")
                    await place_tp_sl_orders(exchange, exchange_id, fill_order_id, fill_side, fill_price, fill_amount)
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] 成交监控异常: {e}")
            await asyncio.sleep(1)

async def place_as_orders(exchange, exchange_id):
    klines = data_store['klines'].get(exchange_id, [])
    order_books = data_store['order_books'].get(exchange_id, [])
    if len(order_books) == 0 or len(klines) < VOL_PERIOD:
        return
    latest_ob = order_books[-1]
    bid1 = float(latest_ob['bids'][0][0]) if latest_ob['bids'] else 0
    ask1 = float(latest_ob['asks'][0][0]) if latest_ob['asks'] else 0
    if bid1 <= 0 or ask1 <= 0:
        return
    S_t = (bid1 + ask1) / 2

    sigma = data_store['volatility'].get(exchange_id, 0.2)
    pos_data = data_store['positions'].get(exchange_id, {})
    net_qt = pos_data.get('net_qt', 0.0)
    abs_qt = pos_data.get('abs_qt', 0.0)
    pos_side = pos_data.get('pos_side', 'flat')
    tau = SCHEDULED_CANCEL_INTERVAL

    if exchange_id in data_store['last_filled_time'] and time.time() - data_store['last_filled_time'][exchange_id] < filled_order_delay:
        return

    Bid_star, Ask_star = calculate_as_quote(S_t, sigma, net_qt, tau, k, gamma)
    Bid_star, Ask_star = adjust_quote_by_market(Bid_star, Ask_star, latest_ob, exchange_id, S_t)
    Bid_star, Ask_star = adjust_quote_by_inventory(Bid_star, Ask_star, net_qt, sigma)
    
    if abs(net_qt) >= max_position:
        print(f"[{datetime.now(timezone.utc)}] 库存超限（net_qt={net_qt:.4f}），仅平仓")
        close_side = 'sell' if net_qt > 0 else 'buy'
        close_price = ask1 if close_side == 'sell' else bid1
        await place_single_order(exchange, exchange_id, close_side, close_price, abs(net_qt), is_close=True)
        return
    
    print(f"\n[{datetime.now(timezone.utc)}] AS模型报价：Bid*={Bid_star}, Ask*={Ask_star} (net_qt={net_qt:.4f}, σ={sigma:.2f})")
    data_store['last_quote_mid'][exchange_id] = S_t

    if pos_side == 'long':
        await place_single_order(exchange, exchange_id, 'buy', Bid_star, order_amount, is_close=False)
        await place_single_order(exchange, exchange_id, 'sell', Ask_star, order_amount, is_close=True)
    elif pos_side == 'short':
        await place_single_order(exchange, exchange_id, 'buy', Bid_star, order_amount, is_close=True)
        await place_single_order(exchange, exchange_id, 'sell', Ask_star, order_amount, is_close=False)
    else:
        await place_single_order(exchange, exchange_id, 'buy', Bid_star, order_amount, is_close=False)
        await place_single_order(exchange, exchange_id, 'sell', Ask_star, order_amount, is_close=False)

    await asyncio.sleep(PUSH_INTERVAL)

async def place_single_order(exchange, exchange_id, side, price, amount, is_close=False):
    try:
        is_duplicate, _ = check_duplicate_order(exchange_id, side, price)
        if is_duplicate:
            return
        params = {'tdMode': 'cross', 'reduceOnly': is_close}
        order = await exchange.create_order(SYMBOL, 'limit', side, amount, price, params=params)
        
        price_aligned = align_to_tick(float(price), TICK_SIZE)
        side_dict_key = 'BuyOpenOrder' if side == 'buy' else 'SellOpenOrder'
        data_store['open_orders'][exchange_id][side_dict_key][price_aligned] = order['id']
        data_store['order_id_map'][order['id']] = (exchange_id, side, price_aligned)
        
        if exchange_id not in data_store['open_orders_all']:
            data_store['open_orders_all'][exchange_id] = []
        data_store['open_orders_all'][exchange_id].append(order['id'])    

        order_type = "平仓" if is_close else "开仓"
        print(f"[{datetime.now(timezone.utc)}] 挂单成功：{order_type} | {side} {amount} CC @ {price} (orderId={order['id']})")
        
        if exchange_id not in data_store['order_history']:
            data_store['order_history'][exchange_id] = []
        data_store['order_history'][exchange_id].append(order['id'])

    except Exception as e:
        order_type = "平仓" if is_close else "开仓"
        print(f"[{datetime.now(timezone.utc)}] 挂单失败 {order_type} {side} @ {price}: {e}")

async def as_market_maker_loop(exchange, exchange_id):
    print(f"[{datetime.now(timezone.utc)}] 启动AS做市主循环（价格偏离+成交即撤双规则）")
    while True:
        try:
            if exchange_id not in data_store['order_books'] or len(data_store['order_books'][exchange_id]) == 0:
                await asyncio.sleep(0.1)
                continue
            order_books = data_store['order_books'].get(exchange_id, [])
            latest_ob = order_books[-1]
            bid1 = float(latest_ob['bids'][0][0]) if latest_ob['bids'] else 0
            ask1 = float(latest_ob['asks'][0][0]) if latest_ob['asks'] else 0
            if bid1 <= 0 or ask1 <= 0:
                await asyncio.sleep(0.1)
                continue
            current_mid = (bid1 + ask1) / 2

            await cancel_orders_by_rules(exchange, exchange_id, current_mid)
            await place_as_orders(exchange, exchange_id)

            await asyncio.sleep(CXL_RULES_INTERVAL)
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] 做市循环错误: {e}")
            await asyncio.sleep(PUSH_INTERVAL)

# ===================== 数据订阅函数（无改动）=====================
async def watch_klines(exchange_id, exchange):
    try:
        while True:
            ohlcv = await exchange.watch_ohlcv(SYMBOL, '1m')
            latest_ohlcv = ohlcv[-1] if isinstance(ohlcv[0], list) else ohlcv
            await handle_kline_update(exchange_id, {
                'timestart': latest_ohlcv[0],
                'open': latest_ohlcv[1],
                'high': latest_ohlcv[2],
                'low': latest_ohlcv[3],
                'close': latest_ohlcv[4],
                'volume': latest_ohlcv[5]
            })
            await asyncio.sleep(KLINE_INTERVAL)
    except Exception as e:
        print(f"[{datetime.now(timezone.utc)}] K线订阅错误: {e}")

async def watch_order_book(exchange_id, exchange):
    try:
        print(f"[{datetime.now(timezone.utc)}] 订阅订单簿（{DEPTH}档）...")
        order_book = await exchange.fetch_order_book(SYMBOL, limit=DEPTH)
        await handle_order_book_update(exchange_id, order_book)
        while True:
            updated_ob = await exchange.watch_order_book(SYMBOL, limit=DEPTH)
            await handle_order_book_update(exchange_id, updated_ob)
    except Exception as e:
        print(f"[{datetime.now(timezone.utc)}] 订单簿订阅错误: {e}")

# ===================== 主函数 =====================
async def run_exchange_bot(exchange_id, config):
    exchange_class = getattr(ccxt.pro, exchange_id)
    exchange = exchange_class(config)
    try:
        await asyncio.gather(
            watch_klines('okx', exchange),          
            watch_order_book('okx', exchange),      
            fetch_positions(exchange, 'okx'),       
            monitor_filled_orders(exchange, 'okx'), 
            as_market_maker_loop(exchange, 'okx'),  
            scheduled_cancel_loop(exchange, 'okx')  
        )
    finally:
        await exchange.close()

async def main():
    utc_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{utc_time}] 启动CC-USDT-SWAP AS做市机器人 → 新增PCT因子调整逻辑")
    print(f"[{utc_time}] PCT因子阈值：上限{PCT_UPPER_THRESHOLD} | 下限{PCT_LOWER_THRESHOLD} | 调整系数{PCT_ADJUST_COEFF}")

    tasks = [run_exchange_bot('okx', exchange_configs['okx'])]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        utc_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n[{utc_time}] 程序已手动终止")
