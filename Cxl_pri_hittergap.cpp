#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iterator>
#include <string>
#include <utility>  // 新增：用于pair
using namespace std;

// 订单结构体（新增timestamp字段记录挂单时间）
struct Order {
    int orderId;
    char side;       // 'b' = buy, 's' = sell
    int size;
    double price;
    double timestamp; // 挂单时间（与Msg的TIME字段对应）
};

// 消息结构体
struct Msg {
    string DATE;
    double TIME;
    string RIC;
    string MSG;      // 事件类型：Cxl/ADD/DEL/Trd等
    char SIDE;       // 买卖方向
    double PRICE;    // 订单价格
    int SIZE;        // 订单数量
    int ORDER_ID;    // 订单ID
};

// 订单簿类
class OrderBook {
private:
    unordered_map<int, Order> orderMap;
    map<double, vector<int>, greater<double>> bids;  // 买盘：价格降序（越高越优先）
    map<double, vector<int>, less<double>> asks;     // 卖盘：价格升序（越低越优先）

public:
    // 重载newOrder：支持传入挂单时间
    void newOrder(int orderId, char side, int size, double price, double timestamp) {
        Order o{orderId, side, size, price, timestamp};
        orderMap[orderId] = o;
        if (side == 'b') bids[price].push_back(orderId);
        else asks[price].push_back(orderId);
    }

    // 兼容原有无时间参数的调用
    void newOrder(int orderId, char side, int size, double price) {
        newOrder(orderId, side, size, price, 0.0);
    }

    // 减少订单数量
    void reduceOrder(int orderId, int newSize) {
        if (!orderMap.count(orderId)) return;
        orderMap[orderId].size = newSize;
        if (newSize == 0) deleteOrder(orderId);
    }

    // 修改订单（先删后加）
    void modifyOrder(int oldOrderId, int orderId, char side, int size, double price) {
        deleteOrder(oldOrderId);
        newOrder(orderId, side, size, price);
    }

    // 删除订单
    void deleteOrder(int orderId) {
        if (!orderMap.count(orderId)) return;
        Order o = orderMap[orderId];
        auto& targetBook = (o.side == 'b') ? bids : asks;
        auto priceIt = targetBook.find(o.price);
        if (priceIt == targetBook.end()) return;

        auto& orderList = priceIt->second;
        orderList.erase(remove(orderList.begin(), orderList.end(), orderId), orderList.end());
        if (orderList.empty()) targetBook.erase(priceIt);
        orderMap.erase(orderId);
    }

    // 获取档位数量
    int getNumLevels(char side) {
        return (side == 'b') ? bids.size() : asks.size();
    }

    // 获取指定档位价格
    double getLevelPrice(char side, int level) {
        if (level < 0) return NAN;
        auto& targetBook = (side == 'b') ? bids : asks;
        if (level >= (int)targetBook.size()) return NAN;
        auto it = targetBook.begin();
        advance(it, level);
        return it->first;
    }

    // 获取指定档位总数量
    int getLevelSize(char side, int level) {
        if (level < 0) return 0;
        auto& targetBook = (side == 'b') ? bids : asks;
        if (level >= (int)targetBook.size()) return 0;
        auto it = targetBook.begin();
        advance(it, level);
        int total = 0;
        for (int id : it->second) total += orderMap[id].size;
        return total;
    }

    // 获取指定档位订单数
    int getLevelOrderCount(char side, int level) {
        if (level < 0) return 0;
        auto& targetBook = (side == 'b') ? bids : asks;
        if (level >= (int)targetBook.size()) return 0;
        auto it = targetBook.begin();
        advance(it, level);
        return it->second.size();
    }

    // 获取指定价格的订单列表
    vector<int>* getLevelOrders(char side, double price) {
        auto& targetBook = (side == 'b') ? bids : asks;
        auto priceIt = targetBook.find(price);
        return (priceIt != targetBook.end()) ? &(priceIt->second) : nullptr;
    }

    // 获取订单映射表（只读）
    const unordered_map<int, Order>& getOrderMap() const {
        return orderMap;
    }

    // 新增：获取买盘/卖盘的价格-订单列表映射（供外部遍历优先级）
    const map<double, vector<int>, greater<double>>& getBids() const { return bids; }
    const map<double, vector<int>, less<double>>& getAsks() const { return asks; }
};

// 全局变量定义
double g_trd_size_ewm = 0.0;         // 交易规模EWM（指数加权移动平均）
double g_snapshot_time = 0.0;        // 快照当前时间
const vector<double> TIME_BUCKETS = {0.0, 0.01, 1.0, 10.0, 300.0, 150000.0}; // 时间区间
OrderBook ob_global;                 // 全局订单簿（供OnSnapshot访问）

// 新增：hitgap相关全局统计变量
double last_stamp = 0.0;                  // 上一次的时间戳（初始为0）
double cum_hitgap_B = 0.0;                // 买盘累计hitgap
double cum_hitgap_S = 0.0;                // 卖盘累计hitgap
int num_B = 0;                            // 买盘Add订单数量
int num_S = 0;                            // 卖盘Add订单数量

// log10(hitgap)的区间定义：[-15,-3], [-3,-2], [-2,-1], [-1,0]
const vector<pair<double, double>> LOG10_BUCKETS = {{-15.0, -3.0}, {-3.0, -2.0}, {-2.0, -1.0}, {-1.0, 0.0}};
vector<double> cum_log_hitgap_B;          // 买盘各log区间累计hitgap
vector<double> cum_log_hitgap_S;          // 卖盘各log区间累计hitgap
vector<int> num_log_B;                    // 买盘各log区间订单数
vector<int> num_log_S;                    // 卖盘各log区间订单数

// hitgap的特定区间：(0.004,0.006)、(0.009,0.011)
double cum_004_006_hitgap_B = 0.0;        // 买盘(0.004,0.006)区间累计hitgap
double cum_004_006_hitgap_S = 0.0;        // 卖盘(0.004,0.006)区间累计hitgap
double cum_009_011_hitgap_B = 0.0;        // 买盘(0.009,0.011)区间累计hitgap
double cum_009_011_hitgap_S = 0.0;        // 卖盘(0.009,0.011)区间累计hitgap

// 新增：初始化hitgap统计向量
void initHitgapStats() {
    cum_log_hitgap_B.resize(LOG10_BUCKETS.size(), 0.0);
    cum_log_hitgap_S.resize(LOG10_BUCKETS.size(), 0.0);
    num_log_B.resize(LOG10_BUCKETS.size(), 0);
    num_log_S.resize(LOG10_BUCKETS.size(), 0);
}

// 时间统计结果结构体
struct TimeStats {
    double avgTimeTillNow; // 平均挂单时长（当前时间-挂单时间）
    int totalCount;        // 总订单数
    double totalAmt;       // 总金额（size*price）
    // 各时间区间的统计
    vector<struct {
        int count;         // 区间内订单数
        double amt;        // 区间内总金额
        double avgTime;    // 区间内平均挂单时长
        double countRatio; // 数量占比
        double amtRatio;   // 金额占比
    }> bucketStats;
};

// 修正核心：基于订单簿自身优先级筛选研究订单（移除外部priority依赖）
vector<int> getStudyOrders(const OrderBook& ob, char side, double trd_size_ewm) {
    vector<int> studyOrders;
    int cumulativeSize = 0;

    // 按订单簿天然优先级遍历：买盘降序、卖盘升序
    if (side == 'b') {
        const auto& bids = ob.getBids();
        for (const auto& [price, orderIds] : bids) {
            for (int orderId : orderIds) {
                const auto& orderMap = ob.getOrderMap();
                auto it = orderMap.find(orderId);
                if (it == orderMap.end()) continue;

                int orderSize = it->second.size();
                // 累加size，超过阈值则终止
                if (cumulativeSize + orderSize > trd_size_ewm) {
                    return studyOrders;
                }
                cumulativeSize += orderSize;
                studyOrders.push_back(orderId);
            }
        }
    } else if (side == 's') {
        const auto& asks = ob.getAsks();
        for (const auto& [price, orderIds] : asks) {
            for (int orderId : orderIds) {
                const auto& orderMap = ob.getOrderMap();
                auto it = orderMap.find(orderId);
                if (it == orderMap.end()) continue;

                int orderSize = it->second.size();
                // 累加size，超过阈值则终止
                if (cumulativeSize + orderSize > trd_size_ewm) {
                    return studyOrders;
                }
                cumulativeSize += orderSize;
                studyOrders.push_back(orderId);
            }
        }
    }

    return studyOrders;
}

// 辅助函数：计算研究订单的时间统计指标
TimeStats calculateTimeStats(const OrderBook& ob, const vector<int>& studyOrders, double snapshotTime, const vector<double>& timeBuckets) {
    TimeStats stats;
    stats.totalCount = studyOrders.size();
    stats.totalAmt = 0.0;
    double totalTimeTillNow = 0.0;

    // 初始化区间统计
    stats.bucketStats.resize(timeBuckets.size() - 1);
    for (auto& bucket : stats.bucketStats) {
        bucket.count = 0;
        bucket.amt = 0.0;
        bucket.avgTime = 0.0;
        bucket.countRatio = 0.0;
        bucket.amtRatio = 0.0;
    }

    const auto& orderMap = ob.getOrderMap();
    // 遍历研究订单，累加基础统计
    for (int orderId : studyOrders) {
        auto it = orderMap.find(orderId);
        if (it == orderMap.end()) continue;

        const Order& order = it->second;
        double timeTillNow = snapshotTime - order.timestamp; // 挂单时长
        double amt = order.size * order.price;               // 订单金额

        totalTimeTillNow += timeTillNow;
        stats.totalAmt += amt;

        // 匹配时间区间并更新统计
        for (int i = 0; i < timeBuckets.size() - 1; ++i) {
            double lower = timeBuckets[i];
            double upper = (i == timeBuckets.size()-2) ? 1e9 : timeBuckets[i+1]; // 最后区间上限设为极大值
            if (timeTillNow >= lower && timeTillNow < upper) {
                stats.bucketStats[i].count++;
                stats.bucketStats[i].amt += amt;
                stats.bucketStats[i].avgTime += timeTillNow;
                break;
            }
        }
    }

    // 计算平均挂单时长
    stats.avgTimeTillNow = (stats.totalCount > 0) ? (totalTimeTillNow / stats.totalCount) : 0.0;

    // 计算区间占比和区间内均值
    for (int i = 0; i < stats.bucketStats.size(); ++i) {
        auto& bucket = stats.bucketStats[i];
        // 数量占比
        bucket.countRatio = (stats.totalCount > 0) ? (double)bucket.count / stats.totalCount : 0.0;
        // 金额占比
        bucket.amtRatio = (stats.totalAmt > 0) ? bucket.amt / stats.totalAmt : 0.0;
        // 区间内平均时间
        bucket.avgTime = (bucket.count > 0) ? bucket.avgTime / bucket.count : 0.0;
    }

    return stats;
}

// 新增：获取hitgap统计结果的辅助函数（方便外部调用查看统计值）
void printHitgapStats() {
    cout << "===== Hitgap 统计结果 =====" << endl;
    cout << "累计hitgap(B): " << cum_hitgap_B << ", 累计hitgap(S): " << cum_hitgap_S << endl;
    cout << "买盘订单数: " << num_B << ", 卖盘订单数: " << num_S << endl;
    
    cout << "\nLog10(hitgap)区间统计:" << endl;
    for (int i = 0; i < LOG10_BUCKETS.size(); ++i) {
        cout << "区间 [" << LOG10_BUCKETS[i].first << ", " << LOG10_BUCKETS[i].second << "): " << endl;
        cout << "  买盘累计hitgap: " << cum_log_hitgap_B[i] << ", 订单数: " << num_log_B[i] << endl;
        cout << "  卖盘累计hitgap: " << cum_log_hitgap_S[i] << ", 订单数: " << num_log_S[i] << endl;
    }

    cout << "\n特定hitgap区间统计:" << endl;
    cout << "(0.004,0.006)区间 - 买盘: " << cum_004_006_hitgap_B << ", 卖盘: " << cum_004_006_hitgap_S << endl;
    cout << "(0.009,0.011)区间 - 买盘: " << cum_009_011_hitgap_B << ", 卖盘: " << cum_009_011_hitgap_S << endl;
    cout << "===========================" << endl;
}

// 消息处理函数（补全Add/Trd/Cxl逻辑）
void OnMsgChange(const Msg& msg, OrderBook& ob, const string& event) {
    if (msg.MSG == "Del"){
        ob.deleteOrder(msg.ORDER_ID); // 处理取消订单
        return;
    }
    if (msg.MSG == "Add"){
        // hitgap = msg.exch_timestamp - last_stamp
        // cum_hitgap_B += hitgap if msg.side=='B' else 0

        // 判断当前np.log10(hitgap) 是否在[-15, -3, -2, -1, 0] 各个区间内，如果在，cum_{bi}_{bj}_hitgap_B += hitgap 以及S
    
        // 判断当前hitgap 是否在[(0.004, 0.006), (0.009, 0.011)] 这两个区间内，如果在对应的cum_{bi}_{bj}_hitgap_B += hitgap 以及S
        // 返回cum_hitgap_B, cum_hitgap_S, num_B, num_S, cum_{bi}_{bj}_hitgap_B, cum_{bi}_{bj}_hitgap_S, cum_004_006_hitgap_B,  cum_004_006_hitgap_S,
        // cum_009_011_hitgap_B, cum_009_011_hitgap_S
        // last_stamp = exch_timestamp
        
        // 1. 新增订单到订单簿
        ob.newOrder(msg.ORDER_ID, msg.SIDE, msg.SIZE, msg.PRICE, msg.TIME);

        // 2. 计算hitgap（当前时间-上一次时间戳）
        double hitgap = 0.0;
        if (last_stamp != 0.0) {  // 非第一次Add消息才计算有效hitgap
            hitgap = msg.TIME - last_stamp;
        }

        // 3. 累计总hitgap和订单数
        if (msg.SIDE == 'b') {
            cum_hitgap_B += hitgap;
            num_B++;
        } else if (msg.SIDE == 's') {
            cum_hitgap_S += hitgap;
            num_S++;
        }

        // 4. 按log10(hitgap)区间统计（避免log10(0)或负数）
        if (hitgap > 0.0) {
            double log_hitgap = log10(hitgap);
            for (int i = 0; i < LOG10_BUCKETS.size(); ++i) {
                double lower = LOG10_BUCKETS[i].first;
                double upper = LOG10_BUCKETS[i].second;
                if (log_hitgap >= lower && log_hitgap < upper) {
                    if (msg.SIDE == 'b') {
                        cum_log_hitgap_B[i] += hitgap;
                        num_log_B[i]++;
                    } else if (msg.SIDE == 's') {
                        cum_log_hitgap_S[i] += hitgap;
                        num_log_S[i]++;
                    }
                    break; // 每个hitgap只匹配一个区间
                }
            }
        }

        // 5. 按特定hitgap区间统计
        // (0.004, 0.006)区间
        if (hitgap > 0.004 && hitgap < 0.006) {
            if (msg.SIDE == 'b') {
                cum_004_006_hitgap_B += hitgap;
            } else if (msg.SIDE == 's') {
                cum_004_006_hitgap_S += hitgap;
            }
        }
        // (0.009, 0.011)区间
        if (hitgap > 0.009 && hitgap < 0.011) {
            if (msg.SIDE == 'b') {
                cum_009_011_hitgap_B += hitgap;
            } else if (msg.SIDE == 's') {
                cum_009_011_hitgap_S += hitgap;
            }
        }

        // 6. 更新上一次时间戳为当前消息时间
        last_stamp = msg.TIME;
        return;
    }
    if (msg.MSG == "Trd"){
        // 更新交易规模EWM：lambda = 0.5^delta_t
        static double last_trd_time = 0.0;
        double delta_t = (last_trd_time == 0.0) ? 1.0 : (msg.TIME - last_trd_time);
        double lambda = pow(0.5, delta_t);
        g_trd_size_ewm = g_trd_size_ewm * lambda + msg.SIZE;
        last_trd_time = msg.TIME;
        return;
    }
    if (msg.MSG == "Replace"){
        ob.modifyOrder(msg.ORDER_ID, msg.ORDER_ID, msg.SIDE, msg.SIZE, msg.PRICE);
        return;
    }
}

// 快照统计核心函数（基于订单簿天然优先级实现）
void OnSnapshot(float snapshot_time = 34200.10) {
    g_snapshot_time = snapshot_time;
    // 可在此处扩展快照时的统计逻辑
    return;
}

// 主函数（测试示例）
int main() {
    // 初始化hitgap统计向量
    initHitgapStats();

    // 初始化测试订单（调整时间戳以产生有效hitgap）
    Msg testMsg1 = {"2022-08-02", 34200.05, "701.hk", "Add", 's', 254.2, 100, 28416};  // hitgap=0（首次）
    Msg testMsg2 = {"2022-08-02", 34200.06, "708.hk", "Add", 's', 262, 100, 190137};   // hitgap=0.01
    Msg testMsg3 = {"2022-08-02", 34200.055, "711.hk", "Add", 'b', 265.4, 100, 190171};// hitgap=-0.005（负数，log统计跳过）
    Msg testMsg4 = {"2022-08-02", 34200.058, "711.hk", "Add", 'b', 265.5, 200, 190172};// hitgap=0.003
    Msg testMsg5 = {"2022-08-02", 34200.06, "711.hk", "Trd", 'b', 265.4, 250, 0};      // 交易事件（trd_size_ewm=250）

    // 处理消息
    OnMsgChange(testMsg1, ob_global, "Add");
    OnMsgChange(testMsg2, ob_global, "Add");
    OnMsgChange(testMsg3, ob_global, "Add");
    OnMsgChange(testMsg4, ob_global, "Add");
    OnMsgChange(testMsg5, ob_global, "Trd");

    // 执行快照统计
    OnSnapshot();

    // 打印hitgap统计结果
    printHitgapStats();

    return 0;
}