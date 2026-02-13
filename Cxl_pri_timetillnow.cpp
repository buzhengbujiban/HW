#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iterator>
#include <string>
#include <cmath>
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

// 全局变量定义（移除了多余的priority向量）
double g_trd_size_ewm = 0.0;         // 交易规模EWM（指数加权移动平均）
double g_snapshot_time = 0.0;        // 快照当前时间
const vector<double> TIME_BUCKETS = {0.0, 0.01, 1.0, 10.0, 300.0, 150000.0}; // 时间区间
OrderBook ob_global;                 // 全局订单簿（供OnSnapshot访问）

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

// 消息处理函数（补全Add/Trd/Cxl逻辑）
void OnMsgChange(const Msg& msg, OrderBook& ob, const string& event) {
    if (msg.MSG == "Cxl"){
        ob.deleteOrder(msg.ORDER_ID); // 处理取消订单
        return;
    }
    if (msg.MSG == "Add"){
        // 新增订单：记录挂单时间
        ob.newOrder(msg.ORDER_ID, msg.SIDE, msg.SIZE, msg.PRICE, msg.TIME);
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
}

// 快照统计核心函数（基于订单簿天然优先级实现）
void OnSnapshot(float g_snapshot_time = 34200.10) {
    // ===================== 步骤1：初始化核心变量 =====================
    // 按照priority 在B,S order_id_queue 分别查找到 cumsize<=trd_size_ewm 的最后一个订单，并将前面的订单视为B,S set （研究对象集合）
    // 获取B,S 研究对象集合中订单的time_till_now（挂单时间-当前时间）的平均值。 并返回time_till_now_B, time_till_now_S, num_B, num_S
    // 找出B,S 研究对象集合中订单的挂单时间在区间[0, 0.01, 1, 10, 300, 150000] 的订单数量占B/S 订单数量的百分比，满足这些区间的订单amt(size*price)占总B/S 分别amt的比例，time_till_now每个区间订单的均值,分别返回这些B/S 的值

    // 步骤1：初始化核心变量
    // 示例快照时间（实际应从业务获取）
    const vector<double>& timeBuckets = TIME_BUCKETS;

    // 步骤2：筛选B/S研究订单集合（直接用订单簿自身优先级）
    vector<int> buyStudyOrders = getStudyOrders(ob_global, 'b', g_trd_size_ewm);
    vector<int> sellStudyOrders = getStudyOrders(ob_global, 's', g_trd_size_ewm);

    // 步骤3：计算时间统计指标
    TimeStats buyStats = calculateTimeStats(ob_global, buyStudyOrders, g_snapshot_time, timeBuckets);
    TimeStats sellStats = calculateTimeStats(ob_global, sellStudyOrders, g_snapshot_time, timeBuckets);

    // 提取核心返回值
    double time_till_now_B = buyStats.avgTimeTillNow;
    double time_till_now_S = sellStats.avgTimeTillNow;
    int num_B = buyStats.totalCount;
    int num_S = sellStats.totalCount;

    // 步骤4：输出统计结果（实际可改为返回/存储）
    cout << "===== 快照统计结果 =====" << endl;
    // 买单统计
    cout << "\n【买单(B) - 价格降序优先级】" << endl;
    cout << "平均挂单时长: " << time_till_now_B << " 秒" << endl;
    cout << "订单总数: " << num_B << endl;
    cout << "总金额: " << buyStats.totalAmt << endl;

    // 买单时间区间统计
    cout << "\n买单时间区间统计：" << endl;
    for (int i = 0; i < timeBuckets.size()-1; ++i) {
        double lower = timeBuckets[i];
        double upper = (i == timeBuckets.size()-2) ? 150000.0 : timeBuckets[i+1];
        auto& bucket = buyStats.bucketStats[i];
        cout << "区间 [" << lower << ", " << upper << "): " << endl;
        cout << "  数量: " << bucket.count << " (占比: " << bucket.countRatio*100 << "%)" << endl;
        cout << "  金额: " << bucket.amt << " (占比: " << bucket.amtRatio*100 << "%)" << endl;
        cout << "  区间均值: " << bucket.avgTime << " 秒" << endl;
    }

    // 卖单统计
    cout << "\n【卖单(S) - 价格升序优先级】" << endl;
    cout << "平均挂单时长: " << time_till_now_S << " 秒" << endl;
    cout << "订单总数: " << num_S << endl;
    cout << "总金额: " << sellStats.totalAmt << endl;

    // 卖单时间区间统计
    cout << "\n卖单时间区间统计：" << endl;
    for (int i = 0; i < timeBuckets.size()-1; ++i) {
        double lower = timeBuckets[i];
        double upper = (i == timeBuckets.size()-2) ? 150000.0 : timeBuckets[i+1];
        auto& bucket = sellStats.bucketStats[i];
        cout << "区间 [" << lower << ", " << upper << "): " << endl;
        cout << "  数量: " << bucket.count << " (占比: " << bucket.countRatio*100 << "%)" << endl;
        cout << "  区间均值: " << bucket.avgTime << " 秒" << endl;
    }
}

// 主函数（测试示例）
int main() {
    // 初始化测试订单
    Msg testMsg1 = {"2022-08-02", 34200.05, "701.hk", "Add", 's', 254.2, 100, 28416};
    Msg testMsg2 = {"2022-08-02", 34200.05, "708.hk", "Add", 's', 262, 100, 190137};
    Msg testMsg3 = {"2022-08-02", 34200.05, "711.hk", "Add", 'b', 265.4, 100, 190171};
    Msg testMsg4 = {"2022-08-02", 34200.05, "711.hk", "Add", 'b', 265.5, 200, 190172}; // 更高价格的买单（优先级更高）
    Msg testMsg5 = {"2022-08-02", 34200.06, "711.hk", "Trd", 'b', 265.4, 250, 0};      // 交易事件（trd_size_ewm=250）

    // 处理消息
    OnMsgChange(testMsg1, ob_global, "Add");
    OnMsgChange(testMsg2, ob_global, "Add");
    OnMsgChange(testMsg3, ob_global, "Add");
    OnMsgChange(testMsg4, ob_global, "Add");
    OnMsgChange(testMsg5, ob_global, "Trd");

    // 执行快照统计
    OnSnapshot();

    return 0;
}