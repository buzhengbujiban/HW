#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iterator>
#include <string>
using namespace std;

// �����ṹ�嶨��
struct Order {
    int orderId;
    char side;       // 'b' = buy, 's' = sell
    int size;
    double price;
};

// ��Ϣ�ṹ�嶨�壨���豣�������ֶΣ�
struct Msg {
    string DATE;
    double TIME;
    string RIC;
    string MSG;      // �¼����ͣ�Cxl/ADD/DEL�ȣ�
    char SIDE;       // ��������
    double PRICE;    // �����۸�
    int SIZE;        // ��������
    int ORDER_ID;    // ����ID
    // �����޹��ֶοɸ���ʵ�ʳ�������
};

// �������ࣨ�޸ķ���Ȩ�ޣ�������Ҫ�ӿڣ�
class OrderBook {
private:
    unordered_map<int, Order> orderMap;
    map<double, vector<int>, greater<double>> bids;  // ���̣��۸���
    map<double, vector<int>, less<double>> asks;     // ���̣��۸�����

public:
    // ��������
    void newOrder(int orderId, char side, int size, double price) {
        Order o{orderId, side, size, price};
        orderMap[orderId] = o;
        if (side == 'b') bids[price].push_back(orderId);
        else asks[price].push_back(orderId);
    }

    // ���ٶ�������
    void reduceOrder(int orderId, int newSize) {
        if (!orderMap.count(orderId)) return;
        orderMap[orderId].size = newSize;
        if (newSize == 0) deleteOrder(orderId);
    }

    // �޸Ķ�����ɾ�ɹ��£�
    void modifyOrder(int oldOrderId, int orderId, char side, int size, double price) {
        deleteOrder(oldOrderId);
        newOrder(orderId, side, size, price);
    }

    // ɾ������
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

    // ��ȡ��λ����
    int getNumLevels(char side) {
        return (side == 'b') ? bids.size() : asks.size();
    }

    // ��ȡָ����λ�۸�
    double getLevelPrice(char side, int level) {
        if (level < 0) return NAN;
        auto& targetBook = (side == 'b') ? bids : asks;
        if (level >= (int)targetBook.size()) return NAN;
        auto it = targetBook.begin();
        advance(it, level);
        return it->first;
    }

    // ��ȡָ����λ������
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

    // ��ȡָ����λ��������
    int getLevelOrderCount(char side, int level) {
        if (level < 0) return 0;
        auto& targetBook = (side == 'b') ? bids : asks;
        if (level >= (int)targetBook.size()) return 0;
        auto it = targetBook.begin();
        advance(it, level);
        return it->second.size();
    }

    // ��������ȡָ���۸�λ�Ķ����б������ⲿ��ѯ��
    vector<int>* getLevelOrders(char side, double price) {
        auto& targetBook = (side == 'b') ? bids : asks;
        auto priceIt = targetBook.find(price);
        return (priceIt != targetBook.end()) ? &(priceIt->second) : nullptr;
    }

    // ��������ȡ����ӳ�䣨���ⲿ��ѯ�������飩
    const unordered_map<int, Order>& getOrderMap() const {
        return orderMap;
    }
};

// ȫ�ֱ������洢1������������Cxl_priority_globָ��
vector<double> g_buy_cxl_priority;
vector<double> g_sell_cxl_priority;

// �����¼���������
void OnMsgChange(const Msg& msg, const OrderBook& ob, const string& event) {
    // ������Cxl�¼�
    if (msg.MSG == "Cxl"){
        
        return;
    }
    if (msg.MSG == "Add"){

        return;
    }

}

// ����ͳ�ƺ�����ÿ���Ӵ�����
// ����ͳ�ƺ�����ÿ���Ӵ�����
/**
 * @brief  订单簿快照统计函数：计算买卖1档位订单优先级偏度（skew）的归一化差值
 * @details 核心功能是针对订单簿最优档位（1档位，对应level 0），分别计算买方、卖方订单的优先级加权偏度，
 *          最终通过归一化公式得到一个反映买卖盘口订单队列结构差异的指标，用于量化盘口的供需或队列拥挤程度。
 * 
 * 关键术语说明：
 * 1.  1档位：订单簿中最优价格档位（买方最高买价、卖方最低卖价），对应代码中level=0（函数getLevelXxx的level从0开始计数）
 * 2.  priority（优先级）：全局变量g_buy_cxl_priority/g_sell_cxl_priority中存储的订单属性，代表订单在同价格队列中的先后位置（距离盘口的远近），
 *     priority值越小，订单越靠近盘口（成交优先级越高）
 * 3.  权重：对应每个订单的size（订单数量），即计算skew时，订单数量越多，对该档位整体skew的贡献越大
 * 4.  priority_skew（优先级偏度）：以订单size为权重，对同档位所有订单的priority进行加权统计得到的偏度值（反映优先级分布的不对称性）
 * 5.  最终归一化公式：(买方偏度 - 卖方偏度) / (买方偏度 + 卖方偏度)，用于将结果映射到[-1, 1]区间，方便后续分析
 */
void OnSnapshot() {
    // ===================== 步骤1：初始化核心变量 =====================
    // 买方1档位优先级加权偏度（B = Buy）
    double B_priority_skew = 0.0;
    // 卖方1档位优先级加权偏度（S = Sell）
    double S_priority_skew = 0.0;
    // 买卖档位总订单量（用于计算加权平均，作为偏度计算的中间变量）
    int total_buy_size = 0;
    int total_sell_size = 0;

    // ===================== 步骤2：计算买方1档位（level=0）的priority_skew =====================
    // 2.1 获取买方1档位的价格（最优买价）
    double best_buy_price = getLevelPrice('b', 0);
    // 2.2 获取该价格档位下的所有订单ID列表
    vector<int>* buy_order_ids = getLevelOrders('b', best_buy_price);
    if (buy_order_ids != nullptr && !buy_order_ids->empty()) {
        // 2.3 遍历该档位所有订单，累加总订单量（用于后续加权计算）
        for (int order_id : *buy_order_ids) {
            // 从订单映射表中获取订单详情，得到该订单的size
            auto& order_map = getOrderMap();
            auto order_it = order_map.find(order_id);
            if (order_it != order_map.end()) {
                total_buy_size += order_it->second.size;
            }
        }

        // 2.4 以size为权重，计算买方优先级加权偏度（此处为核心计算逻辑，需结合g_buy_cxl_priority的存储规则实现）
        // 注：g_buy_cxl_priority需与订单ID/订单对应，此处假设已建立有效映射，仅展示逻辑框架
        for (int order_id : *buy_order_ids) {
            auto& order_map = getOrderMap();
            auto order_it = order_map.find(order_id);
            if (order_it != order_map.end() && total_buy_size > 0) {
                int order_size = order_it->second.size;
                double order_priority = 0.0; // 从g_buy_cxl_priority中获取该订单对应的priority值
                // 加权累加：订单size占总size的比例 * 该订单priority（偏度计算的基础步骤，可根据具体skew公式调整）
                B_priority_skew += (static_cast<double>(order_size) / total_buy_size) * order_priority;
            }
        }
    }

    // ===================== 步骤3：计算卖方1档位（level=0）的priority_skew =====================
    // 3.1 获取卖方1档位的价格（最优卖价）
    double best_sell_price = getLevelPrice('s', 0);
    // 3.2 获取该价格档位下的所有订单ID列表
    vector<int>* sell_order_ids = getLevelOrders('s', best_sell_price);
    if (sell_order_ids != nullptr && !sell_order_ids->empty()) {
        // 3.3 遍历该档位所有订单，累加总订单量（用于后续加权计算）
        for (int order_id : *sell_order_ids) {
            auto& order_map = getOrderMap();
            auto order_it = order_map.find(order_id);
            if (order_it != order_map.end()) {
                total_sell_size += order_it->second.size;
            }
        }

        // 3.4 以size为权重，计算卖方优先级加权偏度（与买方逻辑一致）
        for (int order_id : *sell_order_ids) {
            auto& order_map = getOrderMap();
            auto order_it = order_map.find(order_id);
            if (order_it != order_map.end() && total_sell_size > 0) {
                int order_size = order_it->second.size;
                double order_priority = 0.0; // 从g_sell_cxl_priority中获取该订单对应的priority值
                S_priority_skew += (static_cast<double>(order_size) / total_sell_size) * order_priority;
            }
        }
    }

    // ===================== 步骤4：计算最终归一化结果 =====================
    double final_result = 0.0;
    // 避免分母为0的除法错误（当买卖偏度之和为0时，结果直接置为0）
    if (fabs(B_priority_skew + S_priority_skew) > 1e-9) {
        final_result = (B_priority_skew - S_priority_skew) / (B_priority_skew + S_priority_skew);
    }

    // ===================== 步骤5：后续处理（可选） =====================
    // 此处可添加结果输出、存储或进一步分析的逻辑
    // 例如：cout << "最终归一化结果：" << final_result << endl;
}

// ������������ѡ��
int main() {
    // ģ�ⶩ�������¼����ͣ�ʵ��ʹ��ʱ�滻Ϊ��ʵ�������룩
    OrderBook ob;
    Msg testMsg1 = {"2022-08-02", 34200.05, "701.hk", "Cxl", 's', 254.2, 100, 28416};
    Msg testMsg2 = {"2022-08-02", 34200.05, "708.hk", "Cxl", 's', 262, 100, 190137};
    Msg testMsg3 = {"2022-08-02", 34200.05, "711.hk", "Cxl", 'b', 265.4, 100, 190171};

    // �����Ӳ��Զ�����������
    ob.newOrder(28416, 's', 100, 254.2);
    ob.newOrder(190137, 's', 100, 262);
    ob.newOrder(190171, 'b', 100, 265.4);
    ob.newOrder(190172, 'b', 200, 265.4);  // ����ͬ��λ���������ڲ���ǰ��sum

    // ����Cxl�¼�
    OnMsgChange(testMsg1, ob, "Cxl");
    OnMsgChange(testMsg2, ob, "Cxl");
    OnMsgChange(testMsg3, ob, "Cxl");

    // ��������ͳ��
    OnSnapshot();

    return 0;
}
