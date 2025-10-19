#include <bits/stdc++.h>
using namespace std;

struct Order {
    int orderId;
    char side;   // 'b' = buy, 's' = sell
    int size;
    double price;
};

class OrderBook {
private:
    // 订单映射: orderId -> Order
    unordered_map<int, Order> orderMap;

    // 买卖盘: price -> list of orders
    // 买盘需要按价格从大到小排序 (greater)
    // 卖盘需要按价格从小到大排序 (less)
    map<double, vector<int>, greater<double>> bids;
    map<double, vector<int>, less<double>> asks;

    // 获取某一边的引用
    map<double, vector<int>, greater<double>>& getBidBook() { return bids; }
    map<double, vector<int>, less<double>>& getAskBook() { return asks; }

public:
    /* 新增订单到订单簿 */
    void newOrder(int orderId, char side, int size, double price) {
        Order o{orderId, side, size, price};
        orderMap[orderId] = o;
        if (side == 'b')
            bids[price].push_back(orderId);
        else
            asks[price].push_back(orderId);
    }

    /* 减少指定订单的数量 */
    void reduceOrder(int orderId, int newSize) {
        if (!orderMap.count(orderId)) return;
        orderMap[orderId].size = newSize;
        if (newSize == 0) deleteOrder(orderId);
    }

    /* 修改订单（先删旧单，再挂新单） */
    void modifyOrder(int oldOrderId,
                     int orderId,
                     char side,
                     int size,
                     double price) {
        deleteOrder(oldOrderId);
        newOrder(orderId, side, size, price);
    }

    /* 从订单簿删除订单 */
    void deleteOrder(int orderId) {
        if (!orderMap.count(orderId)) return;
        Order o = orderMap[orderId];
        if (o.side == 'b') {
            auto &vec = bids[o.price];
            vec.erase(remove(vec.begin(), vec.end(), orderId), vec.end());
            if (vec.empty()) bids.erase(o.price);
        } else {
            auto &vec = asks[o.price];
            vec.erase(remove(vec.begin(), vec.end(), orderId), vec.end());
            if (vec.empty()) asks.erase(o.price);
        }
        orderMap.erase(orderId);
    }

    /* 返回某一边（买/卖）的价格档位数 */
    int getNumLevels(char side) {
        if (side == 'b') return bids.size();
        else return asks.size();
    }

    /* 返回某一边指定档位的价格（level 0 为 top-of-book） */
    double getLevelPrice(char side, int level) {
        if (level < 0) return NAN;
        if (side == 'b') {
            if (level >= (int)bids.size()) return NAN;
            auto it = bids.begin();
            advance(it, level);
            return it->first;
        } else {
            if (level >= (int)asks.size()) return NAN;
            auto it = asks.begin();
            advance(it, level);
            return it->first;
        }
    }

    /* 返回某一边指定档位的总数量 */
    int getLevelSize(char side, int level) {
        if (level < 0) return 0;
        if (side == 'b') {
            if (level >= (int)bids.size()) return 0;
            auto it = bids.begin();
            advance(it, level);
            int total = 0;
            for (int id : it->second) total += orderMap[id].size;
            return total;
        } else {
            if (level >= (int)asks.size()) return 0;
            auto it = asks.begin();
            advance(it, level);
            int total = 0;
            for (int id : it->second) total += orderMap[id].size;
            return total;
        }
    }

    /* 返回某一边指定档位包含的订单笔数 */
    int getLevelOrderCount(char side, int level) {
        if (level < 0) return 0;
        if (side == 'b') {
            if (level >= (int)bids.size()) return 0;
            auto it = bids.begin();
            advance(it, level);
            return it->second.size();
        } else {
            if (level >= (int)asks.size()) return 0;
            auto it = asks.begin();
            advance(it, level);
            return it->second.size();
        }
    }
};


