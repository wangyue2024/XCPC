# 问题： 对于给出的1000个平面上的圆，判断它们是否存在公共交点（即，是否存在一个点被所有圆包含）
好的，这是一个在信息学竞赛（尤其是计算几何领域）中非常经典的问题。对于给出的1000个平面上的圆，判断它们是否存在公共交点（即，是否存在一个点被所有圆包含），答案是：**有，并且有高效的算法可以解决**。

这个问题等价于“判断N个凸集（圆盘）的交集是否非空”。下面我将介绍几种解决此问题的算法，从简单直观到高效实用。

---

### 问题分析

首先，我们把问题形式化。每个圆 Cᵢ 由其圆心 (xᵢ, yᵢ) 和半径 rᵢ 定义。一个点 P(x, y) 被圆 Cᵢ 包含的条件是：
(x - xᵢ)² + (y - yᵢ)² ≤ rᵢ²

我们需要找到一个点 P(x, y) 使得对于 **所有** i = 1, 2, ..., 1000，这个不等式都成立。

所有圆的交集是一个**凸集**（因为每个圆本身是凸集，而凸集的交集仍然是凸集）。我们需要判断这个最终的凸集是否为空。

---

### 算法一：基于候选点的暴力枚举（O(N³)）

**核心思想**：
如果所有圆的交集区域 `S` 非空，那么 `S` 也是一个凸集。这个凸集的边界是由某些圆的圆弧组成的。`S` 区域的“顶点”（即不同圆弧的连接点）必然是某两个圆的交点。

此外，整个交集区域 `S` 中 y 坐标最小的点，一定满足下面两个条件之一：

1. 它是某两个圆的边界的交点。
2. 它是某个圆自身的最低点 (xᵢ, yᵢ - rᵢ)。

这个性质为我们提供了一个有限的“候选点”集合。我们只需要检查这些候选点中，有没有一个点同时被所有1000个圆包含。

**候选点集**：

1. **所有圆两两之间的交点**：对于任意两个圆 Cᵢ 和 Cⱼ，计算它们的交点（0个、1个或2个）。
2. **所有圆各自的最低点**：对于每个圆 Cᵢ，其最低点为 (xᵢ, yᵢ - rᵢ)。

**算法步骤**：

1. 创建一个空的候选点列表。
2. 对于每对不同的圆 (Cᵢ, Cⱼ)，计算它们的交点。如果存在交点，将它们加入候选点列表。
3. （为严谨起见）对于每个圆 Cᵢ，将其最低点 (xᵢ, yᵢ - rᵢ) 加入候选点列表。
4. 遍历候选点列表中的每一个点 `P`。
5. 对于每个点 `P`，检查它是否被 **所有1000个** 圆包含。
6. 如果在第5步中找到了这样一个点 `P`，那么所有圆存在公共交点，算法结束，返回“是”。
7. 如果遍历完所有候选点都没有找到满足条件的点，则公共交点不存在，返回“否”。

**复杂度分析**：

- 圆的对数约为 N²/2。每对圆的交点计算是 O(1) 的。因此，候选点的数量是 O(N²)。
- 对于 O(N²) 个候选点，每个点需要与 N 个圆进行检查。
- 总时间复杂度为 **O(N³) **。
- 对于 N=1000，1000³ = 10⁹，这个算法**太慢了**，在信息学竞赛中通常会超时。

---

### 算法二：基于海利定理（Helly's Theorem）的方法 (O(N³))

**核心思想**：
海利定理是一个优美的几何定理，它表明：

> 在 d 维空间中，有一组凸集。如果这组凸集中的任意 d+1 个集合的交集都非空，那么整个集合的交集也非空。

在我们的二维平面问题中 (d=2)，圆盘是凸集。因此，海利定理告诉我们：

> 如果任意 **3** 个圆的交集都非空，那么这 **1000** 个圆的交集也非空。

**算法步骤**：

1. 遍历所有三个圆的组合 (Cᵢ, Cⱼ, Cₖ)。
2. 对于每一组三个圆，判断它们的交集是否为空。
3. **如何判断3个圆的交集是否为空？** 这可以利用算法一的思想：找出这3个圆两两之间的交点（最多6个），然后检查这些交点中是否有任何一个被全部3个圆所包含。如果没有任何交点满足，还需检查是否存在某个圆完全包含在另外两个圆的交集中的情况（例如，检查其中一个圆的圆心是否在另外两个圆内）。
4. 如果在遍历中发现 **任何** 一组三个圆的交集为空，那么根据海利定理（的逆否命题），这1000个圆的交集一定也为空。算法结束，返回“否”。
5. 如果所有三元组的交集都非空，那么根据海利定理，1000个圆的交集也非空。算法结束，返回“是”。

**复杂度分析**：

- 从 N 个圆中选出 3 个的组合数是 C(N, 3) = N(N-1)(N-2)/6，即 O(N³)。
- 判断 3 个圆是否有交集是一个常数时间 O(1) 的操作。
- 总时间复杂度为 **O(N³) **。
- 这个算法虽然和算法一的复杂度一样，但常数更小，且逻辑上更优美。不过对于 N=1000，它仍然处在超时的边缘，风险很高。

---

### 算法三：随机增量法 (Expected O(N))

这是一种更高级的算法，类似于求解“最小覆盖圆”的 Welzl 算法。

**核心思想**：
随机打乱圆的顺序。依次加入每个圆，并维护当前已加入圆的交集中的一个“可行点”。

1. 设 P₀ 是第一个圆 C₁ 的圆心。
2. 处理到第 i 个圆 Cᵢ 时，我们有一个可行点 Pᵢ₋₁ 它是前 i-1 个圆交集中的一个点。
3. 检查 Pᵢ₋₁ 是否在 Cᵢ 内部。
   - 如果在，那么 Pᵢ₋₁ 也是前 i 个圆交集中的点，令 Pᵢ = Pᵢ₋₁，继续处理下一个圆。
   - 如果不在，说明新的可行点（如果存在）**必定在 Cᵢ 的边界上**。问题转化为一个子问题：在前 i-1 个圆的交集中，寻找一个在 Cᵢ 边界上的点。
4. 这个子问题可以递归解决，问题的维度每次减一（从找一个2D点，到找一个在1D圆弧上的点）。

**复杂度分析**：

- 通过精巧的数学证明，可以得出此算法的期望时间复杂度为 **O(N)**。
- 尽管期望复杂度很低，但实现起来非常复杂，需要处理递归、边界情况等，不适合在紧张的比赛中现场编写。

---

### 算法四：三分法（Ternary Search）降维 (O(N log C)) - **推荐算法**

这是解决此类问题在信息学竞赛中的**标准高效解法**。

**核心思想**：
我们将二维搜索问题降维成一维搜索问题。

1. 我们不直接找一个点 (x, y)，而是先尝试确定 x 坐标。
2. 对于一个**固定**的 x 坐标，一个点要被所有圆包含，它的 y 坐标必须满足什么条件？
   - 对于每个圆 Cᵢ，(x - xᵢ)² + (yᵢ - y)² ≤ rᵢ²  => (yᵢ - y)² ≤ rᵢ² - (x - xᵢ)²。
   - 如果 rᵢ² - (x - xᵢ)² < 0，说明这个 x 坐标在圆 Cᵢ 的投影范围之外，此时不可能有交点。
   - 否则，y 必须满足 `yᵢ - sqrt(rᵢ² - (x - xᵢ)²) ≤ y ≤ yᵢ + sqrt(rᵢ² - (x - xᵢ)²) `。
   - 这为每个圆都定义了一个关于 y 的可行区间 `[y_downᵢ(x), y_upᵢ(x)]`。
3. 要满足所有圆的条件，y 必须在**所有这些区间的交集**中。所有区间的交集是 `[max(y_downᵢ(x)), min(y_upᵢ(x))]`。
4. 因此，对于一个固定的 x，存在公共交点的充要条件是 `max(y_downᵢ(x)) ≤ min(y_upᵢ(x))`。
5. 我们定义一个函数 `f(x) = min(y_upᵢ(x)) - max(y_downᵢ(x))`。当 `f(x) ≥ 0` 时，在 x 处存在一个满足条件的 y。我们的目标就是找到是否存在一个 x，使得 `f(x) ≥ 0`。
6. **关键性质**：函数 `y_upᵢ(x)` 是凹函数（像倒置的碗），`y_downᵢ(x)` 是凸函数（像正放的碗）。`min` of 凹函数仍然是凹函数，`max` of 凸函数仍然是凸函数。因此 `f(x)` 是一个凹函数减去一个凸函数，这类函数不一定是单峰的。

**修正和简化思路**：
函数 `h(x) = min_i(y_up_i(x))` 是凹函数（下凸）。函数 `g(x) = max_i(y_down_i(x))` 是凸函数（上凸）。我们要寻找是否存在 `x`使得 `h(x) >= g(x)`。
函数 `F(x) = h(x) - g(x)`是一个凹函数，因此它只有一个最大值（即函数是**单峰**的）。
对于单峰函数，我们可以使用**三分法（Ternary Search）**来找到它的最大值。

**算法步骤**：

1. 确定 x 的搜索范围 `[L, R]`。L 可以是所有 `xᵢ - rᵢ` 的最小值，R 可以是所有 `xᵢ + rᵢ` 的最大值。
2. 在这个范围 `[L, R]` 内对 x 进行三分搜索。
3. 在三分法的每一步，我们取两个三等分点 `m1 = L + (R-L)/3` 和 `m2 = R - (R-L)/3`。
4. 分别计算 `f(m1)` 和 `f(m2)`。
   - 计算 `f(x)` 的过程需要遍历所有 N 个圆，复杂度 O(N)。
5. 比较 `f(m1)` 和 `f(m2)` 的大小，根据比较结果缩小搜索范围 `[L, R]`。
6. 重复迭代足够多次（例如 100 次，这能保证很高的精度）。
7. 最后，在最佳的 x 点（近似），检查 `f(x)` 是否大于等于 0（考虑浮点误差，通常是 `f(x) >= -epsilon`）。如果是，则存在公共交点；否则不存在。

**复杂度分析**：

- 三分法迭代次数是固定的，与精度有关，可以看作一个常数（比如 100），或者写作 `log C`，其中 C 是坐标范围与精度的比值。
- 每次迭代内部计算 `f(x)` 需要遍历 N 个圆，复杂度 O(N)。
- 总时间复杂度为 **O(N log C)**。
- 对于 N=1000，这个算法非常快，是信息学竞赛中的完美解决方案。

---

### 总结与推荐

| 算法             | 时间复杂度           | N=1000时的可行性 | 实现难度       | 推荐度                |
| :--------------- | :------------------- | :--------------- | :------------- | :-------------------- |
| 候选点法         | O(N³)               | 不可行           | 中等           | ★☆☆☆☆            |
| 海利定理法       | O(N³)               | 边缘/风险高      | 中等           | ★★☆☆☆            |
| 随机增量法       | O(N) (期望)          | 非常快           | 困难           | ★★★☆☆ (理论优美) |
| **三分法** | **O(N log C)** | **非常快** | **中等** | ★★★★★            |

对于信息学竞赛，**三分法**无疑是解决此问题的最佳选择。它在效率、稳定性和实现难度之间取得了完美的平衡。你需要熟悉计算几何基础（点、圆关系）和三分搜索算法。在实现时，要特别注意处理浮点数的精度问题。



好的，没有问题。这是一个为XCPC竞赛准备的、基于您提供的计算几何基础模板的 `namespace`。它包含了解决“N个圆是否有公共交点”问题的四种不同方法，以及必要的辅助函数。

**核心思路**：

1. **`solve_brute_force_O_N3`**: 实现了基于候选点（圆与圆的交点、圆心）的 O(N³) 暴力算法。
2. **`solve_helly_theorem_O_N3`**: 实现了基于海利定理的 O(N³) 算法，检查所有三元组。
3. **`solve_randomized_incremental_O_N`**: 实现了理论上期望 O(N) 的随机增量算法。**注意：此算法实现复杂，容易出错，不建议在比赛中现场编写，但作为模板很有价值。**
4. **`solve_ternary_search_O_N_log_C`**: 实现了最实用、最推荐的 O(N log C) 三分搜索降维算法。

这个 `namespace`可以直接复制粘贴到您的代码中，并与您已有的几何模板无缝协作。

```cpp
namespace CircleIntersection {

    // 使用 struct 内部表示圆，更清晰
    struct Circle {
        P center;
        D r;
    };

    // 辅助函数：判断一个点是否在单个圆内部（或边界上）
    bool is_inside(P p, const Circle& c) {
        return sign(dist2(p - c.center) - c.r * c.r) <= 0;
    }

    // 辅助函数：判断一个点是否在所有圆的内部
    // 被 O(N^3) 暴力算法使用
    bool is_inside_all(P p, const vector<Circle>& circles) {
        for (const auto& c : circles) {
            if (!is_inside(p, c)) {
                return false;
            }
        }
        return true;
    }

    // 函数1：返回两个圆的交点
    // 返回一个 vector，包含 0, 1, 或 2 个交点
    vector<P> get_circle_intersections(const Circle& c1, const Circle& c2) {
        vector<P> res;
        D d2 = dist2(c1.center - c2.center);
        D d = sqrt(d2);

        // 两圆分离或内含（无切点）
        if (sign(d - (c1.r + c2.r)) > 0 || sign(d - abs(c1.r - c2.r)) < 0) {
            return res;
        }
    
        // 两圆心重合
        if (sign(d) == 0) {
            return res; // 同心圆无交点（除非半径相同，但那有无限交点，不处理）
        }

        // 计算交点
        // 使用余弦定理的推论公式
        D a = (c1.r * c1.r - c2.r * c2.r + d2) / (2 * d);
        D h = sqrt(c1.r * c1.r - a * a);

        P mid_point = c1.center + (a / d) * (c2.center - c1.center);
        P v_perp = {-(c2.center.y - c1.center.y), c2.center.x - c1.center.x};
    
        res.push_back(mid_point + (h / d) * v_perp);
        if (sign(h) != 0) { // 如果不是切点，有第二个交点
            res.push_back(mid_point - (h / d) * v_perp);
        }
        return res;
    }

    // 算法1: 暴力候选点法 O(N^3)
    bool solve_brute_force_O_N3(const vector<pair<P, D>>& circles_in) {
        int n = circles_in.size();
        if (n == 0) return true;
        vector<Circle> C(n);
        for(int i=0; i<n; ++i) C[i] = {circles_in[i].first, circles_in[i].second};

        vector<P> candidates;
        // 候选点1: 所有圆的圆心
        for (int i = 0; i < n; ++i) {
            candidates.push_back(C[i].center);
        }
        // 候选点2: 所有圆两两之间的交点
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                vector<P> intersections = get_circle_intersections(C[i], C[j]);
                for (const auto& p : intersections) {
                    candidates.push_back(p);
                }
            }
        }

        for (const auto& p : candidates) {
            if (is_inside_all(p, C)) {
                return true;
            }
        }
        return false;
    }

    // 算法2: 海利定理法 O(N^3)
    bool solve_helly_theorem_O_N3(const vector<pair<P, D>>& circles_in) {
        int n = circles_in.size();
        if (n <= 3) {
            return solve_brute_force_O_N3(circles_in);
        }
    
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                for (int k = j + 1; k < n; ++k) {
                    vector<pair<P, D>> sub_problem = {circles_in[i], circles_in[j], circles_in[k]};
                    if (!solve_brute_force_O_N3(sub_problem)) {
                        return false; // 找到一个交集为空的三元组
                    }
                }
            }
        }
        return true;
    }

    // 算法3: 随机增量法 O(N) 期望
    // 注意: 实现复杂，代码较长
    namespace RandomizedIncremental {
        vector<Circle> current_circles;
        optional<P> result;

        // 递归函数，constraints_indices是当前点必须在其边界上的圆的索引
        optional<P> find_point(int k, vector<int> constraints_indices) {
            if (k == -1) {
                if (constraints_indices.size() == 0) return P{0, 0};
                if (constraints_indices.size() == 1) {
                    const auto& c = current_circles[constraints_indices[0]];
                    return c.center + P{c.r, 0};
                }
                if (constraints_indices.size() == 2) {
                    auto intersections = get_circle_intersections(
                        current_circles[constraints_indices[0]],
                        current_circles[constraints_indices[1]]
                    );
                    if (intersections.empty()) return nullopt;
                    return intersections[0];
                }
                return nullopt; // 超过2个约束，交点最多2个，3个圆边界交于一点是罕见情况
            }

            optional<P> p = find_point(k - 1, constraints_indices);
        
            if (p.has_value() && is_inside(p.value(), current_circles[k])) {
                return p;
            }

            if (constraints_indices.size() < 2) {
                constraints_indices.push_back(k);
                return find_point(k - 1, constraints_indices);
            } else { // 已经有两个约束了，现在需要找这三个圆边界的交点
                auto intersections = get_circle_intersections(
                    current_circles[constraints_indices[0]],
                    current_circles[constraints_indices[1]]
                );
                for(const auto& pt : intersections) {
                    if(is_inside(pt, current_circles[k])) {
                        return pt;
                    }
                }
            }
            return nullopt;
        }

        bool solve(vector<pair<P, D>> circles_in) {
            int n = circles_in.size();
            if (n == 0) return true;
        
            current_circles.assign(n, Circle());
            for(int i=0; i<n; ++i) current_circles[i] = {circles_in[i].first, circles_in[i].second};

            random_device rd;
            mt19937 g(rd());
            shuffle(current_circles.begin(), current_circles.end(), g);

            optional<P> p = P{current_circles[0].center};
            for (int i = 1; i < n; ++i) {
                if (!p.has_value() || !is_inside(p.value(), current_circles[i])) {
                    vector<int> constraints = {i};
                    p = find_point(i - 1, constraints);
                }
            }
            return p.has_value();
        }
    } // namespace RandomizedIncremental

    bool solve_randomized_incremental_O_N(const vector<pair<P, D>>& circles_in) {
        return RandomizedIncremental::solve(circles_in);
    }


    // 算法4: 三分搜索法 O(N log C) - 竞赛首选
    namespace TernarySearch {
        vector<Circle> C;
    
        // 计算在给定x坐标下，y坐标的有效区间长度
        D check(D x) {
            D y_max_down = -1e18; // DBL_MIN is not negative infinity
            D y_min_up = 1e18;   // DBL_MAX

            for (const auto& c : C) {
                D dx2 = (x - c.center.x) * (x - c.center.x);
                D r2 = c.r * c.r;
                D h2 = r2 - dx2;
                if (sign(h2) < 0) { // x 在此圆的水平投影之外
                    return -1.0; // 表示无解
                }
                D h = sqrt(h2);
                y_max_down = max(y_max_down, c.center.y - h);
                y_min_up = min(y_min_up, c.center.y + h);
            }
            return y_min_up - y_max_down;
        }

        bool solve(const vector<pair<P, D>>& circles_in) {
            int n = circles_in.size();
            if (n == 0) return true;
        
            C.assign(n, Circle());
            for(int i=0; i<n; ++i) C[i] = {circles_in[i].first, circles_in[i].second};

            D lx = -2e5, rx = 2e5; // 一个足够大的范围，也可以根据输入动态计算
            // 更精确的范围计算
            // lx = C[0].center.x - C[0].r;
            // rx = C[0].center.x + C[0].r;
            // for(int i=1; i<n; ++i) {
            //     lx = min(lx, C[i].center.x - C[i].r);
            //     rx = max(rx, C[i].center.x + C[i].r);
            // }

            for (int i = 0; i < 100; ++i) { // 100次迭代对于long double精度足够
                D m1 = lx + (rx - lx) / 3.0;
                D m2 = rx - (rx - lx) / 3.0;
                if (check(m1) < check(m2)) {
                    lx = m1;
                } else {
                    rx = m2;
                }
            }

            return check((lx + rx) / 2.0) >= -eps;
        }
    } // namespace TernarySearch

    bool solve_ternary_search_O_N_log_C(const vector<pair<P, D>>& circles_in) {
        return TernarySearch::solve(circles_in);
    }

} // namespace CircleIntersection

// =================== 使用示例 ===================
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    cin >> n;

    vector<pair<P, D>> circles(n);
    for (int i = 0; i < n; ++i) {
        cin >> circles[i].first.x >> circles[i].first.y >> circles[i].second;
    }

    // 在比赛中，通常只会使用一种解法，这里为了演示全部调用
    cout << "Method 1 (Brute Force O(N^3)): " 
         << (CircleIntersection::solve_brute_force_O_N3(circles) ? "Yes" : "No") << endl;

    cout << "Method 2 (Helly's Theorem O(N^3)): "
         << (CircleIntersection::solve_helly_theorem_O_N3(circles) ? "Yes" : "No") << endl;
  
    cout << "Method 3 (Randomized Incremental O(N)): "
         << (CircleIntersection::solve_randomized_incremental_O_N(circles) ? "Yes" : "No") << endl;

    cout << "Method 4 (Ternary Search O(N log C)): "
         << (CircleIntersection::solve_ternary_search_O_N_log_C(circles) ? "Yes" : "No") << endl;

    return 0;
}
```
当然，这是一个非常好的问题，它完美地展示了如何将一个成功的降维思想从二维扩展到三维。

将“N个球体求交集”问题降维成“N个圆求交集”问题的核心思想是：**通过固定一个坐标轴，将三维问题转化为在特定平面上的二维问题。**

下面是详细的实现步骤和逻辑：

### 1. 从二维到三维的类比

我们先回顾一下二维圆问题的解法：
*   **二维问题**：寻找一个点 `(x, y)`。
*   **降维方法**：我们**固定 `x`**，然后寻找是否存在一个合法的 `y`。这相当于用一条垂直于x轴的直线 `X = x_const` 去切割所有的圆，得到了一堆在y轴上的一维区间。我们求这些区间的交集。
*   **目标函数**：`F(x) = y_range_length(x)`，我们用三分法找到 `max(F(x))`。

现在，我们将这个思想提升一个维度：
*   **三维问题**：寻找一个点 `(x, y, z)`。
*   **降维方法**：我们同样先**固定 `x`**，然后问题就变成了“在 `X = x_const` 这个固定的平面上，是否存在一个点 `(y, z)` 被所有球体包含？”
*   **目标函数**：`H(x) = some_value`，我们希望用三分法找到 `max(H(x))`。

### 2. “降维”的具体实现：用平面切割球体

当你用一个平面（比如 `x = x_0`）去切割一个三维球体时，会发生什么？

*   **几何上**：切割的结果是一个**二维的圆**（或者一个点，或者没有交集）。
*   **代数上**：
    *   原始球体 `i` 的方程为: `(x - xᵢ)² + (y - yᵢ)² + (z - zᵢ)² ≤ rᵢ²`
    *   我们将 `x = x_0` 代入方程中: `(x_0 - xᵢ)² + (y - yᵢ)² + (z - zᵢ)² ≤ rᵢ²`
    *   整理一下，把常数项移到右边: `(y - yᵢ)² + (z - zᵢ)² ≤ rᵢ² - (x_0 - xᵢ)²`

现在我们来分析这个新方程：

1.  这是一个在 `yz` 平面上的二维问题。
2.  它的形式 `(y - y_center)² + (z - z_center)² ≤ R_new²` 正是一个**圆的方程**。
3.  这个新圆的参数是：
    *   **圆心**: `(yᵢ, zᵢ)` （注意，圆心在yz平面上的坐标就是原始球心的y和z坐标）
    *   **新半径的平方**: `R'_i² = rᵢ² - (x_0 - xᵢ)²`

    **重要**：如果 `rᵢ² - (x_0 - xᵢ)² < 0`，说明 `x_0` 这个平面根本没有碰到球体 `i`。在这种情况下，不可能存在公共交点，对于这个 `x_0`，我们可以直接返回一个负无穷大的结果。否则，新半径就是 `R'_i = sqrt(rᵢ² - (x_0 - xᵢ)²)`.

### 3. 算法的嵌套结构：三分套三分

现在，我们的逻辑链条是这样的：

1.  为了解决三维球体交集问题，我们想找到一个最优的 `x`。
2.  我们发现，对于任何一个固定的 `x`，这个问题都会**转化**为：**“给定N个新的二维圆（在yz平面上），判断它们是否有公共交集。”**
3.  而后面这个问题（N个圆求交集），我们已经知道如何用**三分法**解决了！我们可以在 `y` 轴上进行三分搜索，找到一个使得 `z` 轴上可行区间长度最大的 `y`，并判断该长度是否大于等于0。
4.  这个“N个圆求交集”的解（`z` 轴最大区间长度），其结果依赖于我们一开始固定的 `x`。我们可以把这个结果定义为一个函数 `H(x)`。可以证明，`H(x)` 同样是一个**单峰函数**。
5.  因此，我们可以用一个**外层的三分法**来寻找 `x`，以求得 `H(x)` 的最大值。

**算法的最终实现被称为“三分套三分”（Ternary Search in Ternary Search）：**

*   **外层三分**: 在 `x` 轴上进行三分搜索。
    *   取 `m1 = L + (R-L)/3`, `m2 = R - (R-L)/3`。
    *   我们需要比较 `H(m1)` 和 `H(m2)` 的大小。

*   **内层三分**: 为了计算 `H(x)` 的值（例如 `H(m1)`）：
    *   我们已经固定了 `x = m1`。
    *   根据这个 `x` 值，计算出所有 N 个新的二维圆的参数（圆心 `(yᵢ, zᵢ)` 和新半径 `R'_i`）。
    *   现在，我们对这个二维圆问题，在 `y` 轴上进行三分搜索来找到其解。
        *   在内层三分的每一步，对于一个给定的 `y`，我们计算 `min(z_upᵢ) - max(z_downᵢ)` 的值。
    *   内层三分结束后，返回的结果就是 `H(x)` 的值。

*   **判断**: 外层三分根据 `H(m1)` 和 `H(m2)` 的结果，缩小 `x` 的搜索范围。
*   **最终结果**: 整个算法结束后，我们得到了 `max(H(x))` 的近似值。如果这个值大于等于0（考虑浮点误差），则存在公共交点；否则不存在。

简而言之，**降维成圆是通过固定一个坐标轴来实现的**。因为这个被固定的坐标可以是任意值，所以我们需要再用一层三分法来找到最优的那个固定值。