# 哑铃策略滚动截面回测 — 架构设计

> 版本: V1.0
> 日期: 2026-06-15
> 状态: 设计阶段
> 负责: AgentQuant 回测实验室

---

## 一、问题定义

### 1.1 当前回测的前视偏差

```
错误做法:
  用2026-06-12的PE/ROE截面 → 选出"低PE+高ROE"30只 
  → 回溯到2021年算收益 → +99.5%
  
  问题: 2021年这些股票可能PE并不低, 甚至还没上市

正确做法:
  每月末(如2021-01-29) → 取当时的PE/ROE截面 
  → 选出当时的"低PE+高ROE"30只 
  → 持有到下个月调仓日 → 计算实际收益
```

### 1.2 三个深水区

| 问题 | 描述 | 严重性 |
|------|------|--------|
| **PIT数据对齐** | Q4年报4月才发布, 2月用年报ROE=未来函数 | 🔴 致命 |
| **幸存者偏差** | 退市股/未上市股导致虚假收益 | 🔴 致命 |
| **异步调仓** | 防守季度/进攻周度, 双轨步长不同 | 🟡 工程 |

---

## 二、因子流水线

### 2.1 防守因子: 低PE + 高ROE

```
数据源: financial_statements (report_type='annual'/'Q1'/'Q2'/'Q3')

PIT规则:
  每年1-3月: 使用最近可用的年报 (上一年Q3)
  每年4-8月: 使用上一年年报 (4月底前发布)
  每年9-10月: 使用当年半年报
  每年11-12月: 使用当年Q3报

ROE_TTM = 最近四个季度净利润之和 / 最新净资产

PE = 当前总市值 / 净利润_TTM
  
  总市值: 当日收盘价 × 总股本
  总股本: 从stock_basic获取 (若NULL则用kline_daily.total_share反推)

因子值:
  score_defense = rank(-PE) × 0.6 + rank(ROE_TTM) × 0.4
  rank: 截面百分位排名 (0-1)
```

### 2.2 进攻因子: 行业动量 + 高波/高换手正交化

```
行业动量:
  窗口: 过去20个交易日
  计算: 行业指数涨幅 rank
  选top3行业

正交化:
  截面回归: volatility_i = α + β × turnover_i + ε_i
  净波动率 = ε_i (残差)
  物理意义: 剔除换手驱动的自然波动, 捕捉异质波动

因子值:
  score_offense = rank(turnover) × 0.4 + rank(net_vol) × 0.6
  仅在前3强势行业股票池内计算
```

### 2.3 因子计算时间线

```
调仓日 T (每月最后一个交易日):
  1. 获取T日所有股票截面数据 (价格/PE/ROE/波动率/换手率)
  2. PIT修正: 财务数据取最近合法报告期
  3. 行业动量: 过去20日行业指数涨幅
  4. 正交化: T日截面回归
  5. 选股: 防守top30 + 进攻top30
  6. 持仓: 到下个调仓日
```

---

## 三、Universe 与数据对齐

### 3.1 动态股票池

```sql
每月末生成候选池:
  SELECT ts_code FROM kline_daily
  WHERE trade_date = '{调仓日}'
    AND close > 0 AND vol > 0           -- 正常交易
    AND ts_code NOT LIKE '%ST%'          -- 非ST (需从name字段判断)
    AND ts_code IN (
      SELECT ts_code FROM kline_daily 
      WHERE trade_date >= '{调仓日 - 1年}'  -- 上市满1年
      GROUP BY ts_code HAVING COUNT(*) >= 200
    )
    AND ts_code NOT IN (
      SELECT ts_code FROM stock_basic WHERE delist_date IS NOT NULL  -- 未退市
    )
```

### 3.2 财报PIT映射表

```
调仓月份 | 可用最新财报 | report_type | 滞后天数
  1月    | 上一年Q3     | Q3          | ~90天
  2月    | 上一年Q3     | Q3          | ~120天
  3月    | 上一年Q3     | Q3          | ~150天
  4月    | 上一年Q4(年报)| annual      | ~0-30天
  5月    | 上一年Q4     | annual      | ~30天
  ...
  8月    | 上一年Q4     | annual      | ~120天
  9月    | 当年Q2(半年报)| Q2          | ~0-30天
  ...
  11月   | 当年Q3       | Q3          | ~0-30天
```

### 3.3 缺失数据处理

```
若某股票指定report_type无数据:
  1. 回退到上一个report_type (如无Q3用Q2)
  2. 仍无数据 → 从该月候选池剔除
  3. 若候选池<20只 → 放宽PE上限到500
```

---

## 四、交易执行与风控

### 4.1 双轨调仓

```
防守端:
  调仓频率: 每季度 (3/6/9/12月最后一个交易日)
  持仓数: 30只
  加权: 等权
  调仓成本: 单边0.15% (低换手)

进攻端:
  调仓频率: 每月 (每月最后一个交易日)
  持仓数: 30只
  加权: 等权
  调仓成本: 单边0.30% (高换手)
```

### 4.2 VIX动态权重

```
每月调仓时评估:

  VIX < 18 (低恐慌):
    防守权重 = base_weight - 0.15
    进攻权重 = base_weight + 0.15
  
  18 ≤ VIX ≤ 25 (正常):
    防守权重 = base_weight
    进攻权重 = base_weight
  
  VIX > 25 (高恐慌):
    防守权重 = base_weight + 0.25
    进攻权重 = base_weight - 0.25
  
  VIX > 35 (危机):
    进攻权重 = 0.20
    全仓退守防守端

  base_weight 默认: 防守0.50 / 进攻0.50
  权重裁剪: [0.20, 0.80]
```

### 4.3 再平衡触发

```
触发条件: |实际权重 - 目标权重| > 0.15
再平衡操作: 卖出超配端, 买入低配端
再平衡成本: 0.15% (只在触发时发生)
```

### 4.4 交易摩擦

```
固定参数:
  佣金: 0.025% (双边)
  印花税: 0.05% (卖出单边, A股)
  滑点: 0.10% (进攻端) / 0.05% (防守端)
  
  防守端总摩擦: 0.30% (买入) + 0.45% (卖出含印花税)
  进攻端总摩擦: 0.40% (买入) + 0.55% (卖出含印花税)
```

---

## 五、绩效评价

### 5.1 核心指标

| 指标 | 公式 | 阈值 |
|------|------|------|
| 年化收益 | `(期末净值/期初)^(1/年)-1` | > 基准+5% |
| 年化波动率 | `日收益std × sqrt(252)` | < 25% |
| 夏普比率 | `(年化收益-无风险)/年化波动` | > 1.0 |
| 最大回撤 | `max(1-净值/前高)` | < 20% |
| 卡玛比率 | `年化收益/最大回撤` | > 0.5 |
| 胜率 | `正收益月数/总月数` | > 55% |

### 5.2 分解归因

```
防守端 vs 进攻端:
  相关性: 两端月度收益的相关系数 (期望 < 0.5, 低相关=有效对冲)
  贡献度: 每端占组合收益的百分比
  VIX效应: VIX<18时 vs VIX>25时 组合表现差异

基准比较:
  vs 沪深300 (sh000300)
  vs 中证500 (sh000905)
  vs 等权全A
```

### 5.3 输出图表

1. 组合净值曲线 (防守/进攻/哑铃 三线)
2. 回撤曲线
3. 12月滚动夏普比率
4. VIX vs 组合权重 时间序列
5. 月度收益热力图 (按年排列)

---

## 六、工程架构

### 6.1 模块结构

```
D:\AgentQuant\our\
├── barbell_rolling_backtest_design.md  ← 本文件
├── factor_pipeline.py                  # 因子计算引擎
│   ├── get_pit_financials(date)        # PIT财务数据
│   ├── calc_defense_score(universe)    # 防守因子
│   └── calc_offense_score(universe)    # 进攻因子(含正交化)
├── universe.py                         # 动态股票池
│   ├── get_universe(date)              # 每月候选池
│   └── filter_delist_st(universe)      # 剔除退市/ST
├── execution.py                        # 交易执行
│   ├── rebalance(date, pool, weight)   # 调仓
│   ├── vix_weight_adjust(vix)          # VIX权重
│   └── apply_friction(turnover)        # 摩擦成本
├── performance.py                      # 绩效评价
│   ├── calc_metrics(nav_curve)         # 核心指标
│   └── plot_report(metrics)            # 图表输出
└── run_rolling.py                      # 主回测循环
```

### 6.2 主循环伪代码

```python
def rolling_backtest(start='2021-01-01', end='2026-06-15'):
    dates = get_monthly_rebalance_dates(start, end)
    
    nav_def = 1.0; nav_off = 1.0
    def_holdings = []; off_holdings = []
    
    for i, date in enumerate(dates):
        # 1. 获取当日截面
        universe = get_universe(date)
        
        # 2. PIT财务数据
        fin_data = get_pit_financials(date, universe)
        
        # 3. 因子计算
        if is_quarter_end(date):  # 防守端季度调仓
            def_scores = calc_defense_score(universe, fin_data)
            def_holdings = select_top(def_scores, 30)
            def_turnover = compute_turnover(old_def, def_holdings)
        
        if True:  # 进攻端每月调仓
            off_scores = calc_offense_score(universe, fin_data, date)
            off_holdings = select_top(off_scores, 30)
            off_turnover = compute_turnover(old_off, off_holdings)
        
        # 4. VIX权重
        vix = get_vix(date)
        w_def, w_off = vix_weight(vix, base=0.5)
        
        # 5. 持仓到下个调仓日
        next_date = dates[i+1] if i+1 < len(dates) else end
        ret_def = compute_holdings_return(def_holdings, date, next_date)
        ret_off = compute_holdings_return(off_holdings, date, next_date)
        
        # 6. 扣费
        ret_def -= def_turnover * DEF_FRICTION
        ret_off -= off_turnover * OFF_FRICTION
        
        # 7. 更新净值
        nav_def *= (1 + ret_def)
        nav_off *= (1 + ret_off)
        
        # 8. 再平衡
        actual_w = nav_off / (nav_def + nav_off)
        if abs(actual_w - w_off) > 0.15:
            nav_def, nav_off = rebalance(nav_def, nav_off, w_def, w_off)
    
    return nav_def, nav_off, nav_def + nav_off
```

---

## 七、已知限制

1. **财报PIT精度**: 当前financial_statements表的report_date是报告期截止日, 非实际发布日期。真实PIT需要爬取公告发布日期。
2. **总股本数据**: stock_basic.total_share全NULL, 市值计算暂时只能用valuation_daily的PE反推。
3. **退市股**: kline_daily中退市股数据完整, 但需确认最后交易日。
4. **行业映射**: 个股→申万行业映射未建立, 行业动量选股暂时无法按行业过滤个股。

---

## 八、backtest_engine 三个关键补丁

### 8.1 交易日历自动对齐

```python
def get_month_end_trade_date(year, month):
    """取该月最后一个真实交易日"""
    target = date(year, month, 28) + timedelta(days=4)  # 确保跨到次月
    target = target.replace(day=1) - timedelta(days=1)   # 月末最后一天
    result = db.execute("""
        SELECT MAX(trade_date) FROM kline_daily
        WHERE trade_date <= ?
    """, [target.isoformat()]).fetchone()
    return result[0]  # 真实交易日
```

**绝不使用 `date(year, month, 30)`。** 不同月份末日期不同(28/29/30/31), 且可能是周末/假期。

### 8.2 摩擦成本按换手率计算

```python
def calc_turnover_friction(old_holdings, new_holdings, nav, friction_rate):
    """只对需要换仓的部分扣费"""
    old_set = set(old_holdings)
    new_set = set(new_holdings)
    
    # 卖出: 旧仓有但新仓没有
    sell_count = len(old_set - new_set)
    # 买入: 新仓有但旧仓没有  
    buy_count = len(new_set - old_set)
    
    turnover_ratio = (sell_count + buy_count) / len(new_holdings)
    turnover_value = nav * turnover_ratio
    
    # A股: 买入佣金+卖出佣金+印花税(卖)
    cost = turnover_value * friction_rate
    if sell_count > 0:
        cost += (nav * sell_count / len(new_holdings)) * 0.0005  # 印花税
    
    return cost
```

**不是每次扣总资产的0.30%。** 如果80%持仓不变, 只扣20%换仓部分的费用。

### 8.3 全退守下的现金管理

```
VIX > 35 → 进攻端全清, 防守端压至10%最低仓
  
  现金部分(90%): 年化2%收益 (模拟GC001/货币基金)
  死仓部分(10%): 持有防守池top10, 等权
  
  恢复条件: VIX < 30 且连续3日不再创新高 → 解除全退守
```

**不是净值横线。** 一成死仓给策略留"气口", 反弹时不会被甩下。

---

## 九、全退守决策矩阵

| VIX区间 | 防守权重 | 进攻权重 | 状态 |
|---------|---------|---------|------|
| < 18 | 35% | 65% | 进攻 |
| 18-25 | 50% | 50% | 均衡 |
| 25-30 | 65% | 35% | 防御 |
| 30-35 | 80% | 20% | 收缩 |
| > 35 | 10%死仓 | 0% | 全退守 |
| > 35超过21天 | 清仓 | 0% | 空仓 |

**VIX > 35超过21天 → 连死仓也清。** 说明已进入系统性危机(如2020年3月), 现金为王。

---

> 📅 下一步: 周末实现 backtest_engine.py (66个月循环 + 三补丁)
> 🔧 factor_pipeline.py 已就位, 冒烟测试通过
> ⚠️ 实际回测需5-8小时 (每月截面5000只计算), 建议周末跑全量
