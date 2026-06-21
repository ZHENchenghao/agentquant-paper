# -*- coding: utf-8 -*-
"""
Build conduction table: every macro -> industry link
with full economic mechanism, saved as parquet
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
from datetime import datetime

# ============================================================
# Each row = one conduction link
# Columns:
#   macro_var: which macro indicator
#   macro_source: DuckDB column name
#   industry: Shenwan industry name (matching stock_industry_map)
#   channel: cost / revenue / substitution / wealth / sentiment / policy
#   direction: +1 (macro up -> sector up), -1 (macro up -> sector down)
#   lag_min, lag_max: trading days
#   threshold_pct: macro change threshold to trigger signal
#   elasticity: how much sector moves per 1% macro move (estimated)
#   confidence: HIGH/MEDIUM/LOW (based on economic theory)
#   mechanism: full economic logic description
#   testable_hypothesis: falsifiable prediction
# ============================================================

CONDUCTION = []

def add(macro_var, macro_source, industry, channel, direction,
        lag_min, lag_max, threshold_pct, elasticity, confidence,
        mechanism, hypothesis):
    CONDUCTION.append({
        'macro_var': macro_var,
        'macro_source': macro_source,
        'industry': industry,
        'channel': channel,
        'direction': direction,
        'lag_min': lag_min,
        'lag_max': lag_max,
        'threshold_pct': threshold_pct,
        'elasticity': elasticity,
        'confidence': confidence,
        'mechanism': mechanism,
        'hypothesis': hypothesis,
        'created': datetime.now().strftime('%Y-%m-%d'),
    })

# ============================================================
# 1. WTI原油 -> 各行业 (5大通道)
# ============================================================
WTI_MECH = {
    '石油石化': {
        'channel': 'revenue',
        'dir': +1, 'lag': (0, 5), 'thresh': 3.0, 'elastic': 2.5, 'conf': 'HIGH',
        'mech': 'WTI涨 -> 三桶油上游开采利润直接增厚(弹性2.5x) -> 滞后0-3天同步。'
               '中游炼化成本升但成品油定价跟涨 -> 净效应正。'
               '注意: 中石化炼化占比高(60%), 弹性弱于中石油(80%上游)。',
        'hyp': 'WTI单日涨>3% -> 石油石化行业未来0-5天正收益, 准确率>60%',
    },
    '煤炭': {
        'channel': 'substitution',
        'dir': +1, 'lag': (2, 8), 'thresh': 3.0, 'elastic': 0.8, 'conf': 'MEDIUM',
        'mech': 'WTI涨 -> 能源替代需求 -> 煤价跟涨 -> 煤企利润增。'
               '传导链: WTI -> 国际煤价(纽卡斯尔) -> 国内动力煤 -> 煤企股价。'
               '注意: 国内煤价受长协价约束, 弹性打折扣。',
        'hyp': 'WTI涨>3% -> 煤炭行业未来2-8天正收益',
    },
    '基础化工': {
        'channel': 'cost',
        'dir': -1, 'lag': (3, 15), 'thresh': 5.0, 'elastic': -0.5, 'conf': 'MEDIUM',
        'mech': 'WTI涨 -> 石脑油涨 -> PX/PTA/EG涨(滞后3-7天) -> 化纤/塑料原料成本升。'
               '传导: 原油 -> 石脑油 -> PX -> PTA -> 涤纶长丝 -> 纺织。'
               '每层加工费+库存 = 累计滞后10-15天。'
               '注意: 油价涨<5%被库存吸收, 需大波动才传导。',
        'hyp': 'WTI涨>5% -> 基础化工未来10-15天负收益',
    },
    '交通运输': {
        'channel': 'cost',
        'dir': -1, 'lag': (1, 8), 'thresh': 3.0, 'elastic': -0.6, 'conf': 'HIGH',
        'mech': 'WTI涨 -> 航油/柴油成本升 -> 航空/物流毛利率压缩。'
               '航油占航空成本30-35%, 柴油占物流成本20-25%。'
               '注意: 航运公司有燃油附加费可转嫁, 航空套保比例(国航40%/南航35%)。',
        'hyp': 'WTI涨>3% -> 交通运输未来1-8天负收益, 滞后1-3天最显著',
    },
    '汽车': {
        'channel': 'substitution',
        'dir': +1, 'lag': (3, 15), 'thresh': 5.0, 'elastic': 0.3, 'conf': 'LOW',
        'mech': 'WTI涨 -> 燃油车使用成本升 -> 新能源车替代需求增 -> 利好新能源车企。'
               '但传导慢(消费者换车决策以月计), 短期效果弱。',
        'hyp': 'WTI涨>5% -> 汽车(新能源)行业未来1-3周正收益, 效果弱',
    },
    '公用事业': {
        'channel': 'cost',
        'dir': -1, 'lag': (3, 10), 'thresh': 5.0, 'elastic': -0.2, 'conf': 'LOW',
        'mech': 'WTI涨 -> 天然气跟涨 -> 燃气发电成本升 -> 电力公司承压。'
               '但国内电价受管制, 成本无法完全传导, 影响有限。',
        'hyp': 'WTI涨>5% -> 公用事业轻微负收益',
    },
}

for ind, info in WTI_MECH.items():
    add('WTI原油', 'wti', ind, info['channel'], info['dir'],
        info['lag'][0], info['lag'][1], info['thresh'], info['elastic'],
        info['conf'], info['mech'], info['hyp'])

# ============================================================
# 2. 伦敦金 -> 各行业
# ============================================================
GOLD_MECH = {
    '有色金属': {
        'channel': 'revenue',
        'dir': +1, 'lag': (0, 3), 'thresh': 1.5, 'elastic': 2.5, 'conf': 'HIGH',
        'mech': '金价涨 -> 黄金矿业利润弹性 = 金价涨幅 x 产量 x 杠杆倍数。'
               '紫金矿业2.5x/山东黄金3x/中金黄金2x。'
               '金价+10% -> 黄金股利润+25-30%, 几乎同步(0-2天)。',
        'hyp': '金价涨>1.5% -> 有色金属(黄金股)当日正收益, 准确率>60%',
    },
    '商贸零售': {
        'channel': 'demand',
        'dir': -1, 'lag': (5, 20), 'thresh': 3.0, 'elastic': -0.3, 'conf': 'LOW',
        'mech': '金价涨 -> 黄金饰品需求降(消费者买涨不买跌的反向逻辑: 金价涨太快观望)。'
               '但高金价也增加珠宝商库存价值, 两力对冲。净效应弱。',
        'hyp': '金价急涨>3% -> 珠宝零售未来1-3周轻微负收益',
    },
    '银行': {
        'channel': 'sentiment',
        'dir': -1, 'lag': (0, 3), 'thresh': 2.0, 'elastic': -0.4, 'conf': 'MEDIUM',
        'mech': '金价急涨 -> 避险情绪 -> 资金从银行(周期)撤出 -> 银行股承压。'
               '注意: 金价和银行股负相关也被视为经济衰退预期指标。',
        'hyp': '金价涨>2% -> 银行股当日或次日负收益',
    },
}

for ind, info in GOLD_MECH.items():
    add('伦敦金', 'gold', ind, info['channel'], info['dir'],
        info['lag'][0], info['lag'][1], info['thresh'], info['elastic'],
        info['conf'], info['mech'], info['hyp'])

# ============================================================
# 3. 伦铜 -> 各行业
# ============================================================
COPPER_MECH = {
    '有色金属': {
        'channel': 'revenue',
        'dir': +1, 'lag': (1, 7), 'thresh': 2.0, 'elastic': 1.8, 'conf': 'HIGH',
        'mech': '铜价涨 -> 铜矿企业(紫金/江铜/西部矿业)库存增值 + 售价上升。'
               '铜是经济晴雨表(Dr.Copper): 铜价涨 -> 全球工业需求复苏预期 -> 有色板块跟涨。'
               '金融通道(0-1天) + 实体通道(1-2周)。',
        'hyp': '铜价涨>2% -> 有色金属未来1-7天正收益, 准确率>65%',
    },
    '电力设备': {
        'channel': 'cost',
        'dir': -1, 'lag': (5, 20), 'thresh': 5.0, 'elastic': -0.3, 'conf': 'MEDIUM',
        'mech': '铜价涨 -> 电线电缆/变压器铜成本占15-20% -> 毛利率压缩。'
               '但大型电力设备企业(特变电工/中国西电)有铜期货套保, 实际影响滞后1-2月。',
        'hyp': '铜价涨>5% -> 电力设备未来1-3周轻微负收益',
    },
    '房地产': {
        'channel': 'demand',
        'dir': +1, 'lag': (3, 15), 'thresh': 3.0, 'elastic': 0.4, 'conf': 'MEDIUM',
        'mech': '铜价涨 -> 反映建筑/基建需求旺盛 -> 房地产开工预期改善 -> 地产股跟涨。'
               '铜价是先行指标: 铜价涨通常领先地产开工2-4周。',
        'hyp': '铜价涨>3% -> 房地产未来3-15天正收益',
    },
}

for ind, info in COPPER_MECH.items():
    add('伦铜', 'copper', ind, info['channel'], info['dir'],
        info['lag'][0], info['lag'][1], info['thresh'], info['elastic'],
        info['conf'], info['mech'], info['hyp'])

# ============================================================
# 4. 美10年期国债利率 -> 各行业 (折现率+资金流双通道)
# ============================================================
US10Y_MECH = {
    '电子': {
        'channel': 'valuation',
        'dir': -1, 'lag': (1, 8), 'thresh': 0.15, 'elastic': -1.5, 'conf': 'HIGH',
        'mech': '美10Y升 -> DCF分母折现率升 -> 高久期/高PE板块估值压缩最敏感。'
               '电子(半导体)远期现金流占比高 -> 估值弹性-1.5x。'
               '叠加: 美10Y升 -> 北向减持成长股(次日) -> 加速下跌。'
               '阈值: 15bp变化才触发, 日常<5bp波动忽略。',
        'hyp': '美10Y升>15bp -> 电子行业未来1-8天负收益, 准确率>55%',
    },
    '计算机': {
        'channel': 'valuation',
        'dir': -1, 'lag': (1, 8), 'thresh': 0.15, 'elastic': -1.3, 'conf': 'HIGH',
        'mech': '同电子: 高PE/高久期 -> 利率敏感。软件/SaaS远期现金流 -> DCF估值压缩。',
        'hyp': '美10Y升>15bp -> 计算机行业未来1-8天负收益',
    },
    '医药生物': {
        'channel': 'valuation',
        'dir': -1, 'lag': (1, 10), 'thresh': 0.15, 'elastic': -1.0, 'conf': 'MEDIUM',
        'mech': '创新药/生物科技远期现金流占比高 -> 利率敏感。'
               '但仿制药/中药现金流近期 -> 利率不敏感 -> 行业内分化大。',
        'hyp': '美10Y升>15bp -> 医药生物未来1-10天轻微负收益',
    },
    '银行': {
        'channel': 'revenue',
        'dir': +1, 'lag': (1, 8), 'thresh': 0.10, 'elastic': 0.5, 'conf': 'MEDIUM',
        'mech': '利率升 -> 息差预期扩张 -> 银行利润增。'
               '但有两力对冲: 利率急升 -> 银行持债浮亏(bond portfolio loss)。'
               '净效应: 缓升(20bp/月)利多; 急升(>50bp/周)利空(资产减值恐慌)。'
               '中国银行: 受美10Y影响间接(中美利差 -> 人民币 -> 北向), '
               '国内利率(MLF/LPR)才是主导。',
        'hyp': '美10Y升>10bp且<30bp -> 银行未来1-8天正收益; >30bp -> 反转为负',
    },
    '通信': {
        'channel': 'valuation',
        'dir': -1, 'lag': (1, 8), 'thresh': 0.15, 'elastic': -1.0, 'conf': 'MEDIUM',
        'mech': '通信(5G/光通信)高资本开支 -> 远期现金流 -> 利率敏感。',
        'hyp': '美10Y升>15bp -> 通信行业未来1-8天负收益',
    },
    '国防军工': {
        'channel': 'valuation',
        'dir': -1, 'lag': (1, 8), 'thresh': 0.15, 'elastic': -0.6, 'conf': 'LOW',
        'mech': '军工高PE(50-80x) -> 理论上利率敏感。但军工受政策/地缘主导, 利率影响弱。',
        'hyp': '美10Y升>15bp -> 国防军工轻微负收益(弱)',
    },
}

for ind, info in US10Y_MECH.items():
    add('美10Y', 'us10y', ind, info['channel'], info['dir'],
        info['lag'][0], info['lag'][1], info['thresh'], info['elastic'],
        info['conf'], info['mech'], info['hyp'])

# ============================================================
# 5. VIX -> 各行业 (恐慌传导)
# ============================================================
VIX_MECH = {
    '银行': {
        'channel': 'sentiment',
        'dir': -1, 'lag': (0, 5), 'thresh': 3.0, 'elastic': -1.2, 'conf': 'HIGH',
        'mech': 'VIX跳升 -> 全球risk-off -> 北向撤出 -> 银行(外资重仓)首当其冲。'
               'VIX>25且单日涨>3点 -> 银行次日下跌概率>70%。',
        'hyp': 'VIX单日升>3点 -> 银行未来0-5天负收益, 准确率>65%',
    },
    '食品饮料': {
        'channel': 'sentiment',
        'dir': -1, 'lag': (0, 5), 'thresh': 3.0, 'elastic': -1.0, 'conf': 'HIGH',
        'mech': '白酒/食品是北向重仓(茅台/五粮液) -> VIX升 -> 外资减仓 -> 白酒跌。'
               '但食品饮料防御属性强, 恐慌后期可能成避风港。',
        'hyp': 'VIX升>3点 -> 食品饮料未来0-5天负收益',
    },
    '电子': {
        'channel': 'sentiment',
        'dir': -1, 'lag': (0, 5), 'thresh': 3.0, 'elastic': -1.5, 'conf': 'HIGH',
        'mech': 'VIX升 -> 风险偏好降 -> 高beta板块(半导体)杀跌最猛。'
               '电子beta~1.3 -> VIX升1% -> 电子跌1.3%。',
        'hyp': 'VIX升>3点 -> 电子未来0-5天负收益, 幅度最大',
    },
    '公用事业': {
        'channel': 'sentiment',
        'dir': +1, 'lag': (0, 5), 'thresh': 3.0, 'elastic': 0.8, 'conf': 'MEDIUM',
        'mech': 'VIX升 -> 避险 -> 资金涌入低beta/高股息(电力/水务) -> 公用事业受益。'
               '公用事业是VIX的天然对冲。',
        'hyp': 'VIX升>3点 -> 公用事业未来0-5天正收益(或跌幅小于大盘)',
    },
}

for ind, info in VIX_MECH.items():
    add('VIX', 'vix', ind, info['channel'], info['dir'],
        info['lag'][0], info['lag'][1], info['thresh'], info['elastic'],
        info['conf'], info['mech'], info['hyp'])

# ============================================================
# 6. 标普500 -> A股行业 (隔夜情绪传导)
# ============================================================
SPX_MECH = {
    '食品饮料': {
        'channel': 'sentiment',
        'dir': +1, 'lag': (0, 2), 'thresh': 1.0, 'elastic': 0.6, 'conf': 'MEDIUM',
        'mech': 'SPX涨 -> 全球风险偏好改善 -> 次日北向流入 -> 白酒等外资重仓受益。'
               '与沪深300高度重叠(外资偏好)。',
        'hyp': 'SPX涨>1% -> 食品饮料次日正收益',
    },
    '银行': {
        'channel': 'sentiment',
        'dir': +1, 'lag': (0, 2), 'thresh': 1.0, 'elastic': 0.5, 'conf': 'MEDIUM',
        'mech': 'SPX涨 -> 金融股情绪传导 -> 银行次日跟涨。',
        'hyp': 'SPX涨>1% -> 银行次日正收益',
    },
    '电子': {
        'channel': 'sentiment',
        'dir': +1, 'lag': (0, 2), 'thresh': 1.0, 'elastic': 0.8, 'conf': 'MEDIUM',
        'mech': 'SPX涨(尤其费城半导体) -> A股半导体次日跟涨。纳指对科创50传导更强。',
        'hyp': 'SPX涨>1% -> 电子次日正收益',
    },
}

for ind, info in SPX_MECH.items():
    add('标普500', 'spx', ind, info['channel'], info['dir'],
        info['lag'][0], info['lag'][1], info['thresh'], info['elastic'],
        info['conf'], info['mech'], info['hyp'])

# ============================================================
# Build DataFrame and save
# ============================================================
df = pd.DataFrame(CONDUCTION)
df.index.name = 'link_id'

print('=' * 70)
print('Conduction Table Built')
print('=' * 70)
print('Total links: %d' % len(df))
print()
print('By macro variable:')
for mv in df['macro_var'].unique():
    n = (df['macro_var'] == mv).sum()
    high = ((df['macro_var'] == mv) & (df['confidence'] == 'HIGH')).sum()
    print('  %s: %d links (%d HIGH confidence)' % (mv, n, high))
print()
print('By channel:')
for ch in df['channel'].unique():
    n = (df['channel'] == ch).sum()
    print('  %s: %d links' % (ch, n))
print()
print('By direction:')
print('  Positive (+1): %d' % (df['direction'] == 1).sum())
print('  Negative (-1): %d' % (df['direction'] == -1).sum())

# Save
df.to_parquet('cache/conduction_table.parquet')
print('\nSaved to cache/conduction_table.parquet')

# Also save as readable CSV
df.to_csv('cache/conduction_table.csv', index=True, encoding='utf-8-sig')
print('Also saved as cache/conduction_table.csv')

print('\nDone. Ready for backtest.')
