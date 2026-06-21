# -*- coding: utf-8 -*-
"""
大盘择时开关: MA200 + 斜率双条件 + 震荡缓冲
牛市→进攻轮动 | 熊市→空仓/国债 | 震荡→维持上期状态
"""
import pandas as pd
import numpy as np


class RegimeFilter:
    """双条件大盘状态过滤器

    规则:
    - BULL: 价格 > MA200 且 MA200 5日斜率 > 0
    - BEAR: 价格 < MA200 或 MA200斜率 < 0
    - 缓冲: 需连续3天满足条件才切换状态 (防止反复打脸)
    """

    def __init__(self, ma_window=200, slope_window=5, buffer_days=3):
        self.ma_window = ma_window
        self.slope_window = slope_window
        self.buffer_days = buffer_days

    def compute_regime(self, benchmark_price):
        """
        计算每日状态

        Args:
            benchmark_price: Series [date] 沪深300收盘价

        Returns:
            regime: Series [date] 'BULL' | 'BEAR'
            ma200: Series [date] MA200值
            slope: Series [date] MA200斜率
        """
        close = benchmark_price.dropna().sort_index()
        if len(close) < self.ma_window + self.slope_window:
            return None, None, None

        ma200 = close.rolling(self.ma_window, min_periods=self.ma_window).mean()
        # MA200斜率: 5日变化率
        slope = ma200.pct_change(self.slope_window)

        # 原始信号: True=牛市, False=熊市
        # 条件1: 价格>MA200 (主要)
        # 条件2: MA200斜率>-1%年化 (防止MA200下行加速时进场, 但不过滤缓慢走平)
        raw_signal = (close > ma200) & (slope > -0.01)

        # 缓冲: 连续buffer_days满足才切换
        regime = self._apply_buffer(raw_signal)

        return regime, ma200, slope

    def _apply_buffer(self, raw_signal):
        """缓冲: 相同方向连续buffer_days天确认后才切换, 方向变化立即重置计数"""
        # 初始状态根据前buffer_days天多数决定
        initial = raw_signal.iloc[:self.buffer_days].mean() >= 0.5
        current_regime = 'BULL' if initial else 'BEAR'

        regime = pd.Series('BULL', index=raw_signal.index)
        regime.iloc[:] = current_regime  # 先填满

        bull_streak = 0   # 连续牛市天数
        bear_streak = 0   # 连续熊市天数

        for i, date in enumerate(raw_signal.index):
            bullish = raw_signal.iloc[i]

            if bullish:
                bull_streak += 1
                bear_streak = 0
                if bull_streak >= self.buffer_days and current_regime != 'BULL':
                    current_regime = 'BULL'
            else:
                bear_streak += 1
                bull_streak = 0
                if bear_streak >= self.buffer_days and current_regime != 'BEAR':
                    current_regime = 'BEAR'

            regime.iloc[i] = current_regime

        return regime

    def add_crash_override(self, regime, benchmark_price, crash_threshold=-0.05, crash_window=3):
        """
        崩盘越级跳闸: 3日跌超5% → 无视缓冲直接切BEAR, 保持10天
        """
        close = benchmark_price.dropna().sort_index()
        ret_3d = close.pct_change(crash_window)
        crash = ret_3d < crash_threshold

        # 崩盘后锁死10天BEAR
        regime_corrected = regime.copy()
        crash_lock = 0
        for i, date in enumerate(regime.index):
            if date in crash.index and crash.loc[date]:
                crash_lock = 10
            if crash_lock > 0:
                regime_corrected.iloc[i] = 'BEAR'
                crash_lock -= 1

        return regime_corrected

    def add_quick_brake(self, regime, benchmark_price):
        """
        急刹车: 连跌3天 + 破20日均线 → 立即切BEAR (不等缓冲)
        """
        close = benchmark_price.dropna().sort_index()
        chg = close.pct_change()
        ma20 = close.rolling(20).mean()

        # 连跌3天
        down_streak = (chg < 0).astype(int)
        down_streak = down_streak.rolling(3).sum() == 3

        # 破20日线
        below_ma20 = close < ma20

        # 两个条件同时满足 → BEAR
        emergency = down_streak & below_ma20

        regime_corrected = regime.copy()
        for i, date in enumerate(regime.index):
            if date in emergency.index and emergency.loc[date]:
                regime_corrected.iloc[i] = 'BEAR'

        return regime_corrected

    def get_fragile_mask(self, regime):
        """
        脆牛期: BEAR→BULL切换后的前40个交易日 = 脆牛
        这段时间选股应偏向低波动/防御型行业
        """
        fragile = pd.Series(False, index=regime.index)
        prev = regime.iloc[0]
        fragile_days = 0

        for i in range(len(regime)):
            cur = regime.iloc[i]
            if prev == 'BEAR' and cur == 'BULL':
                fragile_days = 40  # 2个月脆牛期
            if fragile_days > 0:
                fragile.iloc[i] = True
                fragile_days -= 1
            prev = cur

        return fragile

    def get_position_mask(self, regime, return_matrix, benchmark_price=None):
        """
        生成仓位掩码 (含崩盘越级跳闸+急刹车修正)
        """
        # 崩盘越级跳闸
        if benchmark_price is not None:
            regime = self.add_crash_override(regime, benchmark_price)
            regime = self.add_quick_brake(regime, benchmark_price)

        aligned_regime = regime.reindex(return_matrix.index, method='ffill').fillna('BEAR')
        mask = aligned_regime == 'BULL'
        active_ret = return_matrix.multiply(mask, axis=0)

        # 脆牛掩码 (用于选股层)
        fragile_mask = self.get_fragile_mask(aligned_regime)

        return active_ret, aligned_regime, fragile_mask
