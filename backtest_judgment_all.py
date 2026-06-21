# -*- coding: utf-8 -*-
"""
AgentQuant · 全部判断因子回测
=============================
10年历史数据(DuckDB) → 逐个验证
"""
import duckdb, pandas as pd, numpy as np, io, sys
DB = 'D:/FreeFinanceData/data/duckdb/finance.db'

def cdb(): return duckdb.connect(DB, read_only=True)

def run_all():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    c = cdb()
    td = c.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0]
    print(f'数据截止: {td}')
    print()

    # ═══ 1: 国家队 — 放宽五重证据 ═══
    print('=== 1. 国家队 (放宽五重) ===')
    df = c.execute("""
        WITH idx AS (
            SELECT k1.trade_date,
                   (k1.close/LAG(k1.close) OVER w1-1)*100 chg50,
                   (k2.close/LAG(k2.close) OVER w2-1)*100 chg688,
                   (k3.close/LAG(k3.close) OVER w3-1)*100 chg300,
                   (LEAD(k3.close,5) OVER w3/k3.close-1)*100 ret5,
                   (LEAD(k3.close,20) OVER w3/k3.close-1)*100 ret20
            FROM kline_daily k1
            JOIN kline_daily k2 ON k1.trade_date=k2.trade_date
            JOIN kline_daily k3 ON k1.trade_date=k3.trade_date
            WHERE k1.ts_code='sh000016' AND k2.ts_code='sh000688' AND k3.ts_code='sh000300'
            WINDOW w1 AS (ORDER BY k1.trade_date), w2 AS (ORDER BY k2.trade_date), w3 AS (ORDER BY k3.trade_date)
        ),
        vol AS (SELECT trade_date, SUM(amount)/1e8 vol, AVG(SUM(amount)/1e8) OVER(ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) avg20 FROM kline_daily GROUP BY trade_date),
        adv AS (SELECT trade_date, COUNT(*) FILTER(WHERE change_pct>0) up, COUNT(*) FILTER(WHERE change_pct<0) dn FROM kline_daily GROUP BY trade_date)
        SELECT i.trade_date, i.chg50, i.chg688, i.chg300, v.vol/v.avg20 vr, a.up*1.0/(a.up+a.dn) br,
               CASE WHEN i.chg50>0.5 AND i.chg688<0 AND v.vol/v.avg20<0.9 THEN '疑似护盘'
                    WHEN i.chg50>1 AND a.up*1.0/(a.up+a.dn)<0.35 AND i.chg300>0 THEN '权重独拉'
                    WHEN i.chg50<-1 AND v.vol/v.avg20>1.3 THEN '国家队撤退+放量'
                    ELSE '正常' END signal,
               i.ret5, i.ret20
        FROM idx i JOIN vol v ON i.trade_date=v.trade_date JOIN adv a ON i.trade_date=a.trade_date
        WHERE i.trade_date>='2015-01-01'
    """).df()
    for label in ['疑似护盘','权重独拉','国家队撤退+放量']:
        sub=df[df['signal']==label]
        if len(sub)>3:
            r5=sub['ret5'].mean(); r20=sub['ret20'].mean(); wr=(sub['ret5']>0).mean()*100
            print('  %s: %d次 5日%+.2f%% 20日%+.2f%% 胜率%.0f%%' % (label, len(sub), r5, r20, wr))
    c.close()
    print()

    # ═══ 2: 北向背离 — 北向出+内资拉 ═══
    print('=== 2. 北向-内资背离 ===')
    c=cdb()
    df=c.execute("""
        WITH nb AS (SELECT trade_date, SUM(net_flow) daily FROM north_bound_flow WHERE net_flow!=0 GROUP BY trade_date),
        idx AS (SELECT trade_date, (close/LAG(close) OVER w-1)*100 chg, (LEAD(close,10) OVER w/close-1)*100 ret10 FROM kline_daily WHERE ts_code='sh000300' WINDOW w AS (ORDER BY trade_date))
        SELECT n.trade_date, n.daily, i.chg,
               CASE WHEN n.daily<-30 AND i.chg>0 THEN '北向流出+指数涨(背离)'
                    WHEN n.daily>30 AND i.chg<0 THEN '北向流入+指数跌(背离)'
                    WHEN n.daily>30 AND i.chg>0 THEN '北向流入+指数涨(共振多)'
                    WHEN n.daily<-30 AND i.chg<0 THEN '北向流出+指数跌(共振空)'
                    ELSE '正常' END sig, i.ret10
        FROM nb n JOIN idx i ON n.trade_date=i.trade_date WHERE i.trade_date>='2015-01-01'
    """).df()
    for label in ['北向流出+指数涨(背离)','北向流入+指数跌(背离)','北向流入+指数涨(共振多)','北向流出+指数跌(共振空)']:
        sub=df[df['sig']==label]
        if len(sub)>5:
            print(f'  {label}: {len(sub)}次 10日{sub["ret10"].mean():+.2f}% 胜率{(sub["ret10"]>0).mean()*100:.0f}%')
    c.close()
    print()

    # ═══ 3: 跳空缺口 ═══
    print('=== 3. 跳空缺口 (沪深300) ===')
    c=cdb()
    df=c.execute("""
        SELECT trade_date,
               (kline_daily.open/LAG(kline_daily.close) OVER w-1)*100 gap,
               (LEAD(close,5) OVER w/close-1)*100 ret5
        FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2015-01-01'
        WINDOW w AS (ORDER BY trade_date)
        QUALIFY gap IS NOT NULL
    """).df()
    gap_up=df[df['gap']>0.8]; gap_dn=df[df['gap']<-0.8]
    print(f'  跳空高开>0.8%: {len(gap_up)}次 5日{gap_up["ret5"].mean():+.2f}% 胜率{(gap_up["ret5"]>0).mean()*100:.0f}%')
    print(f'  跳空低开>0.8%: {len(gap_dn)}次 5日{gap_dn["ret5"].mean():+.2f}% 胜率{(gap_dn["ret5"]>0).mean()*100:.0f}%')
    c.close()
    print()

    # ═══ 4: 缩量下跌 — 是洗盘还是真跌 ═══
    print('=== 4. 缩量下跌 vs 放量下跌 ===')
    c=cdb()
    df=c.execute("""
        WITH v AS (SELECT trade_date, SUM(amount)/1e8 vol, AVG(SUM(amount)/1e8) OVER(ORDER BY trade_date ROWS 19 PRECEDING) avg20 FROM kline_daily GROUP BY trade_date),
        p AS (SELECT trade_date, (close/LAG(close) OVER w-1)*100 chg, (LEAD(close,5) OVER w/close-1)*100 ret5, (LEAD(close,20) OVER w/close-1)*100 ret20 FROM kline_daily WHERE ts_code='sh000300' WINDOW w AS (ORDER BY trade_date))
        SELECT p.trade_date, p.chg, v.vol/v.avg20 vr,
               CASE WHEN p.chg<-1 AND v.vol/v.avg20<0.7 THEN '缩量下跌(洗盘?)'
                    WHEN p.chg<-1 AND v.vol/v.avg20>1.3 THEN '放量下跌(真跌)'
                    WHEN p.chg>1 AND v.vol/v.avg20<0.7 THEN '缩量上涨(诱多?)'
                    WHEN p.chg>1 AND v.vol/v.avg20>1.3 THEN '放量上涨(真涨)'
                    ELSE '正常' END sig, p.ret5, p.ret20
        FROM p JOIN v ON p.trade_date=v.trade_date WHERE p.trade_date>='2015-01-01'
    """).df()
    for label in ['缩量下跌(洗盘?)','放量下跌(真跌)','缩量上涨(诱多?)','放量上涨(真涨)']:
        sub=df[df['sig']==label]
        if len(sub)>5:
            r5=sub['ret5'].mean(); r20=sub['ret20'].mean(); wr=(sub['ret5']>0).mean()*100
            print('  %s: %d次 5日%+.2f%% 20日%+.2f%% 胜率%.0f%%' % (label, len(sub), r5, r20, wr))
    c.close()
    print()

    # ═══ 5: 连涨/连跌 — 第N天反转概率 ═══
    print('=== 5. 连涨连跌反转 ===')
    c=cdb()
    df=c.execute("""
        WITH streak AS (
            SELECT trade_date,
                   CASE WHEN close>LAG(close) OVER w THEN 1 ELSE 0 END up,
                   (LEAD(close,1) OVER w/close-1)*100 ret1
            FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2015-01-01'
            WINDOW w AS (ORDER BY trade_date)
        ),
        labeled AS (
            SELECT trade_date, up,
                   up+LAG(up) OVER o+LAG(up,2) OVER o+LAG(up,3) OVER o+LAG(up,4) OVER o streak,
                   ret1
            FROM streak WINDOW o AS (ORDER BY trade_date)
        )
        SELECT CASE WHEN streak=5 THEN '5连涨'
                    WHEN streak=0 THEN '5连跌'
                    WHEN streak=4 THEN '4涨1跌'
                    WHEN streak=1 THEN '1涨4跌' END sig, ret1
        FROM labeled WHERE streak IN (0,1,4,5) AND ret1 IS NOT NULL
    """).df()
    for label in ['5连涨','5连跌','4涨1跌','1涨4跌']:
        sub=df[df['sig']==label]
        if len(sub)>5:
            r1=sub['ret1'].mean()
            if '涨' in label: rev_rate=(sub['ret1']<0).mean()*100
            else: rev_rate=(sub['ret1']>0).mean()*100
            print('  %s: %d次 次日%+.2f%% 反转概率%.0f%%' % (label, len(sub), r1, rev_rate))
    c.close()
    print()

    # ═══ 6: 节前效应 — 长假前3天 ═══
    print('=== 6. 节前效应 ===')
    c=cdb()
    df=c.execute("""
        SELECT trade_date,
               (LEAD(close,1) OVER w/close-1)*100 ret1,
               (LEAD(close,3) OVER w/close-1)*100 ret3,
               EXTRACT(MONTH FROM trade_date) m, EXTRACT(DAY FROM trade_date) d
        FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2015-01-01'
        WINDOW w AS (ORDER BY trade_date)
    """).df()
    # 春节前(1-2月最后3天)/国庆前(9月最后3天) 简化: 月末最后3天
    df['月末']=df['trade_date'].apply(lambda x: x.day>=28)
    eom=df[df['月末']]
    print(f'  月末最后3天: {len(eom)}次 次日{eom["ret1"].mean():+.2f}% 3日{eom["ret3"].mean():+.2f}%')
    c.close()
    print()

    # ═══ 7: 放量滞涨 — 天量天价 ═══
    print('=== 7. 放量滞涨 (天量天价) ===')
    c=cdb()
    df=c.execute("""
        WITH v AS (SELECT trade_date, SUM(amount)/1e8 vol, AVG(SUM(amount)/1e8) OVER(ORDER BY trade_date ROWS 19 PRECEDING) avg20 FROM kline_daily GROUP BY trade_date),
        p AS (SELECT trade_date, close, (close/LAG(close) OVER w-1)*100 chg, (LEAD(close,10) OVER w/close-1)*100 ret10 FROM kline_daily WHERE ts_code='sh000300' WINDOW w AS (ORDER BY trade_date))
        SELECT p.trade_date, p.chg, v.vol/v.avg20 vr,
               CASE WHEN p.chg BETWEEN -0.3 AND 0.3 AND v.vol/v.avg20>1.5 THEN '放量滞涨(天量天价)'
                    WHEN p.chg>1.5 AND v.vol/v.avg20>1.5 THEN '放量大涨' END sig, p.ret10
        FROM p JOIN v ON p.trade_date=v.trade_date WHERE p.trade_date>='2015-01-01'
    """).df()
    for label in ['放量滞涨(天量天价)','放量大涨']:
        sub=df[df['sig']==label]
        if len(sub)>5:
            print(f'  {label}: {len(sub)}次 10日{sub["ret10"].mean():+.2f}% 胜率{(sub["ret10"]>0).mean()*100:.0f}%')
    c.close()
    print()

    # ═══ 8: 融资余额变动 ═══
    print('=== 8. 融资余额急降 (杠杆踩踏) ===')
    c=cdb()
    df=c.execute("""
        WITH margin AS (
            SELECT trade_date, margin_balance,
                   (margin_balance/LAG(margin_balance) OVER w-1)*100 chg,
                   (LEAD(margin_balance,5) OVER w/margin_balance-1)*100 ret5
            FROM margin_trading WHERE trade_date>='2015-01-01' AND margin_balance IS NOT NULL
            WINDOW w AS (ORDER BY trade_date)
        ),
        p AS (SELECT trade_date, (LEAD(close,10) OVER w/close-1)*100 ret10 FROM kline_daily WHERE ts_code='sh000300' WINDOW w AS (ORDER BY trade_date))
        SELECT m.trade_date, m.chg margin_chg,
               CASE WHEN m.chg<-5 THEN '融资急降>5%' WHEN m.chg<-2 THEN '融资下降2-5%' WHEN m.chg>5 THEN '融资急增>5%' ELSE '正常' END sig, p.ret10
        FROM margin m JOIN p ON m.trade_date=p.trade_date
    """).df()
    for label in ['融资急降>5%','融资下降2-5%','融资急增>5%']:
        sub=df[df['sig']==label]
        if len(sub)>5:
            print(f'  {label}: {len(sub)}次 10日{sub["ret10"].mean():+.2f}% 胜率{(sub["ret10"]>0).mean()*100:.0f}%')
    c.close()
    print()

    # ═══ 9: 行业轮动速度 ═══
    print('=== 9. 行业轮动加速 (热点切换快) ===')
    c=cdb()
    df=c.execute("""
        WITH ind_ret AS (
            SELECT stock_code, trade_date,
                   (close/LAG(close) OVER(PARTITION BY stock_code ORDER BY trade_date)-1)*100 ret
            FROM proxy_industry_daily WHERE trade_date>='2015-01-01'
        ),
        rank_daily AS (
            SELECT trade_date, stock_code, ret,
                   RANK() OVER(PARTITION BY trade_date ORDER BY ret DESC) rk
            FROM ind_ret WHERE ret IS NOT NULL
        ),
        turnover AS (
            SELECT trade_date,
                   COUNT(DISTINCT CASE WHEN rk=1 THEN stock_code END) top_changes,
                   -- 连续2天排名第一的行业是否换人
                   COUNT(DISTINCT stock_code) total
            FROM rank_daily WHERE rk<=5 GROUP BY trade_date
        ),
        p AS (SELECT trade_date, (LEAD(close,5) OVER w/close-1)*100 ret5 FROM kline_daily WHERE ts_code='sh000300' WINDOW w AS (ORDER BY trade_date))
        SELECT t.trade_date, t.top_changes,
               CASE WHEN t.top_changes>=4 THEN '轮动极快(日换4+行业)' ELSE '轮动正常' END sig, p.ret5
        FROM turnover t JOIN p ON t.trade_date=p.trade_date
    """).df()
    for label in ['轮动极快(日换4+行业)','轮动正常']:
        sub=df[df['sig']==label]
        if len(sub)>5:
            print(f'  {label}: {len(sub)}次 5日{sub["ret5"].mean():+.2f}% 胜率{(sub["ret5"]>0).mean()*100:.0f}%')
    c.close()

    print('\n' + '='*60)
    print('  汇总: 哪些判断因子值得用?')
    print('='*60)


if __name__ == '__main__':
    run_all()
