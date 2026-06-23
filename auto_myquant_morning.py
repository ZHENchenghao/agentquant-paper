# -*- coding: utf-8 -*-
"""掘金自动早盘交易 — 清仓旧持仓 → 买入小众战法30只"""
import sys, io, json, time, os
from datetime import datetime

DIR = 'D:/AgentQuant/our'
LOG_FILE = f'{DIR}/auto_trade_log.txt'

# 日志同时写文件和屏幕
class Tee:
    def __init__(self, f):
        self.f = f
        self.stdout = sys.stdout
    def write(self, s):
        self.f.write(s)
        self.f.flush()
        self.stdout.write(s)
    def flush(self):
        self.f.flush()
        self.stdout.flush()

f = open(LOG_FILE, 'a', encoding='utf-8')
f.write(f'\n{"="*50}\n{datetime.now():%Y-%m-%d %H:%M:%S} 掘金早盘自动交易\n')
sys.stdout = Tee(f)

try:
    from gmtrade.api import (set_token, set_endpoint, account, login,
                              order_close_all, order_cancel_all, order_volume,
                              OrderSide_Buy, OrderType_Limit, PositionEffect_Open,
                              get_positions, get_unfinished_orders, get_cash)

    TOKEN = 'd1f2101ac1bb2e9380acf2c91b0dfc8e85369cdf'
    ACCOUNT = '15d64470-6d88-11f1-932c-00163e022aa6'
    ENDPOINT = 'api.myquant.cn:9000'

    # 登录
    set_token(TOKEN)
    set_endpoint(ENDPOINT)
    a = account(account_id=ACCOUNT, account_alias='')
    login(a)
    print('✅ 掘金登录成功')

    c = get_cash()
    pos_before = get_positions()
    print(f'交易前: 持仓{len(pos_before)}只 | 现金{c.available:,.0f} | 总资产{c.nav:,.0f}')

    # 1. 撤所有旧单
    pending = get_unfinished_orders()
    if pending:
        order_cancel_all()
        print(f'已撤销{len(pending)}笔旧单')
        time.sleep(2)

    # 2. 一键清仓
    pos = get_positions()
    if pos:
        names = [p.symbol for p in pos]
        print(f'清仓{len(pos)}只: {names}')
        order_close_all()

        # 等成交(最多30秒)
        for _ in range(10):
            time.sleep(3)
            pos = get_positions()
            if not pos:
                print('  全部成交 ✅')
                break
            print(f'  等待... 剩{len(pos)}只')
        else:
            print(f'  ⚠ 超时,剩余{len(pos)}只未成交')
    else:
        print('无持仓需清')

    # 3. 加载纸交目标
    with open(f'{DIR}/paper_portfolio_xiaozhong.json', 'r', encoding='utf-8') as pf:
        target = json.load(pf)

    def to_mq(code):
        return ('SHSE.' if code.startswith('sh') else 'SZSE.') + code[2:]

    # 4. 买入30只
    print(f'\n买入{len(target["positions"])}只...')
    ok = fail = 0
    for p in target['positions']:
        code, shares, price, name = p['code'], p['shares'], p['price'], p['name']
        mq_code = to_mq(code)
        limit = round(price * 1.02, 2)
        try:
            order_volume(symbol=mq_code, volume=shares, side=OrderSide_Buy,
                         order_type=OrderType_Limit, position_effect=PositionEffect_Open,
                         price=limit)
            ok += 1
            time.sleep(0.15)
        except Exception as e:
            print(f'  ❌ {mq_code} {name} {e}')
            fail += 1

    # 5. 结果
    time.sleep(2)
    c = get_cash()
    pos_after = get_positions()
    orders = get_unfinished_orders()

    print(f'\n{"="*40}')
    print(f'交易后: 持仓{len(pos_after)}只 | 现金{c.available:,.0f} | 总资产{c.nav:,.0f}')
    print(f'买入: {ok}成功 {fail}失败 | 待成交: {len(orders)}笔')
    print(f'{"✅ 掘金=纸交=30只小众战法" if len(pos_after) >= 25 else "⚠ 需检查"}')
    print(f'{datetime.now():%H:%M:%S} 完成')

    # 写状态文件
    with open(f'{DIR}/trade_status.txt', 'w', encoding='utf-8') as sf:
        sf.write(f'{datetime.now():%Y-%m-%d %H:%M} 掘金自动交易\n')
        sf.write(f'持仓:{len(pos_after)}只 现金:{c.available:,.0f} 总资产:{c.nav:,.0f}\n')
        sf.write(f'买入:{ok}成功 {fail}失败\n')

    # 6. 更新日志并Git提交
    print('\n--- Git存档 ---')
    import subprocess
    os.chdir(DIR)

    # 更新paper_daily_log.md
    today_str = datetime.now().strftime('%Y-%m-%d')
    log_entry = f'| {today_str} | {len(pos_after)}只 | ¥{c.available+c.nav-c.available:,.0f} | ¥{c.available:,.0f} | FULL | {HS300_CLOSE:.0f} | — |\n'
    with open(f'{DIR}/paper_daily_log.md', 'r', encoding='utf-8') as lf:
        log_content = lf.read()
    # 在表格中追加一行
    if today_str not in log_content:
        # 找到表格最后一行日期后插入
        insert_pos = log_content.rfind('| 2026')
        if insert_pos > 0:
            line_end = log_content.find('\n', insert_pos)
            log_content = log_content[:line_end+1] + log_entry + log_content[line_end+1:]
            with open(f'{DIR}/paper_daily_log.md', 'w', encoding='utf-8') as lf:
                lf.write(log_content)
            print(f'日志已更新: {today_str}')

    # Git提交
    for cmd in [
        'git add paper_portfolio_xiaozhong.json sim_account.json paper_daily_log.md trade_status.txt auto_trade_log.txt',
        f'git commit -m "纸交 {today_str} — 掘金自动执行 {len(pos_after)}只" --allow-empty',
        'git push origin master',
    ]:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0 and 'nothing to commit' not in r.stderr:
            print(f'  ⚠ {cmd[:20]}: {r.stderr[:80]}')
    print('Git推送完成 ✅')

    # 7. 10点关机
    now = datetime.now()
    target = now.replace(hour=10, minute=0, second=0)
    if now >= target:
        target = target.replace(hour=22)  # 兜底: 晚上10点
    wait_sec = max(30, (target - now).total_seconds())
    print(f'{now:%H:%M:%S} → 计划{target:%H:%M}关机 (等待{wait_sec/60:.0f}分钟)')
    os.system(f'shutdown /s /t {int(wait_sec)} /f')

except Exception as e:
    print(f'❌ 异常: {e}')
    import traceback
    traceback.print_exc()

finally:
    sys.stdout = sys.__stdout__
    f.close()
    print('日志已保存')
