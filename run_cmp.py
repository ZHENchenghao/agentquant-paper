# -*- coding: utf-8 -*-
""" Run one config from backtest_full_compare.py """
import sys, json
config_name = sys.argv[1]
# Import the shared pipeline + data from backtest_full_compare
# But run only one config
exec(compile(open('backtest_full_compare.py').read(), 'backtest_full_compare.py', 'exec'))
