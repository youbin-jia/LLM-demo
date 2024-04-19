#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:08:31 2024

@author: jyb
"""

import webbrowser

def open_gaode_navigation(origin, destination):
    # 构造高德地图导航页面的 URL
    url = f"https://uri.amap.com/navigation?from={origin}&to={destination}&mode=car&policy=0"
    
    # 打开构造的 URL
    webbrowser.open(url)

# 使用函数调用高德地图导航
origin = "Beijing"
destination = "Shanghai"
open_gaode_navigation(origin, destination)