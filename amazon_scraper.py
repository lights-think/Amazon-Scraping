import asyncio
import re
import pandas as pd
import click
from tqdm import tqdm
from playwright.async_api import async_playwright
import logging
import random
import os
import json
import time
import multiprocessing
from multiprocessing import Process, Manager


# 日志配置：仅写入文件 spider.log
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 清除已有的 handlers
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler('spider.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 国家域名映射
DOMAIN_MAP = {
    'US': 'amazon.com',
    'UK': 'amazon.co.uk',
    'DE': 'amazon.de',
    'FR': 'amazon.fr',
    'ES': 'amazon.es',
    'IT': 'amazon.it',
    'CA': 'amazon.ca',
    'JP': 'amazon.co.jp',
    'MX': 'amazon.com.mx',
    'IN': 'amazon.in',
    'NL': 'amazon.nl',
    'SE': 'amazon.se',
    'BE': 'amazon.com.be',
    'IE': 'amazon.ie',
    'AU': 'amazon.com.au',
    'BR': 'amazon.com.br',
    'SG': 'amazon.sg'
}

# 国家语言环境映射
LOCALE_MAP = {
    'US': 'en-US',
    'UK': 'en-GB', 
    'DE': 'de-DE',
    'FR': 'fr-FR',
    'ES': 'es-ES',
    'IT': 'it-IT',
    'CA': 'en-CA',
    'JP': 'ja-JP',
    'MX': 'es-MX',
    'IN': 'en-IN',
    'NL': 'nl-NL',
    'SE': 'sv-SE',
    'BE': 'nl-BE',
    'IE': 'en-IE',
    'AU': 'en-AU',
    'BR': 'pt-BR',
    'SG': 'en-SG'
}

# 国家时区映射
TIMEZONE_MAP = {
    'US': 'America/New_York',
    'UK': 'Europe/London',
    'DE': 'Europe/Berlin', 
    'FR': 'Europe/Paris',
    'ES': 'Europe/Madrid',
    'IT': 'Europe/Rome',
    'CA': 'America/Toronto',
    'JP': 'Asia/Tokyo',
    'MX': 'America/Mexico_City',
    'IN': 'Asia/Kolkata',
    'NL': 'Europe/Amsterdam',
    'SE': 'Europe/Stockholm',
    'BE': 'Europe/Brussels',
    'IE': 'Europe/Dublin',
    'AU': 'Australia/Sydney',
    'BR': 'America/Sao_Paulo',
    'SG': 'Asia/Singapore'
}

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.114 Safari/537.36"

# === 新增：递归提取BSR主类与所有子类的通用函数 ===
async def extract_bsr_from_node(node):
    """
    递归遍历ul/li结构，提取所有li文本中的BSR排名和类别，兼容多语言。
    返回列表[(rank, category), ...]
    优先英文正则（in），未命中再尝试其他语言正则。
    """
    results = []
    patterns_en = [
        r'(?:#|No\.)?\s*([0-9]+(?:,[0-9]{3})*)\s+in\s+([^(#]+?)(?:\s*\(|$)'
    ]
    patterns_other = [
        r'#([0-9]+(?:\s+[0-9]{3})*)\s+i\s+([^(#]+?)(?:\s*\(|$)',  # 瑞典
        r'#([0-9]+(?:\.[0-9]{3})*)\s+in\s+([^(#]+?)(?:\s*\(|$)',  # 荷兰/英文
        r'(?:nº|n°|No\.)\s*([0-9]+(?:\.[0-9]{3})*)\s+en\s+([^(#]+?)(?:\s*\(|$)',  # 西班牙
        r'(?:n°|No\.)\s*([0-9]+(?:\.[0-9]{3})*)\s+dans\s+([^(#]+?)(?:\s*\(|$)',  # 法语
        r'(?:Nr\.|Rang\s+Nr\.)\s*([0-9]+(?:\.[0-9]{3})*)\s+(?:in|bei)\s+([^(#]+?)(?:\s*\(|$)',  # 德语
        r'n\.\s*([0-9]+(?:\.[0-9]{3})*)\s+in\s+([^(#]+?)(?:\s*\(|$)',  # 意大利
        r'([0-9]+(?:,[0-9]{3})*)位([^0-9（\(]+?)(?:\s*（|\s*\(|$)',  # 日语
        r'(?:排第|排名第)([0-9,，]+)名\s*([^排名（]+?)(?:\s*（|$)'  # 中文
    ]
    
    # 新增：处理特殊情况，当节点是包含"Best Sellers Rank"的span时，先提取父类再递归处理子类
    span_text = ""
    try:
        span_text = await node.inner_text()
        span_text = re.sub(r'\s+', ' ', span_text).strip()
    except:
        pass
    
    if "Best Sellers Rank" in span_text or "Amazon Best Sellers Rank" in span_text:
        # 这可能是一个包含父类和子类的复合结构
        logging.info(f"[BSR-SPAN] 发现复合BSR结构: '{span_text}'")
        
        # 1. 先从span文本中提取父类信息
        for pattern in patterns_en:
            matches = re.findall(pattern, span_text, re.IGNORECASE)
            for m in matches:
                rank_str = m[0].replace(',', '').replace('.', '').replace(' ', '').replace('，', '')
                cat = m[1].strip().rstrip(')）')
                if rank_str.isdigit() and 1 <= int(rank_str) <= 10000000:
                    results.append((rank_str, cat))
                    logging.info(f"[BSR-SPAN-PARENT] 提取到父类: ({rank_str}, {cat})")
        
        # 2. 然后查找并处理内部的ul/li结构获取子类信息
        ul_nodes = await node.query_selector_all('ul')
        for ul_node in ul_nodes:
            sub_results = await extract_bsr_from_node(ul_node)
            if sub_results:
                results.extend(sub_results)
                logging.info(f"[BSR-SPAN-CHILDREN] 提取到子类: {sub_results}")
        
        # 如果已经找到结果，直接返回
        if results:
            return results
    
    # 继续原有的处理逻辑
    li_nodes = await node.query_selector_all(':scope > li')
    if li_nodes:
        for li in li_nodes:
            sub_ul = await li.query_selector('ul')
            if sub_ul:
                results.extend(await extract_bsr_from_node(sub_ul))
            span = await li.query_selector('span')
            if span:
                text = await span.inner_text()
            else:
                a = await li.query_selector('a')
                if a:
                    text = await a.inner_text()
                else:
                    text = await li.inner_text()
            text = re.sub(r'\s+', ' ', text).strip()
            # --- 增强：结构化提取BSR，精准区分导航li和真实BSR li ---
            NAV_PHRASES = [
                'See Top 100 in ', 'Top 100 in ',
                'Visualizza i Top 100 nella categoria', 'Ver el Top 100 en',
                'Visa Topp 100 i', 'Voir les 100 premiers en',
                'Ver os 100 mais vendidos em', 'Top 100 auf',
                'Top 100 in', 'Top 100 dans', 'Top 100 en',
                'Top 100 dei', 'Top 100 der', 'Top 100 de',
            ]
            text_stripped = text.strip()
            # 1. 导航li判别：以导航短语开头或包含导航短语，直接跳过
            # 增强：特别过滤掉排名为100的导航链接项
            if (any(nav_phrase in text_stripped for nav_phrase in NAV_PHRASES) and 
                (not re.search(r'[#nNoNr\d]', text_stripped) or 
                 re.search(r'[#nNoNr\.\s]*100\s+in\s+', text_stripped, re.IGNORECASE))):
                logging.info(f"[BSR-NAV-SKIP] 跳过导航链接: '{text_stripped}'")
                continue
            # 2. 若li主文本为"#N in "，a标签文本为category，则category取a标签文本
            m = re.match(r'[#nNoNr\.\s]*(\d+[,.\d]*)\s+in\s*$', text_stripped, re.IGNORECASE)
            if m:
                rank_str = m.group(1).replace(',', '').replace('.', '').replace(' ', '').replace('，', '')
                a = await li.query_selector('a')
                if a:
                    cat = (await a.inner_text()).strip().rstrip(')）')
                    if rank_str.isdigit() and 1 <= int(rank_str) <= 10000000:
                        results.append((rank_str, cat))
                        continue
            # 3. 其余情况按原有正则提取
            found = False
            # 优先英文正则
            for pattern in patterns_en:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for m in matches:
                    rank_str = m[0].replace(',', '').replace('.', '').replace(' ', '').replace('，', '')
                    cat = m[1].strip().rstrip(')）')
                    if rank_str.isdigit() and 1 <= int(rank_str) <= 10000000:
                        # 增强：排除可能的导航链接（排名为100且文本包含导航短语）
                        if int(rank_str) == 100 and any(nav_phrase in text for nav_phrase in NAV_PHRASES):
                            logging.info(f"[BSR-FILTER] 排除导航链接: rank={rank_str}, category={cat}, text='{text}'")
                            continue
                        results.append((rank_str, cat))
                        found = True
            # 只要英文正则没命中，再尝试其他语言
            if not found:
                for pattern in patterns_other:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for m in matches:
                        rank_str = m[0].replace(',', '').replace('.', '').replace(' ', '').replace('，', '')
                        cat = m[1].strip().rstrip(')）')
                        if rank_str.isdigit() and 1 <= int(rank_str) <= 10000000:
                            # 增强：排除可能的导航链接（排名为100且文本包含导航短语）
                            if int(rank_str) == 100 and any(nav_phrase in text for nav_phrase in NAV_PHRASES):
                                logging.info(f"[BSR-FILTER] 排除导航链接: rank={rank_str}, category={cat}, text='{text}'")
                                continue
                            results.append((rank_str, cat))
            logging.info(f"[BSR-LI] li_text='{text}' matches={results}")
    else:
        text = await node.inner_text()
        text = re.sub(r'\s+', ' ', text).strip()
        found = False
        for pattern in patterns_en:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                rank_str = m[0].replace(',', '').replace('.', '').replace(' ', '').replace('，', '')
                cat = m[1].strip().rstrip(')）')
                if rank_str.isdigit() and 1 <= int(rank_str) <= 10000000:
                    # 增强：排除可能的导航链接（排名为100且文本包含导航短语）
                    if int(rank_str) == 100 and any(nav_phrase in text for nav_phrase in NAV_PHRASES):
                        logging.info(f"[BSR-FILTER-NON-LI] 排除导航链接: rank={rank_str}, category={cat}, text='{text}'")
                        continue
                    results.append((rank_str, cat))
                    found = True
        if not found:
            for pattern in patterns_other:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for m in matches:
                    rank_str = m[0].replace(',', '').replace('.', '').replace(' ', '').replace('，', '')
                    cat = m[1].strip().rstrip(')）')
                    if rank_str.isdigit() and 1 <= int(rank_str) <= 10000000:
                        # 增强：排除可能的导航链接（排名为100且文本包含导航短语）
                        if int(rank_str) == 100 and any(nav_phrase in text for nav_phrase in NAV_PHRASES):
                            logging.info(f"[BSR-FILTER-NON-LI] 排除导航链接: rank={rank_str}, category={cat}, text='{text}'")
                            continue
                        results.append((rank_str, cat))
        logging.info(f"[BSR-NON-LI] text='{text}' matches={results}")
    return results

async def fetch_product_data(page, url):
    # 随机延时，模拟人类访问
    await asyncio.sleep(random.uniform(0.8, 1.5)) # 稍微增加延时
    
    # 不强制添加语言参数，让Amazon根据地区自然显示
    # 我们依赖精确的多语言BSR解析策略来处理各种格式
    
    try:
        await page.goto(url, timeout=60000, wait_until='domcontentloaded')
        # 新增：检测并处理"继续购物"确认页面
        await handle_continue_shopping(page)
    except Exception as e:
        logging.error(f"Error navigating to {url}: {e}") # 使用 logging
        return '', '', '', '', '', '' # 返回空值以便重试逻辑判断

    # 根据展开图标状态决定是否点击展开 BSR 区块
    try:
        expander_link = await page.query_selector('#productDetails_expanderTables_depthLeftSections > div > span > a')
        if expander_link:
            icon = await expander_link.query_selector('i')
            if icon:
                cls = await icon.get_attribute('class') or ''
                if 'a-icon-section-expand' in cls:
                    await expander_link.click()
                    await page.wait_for_selector('#productDetails_expanderTables_depthLeftSections table', timeout=7000) # 增加超时
    except Exception as e:
        logging.debug(f"ASIN from URL {url} - Error or no expander for BSR: {e}") # 使用 debug 级别
        pass

    # 固定等待 2 秒，再进行 BSR 解析；如果第一次失败，再等待 2 秒重试一次
    await page.wait_for_timeout(2000)

    # === 基于真实格式的精确多语言BSR解析策略 V2 ===
    # 瑞典格式: "#49 449 i Bygg" (空格分隔千位)
    # 荷兰格式: "#4.328 in Kantoorproducten" (句号分隔千位)  
    # 西班牙格式: "nº20.691 en Productos" (句号分隔千位)
    # 日本格式: "4,312位ペット用品"
    # 意大利格式: "n. 2.477 in Cancelleria"
    
    main_rank = ''
    main_category = ''
    sub_rank = ''
    sub_category = ''
    
    # 策略1: 优先查找标准BSR表格结构
    try:
        bsr_table_rows = await page.query_selector_all('xpath=//tr[.//th[contains(text(),"Best Sellers Rank") or contains(text(),"Amazon Best Sellers Rank") or contains(text(),"Sales Rank") or contains(text(),"Rangordning för bästsäljare") or contains(text(),"Plaats in bestsellerlijst") or contains(text(),"Clasificación en los más vendidos") or contains(text(),"Classement des meilleures ventes") or contains(text(),"Bestseller-Rang") or contains(text(),"Posizione nella classifica")]]')
        for row in bsr_table_rows:
            td_content = await row.query_selector('td')
            if td_content:
                # 优先查找ul/li结构
                ul_node = await td_content.query_selector('ul')
                if ul_node:
                    bsr_list = await extract_bsr_from_node(ul_node)
                    main_rank, main_category, sub_rank, sub_category = '', '', '', ''
                    main_idx = -1
                    # 1. 优先li文本含'Top 100'或'See Top 100'为主类
                    # 添加局部正则列表，供下方提取排名使用
                    patterns_en = [
                        r'(?:#|No\.)?\s*([0-9]+(?:,[0-9]{3})*)\s+in\s+([^(#]+?)(?:\s*\(|$)'
                    ]
                    patterns_other = [
                        r'#([0-9]+(?:\s+[0-9]{3})*)\s+i\s+([^(#]+?)(?:\s*\(|$)',  # 瑞典
                        r'#([0-9]+(?:\.[0-9]{3})*)\s+in\s+([^(#]+?)(?:\s*\(|$)',  # 荷兰/英文
                        r'(?:nº|n°|No\.)\s*([0-9]+(?:\.[0-9]{3})*)\s+en\s+([^(#]+?)(?:\s*\(|$)',  # 西班牙
                        r'(?:n°|No\.)\s*([0-9]+(?:\.[0-9]{3})*)\s+dans\s+([^(#]+?)(?:\s*\(|$)',  # 法语
                        r'(?:Nr\.|Rang\s+Nr\.)\s*([0-9]+(?:\.[0-9]{3})*)\s+(?:in|bei)\s+([^(#]+?)(?:\s*\(|$)',  # 德语
                        r'n\.\s*([0-9]+(?:\.[0-9]{3})*)\s+in\s+([^(#]+?)(?:\s*\(|$)',  # 意大利
                        r'([0-9]+(?:,[0-9]{3})*)位([^0-9（\(]+?)(?:\s*（|\s*\(|$)',  # 日语
                        r'(?:排第|排名第)([0-9,，]+)名\s*([^排名（]+?)(?:\s*（|$)'  # 中文
                    ]
                    NAV_PHRASES = [
                        'See Top 100 in ', 'Top 100 in ',
                        'Visualizza i Top 100 nella categoria', 'Ver el Top 100 en',
                        'Visa Topp 100 i', 'Voir les 100 premiers en',
                        'Ver os 100 mais vendidos em', 'Top 100 auf',
                        'Top 100 in', 'Top 100 dans', 'Top 100 en',
                        'Top 100 dei', 'Top 100 der', 'Top 100 de',
                    ]
                    li_texts = []  # 仅记录产生排名的li文本，顺序与bsr_list对齐
                    li_nodes = await ul_node.query_selector_all(':scope > li')
                    for li in li_nodes:
                        text = ''
                        span = await li.query_selector('span')
                        if span:
                            text = await span.inner_text()
                        else:
                            a = await li.query_selector('a')
                            if a:
                                text = await a.inner_text()
                            else:
                                text = await li.inner_text()
                        text = re.sub(r'\s+', ' ', text).strip()
                        # 同步 regex 提取判断以决定是否加入 bsr_list 和 li_texts
                        found_local = False
                        tmp_results = []
                        for pattern in patterns_en:
                            matches = re.findall(pattern, text, re.IGNORECASE)
                            for m in matches:
                                rank_str = m[0].replace(',', '').replace('.', '').replace(' ', '').replace('，', '')
                                cat = m[1].strip().rstrip(')）')
                                if rank_str.isdigit() and 1 <= int(rank_str) <= 10000000:
                                    tmp_results.append((rank_str, cat))
                                    found_local = True
                        if not found_local:
                            for pattern in patterns_other:
                                matches = re.findall(pattern, text, re.IGNORECASE)
                                for m in matches:
                                    rank_str = m[0].replace(',', '').replace('.', '').replace(' ', '').replace('，', '')
                                    cat = m[1].strip().rstrip(')）')
                                    if rank_str.isdigit() and 1 <= int(rank_str) <= 10000000:
                                        tmp_results.append((rank_str, cat))
                                        found_local = True
                        if found_local:
                            bsr_list.extend(tmp_results)  # 按出现顺序追加
                            li_texts.append(text)  # 仅记录有效li文本
                            logging.info(f"[BSR-LI-VALID] li_text='{text}' matches={tmp_results}")
                        else:
                            logging.info(f"[BSR-LI-SKIP] li_text='{text}' skipped (no rank)")
                    # 匹配li_texts和bsr_list顺序现在天然一致
                    # 1. 优先li文本含'Top 100'或'See Top 100'为主类
                    main_idx = -1
                    for i, raw_text in enumerate(li_texts):
                        cleaned = re.sub(r'\([^\)]*\)', '', raw_text)
                        if 'Top 100' in cleaned or 'See Top 100' in cleaned:
                            main_idx = i
                            break
                    reason = ''
                    if main_idx >= 0 and main_idx < len(bsr_list):
                        main_rank, main_category = bsr_list[main_idx]
                        reason = '主类通过Top 100语义判别'
                    elif bsr_list:
                        # 2. 无Top 100则排名最大为主类
                        ranks = [(i, int(r[0])) for i, r in enumerate(bsr_list) if r[0].isdigit()]
                        if ranks:
                            main_idx = max(ranks, key=lambda x: x[1])[0]
                            main_rank, main_category = bsr_list[main_idx]
                            reason = '主类通过排名最大判别'
                    # 3. 子类为其余项中排名最小且与主类不同的项
                    sub_idx = -1
                    if main_idx >= 0 and len(bsr_list) > 1:
                        # 修正：子类的品类名必须和主类不同，且排除排名为100的可能干扰项
                        sub_candidates = []
                        for i, r in enumerate(bsr_list):
                            if (i != main_idx and r[0].isdigit() and r[1] and 
                                r[1] != main_category and 
                                (int(r[0]) != 100 or not any(nav_phrase in li_texts[i] for nav_phrase in NAV_PHRASES))):
                                sub_candidates.append((i, int(r[0])))
                        
                        if sub_candidates:
                            # 选择排名最小的作为子类
                            sub_idx = min(sub_candidates, key=lambda x: x[1])[0]
                            sub_rank, sub_category = bsr_list[sub_idx]
                            logging.info(f"[子类选择] 从{len(sub_candidates)}个候选项中选择: ({sub_rank},{sub_category})")
                    logging.info(f"[BSR分配] li_texts={li_texts} bsr_list={bsr_list} main=({main_rank},{main_category}) sub=({sub_rank},{sub_category}) 判别理由={reason}")
                    break
                # 否则按原有正则处理
                text = await td_content.inner_text()
                text = re.sub(r'\s+', ' ', text).strip()
                patterns = [
                    r'#([0-9]+(?:\s+[0-9]{3})*)\s+i\s+([^(#]+?)(?:\s*\(|$)',
                    r'#([0-9]+(?:\.[0-9]{3})*)\s+in\s+([^(#]+?)(?:\s*\(|$)',
                    r'(?:nº|n°|No\.)\s*([0-9]+(?:\.[0-9]{3})*)\s+en\s+([^(#]+?)(?:\s*\(|$)',
                    r'(?:n°|No\.)\s*([0-9]+(?:\.[0-9]{3})*)\s+dans\s+([^(#]+?)(?:\s*\(|$)',
                    r'(?:Nr\.|Rang\s+Nr\.)\s*([0-9]+(?:\.[0-9]{3})*)\s+(?:in|bei)\s+([^(#]+?)(?:\s*\(|$)',
                    r'n\.\s*([0-9]+(?:\.[0-9]{3})*)\s+in\s+([^(#]+?)(?:\s*\(|$)',
                    r'([0-9]+(?:,[0-9]{3})*)位([^0-9（\(]+?)(?:\s*（|\s*\(|$)',
                    r'(?:#|No\.)\s*([0-9]+(?:,[0-9]{3})*)\s+in\s+([^(#]+?)(?:\s*\(|$)',
                    r'(?:排第|排名第)([0-9,，]+)名\s*([^排名（]+?)(?:\s*（|$)'
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                        rank_str = matches[0][0].replace(',', '').replace('.', '').replace(' ', '').replace('，', '')
                        if rank_str.isdigit() and 1 <= int(rank_str) <= 10000000:
                            main_rank = rank_str
                            main_category = matches[0][1].strip()
                            if len(matches) > 1:
                                sub_rank_str = matches[1][0].replace(',', '').replace('.', '').replace(' ', '').replace('，', '')
                                if sub_rank_str.isdigit() and 1 <= int(sub_rank_str) <= 10000000:
                                    sub_rank = sub_rank_str
                                    sub_category = matches[1][1].strip()
                            logging.info(f"ASIN from URL {url} - BSR parsed via Strategy 1 (Table): main={main_rank}/{main_category}, sub={sub_rank}/{sub_category}")
                            break
                if main_rank:
                    break
    except Exception as e:
        logging.warning(f"Strategy 1 (Table) error: {e}")

    # === 新增：ul/li结构通用处理，兼容IE/EN/FR等 ===
    if not main_rank:
        try:
            # 直接查找所有ul/li结构（如IE/EN/FR等）
            ul_nodes = await page.query_selector_all('ul')
            for ul_node in ul_nodes:
                bsr_list = await extract_bsr_from_node(ul_node)
                if bsr_list:
                    # 修改：不再简单地取前两个元素作为主类和子类
                    # 而是应用与表格结构相同的主类/子类选择逻辑
                    NAV_PHRASES = [
                        'See Top 100 in ', 'Top 100 in ',
                        'Visualizza i Top 100 nella categoria', 'Ver el Top 100 en',
                        'Visa Topp 100 i', 'Voir les 100 premiers en',
                        'Ver os 100 mais vendidos em', 'Top 100 auf',
                        'Top 100 in', 'Top 100 dans', 'Top 100 en',
                        'Top 100 dei', 'Top 100 der', 'Top 100 de',
                    ]
                    
                    # 获取每个BSR项对应的原始文本，用于后续判断
                    li_nodes = await ul_node.query_selector_all(':scope > li')
                    li_texts = []
                    for li in li_nodes:
                        text = await li.inner_text()
                        text = re.sub(r'\s+', ' ', text).strip()
                        li_texts.append(text)
                    
                    # 1. 优先li文本含'Top 100'或'See Top 100'为主类
                    main_idx = -1
                    for i, raw_text in enumerate(li_texts):
                        if i < len(bsr_list):  # 确保索引在范围内
                            cleaned = re.sub(r'\([^\)]*\)', '', raw_text)
                            if 'Top 100' in cleaned or 'See Top 100' in cleaned:
                                main_idx = i
                                break
                    
                    # 2. 无Top 100则排名最大为主类
                    if main_idx < 0 and bsr_list:
                        ranks = [(i, int(r[0])) for i, r in enumerate(bsr_list) if r[0].isdigit()]
                        if ranks:
                            main_idx = max(ranks, key=lambda x: x[1])[0]
                    
                    # 设置主类
                    if main_idx >= 0 and main_idx < len(bsr_list):
                        main_rank, main_category = bsr_list[main_idx]
                        
                        # 3. 子类为其余项中排名最小且与主类不同的项
                        sub_idx = -1
                        if main_idx >= 0 and len(bsr_list) > 1:
                            # 修正：子类的品类名必须和主类不同，且排除排名为100的可能干扰项
                            sub_candidates = []
                            for i, r in enumerate(bsr_list):
                                if (i != main_idx and r[0].isdigit() and r[1] and 
                                    r[1] != main_category and 
                                    (int(r[0]) != 100 or not any(nav_phrase in li_texts[i] if i < len(li_texts) else False for nav_phrase in NAV_PHRASES))):
                                    sub_candidates.append((i, int(r[0])))
                            
                            if sub_candidates:
                                # 选择排名最小的作为子类
                                sub_idx = min(sub_candidates, key=lambda x: x[1])[0]
                                sub_rank, sub_category = bsr_list[sub_idx]
                                logging.info(f"[子类选择] 从{len(sub_candidates)}个候选项中选择: ({sub_rank},{sub_category})")
                    else:
                        # 如果没有找到合适的主类但有BSR数据，使用第一个作为主类
                        if bsr_list:
                            main_rank, main_category = bsr_list[0]
                            # 如果有多个项，使用第二个作为子类（如果与主类不同）
                            if len(bsr_list) > 1 and bsr_list[1][1] != main_category:
                                sub_rank, sub_category = bsr_list[1]
                    
                    logging.info(f"ASIN from URL {url} - BSR parsed via UL structure: {bsr_list}, 主类=({main_rank},{main_category}), 子类=({sub_rank},{sub_category})")
                    break
        except Exception as e:
            logging.warning(f"UL structure BSR extraction error: {e}")

    # 新增策略：直接查找包含"Best Sellers Rank"的span元素
    if not main_rank:
        try:
            # 查找包含"Best Sellers Rank"的span元素
            bsr_spans = await page.query_selector_all('span:has-text("Best Sellers Rank"), span:has-text("Amazon Best Sellers Rank")')
            for span in bsr_spans:
                bsr_list = await extract_bsr_from_node(span)
                if bsr_list and len(bsr_list) >= 1:
                    # 第一个通常是父类，后面的是子类
                    main_rank, main_category = bsr_list[0]
                    
                    # 如果有多个BSR项，找出子类（排名最小且与主类不同的项）
                    if len(bsr_list) > 1:
                        sub_candidates = [(i, int(r[0])) for i, r in enumerate(bsr_list) if i > 0 and r[0].isdigit() and r[1] != main_category]
                        if sub_candidates:
                            sub_idx = min(sub_candidates, key=lambda x: x[1])[0]
                            sub_rank, sub_category = bsr_list[sub_idx]
                    
                    logging.info(f"ASIN from URL {url} - BSR parsed via span structure: {bsr_list}, 主类=({main_rank},{main_category}), 子类=({sub_rank},{sub_category})")
                    break
        except Exception as e:
            logging.warning(f"Span BSR extraction error: {e}")

    # 策略2: 日语格式专门处理 (X位Category)
    if not main_rank:
        try:
            jp_elements = await page.query_selector_all('xpath=//tr[contains(.,"売れ筋ランキング") or contains(.,"位")] | //li[contains(.,"売れ筋ランキング") or contains(.,"位")] | //td[contains(.,"売れ筋ランキング") or contains(.,"位")] | //div[contains(.,"売れ筋ランキング") or contains(.,"位")]')
            for element in jp_elements:
                text = await element.inner_text()
                text = re.sub(r'\s+', ' ', text).strip()
                
                # 日语格式: 4,312位ペット用品 或 15位水槽用エアポンプアクセサリ
                jp_matches = re.findall(r'([0-9,，]+)位([^0-9（\(]+?)(?:\s*（|\s*\(|$)', text)
                if jp_matches and not main_rank:
                    main_rank = jp_matches[0][0].replace(',', '').replace('，', '')
                    main_category = jp_matches[0][1].strip()
                    if len(jp_matches) > 1:
                        sub_rank = jp_matches[1][0].replace(',', '').replace('，', '')
                        sub_category = jp_matches[1][1].strip()
                    logging.info(f"ASIN from URL {url} - BSR parsed via Strategy 2 (JP): main={main_rank}/{main_category}, sub={sub_rank}/{sub_category}")
                    break
        except Exception as e:
            logging.warning(f"Strategy 2 (JP) error: {e}")

    # 策略3: 意大利语格式专门处理 (n. X in Category)
    if not main_rank:
        try:
            it_elements = await page.query_selector_all('xpath=//tr[contains(.,"Posizione") or contains(.,"classifica") or contains(.,"Bestseller")] | //li[contains(.,"Posizione") or contains(.,"classifica") or contains(.,"Bestseller")] | //td[contains(.,"Posizione") or contains(.,"classifica") or contains(.,"Bestseller")] | //div[contains(.,"Posizione") or contains(.,"classifica") or contains(.,"Bestseller")]')
            for element in it_elements:
                text = await element.inner_text()
                text = re.sub(r'\s+', ' ', text).strip()
                
                # 意大利语格式: n. 2.477 in Cancelleria 或 n. 53 in Porta badge
                it_matches = re.findall(r'n\.\s*([0-9.,]+)\s+in\s+([^(0-9]+?)(?:\s*\(|$)', text, re.IGNORECASE)
                if it_matches and not main_rank:
                    main_rank = it_matches[0][0].replace('.', '').replace(',', '')
                    main_category = it_matches[0][1].strip()
                    if len(it_matches) > 1:
                        sub_rank = it_matches[1][0].replace('.', '').replace(',', '')
                        sub_category = it_matches[1][1].strip()
                    logging.info(f"ASIN from URL {url} - BSR parsed via Strategy 3 (IT): main={main_rank}/{main_category}, sub={sub_rank}/{sub_category}")
                    break
        except Exception as e:
            logging.warning(f"Strategy 3 (IT) error: {e}")

    # 策略4: 瑞典格式专门处理 (#X X i Category)  
    if not main_rank:
        try:
            se_elements = await page.query_selector_all('xpath=//tr[contains(.,"Rangordning") or contains(.,"bästsäljare")] | //li[contains(.,"Rangordning") or contains(.,"bästsäljare")] | //td[contains(.,"Rangordning") or contains(.,"bästsäljare")] | //div[contains(.,"Rangordning") or contains(.,"bästsäljare")]')
            for element in se_elements:
                text = await element.inner_text()
                text = re.sub(r'\s+', ' ', text).strip()
                
                # 瑞典格式: #49 449 i Bygg (空格分隔千位)
                se_matches = re.findall(r'#([0-9]+(?:\s+[0-9]{3})*)\s+i\s+([^(#]+?)(?:\s*\(|$)', text, re.IGNORECASE)
                if se_matches and not main_rank:
                    main_rank = se_matches[0][0].replace(' ', '')
                    main_category = se_matches[0][1].strip()
                    if len(se_matches) > 1:
                        sub_rank = se_matches[1][0].replace(' ', '')
                        sub_category = se_matches[1][1].strip()
                    logging.info(f"ASIN from URL {url} - BSR parsed via Strategy 4 (SE): main={main_rank}/{main_category}, sub={sub_rank}/{sub_category}")
                    break
        except Exception as e:
            logging.warning(f"Strategy 4 (SE) error: {e}")

    # 策略5: 荷兰格式专门处理 (#X.XXX in Category)
    if not main_rank:
        try:
            nl_elements = await page.query_selector_all('xpath=//tr[contains(.,"Plaats") or contains(.,"bestsellerlijst")] | //li[contains(.,"Plaats") or contains(.,"bestsellerlijst")] | //td[contains(.,"Plaats") or contains(.,"bestsellerlijst")] | //div[contains(.,"Plaats") or contains(.,"bestsellerlijst")]')
            for element in nl_elements:
                text = await element.inner_text()
                text = re.sub(r'\s+', ' ', text).strip()
                
                # 荷兰格式: #4.328 in Kantoorproducten (句号分隔千位)
                nl_matches = re.findall(r'#([0-9]+(?:\.[0-9]{3})*)\s+in\s+([^(#]+?)(?:\s*\(|$)', text, re.IGNORECASE)
                if nl_matches and not main_rank:
                    main_rank = nl_matches[0][0].replace('.', '')
                    main_category = nl_matches[0][1].strip()
                    if len(nl_matches) > 1:
                        sub_rank = nl_matches[1][0].replace('.', '')
                        sub_category = nl_matches[1][1].strip()
                    logging.info(f"ASIN from URL {url} - BSR parsed via Strategy 5 (NL): main={main_rank}/{main_category}, sub={sub_rank}/{sub_category}")
                    break
        except Exception as e:
            logging.warning(f"Strategy 5 (NL) error: {e}")

    # 策略6: 西班牙格式专门处理 (nºX.XXX en Category)
    if not main_rank:
        try:
            es_elements = await page.query_selector_all('xpath=//tr[contains(.,"Clasificación") or contains(.,"vendidos") or contains(.,"Bestseller")] | //li[contains(.,"Clasificación") or contains(.,"vendidos") or contains(.,"Bestseller")] | //td[contains(.,"Clasificación") or contains(.,"vendidos") or contains(.,"Bestseller")] | //div[contains(.,"Clasificación") or contains(.,"vendidos") or contains(.,"Bestseller")]')
            for element in es_elements:
                text = await element.inner_text()
                text = re.sub(r'\s+', ' ', text).strip()
                
                # 西班牙格式: nº20.691 en Productos (句号分隔千位)
                es_matches = re.findall(r'(?:nº|n°|No\.)\s*([0-9]+(?:\.[0-9]{3})*)\s+en\s+([^(#]+?)(?:\s*\(|$)', text, re.IGNORECASE)
                if es_matches and not main_rank:
                    main_rank = es_matches[0][0].replace('.', '')
                    main_category = es_matches[0][1].strip()
                    if len(es_matches) > 1:
                        sub_rank = es_matches[1][0].replace('.', '')
                        sub_category = es_matches[1][1].strip()
                    logging.info(f"ASIN from URL {url} - BSR parsed via Strategy 6 (ES): main={main_rank}/{main_category}, sub={sub_rank}/{sub_category}")
                    break
        except Exception as e:
            logging.warning(f"Strategy 6 (ES) error: {e}")

    # 策略7: 法语格式专门处理 (n° X.XXX dans Category)
    if not main_rank:
        try:
            fr_elements = await page.query_selector_all('xpath=//tr[contains(.,"Classement") or contains(.,"meilleures ventes")] | //li[contains(.,"Classement") or contains(.,"meilleures ventes")] | //td[contains(.,"Classement") or contains(.,"meilleures ventes")] | //div[contains(.,"Classement") or contains(.,"meilleures ventes")]')
            for element in fr_elements:
                text = await element.inner_text()
                text = re.sub(r'\s+', ' ', text).strip()
                
                # 法语格式: n° 1.234 dans Catégorie (句号分隔千位)
                fr_matches = re.findall(r'(?:n°|No\.)\s*([0-9]+(?:\.[0-9]{3})*)\s+dans\s+([^(#]+?)(?:\s*\(|$)', text, re.IGNORECASE)
                if fr_matches and not main_rank:
                    main_rank = fr_matches[0][0].replace('.', '')
                    main_category = fr_matches[0][1].strip()
                    if len(fr_matches) > 1:
                        sub_rank = fr_matches[1][0].replace('.', '')
                        sub_category = fr_matches[1][1].strip()
                    logging.info(f"ASIN from URL {url} - BSR parsed via Strategy 7 (FR): main={main_rank}/{main_category}, sub={sub_rank}/{sub_category}")
                    break
        except Exception as e:
            logging.warning(f"Strategy 7 (FR) error: {e}")

    # 策略8: 德语格式专门处理 (Nr. X.XXX in Category)
    if not main_rank:
        try:
            de_elements = await page.query_selector_all('xpath=//tr[contains(.,"Bestseller-Rang") or contains(.,"Verkaufsrang")] | //li[contains(.,"Bestseller-Rang") or contains(.,"Verkaufsrang")] | //td[contains(.,"Bestseller-Rang") or contains(.,"Verkaufsrang")] | //div[contains(.,"Bestseller-Rang") or contains(.,"Verkaufsrang")]')
            for element in de_elements:
                text = await element.inner_text()
                text = re.sub(r'\s+', ' ', text).strip()
                
                # 德语格式: Nr. 1.234 in Kategorie (句号分隔千位)
                de_matches = re.findall(r'(?:Nr\.|Rang\s+Nr\.)\s*([0-9]+(?:\.[0-9]{3})*)\s+(?:in|bei)\s+([^(#]+?)(?:\s*\(|$)', text, re.IGNORECASE)
                if de_matches and not main_rank:
                    main_rank = de_matches[0][0].replace('.', '')
                    main_category = de_matches[0][1].strip()
                    if len(de_matches) > 1:
                        sub_rank = de_matches[1][0].replace('.', '')
                        sub_category = de_matches[1][1].strip()
                    logging.info(f"ASIN from URL {url} - BSR parsed via Strategy 8 (DE): main={main_rank}/{main_category}, sub={sub_rank}/{sub_category}")
                    break
        except Exception as e:
            logging.warning(f"Strategy 8 (DE) error: {e}")

    # 策略9: 标准英文格式 (#1,234 in Category)
    if not main_rank:
        try:
            en_elements = await page.query_selector_all('xpath=//tr[contains(.,"Best Sellers Rank") or contains(.,"Amazon Best Sellers Rank") or contains(.,"Sales Rank")] | //li[contains(.,"Best Sellers Rank") or contains(.,"Amazon Best Sellers Rank") or contains(.,"Sales Rank")] | //td[contains(.,"Best Sellers Rank") or contains(.,"Amazon Best Sellers Rank") or contains(.,"Sales Rank")] | //div[contains(.,"Best Sellers Rank") or contains(.,"Amazon Best Sellers Rank") or contains(.,"Sales Rank")]')
            for element in en_elements:
                text = await element.inner_text()
                text = re.sub(r'\s+', ' ', text).strip()
                
                # 英文格式: #1,234 in Category or No. 1,234 in Category
                en_matches = re.findall(r'(?:#|No\.)\s*([0-9,，]+)\s+in\s+([^(#]+?)(?:\s*\(|$)', text, re.IGNORECASE)
                if en_matches and not main_rank:
                    main_rank = en_matches[0][0].replace(',', '').replace('，', '')
                    main_category = en_matches[0][1].strip()
                    if len(en_matches) > 1:
                        sub_rank = en_matches[1][0].replace(',', '').replace('，', '')
                        sub_category = en_matches[1][1].strip()
                    logging.info(f"ASIN from URL {url} - BSR parsed via Strategy 9 (EN): main={main_rank}/{main_category}, sub={sub_rank}/{sub_category}")
                    break
        except Exception as e:
            logging.warning(f"Strategy 9 (EN) error: {e}")

    # 策略10: 通用兜底 - 基于用户提供的CSS选择器
    if not main_rank:
        try:
            # 使用用户提供的具体CSS选择器
            specific_selectors = [
                '#productDetailsVoyagerAccordion_feature_div table tbody tr',
                '.prodDetSectionEntry',
                '[class*="prodDet"]'
            ]
            
            for selector in specific_selectors:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    text = await element.inner_text()
                    if 'Best Sellers' in text or 'Rangordning' in text or 'Plaats' in text or 'Clasificación' in text:
                        text = re.sub(r'\s+', ' ', text).strip()
                        
                        # 超宽松的通用模式
                        universal_patterns = [
                            r'#([0-9]+(?:\s+[0-9]{3})*)\s+i\s+([^(#]+?)(?:\s*\(|$)',  # 瑞典空格格式
                            r'#([0-9]+(?:\.[0-9]{3})*)\s+in\s+([^(#]+?)(?:\s*\(|$)',  # 荷兰句号格式
                            r'(?:nº|n°|Nr\.|No\.)\s*([0-9]+(?:[\s.,][0-9]{3})*)\s+(?:en|in|dans|bei)\s+([^(#]+?)(?:\s*\(|$)',  # 多语言通用
                            r'([0-9,，.]+)位([^0-9（\(]+?)(?:\s*（|\s*\(|$)',  # 日语位格式
                            r'(?:^|[^0-9])([0-9]{1,7}(?:[,.，\s][0-9]{3})*)\s*(?:in|dans|en|bei|位|名|i)\s+([^0-9(]{3,}?)(?:\s*\(|[0-9]|$)'  # 最宽松匹配
                        ]
                        
                        for pattern in universal_patterns:
                            matches = re.findall(pattern, text, re.IGNORECASE)
                            if matches:
                                rank_str = matches[0][0].replace(',', '').replace('.', '').replace(' ', '').replace('，', '')
                                if rank_str.isdigit() and 1 <= int(rank_str) <= 10000000:
                                    main_rank = rank_str
                                    main_category = matches[0][1].strip()
                                    logging.info(f"ASIN from URL {url} - BSR parsed via Strategy 10 (Universal CSS): main={main_rank}/{main_category}")
                                    break
                        
                        if main_rank:
                            break
                
                if main_rank:
                    break
        except Exception as e:
            logging.warning(f"Strategy 10 error: {e}")

    # 重试机制：如果仍未找到BSR，等待2秒后重试最宽松匹配
    if not main_rank:
        await page.wait_for_timeout(2000)
        try:
            logging.info(f"Retrying BSR extraction for {url}")
            retry_elements = await page.query_selector_all('xpath=//*[contains(text(),"#") or contains(text(),"No.") or contains(text(),"Nr.") or contains(text(),"n°") or contains(text(),"nº") or contains(text(),"位") or contains(text(),"排")]')
            
            for element in retry_elements:
                text = await element.inner_text()
                text = re.sub(r'\s+', ' ', text).strip()
                
                # 最终兜底的极宽松匹配
                final_patterns = [
                    r'#([0-9]+(?:\s+[0-9]{3})*)\s+i\s+([^0-9(]{2,}?)(?:\s*\(|$)',  # 瑞典空格
                    r'#([0-9]+(?:\.[0-9]{3})*)\s+in\s+([^0-9(]{3,}?)(?:\s*\(|$)',  # 荷兰句号
                    r'(?:nº|n°|Nr\.|No\.)\s*([0-9]+(?:[\s.,][0-9]{3})*)\s+(?:en|in|dans|bei)\s+([^0-9(]{3,}?)(?:\s*\(|$)',  # 欧洲通用
                    r'([0-9,，.]+)位([^0-9（\(]{2,}?)(?:\s*（|\s*\(|$)',  # 日语位
                    r'(?:^|[^0-9])([0-9]{1,7}(?:[,.\s][0-9]{3})*)\s*(?:in|i|en|dans|bei|位|名)\s+([^0-9(]{2,}?)(?:\s*\(|[0-9]|$)'  # 超宽松
                ]
                
                for pattern in final_patterns:
                    loose_matches = re.findall(pattern, text, re.IGNORECASE)
                    if loose_matches:
                        rank_str = loose_matches[0][0].replace(',', '').replace('，', '').replace('.', '').replace(' ', '')
                        if rank_str.isdigit() and 1 <= int(rank_str) <= 10000000:
                            main_rank = rank_str
                            main_category = loose_matches[0][1].strip()
                            logging.info(f"ASIN from URL {url} - BSR parsed via Retry (Final): main={main_rank}/{main_category}")
                            break
                
                if main_rank:
                    break
        except Exception as e:
            logging.warning(f"Retry BSR extraction error: {e}")

    # === 多语言评分解析策略 ===
    # 评分和评论数解析
    rating = ''
    review_count = ''

    # 多重兜底评分解析策略
    rating_strategies = [
        # 策略1: 标准popover
        {
            'name': 'standard_popover',
            'selector': '#acrPopover',
            'attr': 'title',
            'pattern': r'(\d+(?:[.,]\d+)?) (?:out of 5|颗星)'
        },
        # 策略2: 小尺寸评分 (用户提供的新格式)
        {
            'name': 'small_rating',
            'selector': '.a-size-small.a-color-base',
            'attr': 'text',
            'pattern': r'^(\d+(?:[.,]\d+)?)$'
        },
        # 策略3: popover链接中的评分
        {
            'name': 'popover_link',
            'selector': '.a-popover-trigger .a-size-base.a-color-base, .a-popover-trigger .a-size-small.a-color-base',
            'attr': 'text',
            'pattern': r'^(\d+(?:[.,]\d+)?)$'
        },
        # 策略4: 图标alt文本
        {
            'name': 'icon_alt',
            'selector': 'i.a-icon-alt, i[class*="a-star"] span.a-icon-alt',
            'attr': 'text',
            'pattern': r'(\d+(?:[.,]\d+)?) (?:out of 5|颗星)'
        },
        # 策略5: Customer Reviews表格
        {
            'name': 'customer_reviews_table',
            'selector': 'xpath=//th[contains(normalize-space(text()),"Customer Reviews") or contains(normalize-space(text()),"推荐度")]/../td',
            'attr': 'text',
            'pattern': r'(\d+(?:[.,]\d+)?) (?:out of 5|颗星)'
        },
        # 策略6: averageCustomerReviews区域
        {
            'name': 'avg_customer_reviews',
            'selector': '#averageCustomerReviews',
            'attr': 'text',
            'pattern': r'(\d+(?:[.,]\d+)?) (?:out of 5|颗星)'
        }
    ]

    for strategy in rating_strategies:
        if rating:  # 如果已获取到评分，跳出
            break
            
        try:
            if strategy['selector'].startswith('xpath='):
                elem = await page.query_selector(strategy['selector'])
            else:
                elem = await page.query_selector(strategy['selector'])
            
            if elem:
                if strategy['attr'] == 'title':
                    text = await elem.get_attribute('title') or ''
                else:
                    text = await elem.inner_text()
                
                text = text.replace(',', '.').strip()
                match = re.search(strategy['pattern'], text)
                if match:
                    rating = match.group(1)
                    logging.info(f"ASIN from URL {url} - Rating parsed via {strategy['name']}: {rating}")
                    break

        except Exception as e:
            logging.debug(f"ASIN from URL {url} - Rating strategy {strategy['name']} failed: {e}")
            continue

    # 多重兜底评论数解析策略
    review_strategies = [
        # 策略1: 标准评论链接
        {
            'name': 'standard_review_link',
            'selector': '#acrCustomerReviewText',
            'pattern': r'([\d,]+)'
        },
        # 策略2: 评论链接变体
        {
            'name': 'review_link_variant',
            'selector': 'a[href*="reviews"] span, span[id*="review"]',
            'pattern': r'([\d,]+)\s*(?:评论|reviews?|ratings?)'
        },
        # 策略3: Customer Reviews表格中的评论数
        {
            'name': 'table_review_count',
            'selector': 'xpath=//th[contains(normalize-space(text()),"Customer Reviews") or contains(normalize-space(text()),"推荐度")]/../td',
            'pattern': r'([\d,]+)\s*(?:评论|reviews?|ratings?)'
        }
    ]

    for strategy in review_strategies:
        if review_count:  # 如果已获取到评论数，跳出
            break
            
        try:
            if strategy['selector'].startswith('xpath='):
                elem = await page.query_selector(strategy['selector'])
            else:
                elem = await page.query_selector(strategy['selector'])
            
            if elem:
                text = await elem.inner_text()
                match = re.search(strategy['pattern'], text, re.IGNORECASE)
                if match:
                    review_count = match.group(1).replace(',', '')
                    logging.info(f"ASIN from URL {url} - Review count parsed via {strategy['name']}: {review_count}")
                    break
                    
        except Exception as e:
            logging.debug(f"ASIN from URL {url} - Review count strategy {strategy['name']} failed: {e}")
            continue

    if rating:
        rating = rating.strip()
    if review_count:
        m = re.search(r'(\d+)', review_count) # 确保只有数字
        if m:
            review_count = m.group(1)
        else:
            review_count = '' # 如果正则匹配失败，说明格式不对，清空
    
    # 最终日志记录
    logging.info(f"ASIN from URL {url} - Final parsed data: BSR={main_rank}/{main_category}, Sub={sub_rank}/{sub_category}, Rating={rating}, Reviews={review_count}")

    # --- 新增：主子类排名对比，数字大的为父类 ---
    # 只要主类和子类排名都存在且为数字
    try:
        main_rank_num = int(str(main_rank).replace(',', '')) if main_rank and str(main_rank).isdigit() else None
        sub_rank_num = int(str(sub_rank).replace(',', '')) if sub_rank and str(sub_rank).isdigit() else None
        if main_rank_num is not None and sub_rank_num is not None:
            if main_rank_num < sub_rank_num:
                # 交换，数字大的为父类
                main_category, sub_category = sub_category, main_category
                main_rank, sub_rank = sub_rank, main_rank
    except Exception as e:
        logging.warning(f"BSR父子类排名对比异常: {e}")

    # 返回前进行父子类顺序检查和修正
    if main_rank and sub_rank and main_rank.isdigit() and sub_rank.isdigit():
        main_rank_int = int(main_rank)
        sub_rank_int = int(sub_rank)
        
        # 如果子类排名大于主类排名，则交换它们
        if sub_rank_int > main_rank_int:
            logging.info(f"[BSR排序修正] 交换父子类顺序: 原父类=({main_rank},{main_category}), 原子类=({sub_rank},{sub_category})")
            main_rank, sub_rank = sub_rank, main_rank
            main_category, sub_category = sub_category, main_category
            logging.info(f"[BSR排序修正] 修正后: 父类=({main_rank},{main_category}), 子类=({sub_rank},{sub_category})")
    
    # 如果主类为空但子类不为空，则将子类提升为主类
    if (not main_rank or not main_category) and sub_rank and sub_category:
        logging.info(f"[BSR修正] 主类为空，将子类提升为主类: 子类=({sub_rank},{sub_category})")
        main_rank, main_category = sub_rank, sub_category
        sub_rank, sub_category = '', ''
    
    return main_category, main_rank, sub_category, sub_rank, rating, review_count

async def fetch_vine_count(page, asin, domain):
    """
    获取所有评论数，计算最后三页并统计 Vine 评论数量
    """
    # 第一步：获取所有评论总数，用于计算页数
    page_size = 10
    first_url = f'https://www.{domain}/product-reviews/{asin}?sortBy=recent&reviewerType=all_reviews&formatType=all_formats&pageNumber=1&pageSize={page_size}'
    logging.info(f"ASIN={asin} - Fetching initial page to get total review count: {first_url}")
    await page.goto(first_url, timeout=60000, wait_until='domcontentloaded')
    await page.wait_for_timeout(2000)
    # 获取评论总数文本
    try:
        await page.wait_for_selector('#filter-info-section > div', timeout=15000)
        count_text = (await page.inner_text('#filter-info-section > div')).strip()
    except:
        await page.wait_for_selector('div[data-hook="cr-filter-info-review-rating-count"]', timeout=5000)
        count_text = (await page.inner_text('div[data-hook="cr-filter-info-review-rating-count"]')).strip()
    m = re.search(r'([\d,]+)', count_text)
    total_reviews = int(m.group(1).replace(',', '')) if m else 0
    # 计算总页数及起始页（最后三页）
    total_pages = max((total_reviews + page_size - 1) // page_size, 1)
    # 限制最多抓取 10 页评论
    total_pages = min(total_pages, 10)

    # 初始化 Vine 评论计数和最后一页前三条评论平均评分
    vine_count = 0
    latest3_rating = 0.0  # 存储最后一页前三条评论的平均评分
    previous_first_review_text = "INITIAL_DUMMY_VALUE_FOR_FIRST_PAGE_COMPARISON_DO_NOT_MATCH_REAL_TEXT" # Initialize

    for page_num in range(1, total_pages + 1):
        current_page_first_review_text_for_next_iteration = "" # Reset for current page

        if page_num > 1: # Navigation and content change check needed for pages 2 onwards
            try:
                await page.click('li.a-last a')
                url_pattern = re.compile(rf"(\?|&)pageNumber={page_num}(&|$)")
                await page.wait_for_url(url_pattern, timeout=30000)
                await page.wait_for_load_state('domcontentloaded', timeout=30000)
                await page.wait_for_timeout(1000) # Fixed delay

                # Explicitly wait for page content to change from the previous page
                content_changed = False
                for attempt in range(10): # Max 5 seconds wait for content to change
                    new_page_first_el = await page.query_selector('span[data-hook="review-body"]:first-of-type')
                    if new_page_first_el:
                        new_page_first_text = await new_page_first_el.inner_text()
                        if new_page_first_text.strip() and new_page_first_text != previous_first_review_text:
                            logging.debug(f"ASIN={asin} - Content for page {page_num} confirmed different from previous page.")
                            current_page_first_review_text_for_next_iteration = new_page_first_text
                            content_changed = True
                            break 
                    await page.wait_for_timeout(500) # Wait 0.5s before retrying
                
                if not content_changed:
                    logging.warning(f"ASIN={asin} - Timed out waiting for content of page {page_num} to differ from previous. Counts may be inaccurate. Previous: '{previous_first_review_text[:50]}...'")

            except Exception as e_nav:
                logging.error(f"ASIN={asin} - Error navigating to page {page_num} (click, URL, load state, fixed delay, or content change wait): {e_nav}")
                break # Break from ASIN's page loop
        
        # Common section for all pages (page 1 and subsequent pages)
        try:
            await page.wait_for_selector('span[data-hook="review-body"]', timeout=6000)
            
            # For page 1, or if content change check above didn't capture text (e.g. first page after nav error, or genuinely no change)
            if not current_page_first_review_text_for_next_iteration:
                first_el_for_capture = await page.query_selector('span[data-hook="review-body"]:first-of-type')
                if first_el_for_capture:
                    current_page_first_review_text_for_next_iteration = await first_el_for_capture.inner_text()

        except Exception as e_content:
            logging.warning(f"ASIN={asin} - Review-body not found on page {page_num} (URL: {page.url}): {e_content}")
            # Continue to JS eval, which might return 0 if content is missing

        # Execute JS to count Vine reviews on the current page
        try:
            count = await page.evaluate('''() => {
                const els = document.querySelectorAll('span.a-color-success.a-text-bold');
                let cnt = 0;
                els.forEach(el => {
                    if (el.textContent.trim() === 'Amazon Vine Customer Review of Free Product') cnt++;
                });
                return cnt;
            }''')
            vine_count += count
            logging.info(f"ASIN={asin} - Page {page_num} Vine reviews via click-nav JS: {count}")
        except Exception as e_js:
            logging.error(f"ASIN={asin} - Error on page {page_num} JS eval: {e_js}")

        # 如果是最后一页，则计算前三条评论的平均评分
        if page_num == total_pages:
            try:
                avg_rating = await page.evaluate(r'''() => {
                    const stars = Array.from(document.querySelectorAll('i[data-hook="review-star-rating"] span.a-icon-alt'));
                    const nums = stars.slice(0, 3).map(el => {
                        const m = el.textContent.match(/(\d+(?:\.\d+)?)/);
                        return m ? parseFloat(m[1]) : 0;
                    });
                    const valid = nums.filter(r => !isNaN(r));
                    const sum = valid.reduce((a, b) => a + b, 0);
                    return valid.length ? sum / valid.length : 0;
                }''')
                latest3_rating = round(avg_rating, 1)
                logging.info(f"ASIN={asin} - Page {page_num} latest3 average rating: {latest3_rating}")
            except Exception as e_rt:
                logging.error(f"ASIN={asin} - Error calculating latest3_rating on page {page_num}: {e_rt}")

        # Update previous_first_review_text for the next iteration
        previous_first_review_text = current_page_first_review_text_for_next_iteration if current_page_first_review_text_for_next_iteration else previous_first_review_text

    logging.info(f"ASIN={asin} - Total Vine reviews found: {vine_count}, latest3_rating: {latest3_rating}")
    return vine_count, latest3_rating

# 新增：Vine 评论独立抓取函数，使用登录会话
async def run_vine_scraper(df, results_list_ref, concurrency, user_data_dir):
    """
    使用系统 Chrome，可复用登录态并固定 UA、视口、语言和时区
    """
    # 使用默认英文环境，依赖精确的多语言BSR解析
    target_locale = 'en-US'
    target_timezone = 'America/New_York'
    accept_language = 'en-US,en;q=0.9'
    
    logging.info(f"[Vine] 使用标准环境: {target_locale}, 时区: {target_timezone}")
    
    async with async_playwright() as pw:
        # 使用系统 Chrome，可复用登录态并固定 UA、视口、语言和时区
        context = await pw.chromium.launch_persistent_context(
            user_data_dir,
            headless=False,
            executable_path=r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            user_agent=DEFAULT_USER_AGENT,
            locale=target_locale,
            timezone_id=target_timezone,
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()
        await page.set_extra_http_headers({
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept-Language": accept_language
        })
        # 初始化 Vine 评论抓取的进度条
        pbar = tqdm(total=len(df), desc='Vine Scraping')
        await page.set_extra_http_headers({
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept-Language": accept_language
        })
        # 登录状态已持久化，直接开始抓取 Vine 评论
        for idx, row in enumerate(df.itertuples(index=False)):
            asin = getattr(row, 'ASIN', '')
            country = getattr(row, 'country', '').upper()
            domain = DOMAIN_MAP.get(country, 'amazon.com')
            # 获取 Vine 评论总数和最后一页前三条评论平均评分，并对单条异常进行捕获，避免批次中断
            try:
                vine_count, latest3_rating = await fetch_vine_count(page, asin, domain)
                status = '成功'
            except Exception as e_vine_one:
                # 捕获单条异常，输出日志但不中断后续流程
                logging.error(f"ASIN={asin} - Vine scraping error: {e_vine_one}")
                vine_count, latest3_rating = 0, 0.0
                status = '失败'

            # 确保 results_list_ref 对应位置存在可写字典
            if idx >= len(results_list_ref) or results_list_ref[idx] is None:
                results_list_ref[idx] = {
                    'ASIN': asin,
                    'country': country
                }

            results_list_ref[idx]['vine_count'] = vine_count
            results_list_ref[idx]['latest3_rating'] = latest3_rating

            # 控制台输出爬取情况，同时写入日志
            msg = f"[Vine] {asin} - {status}: count={vine_count}, rating={latest3_rating}"
            print(msg)
            if status == '成功':
                logging.info(msg)
            else:
                logging.error(msg)
            # 更新进度条
            pbar.update(1)
        # 完成后关闭进度条并关闭浏览器
        pbar.close()
        await context.close()

# 新增：登录流程，通过任意VINE链接触发登录
async def login_flow(profile_dir, login_url):
    """
    使用系统 Chrome，可复用登录态并固定 UA、视口、语言和时区
    """
    # 使用默认英文环境，依赖精确的多语言BSR解析
    target_locale = 'en-US'
    target_timezone = 'America/New_York'
    accept_language = 'en-US,en;q=0.9'
    
    logging.info(f"[Login] 使用标准环境: {target_locale}, 时区: {target_timezone}")
    
    async with async_playwright() as pw:
        # 使用系统 Chrome，可复用登录态并固定 UA、视口、语言和时区
        context = await pw.chromium.launch_persistent_context(
            profile_dir,
            headless=False,
            executable_path=r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            user_agent=DEFAULT_USER_AGENT,
            locale=target_locale,
            timezone_id=target_timezone,
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()
        await page.set_extra_http_headers({
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept-Language": accept_language
        })
        await page.goto(login_url)
        input("请在浏览器中完成登录后按回车继续...")
        # 会话预热：模拟人类行为，确保 Cookie 写入
        # 访问产品页面
        await page.goto(login_url.split('?')[0].replace('/product-reviews/', '/'), timeout=60000)
        await page.wait_for_timeout(2000)
        # 滚动页面
        await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1000)
        await page.evaluate("window.scrollBy(0, -document.body.scrollHeight)")
        await page.wait_for_timeout(1000)
        # 返回首页
        await page.goto(f"https://{login_url.split('/')[2]}", timeout=60000)
        await page.wait_for_timeout(2000)
        await context.close()

def ensure_login(profile_dir, df):
    login_file = os.path.join(profile_dir, 'login_state.json')
    os.makedirs(profile_dir, exist_ok=True)
    if os.path.exists(login_file):
        try:
            with open(login_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('logged_in'):
                return
        except:
            pass
    # 未登录，开始登录流程
    asin = str(df.iloc[0].get('ASIN', '')).strip()
    country = str(df.iloc[0].get('country', '')).strip().upper()
    domain = DOMAIN_MAP.get(country, 'amazon.com')
    login_url = f'https://www.{domain}/product-reviews/{asin}?sortBy=recent&reviewerType=all_reviews&formatType=all_formats&pageNumber=1&pageSize=10'
    print(f"未检测到登录状态，开始登录流程，登录 URL: {login_url}")
    asyncio.run(login_flow(profile_dir, login_url))
    with open(login_file, 'w', encoding='utf-8') as f:
        json.dump({'logged_in': True}, f, ensure_ascii=False)
    print("登录完成，已记录状态。")

async def run_scraper(df, results_list_ref, concurrency, profile_dir, progress_counter=None):
    """
    使用系统 Chrome，可复用登录态并固定 UA、视口、语言和时区
    """
    # 使用默认英文环境，依赖精确的多语言BSR解析
    target_locale = 'en-US'
    target_timezone = 'America/New_York'
    accept_language = 'en-US,en;q=0.9'
    
    logging.info(f"使用标准环境: {target_locale}, 时区: {target_timezone}")
    
    async with async_playwright() as pw:
        # 使用系统 Chrome，可复用登录态并固定 UA、视口、语言和时区
        context = await pw.chromium.launch_persistent_context(
            profile_dir,
            headless=False,
            executable_path=r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            user_agent=DEFAULT_USER_AGENT,
            locale=target_locale,
            timezone_id=target_timezone,
            viewport={'width': 1920, 'height': 1080}
        )
        
        pages = []
        for _ in range(concurrency):
            page = await context.new_page()
            await page.set_extra_http_headers({
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept-Language": accept_language
            })
            # 注释掉静态资源拦截，恢复完整资源加载
            # async def _intercept_route(route):
            #     if route.request.resource_type in ["image", "stylesheet", "font", "media"]:
            #         await route.abort()
            #     else:
            #         await route.continue_()
            # await page.route("**/*", _intercept_route)
            pages.append(page)
            
        page_queue = asyncio.Queue()
        results_list_ref.clear() # 使用传入的列表引用
        results_list_ref.extend([None] * len(df)) # 预分配，保持顺序

        for p in pages:
            await page_queue.put(p)
            
        # 子进程禁用局部进度条，由主进程统一展示
        pbar = tqdm(total=len(df), desc='Scraping', disable=(progress_counter is not None))

        async def process_row(pos, row_data): # Renamed row to row_data
            page = await page_queue.get()
            try:
                asin = str(row_data.get('ASIN', '')).strip()
                country = str(row_data.get('country', '')).strip().upper()
                domain = DOMAIN_MAP.get(country, 'amazon.com')
                url = f'https://www.{domain}/dp/{asin}'
                
                record = {
                    'ASIN': asin, 'country': country, 'url': url,
                    'bsr_main_category': '', 'bsr_main_rank': '',
                    'bsr_sub_category': '', 'bsr_sub_rank': '',
                    'vine_count': 0, 'rating': '', 'review_count': ''
                }
                logging.info(f"开始抓取 ASIN={asin}, Country={country}, URL={url}")

                max_retries = 3
                main_cat, main_rank, sub_cat, sub_rank, rating, reviews = '', '', '', '', '', ''

                # --- 强制Amazon页面尽量以英文显示：为URL拼接language=en_US参数 ---
                if '?' in url:
                    url_with_lang = url + '&language=en_US'
                else:
                    url_with_lang = url + '?language=en_US'
                # 传递拼接后的URL给fetch_product_data
                # 注：解析规则不变，兼容多语言
                for attempt in range(1, max_retries + 1):
                    logging.info(f"ASIN={asin} - Attempt {attempt}/{max_retries} for product data.")
                    try:
                        main_cat, main_rank, sub_cat, sub_rank, rating, reviews = await fetch_product_data(page, url_with_lang)
                        # 检查核心数据是否都获取到
                        if main_rank and rating and reviews: # 主要BSR排名，评分和评论数是核心
                            logging.info(f"ASIN={asin} - Attempt {attempt} successful for core product data.")
                            break
                        else:
                            logging.warning(f"ASIN={asin} - Attempt {attempt}: Missing core data (MainRank:{main_rank}, Rating:{rating}, Reviews:{reviews}). Retrying...")
                    except Exception as e:
                        logging.error(f"ASIN={asin} - Attempt {attempt} fetch_product_data error: {e}")
                    
                    if attempt < max_retries:
                        await asyncio.sleep(random.uniform(2.0, 4.0)) # 重试前增加延时
                    else:
                        logging.error(f"ASIN={asin} - Failed to fetch core product data after {max_retries} attempts.")
                
                record.update({
                    'bsr_main_category': main_cat, 'bsr_main_rank': main_rank,
                    'bsr_sub_category': sub_cat, 'bsr_sub_rank': sub_rank,
                    'rating': rating, 'review_count': reviews
                })

                # 日志最终获取到的数据
                logging.info(
                    f"ASIN={asin} - Final Data: "
                    f"MainBSR: {main_rank}/{main_cat}, SubBSR: {sub_rank}/{sub_cat}, "
                    f"Rating: {rating}, Reviews: {reviews}, Vine: {record['vine_count']}"
                )
                if not (main_rank and rating and reviews):
                     logging.warning(f"ASIN={asin} - Final check: Core product data might be incomplete.")

                results_list_ref[pos] = record
            except Exception as e_process:
                logging.critical(f"ASIN={asin} (pos={pos}) - Unhandled error in process_row: {e_process}")
                # 即使单行出错，也确保有个记录占位，避免后续DataFrame创建问题
                if results_list_ref[pos] is None: # 如果还没赋值
                    results_list_ref[pos] = {'ASIN': str(row_data.get('ASIN', '')).strip(), 'country': str(row_data.get('country', '')).strip().upper(), 'ERROR': str(e_process)}
            finally:
                await page_queue.put(page)
                pbar.update(1)
                # 更新跨进程共享计数器
                if progress_counter is not None:
                    try:
                        with progress_counter.get_lock():
                            progress_counter.value += 1
                    except AttributeError:
                        # Manager.Value 返回的 ValueProxy 无 get_lock()
                        progress_counter.value += 1

        tasks = [asyncio.create_task(process_row(pos, row_data)) for pos, (_, row_data) in enumerate(df.iterrows())]
        await asyncio.gather(*tasks)
        
        pbar.close()
        logging.info("All scraping tasks completed. Closing browser.")
        for p_item in pages:
            await p_item.close()
        await context.close()

def clean_category_name(name):
    """
    去除品类名称中的特殊字符，仅保留中英文、数字和空格。
    """
    if pd.isna(name):
        return name
    # 保留中文、英文、数字和空格
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9 ]', '', str(name))

def clean_and_adjust_categories(csv_path='output.csv'):
    """
    读取csv，清洗品类名称，调整父子类关系，覆盖写回。
    """
    df = pd.read_csv(csv_path, dtype=str)
    # 清洗品类名称
    df['bsr_main_category'] = df['bsr_main_category'].apply(clean_category_name)
    df['bsr_sub_category'] = df['bsr_sub_category'].apply(clean_category_name)

    # 父子类调整
    def adjust_row(row):
        # 检查主类/子类及排名是否缺失
        main_cat, sub_cat = row['bsr_main_category'], row['bsr_sub_category']
        main_rank, sub_rank = row['bsr_main_rank'], row['bsr_sub_rank']
        # 只要有一个排名缺失，跳过调整
        if (pd.isna(main_cat) or pd.isna(sub_cat) or
            pd.isna(main_rank) or pd.isna(sub_rank) or
            main_cat == '' or sub_cat == '' or
            main_rank == '' or sub_rank == ''):
            return row
        try:
            main_rank_num = int(str(main_rank).replace(',', ''))
            sub_rank_num = int(str(sub_rank).replace(',', ''))
        except Exception:
            return row
        # 如果主类排名大于子类排名，则交换
        if main_rank_num > sub_rank_num:
            row['bsr_main_category'], row['bsr_sub_category'] = sub_cat, main_cat
            row['bsr_main_rank'], row['bsr_sub_rank'] = sub_rank, main_rank
        return row

    df = df.apply(adjust_row, axis=1)
    # 覆盖写回
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

async def handle_continue_shopping(page):
    """
    检测并自动点击亚马逊多语言"继续购物"确认页面的按钮，规避反爬虫。
    """
    import random
    import logging
    try:
        # 1. 结构+属性联合定位
        buttons = await page.query_selector_all('button[type="submit"].a-button-text')
        for btn in buttons:
            visible = await btn.is_visible()
            enabled = await btn.is_enabled()
            # 检查父级div是否有提示信息或info图标
            parent = await btn.evaluate_handle('node => node.parentElement')
            parent_html = await parent.evaluate('node => node.outerHTML') if parent else ''
            # 只要按钮可见可用，且父级结构有info或提示框特征
            if visible and enabled and (('a-box-info' in parent_html) or ('a-icon-alert' in parent_html) or ('info' in parent_html)):
                await asyncio.sleep(random.uniform(0.8, 1.5))
                await btn.hover()
                await asyncio.sleep(random.uniform(0.2, 0.6))
                await btn.click()
                logging.info('[CONTINUE_SHOPPING] Clicked button via structure+parent check')
                await page.wait_for_timeout(1200)
                return True
        # 2. 兜底：查找所有可见的type=submit按钮
        all_submit_btns = await page.query_selector_all('button[type="submit"]')
        for btn in all_submit_btns:
            visible = await btn.is_visible()
            enabled = await btn.is_enabled()
            if visible and enabled:
                await asyncio.sleep(random.uniform(0.8, 1.5))
                await btn.hover()
                await asyncio.sleep(random.uniform(0.2, 0.6))
                await btn.click()
                logging.info('[CONTINUE_SHOPPING] Clicked fallback submit button')
                await page.wait_for_timeout(1200)
                return True
        # 3. 兜底：XPath查找所有可见的按钮
        xpath_btns = await page.query_selector_all('//button[@type="submit"]')
        for btn in xpath_btns:
            visible = await btn.is_visible()
            enabled = await btn.is_enabled()
            if visible and enabled:
                await asyncio.sleep(random.uniform(0.8, 1.5))
                await btn.hover()
                await asyncio.sleep(random.uniform(0.2, 0.6))
                await btn.click()
                logging.info('[CONTINUE_SHOPPING] Clicked fallback XPath button')
                await page.wait_for_timeout(1200)
                return True
        logging.info('[CONTINUE_SHOPPING] No continue button found')
        return False
    except Exception as e:
        logging.error(f'[CONTINUE_SHOPPING] Exception: {e}')
        return False

def _worker_process(df_slice, profile_dir, concurrency, shared_results, progress_counter):
    """子进程执行函数：直接运行爬虫并写结果（登录已在主进程完成）"""
    try:
        results_data: list = []
        asyncio.run(run_scraper(df_slice, results_data, concurrency, profile_dir, progress_counter))
        shared_results.extend([r for r in results_data if r is not None])
    except Exception as e:
        logging.error(f"[Worker-{profile_dir}] 发生异常: {e}")


def multi_process_scraper(df, profile_template: str, profile_count: int, concurrency: int, output_file: str):
    """多进程调度：根据模板前缀批量创建 profile 目录并切分任务"""
    manager = Manager()
    shared_results = manager.list()
    progress_counter = manager.Value('i', 0)

    total = len(df)
    slice_size = (total + profile_count - 1) // profile_count  # 向上取整

    processes = []
    for idx in range(profile_count):
        start_idx = idx * slice_size
        end_idx = min((idx + 1) * slice_size, total)
        if start_idx >= end_idx:
            continue  # 可能最后一个切片为空
        df_slice = df.iloc[start_idx:end_idx].copy()
        profile_dir = f"{profile_template}{idx}"
        # --- 主进程内先确保登录（可交互） ---
        ensure_login(profile_dir, df_slice)
        p = Process(target=_worker_process, args=(df_slice, profile_dir, concurrency, shared_results, progress_counter))
        p.start()
        processes.append(p)
        logging.info(f"[Master] 已启动子进程 {p.pid}，处理行 {start_idx}-{end_idx - 1}，使用 profile '{profile_dir}'")

    # 主进程统一显示总进度条
    total_tasks = len(df)
    pbar_total = tqdm(total=total_tasks, desc='Total Progress')
    while any(p.is_alive() for p in processes):
        try:
            with progress_counter.get_lock():
                completed = progress_counter.value
        except AttributeError:
            completed = progress_counter.value
        pbar_total.update(completed - pbar_total.n)
        time.sleep(0.5)
    # 子进程已全部完成，再做一次最终更新
    try:
        with progress_counter.get_lock():
            completed = progress_counter.value
    except AttributeError:
        completed = progress_counter.value
    pbar_total.update(completed - pbar_total.n)
    pbar_total.close()

    # 等待子进程结束并记录退出码
    for p in processes:
        p.join()
        logging.info(f"[Master] 子进程 {p.pid} 已结束，exitcode={p.exitcode}")

    # 汇总结果
    results = [r for r in shared_results if r]
    if not results:
        print("警告: 未收集到任何有效数据。请检查日志文件 'spider.log'")
        logging.warning("multi_process_scraper 未收集到任何有效数据。")
        return

    out_df = pd.DataFrame(results)
    abs_out_path = os.path.abspath(output_file)
    try:
        out_df.to_csv(abs_out_path, index=False, encoding='utf-8-sig')
        print(f'数据已保存到 {abs_out_path}')
        logging.info(f'[Master] 多进程模式: 数据已保存到 {abs_out_path}')
    except Exception as e:
        print(f"错误: 无法保存结果到 '{abs_out_path}': {e}")
        logging.error(f"无法保存结果到 '{abs_out_path}': {e}")
        return

    # 自动清洗品类
    clean_and_adjust_categories(output_file)

@click.command()
@click.option('--input', '-i', 'input_file', default='data/test_input.csv', help='输入CSV/Excel文件路径，包含ASIN和country列')
@click.option('--encoding', '-e', 'encoding', default='utf-8-sig', help='输入CSV文件编码 (例如 utf-8, utf-8-sig, gbk)')
@click.option('--sep', '-s', 'sep', default=',', help='输入CSV文件分隔符 (例如 ",", "\\t", ";")')
@click.option('--concurrency', '-c', 'concurrency', default=3, type=int, help='单进程内协程并发数 (建议≤5)')
@click.option('--profile-template', 'profile_template', default='my_browser_profile_', help='用户数据目录前缀模板 (启用多进程时使用)')
@click.option('--profile-count', 'profile_count', default=0, type=int, help='根据模板创建的目录数量 (>0 时启用多进程)')
@click.option('--profile-dir', '-p', 'profile_dir', default='my_browser_profile', help='单进程模式下的用户数据目录')
@click.option('--output', '-o', 'output_file', default='output.csv', help='输出CSV文件路径')
def main(input_file, output_file, encoding, sep, concurrency, profile_template, profile_count, profile_dir):
    """
    入口函数：读取输入CSV/Excel，运行爬虫并保存结果到输出CSV
    """
    try:
        if input_file.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(input_file)
        else:
            # 处理分隔符转义，特别是制表符
            actual_sep = sep.replace('\\t', '\t')
            df = pd.read_csv(input_file, encoding=encoding, sep=actual_sep, dtype=str) # 读取为字符串以保留ASIN格式
        logging.info(f"成功读取输入文件: {input_file}, 共 {len(df)} 条记录.")
    except Exception as e:
        print(f"错误: 无法读取输入文件 '{input_file}'. 请检查文件路径、编码和分隔符.")
        logging.error(f"无法读取输入文件 '{input_file}': {e}")
        return

    if 'ASIN' not in df.columns or 'country' not in df.columns:
        print("错误: 输入文件必须包含 'ASIN' 和 'country' 列.")
        logging.error("输入文件缺少 'ASIN' 或 'country' 列.")
        return
    
    # 确保 ASIN 和 country 列是字符串类型，并去除前后空格
    df['ASIN'] = df['ASIN'].astype(str).str.strip()
    df['country'] = df['country'].astype(str).str.strip()

    # --- 多进程模式判断 ---
    if profile_count and profile_count > 0:
        # 使用模板批量创建 profile 目录
        multi_process_scraper(df, profile_template, profile_count, concurrency, output_file)
        return  # 结束主函数

    # ===== 兼容原单进程模式 =====
    # 设置用户数据目录并规范化路径
    if not os.path.isabs(profile_dir):
        user_data_dir = os.path.normpath(os.path.join(os.getcwd(), profile_dir))
    else:
        user_data_dir = os.path.normpath(profile_dir)
    # 登录检查与流程
    ensure_login(user_data_dir, df)
    # 抓取详情数据（无头模式，使用持久化登录状态）
    results_data = []  # 重命名以避免与模块名冲突
    try:
        asyncio.run(run_scraper(df, results_data, concurrency, user_data_dir))
    except Exception as e_run:
        logging.critical(f"运行爬虫时发生未处理的严重错误: {e_run}")
        print(f"爬虫运行时发生严重错误: {e_run}")
        return

    # 过滤掉可能的 None 值（如果某行处理完全失败且未被占位）
    valid_results = [r for r in results_data if r is not None]
    if not valid_results:
        print("警告: 未收集到任何有效数据。请检查日志文件 'spider.log' 获取详细信息。")
        logging.warning("未收集到任何有效数据。")
        return

    out_df = pd.DataFrame(valid_results)
    abs_out_path = os.path.abspath(output_file)
    try:
        out_df.to_csv(abs_out_path, index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 确保 Excel 正确打开
        print(f'数据已保存到 {abs_out_path}')
        logging.info(f'数据已保存到 {abs_out_path}')
    except Exception as e_save:
        print(f"错误: 无法保存结果到 '{abs_out_path}': {e_save}")
        logging.error(f"无法保存结果到 '{abs_out_path}': {e_save}")

    # 再跑 Vine 抓取，使用指定的用户数据目录持久化登录信息
    try:
        asyncio.run(run_vine_scraper(df, results_data, concurrency, user_data_dir))
    except Exception as e_vine:
        logging.critical(f"运行 Vine 爬虫时发生未处理的严重错误: {e_vine}")
        print(f"Vine 爬虫运行时发生严重错误: {e_vine}")
        return
    # 最终保存包括 Vine 的完整结果
    final_df = pd.DataFrame(results_data)
    abs_final_path = os.path.abspath(output_file)
    try:
        final_df.to_csv(abs_final_path, index=False, encoding='utf-8-sig')
        if os.path.exists(abs_final_path):
            print(f'完整数据已保存至 {abs_final_path}')
            logging.info(f'完整数据已保存至 {abs_final_path}')
        else:
            print(f'警告: 尝试保存到 {abs_final_path} 但文件未找到，请检查写入权限。')
            logging.warning(f'尝试保存到 {abs_final_path} 但文件未找到，请检查写入权限。')
    except Exception as e_final:
        print(f"错误: 无法保存完整结果到 '{abs_final_path}': {e_final}")
        logging.error(f"无法保存完整结果到 '{abs_final_path}': {e_final}")

    # --- 新增：主流程后自动清洗和调整品类 ---
    clean_and_adjust_categories('output.csv')

if __name__ == '__main__':
    main()