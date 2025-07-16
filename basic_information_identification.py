import asyncio
import logging
import pandas as pd
import click
from playwright.async_api import async_playwright
from tqdm import tqdm
import os
import json
import time
import random
from multiprocessing import Manager

# 日志配置
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('basic_info_spider.log', encoding='utf-8')
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

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.114 Safari/537.36"

async def extract_basic_information(page):
    """
    从亚马逊页面提取基本信息：标题、五点描述和产品概览表格，并抓取主图链接
    """
    result = {
        'title': '',
        'bullet_points': [],
        'product_overview': {},  # 新增产品概览字段
        'main_image': ''         # 新增主图字段
    }
    
    # 提取标题
    try:
        # 尝试使用CSS选择器获取标题
        title_element = await page.query_selector("#title")
        if title_element:
            title_text = await title_element.inner_text()
            result['title'] = title_text.strip()
            logger.info(f"成功提取标题(CSS): {result['title'][:50]}...")
        else:
            # 尝试使用XPath获取标题
            title_element = await page.query_selector("//*[@id='title']")
            if title_element:
                title_text = await title_element.inner_text()
                result['title'] = title_text.strip()
                logger.info(f"成功提取标题(XPath): {result['title'][:50]}...")
            else:
                logger.warning("未找到标题元素")
    except Exception as e:
        logger.error(f"提取标题时出错: {str(e)}")
    
    # 提取五点描述
    try:
        # 尝试使用CSS选择器获取五点描述
        bullet_points_element = await page.query_selector("#feature-bullets")
        if bullet_points_element:
            # 获取所有列表项
            bullet_points = await bullet_points_element.query_selector_all("ul li span.a-list-item")
            for point in bullet_points:
                point_text = await point.inner_text()
                result['bullet_points'].append(point_text.strip())
            logger.info(f"成功提取五点描述(CSS): 共{len(result['bullet_points'])}点")
        else:
            # 尝试使用XPath获取五点描述
            bullet_points_element = await page.query_selector("//div[@id='feature-bullets']")
            if bullet_points_element:
                bullet_points = await bullet_points_element.query_selector_all("ul li span.a-list-item")
                for point in bullet_points:
                    point_text = await point.inner_text()
                    result['bullet_points'].append(point_text.strip())
                logger.info(f"成功提取五点描述(XPath): 共{len(result['bullet_points'])}点")
            else:
                logger.warning("未找到五点描述元素")
    except Exception as e:
        logger.error(f"提取五点描述时出错: {str(e)}")
    
    # 新增：提取产品概览表格
    try:
        # 尝试使用CSS选择器获取产品概览表格
        overview_table = await page.query_selector("#productOverview_feature_div > div > table > tbody")
        if not overview_table:
            # 尝试使用XPath获取产品概览表格（修复选择器语法）
            overview_table = await page.query_selector("xpath=//table[@id='productOverview']//tbody | //div[contains(@id,'productOverview')]//tbody | //table[contains(@class,'prodDetTable')]//tbody")
            
        if overview_table:
            # 获取所有行
            rows = await overview_table.query_selector_all("tr")
            for row in rows:
                try:
                    # 获取标题单元格和值单元格
                    title_cell = await row.query_selector("td:first-child span")
                    value_cell = await row.query_selector("td:last-child span")
                    
                    if title_cell and value_cell:
                        title_text = await title_cell.inner_text()
                        value_text = await value_cell.inner_text()
                        
                        # 去除空白字符
                        title_text = title_text.strip()
                        value_text = value_text.strip()
                        
                        # 添加到结果中
                        result['product_overview'][title_text] = value_text
                except Exception as e:
                    logger.error(f"提取产品概览行时出错: {str(e)}")
                    continue
            
            logger.info(f"成功提取产品概览表格: 共{len(result['product_overview'])}个属性")
        else:
            # 尝试其他可能的选择器
            alternative_selectors = [
                "table.a-normal.a-spacing-micro > tbody",
                "#productDetails_techSpec_section_1 > tbody",
                "#detailBullets_feature_div > ul",
                "#productDetails_detailBullets_sections1 > tbody"
            ]
            
            for selector in alternative_selectors:
                overview_element = await page.query_selector(selector)
                if overview_element:
                    if selector.endswith("tbody"):
                        # 表格形式
                        rows = await overview_element.query_selector_all("tr")
                        for row in rows:
                            try:
                                # 获取标题单元格和值单元格
                                title_cell = await row.query_selector("th")
                                value_cell = await row.query_selector("td")
                                
                                if title_cell and value_cell:
                                    title_text = await title_cell.inner_text()
                                    value_text = await value_cell.inner_text()
                                    
                                    # 去除空白字符
                                    title_text = title_text.strip()
                                    value_text = value_text.strip()
                                    
                                    # 添加到结果中
                                    result['product_overview'][title_text] = value_text
                            except Exception as e:
                                logger.error(f"提取产品概览行时出错: {str(e)}")
                                continue
                    elif selector.endswith("ul"):
                        # 列表形式
                        items = await overview_element.query_selector_all("li")
                        for item in items:
                            try:
                                item_text = await item.inner_text()
                                # 分割键值对
                                parts = item_text.split(":")
                                if len(parts) >= 2:
                                    title_text = parts[0].strip()
                                    value_text = ":".join(parts[1:]).strip()
                                    result['product_overview'][title_text] = value_text
                            except Exception as e:
                                logger.error(f"提取产品概览项时出错: {str(e)}")
                                continue
                    
                    logger.info(f"成功提取产品概览(替代选择器 {selector}): 共{len(result['product_overview'])}个属性")
                    break
            
            if not result['product_overview']:
                logger.warning("未找到产品概览表格")
    except Exception as e:
        logger.error(f"提取产品概览表格时出错: {str(e)}")
    
    # 提取主图链接
    try:
        img_element = await page.query_selector('#landingImage')
        if img_element:
            src = await img_element.get_attribute('src')
            if src:
                result['main_image'] = src.strip()
            else:
                # 兜底：尝试data-old-hires属性
                old_src = await img_element.get_attribute('data-old-hires')
                if old_src:
                    result['main_image'] = old_src.strip()
        else:
            # 兜底：尝试XPath
            img_element = await page.query_selector('//img[@id="landingImage"]')
            if img_element:
                src = await img_element.get_attribute('src')
                if src:
                    result['main_image'] = src.strip()
    except Exception as e:
        logger.error(f"提取主图链接时出错: {str(e)}")
    
    return result

async def handle_continue_shopping(page):
    """
    检测并自动点击亚马逊多语言"继续购物"确认页面的按钮，规避反爬虫。
    """
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
        logger.info('[CONTINUE_SHOPPING] No continue button found')
        return False
    except Exception as e:
        logger.error(f'[CONTINUE_SHOPPING] Exception: {e}')
        return False

async def fetch_product_data(page, url):
    """
    访问产品页面并提取数据
    """
    # 随机延时，模拟人类访问
    await asyncio.sleep(random.uniform(0.8, 1.5))
    
    try:
        # 访问产品页面
        logger.info(f"正在访问: {url}")
        await page.goto(url, timeout=60000, wait_until='domcontentloaded')
        
        # 检测并处理"继续购物"确认页面
        await handle_continue_shopping(page)
        
        # 等待页面加载完成
        await page.wait_for_selector("body", timeout=10000)
        
        # 随机延时，模拟人类访问
        await asyncio.sleep(random.uniform(1.0, 2.0))
        
        # 提取基本信息
        basic_info = await extract_basic_information(page)
        
        return {
            'url': url,
            'title': basic_info['title'],
            'bullet_points': basic_info['bullet_points'],
            'product_overview': basic_info['product_overview'],  # 添加产品概览数据
            'main_image': basic_info['main_image'] # 添加主图链接
        }
    except Exception as e:
        logger.error(f"获取产品数据时出错: {str(e)}")
        return {
            'url': url,
            'title': '',
            'bullet_points': [],
            'product_overview': {},  # 添加空的产品概览字典
            'main_image': '' # 添加空的主图链接
        }

async def run_scraper(df, results_list_ref, concurrency, profile_dir):
    """
    运行爬虫，处理数据
    """
    async with async_playwright() as p:
        # 创建浏览器实例
        context = await p.chromium.launch_persistent_context(
            profile_dir,
            headless=True,  # 设置为True可隐藏浏览器窗口
            executable_path=r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            user_agent=DEFAULT_USER_AGENT,
            locale='en-US',
            timezone_id='America/New_York',
            viewport={'width': 1920, 'height': 1080},
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-features=IsolateOrigins,site-per-process'
            ]
        )
        
        # 创建页面池
        pages = []
        for _ in range(concurrency):
            page = await context.new_page()
            await page.set_extra_http_headers({
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9"
            })
            pages.append(page)
            
        page_queue = asyncio.Queue()
        results_list_ref.clear() # 使用传入的列表引用
        results_list_ref.extend([None] * len(df)) # 预分配，保持顺序

        for p in pages:
            await page_queue.put(p)
            
        # 创建进度条
        pbar = tqdm(total=len(df), desc='抓取进度')

        async def process_row(pos, row_data):
            page = await page_queue.get()
            try:
                asin = str(row_data.get('ASIN', '')).strip()
                country = str(row_data.get('country', '')).strip().upper()
                domain = DOMAIN_MAP.get(country, 'amazon.com')
                
                # 构建URL
                url = f"https://www.{domain}/dp/{asin}"
                
                try:
                    # 获取产品数据
                    result = await fetch_product_data(page, url)
                    
                    # 添加ASIN和国家信息
                    result['ASIN'] = asin
                    result['country'] = country
                    
                    # 添加到结果列表
                    results_list_ref[pos] = result
                    
                    logger.info(f"完成 {pos+1}/{len(df)}: {asin} ({country})")
                except Exception as e:
                    logger.error(f"获取产品数据时出错: {str(e)}")
                    # 确保即使出错也有记录占位
                    if results_list_ref[pos] is None:
                        results_list_ref[pos] = {'ASIN': asin, 'country': country, 'ERROR': str(e)}
            except Exception as e:
                logger.error(f"处理行 {pos} 时出错: {str(e)}")
                # 确保即使出错也有记录占位
                if results_list_ref[pos] is None:
                    results_list_ref[pos] = {'ASIN': str(row_data.get('ASIN', '')).strip(), 'country': str(row_data.get('country', '')).strip().upper(), 'ERROR': str(e)}
            finally:
                await page_queue.put(page)
                pbar.update(1)
        
        # 创建任务列表
        tasks = [asyncio.create_task(process_row(pos, row_data)) for pos, (_, row_data) in enumerate(df.iterrows())]
        await asyncio.gather(*tasks)
        
        pbar.close()
        
        # 关闭所有页面和浏览器
        for p_item in pages:
            await p_item.close()
        await context.close()

@click.command()
@click.option('--input', '-i', 'input_file', default='data/test_input.csv', help='输入CSV/Excel文件路径，包含ASIN和country列')
@click.option('--encoding', '-e', 'encoding', default='utf-8-sig', help='输入CSV文件编码 (例如 utf-8, utf-8-sig, gbk)')
@click.option('--sep', '-s', 'sep', default=',', help='输入CSV文件分隔符 (例如 ",", "\\t", ";")')
@click.option('--concurrency', '-c', 'concurrency', default=3, type=int, help='协程并发数 (建议≤5)')
@click.option('--profile-template', 'profile_template', default='my_browser_profile_', help='用户数据目录前缀模板 (启用多进程时使用)')
@click.option('--profile-count', 'profile_count', default=0, type=int, help='根据模板创建的目录数量 (>0 时启用多进程)')
@click.option('--profile-dir', '-p', 'profile_dir', default='my_browser_profile', help='单进程模式下的用户数据目录')
@click.option('--output', '-o', 'output_file', default='basic_info_output.csv', help='输出CSV文件路径')
def main(input_file, output_file, encoding, sep, concurrency, profile_template, profile_count, profile_dir):
    """抓取亚马逊产品的基本信息：标题和五点描述"""
    try:
        if input_file.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(input_file)
        else:
            # 处理分隔符转义，特别是制表符
            actual_sep = sep.replace('\\t', '\t')
            df = pd.read_csv(input_file, encoding=encoding, sep=actual_sep, dtype=str) # 读取为字符串以保留ASIN格式
        logger.info(f"成功读取输入文件: {input_file}, 共 {len(df)} 条记录.")
    except Exception as e:
        print(f"错误: 无法读取输入文件 '{input_file}'. 请检查文件路径、编码和分隔符.")
        logger.error(f"无法读取输入文件 '{input_file}': {e}")
        return

    if 'ASIN' not in df.columns or 'country' not in df.columns:
        print("错误: 输入文件必须包含 'ASIN' 和 'country' 列.")
        logger.error("输入文件缺少 'ASIN' 或 'country' 列.")
        return
    
    # 确保 ASIN 和 country 列是字符串类型，并去除前后空格
    df['ASIN'] = df['ASIN'].astype(str).str.strip()
    df['country'] = df['country'].astype(str).str.strip()

    # --- 多进程模式判断 ---
    if profile_count and profile_count > 0:
        # 使用模板批量创建 profile 目录
        print("多进程模式暂未实现，使用单进程模式")
    
    # 设置用户数据目录并规范化路径
    if not os.path.isabs(profile_dir):
        user_data_dir = os.path.normpath(os.path.join(os.getcwd(), profile_dir))
    else:
        user_data_dir = os.path.normpath(profile_dir)
    
    # 创建共享结果列表
    results_list = []
    
    # 运行爬虫
    try:
        asyncio.run(run_scraper(df, results_list, concurrency, user_data_dir))
    except Exception as e_run:
        logger.critical(f"运行爬虫时发生未处理的严重错误: {e_run}")
        print(f"爬虫运行时发生严重错误: {e_run}")
        return

    # 过滤掉可能的 None 值
    valid_results = [r for r in results_list if r is not None]
    if not valid_results:
        print("警告: 未收集到任何有效数据。请检查日志文件获取详细信息。")
        logger.warning("未收集到任何有效数据。")
        return

    # 将结果转换为DataFrame
    results_df = pd.DataFrame(valid_results)
    
    # 保存结果
    abs_out_path = os.path.abspath(output_file)
    try:
        # 将bullet_points列表转换为字符串，以便保存到CSV
        if 'bullet_points' in results_df.columns:
            results_df['bullet_points'] = results_df['bullet_points'].apply(lambda x: '\n'.join(x) if isinstance(x, list) else '')
        
        # 将product_overview字典转换为字符串，以便保存到CSV
        if 'product_overview' in results_df.columns:
            results_df['product_overview'] = results_df['product_overview'].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else '{}'
            )
            
            # 提取常见的产品概览字段作为独立列
            common_fields = ['Brand', 'Color', 'Material', 'Shape', 'Size', 'Weight']
            for field in common_fields:
                results_df[f'overview_{field.lower()}'] = results_df['product_overview'].apply(
                    lambda x: json.loads(x).get(field, '') if x and x != '{}' else ''
                )
        
        # 新增：确保main_image字段输出
        if 'main_image' not in results_df.columns:
            results_df['main_image'] = ''
        # 保存结果
        results_df.to_csv(abs_out_path, index=False, encoding='utf-8-sig')
        logger.info(f"已将结果保存到 {abs_out_path}")
        print(f"已将结果保存到 {abs_out_path}")
    except Exception as e_save:
        print(f"错误: 无法保存结果到 '{abs_out_path}': {e_save}")
        logger.error(f"无法保存结果到 '{abs_out_path}': {e_save}")

if __name__ == "__main__":
    main() 