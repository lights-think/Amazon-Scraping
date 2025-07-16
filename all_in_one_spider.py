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
from multiprocessing import Process, Manager, Queue, Value

# 创建temp目录
os.makedirs('temp', exist_ok=True)

# 日志配置：输出到temp目录
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler('temp/all_in_one_spider.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 导入现有模块的关键函数和配置
from amazon_scraper import (
    fetch_product_data, handle_continue_shopping,
    DOMAIN_MAP, DEFAULT_USER_AGENT, extract_bsr_from_node
)
from basic_information_identification import extract_basic_information





async def fetch_all_product_info(page, asin, country):
    """
    一次性抓取产品的所有信息：BSR、评分、评论数、标题、五点描述、产品概览、主图等
    """
    domain = DOMAIN_MAP.get(country, 'amazon.com')
    url = f'https://www.{domain}/dp/{asin}'
    
    # 强制Amazon页面尽量以英文显示
    if '?' in url:
        url_with_lang = url + '&language=en_US'
    else:
        url_with_lang = url + '?language=en_US'
    
    result = {
        'ASIN': asin,
        'country': country,
        'url': url,
        'title': '',
        'bullet_points': '',
        'product_overview': '{}',
        'main_image': '',
        'bsr_main_category': '',
        'bsr_main_rank': '',
        'bsr_sub_category': '',
        'bsr_sub_rank': '',
        'vine_count': 0,
        'rating': '',
        'review_count': '',
        'latest3_rating': 0.0
    }
    
    logger.info(f"开始抓取 ASIN={asin}, Country={country}, URL={url}")
    
    try:
        # 访问产品页面
        await page.goto(url_with_lang, timeout=60000, wait_until='domcontentloaded')
        await handle_continue_shopping(page)
        await page.wait_for_timeout(2000)
        
        # 1. 抓取Amazon基本信息（BSR、评分、评论数）
        main_cat, main_rank, sub_cat, sub_rank, rating, reviews = await fetch_product_data(page, url_with_lang)
        result.update({
            'bsr_main_category': main_cat,
            'bsr_main_rank': main_rank,
            'bsr_sub_category': sub_cat,
            'bsr_sub_rank': sub_rank,
            'rating': rating,
            'review_count': reviews
        })
        
        # 2. 抓取基本信息（标题、五点描述、产品概览、主图）
        basic_info = await extract_basic_information(page)
        result.update({
            'title': basic_info.get('title', ''),
            'bullet_points': '\n'.join(basic_info.get('bullet_points', [])),
            'product_overview': json.dumps(basic_info.get('product_overview', {}), ensure_ascii=False),
            'main_image': basic_info.get('main_image', '')
        })
        
        # 3. 抓取Vine评论数和最新评分
        # try:
        #     vine_count, latest3_rating = await fetch_vine_count(page, asin, domain)
        #     result.update({
        #         'vine_count': vine_count,
        #         'latest3_rating': latest3_rating
        #     })
        # except Exception as e:
        #     logger.warning(f"ASIN={asin} - Vine抓取失败: {e}")
        #     result.update({
        #         'vine_count': 0,
        #         'latest3_rating': 0.0
        #     })
        
        logger.info(f"ASIN={asin} - 原始数据抓取完成")
        return result
        
    except Exception as e:
        logger.error(f"ASIN={asin} - 抓取失败: {e}")
        return result


async def run_scraper_batch(df_batch, profile_dir, concurrency, shared_results, progress_counter):
    """
    运行一个批次的爬虫，每个ASIN抓取所有原始信息，使用固定的用户资料目录
    """
    context = None
    pages = []
    
    try:
        async with async_playwright() as pw:
            # 创建页面池
            page_queue = asyncio.Queue()
            
            # 创建browser context
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    context = await pw.chromium.launch_persistent_context(
                        profile_dir,
                        headless=True,
                        executable_path=r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                        user_agent=DEFAULT_USER_AGENT,
                        locale='en-US',
                        timezone_id='America/New_York',
                        viewport={'width': 1920, 'height': 1080}
                    )
                    break
                except Exception as e:
                    logger.warning(f"创建浏览器上下文失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(1)
            
            # 创建页面池
            for _ in range(concurrency):
                page = await context.new_page()
                await page.set_extra_http_headers({
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Accept-Language": "en-US,en;q=0.9"
                })
                pages.append(page)
                await page_queue.put(page)
            
            logger.info(f"已创建浏览器上下文，profile: {profile_dir}")
            
            async def process_row(row_data):
                page = await page_queue.get()
                try:
                    asin = str(row_data.get('ASIN', '')).strip()
                    country = str(row_data.get('country', '')).strip().upper()
                    
                    # 抓取所有原始信息
                    result = await fetch_all_product_info(page, asin, country)
                    shared_results.append(result)
                    
                    # 更新进度计数器
                    if progress_counter is not None:
                        try:
                            with progress_counter.get_lock():
                                progress_counter.value += 1
                        except AttributeError:
                            progress_counter.value += 1
                            
                except Exception as e:
                    logger.error(f"处理行失败: {e}")
                    # 确保有记录占位
                    shared_results.append({
                        'ASIN': str(row_data.get('ASIN', '')).strip(),
                        'country': str(row_data.get('country', '')).strip().upper(),
                        'ERROR': str(e)
                    })
                finally:
                    # 检查页面是否仍然有效再放回队列
                    try:
                        if not page.is_closed():
                            await page_queue.put(page)
                    except Exception as e:
                        logger.warning(f"页面放回队列时出错: {e}")

            # 创建任务
            tasks = [asyncio.create_task(process_row(row_data)) for _, row_data in df_batch.iterrows()]
            await asyncio.gather(*tasks)
            
    finally:
        # 关闭页面和浏览器
        if pages:
            for p in pages:
                try:
                    if not p.is_closed():
                        await p.close()
                except Exception as e:
                    logger.debug(f"关闭页面时出错: {e}")
        if context:
            try:
                await context.close()
            except Exception as e:
                logger.debug(f"关闭浏览器上下文时出错: {e}")


def worker_process(df_batch, profile_dir, concurrency, shared_results, progress_counter):
    """子进程执行函数，使用固定的用户资料目录"""
    pid = os.getpid()
    logger.info(f"[Worker-{pid}] 启动子进程，处理 {len(df_batch)} 条记录")
    
    try:
        asyncio.run(run_scraper_batch(df_batch, profile_dir, concurrency, shared_results, progress_counter))
        logger.info(f"[Worker-{pid}] 处理完成")
            
    except Exception as e:
        logger.error(f"[Worker-{pid}] 发生异常: {e}")
        import traceback
        logger.error(f"[Worker-{pid}] 异常详情: {traceback.format_exc()}")


async def fetch_bsr_only(page, asin, country):
    """
    仅抓取产品的BSR信息
    """
    domain = DOMAIN_MAP.get(country, 'amazon.com')
    url = f'https://www.{domain}/dp/{asin}'
    
    # 强制Amazon页面尽量以英文显示
    if '?' in url:
        url_with_lang = url + '&language=en_US'
    else:
        url_with_lang = url + '?language=en_US'
    
    result = {
        'ASIN': asin,
        'country': country,
        'bsr_main_category': '',
        'bsr_main_rank': '',
        'bsr_sub_category': '',
        'bsr_sub_rank': ''
    }
    
    logger.info(f"开始抓取BSR ASIN={asin}, Country={country}, URL={url}")
    
    try:
        # 访问产品页面
        logger.debug(f"ASIN={asin} - 正在访问页面...")
        await page.goto(url_with_lang, timeout=60000, wait_until='domcontentloaded')
        await handle_continue_shopping(page)
        await page.wait_for_timeout(2000)
        logger.debug(f"ASIN={asin} - 页面加载完成，开始提取BSR数据...")
        
        # 只抓取BSR信息
        main_cat, main_rank, sub_cat, sub_rank, _, _ = await fetch_product_data(page, url_with_lang)
        result.update({
            'bsr_main_category': main_cat,
            'bsr_main_rank': main_rank,
            'bsr_sub_category': sub_cat,
            'bsr_sub_rank': sub_rank
        })
        
        logger.info(f"ASIN={asin} - BSR抓取完成: 主类={main_cat}#{main_rank}, 子类={sub_cat}#{sub_rank}")
        return result
        
    except Exception as e:
        logger.error(f"ASIN={asin} - BSR抓取失败: {e}")
        return result


async def run_bsr_update_batch(df_batch, profile_dir, concurrency, shared_results, progress_counter):
    """
    运行BSR更新批次，只抓取BSR信息，使用固定的用户资料目录
    """
    context = None
    pages = []
    
    try:
        async with async_playwright() as pw:
            # 创建页面池
            page_queue = asyncio.Queue()
            
            # 创建browser context
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    context = await pw.chromium.launch_persistent_context(
                        profile_dir,
                        headless=True,
                        executable_path=r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                        user_agent=DEFAULT_USER_AGENT,
                        locale='en-US',
                        timezone_id='America/New_York',
                        viewport={'width': 1920, 'height': 1080}
                    )
                    break
                except Exception as e:
                    logger.warning(f"BSR创建浏览器上下文失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(1)
            
            # 创建页面池
            for _ in range(concurrency):
                page = await context.new_page()
                await page.set_extra_http_headers({
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Accept-Language": "en-US,en;q=0.9"
                })
                pages.append(page)
                await page_queue.put(page)
            
            logger.info(f"BSR已创建浏览器上下文，profile: {profile_dir}")

            async def process_bsr_row(row_data):
                page = await page_queue.get()
                try:
                    asin = str(row_data.get('ASIN', '')).strip()
                    country = str(row_data.get('country', '')).strip().upper()
                    
                    # 只抓取BSR信息
                    result = await fetch_bsr_only(page, asin, country)
                    shared_results.append(result)
                    
                    # 更新进度计数器
                    if progress_counter is not None:
                        try:
                            with progress_counter.get_lock():
                                progress_counter.value += 1
                                logger.debug(f"BSR进度更新: {progress_counter.value} - ASIN={asin}")
                        except AttributeError:
                            progress_counter.value += 1
                            logger.debug(f"BSR进度更新: {progress_counter.value} - ASIN={asin}")
                            
                except Exception as e:
                    logger.error(f"处理BSR行失败: {e}")
                    # 确保有记录占位
                    shared_results.append({
                        'ASIN': str(row_data.get('ASIN', '')).strip(),
                        'country': str(row_data.get('country', '')).strip().upper(),
                        'ERROR': str(e)
                    })
                finally:
                    # 检查页面是否仍然有效再放回队列
                    try:
                        if not page.is_closed():
                            await page_queue.put(page)
                    except Exception as e:
                        logger.warning(f"BSR页面放回队列时出错: {e}")

            # 创建任务
            tasks = [asyncio.create_task(process_bsr_row(row_data)) for _, row_data in df_batch.iterrows()]
            await asyncio.gather(*tasks)
            
    finally:
        # 关闭页面和浏览器
        if pages:
            for p in pages:
                try:
                    if not p.is_closed():
                        await p.close()
                except Exception as e:
                    logger.debug(f"BSR关闭页面时出错: {e}")
        if context:
            try:
                await context.close()
            except Exception as e:
                logger.debug(f"BSR关闭浏览器上下文时出错: {e}")


def bsr_update_worker_process(df_batch, profile_dir, concurrency, shared_results, progress_counter):
    """BSR更新子进程执行函数，使用固定的用户资料目录"""
    try:
        logger.info(f"[BSR-Worker] 进程启动，处理 {len(df_batch)} 条记录，并发数: {concurrency}")
        
        asyncio.run(run_bsr_update_batch(df_batch, profile_dir, concurrency, shared_results, progress_counter))
        logger.info(f"[BSR-Worker] 进程完成")
            
    except Exception as e:
        logger.error(f"[BSR-Worker] 发生异常: {e}")


def load_completed_asins(temp_file):
    """加载已完成的ASIN列表"""
    if os.path.exists(temp_file):
        try:
            df = pd.read_csv(temp_file)
            if 'ASIN' in df.columns and 'country' in df.columns:
                completed = set()
                for _, row in df.iterrows():
                    completed.add((str(row['ASIN']).strip(), str(row['country']).strip().upper()))
                logger.info(f"从 {temp_file} 加载了 {len(completed)} 个已完成的ASIN")
                return completed, df
        except Exception as e:
            logger.warning(f"读取临时文件失败: {e}")
    return set(), pd.DataFrame()


def save_temp_results(results, temp_file):
    """保存临时结果"""
    if results:
        df = pd.DataFrame(results)
        df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        logger.info(f"已保存 {len(results)} 条结果到 {temp_file}")


def update_bsr_mode(df, input_file, processes, concurrency):
    """
    BSR更新模式主函数
    """
    logger.info("=== BSR更新模式：只更新BSR品类为空的记录 ===")
    
    # 确保BSR相关列存在
    bsr_columns = ['bsr_main_category', 'bsr_main_rank', 'bsr_sub_category', 'bsr_sub_rank']
    for col in bsr_columns:
        if col not in df.columns:
            df[col] = ''
    
    # 筛选需要更新BSR的记录
    needs_bsr_update = df[
        df['bsr_main_category'].isna() | 
        (df['bsr_main_category'] == '') |
        df['bsr_main_rank'].isna() | 
        (df['bsr_main_rank'] == '')
    ].copy()
    
    if needs_bsr_update.empty:
        logger.info("没有需要更新BSR的记录，退出程序")
        return
        
    logger.info(f"找到 {len(needs_bsr_update)} 条需要更新BSR的记录")
    
    # 启动多进程BSR更新
    manager = Manager()
    shared_results = manager.list()
    progress_counter = manager.Value('i', 0)
    
    # 分批分配到各进程
    total_to_update = len(needs_bsr_update)
    slice_size = (total_to_update + processes - 1) // processes
    
    print(f"\n=== 启动 {processes} 个BSR更新进程 ===")
    print(f"每个进程并发数: {concurrency}")
    print(f"总计需要更新: {total_to_update} 条记录")
    print("正在启动浏览器进程...")
    
    processes_list = []
    for idx in range(processes):
        start_idx = idx * slice_size
        end_idx = min((idx + 1) * slice_size, total_to_update)
        if start_idx >= end_idx:
            continue
            
        df_slice = needs_bsr_update.iloc[start_idx:end_idx].copy()
        # 为每个进程创建独立的用户资料目录
        profile_dir = f"temp/bsr_profile_{idx}"
        
        p = Process(target=bsr_update_worker_process, args=(df_slice, profile_dir, concurrency, shared_results, progress_counter))
        p.start()
        processes_list.append(p)
        logger.info(f"启动BSR更新子进程 {p.pid}，处理行 {start_idx}-{end_idx-1}，使用profile: {profile_dir}")
        print(f"✓ 进程 {idx+1}/{processes} 已启动 (PID: {p.pid})")
    
    print(f"\n=== 所有进程已启动，等待浏览器初始化完成 ===")
    print("正在等待浏览器启动和页面池创建...")
    
    # 等待一段时间让浏览器初始化，同时显示初始化进度
    init_pbar = tqdm(total=100, desc='浏览器初始化进度', leave=False)
    init_time = 0
    while init_time < 30:  # 最多等待30秒
        # 检查是否有任何进程已经开始工作
        try:
            with progress_counter.get_lock():
                completed = progress_counter.value
        except AttributeError:
            completed = progress_counter.value
            
        if completed > 0:
            # 有进程开始工作了，可以切换到主进度条
            init_pbar.update(100 - init_pbar.n)
            init_pbar.close()
            break
            
        # 更新初始化进度
        init_progress = min(init_time * 3, 95)  # 前30秒内逐渐增加到95%
        init_pbar.update(init_progress - init_pbar.n)
        
        time.sleep(1)
        init_time += 1
    
    # 关闭初始化进度条
    if init_pbar.n < 100:
        init_pbar.update(100 - init_pbar.n)
    init_pbar.close()
    
    print("=== 开始BSR数据抓取 ===")
    
    # 显示总进度
    pbar_total = tqdm(total=total_to_update, desc='BSR更新进度')
    while any(p.is_alive() for p in processes_list):
        try:
            with progress_counter.get_lock():
                completed = progress_counter.value
        except AttributeError:
            completed = progress_counter.value
        pbar_total.update(completed - pbar_total.n)
        time.sleep(1)
    
    # 最终更新
    try:
        with progress_counter.get_lock():
            completed = progress_counter.value
    except AttributeError:
        completed = progress_counter.value
    pbar_total.update(completed - pbar_total.n)
    pbar_total.close()
    
    # 等待所有进程结束
    for p in processes_list:
        p.join()
        logger.info(f"BSR更新子进程 {p.pid} 已结束，exitcode={p.exitcode}")
    
    # 收集所有结果
    bsr_results = [r for r in shared_results if r and 'ERROR' not in r]
    logger.info(f"BSR更新阶段完成，成功抓取 {len(bsr_results)} 条BSR记录")
    
    # 更新原始数据
    print("\n=== 开始更新BSR信息到源文件 ===")
    
    # 创建ASIN和country的索引，用于快速查找
    asin_country_index = {}
    for i, row in df.iterrows():
        key = (str(row['ASIN']).strip(), str(row['country']).strip().upper())
        asin_country_index[key] = i
    
    # 更新BSR信息
    update_count = 0
    
    # 添加进度条
    pbar = tqdm(total=len(bsr_results), desc='数据更新进度')
    
    for result in bsr_results:
        asin = str(result['ASIN']).strip()
        country = str(result['country']).strip().upper()
        key = (asin, country)
        
        if key in asin_country_index:
            idx = asin_country_index[key]
            # 只更新BSR相关字段
            df.at[idx, 'bsr_main_category'] = result['bsr_main_category']
            df.at[idx, 'bsr_main_rank'] = result['bsr_main_rank']
            df.at[idx, 'bsr_sub_category'] = result['bsr_sub_category']
            df.at[idx, 'bsr_sub_rank'] = result['bsr_sub_rank']
            update_count += 1
        
        # 更新进度条
        pbar.update(1)
        # 确保进度条立即显示
        pbar.refresh()
    
    # 关闭进度条
    pbar.close()
    
    logger.info(f"BSR更新模式：已更新 {update_count} 条记录的BSR信息")
    
    # 保存更新后的数据
    print(f"\n=== 保存更新后的数据到 {os.path.abspath(input_file)} ===")
    df.to_csv(input_file, index=False, encoding='utf-8-sig')
    print(f"=== BSR更新完成！结果已保存到 {os.path.abspath(input_file)} ===")
    print(f"共更新 {update_count} 条记录的BSR信息")
    logger.info(f"BSR更新任务完成，结果保存到 {os.path.abspath(input_file)}")


@click.command()
@click.option('--input', '-i', 'input_file', default='data/test_input.csv', help='输入CSV/Excel文件路径，包含ASIN和country列')
@click.option('--encoding', '-e', 'encoding', default='utf-8-sig', help='输入CSV文件编码')
@click.option('--sep', '-s', 'sep', default=',', help='输入CSV文件分隔符')
@click.option('--batch-size', '-b', 'batch_size', default=50, type=int, help='每批处理多少个ASIN')
@click.option('--sleep-time', '-t', 'sleep_time', default=5, type=int, help='批次间隔秒数')
@click.option('--processes', '-p', 'processes', default=2, type=int, help='爬虫进程数')
@click.option('--concurrency', '-c', 'concurrency', default=3, type=int, help='每进程协程数')
@click.option('--output', '-o', 'output_file', default='temp/spider_raw_output.csv', help='爬虫原始数据输出CSV文件路径')
@click.option('--update-bsr', '-u', 'update_bsr', is_flag=True, help='仅更新源文件中BSR品类为空的记录')
def main(input_file, encoding, sep, batch_size, sleep_time, processes, concurrency, 
         output_file, update_bsr):
    """
    Amazon产品信息爬虫，专门负责抓取产品的原始数据
    包括BSR、评分、评论数、标题、描述、产品概览、主图等
    
    使用固定的用户资料目录，每个进程使用独立的profile目录
    """
    
    logger.info("=== 启动 All-in-One Amazon Spider ===")
    
    # 1. 读取输入文件
    try:
        if input_file.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(input_file)
        else:
            actual_sep = sep.replace('\\t', '\t')
            df = pd.read_csv(input_file, encoding=encoding, sep=actual_sep, dtype=str)
        logger.info(f"成功读取输入文件: {input_file}, 共 {len(df)} 条记录")
    except Exception as e:
        print(f"错误: 无法读取输入文件 '{input_file}': {e}")
        logger.error(f"无法读取输入文件: {e}")
        return

    if 'ASIN' not in df.columns or 'country' not in df.columns:
        print("错误: 输入文件必须包含 'ASIN' 和 'country' 列")
        logger.error("输入文件缺少必要的列")
        return
    
    # 数据清洗
    df['ASIN'] = df['ASIN'].astype(str).str.strip()
    df['country'] = df['country'].astype(str).str.strip().str.upper()
    
    # 如果是BSR更新模式，走独立的更新流程
    if update_bsr:
        update_bsr_mode(df, input_file, processes, concurrency)
        return
        
    # 正常模式：检查断点续爬
    temp_raw_file = 'temp/all_info_raw.csv'
    completed_asins, existing_df = load_completed_asins(temp_raw_file)
    
    # 过滤出未完成的ASIN
    def is_completed(row):
        return (str(row['ASIN']).strip(), str(row['country']).strip().upper()) in completed_asins
    
    if completed_asins:
        remaining_df = df[~df.apply(is_completed, axis=1)].copy()
        logger.info(f"跳过 {len(df) - len(remaining_df)} 个已完成的ASIN，剩余 {len(remaining_df)} 个待处理")
    else:
        remaining_df = df.copy()
        logger.info(f"开始处理 {len(remaining_df)} 个ASIN")
    
    if remaining_df.empty:
        logger.info("所有ASIN都已完成，跳过爬虫阶段")
        final_df = existing_df
    else:
        # 3. 多进程爬虫阶段
        logger.info("=== 开始多进程爬虫阶段 ===")
        
        manager = Manager()
        shared_results = manager.list()
        progress_counter = manager.Value('i', 0)
        
        # 分批分配到各进程
        total_remaining = len(remaining_df)
        slice_size = (total_remaining + processes - 1) // processes
        
        processes_list = []
        for idx in range(processes):
            start_idx = idx * slice_size
            end_idx = min((idx + 1) * slice_size, total_remaining)
            if start_idx >= end_idx:
                continue
                
            df_slice = remaining_df.iloc[start_idx:end_idx].copy()
            # 为每个进程创建独立的用户资料目录
            profile_dir = f"temp/spider_profile_{idx}"
            
            p = Process(target=worker_process, args=(df_slice, profile_dir, concurrency, shared_results, progress_counter))
            p.start()
            processes_list.append(p)
            logger.info(f"启动爬虫子进程 {p.pid}，处理行 {start_idx}-{end_idx-1}，使用profile: {profile_dir}")
        
        # 显示总进度
        pbar_total = tqdm(total=total_remaining, desc='爬虫总进度')
        while any(p.is_alive() for p in processes_list):
            try:
                with progress_counter.get_lock():
                    completed = progress_counter.value
            except AttributeError:
                completed = progress_counter.value
            pbar_total.update(completed - pbar_total.n)
            time.sleep(1)
        
        # 最终更新
        try:
            with progress_counter.get_lock():
                completed = progress_counter.value
        except AttributeError:
            completed = progress_counter.value
        pbar_total.update(completed - pbar_total.n)
        pbar_total.close()
        
        # 等待所有进程结束
        for p in processes_list:
            p.join()
            logger.info(f"爬虫子进程 {p.pid} 已结束，exitcode={p.exitcode}")
        
        # 收集所有结果
        new_results = [r for r in shared_results if r and 'ERROR' not in r]
        logger.info(f"爬虫阶段完成，新抓取 {len(new_results)} 条记录")
        
        # 如果是BSR更新模式，只更新BSR相关字段
        if update_bsr:
            print("\n=== 爬虫阶段完成，开始更新BSR信息 ===")
            logger.info("开始更新BSR信息到源文件")
            
            # 创建ASIN和country的索引，用于快速查找
            asin_country_index = {}
            for i, row in existing_df.iterrows():
                key = (str(row['ASIN']).strip(), str(row['country']).strip().upper())
                asin_country_index[key] = i
            
            # 更新BSR信息
            update_count = 0
            
            # 添加进度条
            pbar = tqdm(total=len(new_results), desc='BSR更新进度')
            
            for result in new_results:
                asin = str(result['ASIN']).strip()
                country = str(result['country']).strip().upper()
                key = (asin, country)
                
                if key in asin_country_index:
                    idx = asin_country_index[key]
                    # 只更新BSR相关字段
                    existing_df.at[idx, 'bsr_main_category'] = result['bsr_main_category']
                    existing_df.at[idx, 'bsr_main_rank'] = result['bsr_main_rank']
                    existing_df.at[idx, 'bsr_sub_category'] = result['bsr_sub_category']
                    existing_df.at[idx, 'bsr_sub_rank'] = result['bsr_sub_rank']
                    update_count += 1
                
                # 更新进度条
                pbar.update(1)
                # 确保进度条立即显示
                pbar.refresh()
            
            # 关闭进度条
            pbar.close()
            
            logger.info(f"BSR更新模式：已更新 {update_count} 条记录的BSR信息")
            
            # 保存更新后的数据
            print(f"\n=== 保存更新后的数据到 {os.path.abspath(output_file)} ===")
            existing_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"=== BSR更新完成！结果已保存到 {os.path.abspath(output_file)} ===")
            print(f"共更新 {update_count} 条记录的BSR信息")
            logger.info(f"BSR更新任务完成，结果保存到 {os.path.abspath(output_file)}")
            return
        else:
            # 正常模式：合并新旧结果
            if existing_df.empty:
                all_results = new_results
            else:
                all_results = existing_df.to_dict('records') + new_results
            
            # 保存原始数据
            save_temp_results(all_results, temp_raw_file)
            final_df = pd.DataFrame(all_results)
    
    # 如果是BSR更新模式，跳过特征分析阶段
    if update_bsr:
        logger.info("BSR更新模式：跳过特征分析阶段")
    
    # 输出爬虫原始数据结果
    # 确保列顺序
    output_columns = [
        'ASIN', 'country', 'url', 'title', 
        'bullet_points', 'product_overview', 'main_image', 'bsr_main_category', 'bsr_main_rank', 
        'bsr_sub_category', 'bsr_sub_rank', 'vine_count', 'rating', 'review_count', 
        'latest3_rating'
    ]
    
    # 确保所有列都存在
    for col in output_columns:
        if col not in final_df.columns:
            final_df[col] = ''
    
    # 选择并重新排序列
    output_df = final_df[output_columns].copy()
    
    # 保存爬虫原始数据
    abs_output_path = os.path.abspath(output_file)
    output_df.to_csv(abs_output_path, index=False, encoding='utf-8-sig')
    
    print(f"=== 爬虫完成！原始数据已保存到 {abs_output_path} ===")
    print(f"共抓取 {len(output_df)} 条记录")
    print(f"如需分析产品特征，请运行: python analyze_product_features.py -i {abs_output_path}")
    logger.info(f"爬虫任务完成，原始数据保存到 {abs_output_path}")


if __name__ == '__main__':
    main() 