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
import tempfile
import shutil

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
    fetch_product_data, fetch_vine_count, handle_continue_shopping,
    DOMAIN_MAP, DEFAULT_USER_AGENT, extract_bsr_from_node
)
from basic_information_identification import extract_basic_information
from analyze_product_features import (
    call_ollama_api, create_prompt, standardize_color, 
    standardize_material, standardize_shape, _extract_from_overview,
    COLOR_KEYS, MATERIAL_KEYS, SHAPE_KEYS, STANDARD_COLORS,
    get_yolo_model, download_image, yolo_detect_main_object,
    get_dominant_color, infer_shape_from_box
)


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
        try:
            vine_count, latest3_rating = await fetch_vine_count(page, asin, domain)
            result.update({
                'vine_count': vine_count,
                'latest3_rating': latest3_rating
            })
        except Exception as e:
            logger.warning(f"ASIN={asin} - Vine抓取失败: {e}")
            result.update({
                'vine_count': 0,
                'latest3_rating': 0.0
            })
        
        logger.info(f"ASIN={asin} - 原始数据抓取完成")
        return result
        
    except Exception as e:
        logger.error(f"ASIN={asin} - 抓取失败: {e}")
        return result


async def run_scraper_batch(df_batch, profile_dir, concurrency, shared_results, progress_counter):
    """
    运行一个批次的爬虫，每个ASIN抓取所有原始信息
    """
    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            profile_dir,
            headless=True,
            executable_path=r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            user_agent=DEFAULT_USER_AGENT,
            locale='en-US',
            timezone_id='America/New_York',
            viewport={'width': 1920, 'height': 1080}
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
        for p in pages:
            await page_queue.put(p)

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
                await page_queue.put(page)

        # 创建任务
        tasks = [asyncio.create_task(process_row(row_data)) for _, row_data in df_batch.iterrows()]
        await asyncio.gather(*tasks)
        
        # 关闭页面和浏览器
        for p in pages:
            await p.close()
        await context.close()


def worker_process(df_batch, profile_dir, concurrency, shared_results, progress_counter):
    """子进程执行函数"""
    try:
        asyncio.run(run_scraper_batch(df_batch, profile_dir, concurrency, shared_results, progress_counter))
    except Exception as e:
        logger.error(f"[Worker-{profile_dir}] 发生异常: {e}")


def analyze_features_batch(df_batch, batch_size=10, sleep_time=2):
    """
    批量分析产品特征（color, material, shape）
    """
    logger.info(f"开始分析 {len(df_batch)} 个产品的特征")
    
    for i in tqdm(range(0, len(df_batch), batch_size), desc="分析特征"):
        batch = df_batch.iloc[i:min(i+batch_size, len(df_batch))]
        
        for idx, row in batch.iterrows():
            try:
                title = str(row.get('title', '')) if not pd.isna(row.get('title')) else ""
                bullet_points = str(row.get('bullet_points', '')) if not pd.isna(row.get('bullet_points')) else ""
                product_overview_str = str(row.get('product_overview', '{}')) if not pd.isna(row.get('product_overview')) else "{}"
                main_image = str(row.get('main_image', '')) if not pd.isna(row.get('main_image')) else ""

                # 先从产品概览中提取原始值
                raw_color = _extract_from_overview(product_overview_str, COLOR_KEYS)
                raw_material = _extract_from_overview(product_overview_str, MATERIAL_KEYS)
                raw_shape = _extract_from_overview(product_overview_str, SHAPE_KEYS)

                # 标准化
                color_val = standardize_color(raw_color)
                material_val = standardize_material(raw_material)
                shape_val = standardize_shape(raw_shape)

                # 如颜色非标准色或包含非颜色信息，用YOLO识别主色
                color_is_standard = color_val != "Unknown" and color_val.lower() in [c.lower() for c in STANDARD_COLORS]
                color_has_noise = bool(re.search(r'\d|cm|mm|inch|x|\s|,|\.|-', color_val, re.IGNORECASE))
                if (not color_is_standard) or color_has_noise:
                    color_val = "Unknown"

                # YOLO识别主图主体颜色/形状
                if (color_val == "Unknown" or shape_val == "Unknown") and main_image.startswith('http'):
                    try:
                        _ = get_yolo_model()  # 确保模型已加载
                        img_path = download_image(main_image)
                        if img_path:
                            crop, class_id, box = yolo_detect_main_object(img_path)
                            if crop is not None:
                                if color_val == "Unknown":
                                    color_val = get_dominant_color(crop)
                                if shape_val == "Unknown":
                                    shape_val = infer_shape_from_box(box, class_id)
                            try:
                                os.remove(img_path)
                            except:
                                pass
                    except Exception as e:
                        logger.warning(f"YOLO分析失败: {e}")

                # AI兜底
                if "Unknown" in [color_val, material_val, shape_val]:
                    prompt = create_prompt(title, bullet_points)
                    features = call_ollama_api(prompt)
                    if color_val == "Unknown":
                        color_val = standardize_color(features.get('color', 'Unknown'))
                    if material_val == "Unknown":
                        material_val = standardize_material(features.get('material', 'Unknown'))
                    if shape_val == "Unknown":
                        shape_val = standardize_shape(features.get('shape', 'Unknown'))

                # 更新DataFrame
                df_batch.at[idx, 'color'] = color_val
                df_batch.at[idx, 'material'] = material_val
                df_batch.at[idx, 'shape'] = shape_val

                logger.info(f"完成分析 ASIN {row.get('ASIN')}: 颜色={color_val}, 材质={material_val}, 形状={shape_val}")
                
            except Exception as e:
                logger.error(f"分析 ASIN {row.get('ASIN', 'Unknown')} 时出错: {str(e)}")
                # 设置默认值
                df_batch.at[idx, 'color'] = 'Unknown'
                df_batch.at[idx, 'material'] = 'Unknown'
                df_batch.at[idx, 'shape'] = 'Unknown'
        
        # 批次间休息
        if i + batch_size < len(df_batch):
            logger.info(f"批次间休息 {sleep_time} 秒...")
            time.sleep(sleep_time)
    
    return df_batch


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


@click.command()
@click.option('--input', '-i', 'input_file', default='data/test_input.csv', help='输入CSV/Excel文件路径，包含ASIN和country列')
@click.option('--encoding', '-e', 'encoding', default='utf-8-sig', help='输入CSV文件编码')
@click.option('--sep', '-s', 'sep', default=',', help='输入CSV文件分隔符')
@click.option('--batch-size', '-b', 'batch_size', default=50, type=int, help='每批处理多少个ASIN')
@click.option('--sleep-time', '-t', 'sleep_time', default=5, type=int, help='批次间隔秒数')
@click.option('--processes', '-p', 'processes', default=2, type=int, help='进程数')
@click.option('--concurrency', '-c', 'concurrency', default=3, type=int, help='每进程协程数')
@click.option('--profile-template', 'profile_template', default='temp/browser_profile_', help='浏览器用户数据目录前缀')
@click.option('--output', '-o', 'output_file', default='temp/all_info_output.csv', help='最终输出CSV文件路径')
@click.option('--analyze-batch-size', 'analyze_batch_size', default=10, type=int, help='分析时每批大小')
@click.option('--analyze-sleep', 'analyze_sleep', default=2, type=int, help='分析批次间隔秒数')
def main(input_file, encoding, sep, batch_size, sleep_time, processes, concurrency, 
         profile_template, output_file, analyze_batch_size, analyze_sleep):
    """
    一次性爬取Amazon产品的所有信息，包括BSR、评分、评论数、Vine、标题、描述等，
    然后使用YOLO+AI分析产品特征
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
    
    # 2. 检查断点续爬
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
            profile_dir = f"{profile_template}{idx}"
            
            p = Process(target=worker_process, args=(df_slice, profile_dir, concurrency, shared_results, progress_counter))
            p.start()
            processes_list.append(p)
            logger.info(f"启动子进程 {p.pid}，处理行 {start_idx}-{end_idx-1}，profile: {profile_dir}")
        
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
            logger.info(f"子进程 {p.pid} 已结束，exitcode={p.exitcode}")
        
        # 收集所有结果
        new_results = [r for r in shared_results if r and 'ERROR' not in r]
        logger.info(f"爬虫阶段完成，新抓取 {len(new_results)} 条记录")
        
        # 合并新旧结果
        if existing_df.empty:
            all_results = new_results
        else:
            all_results = existing_df.to_dict('records') + new_results
        
        # 保存原始数据
        save_temp_results(all_results, temp_raw_file)
        final_df = pd.DataFrame(all_results)
    
    # 4. 主进程分析阶段
    logger.info("=== 开始主进程分析阶段 ===")
    
    # 检查哪些需要分析（先确保列存在）
    for col in ['color', 'material', 'shape']:
        if col not in final_df.columns:
            final_df[col] = ''
    
    analysis_needed = final_df[
        final_df['color'].isna() | 
        final_df['material'].isna() | 
        final_df['shape'].isna() |
        (final_df['color'] == '') |
        (final_df['material'] == '') |
        (final_df['shape'] == '')
    ].copy()
    
    if not analysis_needed.empty:
        logger.info(f"需要分析 {len(analysis_needed)} 个产品的特征")
        analyzed_df = analyze_features_batch(analysis_needed, analyze_batch_size, analyze_sleep)
        
        # 更新final_df
        for idx, row in analyzed_df.iterrows():
            final_df.at[idx, 'color'] = row['color']
            final_df.at[idx, 'material'] = row['material']
            final_df.at[idx, 'shape'] = row['shape']
    else:
        logger.info("所有产品都已完成特征分析")
    
    # 5. 输出最终结果
    # 确保列顺序
    output_columns = [
        'ASIN', 'country', 'url', 'color', 'material', 'shape', 'title', 
        'bullet_points', 'product_overview', 'bsr_main_category', 'bsr_main_rank', 
        'bsr_sub_category', 'bsr_sub_rank', 'vine_count', 'rating', 'review_count', 
        'latest3_rating'
    ]
    
    # 确保所有列都存在
    for col in output_columns:
        if col not in final_df.columns:
            final_df[col] = ''
    
    # 选择并重新排序列
    output_df = final_df[output_columns].copy()
    
    # 保存最终结果
    abs_output_path = os.path.abspath(output_file)
    output_df.to_csv(abs_output_path, index=False, encoding='utf-8-sig')
    
    print(f"=== 完成！最终结果已保存到 {abs_output_path} ===")
    print(f"共处理 {len(output_df)} 条记录")
    logger.info(f"所有任务完成，最终结果保存到 {abs_output_path}")


if __name__ == '__main__':
    main() 