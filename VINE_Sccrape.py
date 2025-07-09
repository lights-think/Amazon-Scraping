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
from multiprocessing import Process, Manager

# 日志配置：写入 spider.log，与原脚本统一
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler('spider.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 扩展国家域名映射（添加更多主流Amazon站点）
DOMAIN_MAP = {
    'US': 'amazon.com', 'UK': 'amazon.co.uk', 'DE': 'amazon.de', 'FR': 'amazon.fr', 'ES': 'amazon.es',
    'IT': 'amazon.it', 'CA': 'amazon.ca', 'JP': 'amazon.co.jp', 'MX': 'amazon.com.mx', 'IN': 'amazon.in',
    'NL': 'amazon.nl', 'SE': 'amazon.se', 'BE': 'amazon.com.be', 'IE': 'amazon.ie', 'AU': 'amazon.com.au',
    'BR': 'amazon.com.br', 'SG': 'amazon.sg', 'AE': 'amazon.ae', 'TR': 'amazon.com.tr', 'PL': 'amazon.pl',
    'SA': 'amazon.sa', 'EG': 'amazon.eg', 'CN': 'amazon.cn', 'TW': 'amazon.com.tw', 'NO': 'amazon.no'
}

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/136.0.7103.114 Safari/537.36"
)

# -------------------- 核心 VINE 抓取逻辑 --------------------
async def fetch_vine_count(page, asin: str, domain: str):
    """统计某 ASIN 在最后最多 10 页评论中的 Vine 评论数量，并返回最后一页前三条评论平均分。"""
    page_size = 10
    first_url = (
        f'https://www.{domain}/product-reviews/{asin}'
        f'?sortBy=recent&reviewerType=all_reviews&formatType=all_formats'
        f'&pageNumber=1&pageSize={page_size}'
    )
    logging.info(f"[VINE] ASIN={asin} initial review page: {first_url}")

    # 页面加载重试机制
    max_retries = 3
    for retry in range(max_retries):
        try:
            await page.goto(first_url, timeout=60000, wait_until='domcontentloaded')
            # 等待页面稳定加载
            await page.wait_for_timeout(2000)
            
            # 确保评论区域已加载
            try:
                await page.wait_for_selector('div[data-hook="review"], .review, #reviews-section', timeout=10000)
                break
            except:
                if retry < max_retries - 1:
                    logging.warning(f"[VINE] ASIN={asin} 评论区域未加载，重试 {retry + 1}/{max_retries}")
                    await page.wait_for_timeout(3000)
                    continue
                else:
                    logging.warning(f"[VINE] ASIN={asin} 评论区域始终未加载，继续尝试")
                    break
        except Exception as e:
            if retry < max_retries - 1:
                logging.warning(f"[VINE] ASIN={asin} 导航失败，重试 {retry + 1}/{max_retries}: {e}")
                await page.wait_for_timeout(5000)
            else:
                logging.error(f"[VINE] ASIN={asin} navigation error after {max_retries} retries: {e}")
                return 0, 0.0

    # 获取评论总数
    total_reviews = 0
    selectors_try = [
        '#filter-info-section > div',
        'div[data-hook="cr-filter-info-review-rating-count"]'
    ]
    for sel in selectors_try:
        try:
            await page.wait_for_selector(sel, timeout=7000)
            txt = (await page.inner_text(sel)).strip()
            m = re.search(r'([\d,]+)', txt)
            if m:
                total_reviews = int(m.group(1).replace(',', ''))
                break
        except Exception:
            continue

    if total_reviews == 0:
        logging.warning(f"[VINE] ASIN={asin} cannot parse review count")
        return 0, 0.0

    total_pages = min(max((total_reviews + page_size - 1) // page_size, 1), 10)

    vine_count = 0
    latest3_rating = 0.0
    previous_first_review_text = ""

    for page_num in range(1, total_pages + 1):
        if page_num > 1:
            try:
                await page.click('li.a-last a')
                url_pat = re.compile(rf"(\\?|&)pageNumber={page_num}(&|$)")
                await page.wait_for_url(url_pat, timeout=30000)
                await page.wait_for_load_state('domcontentloaded', timeout=30000)
                await page.wait_for_timeout(1000)
            except Exception as nav_e:
                logging.error(f"[VINE] ASIN={asin} nav page {page_num} error: {nav_e}")
                break

        try:
            await page.wait_for_selector('span[data-hook="review-body"]', timeout=6000)
            first_el = await page.query_selector('span[data-hook="review-body"]:first-of-type')
            if first_el:
                txt_now = await first_el.inner_text()
                if txt_now == previous_first_review_text:
                    logging.debug(f"[VINE] ASIN={asin} page {page_num} duplicated, skip counting")
                previous_first_review_text = txt_now
        except Exception:
            pass

        # JS 评估 Vine 标签 - 改进的健壮性实现
        try:
            count = await page.evaluate('''() => {
                // 多种VINE标识符策略
                let vineCount = 0;
                
                // 策略1: 标准VINE文本检测（支持多语言）
                const vineTexts = [
                    'Amazon Vine Customer Review of Free Product',
                    'Vine Customer Review of Free Product',
                    'Vine Customer Review',
                    'Amazon Vine Customer Review',
                    'Amazon Vine Kundenrezension',  // 德语
                    'Vine Kundenrezension',
                    'Kostenlose Probe'  // 德语："免费样品"
                ];
                
                vineTexts.forEach(vineText => {
                    const elements = Array.from(document.querySelectorAll('span.a-color-success.a-text-bold, span.a-color-success, .a-color-success'));
                    const count = elements.filter(el => {
                        const text = el.textContent ? el.textContent.trim() : '';
                        return text.includes(vineText);
                    }).length;
                    vineCount = Math.max(vineCount, count);
                });
                
                // 策略2: 通过class属性检测vine标识
                const vineClassElements = Array.from(document.querySelectorAll('[class*="vine"], [data-hook*="vine"], .vine-review, .amazon-vine'));
                if (vineClassElements.length > 0) {
                    vineCount = Math.max(vineCount, vineClassElements.length);
                }
                
                // 策略3: 通过绿色文本检测(VINE标识通常是绿色)
                if (vineCount === 0) {
                    const greenElements = Array.from(document.querySelectorAll('span.a-color-success, .a-color-success'));
                    const vineKeywords = [
                        'vine', 'free product', '免费产品', 
                        'kostenlos', 'kostenloses produkt', 'gratis',  // 德语关键词
                        'produit gratuit', 'gratuito',  // 法语、意大利语
                        'producto gratuito'  // 西班牙语
                    ];
                    greenElements.forEach(el => {
                        const text = el.textContent ? el.textContent.toLowerCase() : '';
                        if (vineKeywords.some(keyword => text.includes(keyword))) {
                            vineCount++;
                        }
                    });
                }
                
                // 策略4: 通过图标检测(有些VINE评论有特殊图标)
                if (vineCount === 0) {
                    const iconElements = Array.from(document.querySelectorAll('i[class*="vine"], img[alt*="vine"], img[title*="vine"]'));
                    vineCount = Math.max(vineCount, iconElements.length);
                }
                
                return vineCount;
            }''')
            vine_count += count
            if count > 0:
                logging.debug(f"[VINE] ASIN={asin} 第{page_num}页找到{count}个VINE评论")
        except Exception as js_e:
            logging.error(f"[VINE] ASIN={asin} JS eval error page {page_num}: {js_e}")

        # 处理最后一页前三条评分 - 改进的健壮性实现
        if page_num == total_pages:
            try:
                avg_rating = await page.evaluate('''() => {
                    // 基于测试结果优化的评分选择器策略
                    const selectors = [
                        // 最有效的选择器放在前面（基于德国站点测试结果）
                        'i[data-hook="cmps-review-star-rating"] span.a-icon-alt',
                        'i.a-icon.a-icon-star span.a-icon-alt',
                        'i[data-hook="review-star-rating"] span.a-icon-alt',
                        'span[data-hook="review-star-rating"] span.a-icon-alt',
                        '.review-rating i span.a-icon-alt',
                        '.cr-original-review-text i span.a-icon-alt'
                    ];
                    
                    let allRatings = [];
                    
                    // 策略1: 优先使用测试验证有效的选择器
                    for (const selector of selectors) {
                        const elements = Array.from(document.querySelectorAll(selector));
                        if (elements.length > 0) {
                            elements.slice(0, 3).forEach(el => {
                                const text = el.textContent || '';
                                // 支持多语言评分文本匹配（英语、德语等）
                                const match = text.match(/(\d+(?:[,\.]\d+)?)\s*(von|out\s*of)\s*5\s*(stern|star)/i) || 
                                             text.match(/(\d+(?:[,\.]\d+)?)\s*out\s*of\s*5/i) || 
                                             text.match(/(\d+(?:[,\.]\d+)?)/);
                                if (match) {
                                    // 处理不同的小数点格式（逗号/点号）
                                    const rating = parseFloat(match[1].replace(',', '.'));
                                    if (rating >= 1 && rating <= 5) {
                                        allRatings.push(rating);
                                    }
                                }
                            });
                            if (allRatings.length >= 3) break; // 找到足够的评分就停止
                        }
                    }
                    
                    // 策略2: 从CSS类名提取评分 (针对a-star-5这种格式)
                    if (allRatings.length < 3) {
                        const starElements = Array.from(document.querySelectorAll('i.a-icon-star, i[data-hook*="review-star-rating"], i[data-hook*="cmps-review-star-rating"]'));
                        starElements.slice(0, 5).forEach(el => {
                            if (allRatings.length >= 3) return;
                            const className = el.className ? el.className.toString() : '';
                            const classMatch = className.match(/a-star-(\d+(?:[,\.]\d+)?)/);
                            if (classMatch) {
                                const rating = parseFloat(classMatch[1].replace(',', '.'));
                                if (rating >= 1 && rating <= 5) {
                                    allRatings.push(rating);
                                }
                            }
                        });
                    }
                    
                    // 策略3: 基于评论容器的精确查找
                    if (allRatings.length < 3) {
                        const reviewContainers = Array.from(document.querySelectorAll('div[data-hook="review"], .review, .cr-original-review-text'));
                        reviewContainers.slice(0, 3).forEach(container => {
                            if (allRatings.length >= 3) return;
                            // 优先使用测试验证有效的选择器
                            const ratingSelectors = [
                                'i[data-hook="cmps-review-star-rating"] span.a-icon-alt',
                                'i.a-icon-star span.a-icon-alt',
                                'i[data-hook="review-star-rating"] span.a-icon-alt'
                            ];
                            
                            for (const selector of ratingSelectors) {
                                const ratingEl = container.querySelector(selector);
                                if (ratingEl) {
                                    const text = ratingEl.textContent || '';
                                    const match = text.match(/(\d+(?:[,\.]\d+)?)\s*(von|out\s*of)\s*5/i) || 
                                                 text.match(/(\d+(?:[,\.]\d+)?)/);
                                    if (match) {
                                        const rating = parseFloat(match[1].replace(',', '.'));
                                        if (rating >= 1 && rating <= 5) {
                                            allRatings.push(rating);
                                            break;
                                        }
                                    }
                                }
                            }
                        });
                    }
                    
                    // 去重并计算平均值，保留一位小数
                    const uniqueRatings = [...new Set(allRatings)].slice(0, 3);
                    const average = uniqueRatings.length ? uniqueRatings.reduce((a,b)=>a+b,0)/uniqueRatings.length : 0;
                    return Math.round(average * 10) / 10; // 保留一位小数
                }''')
                latest3_rating = round(avg_rating, 1) if avg_rating > 0 else 0.0
                if latest3_rating > 0:
                    logging.info(f"[VINE] ASIN={asin} 成功获取评分: {latest3_rating} (使用优化后的多语言选择器)")
                else:
                    logging.warning(f"[VINE] ASIN={asin} 未能获取到有效评分，请检查页面结构")
            except Exception as rt_e:
                logging.error(f"[VINE] ASIN={asin} rating calc error: {rt_e}")
                latest3_rating = 0.0

    logging.info(f"[VINE] ASIN={asin} vine_count={vine_count}, latest3_rating={latest3_rating}")
    return vine_count, latest3_rating


async def run_vine_scraper(df: pd.DataFrame, results_list_ref: list, concurrency: int, user_data_dir: str, login_states: dict = None):
    """
    并发抓取整个 DataFrame 的 VINE 数据，结果写入 results_list_ref。
    
    Args:
        df: 数据表，包含ASIN和country列
        results_list_ref: 结果列表引用
        concurrency: 并发数
        user_data_dir: 浏览器配置目录
        login_states: 各国家登录状态字典 {'US': True, 'UK': False, ...}
    """
    target_locale = 'en-US'
    target_timezone = 'America/New_York'
    accept_language = 'en-US,en;q=0.9'

    # 如果未提供登录状态，默认所有国家都已登录
    if login_states is None:
        login_states = {country: True for country in df['country'].str.strip().str.upper().unique()}
    
    # 过滤出已登录国家的记录
    logged_in_mask = df['country'].str.strip().str.upper().apply(lambda c: c in login_states and login_states[c])
    logged_in_df = df[logged_in_mask].copy()
    
    # 如果没有已登录的国家记录，直接返回
    if logged_in_df.empty:
        logging.warning("[VINE] 没有已登录国家的记录，无法执行爬取")
        print("警告: 没有已登录国家的记录，请先登录相关国家")
        return
    
    # 记录未登录的国家
    not_logged_in_countries = df.loc[~logged_in_mask, 'country'].str.strip().str.upper().unique()
    if len(not_logged_in_countries) > 0:
        logging.warning(f"[VINE] 以下国家未登录，相关记录将被跳过: {', '.join(not_logged_in_countries)}")
        print(f"注意: 以下国家未登录，相关记录将被跳过: {', '.join(not_logged_in_countries)}")

    logging.info(f"[VINE] 环境 locale={target_locale}, timezone={target_timezone}, 处理 {len(logged_in_df)} 条记录")

    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            user_data_dir,
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
            pages.append(page)

        page_queue = asyncio.Queue()
        # 预分配结果占位，保持顺序
        results_list_ref.clear()
        results_list_ref.extend([None] * len(df))

        for p in pages:
            await page_queue.put(p)

        pbar = tqdm(total=len(logged_in_df), desc='Vine Scraping')

        async def process_row(pos: int, row_data):
            page = await page_queue.get()
            try:
                asin = str(row_data.get('ASIN', '')).strip()
                country = str(row_data.get('country', '')).strip().upper()
                domain = DOMAIN_MAP.get(country, 'amazon.com')

                vine_count, latest3_rating = await fetch_vine_count(page, asin, domain)
                record = {
                    'ASIN': asin,
                    'country': country,
                    'vine_count': vine_count,
                    'latest3_rating': latest3_rating
                }
                results_list_ref[pos] = record
            except Exception as e_row:
                logging.error(f"[VINE] ASIN={row_data.get('ASIN','')} row error: {e_row}")
                if results_list_ref[pos] is None:
                    results_list_ref[pos] = {
                        'ASIN': str(row_data.get('ASIN', '')).strip(),
                        'country': str(row_data.get('country', '')).strip().upper(),
                        'ERROR': str(e_row)
                    }
            finally:
                await page_queue.put(page)
                pbar.update(1)

        # 只处理已登录国家的记录
        tasks = []
        for i, (_, row) in enumerate(df.iterrows()):
            country = str(row.get('country', '')).strip().upper()
            if country in login_states and login_states[country]:
                tasks.append(asyncio.create_task(process_row(i, row)))
        
        if tasks:
            await asyncio.gather(*tasks)
        pbar.close()

        for p in pages:
            await p.close()
        await context.close()


# -------------------- 登录及会话复用 --------------------
async def login_flow(profile_dir: str, login_url: str, country: str):
    """登录流程，支持指定国家的登录，并返回登录成功状态"""
    target_locale = 'en-US'
    target_timezone = 'America/New_York'
    accept_language = 'en-US,en;q=0.9'

    async with async_playwright() as pw:
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
        
        print(f"\n开始 {country} 站点登录流程...")
        print(f"1. 正在打开浏览器访问: {login_url}")
        await page.goto(login_url)
        
        print(f"2. 请在浏览器中完成 {country} 站点的登录")
        print("   - 输入账号密码并通过可能的验证")
        print("   - 确认您能看到评论页面")
        print("   - 完成后回到此命令行窗口")
        
        user_input = input("\n请选择操作: \n[1] 登录成功(默认) \n[2] 跳过此国家 \n[3] 登录失败，重试 \n请输入选项(1/2/3): ")
        
        if user_input.strip() == '3':
            print(f"重新尝试登录 {country}...")
            await page.reload()
            user_input = input("\n请再次选择操作: \n[1] 登录成功(默认) \n[2] 跳过此国家 \n请输入选项(1/2): ")
            login_success = user_input.strip() != '2'
        else:
            login_success = user_input.strip() != '2'
        
        if login_success:
            print(f"{country} 登录已确认成功!")
            # 会话预热：模拟人类行为，确保 Cookie 写入
            try:
                await page.evaluate("window.scrollBy(0, 500)")
                await page.wait_for_timeout(1000)
                await page.evaluate("window.scrollBy(0, -300)")
                await page.wait_for_timeout(500)
            except Exception:
                pass
        else:
            print(f"{country} 登录已跳过")
            
        await context.close()
        return login_success


def ensure_login(profile_dir: str, df: pd.DataFrame, force_login: bool = False):
    """
    确保所有需要的国家都已登录
    
    Args:
        profile_dir: 浏览器配置目录
        df: 数据表，包含country列
        force_login: 是否强制重新登录
    
    Returns:
        dict: 各国家登录状态字典 {'US': True, 'UK': False, ...}
    """
    login_file = os.path.join(profile_dir, 'login_state.json')
    os.makedirs(profile_dir, exist_ok=True)
    
    # 初始化登录状态字典
    login_states = {}
    
    # 读取现有登录状态
    if os.path.exists(login_file) and not force_login:
        try:
            with open(login_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 处理旧格式兼容
                if isinstance(data, dict) and 'logged_in' in data and data['logged_in'] is True:
                    # 旧格式，转换为新格式
                    first_country = df['country'].iloc[0].strip().upper() if not df.empty else 'US'
                    login_states = {first_country: True}
                    print(f"检测到旧版登录状态文件，已转换为新格式: {first_country} 已登录")
                elif isinstance(data, dict):
                    # 新格式，直接使用
                    login_states = data
        except Exception as e:
            logging.warning(f"读取登录状态文件失败: {e}")
            login_states = {}
    
    # 获取数据中的所有国家
    countries = df['country'].str.strip().str.upper().unique()
    
    # 检查哪些国家需要登录
    need_login = []
    for country in countries:
        if country not in login_states or not login_states[country]:
            need_login.append(country)
    
    if not need_login:
        print("所有需要的国家站点均已登录")
        return login_states
    
    # 显示登录状态
    print("\n===== 当前登录状态 =====")
    for country in countries:
        status = "已登录" if country in login_states and login_states[country] else "未登录"
        print(f"{country}: {status}")
    
    # 询问用户是否要登录所有未登录的国家
    if need_login:
        print(f"\n以下国家需要登录: {', '.join(need_login)}")
        choice = input("是否要登录所有未登录国家? (y/n，或输入特定国家代码如'US,UK'): ").strip().upper()
        
        countries_to_login = []
        if choice == 'Y':
            countries_to_login = need_login
        elif choice != 'N':
            # 用户指定了特定国家
            countries_to_login = [c.strip() for c in choice.split(',') if c.strip() in need_login]
        
        # 执行登录流程
        for country in countries_to_login:
            if country not in DOMAIN_MAP:
                print(f"警告: 不支持的国家代码 {country}")
                continue
                
            # 获取该国家的一个ASIN用于登录
            country_df = df[df['country'].str.strip().str.upper() == country]
            if country_df.empty:
                print(f"错误: 数据中没有 {country} 的记录")
                continue
                
            asin = str(country_df.iloc[0].get('ASIN', '')).strip()
            domain = DOMAIN_MAP.get(country, 'amazon.com')
            login_url = (
                f'https://www.{domain}/product-reviews/{asin}'
                f'?sortBy=recent&reviewerType=all_reviews&formatType=all_formats&pageNumber=1&pageSize=10'
            )
            
            print(f"\n开始 {country} 登录流程，URL: {login_url}")
            login_success = asyncio.run(login_flow(profile_dir, login_url, country))
            
            # 更新登录状态
            login_states[country] = login_success
            
            # 实时保存登录状态
            with open(login_file, 'w', encoding='utf-8') as f:
                json.dump(login_states, f, ensure_ascii=False, indent=2)
            
            print(f"{country} 登录{'成功' if login_success else '已跳过'}")
    
    # 显示最终登录状态
    print("\n===== 最终登录状态 =====")
    for country in countries:
        status = "已登录" if country in login_states and login_states[country] else "未登录"
        print(f"{country}: {status}")
    
    return login_states


# -------------------- 多进程封装 --------------------

def _worker_process(df_slice: pd.DataFrame, profile_dir: str, concurrency: int, shared_results, progress_counter):
    """子进程处理函数，处理数据切片并更新共享结果"""
    try:
        # 读取该进程的登录状态
        login_states = {}
        login_file = os.path.join(profile_dir, 'login_state.json')
        if os.path.exists(login_file):
            try:
                with open(login_file, 'r', encoding='utf-8') as f:
                    login_states = json.load(f)
            except Exception as e:
                logging.error(f"[Worker-{profile_dir}] 读取登录状态失败: {e}")
        
        results_temp = []
        asyncio.run(run_vine_scraper(df_slice, results_temp, concurrency, profile_dir, login_states))
        shared_results.extend([r for r in results_temp if r is not None])
        with progress_counter.get_lock():
            progress_counter.value += len(df_slice)
    except Exception as e:
        logging.error(f"[Worker-{profile_dir}] error: {e}")


def multi_process_scraper(df: pd.DataFrame, profile_template: str, profile_count: int, concurrency: int, output_file: str, force_login: bool = False):
    """
    多进程爬虫调度器
    
    Args:
        df: 数据表，包含ASIN和country列
        profile_template: 浏览器配置目录模板
        profile_count: 进程数
        concurrency: 每个进程的并发数
        output_file: 输出文件路径
        force_login: 是否强制重新登录
    """
    manager = Manager()
    shared_results = manager.list()
    progress_counter = manager.Value('i', 0)

    total = len(df)
    slice_size = (total + profile_count - 1) // profile_count

    # 首先确保所有国家都有对应的登录状态
    all_countries = df['country'].str.strip().str.upper().unique()
    print(f"数据中包含以下国家: {', '.join(all_countries)}")
    
    # 检查每个进程目录的登录状态
    all_login_states = {}
    for idx in range(profile_count):
        profile_dir = f"{profile_template}{idx}"
        os.makedirs(profile_dir, exist_ok=True)
        login_file = os.path.join(profile_dir, 'login_state.json')
        
        # 读取现有登录状态
        profile_login_states = {}
        if os.path.exists(login_file) and not force_login:
            try:
                with open(login_file, 'r', encoding='utf-8') as f:
                    profile_login_states = json.load(f)
            except Exception as e:
                logging.warning(f"读取登录状态文件失败 {login_file}: {e}")
        
        # 更新总登录状态
        for country in profile_login_states:
            if country not in all_login_states or not all_login_states[country]:
                all_login_states[country] = profile_login_states[country]
    
    # 显示当前登录状态
    print("\n===== 当前所有进程登录状态汇总 =====")
    for country in all_countries:
        status = "已登录" if country in all_login_states and all_login_states[country] else "未登录"
        print(f"{country}: {status}")
    
    # 检查是否需要登录
    need_login = [c for c in all_countries if c not in all_login_states or not all_login_states[c]]
    if need_login:
        print(f"\n以下国家需要登录: {', '.join(need_login)}")
        choice = input("是否要登录所有未登录国家? (y/n，或输入特定国家代码如'US,UK'): ").strip().upper()
        
        countries_to_login = []
        if choice == 'Y':
            countries_to_login = need_login
        elif choice != 'N':
            # 用户指定了特定国家
            countries_to_login = [c.strip() for c in choice.split(',') if c.strip() in need_login]
        
        # 按国家分配登录任务给不同进程
        if countries_to_login:
            for i, country in enumerate(countries_to_login):
                # 轮流分配给不同进程
                profile_idx = i % profile_count
                profile_dir = f"{profile_template}{profile_idx}"
                
                # 获取该国家的一个ASIN用于登录
                country_df = df[df['country'].str.strip().str.upper() == country]
                if country_df.empty:
                    print(f"错误: 数据中没有 {country} 的记录")
                    continue
                    
                asin = str(country_df.iloc[0].get('ASIN', '')).strip()
                domain = DOMAIN_MAP.get(country, 'amazon.com')
                login_url = (
                    f'https://www.{domain}/product-reviews/{asin}'
                    f'?sortBy=recent&reviewerType=all_reviews&formatType=all_formats&pageNumber=1&pageSize=10'
                )
                
                print(f"\n使用进程 {profile_idx} 登录 {country}，URL: {login_url}")
                login_success = asyncio.run(login_flow(profile_dir, login_url, country))
                
                # 更新该进程的登录状态文件
                login_file = os.path.join(profile_dir, 'login_state.json')
                profile_login_states = {}
                if os.path.exists(login_file):
                    try:
                        with open(login_file, 'r', encoding='utf-8') as f:
                            profile_login_states = json.load(f)
                    except:
                        profile_login_states = {}
                
                profile_login_states[country] = login_success
                with open(login_file, 'w', encoding='utf-8') as f:
                    json.dump(profile_login_states, f, ensure_ascii=False, indent=2)
                
                print(f"{country} 登录{'成功' if login_success else '已跳过'}")
                
                # 更新总登录状态
                all_login_states[country] = login_success
    
    # 显示登录状态并确认是否继续
    print("\n===== 爬取前登录状态确认 =====")
    logged_in_countries = []
    not_logged_countries = []
    
    for country in all_countries:
        if country in all_login_states and all_login_states[country]:
            logged_in_countries.append(country)
            status = "已登录"
        else:
            not_logged_countries.append(country)
            status = "未登录"
        print(f"{country}: {status}")
    
    if not logged_in_countries:
        print("\n警告: 没有任何国家处于登录状态，无法执行爬取")
        if input("是否返回登录流程? (y/n): ").lower() == 'y':
            # 重新调用自身，强制登录
            return multi_process_scraper(df, profile_template, profile_count, concurrency, output_file, True)
        else:
            print("程序退出")
            return
    
    if not_logged_countries:
        print(f"\n注意: 以下国家未登录，相关记录将被跳过: {', '.join(not_logged_countries)}")
    
    confirm = input("\n所有登录已完成，是否开始爬取? (y/n): ")
    if confirm.lower() != 'y':
        print("爬取已取消，程序退出")
        return

    processes = []
    for idx in range(profile_count):
        start, end = idx * slice_size, min((idx + 1) * slice_size, total)
        if start >= end:
            continue
        df_slice = df.iloc[start:end].copy()
        profile_dir = f"{profile_template}{idx}"
        p = Process(target=_worker_process, args=(df_slice, profile_dir, concurrency, shared_results, progress_counter))
        p.start()
        processes.append(p)
        logging.info(f"[Master] 子进程 {p.pid} 处理 {start}-{end - 1}")

    pbar_total = tqdm(total=total, desc='Total Progress')
    while any(p.is_alive() for p in processes):
        with progress_counter.get_lock():
            completed = progress_counter.value
        pbar_total.update(completed - pbar_total.n)
        time.sleep(0.5)
    with progress_counter.get_lock():
        pbar_total.update(progress_counter.value - pbar_total.n)
    pbar_total.close()

    for p in processes:
        p.join()

    results = [r for r in shared_results if r]
    if not results:
        print("警告: 未收集到任何有效数据，详见 spider.log")
        return

    out_df = pd.DataFrame(results)
    abs_path = os.path.abspath(output_file)
    out_df.to_csv(abs_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存到 {abs_path}")
    logging.info(f"[Master] 完成，多进程结果写入 {abs_path}")


# -------------------- CLI 入口 --------------------
@click.command()
@click.option('--input', '-i', 'input_file', default='data/test_input.csv', help='输入CSV/Excel，需含ASIN和country列')
@click.option('--encoding', '-e', default='utf-8-sig', help='CSV 编码')
@click.option('--sep', '-s', default=',', help='CSV 分隔符')
@click.option('--concurrency', '-c', default=3, type=int, help='单进程内协程并发数')
@click.option('--profile-template', default='my_browser_profile_', help='多进程用户数据目录前缀')
@click.option('--profile-count', default=0, type=int, help='>0 启用多进程子进程数量')
@click.option('--profile-dir', '-p', default='my_browser_profile', help='单进程模式用户数据目录')
@click.option('--output', '-o', 'output_file', default='vine_output.csv', help='输出CSV路径')
@click.option('--force-login', is_flag=True, help='强制重新登录所有国家')
@click.option('--login-only', is_flag=True, help='仅执行登录流程，不进行爬取')

def main(input_file, output_file, encoding, sep, concurrency, profile_template, profile_count, profile_dir, force_login, login_only):
    """VINE 爬虫主入口，与原脚本参数保持一致。"""
    try:
        if input_file.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(input_file)
        else:
            df = pd.read_csv(input_file, encoding=encoding, sep=sep.replace('\\t', '\t'), dtype=str)
        logging.info(f"读取输入 {input_file} 共 {len(df)} 条")
    except Exception as e:
        print(f"错误: 无法读取 {input_file}: {e}")
        logging.error(f"无法读取输入文件: {e}")
        return

    if 'ASIN' not in df.columns or 'country' not in df.columns:
        print("错误: 输入文件必须包含 ASIN 和 country 列")
        logging.error("缺少必需列 ASIN/country")
        return

    df['ASIN'] = df['ASIN'].astype(str).str.strip()
    df['country'] = df['country'].astype(str).str.strip()

    if profile_count and profile_count > 0:
        multi_process_scraper(df, profile_template, profile_count, concurrency, output_file, force_login)
        return

    if not os.path.isabs(profile_dir):
        user_dir = os.path.join(os.getcwd(), profile_dir)
    else:
        user_dir = profile_dir

    # 确保登录
    login_states = ensure_login(user_dir, df, force_login)
    
    # 如果仅登录模式，则结束
    if login_only:
        print("登录流程已完成，退出程序")
        return
    
    # 显示登录状态并确认是否继续
    print("\n===== 爬取前登录状态确认 =====")
    countries = df['country'].str.strip().str.upper().unique()
    logged_in_countries = []
    not_logged_countries = []
    
    for country in countries:
        if country in login_states and login_states[country]:
            logged_in_countries.append(country)
            status = "已登录"
        else:
            not_logged_countries.append(country)
            status = "未登录"
        print(f"{country}: {status}")
    
    if not logged_in_countries:
        print("\n警告: 没有任何国家处于登录状态，无法执行爬取")
        if input("是否返回登录流程? (y/n): ").lower() == 'y':
            login_states = ensure_login(user_dir, df, True)  # 强制重新登录
            if not any(login_states.values()):
                print("仍然没有登录任何国家，程序退出")
                return
        else:
            print("程序退出")
            return
    
    if not_logged_countries:
        print(f"\n注意: 以下国家未登录，相关记录将被跳过: {', '.join(not_logged_countries)}")
    
    confirm = input("\n所有登录已完成，是否开始爬取? (y/n): ")
    if confirm.lower() != 'y':
        print("爬取已取消，程序退出")
        return

    results = [None] * len(df)
    try:
        asyncio.run(run_vine_scraper(df, results, concurrency, user_dir, login_states))
    except Exception as exc:
        logging.critical(f"运行 VINE 爬虫错误: {exc}")
        print(f"严重错误: {exc}")
        return

    valid = [r for r in results if r]
    if not valid:
        print("警告: 没有有效数据，请检查日志")
        return

    out_df = pd.DataFrame(valid)
    abs_path = os.path.abspath(output_file)
    out_df.to_csv(abs_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存到 {abs_path}")
    logging.info(f"单进程结果写入 {abs_path}")


if __name__ == '__main__':
    main() 