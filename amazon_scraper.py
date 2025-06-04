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
    'IN': 'amazon.in'
}

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.114 Safari/537.36"

async def fetch_product_data(page, url):
    # 随机延时，模拟人类访问
    await asyncio.sleep(random.uniform(0.8, 1.5)) # 稍微增加延时
    try:
        await page.goto(url, timeout=60000, wait_until='domcontentloaded')
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

    # 默认值
    main_category = ''
    main_rank = ''
    sub_category = ''
    sub_rank = ''

    # BSR解析：先通过表头文本定位 Best Sellers Rank，再解析大类/小类及排名
    try:
        bsr_td = await page.query_selector('xpath=//th[contains(normalize-space(text()),"Best Sellers Rank")]/following-sibling::td | //th[contains(normalize-space(text()),"Best Sellers Rank")]/../td') # 兼容不同层级
        if bsr_td:
            td_text = await bsr_td.inner_text()
            lines = [line.strip() for line in td_text.splitlines() if line.strip()]
            if len(lines) >= 1:
                m = re.match(r'#([\d,]+) in (.+?)(?: \(|$)', lines[0])
                if m:
                    main_rank = m.group(1).replace(',', '')
                    main_category = m.group(2).strip()
            if len(lines) >= 2:
                m2 = re.match(r'#([\d,]+) in (.+)', lines[1])
                if m2:
                    sub_rank = m2.group(1).replace(',', '')
                    sub_category = m2.group(2).strip()
        else: # 回退到旧选择器或更通用的列表项查找
            logging.debug(f"ASIN from URL {url} - BSR table cell not found via th, trying list item fallback.")
            # BSR 备用解析：通用 li 匹配 Best Sellers Rank 并文本规范化 (移到这里作为主要回退)
            fb_elem = await page.query_selector(
                'xpath=//li[contains(normalize-space(.),"Best Sellers Rank")] | //div[contains(normalize-space(.),"Best Sellers Rank") and contains(@id, "detailBullets")]//li[contains(normalize-space(.),"#")]'
            )
            if fb_elem:
                fb_text = await fb_elem.inner_text()
                fb_text = re.sub(r'\s+', ' ', fb_text)
                matches = re.findall(r'#([\d,]+) in ([^#\(]+?)(?:\s*\(See Top 100|\s*$)', fb_text) # 改进正则，避免捕获 "See Top 100"
                if matches:
                    if not main_rank and len(matches) > 0: # 仅当主BSR未找到时填充
                        main_rank = matches[0][0].replace(',', '')
                        main_category = matches[0][1].strip().rstrip('.')
                    if not sub_rank and len(matches) > 1: # 仅当子BSR未找到时填充
                        sub_rank = matches[1][0].replace(',', '')
                        sub_category = matches[1][1].strip().rstrip('.')
    except Exception as e:
        logging.warning(f"ASIN from URL {url} - Error parsing BSR: {e}")
        pass

    # 评分和评论数解析
    rating = ''
    review_count = ''

    try:
        review_elem = await page.query_selector('#acrCustomerReviewText')
        if review_elem:
            review_text_raw = (await review_elem.inner_text()).strip()
            # 提取数字部分，例如 "1,234 ratings" -> "1234"
            m_rev = re.search(r'([\d,]+)', review_text_raw)
            if m_rev:
                review_count = m_rev.group(1).replace(',', '')
    except Exception as e:
        logging.debug(f"ASIN from URL {url} - Error parsing review count from #acrCustomerReviewText: {e}")
        pass

    try:
        popover = await page.query_selector('#acrPopover')
        if popover:
            title = await popover.get_attribute('title')
            if title:
                m = re.search(r'(\d+(?:[.,]\d+)?) out of 5', title.replace(',', '.')) # 兼容小数点和逗号
                if m:
                    rating = m.group(1)
    except Exception as e:
        logging.debug(f"ASIN from URL {url} - Error parsing rating from #acrPopover: {e}")
        pass

    if not rating:
        try:
            alt_elem = await page.query_selector('i.a-icon-alt')
            if alt_elem:
                text = (await alt_elem.inner_text()).strip()
                m = re.search(r'(\d+(?:[.,]\d+)?) out of 5', text.replace(',', '.'))
                if m:
                    rating = m.group(1)
        except Exception as e:
            logging.debug(f"ASIN from URL {url} - Error parsing rating from i.a-icon-alt: {e}")
            pass

    if not rating:
        try:
            td = await page.query_selector('xpath=//th[contains(normalize-space(text()),"Customer Reviews")]/following-sibling::td | //div[@id="averageCustomerReviews"]//span[contains(@class, "a-size-base")]')
            if td:
                td_text = await td.inner_text()
                m = re.search(r'(\d+(?:[.,]\d+)?) out of 5', td_text.replace(',', '.'))
                if m:
                    rating = m.group(1)
                # 如果上面没匹配到，尝试从td_text中直接找评论数（作为备用）
                if not review_count:
                    m_rev_alt = re.search(r'([\d,]+)\s+(?:ratings|customer reviews)', td_text, re.IGNORECASE)
                    if m_rev_alt:
                        review_count = m_rev_alt.group(1).replace(',', '')
            else: # 终极备用
                 customer_reviews_text_element = await page.query_selector('#averageCustomerReviews_feature_div span.a-declarative a span.a-size-base')
                 if customer_reviews_text_element:
                     text = await customer_reviews_text_element.inner_text()
                     m = re.search(r'(\d+(?:[.,]\d+)?) out of 5', text.replace(',', '.'))
                     if m:
                         rating = m.group(1)


        except Exception as e:
            logging.debug(f"ASIN from URL {url} - Error parsing rating from Customer Reviews td/averageCustomerReviews: {e}")
            pass
            
    # 再次尝试从 detailBullets 区域获取评分和评论数 (作为补充)
    if not rating or not review_count:
        try:
            detail_bullets_acr_text = await page.query_selector('#detailBullets_averageCustomerReviews #acrCustomerReviewText')
            if detail_bullets_acr_text:
                review_text_raw_db = (await detail_bullets_acr_text.inner_text()).strip()
                m_rev_db = re.search(r'([\d,]+)', review_text_raw_db)
                if m_rev_db:
                    review_count = m_rev_db.group(1).replace(',', '') # 可能会覆盖之前的，确保取到

            detail_bullets_popover = await page.query_selector('#detailBullets_averageCustomerReviews #acrPopover')
            if detail_bullets_popover:
                title_db = await detail_bullets_popover.get_attribute('title')
                if title_db:
                    m_db = re.search(r'(\d+(?:[.,]\d+)?) out of 5', title_db.replace(',', '.'))
                    if m_db:
                        rating = m_db.group(1) # 可能会覆盖之前的
        except Exception as e:
            logging.debug(f"ASIN from URL {url} - Error parsing rating/reviews from detailBullets: {e}")
            pass

    if rating:
        rating = rating.strip()
    if review_count:
        m = re.search(r'(\d+)', review_count) # 确保只有数字
        if m:
            review_count = m.group(1)
        else:
            review_count = '' # 如果正则匹配失败，说明格式不对，清空

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
    async with async_playwright() as pw:
        # 使用系统 Chrome，可复用登录态并固定 UA、视口、语言和时区
        context = await pw.chromium.launch_persistent_context(
            user_data_dir,
            headless=True,
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            user_agent=DEFAULT_USER_AGENT,
            locale='en-US',
            timezone_id='America/Los_Angeles',
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()
        # 初始化 Vine 评论抓取的进度条
        pbar = tqdm(total=len(df), desc='Vine Scraping')
        await page.set_extra_http_headers({"User-Agent": DEFAULT_USER_AGENT})
        # 登录状态已持久化，直接开始抓取 Vine 评论
        for idx, row in enumerate(df.itertuples(index=False)):
            asin = getattr(row, 'ASIN', '')
            country = getattr(row, 'country', '').upper()
            domain = DOMAIN_MAP.get(country, 'amazon.com')
            # 获取 Vine 评论总数和最后一页前三条评论平均评分
            vine_count, latest3_rating = await fetch_vine_count(page, asin, domain)
            results_list_ref[idx]['vine_count'] = vine_count
            results_list_ref[idx]['latest3_rating'] = latest3_rating
            logging.info(f"ASIN={asin} - Vine count updated: {vine_count}, latest3_rating: {latest3_rating}")
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
    async with async_playwright() as pw:
        # 使用系统 Chrome，可复用登录态并固定 UA、视口、语言和时区
        context = await pw.chromium.launch_persistent_context(
            profile_dir,
            headless=False,
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            user_agent=DEFAULT_USER_AGENT,
            locale='en-US',
            timezone_id='America/Los_Angeles',
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()
        await page.set_extra_http_headers({"User-Agent": DEFAULT_USER_AGENT})
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

async def run_scraper(df, results_list_ref, concurrency, profile_dir):
    """
    使用系统 Chrome，可复用登录态并固定 UA、视口、语言和时区
    """
    async with async_playwright() as pw:
        # 使用系统 Chrome，可复用登录态并固定 UA、视口、语言和时区
        context = await pw.chromium.launch_persistent_context(
            profile_dir,
            headless=True,
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            user_agent=DEFAULT_USER_AGENT,
            locale='en-US',
            timezone_id='America/Los_Angeles',
            viewport={'width': 1920, 'height': 1080}
        )
        pages = []
        for _ in range(concurrency):
            page = await context.new_page()
            await page.set_extra_http_headers({"User-Agent": DEFAULT_USER_AGENT})
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
            
        pbar = tqdm(total=len(df), desc='Scraping')

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

                for attempt in range(1, max_retries + 1):
                    logging.info(f"ASIN={asin} - Attempt {attempt}/{max_retries} for product data.")
                    try:
                        main_cat, main_rank, sub_cat, sub_rank, rating, reviews = await fetch_product_data(page, url)
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

        tasks = [asyncio.create_task(process_row(pos, row_data)) for pos, (_, row_data) in enumerate(df.iterrows())]
        await asyncio.gather(*tasks)
        
        pbar.close()
        logging.info("All scraping tasks completed. Closing browser.")
        for p_item in pages:
            await p_item.close()
        await context.close()

@click.command()
@click.option('--input', '-i', 'input_file', required=True, help='输入CSV/Excel文件路径，包含ASIN和country列')
@click.option('--encoding', '-e', 'encoding', default='utf-8-sig', help='输入CSV文件编码 (例如 utf-8, utf-8-sig, gbk)')
@click.option('--sep', '-s', 'sep', default=',', help='输入CSV文件分隔符 (例如 ",", "\\t", ";")')
@click.option('--concurrency', '-c', 'concurrency', default=5, type=int, help='并发任务数 (建议根据网络和机器性能调整，过高易被封)')
@click.option('--profile-dir', '-p', 'profile_dir', default='my_browser_profile', help='用于持久化浏览器登录信息的用户数据目录')
@click.option('--output', '-o', 'output_file', default='output.csv', help='输出CSV文件路径')
def main(input_file, output_file, encoding, sep, concurrency, profile_dir):
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
    try:
        out_df.to_csv(output_file, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 确保 Excel 正确打开
        print(f'数据已保存到 {output_file}')
        logging.info(f'数据已保存到 {output_file}')
    except Exception as e_save:
        print(f"错误: 无法保存结果到 '{output_file}': {e_save}")
        logging.error(f"无法保存结果到 '{output_file}': {e_save}")

    # 再跑 Vine 抓取，使用指定的用户数据目录持久化登录信息
    try:
        asyncio.run(run_vine_scraper(df, results_data, concurrency, user_data_dir))
    except Exception as e_vine:
        logging.critical(f"运行 Vine 爬虫时发生未处理的严重错误: {e_vine}")
        print(f"Vine 爬虫运行时发生严重错误: {e_vine}")
        return
    # 最终保存包括 Vine 的完整结果
    final_df = pd.DataFrame(results_data)
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'完整数据已保存至 {output_file}')

if __name__ == '__main__':
    main()