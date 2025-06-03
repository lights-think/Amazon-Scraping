import asyncio
import re
import pandas as pd
import click
from tqdm import tqdm
from playwright.async_api import async_playwright
import logging
import random

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

async def fetch_product_data(page, url):
    # 随机延时，模拟人类访问
    await asyncio.sleep(random.uniform(0.5, 1.0))
    try:
        await page.goto(url, timeout=60000, wait_until='domcontentloaded')
    except Exception as e:
        print(f"Error navigating to {url}: {e}")
    # 根据展开图标状态决定是否点击展开 BSR 区块
    try:
        expander_link = await page.query_selector('#productDetails_expanderTables_depthLeftSections > div > span > a')
        if expander_link:
            icon = await expander_link.query_selector('i')
            if icon:
                cls = await icon.get_attribute('class') or ''
                # 仅在折叠状态时点击展开
                if 'a-icon-section-expand' in cls:
                    await expander_link.click()
                    # 等待展开后的表格加载
                    await page.wait_for_selector('#productDetails_expanderTables_depthLeftSections table', timeout=5000)
    except Exception:
        pass
    # 默认值
    main_category = ''
    main_rank = ''
    sub_category = ''
    sub_rank = ''
    # BSR解析：先通过表头文本定位 Best Sellers Rank，再解析大类/小类及排名
    try:
        # 定位含 "Best Sellers Rank" 的 <th> 对应的 <td>
        bsr_td = await page.query_selector('xpath=//th[contains(text(),"Best Sellers Rank")]/following-sibling::td')
        if bsr_td:
            td_text = await bsr_td.inner_text()
        else:
            # 回退到旧选择器
            td_text = await page.inner_text('#productDetails_expanderTables_depthLeftSections table tbody tr:nth-child(4) td')
        lines = [line.strip() for line in td_text.splitlines() if line.strip()]
        # 解析大类及排名，如 "#32,726 in Office Products (... )"
        if len(lines) >= 1:
            m = re.match(r'#([\d,]+) in (.+?)(?: \(|$)', lines[0])
            if m:
                main_rank = m.group(1).replace(',', '')
                main_category = m.group(2).strip()
        # 解析小类及排名，如 "#181 in Badge Lanyards"
        if len(lines) >= 2:
            m2 = re.match(r'#([\d,]+) in (.+)', lines[1])
            if m2:
                sub_rank = m2.group(1).replace(',', '')
                sub_category = m2.group(2).strip()
    except Exception:
        pass
    # 评分和评论数解析，层级备选逻辑
    rating = ''
    review_count = ''
    # 评论数
    try:
        review_elem = await page.query_selector('#acrCustomerReviewText')
        if review_elem:
            review_count = (await review_elem.inner_text()).strip().strip('()').replace(',', '')
    except:
        pass
    # 1. 优先从 #acrPopover.title 获取 rating
    try:
        popover = await page.query_selector('#acrPopover')
        if popover:
            title = await popover.get_attribute('title')
            if title:
                m = re.search(r'(\d+(?:\.\d+)?) out of 5', title)
                if m:
                    rating = m.group(1)
    except:
        pass
    # 2. 从 i.a-icon-alt 元素获取
    if not rating:
        try:
            alt_elem = await page.query_selector('i.a-icon-alt')
            if alt_elem:
                text = (await alt_elem.inner_text()).strip()
                m = re.search(r'(\d+(?:\.\d+)?) out of 5', text)
                if m:
                    rating = m.group(1)
        except:
            pass
    # 3. 从 Customer Reviews td 整体文本获取
    if not rating:
        try:
            # 定位 Customer Reviews 对应的 td
            td = await page.query_selector('xpath=//th[contains(text(),"Customer Reviews")]/following-sibling::td')
            if td:
                td_text = await td.inner_text()
            else:
                td_text = await page.eval_on_selector('#averageCustomerReviews', 'el => el.parentElement.innerText')
            m = re.search(r'(\d+(?:\.\d+)?) out of 5', td_text)
            if m:
                rating = m.group(1)
        except:
            pass
    # BSR 备用解析：detailBullets 区域
    if not main_rank or not sub_rank:
        try:
            # 定位 detailBullets 中的 Best Sellers Rank 对应 <li>
            fallback_li = await page.query_selector(
                'xpath=//div[@id="detailBullets_feature_div"]//li[.//span[contains(text(),"Best Sellers Rank")]]'
            )
            if fallback_li:
                fb_text = await fallback_li.inner_text()
                fb_lines = [l.strip() for l in fb_text.splitlines() if l.strip()]
                # 主分类排名
                if fb_lines:
                    m = re.search(r'#([\d,]+) in (.+?)(?: \(|$)', fb_lines[0])
                    if m:
                        main_rank = m.group(1).replace(',', '')
                        main_category = m.group(2).strip()
                # 子分类排名
                if len(fb_lines) >= 2:
                    m2 = re.search(r'#([\d,]+) in (.+)', fb_lines[1])
                    if m2:
                        sub_rank = m2.group(1).replace(',', '')
                        sub_category = m2.group(2).strip()
        except Exception:
            pass
    # 清洗 rating
    if rating:
        rating = rating.strip()
    # 评分备用解析：detailBullets 区域
    if not rating or not review_count:
        try:
            # 文本评论数
            rb_elem = await page.query_selector('#detailBullets_averageCustomerReviews #acrCustomerReviewText')
            if rb_elem:
                review_count = (await rb_elem.inner_text()).strip().strip('()').replace(',', '')
            # 弹出框评分
            pop_el = await page.query_selector('#detailBullets_averageCustomerReviews #acrPopover')
            if pop_el:
                t = await pop_el.get_attribute('title')
                if t:
                    m = re.search(r'(\d+(?:\.\d+)?) out of 5', t)
                    if m:
                        rating = m.group(1)
        except Exception:
            pass
    # 清洗 review_count，只保留数字
    if review_count:
        m = re.search(r'(\d+)', review_count)
        if m:
            review_count = m.group(1)
    return main_category, main_rank, sub_category, sub_rank, rating, review_count

async def run_scraper(df, results, concurrency):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        USER_AGENT_LIST = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"
        ]
        # 预创建页面池并复用，设置随机 UA 并拦截静态资源
        pages = []
        for _ in range(concurrency):
            page = await browser.new_page()
            # 随机 User-Agent
            ua = random.choice(USER_AGENT_LIST)
            await page.set_extra_http_headers({"User-Agent": ua})
            # 拦截静态资源
            async def _intercept_route(route):
                if route.request.resource_type in ["image", "stylesheet", "font"]:
                    await route.abort()
                else:
                    await route.continue_()
            await page.route("**/*", _intercept_route)
            pages.append(page)
        page_queue = asyncio.Queue()
        for p in pages:
            await page_queue.put(p)
        pbar = tqdm(total=len(df), desc='Scraping')
        # 定义并发处理函数
        async def process_row(row):
            # 从池中获取页面
            page = await page_queue.get()
            asin = str(row.get('ASIN', '')).strip()
            country = str(row.get('country', '')).strip().upper()
            domain = DOMAIN_MAP.get(country, 'amazon.com')
            url = f'https://www.{domain}/dp/{asin}'
            logging.info(f"开始抓取 ASIN={asin}, URL={url}")
            try:
                main_cat, main_rank, sub_cat, sub_rank, rating, reviews = await fetch_product_data(page, url)
                if not main_rank or not sub_rank:
                    logging.warning(f"ASIN={asin} 未获取到 BSR (主:{main_rank}, 子:{sub_rank})")
                else:
                    logging.info(f"ASIN={asin} BSR 主:{main_rank}/{main_cat}, 子:{sub_rank}/{sub_cat}")
                logging.info(f"ASIN={asin} rating={rating}, reviews={reviews}")
                results.append({
                    'ASIN': asin,
                    'country': country,
                    'url': url,
                    'bsr_main_category': main_cat,
                    'bsr_main_rank': main_rank,
                    'bsr_sub_category': sub_cat,
                    'bsr_sub_rank': sub_rank,
                    'rating': rating,
                    'review_count': reviews
                })
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
            finally:
                # 归还页面到池中
                await page_queue.put(page)
            pbar.update(1)
        # 并发执行所有任务（任务数不限，由页面池大小控制并发）
        tasks = [asyncio.create_task(process_row(row)) for _, row in df.iterrows()]
        await asyncio.gather(*tasks)
        # 关闭所有页面和浏览器
        for p in pages:
            await p.close()
        await browser.close()
        pbar.close()

@click.command()
@click.option('--input', '-i', 'input_file', required=True, help='输入CSV文件路径，包含ASIN和country列')
@click.option('--encoding', '-e', 'encoding', default='utf-8', help='输入CSV文件编码，比如 utf-8、gbk')
@click.option('--sep', '-s', 'sep', default=',', help='输入CSV文件分隔符，例如 ,、\\t、; 等')
@click.option('--concurrency', '-c', 'concurrency', default=10, help='并发任务数')
@click.option('--output', '-o', 'output_file', default='output.csv', help='输出CSV文件路径')
def main(input_file, output_file, encoding, sep, concurrency):
    """
    入口函数：读取输入CSV，运行爬虫并保存结果到输出CSV
    """
    # 读取输入文件，支持 CSV 和 Excel (.xls/.xlsx)
    if input_file.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(input_file)
    else:
        # 处理分隔符转义
        sep_char = '\t' if sep == '\t' else sep
        df = pd.read_csv(input_file, encoding=encoding, sep=sep_char)
    results = []
    # 异步执行爬虫
    asyncio.run(run_scraper(df, results, concurrency))
    # 保存结果
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, index=False)
    print(f'数据已保存到 {output_file}')

if __name__ == '__main__':
    main() 