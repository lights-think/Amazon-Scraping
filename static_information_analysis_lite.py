"""
静态信息分析工具 - 轻量版（无YOLO依赖）

核心功能:
1. 直接颜色分析: 使用K-means聚类分析整张图片的主色调
2. 几何形状识别: 使用OpenCV轮廓分析、椭圆拟合、顶点检测等方法
3. 亚马逊五点描述爬取: 获取详细产品信息
4. AI文本分析: 使用Ollama的gemma3:latest模型进行材质、颜色、形状识别

优势:
- 无需YOLO模型，减少依赖和安装复杂度
- 直接分析整张图片，适用于产品图片（通常主体就是产品）
- 保持完整的AI分析能力
- 更轻量，启动更快

CSV输出格式:
sku,color,material,shape,image_color,geometric_shape,llm_color,llm_material,llm_shape

使用示例:
    # 基础使用
    python static_information_analysis_lite.py -e products.xlsx
    
    # 启用亚马逊五点描述爬取
    python static_information_analysis_lite.py -e products.xlsx --amazon
    
    # 测试模式
    python static_information_analysis_lite.py -e products.xlsx --amazon --test

环境要求:
    - Ollama服务运行中，已安装gemma3:latest模型
    - Chrome浏览器（用于亚马逊爬取）
    - OpenCV、scikit-learn等基础依赖包（无需ultralytics）

Excel文件要求:
    必需列: product_sku
    可选列: product_title_en, ASIN, country
"""

import pandas as pd
import json
import time
import random
from tqdm import tqdm
import click
import logging
import os
import re
from collections import Counter

# ========== 基础依赖 ==========
# 需要: pip install opencv-python scikit-learn openpyxl playwright
import cv2
from sklearn.cluster import KMeans
import numpy as np
from ollama import chat as ollama_chat
import asyncio
from playwright.async_api import async_playwright

# ========== 配置和常量 ==========
# 标准色RGB字典
STANDARD_COLOR_RGB = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Red": (220, 20, 60),
    "Blue": (30, 144, 255),
    "Green": (34, 139, 34),
    "Yellow": (255, 215, 0),
    "Orange": (255, 140, 0),
    "Purple": (128, 0, 128),
    "Pink": (255, 105, 180),
    "Brown": (139, 69, 19),
    "Gray": (128, 128, 128),
    "Silver": (192, 192, 192),
    "Gold": (255, 215, 0),
    "Beige": (245, 245, 220),
    "Multicolor": (127, 127, 127)
}

# 亚马逊爬取相关配置
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('static_analysis_lite.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ---------- 标准映射及工具函数 ----------
STANDARD_COLORS = [
    "Black", "White", "Red", "Blue", "Green", "Yellow", "Orange", "Purple",
    "Pink", "Brown", "Gray", "Silver", "Gold", "Beige", "Multicolor"
]

STANDARD_MATERIALS = [
    "Plastic", "Metal", "Stainless Steel", "Aluminum", "Wood", "Glass",
    "Ceramic", "Silicone", "Leather", "Fabric", "Cotton", "Paper",
    "Rubber", "Foam", "Stone"
]

STANDARD_SHAPES = [
    "Round", "Square", "Rectangular", "Oval", "Triangular", "Hexagonal",
    "Octagonal", "Heart", "Star", "Cylindrical"
]

def _standardize_attribute(value: str, standard_list: list) -> str:
    """在标准列表中查找匹配值；若未匹配则返回 'Unknown'"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unknown"
    parts = re.split(r"[;,/|]", str(value))
    matched = []
    for part in parts:
        part_clean = part.strip().lower()
        if not part_clean:
            continue
        for std in standard_list:
            if std.lower() in part_clean or part_clean in std.lower():
                if std not in matched:
                    matched.append(std)
    if matched:
        return ", ".join(matched)
    cleaned_original = ", ".join([p.strip() for p in parts if p.strip()])
    return cleaned_original if cleaned_original else "Unknown"

def standardize_color(value: str) -> str:
    return _standardize_attribute(value, STANDARD_COLORS)

def standardize_material(value: str) -> str:
    return _standardize_attribute(value, STANDARD_MATERIALS)

def standardize_shape(value: str) -> str:
    return _standardize_attribute(value, STANDARD_SHAPES)

# ========== 图像分析函数（无YOLO版本）==========

def get_dominant_color_from_image(image_path, k=3):
    """
    直接从整张图片获取主色调并映射到标准色彩（无需YOLO检测框）
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"无法读取图片: {image_path}")
            return 'Unknown'
        
        # 转换颜色空间并重塑为像素向量
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pixels = img_rgb.reshape((-1, 3))
        
        # 使用K-means聚类找到主色调
        kmeans = KMeans(n_clusters=k, n_init=3, random_state=42).fit(img_pixels)
        counts = np.bincount(kmeans.labels_)
        dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
        
        # 映射到标准色
        min_dist, best_color = float('inf'), 'Multicolor'
        for color_name, rgb in STANDARD_COLOR_RGB.items():
            dist = np.linalg.norm(dominant_color - np.array(rgb))
            if dist < min_dist:
                min_dist, best_color = dist, color_name
        
        logger.info(f"图片主色分析结果: {best_color}, RGB={dominant_color}")
        return best_color
        
    except Exception as e:
        logger.warning(f"主色识别失败: {image_path}, {e}")
        return 'Unknown'

def analyze_shape_from_image(image_path):
    """
    直接从整张图片分析几何形状（无需YOLO检测框）
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"无法读取图片: {image_path}")
            return 'Unknown'
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 如果没有明显轮廓，基于图片宽高比进行基础判断
            h, w = img.shape[:2]
            ratio = w / h if h > 0 else 1
            
            if 0.95 < ratio < 1.05:
                return 'Square'
            elif ratio >= 1.5:
                return 'Rectangular'
            else:
                return 'Oval'
        
        # 找到最大的轮廓（假设是主要物体）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓面积
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < 100:  # 面积太小，回退到宽高比
            h, w = img.shape[:2]
            ratio = w / h if h > 0 else 1
            if 0.95 < ratio < 1.05:
                return 'Square'
            elif ratio >= 1.5:
                return 'Rectangular'
            else:
                return 'Oval'
        
        # 轮廓近似
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 根据近似后的顶点数量判断形状
        vertices = len(approx)
        
        if vertices == 3:
            return 'Triangular'
        elif vertices == 4:
            # 进一步判断是正方形还是矩形
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:
                return 'Square'
            else:
                return 'Rectangular'
        elif vertices == 6:
            return 'Hexagonal'
        elif vertices == 8:
            return 'Octagonal'
        elif vertices > 8:
            # 很多顶点，可能是圆形或椭圆
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                (center, axes, orientation) = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                
                if major_axis > 0:
                    eccentricity = minor_axis / major_axis
                    if eccentricity > 0.9:  # 接近圆形
                        return 'Round'
                    else:  # 椭圆形
                        return 'Oval'
        
        # 备用：使用轮廓的紧致度判断
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = (4 * np.pi * contour_area) / (perimeter * perimeter)
            if circularity > 0.85:
                return 'Round'
            elif circularity > 0.65:
                return 'Oval'
        
        # 默认回退到基于宽高比的判断
        h, w = img.shape[:2]
        ratio = w / h if h > 0 else 1
        if 0.95 < ratio < 1.05:
            return 'Square'
        elif ratio >= 1.5:
            return 'Rectangular'
        else:
            return 'Oval'
        
    except Exception as e:
        logger.warning(f"几何形状分析失败: {image_path}, {e}")
        return 'Unknown'

# ========== AI分析函数 ==========

def create_prompt(title, bullet_points=None):
    """创建发送给AI的提示文本，结合标题和五点描述"""
    prompt = (
        "Here is a product title and bullet points. "
        "Extract its color, material and shape. "
        "Respond ONLY with JSON using keys color, material, shape. "
        "Use English words taken ONLY from the lists below. "
        "If any information is missing, put 'Unknown'.\n\n"
        f"Allowed Colors: {', '.join(STANDARD_COLORS)}\n"
        f"Allowed Materials: {', '.join(STANDARD_MATERIALS)}\n"
        f"Allowed Shapes: {', '.join(STANDARD_SHAPES)}\n\n"
        f"Title: {title}"
    )
    
    if bullet_points and len(bullet_points) > 0:
        prompt += "\n\nBullet Points:\n"
        for i, point in enumerate(bullet_points[:5], 1):  # 最多使用前5个要点
            prompt += f"{i}. {point}\n"
    
    return prompt

def call_ollama_api(prompt, max_retries=3, retry_delay=2):
    """使用ollama本地API调用gemma3:latest模型，返回JSON格式特征"""
    for attempt in range(max_retries):
        try:
            response = ollama_chat(model='gemma3:latest', messages=[{'role': 'user', 'content': prompt}])
            content = response['message']['content'] if isinstance(response, dict) else response.message.content
            
            # 只提取第一个JSON对象
            json_match = re.search(r'\{[\s\S]*?\}', content)
            if json_match:
                json_str = json_match.group(0)
                try:
                    features = json.loads(json_str)
                    # 确保返回的JSON包含所需字段
                    default_keys = {'color': 'Unknown', 'material': 'Unknown', 'shape': 'Unknown'}
                    for key in default_keys:
                        if key not in features:
                            features[key] = default_keys[key]
                    logger.info(f"Ollama结果: {features}")
                    return features
                except json.JSONDecodeError:
                    logger.warning(f"Ollama返回的JSON解析失败: {json_str}")
                    if attempt == max_retries - 1:
                        return {'color': 'Unknown', 'material': 'Unknown', 'shape': 'Unknown'}
            else:
                logger.warning(f"Ollama未返回JSON，原始内容: {content[:100]}...")
                if attempt == max_retries - 1:
                    return {'color': 'Unknown', 'material': 'Unknown', 'shape': 'Unknown'}
        except Exception as e:
            logger.warning(f"Ollama请求失败: {str(e)}")
            if attempt == max_retries - 1:
                return {'color': 'Unknown', 'material': 'Unknown', 'shape': 'Unknown'}
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    return {'color': 'Unknown', 'material': 'Unknown', 'shape': 'Unknown'}

def ai_voting_analysis(title, bullet_points=None):
    """AI投票机制分析产品特征，结合标题和五点描述"""
    prompt = create_prompt(title, bullet_points)
    
    # 第一轮
    result1 = call_ollama_api(prompt)
    time.sleep(random.uniform(1, 2))
    
    # 第二轮
    result2 = call_ollama_api(prompt)
    
    # 比较结果
    if result1 == result2:
        logger.info(f"AI投票: 两轮一致，直接输出 {result1}")
        return result1
    
    logger.info(f"AI投票: 两轮不一致，进行第三轮投票")
    logger.info(f"第一轮: {result1}")
    logger.info(f"第二轮: {result2}")
    
    time.sleep(random.uniform(1, 2))
    
    # 第三轮
    result3 = call_ollama_api(prompt)
    logger.info(f"第三轮: {result3}")
    
    # 投票决策
    final_result = {}
    for key in ['color', 'material', 'shape']:
        votes = [result1[key], result2[key], result3[key]]
        vote_count = Counter(votes)
        
        # 找到最多票数
        max_votes = max(vote_count.values())
        winners = [k for k, v in vote_count.items() if v == max_votes]
        
        if len(winners) == 1:
            # 有明确多数
            final_result[key] = winners[0]
            logger.info(f"{key}投票结果: {winners[0]} (得票{max_votes})")
        else:
            # 无多数，选择第一个结果
            final_result[key] = result1[key]
            logger.info(f"{key}投票结果: 无多数，选择第一轮结果 {result1[key]}")
    
    logger.info(f"AI投票最终结果: {final_result}")
    return final_result

# ========== 亚马逊爬取函数 ==========

async def handle_continue_shopping(page):
    """检测并自动点击亚马逊多语言"继续购物"确认页面的按钮"""
    try:
        buttons = await page.query_selector_all('button[type="submit"].a-button-text')
        for btn in buttons:
            visible = await btn.is_visible()
            enabled = await btn.is_enabled()
            if visible and enabled:
                await asyncio.sleep(random.uniform(0.8, 1.5))
                await btn.click()
                logger.info('[CONTINUE_SHOPPING] Clicked button')
                await page.wait_for_timeout(1200)
                return True
        return False
    except Exception as e:
        logger.error(f'[CONTINUE_SHOPPING] Exception: {e}')
        return False

async def extract_amazon_bullet_points(page):
    """从亚马逊页面提取五点描述"""
    bullet_points = []
    try:
        bullet_points_element = await page.query_selector("#feature-bullets")
        if bullet_points_element:
            points = await bullet_points_element.query_selector_all("ul li span.a-list-item")
            for point in points:
                point_text = await point.inner_text()
                if point_text.strip():
                    bullet_points.append(point_text.strip())
            logger.info(f"成功提取五点描述: 共{len(bullet_points)}点")
        else:
            logger.warning("未找到五点描述元素")
    except Exception as e:
        logger.error(f"提取五点描述时出错: {str(e)}")
    
    return bullet_points

async def fetch_amazon_bullet_points(asin, country='US'):
    """从亚马逊获取产品的五点描述"""
    domain = DOMAIN_MAP.get(country.upper(), 'amazon.com')
    url = f"https://www.{domain}/dp/{asin}"
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                executable_path=r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
            )
            
            page = await browser.new_page()
            await page.set_extra_http_headers({
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9"
            })
            
            logger.info(f"正在获取ASIN {asin} 的五点描述: {url}")
            await page.goto(url, timeout=60000, wait_until='domcontentloaded')
            
            # 检测并处理"继续购物"确认页面
            await handle_continue_shopping(page)
            
            # 等待页面加载完成
            await page.wait_for_selector("body", timeout=10000)
            
            # 随机延时，模拟人类访问
            await asyncio.sleep(random.uniform(1.0, 2.0))
            
            # 提取五点描述
            bullet_points = await extract_amazon_bullet_points(page)
            
            await browser.close()
            
            return bullet_points
            
    except Exception as e:
        logger.error(f"获取ASIN {asin} 五点描述时出错: {str(e)}")
        return []

# ========== 预爬取函数 ==========

async def pre_crawl_amazon_data(sku_info_map, enable_amazon_crawl, target_skus=None):
    """预先爬取需要的亚马逊五点描述数据"""
    # 检查是否有可爬取的ASIN
    has_asin = any(info.get('asin', '').strip() for info in sku_info_map.values())
    
    if not enable_amazon_crawl:
        if has_asin:
            logger.warning("检测到Excel中有ASIN信息，建议启用 --amazon 参数获取五点描述以提高分析准确性")
        logger.info("未启用亚马逊爬取功能，跳过五点描述预爬取")
        return {}
    
    if not has_asin:
        logger.warning("Excel文件中没有ASIN信息，无法进行五点描述爬取")
        return {}
    
    # 收集需要爬取的ASIN
    asin_to_crawl = {}
    skus_to_process = target_skus if target_skus is not None else sku_info_map.keys()
    
    logger.info(f"开始收集ASIN信息，目标SKU数量: {len(skus_to_process)}")
    
    for sku in skus_to_process:
        if sku not in sku_info_map:
            logger.warning(f"SKU {sku} 不在Excel数据中，跳过")
            continue
            
        info = sku_info_map[sku]
        asin = info.get('asin', '')
        country = info.get('country', 'US')
        
        if asin and asin.strip():
            asin_key = f"{asin}_{country}"
            if asin_key not in asin_to_crawl:
                asin_to_crawl[asin_key] = {
                    'asin': asin,
                    'country': country,
                    'skus': [sku]
                }
            else:
                asin_to_crawl[asin_key]['skus'].append(sku)
        else:
            logger.warning(f"SKU {sku} 没有ASIN信息，无法爬取五点描述")
    
    if not asin_to_crawl:
        logger.warning("没有找到需要爬取的ASIN信息")
        return {}
    
    logger.info(f"开始预爬取 {len(asin_to_crawl)} 个不同的ASIN五点描述...")
    
    # 批量爬取五点描述
    bullet_points_cache = {}
    crawl_errors = []
    
    for i, (asin_key, asin_info) in enumerate(asin_to_crawl.items(), 1):
        asin = asin_info['asin']
        country = asin_info['country']
        skus = asin_info['skus']
        
        try:
            logger.info(f"[{i}/{len(asin_to_crawl)}] 正在爬取ASIN {asin} ({country}) - 关联SKU: {', '.join(skus[:3])}{'...' if len(skus) > 3 else ''}")
            
            bullet_points = await fetch_amazon_bullet_points(asin, country)
            
            if bullet_points:
                bullet_points_cache[asin_key] = bullet_points
                logger.info(f"✓ ASIN {asin} 成功获取 {len(bullet_points)} 条五点描述")
            else:
                logger.warning(f"✗ ASIN {asin} 未获取到五点描述")
                crawl_errors.append(f"ASIN {asin} ({country}): 未获取到数据")
            
            # 添加延时避免被反爬虫
            if i < len(asin_to_crawl):
                await asyncio.sleep(random.uniform(2, 4))
                
        except Exception as e:
            logger.error(f"✗ ASIN {asin} 爬取失败: {e}")
            crawl_errors.append(f"ASIN {asin} ({country}): {str(e)}")
    
    # 输出爬取统计
    success_count = len(bullet_points_cache)
    total_count = len(asin_to_crawl)
    logger.info(f"五点描述预爬取完成: 成功 {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    return bullet_points_cache

# ========== 工具函数 ==========

def get_sku_from_filename(filename):
    """从文件名中提取SKU，去除扩展名"""
    return os.path.splitext(filename)[0]

def find_image_files(image_folder):
    """查找图片文件夹中的所有图片文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    if not os.path.exists(image_folder):
        logger.error(f"图片文件夹不存在: {image_folder}")
        return []
    
    for filename in os.listdir(image_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    logger.info(f"在 {image_folder} 中找到 {len(image_files)} 个图片文件")
    return image_files

def load_excel_data(excel_file):
    """读取Excel文件，返回SKU到产品信息的映射"""
    try:
        df = pd.read_excel(excel_file)
        logger.info(f"成功读取Excel文件: {excel_file}, 共 {len(df)} 条记录")
        
        # 检查必要的列
        required_columns = ['product_sku']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Excel文件缺少必要的列: {', '.join(missing_columns)}")
            return {}
        
        # 创建SKU到产品信息的映射
        sku_info_map = {}
        for _, row in df.iterrows():
            sku = str(row['product_sku']).strip()
            if sku and not pd.isna(row['product_sku']):
                info = {'sku': sku}
                
                # 提取标题
                if 'product_title_en' in df.columns and not pd.isna(row['product_title_en']):
                    info['title'] = str(row['product_title_en']).strip()
                
                # 提取ASIN
                if 'ASIN' in df.columns and not pd.isna(row['ASIN']):
                    info['asin'] = str(row['ASIN']).strip()
                
                # 提取country
                if 'country' in df.columns and not pd.isna(row['country']):
                    info['country'] = str(row['country']).strip().upper()
                else:
                    info['country'] = 'US'  # 默认美国
                
                sku_info_map[sku] = info
        
        logger.info(f"创建了 {len(sku_info_map)} 个SKU到产品信息的映射")
        return sku_info_map
        
    except Exception as e:
        logger.error(f"读取Excel文件失败: {str(e)}")
        return {}

# ========== 主程序 ==========

@click.command()
@click.option('--image-folder', '-i', default='data/产品图片', help='图片文件夹路径')
@click.option('--excel-file', '-e', required=True, help='Excel文件路径，包含product_sku列，可选ASIN、country、product_title_en列')
@click.option('--output', '-o', 'output_file', default='static_features_lite.csv', help='输出CSV文件路径')
@click.option('--batch-size', '-b', 'batch_size', default=10, type=int, help='每次处理的批次大小')
@click.option('--enable-amazon-crawl', '--amazon', is_flag=True, help='启用亚马逊五点描述爬取功能（强烈推荐）')
@click.option('--test', is_flag=True, help='测试模式，仅随机抽取50个图片进行分析')
def main(image_folder, excel_file, output_file, batch_size, enable_amazon_crawl, test):
    """静态信息分析：轻量版 - 无YOLO依赖的产品特征分析"""
    try:
        # 加载Excel数据
        sku_info_map = load_excel_data(excel_file)
        if not sku_info_map:
            logger.error("无法从Excel文件中加载数据")
            return
        
        # 查找图片文件
        logger.info("=" * 60)
        logger.info("步骤1: 扫描图片文件")
        logger.info("=" * 60)
        image_files = find_image_files(image_folder)
        if not image_files:
            logger.error("图片文件夹中没有找到图片文件")
            return
        
        # 测试模式：随机抽取50个图片
        if test:
            sample_size = min(50, len(image_files))
            image_files = random.sample(image_files, sample_size)
            logger.info(f"测试模式启用，仅抽取{sample_size}个图片进行分析")
        
        # 根据要处理的图片文件确定需要爬取的SKU
        target_skus = []
        for filename in image_files:
            sku = get_sku_from_filename(filename)
            target_skus.append(sku)
        
        logger.info(f"确定需要分析的SKU数量: {len(target_skus)}")
        
        # 预爬取亚马逊五点描述数据
        logger.info("=" * 60)
        logger.info("步骤2: 预爬取亚马逊五点描述数据")
        logger.info("=" * 60)
        bullet_points_cache = asyncio.run(pre_crawl_amazon_data(sku_info_map, enable_amazon_crawl, target_skus))
        
        # 准备结果数据
        results = []
        total_files = len(image_files)
        
        logger.info("=" * 60)
        logger.info(f"步骤3: 开始分析 {total_files} 个图片文件")
        logger.info("=" * 60)
        
        # 批量处理图片
        for i in tqdm(range(0, total_files, batch_size), desc="处理批次"):
            batch_files = image_files[i:min(i+batch_size, total_files)]
            
            for filename in batch_files:
                try:
                    # 提取SKU
                    sku = get_sku_from_filename(filename)
                    image_path = os.path.join(image_folder, filename)
                    
                    logger.info(f"处理图片: {filename}, SKU: {sku}")
                    
                    # 初始化结果
                    image_color = 'Unknown'
                    geometric_shape = 'Unknown'
                    llm_color = 'Unknown'
                    llm_material = 'Unknown'
                    llm_shape = 'Unknown'
                    
                    # 图像分析（直接分析整张图片，无需YOLO）
                    image_color = get_dominant_color_from_image(image_path)
                    geometric_shape = analyze_shape_from_image(image_path)
                    
                    logger.info(f"图像分析完成: 颜色={image_color}, 形状={geometric_shape}")
                    
                    # AI分析材质和其他信息
                    if sku in sku_info_map:
                        sku_info = sku_info_map[sku]
                        title = sku_info.get('title', '')
                        asin = sku_info.get('asin', '')
                        country = sku_info.get('country', 'US')
                        
                        bullet_points = []
                        
                        # 从预爬取的缓存中获取五点描述
                        if asin and asin.strip():
                            asin_key = f"{asin}_{country}"
                            if asin_key in bullet_points_cache:
                                bullet_points = bullet_points_cache[asin_key]
                                logger.info(f"从缓存获取ASIN {asin} 的 {len(bullet_points)} 条五点描述")
                            else:
                                logger.warning(f"缓存中未找到ASIN {asin} ({country}) 的五点描述")
                        
                        # 使用标题和五点描述进行AI分析
                        if title or bullet_points:
                            logger.info(f"开始AI分析: 标题='{title[:50]}...', 五点描述数量={len(bullet_points)}")
                            ai_result = ai_voting_analysis(title, bullet_points)
                            
                            # 记录LLM分析结果
                            llm_color = standardize_color(ai_result.get('color', 'Unknown'))
                            llm_material = standardize_material(ai_result.get('material', 'Unknown'))
                            llm_shape = standardize_shape(ai_result.get('shape', 'Unknown'))
                            
                            logger.info(f"AI分析完成: LLM颜色={llm_color}, 材质={llm_material}, 形状={llm_shape}")
                        else:
                            logger.warning(f"SKU {sku} 缺少标题和五点描述信息，无法进行AI分析")
                        
                    else:
                        logger.warning(f"SKU {sku} 在Excel文件中未找到对应信息")
                    
                    # 生成最终结果（优先级：AI > 图像分析）
                    final_color = llm_color if llm_color != 'Unknown' else image_color
                    final_material = llm_material  # 材质主要依赖AI分析
                    final_shape = llm_shape if llm_shape != 'Unknown' else geometric_shape
                    
                    # 标准化最终结果
                    final_color = standardize_color(final_color)
                    final_material = standardize_material(final_material)
                    final_shape = standardize_shape(final_shape)
                    
                    # 标准化分析结果
                    image_color = standardize_color(image_color)
                    geometric_shape = standardize_shape(geometric_shape)
                    llm_color = standardize_color(llm_color)
                    llm_material = standardize_material(llm_material)
                    llm_shape = standardize_shape(llm_shape)
                    
                    # 添加到结果
                    results.append({
                        'sku': sku,
                        # 最终结果
                        'color': final_color,
                        'material': final_material,
                        'shape': final_shape,
                        # 图像分析结果
                        'image_color': image_color,
                        'geometric_shape': geometric_shape,
                        # LLM分析结果
                        'llm_color': llm_color,
                        'llm_material': llm_material,
                        'llm_shape': llm_shape
                    })
                    
                    logger.info(f"✓ SKU {sku} 处理完成: 颜色={final_color}, 材质={final_material}, 形状={final_shape}")
                    logger.info(f"  └ 分析详情: 图像[颜色={image_color}, 形状={geometric_shape}] | LLM[颜色={llm_color}, 材质={llm_material}, 形状={llm_shape}]")
                    
                    # 减少延时
                    time.sleep(random.uniform(0.5, 1.0))
                    
                except Exception as e:
                    logger.error(f"处理图片 {filename} 时出错: {str(e)}")
                    # 即使出错也要记录基本信息
                    sku = get_sku_from_filename(filename)
                    results.append({
                        'sku': sku,
                        'color': 'Unknown',
                        'material': 'Unknown',
                        'shape': 'Unknown',
                        'image_color': 'Unknown',
                        'geometric_shape': 'Unknown',
                        'llm_color': 'Unknown',
                        'llm_material': 'Unknown',
                        'llm_shape': 'Unknown'
                    })
            
            # 每批次保存一次临时结果
            if results:
                temp_df = pd.DataFrame(results)
                temp_output = f"{os.path.splitext(output_file)[0]}_temp.csv"
                temp_df.to_csv(temp_output, index=False, encoding='utf-8-sig')
                logger.info(f"已处理 {len(results)}/{total_files} 个文件，临时保存到 {temp_output}")
        
        # 保存最终结果
        if results:
            final_df = pd.DataFrame(results)
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"分析完成，结果已保存到 {output_file}")
            
            # 删除临时文件
            temp_output = f"{os.path.splitext(output_file)[0]}_temp.csv"
            if os.path.exists(temp_output):
                os.remove(temp_output)
                logger.info(f"已删除临时文件 {temp_output}")
            
            # 打印统计信息
            logger.info(f"总共处理 {len(results)} 个SKU")
            logger.info(f"颜色分布: {final_df['color'].value_counts().to_dict()}")
            logger.info(f"材质分布: {final_df['material'].value_counts().to_dict()}")
            logger.info(f"形状分布: {final_df['shape'].value_counts().to_dict()}")
        else:
            logger.error("没有成功处理任何图片")
    
    except Exception as e:
        logger.error(f"主程序出错: {str(e)}")
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main() 