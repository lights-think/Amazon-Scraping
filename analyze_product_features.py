import pandas as pd
import requests
import json
import time
import random  # 添加random模块导入
from tqdm import tqdm
import click
import logging
import os
import re  # 用于正则分割实现标准化
import multiprocessing
from multiprocessing import Process, Manager, Queue, Value

# ========== 新增依赖说明 ==========
# 需要: pip install ultralytics opencv-python scikit-learn
from ultralytics import YOLO
import cv2
import tempfile
import shutil
from sklearn.cluster import KMeans
import numpy as np
from ollama import chat as ollama_chat

# ========== YOLO辅助函数 ==========
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

# 全局YOLO模型变量
YOLO_MODEL = None

def get_yolo_model():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = YOLO('yolov8s.pt')  # 使用最新小模型
    return YOLO_MODEL

def download_image(url):
    try:
        import requests
        logger.info(f"开始下载图片: {url}")
        resp = requests.get(url, timeout=10, stream=True)
        if resp.status_code == 200:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            with open(tmp.name, 'wb') as f:
                shutil.copyfileobj(resp.raw, f)
            logger.info(f"图片下载成功: {tmp.name}")
            return tmp.name
        else:
            logger.warning(f"图片下载失败，状态码: {resp.status_code}, url: {url}")
    except Exception as e:
        logger.warning(f"图片下载失败: {url}, {e}")
    return None

def yolo_detect_main_object(image_path):
    try:
        logger.info(f"开始YOLO分析图片: {image_path}")
        model = get_yolo_model()
        results = model(image_path)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
        confs = results[0].boxes.conf.cpu().numpy() if results and results[0].boxes is not None else []
        clss = results[0].boxes.cls.cpu().numpy() if results and results[0].boxes is not None else []
        img = cv2.imread(image_path)
        if len(boxes) == 0:
            logger.warning(f"YOLO未检测到任何目标: {image_path}")
            # 用整图做主色和形状分析
            if img is not None:
                h, w = img.shape[:2]
                box = [0, 0, w, h]
                class_id = None
                logger.info(f"用整图推断形状: box={box}")
                return img, class_id, box
            else:
                logger.warning(f"cv2无法读取图片: {image_path}")
                return None, None, None
        # 取最大面积box
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        idx = int(np.argmax(areas))
        box = boxes[idx]
        class_id = int(clss[idx])
        conf = confs[idx]
        logger.info(f"YOLO检测到目标: class_id={class_id}, box={box}, conf={conf}")
        if img is None:
            logger.warning(f"cv2无法读取图片: {image_path}")
            return None, None, None
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        return crop, class_id, box
    except Exception as e:
        logger.warning(f"YOLO识别失败: {image_path}, {e}")
        return None, None, None

def get_dominant_color(image, k=3):
    try:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k, n_init=3, random_state=42).fit(img)
        counts = np.bincount(kmeans.labels_)
        dom_color = kmeans.cluster_centers_[np.argmax(counts)]
        # 映射到标准色
        min_dist, best_color = float('inf'), 'Multicolor'
        for cname, rgb in STANDARD_COLOR_RGB.items():
            dist = np.linalg.norm(dom_color - np.array(rgb))
            if dist < min_dist:
                min_dist, best_color = dist, cname
        return best_color
    except Exception as e:
        logger.warning(f"主色识别失败: {e}")
        return 'Unknown'

YOLO_CYLINDRICAL_CLASSES = {39, 41, 46}  # 39:bottle, 41:cup, 46:can (COCO)
YOLO_CYLINDRICAL_NAMES = {'bottle', 'cup', 'can'}

def infer_shape_from_box(box, class_id):
    # YOLO类别可用COCO80类，部分可推断形状
    # box: [x1, y1, x2, y2]
    # class_id: int or None
    if class_id is not None:
        # 优先类别判断
        if class_id in YOLO_CYLINDRICAL_CLASSES:
            return 'Cylindrical'
    if box is None:
        return 'Unknown'
    w = box[2] - box[0]
    h = box[3] - box[1]
    ratio = w / h if h > 0 else 1
    if 0.9 < ratio < 1.1:
        return 'Square'
    elif ratio >= 1.1 or ratio <= 0.8:
        return 'Rectangular'
    else:
        return 'Unknown'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('analyze_features.log', encoding='utf-8'),
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
    """
    在标准列表中查找匹配值；若未匹配则返回 'Unknown'。
    支持输入包含逗号/分号/斜杠等分隔的多值。
    """
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
    # 若未匹配任何标准值，但原始字符串非空，则直接返回清洗后的原始值
    cleaned_original = ", ".join([p.strip() for p in parts if p.strip()])
    return cleaned_original if cleaned_original else "Unknown"

def standardize_color(value: str) -> str:
    return _standardize_attribute(value, STANDARD_COLORS)

def standardize_material(value: str) -> str:
    return _standardize_attribute(value, STANDARD_MATERIALS)

def standardize_shape(value: str) -> str:
    return _standardize_attribute(value, STANDARD_SHAPES)
# ---------- End ----------

# ---------- Product Overview 提取辅助 ----------
COLOR_KEYS = ["Color", "Colour", "颜色"]
MATERIAL_KEYS = ["Material", "材质", "材料"]
SHAPE_KEYS = ["Shape", "形状", "风格", "Style"]


def _extract_from_overview(overview_json_str: str, key_candidates: list):
    """从 JSON 字符串中提取候选键对应的值"""
    if not overview_json_str or overview_json_str == '{}' or pd.isna(overview_json_str):
        return None
    try:
        data = json.loads(overview_json_str)
        for k in key_candidates:
            # 不区分大小写匹配
            for actual_key in data.keys():
                if actual_key.lower() == k.lower():
                    return str(data[actual_key]).strip()
    except Exception:
        pass
    return None
# ---------- End ----------

# Groq API配置
GROQ_API_KEY = ""
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "qwen/qwen3-32b"

def call_groq_api(prompt, max_retries=3, retry_delay=2):
    """
    调用Groq API提取产品特征
    
    Args:
        prompt (str): 发送给API的提示文本
        max_retries (int): 最大重试次数
        retry_delay (int): 重试间隔秒数
    
    Returns:
        dict: 包含颜色、材质和形状的字典
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_message = (
        "You are a professional product feature analysis assistant. "
        "Extract color, material and shape from the product title and description. "
        "Choose only from the following lists and respond in English.\n"
        f"Colors: {', '.join(STANDARD_COLORS)}\n"
        f"Materials: {', '.join(STANDARD_MATERIALS)}\n"
        f"Shapes: {', '.join(STANDARD_SHAPES)}\n"
        "Return ONLY JSON with keys color, material, shape. "
        "If information is missing, output 'Unknown'."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            
            # 尝试解析JSON响应
            try:
                features = json.loads(content)
                # 确保返回的JSON包含所需字段
                if not all(key in features for key in ['color', 'material', 'shape']):
                    default_keys = {'color': 'Unknown', 'material': 'Unknown', 'shape': 'Unknown'}
                    for key in default_keys:
                        if key not in features:
                            features[key] = default_keys[key]
                return features
            except json.JSONDecodeError:
                logger.warning(f"API返回的不是有效JSON: {content}")
                if attempt == max_retries - 1:
                    return {'color': 'Unknown', 'material': 'Unknown', 'shape': 'Unknown'}
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"API请求失败: {str(e)}")
            if attempt == max_retries - 1:
                return {'color': 'Unknown', 'material': 'Unknown', 'shape': 'Unknown'}
        
        # 如果不是最后一次尝试，等待后重试
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    return {'color': 'Unknown', 'material': 'Unknown', 'shape': 'Unknown'}

def create_prompt(title, bullet_points):
    """
    创建发送给API的提示文本
    
    Args:
        title (str): 产品标题
        bullet_points (str): 产品五点描述
    
    Returns:
        str: 格式化的提示文本
    """
    prompt = (
        "Here are a product title and description. "
        "Extract its color, material and shape. "
        "Respond ONLY with JSON using keys color, material, shape. "
        "Use English words taken ONLY from the lists below. "
        "If any information is missing, put 'Unknown'.\n\n"
        f"Allowed Colors: {', '.join(STANDARD_COLORS)}\n"
        f"Allowed Materials: {', '.join(STANDARD_MATERIALS)}\n"
        f"Allowed Shapes: {', '.join(STANDARD_SHAPES)}\n\n"
        f"Title: {title}\n\nDescription:\n{bullet_points}"
    )
    
    return prompt

def call_ollama_api(prompt, max_retries=3, retry_delay=2):
    """
    使用ollama本地API调用qwen3:latest模型，返回JSON格式特征，自动过滤思考过程。
    """
    import re
    for attempt in range(max_retries):
        try:
            response = ollama_chat(model='qwen3:latest', messages=[{'role': 'user', 'content': prompt}])
            content = response['message']['content'] if isinstance(response, dict) else response.message.content
            # 只提取第一个JSON对象
            json_match = re.search(r'\{[\s\S]*?\}', content)
            if json_match:
                json_str = json_match.group(0)
                try:
                    features = json.loads(json_str)
                    # 确保返回的JSON包含所需字段
                    if not all(key in features for key in ['color', 'material', 'shape']):
                        default_keys = {'color': 'Unknown', 'material': 'Unknown', 'shape': 'Unknown'}
                        for key in default_keys:
                            if key not in features:
                                features[key] = default_keys[key]
                    logger.info(f"Ollama最终结果: {features}")
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

def analyze_single_product(row_data, result_queue, progress_counter=None):
    """
    分析单个产品的特征（color, material, shape）
    """
    try:
        asin = row_data.get('ASIN', 'Unknown')
        title = str(row_data.get('title', '')) if not pd.isna(row_data.get('title')) else ""
        bullet_points = str(row_data.get('bullet_points', '')) if not pd.isna(row_data.get('bullet_points')) else ""
        product_overview_str = str(row_data.get('product_overview', '{}')) if not pd.isna(row_data.get('product_overview')) else "{}"
        main_image = str(row_data.get('main_image', '')) if not pd.isna(row_data.get('main_image')) else ""

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

        # 准备结果
        result = {
            'index': row_data.get('index'),
            'color': color_val,
            'material': material_val,
            'shape': shape_val
        }
        
        logger.info(f"完成分析 ASIN {asin}: 颜色={color_val}, 材质={material_val}, 形状={shape_val}")
        
        # 放入结果队列
        result_queue.put(result)
        
        # 更新进度计数器
        if progress_counter is not None:
            try:
                with progress_counter.get_lock():
                    progress_counter.value += 1
            except AttributeError:
                progress_counter.value += 1
                
    except Exception as e:
        logger.error(f"分析 ASIN {row_data.get('ASIN', 'Unknown')} 时出错: {str(e)}")
        # 放入错误结果
        result_queue.put({
            'index': row_data.get('index'),
            'color': 'Unknown',
            'material': 'Unknown',
            'shape': 'Unknown',
            'ERROR': str(e)
        })
        
        # 仍然更新进度计数器
        if progress_counter is not None:
            try:
                with progress_counter.get_lock():
                    progress_counter.value += 1
            except AttributeError:
                progress_counter.value += 1


def analyze_worker_process(task_queue, result_queue, progress_counter, batch_size=5, sleep_time=1):
    """分析特征的子进程执行函数"""
    while True:
        try:
            # 从任务队列获取一批数据
            batch_data = []
            for _ in range(batch_size):
                if task_queue.empty():
                    break
                item = task_queue.get()
                if item is None:  # 结束信号
                    task_queue.put(None)  # 放回结束信号给其他进程
                    return
                batch_data.append(item)
            
            if not batch_data:
                # 队列为空，等待一会再试
                time.sleep(0.1)
                continue
                
            # 处理批次中的每个项目
            for row_data in batch_data:
                analyze_single_product(row_data, result_queue, progress_counter)
                
            # 批次间休息
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"分析进程发生异常: {e}")
            time.sleep(1)  # 出错后休息一下再继续


def analyze_features_multiprocess(df, analyze_processes=2, analyze_batch_size=10, analyze_sleep=2):
    """
    多进程分析产品特征（color, material, shape）
    """
    if df.empty:
        logger.info("没有需要分析的产品")
        return df
        
    logger.info(f"开始多进程分析 {len(df)} 个产品的特征，进程数={analyze_processes}")
    
    # 添加索引列以便后续更新
    df = df.reset_index(drop=False)
    
    # 创建共享对象
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    progress_counter = manager.Value('i', 0)
    
    # 将所有任务放入队列
    for _, row in df.iterrows():
        task_queue.put(row.to_dict())
    
    # 添加结束信号
    task_queue.put(None)
    
    # 启动分析进程
    processes = []
    for i in range(analyze_processes):
        p = Process(target=analyze_worker_process, 
                   args=(task_queue, result_queue, progress_counter, analyze_batch_size, analyze_sleep))
        p.daemon = True
        p.start()
        processes.append(p)
        logger.info(f"启动分析子进程 {p.pid}")
    
    # 显示总进度
    total_items = len(df)
    pbar = tqdm(total=total_items, desc='特征分析进度')
    
    # 收集结果
    results = []
    completed = 0
    
    while completed < total_items:
        # 更新进度条
        try:
            with progress_counter.get_lock():
                current = progress_counter.value
        except AttributeError:
            current = progress_counter.value
            
        pbar.update(current - pbar.n)
        
        # 收集已完成的结果
        while not result_queue.empty():
            result = result_queue.get()
            results.append(result)
            completed += 1
            
        # 检查是否所有进程都还活着
        if not any(p.is_alive() for p in processes) and completed < total_items:
            logger.error("所有分析进程都已结束，但任务未完成")
            break
            
        time.sleep(0.5)
    
    # 最终更新进度条
    pbar.update(total_items - pbar.n)
    pbar.close()
    
    # 等待所有进程结束
    for p in processes:
        p.join(timeout=1)
        if p.is_alive():
            p.terminate()
    
    logger.info(f"分析阶段完成，收集了 {len(results)} 条结果")
    
    # 更新原始DataFrame
    for result in results:
        if 'index' in result and result['index'] in df.index:
            idx = result['index']
            if 'color' in result:
                df.at[idx, 'color'] = result['color']
            if 'material' in result:
                df.at[idx, 'material'] = result['material']
            if 'shape' in result:
                df.at[idx, 'shape'] = result['shape']
    
    # 删除临时索引列
    if 'index' in df.columns:
        df = df.drop('index', axis=1)
        
    return df

@click.command()
@click.option('--input', '-i', 'input_file', default='temp/spider_raw_output.csv', help='输入CSV文件路径，包含产品原始数据')
@click.option('--output', '-o', 'output_file', default='temp/product_features_analyzed.csv', help='输出CSV文件路径')
@click.option('--batch-size', '-b', 'batch_size', default=10, type=int, help='每次处理的批次大小')
@click.option('--start-index', '-s', 'start_index', default=0, type=int, help='开始处理的索引位置')
@click.option('--end-index', '-e', 'end_index', default=-1, type=int, help='结束处理的索引位置，-1表示处理到末尾')
@click.option('--processes', '-p', 'processes', default=2, type=int, help='分析进程数')
@click.option('--sleep-time', '-t', 'sleep_time', default=2, type=int, help='批次间隔秒数')
@click.option('--use-multiprocess', '-m', 'use_multiprocess', is_flag=True, help='使用多进程并行分析')
def main(input_file, output_file, batch_size, start_index, end_index, processes, sleep_time, use_multiprocess):
    """分析产品特征并提取颜色、材质和形状"""
    try:
        # 读取输入文件
        if not os.path.exists(input_file):
            logger.error(f"输入文件 {input_file} 不存在")
            return
        
        df = pd.read_csv(input_file)
        logger.info(f"成功读取输入文件: {input_file}, 共 {len(df)} 条记录")
        
        # 检查必要的列（兼容all_in_one_spider.py的输出格式）
        required_columns = ['ASIN', 'country', 'title', 'bullet_points']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"输入文件缺少必要的列: {', '.join(missing_columns)}")
            return
        
        # 确保可选列存在（这些列可能来自all_in_one_spider.py）
        optional_columns = ['product_overview', 'main_image']
        for col in optional_columns:
            if col not in df.columns:
                df[col] = ''
                logger.info(f"添加缺失的可选列: {col}")
        
        # 处理索引范围
        if end_index == -1 or end_index >= len(df):
            end_index = len(df)
        
        # 只处理指定范围内的记录
        df_to_process = df.iloc[start_index:end_index].copy()
        total = len(df_to_process)
        logger.info(f"将处理 {total} 条记录，从索引 {start_index} 到 {end_index-1}")
        
        # 检查哪些记录需要分析
        for col in ['color', 'material', 'shape']:
            if col not in df_to_process.columns:
                df_to_process[col] = 'Unknown'
        
        # 筛选需要分析的记录
        needs_analysis = df_to_process[
            df_to_process['color'].isna() | 
            df_to_process['material'].isna() | 
            df_to_process['shape'].isna() |
            (df_to_process['color'] == '') |
            (df_to_process['material'] == '') |
            (df_to_process['shape'] == '') |
            (df_to_process['color'] == 'Unknown') |
            (df_to_process['material'] == 'Unknown') |
            (df_to_process['shape'] == 'Unknown')
        ].copy()
        
        if needs_analysis.empty:
            logger.info("所有记录都已完成特征分析")
        else:
            logger.info(f"发现 {len(needs_analysis)} 条记录需要特征分析")
            
            if use_multiprocess and len(needs_analysis) > 5:
                # 使用多进程分析
                logger.info(f"启动多进程分析，进程数: {processes}")
                analyzed_df = analyze_features_multiprocess(
                    needs_analysis, 
                    analyze_processes=processes,
                    analyze_batch_size=batch_size, 
                    analyze_sleep=sleep_time
                )
                
                # 更新原始DataFrame
                for idx, row in analyzed_df.iterrows():
                    if idx in df_to_process.index:
                        df_to_process.at[idx, 'color'] = row['color']
                        df_to_process.at[idx, 'material'] = row['material']
                        df_to_process.at[idx, 'shape'] = row['shape']
            else:
                # 使用单进程批量分析
                logger.info("使用单进程批量分析")
                for i in tqdm(range(0, len(needs_analysis), batch_size), desc="处理批次"):
                    batch = needs_analysis.iloc[i:min(i+batch_size, len(needs_analysis))]
                    
                    for idx, row in batch.iterrows():
                        try:
                            title = str(row['title']) if not pd.isna(row['title']) else ""
                            bullet_points = str(row['bullet_points']) if not pd.isna(row['bullet_points']) else ""
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
                                    _ = get_yolo_model()
                                    img_path = download_image(main_image)
                                    if img_path:
                                        crop, class_id, box = yolo_detect_main_object(img_path)
                                        if crop is not None:
                                            if color_val == "Unknown":
                                                logger.info(f"YOLO主色分析: {img_path}")
                                                color_val = get_dominant_color(crop)
                                            if shape_val == "Unknown":
                                                logger.info(f"YOLO形状推断: {img_path}")
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
                                    color_val = standardize_color(features.get('color'))
                                if material_val == "Unknown":
                                    material_val = standardize_material(features.get('material'))
                                if shape_val == "Unknown":
                                    shape_val = standardize_shape(features.get('shape'))

                            # 更新DataFrame
                            df_to_process.at[idx, 'color'] = color_val
                            df_to_process.at[idx, 'material'] = material_val
                            df_to_process.at[idx, 'shape'] = shape_val

                            logger.info(f"成功处理 ASIN {row['ASIN']}: 颜色={color_val}, 材质={material_val}, 形状={shape_val}")
                            
                            # 添加随机延时，限制请求速度
                            sleep_time_random = random.uniform(1, 3)
                            time.sleep(sleep_time_random)
                            
                        except Exception as e:
                            logger.error(f"处理 ASIN {row.get('ASIN', 'Unknown')} 时出错: {str(e)}")
                            df_to_process.at[idx, 'color'] = 'Unknown'
                            df_to_process.at[idx, 'material'] = 'Unknown'
                            df_to_process.at[idx, 'shape'] = 'Unknown'
                    
                    # 批次间休息
                    if i + batch_size < len(needs_analysis):
                        logger.info(f"批次间休息 {sleep_time} 秒...")
                        time.sleep(sleep_time)
        
        # 如果是处理全部数据，直接保存为最终输出
        if start_index == 0 and end_index == len(df):
            output_df = df_to_process.copy()
            output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"已将结果保存到 {output_file}")
        else:
            # 如果是部分处理，需要合并结果
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                # 更新已存在的记录
                for idx, row in df_to_process.iterrows():
                    asin = row['ASIN']
                    country = row['country']
                    mask = (existing_df['ASIN'] == asin) & (existing_df['country'] == country)
                    if mask.any():
                        existing_df.loc[mask, ['color', 'material', 'shape']] = row[['color', 'material', 'shape']].values
                    else:
                        # 添加新记录
                        new_row = row
                        existing_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
                
                existing_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            else:
                # 如果输出文件不存在，直接保存处理的部分
                df_to_process.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"已将结果合并保存到 {output_file}")
        
        print(f"=== 特征分析完成！结果已保存到 {os.path.abspath(output_file)} ===")
        print(f"共处理 {len(df_to_process)} 条记录")
        logger.info(f"特征分析任务完成，结果保存到 {os.path.abspath(output_file)}")
    
    except Exception as e:
        logger.error(f"主程序出错: {str(e)}")
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main() 