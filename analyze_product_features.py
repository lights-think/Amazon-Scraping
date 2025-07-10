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

@click.command()
@click.option('--input', '-i', 'input_file', default='basic_info_output.csv', help='输入CSV文件路径，包含产品标题和描述')
@click.option('--output', '-o', 'output_file', default='product_features.csv', help='输出CSV文件路径')
@click.option('--batch-size', '-b', 'batch_size', default=10, type=int, help='每次处理的批次大小')
@click.option('--start-index', '-s', 'start_index', default=0, type=int, help='开始处理的索引位置')
@click.option('--end-index', '-e', 'end_index', default=-1, type=int, help='结束处理的索引位置，-1表示处理到末尾')
def main(input_file, output_file, batch_size, start_index, end_index):
    """分析产品特征并提取颜色、材质和形状"""
    try:
        # 读取输入文件
        if not os.path.exists(input_file):
            logger.error(f"输入文件 {input_file} 不存在")
            return
        
        df = pd.read_csv(input_file)
        logger.info(f"成功读取输入文件: {input_file}, 共 {len(df)} 条记录")
        
        # 检查必要的列
        required_columns = ['ASIN', 'country', 'title', 'bullet_points']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"输入文件缺少必要的列: {', '.join(missing_columns)}")
            return
        
        # 处理索引范围
        if end_index == -1 or end_index >= len(df):
            end_index = len(df)
        
        # 只处理指定范围内的记录
        df_to_process = df.iloc[start_index:end_index].copy()
        total = len(df_to_process)
        logger.info(f"将处理 {total} 条记录，从索引 {start_index} 到 {end_index-1}")
        
        # 创建结果列
        df_to_process['color'] = 'Unknown'
        df_to_process['material'] = 'Unknown'
        df_to_process['shape'] = 'Unknown'
        
        # 批量处理记录
        for i in tqdm(range(0, total, batch_size), desc="处理批次"):
            batch = df_to_process.iloc[i:min(i+batch_size, total)]
            
            for idx, row in batch.iterrows():
                try:
                    title = str(row['title']) if not pd.isna(row['title']) else ""
                    bullet_points = str(row['bullet_points']) if not pd.isna(row['bullet_points']) else ""

                    # ---------------- 先从 overview_* 或 JSON 中提取原始值 ----------------
                    raw_color = None
                    raw_material = None
                    raw_shape = None

                    # 1) 直接列
                    if 'overview_color' in df_to_process.columns and not pd.isna(row['overview_color']):
                        raw_color = str(row['overview_color']).strip()
                    if 'overview_material' in df_to_process.columns and not pd.isna(row['overview_material']):
                        raw_material = str(row['overview_material']).strip()
                    if 'overview_shape' in df_to_process.columns and not pd.isna(row['overview_shape']):
                        raw_shape = str(row['overview_shape']).strip()

                    # 2) product_overview JSON
                    if (raw_color is None or raw_color == "") and 'product_overview' in row:
                        raw_color = _extract_from_overview(row['product_overview'], COLOR_KEYS)
                    if (raw_material is None or raw_material == "") and 'product_overview' in row:
                        raw_material = _extract_from_overview(row['product_overview'], MATERIAL_KEYS)
                    if (raw_shape is None or raw_shape == "") and 'product_overview' in row:
                        raw_shape = _extract_from_overview(row['product_overview'], SHAPE_KEYS)

                    # ---------------- 标准化 ----------------
                    color_val = standardize_color(raw_color)
                    material_val = standardize_material(raw_material)
                    shape_val = standardize_shape(raw_shape)

                    # ========== 新增：如颜色非标准色或包含非颜色信息，直接用YOLO识别主色 ==========
                    color_is_standard = color_val != "Unknown" and color_val.lower() in [c.lower() for c in STANDARD_COLORS]
                    # 检查是否包含数字、单位、逗号、空格等非颜色信息
                    import re
                    color_has_noise = bool(re.search(r'\d|cm|mm|inch|x|\s|,|\.|-', color_val, re.IGNORECASE))
                    if (not color_is_standard) or color_has_noise:
                        color_val = "Unknown"

                    # ========== YOLO识别主图主体颜色/形状 ==========
                    if (color_val == "Unknown" or shape_val == "Unknown") and 'main_image' in row and isinstance(row['main_image'], str) and row['main_image'].startswith('http'):
                        # 确保模型已加载（首次会自动下载）
                        _ = get_yolo_model()
                        img_path = download_image(row['main_image'])
                        if img_path:
                            crop, class_id, box = yolo_detect_main_object(img_path)
                            if crop is not None:
                                if color_val == "Unknown":
                                    logger.info(f"YOLO主色分析: {img_path}")
                                    color_val = get_dominant_color(crop)
                                if shape_val == "Unknown":
                                    logger.info(f"YOLO形状推断: {img_path}")
                                    shape_val = infer_shape_from_box(box, class_id)
                            else:
                                logger.warning(f"YOLO未能识别出主主体，图片: {img_path}")
                            try:
                                os.remove(img_path)
                                logger.info(f"已删除临时图片: {img_path}")
                            except Exception as e:
                                logger.warning(f"删除临时图片失败: {img_path}, {e}")

                    # ========== AI兜底 ==========
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

                    logger.info(
                        f"成功处理 ASIN {row['ASIN']}: 颜色={color_val}, 材质={material_val}, 形状={shape_val}"
                    )
                    
                    # 添加随机延时，限制请求速度
                    sleep_time = random.uniform(1, 3)
                    logger.info(f"请求限速，等待 {sleep_time:.2f} 秒...")
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"处理 ASIN {row.get('ASIN', 'Unknown')} 时出错: {str(e)}")
            
            # 每批次保存一次，防止中断丢失数据
            temp_output = f"{os.path.splitext(output_file)[0]}_temp.csv"
            df_to_process.to_csv(temp_output, index=False, encoding='utf-8-sig')
            logger.info(f"已处理 {min(i+batch_size, total)}/{total} 条记录，临时保存到 {temp_output}")
        
        # 如果是处理全部数据，直接保存为最终输出
        if start_index == 0 and end_index == len(df):
            # 只保留需要的列
            result_df = df_to_process[['ASIN', 'country', 'color', 'material', 'shape']]
            result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
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
                        new_row = row[['ASIN', 'country', 'color', 'material', 'shape']]
                        existing_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
                
                existing_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            else:
                # 如果输出文件不存在，直接保存处理的部分
                result_df = df_to_process[['ASIN', 'country', 'color', 'material', 'shape']]
                result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"已将结果合并保存到 {output_file}")
        
        # 删除临时文件
        temp_output = f"{os.path.splitext(output_file)[0]}_temp.csv"
        if os.path.exists(temp_output):
            os.remove(temp_output)
            logger.info(f"已删除临时文件 {temp_output}")
    
    except Exception as e:
        logger.error(f"主程序出错: {str(e)}")
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main() 