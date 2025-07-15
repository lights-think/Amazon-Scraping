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

# ========== 依赖说明 ==========
# 需要: pip install ultralytics opencv-python scikit-learn openpyxl
from ultralytics import YOLO
import cv2
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

def yolo_detect_largest_object(image_path):
    """
    使用YOLO检测图片中的所有物体，返回占比最大的物体区域
    """
    try:
        logger.info(f"开始YOLO分析图片: {image_path}")
        model = get_yolo_model()
        results = model(image_path)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
        confs = results[0].boxes.conf.cpu().numpy() if results and results[0].boxes is not None else []
        clss = results[0].boxes.cls.cpu().numpy() if results and results[0].boxes is not None else []
        
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"cv2无法读取图片: {image_path}")
            return None, None, None
            
        img_height, img_width = img.shape[:2]
        total_area = img_height * img_width
        
        if len(boxes) == 0:
            logger.warning(f"YOLO未检测到任何目标: {image_path}")
            # 用整图做主色和形状分析
            box = [0, 0, img_width, img_height]
            class_id = None
            logger.info(f"用整图推断: box={box}")
            return img, class_id, box
        
        # 计算所有检测框的面积占比，选择最大的
        max_area = 0
        best_idx = 0
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / total_area
            
            if area > max_area:
                max_area = area
                best_idx = i
        
        # 选择占比最大的目标
        box = boxes[best_idx]
        class_id = int(clss[best_idx])
        conf = confs[best_idx]
        area_ratio = max_area / total_area
        
        logger.info(f"YOLO检测到最大目标: class_id={class_id}, box={box}, conf={conf}, area_ratio={area_ratio:.3f}")
        
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        
        return crop, class_id, box
        
    except Exception as e:
        logger.warning(f"YOLO识别失败: {image_path}, {e}")
        return None, None, None

def get_dominant_color(image, k=3):
    """
    获取图像的主色调并映射到标准色彩
    """
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
        
        logger.info(f"主色分析结果: {best_color}, RGB={dom_color}")
        return best_color
    except Exception as e:
        logger.warning(f"主色识别失败: {e}")
        return 'Unknown'

YOLO_CYLINDRICAL_CLASSES = {39, 41, 46}  # 39:bottle, 41:cup, 46:can (COCO)

def infer_shape_from_box(box, class_id):
    """
    从检测框和类别推断形状
    """
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
    elif ratio >= 1.5:
        return 'Rectangular'
    elif ratio <= 0.67:
        return 'Rectangular'
    else:
        return 'Oval'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('static_analysis.log', encoding='utf-8'),
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

def create_prompt(title):
    """
    创建发送给AI的提示文本
    """
    prompt = (
        "Here is a product title. "
        "Extract its color, material and shape. "
        "Respond ONLY with JSON using keys color, material, shape. "
        "Use English words taken ONLY from the lists below. "
        "If any information is missing, put 'Unknown'.\n\n"
        f"Allowed Colors: {', '.join(STANDARD_COLORS)}\n"
        f"Allowed Materials: {', '.join(STANDARD_MATERIALS)}\n"
        f"Allowed Shapes: {', '.join(STANDARD_SHAPES)}\n\n"
        f"Title: {title}"
    )
    
    return prompt

def call_ollama_api(prompt, max_retries=3, retry_delay=2):
    """
    使用ollama本地API调用qwen3:latest模型，返回JSON格式特征
    """
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

def ai_voting_analysis(title, feature_type='all'):
    """
    AI投票机制分析产品特征
    feature_type: 'all', 'material_only', 'color_only', 'shape_only'
    """
    prompt = create_prompt(title)
    
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

def get_sku_from_filename(filename):
    """
    从文件名中提取SKU，去除扩展名
    """
    return os.path.splitext(filename)[0]

def find_image_files(image_folder):
    """
    查找图片文件夹中的所有图片文件
    """
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
    """
    读取Excel文件，返回SKU到标题的映射
    """
    try:
        df = pd.read_excel(excel_file)
        logger.info(f"成功读取Excel文件: {excel_file}, 共 {len(df)} 条记录")
        
        # 检查必要的列
        required_columns = ['product_sku', 'product_title_en']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Excel文件缺少必要的列: {', '.join(missing_columns)}")
            return {}
        
        # 创建SKU到标题的映射，过滤空值
        sku_title_map = {}
        for _, row in df.iterrows():
            sku = str(row['product_sku']).strip()
            title = str(row['product_title_en']).strip()
            if sku and title and not pd.isna(row['product_sku']) and not pd.isna(row['product_title_en']):
                sku_title_map[sku] = title
        
        logger.info(f"创建了 {len(sku_title_map)} 个SKU到标题的映射")
        return sku_title_map
        
    except Exception as e:
        logger.error(f"读取Excel文件失败: {str(e)}")
        return {}

@click.command()
@click.option('--image-folder', '-i', default='data/产品图片', help='图片文件夹路径')
@click.option('--excel-file', '-e', required=True, help='Excel文件路径，包含product_sku和product_title_en列')
@click.option('--output', '-o', 'output_file', default='static_features.csv', help='输出CSV文件路径')
@click.option('--batch-size', '-b', 'batch_size', default=10, type=int, help='每次处理的批次大小')
def main(image_folder, excel_file, output_file, batch_size):
    """静态信息分析：基于本地图片和Excel文件分析产品特征"""
    try:
        # 加载Excel数据
        sku_title_map = load_excel_data(excel_file)
        if not sku_title_map:
            logger.error("无法从Excel文件中加载数据")
            return
        
        # 查找图片文件
        image_files = find_image_files(image_folder)
        if not image_files:
            logger.error("图片文件夹中没有找到图片文件")
            return
        
        # 准备结果数据
        results = []
        total_files = len(image_files)
        
        logger.info(f"开始处理 {total_files} 个图片文件")
        
        # 确保YOLO模型已加载
        _ = get_yolo_model()
        
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
                    color_val = 'Unknown'
                    material_val = 'Unknown'
                    shape_val = 'Unknown'
                    
                    # YOLO分析图片
                    crop, class_id, box = yolo_detect_largest_object(image_path)
                    if crop is not None:
                        # 获取颜色
                        color_val = get_dominant_color(crop)
                        # 推断形状
                        shape_val = infer_shape_from_box(box, class_id)
                        logger.info(f"YOLO分析完成: 颜色={color_val}, 形状={shape_val}")
                    
                    # AI分析材质和缺失信息
                    if sku in sku_title_map:
                        title = sku_title_map[sku]
                        logger.info(f"找到标题: {title}")
                        
                        # 确定需要AI分析的特征
                        ai_result = ai_voting_analysis(title)
                        
                        # 如果YOLO没有获得结果，使用AI结果
                        if color_val == 'Unknown':
                            color_val = standardize_color(ai_result.get('color', 'Unknown'))
                        if shape_val == 'Unknown':
                            shape_val = standardize_shape(ai_result.get('shape', 'Unknown'))
                        
                        # 材质主要依赖AI分析
                        material_val = standardize_material(ai_result.get('material', 'Unknown'))
                        
                    else:
                        logger.warning(f"SKU {sku} 在Excel文件中未找到对应标题")
                    
                    # 标准化所有结果
                    color_val = standardize_color(color_val)
                    material_val = standardize_material(material_val)
                    shape_val = standardize_shape(shape_val)
                    
                    # 添加到结果
                    results.append({
                        'sku': sku,
                        'color': color_val,
                        'material': material_val,
                        'shape': shape_val
                    })
                    
                    logger.info(f"SKU {sku} 处理完成: 颜色={color_val}, 材质={material_val}, 形状={shape_val}")
                    
                    # 添加随机延时
                    time.sleep(random.uniform(1, 2))
                    
                except Exception as e:
                    logger.error(f"处理图片 {filename} 时出错: {str(e)}")
                    # 即使出错也要记录基本信息
                    sku = get_sku_from_filename(filename)
                    results.append({
                        'sku': sku,
                        'color': 'Unknown',
                        'material': 'Unknown',
                        'shape': 'Unknown'
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