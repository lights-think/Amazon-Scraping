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

                    # 如果仍有 Unknown 字段，再调用 AI 解析缺失部分
                    if "Unknown" in [color_val, material_val, shape_val]:
                        prompt = create_prompt(title, bullet_points)
                        features = call_groq_api(prompt)
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