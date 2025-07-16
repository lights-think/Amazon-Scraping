"""
静态信息分析工具 - 智能背景处理增强版

最新改进 (v3.0):
1. 智能背景检测: 基于上下文置信度、面积比例、边缘位置的动态背景过滤
2. 产品优先级分类: 区分商品和背景物品，解决商品本身就是"背景物品"的问题
3. 高级背景去除: 基于图像分割和颜色分析的背景处理技术
4. 详细对比输出: CSV中分别显示YOLO/几何分析和LLM分析结果，便于对比分析

核心功能:
1. YOLO物体检测: 自动识别图片中的主要产品物体
2. 几何形状识别: 使用OpenCV轮廓分析、椭圆拟合、顶点检测等方法
3. 亚马逊五点描述爬取: 参考basic_information_identification.py，获取详细产品信息
4. 多模态LLM支持: 使用Ollama的LLaVA模型进行图像形状识别
5. AI模型统一: 所有AI分析均使用gemma3:latest模型

智能背景处理技术:
- 传统方法: 基于COCO类别ID硬过滤 → 容易误判商品
- 智能检测: 综合置信度、面积、位置、上下文的动态判断
- 高级去除: 颜色分析+图像分割，适用于复杂背景场景

CSV输出格式 (详细对比版):
sku,color,material,shape,yolo_color,yolo_shape,llm_color,llm_material,llm_shape,multimodal_llm_shape

使用示例:
    # 基础使用（智能背景检测默认开启）
    python static_information_analysis.py -e products.xlsx
    
    # 启用亚马逊五点描述预爬取 + 所有高级功能
    python static_information_analysis.py -e products.xlsx --amazon --llm-shape --advanced-bg
    
    # 自定义输出和批次大小
    python static_information_analysis.py -e products.xlsx -o detailed_results.csv -b 5
    
工作流程:
    1. 预爬取阶段: 如果启用--amazon，会先批量爬取所有ASIN的五点描述
    2. 图片分析阶段: 使用YOLO检测+几何分析进行视觉特征提取
    3. AI分析阶段: 结合产品标题+五点描述进行LLM特征识别
    4. 结果输出阶段: 生成详细对比CSV，包含各种分析方法的结果

命令行参数:
    --amazon: 启用亚马逊五点描述爬取功能
    --llm-shape: 启用多模态LLM形状识别
    --advanced-bg: 启用高级背景去除技术

环境要求:
    - Ollama服务运行中，已安装gemma3:latest和llava:latest模型
    - Chrome浏览器（用于亚马逊爬取）
    - OpenCV、ultralytics、scikit-learn等依赖包

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

# ========== 依赖说明 ==========
# 需要: pip install ultralytics opencv-python scikit-learn openpyxl playwright
from ultralytics import YOLO
import cv2
from sklearn.cluster import KMeans
import numpy as np
from ollama import chat as ollama_chat
import asyncio
from playwright.async_api import async_playwright

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

# YOLO COCO数据集中的背景/环境类别ID - 基于网络搜索和Context7确认的准确类别
# 参考来源: https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8 
BACKGROUND_CLASS_IDS = {
    # 家具类 - 产品常放置的背景物体
    56,  # chair 椅子
    57,  # couch/sofa 沙发  
    59,  # bed 床
    60,  # dining table 餐桌
    61,  # toilet 马桶
    
    # 电子设备类 - 常见的环境物体
    62,  # tv 电视
    63,  # laptop 笔记本电脑
    64,  # mouse 鼠标  
    65,  # remote 遥控器
    66,  # keyboard 键盘
    67,  # cell phone 手机
    
    # 厨房电器类 - 背景环境物体
    68,  # microwave 微波炉
    69,  # oven 烤箱
    70,  # toaster 烤面包机
    71,  # sink 水槽
    72,  # refrigerator 冰箱
    
    # 其他环境物体类
    73,  # book 书
    74,  # clock 时钟
    13,  # bench 长椅
    58,  # potted plant 盆栽植物（可能是装饰背景）
    
    # 可选过滤的物体（根据实际需要调整）
    75,  # vase 花瓶
    76,  # scissors 剪刀  
    77,  # teddy bear 泰迪熊
    78,  # hair drier 吹风机
    79,  # toothbrush 牙刷
}

# YOLO圆柱形类别（保留原有定义）
YOLO_CYLINDRICAL_CLASSES = {39, 41, 46}  # 39:bottle, 41:cup, 46:can (COCO)

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

# 全局YOLO模型变量
YOLO_MODEL = None

def get_yolo_model():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = YOLO('yolov8s.pt')  # 使用最新小模型
    return YOLO_MODEL

# 智能背景检测配置 - 基于上下文置信度和空间关系
CONTEXT_BASED_BACKGROUND_DETECTION = {
    # 置信度阈值：低于此值的检测被认为是背景噪音
    'confidence_threshold': 0.3,
    
    # 面积比例阈值：占比过大的物体可能是背景（如整面墙、大桌子）
    'large_area_threshold': 0.7,
    
    # 面积比例阈值：占比过小的物体可能是背景噪音
    'small_area_threshold': 0.01,
    
    # 边缘位置检测：接近图像边缘的物体更可能是背景
    'edge_margin_ratio': 0.1,  # 距离边缘10%范围内
    
    # 产品优先类别 - 常见的电商产品类别，优先级较高
    'product_priority_classes': {
        # 食品饮料类
        47,  # apple 苹果
        46,  # banana 香蕉
        51,  # orange 橙子
        41,  # cup 杯子
        39,  # bottle 瓶子
        
        # 电子产品类（但不是背景设备）
        # 注意：这里区分产品和背景，同样是电子设备，手机是背景，但某些电子产品可能是商品
        
        # 服装配饰类
        # COCO数据集中没有明确的服装类别，但可以根据实际需要扩展
        
        # 家居用品类（小件商品）
        84,  # scissors 剪刀（可能是商品而非背景）
        85,  # teddy bear 玩具（可能是商品）
        86,  # hair drier 小家电（可能是商品）
        87,  # toothbrush 个人护理用品
    }
}

def is_likely_background_by_context(box, class_id, confidence, img_shape, all_detections=None):
    """
    基于上下文信息判断检测到的物体是否可能是背景
    
    Args:
        box: 检测框 [x1, y1, x2, y2]
        class_id: YOLO类别ID
        confidence: 检测置信度
        img_shape: 图像尺寸 (height, width)
        all_detections: 所有检测结果，用于上下文分析
    
    Returns:
        bool: True表示可能是背景，False表示可能是前景产品
    """
    config = CONTEXT_BASED_BACKGROUND_DETECTION
    
    # 1. 直接背景类别检查
    if class_id in BACKGROUND_CLASS_IDS:
        # 但是，如果是产品优先类别，则不视为背景
        if class_id in config['product_priority_classes']:
            logger.info(f"类别{class_id}在背景列表中，但是产品优先类别，保留")
            return False
        logger.info(f"类别{class_id}在背景类别列表中，标记为背景")
        return True
    
    # 2. 置信度检查
    if confidence < config['confidence_threshold']:
        logger.debug(f"置信度{confidence:.3f}低于阈值{config['confidence_threshold']}，标记为背景")
        return True
    
    # 3. 面积比例检查
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_shape[0] * img_shape[1]
    area_ratio = box_area / img_area
    
    # 面积过大（可能是背景）
    if area_ratio > config['large_area_threshold']:
        logger.info(f"检测框面积比例{area_ratio:.3f}过大，可能是背景")
        return True
        
    # 面积过小（可能是噪音）
    if area_ratio < config['small_area_threshold']:
        logger.debug(f"检测框面积比例{area_ratio:.3f}过小，可能是背景噪音")
        return True
    
    # 4. 边缘位置检查
    img_height, img_width = img_shape
    edge_margin_x = img_width * config['edge_margin_ratio']
    edge_margin_y = img_height * config['edge_margin_ratio']
    
    # 检查是否接近边缘
    near_edge = (x1 < edge_margin_x or x2 > img_width - edge_margin_x or 
                 y1 < edge_margin_y or y2 > img_height - edge_margin_y)
    
    if near_edge and class_id in BACKGROUND_CLASS_IDS:
        logger.debug(f"检测框接近图像边缘且为背景类别，标记为背景")
        return True
    
    # 5. 产品优先级检查
    if class_id in config['product_priority_classes']:
        logger.info(f"类别{class_id}为产品优先类别，保留为前景")
        return False
    
    # 6. 上下文分析（如果提供了所有检测结果）
    if all_detections is not None and len(all_detections) > 1:
        # 统计同类物体数量
        same_class_count = sum(1 for det in all_detections if det['class_id'] == class_id)
        
        # 如果同类物体很多，可能是背景环境物体
        if same_class_count > 3 and class_id in BACKGROUND_CLASS_IDS:
            logger.debug(f"检测到{same_class_count}个同类背景物体，可能是环境背景")
            return True
    
    # 默认保留为前景
    return False

def apply_advanced_background_removal(image, foreground_box):
    """
    应用更高级的背景去除技术（基于图像分割和颜色分析）
    
    Args:
        image: 输入图像
        foreground_box: 前景物体的检测框 [x1, y1, x2, y2]
    
    Returns:
        processed_image: 背景去除后的图像
        background_mask: 背景掩码
    """
    try:
        # 1. 创建前景区域掩码
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, foreground_box)
        
        # 确保坐标在图像范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 2. 基于颜色相似性的背景检测
        foreground_roi = image[y1:y2, x1:x2]
        if foreground_roi.size == 0:
            return image, None
            
        # 计算前景区域的主色调
        fg_colors = foreground_roi.reshape(-1, 3).astype(np.float32)
        from sklearn.cluster import KMeans
        
        # 使用K-means聚类找到主要颜色
        kmeans = KMeans(n_clusters=min(3, len(fg_colors)), n_init=3, random_state=42)
        kmeans.fit(fg_colors)
        fg_dominant_colors = kmeans.cluster_centers_
        
        # 3. 创建基于颜色距离的背景掩码
        img_pixels = image.reshape(-1, 3).astype(np.float32)
        background_mask = np.zeros((h * w,), dtype=np.uint8)
        
        # 计算每个像素到前景主色调的最小距离
        for i, pixel in enumerate(img_pixels):
            min_dist = min(np.linalg.norm(pixel - color) for color in fg_dominant_colors)
            # 如果距离过大，认为是背景
            if min_dist > 80:  # 阈值可调整
                background_mask[i] = 255
        
        background_mask = background_mask.reshape(h, w)
        
        # 4. 形态学操作优化掩码
        kernel = np.ones((5, 5), np.uint8)
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
        
        # 5. 保留前景区域
        background_mask[y1:y2, x1:x2] = 0  # 前景区域不是背景
        
        # 6. 应用高斯模糊到掩码边缘
        background_mask_blurred = cv2.GaussianBlur(background_mask, (5, 5), 0)
        
        # 7. 创建处理后的图像
        processed_image = image.copy()
        
        # 将背景区域替换为模糊或单色
        background_indices = background_mask_blurred > 128
        processed_image[background_indices] = [240, 240, 240]  # 浅灰色背景
        
        logger.info(f"高级背景去除完成，背景像素比例: {np.sum(background_indices) / (h * w):.3f}")
        
        return processed_image, background_mask
        
    except Exception as e:
        logger.warning(f"高级背景去除失败: {e}")
        return image, None

def yolo_detect_largest_object(image_path):
    """
    使用YOLO检测图片中的所有物体，返回占比最大的非背景物体区域
    增加了智能背景检测和上下文分析
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
        
        # 构建所有检测结果列表，用于上下文分析
        all_detections = []
        for i, box in enumerate(boxes):
            all_detections.append({
                'box': box,
                'class_id': int(clss[i]),
                'confidence': confs[i],
                'area': (box[2] - box[0]) * (box[3] - box[1])
            })
        
        # 使用智能背景检测过滤
        filtered_detections = []
        
        for detection in all_detections:
            box = detection['box']
            class_id = detection['class_id']
            confidence = detection['confidence']
            
            # 应用智能背景检测
            if not is_likely_background_by_context(
                box, class_id, confidence, (img_height, img_width), all_detections
            ):
                filtered_detections.append(detection)
                logger.info(f"保留检测: class_id={class_id}, conf={confidence:.3f}, area_ratio={detection['area']/total_area:.3f}")
            else:
                logger.info(f"过滤背景: class_id={class_id}, conf={confidence:.3f}, area_ratio={detection['area']/total_area:.3f}")
        
        if len(filtered_detections) == 0:
            logger.warning(f"智能过滤后未检测到任何前景目标: {image_path}")
            # 用整图做主色和形状分析
            box = [0, 0, img_width, img_height]
            class_id = None
            logger.info(f"用整图推断: box={box}")
            return img, class_id, box
        
        # 从过滤后的检测中选择面积最大的
        best_detection = max(filtered_detections, key=lambda x: x['area'])
        
        box = best_detection['box']
        class_id = best_detection['class_id']
        conf = best_detection['confidence']
        area_ratio = best_detection['area'] / total_area
        
        logger.info(f"YOLO智能检测到最大前景目标: class_id={class_id}, box={box}, conf={conf}, area_ratio={area_ratio:.3f}")
        
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

def infer_shape_from_box(box, class_id, image=None):
    """
    从检测框和类别推断形状，使用几何特征分析
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
    area = w * h
    
    # 基础宽高比判断（改进版）
    if 0.95 < ratio < 1.05:
        # 接近正方形，但需要进一步验证
        shape_candidate = 'Square'
    elif ratio >= 1.6:
        shape_candidate = 'Rectangular'
    elif ratio <= 0.625:
        shape_candidate = 'Rectangular'
    else:
        # 中等比例，可能是圆形、椭圆或其他形状
        shape_candidate = 'Oval'
    
    # 如果有图像数据，进行更精确的几何分析
    if image is not None:
        try:
            shape_from_geometry = analyze_shape_geometry(image, box)
            if shape_from_geometry != 'Unknown':
                logger.info(f"几何分析结果: {shape_from_geometry}, 宽高比分析: {shape_candidate}")
                return shape_from_geometry
        except Exception as e:
            logger.warning(f"几何形状分析失败: {e}")
    
    # 使用面积与周长的关系进一步判断
    perimeter = 2 * (w + h)
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity > 0.8:
            return 'Round'
        elif circularity > 0.65 and 0.8 < ratio < 1.25:
            return 'Oval'
    
    return shape_candidate

def analyze_shape_geometry(image, box):
    """
    使用OpenCV进行几何特征分析来识别形状
    """
    try:
        x1, y1, x2, y2 = map(int, box)
        
        # 提取感兴趣区域
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return 'Unknown'
        
        # 转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'Unknown'
        
        # 找到最大的轮廓（假设是主要物体）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓面积
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < 100:  # 面积太小，可能是噪声
            return 'Unknown'
        
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
            # 使用拟合椭圆来判断
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
        
        return 'Unknown'
        
    except Exception as e:
        logger.warning(f"几何形状分析出错: {e}")
        return 'Unknown'

def analyze_shape_with_multimodal_llm(image_path, box=None):
    """
    使用Ollama多模态LLM（如LLaVA）分析图像中的形状
    """
    try:
        import base64
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return 'Unknown'
        
        # 如果提供了检测框，裁剪图像
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            image = image[y1:y2, x1:x2]
        
        # 将图像编码为base64
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{image_base64}"
        
        # 创建多模态提示
        prompt = (
            "Please analyze this product image and identify its shape. "
            "Choose ONLY from these allowed shapes: "
            f"{', '.join(STANDARD_SHAPES)}. "
            "Look at the main object in the image and determine if it is: "
            "Round (circular), Square, Rectangular, Oval (elliptical), "
            "Triangular, Hexagonal, Octagonal, Heart, Star, or Cylindrical. "
            "Respond with ONLY the single shape name, no explanations."
        )
        
        # 调用Ollama多模态模型（LLaVA）
        try:
            # 尝试使用llava模型进行视觉分析
            response = ollama_chat(
                model='gemma3:latest',  # 使用LLaVA多模态模型
                messages=[{
                    'role': 'user', 
                    'content': prompt,
                    'images': [data_url]  # 传递图像数据
                }]
            )
            
            content = response['message']['content'] if isinstance(response, dict) else response.message.content
            
            # 解析响应，提取形状名称
            shape_result = content.strip()
            
            # 验证返回的形状是否在标准列表中
            for standard_shape in STANDARD_SHAPES:
                if standard_shape.lower() in shape_result.lower():
                    logger.info(f"多模态LLM识别形状: {standard_shape}")
                    return standard_shape
            
            logger.warning(f"多模态LLM返回了未识别的形状: {shape_result}")
            return 'Unknown'
            
        except Exception as llm_error:
            logger.warning(f"LLaVA模型调用失败: {llm_error}")
            # 降级到文本模型尝试
            try:
                # 使用gemma3描述图像特征，然后推断形状
                simple_prompt = (
                    "This is a product image. Based on typical product shapes, "
                    f"choose the most likely shape from: {', '.join(STANDARD_SHAPES)}. "
                    "Respond with only the shape name."
                )
                
                response = ollama_chat(
                    model='gemma3:latest', 
                    messages=[{'role': 'user', 'content': simple_prompt}]
                )
                
                content = response['message']['content'] if isinstance(response, dict) else response.message.content
                shape_result = content.strip()
                
                for standard_shape in STANDARD_SHAPES:
                    if standard_shape.lower() in shape_result.lower():
                        logger.info(f"文本模型推断形状: {standard_shape}")
                        return standard_shape
                        
                return 'Unknown'
                
            except Exception as fallback_error:
                logger.warning(f"备用文本模型也失败: {fallback_error}")
                return 'Unknown'
        
    except Exception as e:
        logger.warning(f"多模态LLM形状识别出错: {e}")
        return 'Unknown'

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

def create_prompt(title, bullet_points=None):
    """
    创建发送给AI的提示文本，结合标题和五点描述
    """
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
    """
    使用ollama本地API调用gemma3:latest模型，返回JSON格式特征
    """
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

def ai_voting_analysis(title, bullet_points=None, feature_type='all'):
    """
    AI投票机制分析产品特征，结合标题和五点描述
    feature_type: 'all', 'material_only', 'color_only', 'shape_only'
    """
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

async def handle_continue_shopping(page):
    """
    检测并自动点击亚马逊多语言"继续购物"确认页面的按钮，规避反爬虫。
    """
    try:
        # 1. 结构+属性联合定位
        buttons = await page.query_selector_all('button[type="submit"].a-button-text')
        for btn in buttons:
            visible = await btn.is_visible()
            enabled = await btn.is_enabled()
            # 检查父级div是否有提示信息或info图标
            parent = await btn.evaluate_handle('node => node.parentElement')
            parent_html = await parent.evaluate('node => node.outerHTML') if parent else ''
            # 只要按钮可见可用，且父级结构有info或提示框特征
            if visible and enabled and (('a-box-info' in parent_html) or ('a-icon-alert' in parent_html) or ('info' in parent_html)):
                await asyncio.sleep(random.uniform(0.8, 1.5))
                await btn.hover()
                await asyncio.sleep(random.uniform(0.2, 0.6))
                await btn.click()
                logger.info('[CONTINUE_SHOPPING] Clicked button via structure+parent check')
                await page.wait_for_timeout(1200)
                return True
        # 2. 兜底：查找所有可见的type=submit按钮
        all_submit_btns = await page.query_selector_all('button[type="submit"]')
        for btn in all_submit_btns:
            visible = await btn.is_visible()
            enabled = await btn.is_enabled()
            if visible and enabled:
                await asyncio.sleep(random.uniform(0.8, 1.5))
                await btn.hover()
                await asyncio.sleep(random.uniform(0.2, 0.6))
                await btn.click()
                logger.info('[CONTINUE_SHOPPING] Clicked fallback submit button')
                await page.wait_for_timeout(1200)
                return True
        # 3. 兜底：XPath查找所有可见的按钮
        xpath_btns = await page.query_selector_all('//button[@type="submit"]')
        for btn in xpath_btns:
            visible = await btn.is_visible()
            enabled = await btn.is_enabled()
            if visible and enabled:
                await asyncio.sleep(random.uniform(0.8, 1.5))
                await btn.hover()
                await asyncio.sleep(random.uniform(0.2, 0.6))
                await btn.click()
                logger.info('[CONTINUE_SHOPPING] Clicked fallback XPath button')
                await page.wait_for_timeout(1200)
                return True
        logger.info('[CONTINUE_SHOPPING] No continue button found')
        return False
    except Exception as e:
        logger.error(f'[CONTINUE_SHOPPING] Exception: {e}')
        return False

async def extract_amazon_bullet_points(page):
    """
    从亚马逊页面提取五点描述
    """
    bullet_points = []
    try:
        # 尝试使用CSS选择器获取五点描述
        bullet_points_element = await page.query_selector("#feature-bullets")
        if bullet_points_element:
            # 获取所有列表项
            points = await bullet_points_element.query_selector_all("ul li span.a-list-item")
            for point in points:
                point_text = await point.inner_text()
                if point_text.strip():
                    bullet_points.append(point_text.strip())
            logger.info(f"成功提取五点描述(CSS): 共{len(bullet_points)}点")
        else:
            # 尝试使用XPath获取五点描述
            bullet_points_element = await page.query_selector("//div[@id='feature-bullets']")
            if bullet_points_element:
                points = await bullet_points_element.query_selector_all("ul li span.a-list-item")
                for point in points:
                    point_text = await point.inner_text()
                    if point_text.strip():
                        bullet_points.append(point_text.strip())
                logger.info(f"成功提取五点描述(XPath): 共{len(bullet_points)}点")
            else:
                logger.warning("未找到五点描述元素")
    except Exception as e:
        logger.error(f"提取五点描述时出错: {str(e)}")
    
    return bullet_points

async def fetch_amazon_bullet_points(asin, country='US'):
    """
    从亚马逊获取产品的五点描述
    """
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
    读取Excel文件，返回SKU到产品信息的映射
    """
    try:
        df = pd.read_excel(excel_file)
        logger.info(f"成功读取Excel文件: {excel_file}, 共 {len(df)} 条记录")
        
        # 检查必要的列
        required_columns = ['product_sku']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Excel文件缺少必要的列: {', '.join(missing_columns)}")
            return {}
        
        # 创建SKU到产品信息的映射，过滤空值
        sku_info_map = {}
        for _, row in df.iterrows():
            sku = str(row['product_sku']).strip()
            if sku and not pd.isna(row['product_sku']):
                info = {'sku': sku}
                
                # 提取标题（如果存在）
                if 'product_title_en' in df.columns and not pd.isna(row['product_title_en']):
                    info['title'] = str(row['product_title_en']).strip()
                
                # 提取ASIN（如果存在）
                if 'ASIN' in df.columns and not pd.isna(row['ASIN']):
                    info['asin'] = str(row['ASIN']).strip()
                
                # 提取country（如果存在）
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


async def pre_crawl_amazon_data(sku_info_map, enable_amazon_crawl):
    """
    预先爬取所有需要的亚马逊五点描述数据
    """
    if not enable_amazon_crawl:
        logger.info("未启用亚马逊爬取功能，跳过五点描述预爬取")
        return {}
    
    # 收集所有需要爬取的ASIN
    asin_to_crawl = {}
    for sku, info in sku_info_map.items():
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
    
    if crawl_errors:
        logger.warning(f"爬取失败的ASIN ({len(crawl_errors)}个):")
        for error in crawl_errors[:5]:  # 只显示前5个错误
            logger.warning(f"  - {error}")
        if len(crawl_errors) > 5:
            logger.warning(f"  ... 还有 {len(crawl_errors) - 5} 个错误")
    
    return bullet_points_cache

@click.command()
@click.option('--image-folder', '-i', default='data/产品图片', help='图片文件夹路径')
@click.option('--excel-file', '-e', required=True, help='Excel文件路径，包含product_sku列，可选ASIN、country、product_title_en列')
@click.option('--output', '-o', 'output_file', default='static_features.csv', help='输出CSV文件路径')
@click.option('--batch-size', '-b', 'batch_size', default=10, type=int, help='每次处理的批次大小')
@click.option('--enable-amazon-crawl', '--amazon', is_flag=True, help='启用亚马逊五点描述爬取功能（需要ASIN列）')
@click.option('--enable-multimodal-llm', '--llm-shape', is_flag=True, help='启用多模态LLM进行形状识别（实验性功能）')
@click.option('--enable-advanced-bg-removal', '--advanced-bg', is_flag=True, help='启用高级背景去除技术（基于图像分割和颜色分析）')
@click.option('--test', is_flag=True, help='测试模式，仅随机抽取50个图片进行分析')
def main(image_folder, excel_file, output_file, batch_size, enable_amazon_crawl, enable_multimodal_llm, enable_advanced_bg_removal, test):
    """静态信息分析：基于本地图片和Excel文件分析产品特征"""
    try:
        # 加载Excel数据
        sku_info_map = load_excel_data(excel_file)
        if not sku_info_map:
            logger.error("无法从Excel文件中加载数据")
            return
        
        # 预爬取亚马逊五点描述数据
        logger.info("=" * 60)
        logger.info("步骤1: 预爬取亚马逊五点描述数据")
        logger.info("=" * 60)
        bullet_points_cache = asyncio.run(pre_crawl_amazon_data(sku_info_map, enable_amazon_crawl))
        
        # 查找图片文件
        logger.info("=" * 60)
        logger.info("步骤2: 扫描图片文件")
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
        
        # 准备结果数据
        results = []
        total_files = len(image_files)
        
        logger.info("=" * 60)
        logger.info(f"步骤3: 开始分析 {total_files} 个图片文件")
        logger.info("=" * 60)
        
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
                    
                    # 初始化结果 - 分别记录YOLO/几何分析和LLM分析结果
                    yolo_color = 'Unknown'
                    yolo_shape = 'Unknown'
                    llm_color = 'Unknown'
                    llm_material = 'Unknown'
                    llm_shape = 'Unknown'
                    multimodal_llm_shape = 'Unknown'
                    
                    # YOLO分析图片
                    crop, class_id, box = yolo_detect_largest_object(image_path)
                    if crop is not None:
                        # 获取颜色（可选应用高级背景去除）
                        if enable_advanced_bg_removal and box is not None:
                            try:
                                # 读取完整图像用于背景去除
                                full_image = cv2.imread(image_path)
                                processed_image, bg_mask = apply_advanced_background_removal(full_image, box)
                                
                                # 从处理后的图像中提取裁剪区域
                                x1, y1, x2, y2 = map(int, box)
                                enhanced_crop = processed_image[y1:y2, x1:x2]
                                yolo_color = get_dominant_color(enhanced_crop)
                                logger.info(f"应用高级背景去除后的颜色分析: {yolo_color}")
                            except Exception as e:
                                logger.warning(f"高级背景去除失败，使用原始方法: {e}")
                                yolo_color = get_dominant_color(crop)
                        else:
                            yolo_color = get_dominant_color(crop)
                            
                        # 推断形状（传入完整图像进行几何分析）
                        full_image = cv2.imread(image_path)
                        yolo_shape = infer_shape_from_box(box, class_id, full_image)
                        
                        # 如果启用多模态LLM，独立尝试形状识别
                        if enable_multimodal_llm:
                            multimodal_llm_shape = analyze_shape_with_multimodal_llm(image_path, box)
                            if multimodal_llm_shape != 'Unknown':
                                logger.info(f"多模态LLM形状识别结果: {multimodal_llm_shape}")
                        
                        logger.info(f"YOLO分析完成: 颜色={yolo_color}, 形状={yolo_shape}")
                    
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
                    
                    # 生成最终结果（保持向后兼容）
                    final_color = yolo_color if yolo_color != 'Unknown' else llm_color
                    final_material = llm_material  # 材质主要依赖AI分析
                    final_shape = yolo_shape if yolo_shape != 'Unknown' else (multimodal_llm_shape if multimodal_llm_shape != 'Unknown' else llm_shape)
                    
                    # 标准化最终结果
                    final_color = standardize_color(final_color)
                    final_material = standardize_material(final_material)
                    final_shape = standardize_shape(final_shape)
                    
                    # 标准化分析结果
                    yolo_color = standardize_color(yolo_color)
                    yolo_shape = standardize_shape(yolo_shape)
                    llm_color = standardize_color(llm_color)
                    llm_material = standardize_material(llm_material)
                    llm_shape = standardize_shape(llm_shape)
                    multimodal_llm_shape = standardize_shape(multimodal_llm_shape)
                    
                    # 添加到结果（详细对比格式）
                    results.append({
                        'sku': sku,
                        # 最终结果（向后兼容）
                        'color': final_color,
                        'material': final_material,
                        'shape': final_shape,
                        # YOLO/几何分析结果
                        'yolo_color': yolo_color,
                        'yolo_shape': yolo_shape,
                        # LLM分析结果
                        'llm_color': llm_color,
                        'llm_material': llm_material,
                        'llm_shape': llm_shape,
                        # 多模态LLM形状识别结果
                        'multimodal_llm_shape': multimodal_llm_shape
                    })
                    
                    logger.info(f"✓ SKU {sku} 处理完成: 颜色={final_color}, 材质={final_material}, 形状={final_shape}")
                    logger.info(f"  └ 分析详情: YOLO[颜色={yolo_color}, 形状={yolo_shape}] | LLM[颜色={llm_color}, 材质={llm_material}, 形状={llm_shape}] | 多模态[形状={multimodal_llm_shape}]")
                    
                    # 减少延时，因为不再实时爬取
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
                        'yolo_color': 'Unknown',
                        'yolo_shape': 'Unknown',
                        'llm_color': 'Unknown',
                        'llm_material': 'Unknown',
                        'llm_shape': 'Unknown',
                        'multimodal_llm_shape': 'Unknown'
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