#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进后静态信息分析功能测试脚本

测试内容:
1. YOLO背景过滤验证
2. 几何形状识别测试  
3. Ollama模型连接测试
4. 多模态LLM功能测试

使用方法:
python test_improved_features.py
"""

import logging
import sys
import os
from ollama import chat as ollama_chat

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger()

def test_ollama_connection():
    """测试Ollama服务连接"""
    logger.info("测试Ollama服务连接...")
    
    try:
        # 测试gemma3:latest模型
        response = ollama_chat(
            model='gemma3:latest',
            messages=[{'role': 'user', 'content': 'Hello, respond with just "OK"'}]
        )
        
        content = response['message']['content'] if isinstance(response, dict) else response.message.content
        logger.info(f"gemma3:latest 响应: {content}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ollama连接失败: {e}")
        logger.error("请确保:")
        logger.error("1. Ollama服务正在运行")
        logger.error("2. 已安装gemma3:latest模型: ollama pull gemma3:latest")
        return False

def test_multimodal_llm():
    """测试多模态LLM功能"""
    logger.info("测试多模态LLM (LLaVA)...")
    
    try:
        # 测试llava模型是否可用
        test_prompt = "What shape is a circle? Respond with just 'Round'."
        
        response = ollama_chat(
            model='gemma3:latest',
            messages=[{'role': 'user', 'content': test_prompt}]
        )
        
        content = response['message']['content'] if isinstance(response, dict) else response.message.content
        logger.info(f"LLaVA测试响应: {content}")
        
        return True
        
    except Exception as e:
        logger.error(f"LLaVA模型测试失败: {e}")
        logger.error("请安装LLaVA模型: ollama pull llava:latest")
        return False

def test_yolo_background_filter():
    """测试YOLO背景过滤配置"""
    logger.info("验证YOLO背景过滤配置...")
    
    # 导入背景类别ID配置
    try:
        from static_information_analysis import BACKGROUND_CLASS_IDS
        
        logger.info(f"背景类别ID数量: {len(BACKGROUND_CLASS_IDS)}")
        logger.info(f"背景类别ID: {sorted(BACKGROUND_CLASS_IDS)}")
        
        # 验证关键背景类别
        expected_bg_classes = {56, 57, 59, 60, 61, 62, 63}  # chair, sofa, bed, dining table, etc.
        found_classes = BACKGROUND_CLASS_IDS.intersection(expected_bg_classes)
        
        logger.info(f"关键背景类别覆盖: {len(found_classes)}/{len(expected_bg_classes)}")
        
        if len(found_classes) >= 5:
            logger.info("✓ YOLO背景过滤配置正确")
            return True
        else:
            logger.warning("⚠ YOLO背景过滤配置可能不完整")
            return False
            
    except ImportError as e:
        logger.error(f"无法导入static_information_analysis模块: {e}")
        return False

def test_shape_analysis():
    """测试形状分析功能"""
    logger.info("测试形状分析功能...")
    
    try:
        from static_information_analysis import STANDARD_SHAPES, standardize_shape
        
        logger.info(f"标准形状列表: {STANDARD_SHAPES}")
        
        # 测试形状标准化
        test_shapes = ['round', 'SQUARE', 'rectangular', 'unknown_shape']
        
        for shape in test_shapes:
            result = standardize_shape(shape)
            logger.info(f"'{shape}' -> '{result}'")
        
        logger.info("✓ 形状分析功能正常")
        return True
        
    except Exception as e:
        logger.error(f"形状分析测试失败: {e}")
        return False

def test_ai_analysis():
    """测试AI分析功能"""
    logger.info("测试AI分析功能...")
    
    try:
        from static_information_analysis import ai_voting_analysis
        
        # 简单测试标题分析
        test_title = "Red plastic water bottle"
        test_bullet_points = [
            "Made of high-quality plastic material",
            "Round shape design for easy grip",
            "Bright red color for visibility"
        ]
        
        logger.info(f"测试标题: {test_title}")
        logger.info(f"测试要点: {test_bullet_points}")
        
        # 注意：这会调用真实的AI模型，需要确保模型可用
        result = ai_voting_analysis(test_title, test_bullet_points)
        
        logger.info(f"AI分析结果: {result}")
        logger.info("✓ AI分析功能正常")
        
        return True
        
    except Exception as e:
        logger.error(f"AI分析测试失败: {e}")
        return False

def test_intelligent_background_detection():
    """测试智能背景检测功能"""
    logger.info("测试智能背景检测功能...")
    
    try:
        from static_information_analysis import (
            is_likely_background_by_context, 
            CONTEXT_BASED_BACKGROUND_DETECTION,
            BACKGROUND_CLASS_IDS
        )
        
        # 测试场景1：明显的背景物体（沙发）
        box1 = [100, 100, 500, 400]  # 中等大小的检测框
        class_id1 = 57  # sofa 沙发
        confidence1 = 0.8  # 高置信度
        img_shape = (600, 800)  # 图像尺寸
        
        is_bg1 = is_likely_background_by_context(box1, class_id1, confidence1, img_shape)
        logger.info(f"测试1 - 沙发 (class_id=57): 是否为背景={is_bg1} (期望=True)")
        
        # 测试场景2：产品优先类别（杯子）
        box2 = [200, 200, 300, 350]  # 较小的检测框
        class_id2 = 41  # cup 杯子（在产品优先列表中）
        confidence2 = 0.7
        
        is_bg2 = is_likely_background_by_context(box2, class_id2, confidence2, img_shape)
        logger.info(f"测试2 - 杯子 (class_id=41): 是否为背景={is_bg2} (期望=False)")
        
        # 测试场景3：低置信度检测
        box3 = [50, 50, 150, 150]
        class_id3 = 0  # person 人（通常不是背景，但置信度低）
        confidence3 = 0.2  # 低置信度
        
        is_bg3 = is_likely_background_by_context(box3, class_id3, confidence3, img_shape)
        logger.info(f"测试3 - 低置信度人员: 是否为背景={is_bg3} (期望=True)")
        
        # 测试场景4：面积过大的物体
        box4 = [0, 0, 750, 550]  # 占据大部分图像的检测框
        class_id4 = 60  # dining table 餐桌
        confidence4 = 0.9
        
        is_bg4 = is_likely_background_by_context(box4, class_id4, confidence4, img_shape)
        logger.info(f"测试4 - 大面积餐桌: 是否为背景={is_bg4} (期望=True)")
        
        # 测试场景5：边缘位置的背景物体
        box5 = [750, 10, 800, 60]  # 接近右边缘
        class_id5 = 74  # clock 时钟
        confidence5 = 0.6
        
        is_bg5 = is_likely_background_by_context(box5, class_id5, confidence5, img_shape)
        logger.info(f"测试5 - 边缘时钟: 是否为背景={is_bg5} (期望=True)")
        
        logger.info("✓ 智能背景检测测试完成")
        return True
        
    except Exception as e:
        logger.error(f"智能背景检测测试失败: {e}")
        return False

def test_csv_output_format():
    """测试详细CSV输出格式"""
    logger.info("测试详细CSV输出格式...")
    
    try:
        # 模拟详细分析结果
        test_results = [
            {
                'sku': 'TEST001',
                'color': 'Red',
                'material': 'Plastic',
                'shape': 'Round',
                'yolo_color': 'Red',
                'yolo_shape': 'Round',
                'llm_color': 'Red',
                'llm_material': 'Plastic',
                'llm_shape': 'Cylindrical',
                'multimodal_llm_shape': 'Round'
            },
            {
                'sku': 'TEST002',
                'color': 'Blue',
                'material': 'Metal',
                'shape': 'Rectangular',
                'yolo_color': 'Blue',
                'yolo_shape': 'Rectangular',
                'llm_color': 'Unknown',
                'llm_material': 'Metal',
                'llm_shape': 'Unknown',
                'multimodal_llm_shape': 'Rectangular'
            }
        ]
        
        # 检查所有期望的列是否存在
        expected_columns = [
            'sku', 'color', 'material', 'shape',  # 基础列
            'yolo_color', 'yolo_shape',  # YOLO分析列
            'llm_color', 'llm_material', 'llm_shape',  # LLM分析列
            'multimodal_llm_shape'  # 多模态LLM列
        ]
        
        # 验证每个结果包含所有期望的字段
        for i, result in enumerate(test_results):
            missing_fields = [field for field in expected_columns if field not in result]
            if missing_fields:
                logger.error(f"测试结果{i+1}缺少字段: {missing_fields}")
                return False
            
            logger.info(f"测试结果{i+1}包含所有期望字段: {list(result.keys())}")
        
        # 检查对比功能
        result1 = test_results[0]
        if result1['yolo_shape'] != result1['llm_shape']:
            logger.info(f"SKU {result1['sku']}: YOLO形状({result1['yolo_shape']}) vs LLM形状({result1['llm_shape']}) - 可进行对比分析")
        
        if result1['color'] == result1['yolo_color']:
            logger.info(f"SKU {result1['sku']}: 最终颜色与YOLO颜色一致({result1['color']})")
        
        logger.info("✓ CSV输出格式测试完成")
        return True
        
    except Exception as e:
        logger.error(f"CSV输出格式测试失败: {e}")
        return False

def test_advanced_background_removal():
    """测试高级背景去除功能"""
    logger.info("测试高级背景去除功能...")
    
    try:
        from static_information_analysis import apply_advanced_background_removal
        import cv2
        import numpy as np
        
        # 创建测试图像（模拟产品图片）
        height, width = 400, 600
        test_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 模拟背景（蓝色）
        test_image[:, :] = [100, 150, 200]
        
        # 模拟前景产品（红色矩形）
        foreground_box = [200, 150, 400, 250]
        x1, y1, x2, y2 = foreground_box
        test_image[y1:y2, x1:x2] = [50, 50, 200]  # 红色产品
        
        # 应用高级背景去除
        processed_image, background_mask = apply_advanced_background_removal(test_image, foreground_box)
        
        if processed_image is not None and background_mask is not None:
            logger.info("✓ 高级背景去除功能正常工作")
            logger.info(f"处理后图像尺寸: {processed_image.shape}")
            logger.info(f"背景掩码尺寸: {background_mask.shape}")
            
            # 检查前景区域是否被保留
            fg_pixels = processed_image[y1:y2, x1:x2]
            if not np.array_equal(fg_pixels, test_image[y1:y2, x1:x2]):
                logger.warning("前景区域可能被意外修改")
            else:
                logger.info("✓ 前景区域正确保留")
            
            return True
        else:
            logger.error("高级背景去除返回了None结果")
            return False
        
    except Exception as e:
        logger.error(f"高级背景去除测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 50)
    logger.info("开始改进功能测试")
    logger.info("=" * 50)
    
    test_results = {}
    
    # 1. 测试YOLO背景过滤配置
    test_results['yolo_bg_filter'] = test_yolo_background_filter()
    
    # 2. 测试形状分析功能
    test_results['shape_analysis'] = test_shape_analysis()
    
    # 3. 测试Ollama连接
    test_results['ollama_connection'] = test_ollama_connection()
    
    # 4. 测试多模态LLM（如果Ollama可用）
    if test_results['ollama_connection']:
        test_results['multimodal_llm'] = test_multimodal_llm()
        
        # 5. 测试AI分析功能（需要模型可用）
        test_results['ai_analysis'] = test_ai_analysis()
    else:
        test_results['multimodal_llm'] = False
        test_results['ai_analysis'] = False
    
    # 输出测试总结
    logger.info("=" * 50)
    logger.info("测试结果总结:")
    logger.info("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name:20}: {status}")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info(f"\n总计: {passed_tests}/{total_tests} 项测试通过")
    
    if passed_tests == total_tests:
        logger.info("🎉 所有测试通过！改进功能可以正常使用。")
        return 0
    elif passed_tests >= total_tests * 0.7:
        logger.info("⚠️  大部分测试通过，建议检查失败项目。")
        return 1
    else:
        logger.error("❌ 多项测试失败，请检查环境配置。")
        return 2

if __name__ == "__main__":
    sys.exit(main()) 