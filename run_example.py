#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon供应链爬虫 - 完整工作流程示例
现在分为两个阶段：
1. 爬虫阶段：使用 all_in_one_spider.py 抓取原始数据
2. 分析阶段：使用 analyze_product_features.py 分析产品特征
"""

import os
import subprocess
import pandas as pd
import time

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"执行命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print(f"✅ {description} - 执行成功!")
            if result.stdout:
                print("输出信息:")
                print(result.stdout[-500:])  # 显示最后500字符
        else:
            print(f"❌ {description} - 执行失败!")
            if result.stderr:
                print("错误信息:")
                print(result.stderr[-500:])
            return False
                
    except Exception as e:
        print(f"❌ {description} - 执行异常: {e}")
        return False
    
    return True

def main():
    """完整的Amazon供应链分析流程"""
    
    print("🚀 Amazon供应链爬虫 - 完整工作流程")
    print("=" * 60)
    
    # 1. 检查输入文件
    input_file = "data/test_input.csv"
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        print("请确保输入文件包含 ASIN 和 country 两列")
        return
    
    # 读取输入文件查看样本
    try:
        df_input = pd.read_csv(input_file)
        print(f"📊 输入文件: {input_file}")
        print(f"   记录数: {len(df_input)}")
        print(f"   列名: {list(df_input.columns)}")
        if len(df_input) > 0:
            print("   样本数据:")
            print(df_input.head(3).to_string(index=False))
    except Exception as e:
        print(f"❌ 读取输入文件失败: {e}")
        return
    
    # 2. 第一阶段：运行爬虫
    spider_output = "temp/spider_raw_output.csv"
    spider_cmd = f"""python all_in_one_spider.py \\
        --input "{input_file}" \\
        --output "{spider_output}" \\
        --processes 2 \\
        --concurrency 3 \\
        --profile-change-interval 100"""
    
    if not run_command(spider_cmd, "第一阶段：爬虫数据抓取"):
        print("❌ 爬虫阶段失败，停止执行")
        return
    
    # 检查爬虫输出
    if not os.path.exists(spider_output):
        print(f"❌ 爬虫输出文件不存在: {spider_output}")
        return
    
    try:
        df_spider = pd.read_csv(spider_output)
        print(f"\n📊 爬虫输出文件: {spider_output}")
        print(f"   记录数: {len(df_spider)}")
        print(f"   列名: {list(df_spider.columns)}")
        
        # 检查数据质量
        success_count = len(df_spider[df_spider['title'].notna() & (df_spider['title'] != '')])
        print(f"   成功抓取的记录: {success_count}/{len(df_spider)} ({success_count/len(df_spider)*100:.1f}%)")
    except Exception as e:
        print(f"❌ 读取爬虫输出失败: {e}")
        return
    
    # 3. 第二阶段：特征分析
    analysis_output = "temp/product_features_analyzed.csv"
    
    # 单进程分析示例
    analysis_cmd_single = f"""python analyze_product_features.py \\
        --input "{spider_output}" \\
        --output "{analysis_output}" \\
        --batch-size 10 \\
        --sleep-time 2"""
    
    # 多进程分析示例
    analysis_cmd_multi = f"""python analyze_product_features.py \\
        --input "{spider_output}" \\
        --output "{analysis_output}" \\
        --batch-size 10 \\
        --processes 2 \\
        --sleep-time 2 \\
        --use-multiprocess"""
    
    # 根据数据量选择分析方式
    if len(df_spider) > 10:
        print("\n🔀 检测到较多数据，使用多进程分析")
        selected_cmd = analysis_cmd_multi
        analysis_desc = "第二阶段：多进程特征分析"
    else:
        print("\n🔀 数据量较少，使用单进程分析")
        selected_cmd = analysis_cmd_single
        analysis_desc = "第二阶段：单进程特征分析"
    
    if not run_command(selected_cmd, analysis_desc):
        print("❌ 特征分析阶段失败")
        return
    
    # 4. 检查最终结果
    if not os.path.exists(analysis_output):
        print(f"❌ 分析输出文件不存在: {analysis_output}")
        return
    
    try:
        df_final = pd.read_csv(analysis_output)
        print(f"\n📊 最终结果文件: {analysis_output}")
        print(f"   记录数: {len(df_final)}")
        print(f"   列名: {list(df_final.columns)}")
        
        # 分析结果统计
        if 'color' in df_final.columns:
            color_stats = df_final['color'].value_counts()
            print(f"\n🎨 颜色分布:")
            print(color_stats.head(5).to_string())
        
        if 'material' in df_final.columns:
            material_stats = df_final['material'].value_counts()
            print(f"\n🧱 材质分布:")
            print(material_stats.head(5).to_string())
        
        if 'shape' in df_final.columns:
            shape_stats = df_final['shape'].value_counts()
            print(f"\n📐 形状分布:")
            print(shape_stats.head(5).to_string())
        
        # 显示样本数据
        print(f"\n📋 最终结果样本:")
        display_columns = ['ASIN', 'country', 'color', 'material', 'shape', 'title']
        available_columns = [col for col in display_columns if col in df_final.columns]
        print(df_final[available_columns].head(3).to_string(index=False))
        
    except Exception as e:
        print(f"❌ 读取最终结果失败: {e}")
        return
    
    # 5. 完成总结
    print(f"\n🎉 完整流程执行完成!")
    print(f"=" * 60)
    print(f"📁 输入文件: {input_file} ({len(df_input)} 条记录)")
    print(f"📁 爬虫输出: {spider_output} ({len(df_spider)} 条记录)")
    print(f"📁 最终结果: {analysis_output} ({len(df_final)} 条记录)")
    print(f"📁 日志文件: temp/all_in_one_spider.log, temp/analyze_features.log")
    
    print(f"\n🔧 后续使用建议:")
    print(f"   • 如需更新BSR信息: python all_in_one_spider.py --update-bsr")
    print(f"   • 如需重新分析特征: python analyze_product_features.py --use-multiprocess")
    print(f"   • 如需部分处理: 添加 --start-index 和 --end-index 参数")

if __name__ == "__main__":
    main() 