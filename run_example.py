#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-in-One Amazon Spider 示例运行脚本
"""

import subprocess
import sys
import os

def run_spider_example():
    """运行示例爬虫"""
    
    # 检查输入文件是否存在
    input_file = "test_input_sample.csv"
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        print("请先创建包含ASIN和country列的CSV文件")
        return
    
    # 基本参数配置
    cmd = [
        sys.executable, "all_in_one_spider.py",
        "--input", input_file,
        "--output", "temp/example_output.csv",
        "--processes", "2",
        "--concurrency", "3",
        "--batch-size", "5",  # 小批量测试
        "--sleep-time", "3",
        "--analyze-batch-size", "5",
        "--analyze-sleep", "2"
    ]
    
    print("=== All-in-One Amazon Spider 示例运行 ===")
    print(f"输入文件: {input_file}")
    print(f"输出文件: temp/example_output.csv")
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    try:
        # 运行爬虫
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n=== 运行完成 ===")
            print("请检查以下文件:")
            print("- temp/example_output.csv (最终结果)")
            print("- temp/all_info_raw.csv (原始爬虫数据)")
            print("- temp/all_in_one_spider.log (详细日志)")
        else:
            print(f"\n运行失败，退出码: {result.returncode}")
            print("请检查日志文件获取详细错误信息")
            
    except FileNotFoundError:
        print("错误: 找不到 all_in_one_spider.py 文件")
        print("请确保在正确的目录下运行此脚本")
    except Exception as e:
        print(f"运行时发生错误: {e}")

def show_help():
    """显示帮助信息"""
    print("All-in-One Amazon Spider 示例脚本")
    print()
    print("用法:")
    print("  python run_example.py        # 运行示例")
    print("  python run_example.py -h     # 显示帮助")
    print()
    print("功能:")
    print("- 使用 test_input_sample.csv 作为输入")
    print("- 小批量配置，适合测试")
    print("- 自动创建 temp/ 目录")
    print("- 输出到 temp/example_output.csv")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        show_help()
    else:
        run_spider_example() 