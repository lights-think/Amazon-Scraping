# Amazon供应链爬虫系统

一个完整的Amazon产品数据爬取和特征分析系统，现已分为两个独立的阶段：

1. **数据爬取阶段**：使用 `all_in_one_spider.py` 抓取Amazon产品原始数据
2. **特征分析阶段**：使用 `analyze_product_features.py` 分析产品特征

## 系统架构

```
输入文件 (ASIN+Country)
    ↓
🕷️ all_in_one_spider.py (爬虫阶段)
    ↓ 
原始数据 (title, bullet_points, images, BSR等)
    ↓
🧠 analyze_product_features.py (分析阶段)
    ↓
最终结果 (color, material, shape)
```

## 主要功能

### 第一阶段：数据爬取 (`all_in_one_spider.py`)
- ✅ 多进程并发爬取Amazon产品数据
- ✅ 动态用户资料目录切换（反爬机制）
- ✅ 断点续爬功能
- ✅ BSR单独更新模式
- ✅ 抓取内容：标题、描述、图片、BSR、评分、评论数等

### 第二阶段：特征分析 (`analyze_product_features.py`)
- ✅ 多进程特征分析
- ✅ YOLO图像识别（颜色/形状）
- ✅ AI语言模型分析（Ollama + qwen3）
- ✅ 标准化特征映射
- ✅ 增量分析支持

## 快速开始

### 1. 环境准备

```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装YOLO模型依赖
pip install ultralytics opencv-python scikit-learn

# 安装并启动Ollama (本地AI模型)
# 访问 https://ollama.ai/ 下载安装
ollama pull qwen3:latest
```

### 2. 准备输入文件

创建包含ASIN和country的CSV文件：

```csv
ASIN,country
B08N5WRWNW,US
B07ZPKN6YR,UK
B09JQCZJQZ,DE
```

### 3. 完整流程执行

```bash
# 方法1：使用示例脚本（推荐）
python run_example.py

# 方法2：手动执行两个阶段
# 第一阶段：爬取数据
python all_in_one_spider.py --input data/test_input.csv --output temp/spider_raw.csv --processes 2

# 第二阶段：分析特征
python analyze_product_features.py --input temp/spider_raw.csv --output temp/final_result.csv --use-multiprocess
```

## 详细使用说明

### 爬虫阶段 (all_in_one_spider.py)

#### 基本用法
```bash
python all_in_one_spider.py --input data/test_input.csv --output temp/spider_output.csv
```

#### 高级配置
```bash
python all_in_one_spider.py \
    --input data/test_input.csv \
    --output temp/spider_output.csv \
    --processes 3 \
    --concurrency 5 \
    --profile-change-interval 50 \
    --batch-size 20
```

#### BSR更新模式
```bash
# 只更新BSR信息为空的记录
python all_in_one_spider.py --input existing_data.csv --update-bsr
```

#### 参数说明
- `--input, -i`: 输入CSV文件路径
- `--output, -o`: 输出CSV文件路径
- `--processes, -p`: 爬虫进程数 (默认2)
- `--concurrency, -c`: 每进程协程数 (默认3)
- `--profile-change-interval, -r`: 反爬机制切换间隔 (默认100)
- `--update-bsr, -u`: BSR更新模式

### 分析阶段 (analyze_product_features.py)

#### 基本用法
```bash
python analyze_product_features.py --input temp/spider_output.csv --output temp/analyzed.csv
```

#### 多进程分析
```bash
python analyze_product_features.py \
    --input temp/spider_output.csv \
    --output temp/analyzed.csv \
    --use-multiprocess \
    --processes 4 \
    --batch-size 15
```

#### 部分数据处理
```bash
# 只处理第100-200条记录
python analyze_product_features.py \
    --input temp/spider_output.csv \
    --output temp/analyzed_partial.csv \
    --start-index 100 \
    --end-index 200
```

#### 参数说明
- `--input, -i`: 输入CSV文件路径
- `--output, -o`: 输出CSV文件路径
- `--use-multiprocess, -m`: 启用多进程分析
- `--processes, -p`: 分析进程数 (默认2)
- `--batch-size, -b`: 批次大小 (默认10)
- `--sleep-time, -t`: 批次间隔 (默认2秒)
- `--start-index, -s`: 开始索引 (默认0)
- `--end-index, -e`: 结束索引 (默认-1，处理到末尾)

## 输出格式

### 爬虫阶段输出
```csv
ASIN,country,url,title,bullet_points,product_overview,main_image,bsr_main_category,bsr_main_rank,bsr_sub_category,bsr_sub_rank,vine_count,rating,review_count,latest3_rating
```

### 分析阶段输出
在爬虫输出基础上添加：
```csv
color,material,shape
```

## 特征分析原理

### 1. 数据源优先级
1. **产品概览提取**：从`product_overview` JSON中提取现有特征
2. **YOLO图像识别**：分析`main_image`识别颜色和形状
3. **AI语言分析**：使用Ollama分析标题和描述

### 2. 标准化映射
- **颜色**：映射到15种标准颜色
- **材质**：映射到15种标准材质
- **形状**：映射到10种标准形状

### 3. 智能过滤
- 自动过滤非颜色信息（尺寸、型号等）
- 检测和纠正低质量特征提取

## 性能优化

### 爬虫优化
- **多进程并发**：每个进程独立的浏览器实例
- **协程池**：每进程内多个页面并发
- **动态代理**：定期切换用户资料目录
- **断点续爬**：自动跳过已完成的记录

### 分析优化
- **多进程分析**：CPU密集型任务并行化
- **增量处理**：只分析缺失特征的记录
- **批量处理**：减少AI API调用开销
- **缓存机制**：避免重复分析

## 监控和日志

### 日志文件
- `temp/all_in_one_spider.log`: 爬虫详细日志
- `temp/analyze_features.log`: 分析详细日志

### 进度监控
- 实时进度条显示
- 详细的成功/失败统计
- 性能指标记录

## 故障排除

### 常见问题

1. **Chrome浏览器路径问题**
   ```bash
   # 修改 all_in_one_spider.py 中的路径
   executable_path=r"你的Chrome路径"
   ```

2. **Ollama连接失败**
   ```bash
   # 确保Ollama服务运行
   ollama serve
   ollama pull qwen3:latest
   ```

3. **内存不足**
   ```bash
   # 减少并发数
   --processes 1 --concurrency 2
   ```

4. **网络超时**
   ```bash
   # 增加反爬间隔
   --profile-change-interval 50
   ```

### 调试模式
```bash
# 启用详细日志
export PYTHONPATH=.
python -u all_in_one_spider.py --input data/test_input.csv --processes 1
```

## 扩展功能

### 自定义特征分析
可以修改 `analyze_product_features.py` 中的标准化列表：
- `STANDARD_COLORS`
- `STANDARD_MATERIALS` 
- `STANDARD_SHAPES`

### 自定义AI提示
修改 `create_prompt()` 函数来自定义AI分析逻辑。

### 新数据源集成
在 `analyze_single_product()` 函数中添加新的特征提取逻辑。

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License

## 更新日志

### v2.0.0 (当前版本)
- 🔄 分离爬虫和分析功能
- ✨ 增加多进程特征分析
- 🎯 改进YOLO图像识别
- 📊 增强数据兼容性
- 🚀 优化性能和稳定性

### v1.0.0
- 🎉 初始版本，一体化爬虫和分析 