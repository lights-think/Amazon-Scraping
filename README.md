# Amazon供应链爬虫系统

一个完整的Amazon产品数据爬取和特征分析系统，现已发展为模块化的两阶段架构：

1. **数据爬取阶段**：使用 `all_in_one_spider.py` 抓取Amazon产品原始数据
2. **特征分析阶段**：使用 `analyze_product_features.py` 分析产品特征

## 🚀 新版本特性

### v2.1.0 主要更新
- ✨ **完整模块化架构**：爬虫和分析完全解耦，可独立运行
- 🧠 **增强AI分析**：集成YOLO图像识别 + Ollama本地大模型
- ⚡ **多进程优化**：爬虫和分析阶段都支持多进程并行
- 🔄 **智能断点续爬**：自动跳过已完成记录，支持BSR单独更新
- 📊 **爬虫有效性测试**：自动评估数据质量和抓取成功率
- 🛠️ **完整工作流**：一键运行端到端流程的示例脚本

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
    ↓
📊 test_scraper_effectiveness.py (质量测试)
```

## 主要功能模块

### 第一阶段：数据爬取 (`all_in_one_spider.py`)
- ✅ 多进程并发爬取Amazon产品数据
- ✅ 智能反爬机制（动态用户资料目录切换）
- ✅ 断点续爬功能，自动跳过已完成记录
- ✅ BSR专项更新模式，只更新缺失的BSR信息
- ✅ 全面数据抓取：标题、描述、图片URL、BSR、评分、评论数、Vine标记

### 第二阶段：特征分析 (`analyze_product_features.py`)
- ✅ 多进程特征分析，CPU密集型任务并行化
- ✅ YOLO图像识别，自动识别产品颜色和形状
- ✅ AI语言模型分析（集成Ollama + qwen3本地模型）
- ✅ 三层特征提取：产品概览 → 图像识别 → AI分析
- ✅ 标准化特征映射，统一15种颜色、15种材质、10种形状
- ✅ 增量分析支持，只处理缺失特征的记录

### 质量控制模块
- 📊 **有效性测试** (`test_scraper_effectiveness.py`)：评估爬虫数据质量
- 🎯 **测试数据生成** (`make_test_input.py`)：基于现有数据创建测试集
- 🔄 **完整工作流** (`run_example.py`)：端到端流程演示和执行

## 快速开始

### 1. 环境准备

```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装Playwright浏览器驱动
playwright install chromium

# 安装并启动Ollama (本地AI模型)
# 访问 https://ollama.ai/ 下载安装
ollama pull qwen3:latest
```

### 2. 准备输入数据

#### 方法1：使用现有ASIN数据生成测试集
```bash
# 将你的ASIN数据放到 data/asin大全.xlsx
python make_test_input.py
```

#### 方法2：手动创建输入文件
创建包含ASIN和country的CSV文件：

```csv
ASIN,country
B08N5WRWNW,US
B07ZPKN6YR,UK
B09JQCZJQZ,DE
```

### 3. 完整流程执行

```bash
# 方法1：使用示例脚本（推荐）- 自动执行全流程
python run_example.py

# 方法2：手动分阶段执行
# 第一阶段：爬取数据
python all_in_one_spider.py --input data/test_input.csv --output temp/spider_raw.csv --processes 2

# 第二阶段：分析特征
python analyze_product_features.py --input temp/spider_raw.csv --output temp/final_result.csv --use-multiprocess

# 第三阶段：质量测试
python test_scraper_effectiveness.py
```

## 详细使用说明

### 爬虫阶段 (all_in_one_spider.py)

#### 标准爬取模式
```bash
python all_in_one_spider.py \
    --input data/test_input.csv \
    --output temp/spider_output.csv \
    --processes 2 \
    --concurrency 3 \
    --profile-change-interval 100
```

#### BSR专项更新模式
```bash
# 只更新BSR信息为空的记录，不重新抓取其他数据
python all_in_one_spider.py \
    --input existing_data.csv \
    --output updated_data.csv \
    --update-bsr
```

#### 参数说明
- `--input, -i`: 输入CSV文件路径
- `--output, -o`: 输出CSV文件路径  
- `--processes, -p`: 爬虫进程数 (默认2)
- `--concurrency, -c`: 每进程协程数 (默认3)
- `--profile-change-interval, -r`: 反爬机制切换间隔 (默认100)
- `--batch-size, -b`: 批处理大小 (默认20)
- `--update-bsr, -u`: BSR专项更新模式

### 分析阶段 (analyze_product_features.py)

#### 多进程分析（推荐）
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

### 质量测试 (test_scraper_effectiveness.py)

```bash
# 测试爬虫数据质量
python test_scraper_effectiveness.py

# 对于有效率>80%认定爬虫可用，会输出详细的质量分析报告
```

测试规则：
- BSR父子类数据完整性检查
- 评分和评论数缺失率统计  
- BSR类目名称合理性验证
- 综合有效率评估

## 输出数据格式

### 爬虫阶段输出
```csv
ASIN,country,url,title,bullet_points,product_overview,main_image,bsr_main_category,bsr_main_rank,bsr_sub_category,bsr_sub_rank,vine_count,rating,review_count,latest3_rating
```

### 分析阶段输出
在爬虫输出基础上添加：
```csv
color,material,shape
```

## 特征分析技术原理

### 1. 三层特征提取策略
1. **产品概览解析**：从`product_overview` JSON结构中提取现有特征
2. **YOLO图像识别**：分析`main_image`，使用计算机视觉识别颜色和形状
3. **AI语言分析**：使用Ollama本地大模型分析标题和描述文本

### 2. 智能标准化映射
- **颜色**：映射到15种标准颜色（Black, White, Red, Blue等）
- **材质**：映射到15种标准材质（Plastic, Metal, Wood, Fabric等）
- **形状**：映射到10种标准形状（Round, Square, Rectangular等）

### 3. 质量过滤机制
- 自动过滤非颜色信息（尺寸、型号、品牌等）
- 噪音检测：识别包含数字、尺寸单位的低质量提取
- 增量分析：只处理缺失或不完整的特征

## 性能优化策略

### 爬虫性能优化
- **多进程架构**：每个进程独立的浏览器实例，避免资源竞争
- **协程池管理**：进程内多页面并发，提高单进程效率
- **智能反爬**：动态切换用户资料目录，降低检测风险
- **断点续爬**：自动跳过已完成记录，支持增量更新

### 分析性能优化  
- **多进程分析**：CPU密集型特征分析任务并行化
- **批量处理**：减少AI模型调用开销
- **增量处理**：只分析缺失特征的记录
- **本地AI模型**：使用Ollama避免API调用限制

## 监控和日志

### 详细日志系统
- `temp/all_in_one_spider.log`: 爬虫详细运行日志
- `temp/analyze_features.log`: 特征分析详细日志

### 实时监控
- 进度条显示：实时显示处理进度和预估完成时间
- 统计报告：成功/失败数量统计和成功率
- 性能指标：处理速度、内存使用等关键指标

## 故障排除

### 常见问题及解决方案

1. **Chrome浏览器路径问题**
   ```bash
   # 修改 all_in_one_spider.py 中的路径
   executable_path=r"你的Chrome安装路径"
   ```

2. **Ollama连接失败**
   ```bash
   # 确保Ollama服务正常运行
   ollama serve
   ollama pull qwen3:latest
   
   # 检查服务状态
   curl http://localhost:11434/api/tags
   ```

3. **内存不足错误**
   ```bash
   # 减少并发数和进程数
   --processes 1 --concurrency 2 --batch-size 5
   ```

4. **网络超时或反爬检测**
   ```bash
   # 增加反爬切换间隔，降低抓取频率
   --profile-change-interval 50
   ```

5. **YOLO模型下载失败**
   ```bash
   # 手动下载YOLO模型
   python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
   ```

### 调试模式
```bash
# 启用详细日志和单进程模式便于调试
python all_in_one_spider.py \
    --input data/test_input.csv \
    --processes 1 \
    --concurrency 1
```

## 扩展功能

### 自定义特征类别
可以在 `analyze_product_features.py` 中修改标准化列表：
```python
STANDARD_COLORS = ["Black", "White", "Red", ...]  # 自定义颜色
STANDARD_MATERIALS = ["Plastic", "Metal", ...]    # 自定义材质  
STANDARD_SHAPES = ["Round", "Square", ...]        # 自定义形状
```

### 自定义AI分析提示
修改 `create_prompt()` 函数来自定义AI分析逻辑：
```python
def create_prompt(title, bullet_points):
    return f"请分析以下产品的特征：\n标题：{title}\n描述：{bullet_points}"
```

### 新站点支持
在 `DOMAIN_MAP` 中添加新的Amazon站点：
```python
DOMAIN_MAP['新国家代码'] = 'amazon.新域名'
```

## 项目文件结构

```
Amazon-Scraping/
├── all_in_one_spider.py          # 主爬虫模块
├── analyze_product_features.py   # 特征分析模块
├── run_example.py                # 完整工作流程示例
├── test_scraper_effectiveness.py # 爬虫有效性测试
├── make_test_input.py            # 测试数据生成
├── amazon_scraper.py             # 爬虫核心功能
├── basic_information_identification.py # 基础信息提取
├── static_information_analysis.py # 静态信息分析
├── VINE_Sccrape.py              # Vine产品专项爬取
├── requirements.txt              # Python依赖列表
├── README.md                     # 项目文档
└── data/                        # 数据目录
    └── test_input.csv           # 测试输入文件
```

## 使用案例

### 案例1：批量产品特征分析
```bash
# 1. 生成测试数据
python make_test_input.py

# 2. 执行完整分析流程  
python run_example.py

# 3. 查看结果
head temp/product_features_analyzed.csv
```

### 案例2：BSR信息批量更新
```bash
# 只更新现有数据中缺失的BSR信息
python all_in_one_spider.py \
    --input existing_products.csv \
    --output updated_products.csv \
    --update-bsr \
    --processes 3
```

### 案例3：大数据集分批处理
```bash
# 分批处理大数据集，每次处理1000条
python analyze_product_features.py \
    --input large_dataset.csv \
    --output batch_1_results.csv \
    --start-index 0 \
    --end-index 1000 \
    --use-multiprocess \
    --processes 4
```

## 贡献指南

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/新功能`)
3. 提交更改 (`git commit -am '添加新功能'`)
4. 推送到分支 (`git push origin feature/新功能`)
5. 创建 Pull Request

### 开发规范
- 遵循PEP 8代码风格
- 添加必要的日志记录
- 编写单元测试
- 更新相关文档

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 更新日志

### v2.1.0 (当前版本)
- 🔄 完整模块化架构重构
- ✨ 新增YOLO图像识别功能
- 🧠 集成Ollama本地大模型分析
- 📊 添加爬虫有效性测试模块
- 🚀 优化多进程性能和稳定性
- 🛠️ 完善工作流程和错误处理

### v2.0.0
- 🔄 分离爬虫和分析功能为独立模块
- ✨ 增加多进程特征分析支持
- 🎯 改进AI语言模型分析
- 📊 增强数据格式兼容性
- 🚀 全面优化性能和稳定性

### v1.0.0
- 🎉 初始版本，一体化爬虫和分析功能

---

## 技术支持

如有问题或建议，请通过以下方式联系：
- 创建 GitHub Issue
- 提交 Pull Request
- 发送邮件至项目维护者

**注意**：本工具仅供学习和研究使用，请遵守Amazon的robots.txt和使用条款，避免过度请求对服务器造成负担。 