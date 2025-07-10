# Amazon 商品信息爬虫

基于 Python 和 Playwright 的亚马逊商品信息爬虫，支持从 CSV/Excel 导入 ASIN 和国家，分为两个独立模块：

1. **主爬虫 (amazon_scraper.py)** - 抓取商品基本信息、BSR排名、评分和评论数
2. **VINE爬虫 (VINE_Sccrape.py)** - 专门抓取商品的VINE评论数量和最近评论评分
3. **一体化主控 (all_in_one_spider.py)** - 一次性完成爬取 + YOLO/AI 分析 + 断点续爬，产出最终17列结果

## 特性

### 共同特性
- 支持无头/可视化运行 Chrome
- 命令行界面（CLI）
- 进度条显示（tqdm）
- CSV/Excel 输入/输出
- 支持多进程并行抓取
- 支持主流国家：US、UK、DE、FR、ES、IT、CA、JP、MX、IN、NL、SE、BE、IE、AU、BR、SG等

### 主爬虫功能
- 抓取商品BSR排名（主类和子类）
- 抓取商品评分和评论数
- 自动处理多语言网站
- 支持多进程并行抓取

### VINE爬虫功能
- 专注抓取VINE评论数量
- 计算最近评论的平均评分
- 支持多国家登录状态管理
- 支持仅登录模式

## 安装

1. 克隆仓库或下载脚本

```bash
git clone https://github.com/lights-think/Amazon-Scraping.git
cd Amazon-Scraping
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 安装 Playwright 浏览器组件

```bash
playwright install
```

> 注意：脚本默认使用系统 Chrome，可执行路径为 Windows: `C:\Program Files\Google\Chrome\Application\chrome.exe`，如安装路径不同，请修改脚本中 `executable_path` 参数。

## 使用方法

### 主爬虫 (amazon_scraper.py)

```bash
# 基本用法
python amazon_scraper.py -i input.csv -o output.csv

# 高级参数
python amazon_scraper.py -i input.csv -o output.csv -e utf-8 -s '\t' -c 3 -p my_browser_profile

# 多进程模式
python amazon_scraper.py -i input.csv -o output.csv --profile-template my_profile_ --profile-count 4
```

参数说明:
- `-i/--input`: 输入文件路径 (CSV或Excel)
- `-o/--output`: 输出文件路径 (CSV)
- `-e/--encoding`: 输入CSV文件编码 (默认 utf-8-sig)
- `-s/--sep`: 输入CSV分隔符 (默认 ,)
- `-c/--concurrency`: 单进程内协程并发数 (默认 3)
- `-p/--profile-dir`: 单进程模式用户数据目录 (默认 my_browser_profile)
- `--profile-template`: 多进程模式用户数据目录前缀
- `--profile-count`: 多进程数量 (>0时启用多进程)

输出字段:
- `ASIN`: 商品ASIN
- `country`: 国家代码
- `url`: 商品URL
- `bsr_main_category`: 主分类名称
- `bsr_main_rank`: 主分类排名
- `bsr_sub_category`: 子分类名称
- `bsr_sub_rank`: 子分类排名
- `rating`: 平均评分
- `review_count`: 评论数量

### VINE爬虫 (VINE_Sccrape.py)

```bash
# 基本用法
python VINE_Sccrape.py -i input.csv -o vine_output.csv

# 仅登录模式
python VINE_Sccrape.py -i input.csv --login-only

# 强制重新登录
python VINE_Sccrape.py -i input.csv -o vine_output.csv --force-login

# 多进程模式
python VINE_Sccrape.py -i input.csv -o vine_output.csv --profile-template vine_profile_ --profile-count 4
```

参数说明:
- `-i/--input`: 输入文件路径 (CSV或Excel)
- `-o/--output`: 输出文件路径 (CSV)
- `-e/--encoding`: 输入CSV文件编码 (默认 utf-8-sig)
- `-s/--sep`: 输入CSV分隔符 (默认 ,)
- `-c/--concurrency`: 单进程内协程并发数 (默认 3)
- `-p/--profile-dir`: 单进程模式用户数据目录 (默认 my_browser_profile)
- `--profile-template`: 多进程模式用户数据目录前缀
- `--profile-count`: 多进程数量 (>0时启用多进程)
- `--force-login`: 强制重新登录所有国家
- `--login-only`: 仅执行登录流程，不进行爬取

输出字段:
- `ASIN`: 商品ASIN
- `country`: 国家代码
- `vine_count`: VINE评论数量
- `latest3_rating`: 最近3条评论的平均评分

## 一体化主控脚本 (all_in_one_spider.py)
`all_in_one_spider.py` 集成了 **爬虫 → YOLO 图片分析 → AI 文本分析 → CSV 合并输出** 的完整流程，支持多进程 + 多协程，并内置断点续爬。

### 主要特性
- **一次执行**：读取 ASIN+国家列表，自动完成所有信息抓取及特征分析。
- **断点续爬**：中途中断后重新运行会跳过已处理记录（依据 `temp/all_info_raw.csv`）。
- **多进程+协程**：`--processes` 控制进程数，`--concurrency` 控制每进程协程数。
- **YOLO+AI**：先用 YOLO 识别主图颜色/形状，再用 AI(ollama) 补全缺失信息。
- **可配置批次**：`--batch-size / --sleep-time` 控制单批 ASIN 数及批间间隔，降低反爬风险。

### 命令行示例
```bash
# 快速测试（单进程）
python all_in_one_spider.py -i input.csv -o result.csv --processes 1 --batch-size 20

# 生产示例（多进程）
python all_in_one_spider.py \
  -i input.csv -o result.csv \
  --processes 4 --concurrency 3 \
  --batch-size 100 --sleep-time 8 \
  --analyze-batch-size 20 --analyze-sleep 3
```
常用参数：
| 参数 | 说明 | 默认 |
|------|------|------|
| `-i/--input` | 输入文件 (CSV/Excel)，需含 ASIN 和 country | `data/test_input.csv` |
| `-o/--output` | 最终输出 CSV | `temp/all_info_output.csv` |
| `--processes` | 进程数 | 2 |
| `--concurrency` | 每进程协程数 | 3 |
| `--batch-size` | 每批 ASIN 数 | 50 |
| `--sleep-time` | 批间休眠秒数 | 5 |
| `--analyze-batch-size` | AI 分析批大小 | 10 |
| `--analyze-sleep` | 分析批间休眠秒数 | 2 |

### 断点续爬说明
- 爬虫阶段完成后，原始数据将写入 `temp/all_info_raw.csv`。
- 再次运行脚本会自动读取该文件并跳过已完成的 `(ASIN,country)` 组合。
- 如需强制重爬某条，可手动删除 `temp/all_info_raw.csv` 或在其中删除对应行。

### 最终输出字段（17 列）
| 列名 | 含义 |
|------|------|
| ASIN | 产品 ASIN |
| country | 国家代码 |
| url | 产品链接 |
| color | 颜色 (标准化) |
| material | 材质 (标准化) |
| shape | 形状 (标准化) |
| title | 产品标题 |
| bullet_points | 五点描述 (多行换行分隔) |
| product_overview | 产品概览(JSON) |
| bsr_main_category | 主类名称 |
| bsr_main_rank | 主类排名 |
| bsr_sub_category | 子类名称 |
| bsr_sub_rank | 子类排名 |
| vine_count | Vine 评论数 |
| rating | 平均评分 |
| review_count | 评论数 |
| latest3_rating | 最近3条评论平均评分 |

> **依赖**：除 `requirements.txt` 外，还需安装 YOLO (`ultralytics`)、`opencv-python`、`ollama` 等，可参考 `requirements_all_in_one.txt`。

## 输入文件格式

输入文件 (CSV或Excel) 必须包含以下列:
- `ASIN`: 商品ASIN编码
- `country`: 国家代码 (如US, UK, DE等)

示例:
```
ASIN,country
B07PXGQC1Q,US
B08F2YD1GM,UK
B07ZGLLWBT,DE
```

## 登录流程

首次运行VINE爬虫时，系统会打开浏览器窗口引导您登录Amazon账户。登录成功后，状态将被保存在用户数据目录中，后续运行无需重复登录。

使用`--force-login`参数可强制重新登录，`--login-only`参数可只执行登录而不进行爬取。

## 定期启动

可以使用系统定时任务（如 `cron`）定期运行：

```cron
# 每天凌晨2点运行主爬虫
0 2 * * * cd /path/to/repo && /usr/bin/python3 amazon_scraper.py -i input.csv -o output.csv

# 每天凌晨3点运行VINE爬虫
0 3 * * * cd /path/to/repo && /usr/bin/python3 VINE_Sccrape.py -i input.csv -o vine_output.csv
```

## 开源协议

本项目基于 MIT 协议开源，详细见 [LICENSE](LICENSE)。 