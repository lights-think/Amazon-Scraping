# Amazon 商品信息爬虫

基于 Python 和 Playwright 的亚马逊商品信息爬虫，支持从 CSV 导入 ASIN 和国家，抓取 BSR（大类和小类）、产品评分和评论数，并输出到 CSV。

## 特性
- 无头运行（Headless Chrome）
- 命令行界面（CLI）
- 进度条显示（tqdm）
- CSV 输入/输出
- 支持主流国家：US、UK、DE、FR、ES、IT、CA、JP、MX、IN

## 安装

1. 克隆仓库或下载脚本

```bash
git clone <your-repo-url>
cd <repo>
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 安装 Playwright 浏览器组件

```bash
playwright install
```

## 使用

```bash
# 对 CSV 文件（使用并发参数示例）
python amazon_scraper.py -i input.csv -o output.csv -e utf-8 -s '\t' -c 10

# 对 Excel 文件 (.xls/.xlsx)
python amazon_scraper.py -i input.xlsx -o output.csv
```

- `input.csv`/`input.xlsx` 应包含 `ASIN` 和 `country` 列
- 对 CSV 文件，可通过 `-e`/`--encoding` 指定输入文件编码（默认 `utf-8`)，通过 `-s`/`--sep` 指定分隔符（默认 `,`)
- 可通过 `-c`/`--concurrency` 指定并发任务数（默认 5）
- 输出文件 `output.csv` 将包含抓取结果：
  - `bsr_main_category`：主分类名称
  - `bsr_main_rank`：主分类排名
  - `bsr_sub_category`：子分类名称
  - `bsr_sub_rank`：子分类排名
  - `rating`：平均评分
  - `review_count`：评论数量

## 定期启动

可以使用系统定时任务（如 `cron`）定期运行：

```cron
0 2 * * * cd /path/to/repo && /usr/bin/python3 amazon_scraper.py -i input.csv -o output.csv
```

## 开源协议

本项目基于 MIT 协议开源，详细见 [LICENSE](LICENSE)。 