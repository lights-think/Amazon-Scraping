# Amazon 商品信息爬虫

基于 Python 和 Playwright 的亚马逊商品信息爬虫，支持从 CSV 导入 ASIN 和国家，抓取 BSR（大类和小类）、产品评分和评论数，并输出到 CSV。

## 特性
- 无头运行（Headless Chrome）
- 命令行界面（CLI）
- 进度条显示（tqdm）
- CSV 输入/输出
- 支持主流国家：US、UK、DE、FR、ES、IT、CA、JP、MX、IN
- 支持抓取 Vine 评论数量和最后三条评论的平均评分（输出字段 `vine_count`、`latest3_rating`）。

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

> 注意：脚本默认使用系统 Chrome，可执行路径为 Windows: `C:\Program Files\Google\Chrome\Application\chrome.exe`，如安装路径不同，请修改 `amazon_scraper.py` 中 `executable_path` 参数。

## 使用

```bash
# 对 CSV 文件（使用并发参数示例）
python amazon_scraper.py -i input.csv -o output.csv -e utf-8 -s '\t' -c 10 -p my_browser_profile

# 对 Excel 文件 (.xls/.xlsx)
python amazon_scraper.py -i input.xlsx -o output.csv -p my_browser_profile
```

- `input.csv`/`input.xlsx` 应包含 `ASIN` 和 `country` 列
- 对 CSV 文件，可通过 `-e`/`--encoding` 指定输入文件编码（默认 `utf-8`)，通过 `-s`/`--sep` 指定分隔符（默认 `,`)
- 可通过 `-c`/`--concurrency` 指定并发任务数（默认 5）
- 可通过 `-p`/`--profile-dir` 指定用户数据目录（默认 `my_browser_profile`），用于持久化浏览器登录信息。

首次运行时如未检测到登录状态，脚本会打开可视化浏览器进行登录 Vine 评论页面，登录完成后会保存登录状态，后续运行无需重复登录。

- 输出文件 `output.csv` 将包含抓取结果：
  - `bsr_main_category`：主分类名称
  - `bsr_main_rank`：主分类排名
  - `bsr_sub_category`：子分类名称
  - `bsr_sub_rank`：子分类排名
  - `rating`：平均评分
  - `review_count`：评论数量
  - `vine_count`：Vine 评论数量
  - `latest3_rating`：最后一页前三条评论的平均评分

## 定期启动

可以使用系统定时任务（如 `cron`）定期运行：

```cron
0 2 * * * cd /path/to/repo && /usr/bin/python3 amazon_scraper.py -i input.csv -o output.csv
```

## 开源协议

本项目基于 MIT 协议开源，详细见 [LICENSE](LICENSE)。 