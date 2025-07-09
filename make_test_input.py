import pandas as pd

# 读取原始数据
df = pd.read_excel('data/asin大全.xlsx')
print(df.head())
# 只保留ASIN和country两列，去除空值
df = df[['子asin', '站点']].dropna()
df['ASIN'] = df['子asin'].astype(str).str.strip()
df['country'] = df['站点'].astype(str).str.strip().str.upper()

# 尝试获取销量/评论数/排名等字段
possible_sales_fields = [col for col in df.columns if '销量' in col or '评论' in col or 'rank' in col.lower()]
# 如果有销量/评论/排名字段，优先用来筛选高低销量
if possible_sales_fields:
    sales_col = possible_sales_fields[0]
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
else:
    sales_col = None

test_rows = []
for country in df['country'].unique():
    sub = df[df['country'] == country]
    # 优先选销量高和低的各1个
    if sales_col:
        sub = sub.sort_values(sales_col, ascending=False)
        high = sub.iloc[0]
        low = sub.iloc[-1]
        test_rows.append(high)
        if not high.equals(low):
            test_rows.append(low)
    else:
        # 没有销量字段则随机选2个
        test_rows.extend(sub.sample(n=min(50, len(sub)), random_state=42).to_dict('records'))

# 去重
test_df = pd.DataFrame(test_rows).drop_duplicates(subset=['ASIN', 'country'])
test_df[['ASIN', 'country']].to_csv('data/test_input.csv', index=False, encoding='utf-8-sig')
print('测试集已保存到 data/test_input.csv')
