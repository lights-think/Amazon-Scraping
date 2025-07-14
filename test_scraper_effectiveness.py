import csv
import pandas as pd

def test_scraper_effectiveness(csv_file='./temp/all_info_raw.csv', debug=False):
    """
    测试爬虫有效性的脚本，根据以下规则：
    1. 如果父类子类只有一个，认定缺数据
    2. 两个都没有认定为有效
    3. 父子类名称一致，认定为无效
    4. Rating和review_count缺失认定无效
    5. BSR父类或子类名称超过10个英文单词认定无效
    
    有效率大于80%认定爬虫可用
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file)
        print(f"成功读取CSV文件: {csv_file}")
        print(f"总样本数: {len(df)}")
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return
    
    # 初始化计数器
    total_samples = len(df)
    missing_data_count = 0
    invalid_count = 0
    valid_count = 0
    
    # 记录无效数据的原因
    invalid_reasons = {
        "父类子类只有一个": 0,
        "父子类名称一致": 0,
        "子类排名为100": 0,
        "Rating缺失": 0,
        "Review Count缺失": 0,
        "BSR类别名称过长": 0
    }
    
    # 逐行检查数据
    for index, row in df.iterrows():
        invalid_reasons_for_row = []
        
        # 检查父类和子类
        has_main_category = pd.notna(row.get('bsr_main_category', '')) and row.get('bsr_main_category', '') != ''
        has_sub_category = pd.notna(row.get('bsr_sub_category', '')) and row.get('bsr_sub_category', '') != ''
        
        if debug:
            print(f"\n检查行 {index+1} (ASIN: {row.get('ASIN', 'N/A')}):")
            print(f"  主类别: {row.get('bsr_main_category', 'N/A')}, 子类别: {row.get('bsr_sub_category', 'N/A')}")
            print(f"  评分: {row.get('rating', 'N/A')}, 评论数: {row.get('review_count', 'N/A')}")

        # 新增规则：子类排名为100 视为缺数据
        sub_rank_value = row.get('bsr_sub_rank', '')
        is_sub_rank_100 = False
        if pd.notna(sub_rank_value):
            # 若能转换为数字并等于100，则视为100；否则检查字符串"100"
            try:
                if float(sub_rank_value) == 100:
                    is_sub_rank_100 = True
            except (ValueError, TypeError):
                pass
            if not is_sub_rank_100 and str(sub_rank_value).strip() == '100':
                is_sub_rank_100 = True
        if is_sub_rank_100:
            missing_data_count += 1
            invalid_reasons_for_row.append("子类排名为100")
            if debug:
                print("  ❌ 子类排名为100 (缺数据)")
        
        # 1. 如果父类子类只有一个，认定缺数据
        if (has_main_category and not has_sub_category) or (not has_main_category and has_sub_category):
            missing_data_count += 1
            invalid_reasons_for_row.append("父类子类只有一个")
            if debug:
                print("  ❌ 父类子类只有一个 (缺数据)")
        
        # 2. 两个都没有认定为有效（不做处理，默认有效）
        if not has_main_category and not has_sub_category and debug:
            print("  ✅ 父类子类都没有 (认定有效)")
        
        # 3. 父子类名称一致，认定为无效
        if has_main_category and has_sub_category and row['bsr_main_category'] == row['bsr_sub_category']:
            invalid_reasons_for_row.append("父子类名称一致")
            if debug:
                print("  ❌ 父子类名称一致")
        
        # 4. Rating和review_count缺失认定无效
        if pd.isna(row.get('rating', '')) or row.get('rating', '') == '':
            invalid_reasons_for_row.append("Rating缺失")
            if debug:
                print("  ❌ Rating缺失")
        
        if pd.isna(row.get('review_count', '')) or row.get('review_count', '') == '':
            invalid_reasons_for_row.append("Review Count缺失")
            if debug:
                print("  ❌ Review Count缺失")
        
        # 5. 检查BSR父类或子类名称是否超过10个英文单词
        def count_english_words(text):
            """统计英文单词数量"""
            if pd.isna(text) or text == '':
                return 0
            # 分割文本并过滤空字符串
            words = [word.strip() for word in str(text).split() if word.strip()]
            return len(words)
        
        main_category_words = count_english_words(row.get('bsr_main_category', ''))
        sub_category_words = count_english_words(row.get('bsr_sub_category', ''))
        
        if main_category_words > 10 or sub_category_words > 10:
            invalid_reasons_for_row.append("BSR类别名称过长")
            if debug:
                print(f"  ❌ BSR类别名称过长 (主类: {main_category_words}词, 子类: {sub_category_words}词)")
        
        # 统计无效原因
        for reason in invalid_reasons_for_row:
            invalid_reasons[reason] += 1
        
        # 判断数据有效性
        if len(invalid_reasons_for_row) == 0:
            # 如果没有任何无效原因，则认为有效
            valid_count += 1
            if debug:
                print("  ✅ 有效数据")
        elif len(invalid_reasons_for_row) == 1 and invalid_reasons_for_row[0] in ["父类子类只有一个", "子类排名为100"]:
            # 如果只有“父类子类只有一个”或“子类排名为100”这一个原因，则只计入缺数据，不计入无效数据
            if debug:
                print("  ❓ 缺数据 (不计入无效)")
        else:
            # 如果有其他无效原因，则认为无效
            invalid_count += 1
            if debug:
                print("  ❌ 无效数据")
    
    # 计算有效率
    effectiveness_rate = (valid_count / total_samples) * 100 if total_samples > 0 else 0
    
    # 打印结果
    print("\n===== 爬虫有效性测试结果 =====")
    print(f"总样本数: {total_samples}")
    print(f"缺数据数: {missing_data_count}")
    print(f"无效数据: {invalid_count}")
    print(f"有效数据: {valid_count}")
    print(f"有效率: {effectiveness_rate:.2f}%")
    
    # 打印无效原因的详细统计
    print("\n===== 无效数据原因统计 =====")
    for reason, count in invalid_reasons.items():
        if count > 0:
            print(f"{reason}: {count}条")
    
    # 判断爬虫是否可用
    if effectiveness_rate >= 80:
        print("\n✅ 爬虫有效率大于80%，认定为可用")
    else:
        print(f"\n❌ 爬虫有效率低于80%（{effectiveness_rate:.2f}%），认定为不可用")
        print("建议改进爬虫以提高数据质量")

if __name__ == "__main__":
    test_scraper_effectiveness(debug=True) 