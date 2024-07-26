import pandas as pd
import json
import tushare as ts
from datetime import datetime, timedelta



token = "bb4d13a08e1b2927a3dfb2e527e226649e05e3749f8cc239c6cb66ac"

def get_quarterly_financials(token, quarter):

    # 设置Tushare的token
    ts.set_token(token)

    # 初始化pro接口
    pro = ts.pro_api()
    fields = 'ts_code,end_date,report_type,total_revenue,total_cogs'
    df =pro.income_vip(period=quarter,fields =fields)
    print(df)
    return df

def process_tushare_data(df):
    # 假设 df 是从 Tushare 获取的 DataFrame
    processed_data = []

    for index, row in df.iterrows():
        # 为每个指标创建一个问答对
        for column in df.columns:
            if column != 'ts_code' and column != 'trade_date':  # 排除股票代码和日期列
                question = f"在 {row['trade_date']} 这一天，{row['ts_code']} 的 {column} 是多少？"
                answer = f"{row['ts_code']} 在 {row['trade_date']} 的 {column} 是 {row[column]}。"

                processed_data.append({
                    "instruction": question,
                    "input": "",
                    "output": answer
                })

    return processed_data


import json
import random


def create_dataset(df):
    """
    将财务数据DataFrame转换为语言模型训练用的问答对数据集

    参数:
    df (pandas.DataFrame): 包含多个公司财务数据的DataFrame

    返回:
    list: 问答对数据集
    """
    dataset = []

    for _, row in df.iterrows():
        ts_code = row['ts_code']
        end_date = pd.to_datetime(str(row['end_date']))
        year = end_date.year
        quarter = (end_date.month - 1) // 3 + 1
        total_revenue = row['total_revenue']
        total_cogs = row['total_cogs']
        gross_profit = total_revenue - total_cogs

        # print(ts_code,year,quarter,total_revenue,total_cogs)

        # 创建多个问答对
        qa_pairs = [
            {
                "instruction": f"请问{ts_code}在{year}年第{quarter}季度的营业收入是多少？",
                "input": "",
                "output": f"{ts_code}在{year}年第{quarter}季度的营业收入是{total_revenue:.2f}元。"
            },
            {
                "instruction": f"{year}年第{quarter}季度，{ts_code}的营业成本是多少？",
                "input": "",
                "output": f"{year}年第{quarter}季度，{ts_code}的营业成本为{total_cogs:.2f}元。"
            },
            {
                "instruction": f"计算{ts_code}在{year}年第{quarter}季度的毛利润。",
                "input": "",
                "output": f"{ts_code}在{year}年第{quarter}季度的毛利润为{gross_profit:.2f}元，计算方法是营业收入{total_revenue:.2f}元减去营业成本{total_cogs:.2f}元。"
            }
        ]

        # 处理毛利率计算的特殊情况
        if total_revenue != 0:
            gross_margin = (gross_profit / total_revenue) * 100
            qa_pairs.append({
                "instruction": f"分析{ts_code}在{year}年第{quarter}季度的毛利率。",
                "input": "",
                "output": f"{ts_code}在{year}年第{quarter}季度的毛利率为{gross_margin:.2f}%。这是通过毛利润{gross_profit:.2f}元除以营业收入{total_revenue:.2f}元计算得出的。"
            })
        else:
            qa_pairs.append({
                "instruction": f"分析{ts_code}在{year}年第{quarter}季度的毛利率。",
                "input": "",
                "output": f"{ts_code}在{year}年第{quarter}季度的毛利率无法计算，因为营业收入为0。"
            })

        # 处理成本占比计算的特殊情况
        if total_revenue != 0:
            cost_ratio = (total_cogs / total_revenue) * 100
            qa_pairs.append({
                "instruction": f"比较{ts_code}在{year}年第{quarter}季度的营业收入和营业成本。",
                "input": "",
                "output": f"{ts_code}在{year}年第{quarter}季度的营业收入为{total_revenue:.2f}元，营业成本为{total_cogs:.2f}元。营业成本占营业收入的{cost_ratio:.2f}%，这反映了公司的成本控制情况。"
            })
        else:
            qa_pairs.append({
                "instruction": f"比较{ts_code}在{year}年第{quarter}季度的营业收入和营业成本。",
                "input": "",
                "output": f"{ts_code}在{year}年第{quarter}季度的营业收入为0元，营业成本为{total_cogs:.2f}元。由于营业收入为0，无法计算成本占比。这可能表明公司在该季度没有产生收入或数据有误。"
            })

        dataset.extend(qa_pairs)

        # 打乱数据集顺序
    random.shuffle(dataset)
    print(dataset)

    return dataset





if __name__ == '__main__':
    token = "bb4d13a08e1b2927a3dfb2e527e226649e05e3749f8cc239c6cb66ac"
    quarter = 20230331
    df =get_quarterly_financials(token=token,quarter=quarter)
    create_dataset(df=df)