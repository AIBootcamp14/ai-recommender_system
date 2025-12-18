"""
추천 시스템 경진대회 - 탐색적 데이터 분석 (EDA)
결과물: 시각화 이미지 + JSON 요약 파일
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import Counter
from datetime import datetime

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리 생성
OUTPUT_DIR = '../eda_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(name):
    """그래프 저장 헬퍼 함수"""
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  저장: {name}.png')

def main():
    print('=' * 60)
    print('추천 시스템 EDA 시작')
    print('=' * 60)

    # 데이터 로드
    print('\n[1/10] 데이터 로드 중...')
    train = pd.read_parquet('../data/train.parquet')
    train['event_time'] = pd.to_datetime(train['event_time'], format='%Y-%m-%d %H:%M:%S %Z')

    # JSON 결과 저장용 딕셔너리
    eda_result = {}

    # =========================================================================
    # 1. 기본 통계
    # =========================================================================
    print('\n[2/10] 기본 통계 분석...')

    basic_stats = {
        'total_interactions': int(len(train)),
        'unique_users': int(train['user_id'].nunique()),
        'unique_items': int(train['item_id'].nunique()),
        'unique_sessions': int(train['user_session'].nunique()),
        'unique_categories': int(train['category_code'].nunique()),
        'unique_brands': int(train['brand'].nunique()),
        'date_range': {
            'start': str(train['event_time'].min()),
            'end': str(train['event_time'].max()),
            'days': int((train['event_time'].max() - train['event_time'].min()).days)
        },
        'price_stats': {
            'min': float(train['price'].min()),
            'max': float(train['price'].max()),
            'mean': float(train['price'].mean()),
            'median': float(train['price'].median()),
            'std': float(train['price'].std())
        },
        'sparsity': float(1 - len(train) / (train['user_id'].nunique() * train['item_id'].nunique()))
    }
    eda_result['basic_stats'] = basic_stats

    # =========================================================================
    # 2. 이벤트 타입 분석
    # =========================================================================
    print('\n[3/10] 이벤트 타입 분석...')

    event_counts = train['event_type'].value_counts()
    event_stats = {
        'distribution': {k: int(v) for k, v in event_counts.items()},
        'percentage': {k: float(v/len(train)*100) for k, v in event_counts.items()},
        'conversion_rate': {
            'view_to_cart': float(event_counts.get('cart', 0) / event_counts.get('view', 1) * 100),
            'cart_to_purchase': float(event_counts.get('purchase', 0) / event_counts.get('cart', 1) * 100),
            'view_to_purchase': float(event_counts.get('purchase', 0) / event_counts.get('view', 1) * 100)
        }
    }
    eda_result['event_type'] = event_stats

    # 시각화: 이벤트 타입 분포
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    axes[0].bar(event_counts.index, event_counts.values, color=colors)
    axes[0].set_title('이벤트 타입별 횟수')
    axes[0].set_ylabel('횟수')
    for i, v in enumerate(event_counts.values):
        axes[0].text(i, v + 50000, f'{v:,}', ha='center')

    axes[1].pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%', colors=colors)
    axes[1].set_title('이벤트 타입 비율')

    save_fig('01_event_type_distribution')

    # =========================================================================
    # 3. 유저 행동 분석
    # =========================================================================
    print('\n[4/10] 유저 행동 분석...')

    user_interactions = train.groupby('user_id').size()
    user_event_types = train.groupby('user_id')['event_type'].value_counts().unstack(fill_value=0)

    # 유저별 구매 여부
    users_with_purchase = (user_event_types.get('purchase', pd.Series([0])) > 0).sum()

    user_stats = {
        'interactions_per_user': {
            'min': int(user_interactions.min()),
            'max': int(user_interactions.max()),
            'mean': float(user_interactions.mean()),
            'median': float(user_interactions.median()),
            'std': float(user_interactions.std())
        },
        'interaction_distribution': {
            '1_interaction': int((user_interactions == 1).sum()),
            '2-5_interactions': int(((user_interactions >= 2) & (user_interactions <= 5)).sum()),
            '6-10_interactions': int(((user_interactions >= 6) & (user_interactions <= 10)).sum()),
            '11-50_interactions': int(((user_interactions >= 11) & (user_interactions <= 50)).sum()),
            '50+_interactions': int((user_interactions > 50).sum())
        },
        'users_with_purchase': int(users_with_purchase),
        'users_without_purchase': int(len(user_interactions) - users_with_purchase),
        'purchase_user_ratio': float(users_with_purchase / len(user_interactions) * 100)
    }
    eda_result['user_behavior'] = user_stats

    # 시각화: 유저별 상호작용 분포
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 히스토그램 (로그 스케일)
    axes[0].hist(user_interactions.clip(upper=100), bins=50, color='#3498db', edgecolor='white')
    axes[0].set_title('유저별 상호작용 수 분포 (100 이하)')
    axes[0].set_xlabel('상호작용 수')
    axes[0].set_ylabel('유저 수')

    # 상호작용 구간별 유저 수
    interaction_bins = list(user_stats['interaction_distribution'].values())
    interaction_labels = ['1회', '2-5회', '6-10회', '11-50회', '50회+']
    axes[1].bar(interaction_labels, interaction_bins, color='#9b59b6')
    axes[1].set_title('상호작용 구간별 유저 수')
    axes[1].set_ylabel('유저 수')
    for i, v in enumerate(interaction_bins):
        axes[1].text(i, v + 5000, f'{v:,}', ha='center')

    save_fig('02_user_interaction_distribution')

    # =========================================================================
    # 4. 아이템 인기도 분석
    # =========================================================================
    print('\n[5/10] 아이템 인기도 분석...')

    item_interactions = train.groupby('item_id').size()
    item_purchases = train[train['event_type'] == 'purchase'].groupby('item_id').size()

    item_stats = {
        'interactions_per_item': {
            'min': int(item_interactions.min()),
            'max': int(item_interactions.max()),
            'mean': float(item_interactions.mean()),
            'median': float(item_interactions.median()),
            'std': float(item_interactions.std())
        },
        'interaction_distribution': {
            '1-5_interactions': int((item_interactions <= 5).sum()),
            '6-50_interactions': int(((item_interactions >= 6) & (item_interactions <= 50)).sum()),
            '51-500_interactions': int(((item_interactions >= 51) & (item_interactions <= 500)).sum()),
            '500+_interactions': int((item_interactions > 500).sum())
        },
        'top_10_items': [
            {'item_id': str(idx), 'count': int(val)}
            for idx, val in item_interactions.nlargest(10).items()
        ],
        'items_never_purchased': int(len(item_interactions) - len(item_purchases)),
        'items_purchased_ratio': float(len(item_purchases) / len(item_interactions) * 100)
    }
    eda_result['item_popularity'] = item_stats

    # 시각화: 아이템 인기도 (롱테일)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 롱테일 분포
    sorted_items = item_interactions.sort_values(ascending=False).values
    axes[0].plot(range(len(sorted_items)), sorted_items, color='#e74c3c')
    axes[0].set_title('아이템 인기도 분포 (Long-tail)')
    axes[0].set_xlabel('아이템 순위')
    axes[0].set_ylabel('상호작용 수')
    axes[0].set_yscale('log')

    # 상위 20개 아이템
    top20 = item_interactions.nlargest(20)
    axes[1].barh(range(20), top20.values, color='#e74c3c')
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([f'Item {i+1}' for i in range(20)])
    axes[1].set_title('상위 20개 인기 아이템')
    axes[1].set_xlabel('상호작용 수')
    axes[1].invert_yaxis()

    save_fig('03_item_popularity_distribution')

    # =========================================================================
    # 5. 카테고리 분석
    # =========================================================================
    print('\n[6/10] 카테고리 분석...')

    # 메인 카테고리 추출 (첫 번째 레벨)
    train['main_category'] = train['category_code'].fillna('unknown').apply(
        lambda x: x.split('.')[0] if pd.notna(x) else 'unknown'
    )

    category_counts = train['main_category'].value_counts()
    category_purchase = train[train['event_type'] == 'purchase']['main_category'].value_counts()

    # 카테고리별 전환율
    category_conversion = {}
    for cat in category_counts.index[:15]:
        cat_data = train[train['main_category'] == cat]
        views = len(cat_data[cat_data['event_type'] == 'view'])
        purchases = len(cat_data[cat_data['event_type'] == 'purchase'])
        if views > 0:
            category_conversion[cat] = float(purchases / views * 100)

    category_stats = {
        'total_main_categories': int(train['main_category'].nunique()),
        'total_sub_categories': int(train['category_code'].nunique()),
        'null_category_count': int(train['category_code'].isna().sum()),
        'null_category_ratio': float(train['category_code'].isna().mean() * 100),
        'top_10_categories': {k: int(v) for k, v in category_counts.head(10).items()},
        'category_conversion_rate': category_conversion,
        'top_purchased_categories': {k: int(v) for k, v in category_purchase.head(10).items()}
    }
    eda_result['category'] = category_stats

    # 시각화: 카테고리 분포
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top10_cat = category_counts.head(10)
    n_cats = len(top10_cat)
    axes[0].barh(range(n_cats), top10_cat.values, color='#2ecc71')
    axes[0].set_yticks(range(n_cats))
    axes[0].set_yticklabels(top10_cat.index)
    axes[0].set_title(f'상위 {n_cats}개 메인 카테고리')
    axes[0].set_xlabel('상호작용 수')
    axes[0].invert_yaxis()

    # 카테고리별 전환율
    conv_cats = list(category_conversion.keys())[:10]
    conv_rates = [category_conversion[c] for c in conv_cats]
    n_conv = len(conv_cats)
    axes[1].barh(range(n_conv), conv_rates, color='#f39c12')
    axes[1].set_yticks(range(n_conv))
    axes[1].set_yticklabels(conv_cats)
    axes[1].set_title('카테고리별 구매 전환율 (%)')
    axes[1].set_xlabel('전환율 (%)')
    axes[1].invert_yaxis()

    save_fig('04_category_analysis')

    # =========================================================================
    # 6. 브랜드 분석
    # =========================================================================
    print('\n[7/10] 브랜드 분석...')

    brand_counts = train['brand'].value_counts()
    brand_purchase = train[train['event_type'] == 'purchase']['brand'].value_counts()

    brand_stats = {
        'total_brands': int(train['brand'].nunique()),
        'null_brand_count': int(train['brand'].isna().sum()),
        'null_brand_ratio': float(train['brand'].isna().mean() * 100),
        'top_10_brands': {str(k): int(v) for k, v in brand_counts.head(10).items()},
        'top_purchased_brands': {str(k): int(v) for k, v in brand_purchase.head(10).items()}
    }
    eda_result['brand'] = brand_stats

    # 시각화: 브랜드 분포
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top10_brand = brand_counts.head(10)
    n_brands = len(top10_brand)
    axes[0].barh(range(n_brands), top10_brand.values, color='#9b59b6')
    axes[0].set_yticks(range(n_brands))
    axes[0].set_yticklabels([str(x)[:20] for x in top10_brand.index])
    axes[0].set_title(f'상위 {n_brands}개 브랜드 (전체 상호작용)')
    axes[0].set_xlabel('상호작용 수')
    axes[0].invert_yaxis()

    top10_brand_purchase = brand_purchase.head(10)
    n_brands_p = len(top10_brand_purchase)
    axes[1].barh(range(n_brands_p), top10_brand_purchase.values, color='#e91e63')
    axes[1].set_yticks(range(n_brands_p))
    axes[1].set_yticklabels([str(x)[:20] for x in top10_brand_purchase.index])
    axes[1].set_title(f'상위 {n_brands_p}개 브랜드 (구매)')
    axes[1].set_xlabel('구매 수')
    axes[1].invert_yaxis()

    save_fig('05_brand_analysis')

    # =========================================================================
    # 7. 가격 분석
    # =========================================================================
    print('\n[8/10] 가격 분석...')

    price_by_event = train.groupby('event_type')['price'].agg(['mean', 'median', 'std'])

    # 가격대별 구매 전환율
    train['price_range'] = pd.cut(train['price'],
                                   bins=[0, 50, 100, 200, 500, 1000, float('inf')],
                                   labels=['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+'])

    price_range_stats = train.groupby('price_range')['event_type'].value_counts().unstack(fill_value=0)

    price_stats = {
        'by_event_type': {
            event: {
                'mean': float(price_by_event.loc[event, 'mean']),
                'median': float(price_by_event.loc[event, 'median']),
                'std': float(price_by_event.loc[event, 'std'])
            }
            for event in price_by_event.index
        },
        'price_range_distribution': {
            str(pr): {
                'view': int(price_range_stats.loc[pr, 'view']) if 'view' in price_range_stats.columns else 0,
                'cart': int(price_range_stats.loc[pr, 'cart']) if 'cart' in price_range_stats.columns else 0,
                'purchase': int(price_range_stats.loc[pr, 'purchase']) if 'purchase' in price_range_stats.columns else 0
            }
            for pr in price_range_stats.index
        }
    }
    eda_result['price'] = price_stats

    # 시각화: 가격 분석
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 이벤트 타입별 가격 분포
    for event_type in ['view', 'cart', 'purchase']:
        data = train[train['event_type'] == event_type]['price'].clip(upper=500)
        axes[0].hist(data, bins=50, alpha=0.5, label=event_type)
    axes[0].set_title('이벤트 타입별 가격 분포 (500 이하)')
    axes[0].set_xlabel('가격')
    axes[0].set_ylabel('빈도')
    axes[0].legend()

    # 가격대별 이벤트 분포
    price_ranges = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']
    x = np.arange(len(price_ranges))
    width = 0.25

    views = [price_range_stats.loc[pr, 'view'] if pr in price_range_stats.index else 0 for pr in price_ranges]
    carts = [price_range_stats.loc[pr, 'cart'] if pr in price_range_stats.index else 0 for pr in price_ranges]
    purchases = [price_range_stats.loc[pr, 'purchase'] if pr in price_range_stats.index else 0 for pr in price_ranges]

    axes[1].bar(x - width, views, width, label='view', color='#3498db')
    axes[1].bar(x, carts, width, label='cart', color='#e74c3c')
    axes[1].bar(x + width, purchases, width, label='purchase', color='#2ecc71')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(price_ranges)
    axes[1].set_title('가격대별 이벤트 분포')
    axes[1].set_xlabel('가격대')
    axes[1].set_ylabel('횟수')
    axes[1].legend()

    save_fig('06_price_analysis')

    # =========================================================================
    # 8. 시간 분석
    # =========================================================================
    print('\n[9/10] 시간 분석...')

    train['date'] = train['event_time'].dt.date
    train['hour'] = train['event_time'].dt.hour
    train['dayofweek'] = train['event_time'].dt.dayofweek
    train['month'] = train['event_time'].dt.month

    daily_counts = train.groupby('date').size()
    hourly_counts = train.groupby('hour').size()
    dow_counts = train.groupby('dayofweek').size()

    # 요일별 구매 전환율
    dow_purchase = train[train['event_type'] == 'purchase'].groupby('dayofweek').size()
    dow_view = train[train['event_type'] == 'view'].groupby('dayofweek').size()
    dow_conversion = (dow_purchase / dow_view * 100).fillna(0)

    time_stats = {
        'daily_interactions': {
            'min': int(daily_counts.min()),
            'max': int(daily_counts.max()),
            'mean': float(daily_counts.mean()),
            'std': float(daily_counts.std())
        },
        'hourly_distribution': {int(k): int(v) for k, v in hourly_counts.items()},
        'dayofweek_distribution': {
            ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][k]: int(v)
            for k, v in dow_counts.items()
        },
        'dayofweek_conversion': {
            ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][k]: float(v)
            for k, v in dow_conversion.items()
        }
    }
    eda_result['time'] = time_stats

    # 시각화: 시간 분석
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 일별 추이
    axes[0, 0].plot(daily_counts.index, daily_counts.values, color='#3498db')
    axes[0, 0].set_title('일별 상호작용 추이')
    axes[0, 0].set_xlabel('날짜')
    axes[0, 0].set_ylabel('상호작용 수')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 시간대별 분포
    axes[0, 1].bar(hourly_counts.index, hourly_counts.values, color='#e74c3c')
    axes[0, 1].set_title('시간대별 상호작용 분포')
    axes[0, 1].set_xlabel('시간 (0-23)')
    axes[0, 1].set_ylabel('상호작용 수')

    # 요일별 분포
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].bar(dow_labels, [dow_counts.get(i, 0) for i in range(7)], color='#2ecc71')
    axes[1, 0].set_title('요일별 상호작용 분포')
    axes[1, 0].set_xlabel('요일')
    axes[1, 0].set_ylabel('상호작용 수')

    # 요일별 전환율
    axes[1, 1].bar(dow_labels, [dow_conversion.get(i, 0) for i in range(7)], color='#9b59b6')
    axes[1, 1].set_title('요일별 구매 전환율')
    axes[1, 1].set_xlabel('요일')
    axes[1, 1].set_ylabel('전환율 (%)')

    save_fig('07_time_analysis')

    # =========================================================================
    # 9. 세션 분석
    # =========================================================================
    print('\n[10/10] 세션 분석...')

    session_stats_df = train.groupby('user_session').agg({
        'user_id': 'first',
        'item_id': 'nunique',
        'event_type': lambda x: (x == 'purchase').sum()
    }).rename(columns={'item_id': 'unique_items', 'event_type': 'purchases'})

    session_length = train.groupby('user_session').size()

    session_stats = {
        'total_sessions': int(train['user_session'].nunique()),
        'avg_session_length': float(session_length.mean()),
        'median_session_length': float(session_length.median()),
        'max_session_length': int(session_length.max()),
        'sessions_with_purchase': int((session_stats_df['purchases'] > 0).sum()),
        'session_purchase_ratio': float((session_stats_df['purchases'] > 0).mean() * 100),
        'avg_unique_items_per_session': float(session_stats_df['unique_items'].mean())
    }
    eda_result['session'] = session_stats

    # =========================================================================
    # 10. Cold Start 분석 (매우 중요!)
    # =========================================================================
    print('\n[BONUS] Cold Start 분석...')

    # 5회 미만 상호작용 유저/아이템 (SASRec 필터링 기준)
    users_under_5 = (user_interactions < 5).sum()
    items_under_5 = (item_interactions < 5).sum()

    cold_start_stats = {
        'users_with_less_than_5_interactions': {
            'count': int(users_under_5),
            'ratio': float(users_under_5 / len(user_interactions) * 100)
        },
        'items_with_less_than_5_interactions': {
            'count': int(items_under_5),
            'ratio': float(items_under_5 / len(item_interactions) * 100)
        },
        'filtered_interactions_estimate': {
            'users_remaining': int(len(user_interactions) - users_under_5),
            'items_remaining': int(len(item_interactions) - items_under_5)
        }
    }
    eda_result['cold_start'] = cold_start_stats

    # =========================================================================
    # 11. 모델 개선을 위한 인사이트
    # =========================================================================
    insights = {
        'data_quality': {
            'null_category_issue': category_stats['null_category_ratio'] > 10,
            'null_brand_issue': brand_stats['null_brand_ratio'] > 10
        },
        'event_weighting_suggestion': {
            'view_weight': 1,
            'cart_weight': 3,  # cart가 view보다 강한 신호
            'purchase_weight': 5  # purchase가 가장 강한 신호
        },
        'cold_start_severity': {
            'is_severe': cold_start_stats['users_with_less_than_5_interactions']['ratio'] > 30,
            'affected_users_ratio': cold_start_stats['users_with_less_than_5_interactions']['ratio']
        },
        'long_tail_issue': {
            'top_10_percent_items_cover': float(
                item_interactions.nlargest(int(len(item_interactions) * 0.1)).sum() / item_interactions.sum() * 100
            )
        }
    }
    eda_result['insights'] = insights

    # =========================================================================
    # JSON 저장
    # =========================================================================
    json_path = os.path.join(OUTPUT_DIR, 'eda_result.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(eda_result, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 60)
    print('EDA 완료!')
    print(f'결과 저장 위치: {OUTPUT_DIR}/')
    print('  - eda_result.json (분석 결과)')
    print('  - 01~07_*.png (시각화)')
    print('=' * 60)

if __name__ == '__main__':
    main()
