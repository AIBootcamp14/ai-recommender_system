"""
Market Basket Analysis (장바구니 분석)
- 세션 내 동시 조회/구매 아이템 분석
- 연관 규칙 마이닝 (Support, Confidence, Lift)
- 카테고리 간 연관성 분석
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import defaultdict

# 스타일 설정
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def load_data(dir_path, data_dir):
    """데이터 로드"""
    print("=" * 50)
    print("Data Loading...")
    print("=" * 50)

    df = pd.read_parquet(os.path.join(dir_path, data_dir))
    df['event_time'] = pd.to_datetime(df['event_time'])

    print(f"  Total interactions: {len(df):,}")
    print(f"  Users: {df['user_id'].nunique():,}")
    print(f"  Items: {df['item_id'].nunique():,}")

    # Event type 분포
    print("\n  Event Type Distribution:")
    for event, count in df['event_type'].value_counts().items():
        print(f"    {event}: {count:,} ({count/len(df)*100:.2f}%)")

    return df


def analyze_session_baskets(df, output_dir):
    """세션 기반 장바구니 분석"""
    print("\n" + "=" * 50)
    print("[1/4] Session-based Basket Analysis")
    print("=" * 50)

    if 'user_session' not in df.columns:
        print("  No session info - using user_id + date as session")
        df['session'] = df['user_id'].astype(str) + '_' + df['event_time'].dt.date.astype(str)
    else:
        df['session'] = df['user_session']

    # 세션별 아이템 집합
    session_items = df.groupby('session')['item_id'].apply(set).reset_index()
    session_items['basket_size'] = session_items['item_id'].apply(len)

    print(f"\n  Total sessions: {len(session_items):,}")
    print(f"  Avg basket size: {session_items['basket_size'].mean():.2f}")
    print(f"  Median basket size: {session_items['basket_size'].median():.0f}")
    print(f"  Max basket size: {session_items['basket_size'].max()}")

    # 장바구니 크기 분포
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 장바구니 크기 히스토그램
    basket_dist = session_items['basket_size'].value_counts().sort_index().head(20)
    axes[0].bar(basket_dist.index, basket_dist.values, color='#3498db')
    axes[0].set_xlabel('Basket Size (items)')
    axes[0].set_ylabel('Number of Sessions')
    axes[0].set_title('Basket Size Distribution')

    # 2. 누적 분포
    size_counts = session_items['basket_size'].value_counts().sort_index()
    cumsum = size_counts.cumsum() / size_counts.sum() * 100
    axes[1].plot(cumsum.index[:50], cumsum.values[:50], 'b-', linewidth=2)
    axes[1].axhline(90, color='red', linestyle='--', alpha=0.7, label='90%')
    axes[1].set_xlabel('Basket Size')
    axes[1].set_ylabel('Cumulative %')
    axes[1].set_title('Cumulative Basket Size Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_basket_size.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_basket_size.png")

    return session_items


def find_frequent_itemsets(df, min_support=0.001, max_items=2):
    """빈발 아이템셋 찾기 (Apriori 간소화 버전)"""
    print("\n" + "=" * 50)
    print("[2/4] Frequent Itemsets Analysis")
    print("=" * 50)

    if 'user_session' not in df.columns:
        df['session'] = df['user_id'].astype(str) + '_' + df['event_time'].dt.date.astype(str)
    else:
        df['session'] = df['user_session']

    # 세션별 아이템 집합
    sessions = df.groupby('session')['item_id'].apply(set).tolist()
    n_sessions = len(sessions)

    print(f"  Total sessions: {n_sessions:,}")
    print(f"  Min support threshold: {min_support} ({int(min_support * n_sessions)} sessions)")

    # 1-itemset 빈도
    item_counts = defaultdict(int)
    for session in sessions:
        for item in session:
            item_counts[item] += 1

    # Support 계산
    item_support = {item: count / n_sessions for item, count in item_counts.items()}
    frequent_1 = {item: sup for item, sup in item_support.items() if sup >= min_support}

    print(f"\n  Frequent 1-itemsets: {len(frequent_1):,}")

    # 2-itemset (pair) 빈도
    if max_items >= 2:
        pair_counts = defaultdict(int)
        for session in sessions:
            if len(session) >= 2:
                session_list = list(session)
                for i in range(len(session_list)):
                    for j in range(i + 1, len(session_list)):
                        pair = tuple(sorted([session_list[i], session_list[j]]))
                        pair_counts[pair] += 1

        pair_support = {pair: count / n_sessions for pair, count in pair_counts.items()}
        frequent_2 = {pair: sup for pair, sup in pair_support.items() if sup >= min_support}

        print(f"  Frequent 2-itemsets: {len(frequent_2):,}")
    else:
        frequent_2 = {}

    return frequent_1, frequent_2, item_support, n_sessions


def calculate_association_rules(frequent_2, item_support, n_sessions, min_confidence=0.1, top_n=30):
    """연관 규칙 계산 (Confidence, Lift)"""
    print("\n" + "=" * 50)
    print("[3/4] Association Rules")
    print("=" * 50)

    rules = []

    for (item_a, item_b), support_ab in frequent_2.items():
        support_a = item_support.get(item_a, 0)
        support_b = item_support.get(item_b, 0)

        if support_a > 0 and support_b > 0:
            # A -> B
            conf_a_to_b = support_ab / support_a
            lift_a_to_b = conf_a_to_b / support_b

            if conf_a_to_b >= min_confidence:
                rules.append({
                    'antecedent': item_a,
                    'consequent': item_b,
                    'support': support_ab,
                    'confidence': conf_a_to_b,
                    'lift': lift_a_to_b
                })

            # B -> A
            conf_b_to_a = support_ab / support_b
            lift_b_to_a = conf_b_to_a / support_a

            if conf_b_to_a >= min_confidence:
                rules.append({
                    'antecedent': item_b,
                    'consequent': item_a,
                    'support': support_ab,
                    'confidence': conf_b_to_a,
                    'lift': lift_b_to_a
                })

    rules_df = pd.DataFrame(rules)

    if len(rules_df) > 0:
        rules_df = rules_df.sort_values('lift', ascending=False)
        print(f"\n  Total rules (conf >= {min_confidence}): {len(rules_df):,}")
        print(f"\n  Top {min(top_n, len(rules_df))} rules by Lift:")
        print("-" * 80)

        for i, row in rules_df.head(top_n).iterrows():
            ant = str(row['antecedent'])[:20]
            cons = str(row['consequent'])[:20]
            print(f"  {ant:>20} -> {cons:<20} | sup={row['support']:.4f} conf={row['confidence']:.3f} lift={row['lift']:.2f}")
    else:
        print("  No rules found with given thresholds")

    return rules_df


def analyze_category_associations(df, output_dir, min_support=0.005):
    """카테고리 간 연관성 분석"""
    print("\n" + "=" * 50)
    print("[4/4] Category Association Analysis")
    print("=" * 50)

    if 'category_code' not in df.columns:
        print("  No category info - skipping")
        return None

    # 상위 카테고리 추출 (첫 번째 레벨)
    df['category_l1'] = df['category_code'].fillna('unknown').apply(
        lambda x: x.split('.')[0] if pd.notna(x) and '.' in str(x) else str(x)
    )

    if 'user_session' not in df.columns:
        df['session'] = df['user_id'].astype(str) + '_' + df['event_time'].dt.date.astype(str)
    else:
        df['session'] = df['user_session']

    # 세션별 카테고리 집합
    sessions = df.groupby('session')['category_l1'].apply(set).tolist()
    n_sessions = len(sessions)

    # 카테고리 쌍 빈도
    cat_pair_counts = defaultdict(int)
    cat_counts = defaultdict(int)

    for session in sessions:
        for cat in session:
            cat_counts[cat] += 1
        if len(session) >= 2:
            for cat1, cat2 in combinations(session, 2):
                pair = tuple(sorted([cat1, cat2]))
                cat_pair_counts[pair] += 1

    # Co-occurrence matrix 생성
    categories = list(cat_counts.keys())
    n_cats = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    cooc_matrix = np.zeros((n_cats, n_cats))
    for (cat1, cat2), count in cat_pair_counts.items():
        i, j = cat_to_idx[cat1], cat_to_idx[cat2]
        cooc_matrix[i, j] = count
        cooc_matrix[j, i] = count

    # 대각선에 단일 카테고리 빈도
    for cat, count in cat_counts.items():
        cooc_matrix[cat_to_idx[cat], cat_to_idx[cat]] = count

    # 상위 카테고리만 시각화
    top_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    top_cat_names = [c[0] for c in top_cats]
    top_cat_idx = [cat_to_idx[c] for c in top_cat_names]

    cooc_sub = cooc_matrix[np.ix_(top_cat_idx, top_cat_idx)]

    # Lift matrix 계산
    lift_matrix = np.zeros_like(cooc_sub)
    for i in range(len(top_cat_names)):
        for j in range(len(top_cat_names)):
            if i != j:
                sup_i = cat_counts[top_cat_names[i]] / n_sessions
                sup_j = cat_counts[top_cat_names[j]] / n_sessions
                sup_ij = cooc_sub[i, j] / n_sessions
                if sup_i > 0 and sup_j > 0:
                    lift_matrix[i, j] = sup_ij / (sup_i * sup_j)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Co-occurrence heatmap
    sns.heatmap(cooc_sub, xticklabels=top_cat_names, yticklabels=top_cat_names,
                cmap='Blues', ax=axes[0], fmt='.0f',
                cbar_kws={'label': 'Co-occurrence Count'})
    axes[0].set_title('Category Co-occurrence Matrix')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # 2. Lift heatmap
    mask = np.eye(len(top_cat_names), dtype=bool)
    sns.heatmap(lift_matrix, xticklabels=top_cat_names, yticklabels=top_cat_names,
                cmap='RdYlGn', center=1, ax=axes[1], mask=mask,
                cbar_kws={'label': 'Lift'})
    axes[1].set_title('Category Lift Matrix (Lift > 1: positive association)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_category_association.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 07_category_association.png")

    # 상위 카테고리 연관 규칙 출력
    print("\n  Top Category Associations (by Lift):")
    print("-" * 60)

    cat_rules = []
    for i in range(len(top_cat_names)):
        for j in range(len(top_cat_names)):
            if i != j and lift_matrix[i, j] > 1:
                cat_rules.append({
                    'from': top_cat_names[i],
                    'to': top_cat_names[j],
                    'lift': lift_matrix[i, j],
                    'cooc': cooc_sub[i, j]
                })

    cat_rules = sorted(cat_rules, key=lambda x: x['lift'], reverse=True)[:15]
    for rule in cat_rules:
        print(f"  {rule['from']:>20} -> {rule['to']:<20} | lift={rule['lift']:.2f} cooc={rule['cooc']:.0f}")

    return cat_rules


def analyze_purchase_patterns(df, output_dir):
    """구매/장바구니 패턴 분석"""
    print("\n" + "=" * 50)
    print("[Bonus] Purchase/Cart Pattern Analysis")
    print("=" * 50)

    # Cart/Purchase 이벤트만 필터
    cart_purchase = df[df['event_type'].isin(['cart', 'purchase'])]

    if len(cart_purchase) == 0:
        print("  No cart/purchase events")
        return

    print(f"  Cart events: {len(df[df['event_type'] == 'cart']):,}")
    print(f"  Purchase events: {len(df[df['event_type'] == 'purchase']):,}")

    # 사용자별 구매 아이템
    if 'user_session' in df.columns:
        purchase_baskets = cart_purchase.groupby('user_session')['item_id'].apply(list)
    else:
        purchase_baskets = cart_purchase.groupby('user_id')['item_id'].apply(list)

    basket_sizes = purchase_baskets.apply(len)

    print(f"\n  Purchase baskets: {len(purchase_baskets):,}")
    print(f"  Avg items per basket: {basket_sizes.mean():.2f}")
    print(f"  Single item baskets: {(basket_sizes == 1).sum():,} ({(basket_sizes == 1).mean()*100:.1f}%)")

    # 자주 구매되는 아이템
    purchase_items = df[df['event_type'] == 'purchase']['item_id'].value_counts()
    print(f"\n  Top 10 purchased items:")
    for item, count in purchase_items.head(10).items():
        print(f"    {str(item)[:30]}: {count} purchases")


def main():
    parser = argparse.ArgumentParser(description='Market Basket Analysis')
    parser.add_argument('--dir_path', type=str, default='../../data/', help='Data path')
    parser.add_argument('--data_dir', type=str, default='train.parquet', help='Data file')
    parser.add_argument('--output_dir', type=str, default='./', help='Output path')
    parser.add_argument('--min_support', type=float, default=0.001, help='Min support')
    parser.add_argument('--min_confidence', type=float, default=0.1, help='Min confidence')
    args = parser.parse_args()

    # 출력 디렉토리 확인
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 데이터 로드
    df = load_data(args.dir_path, args.data_dir)

    # 1. 세션 기반 장바구니 분석
    session_items = analyze_session_baskets(df, args.output_dir)

    # 2. 빈발 아이템셋 찾기
    frequent_1, frequent_2, item_support, n_sessions = find_frequent_itemsets(
        df, min_support=args.min_support
    )

    # 3. 연관 규칙 계산
    if len(frequent_2) > 0:
        rules_df = calculate_association_rules(
            frequent_2, item_support, n_sessions,
            min_confidence=args.min_confidence
        )

        # 규칙 저장
        if len(rules_df) > 0:
            rules_df.to_csv(os.path.join(args.output_dir, 'association_rules.csv'), index=False)
            print(f"\n  Saved: association_rules.csv")

    # 4. 카테고리 연관성 분석
    analyze_category_associations(df, args.output_dir)

    # 5. 구매 패턴 분석
    analyze_purchase_patterns(df, args.output_dir)

    print("\n" + "=" * 50)
    print("Basket Analysis Complete!")
    print("=" * 50)
    print(f"  Output: {args.output_dir}")


if __name__ == '__main__':
    main()
