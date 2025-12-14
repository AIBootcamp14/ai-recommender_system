"""
EDA 분석 및 시각화
- 데이터 기본 통계
- 사용자/아이템 분포
- 시간 패턴 분석
- Event type 분석
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 폰트 설정 (영문 사용으로 호환성 보장)
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data(dir_path, data_dir):
    """데이터 로드"""
    print("=" * 50)
    print("데이터 로드 중...")
    print("=" * 50)

    df = pd.read_parquet(os.path.join(dir_path, data_dir))
    df['event_time'] = pd.to_datetime(df['event_time'])

    print(f"  총 상호작용: {len(df):,}")
    print(f"  사용자 수: {df['user_id'].nunique():,}")
    print(f"  아이템 수: {df['item_id'].nunique():,}")
    print(f"  기간: {df['event_time'].min().date()} ~ {df['event_time'].max().date()}")

    return df


def plot_basic_stats(df, output_dir):
    """기본 통계 시각화"""
    print("\n[1/5] 기본 통계 시각화...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Event type 분포
    event_counts = df['event_type'].value_counts()
    axes[0].pie(event_counts.values, labels=event_counts.index, autopct='%1.2f%%',
                colors=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_title('Event Type Distribution')

    # 2. 사용자당 상호작용 수
    user_counts = df.groupby('user_id').size()
    axes[1].hist(user_counts.values, bins=50, color='#3498db', edgecolor='white', log=True)
    axes[1].set_xlabel('Interactions')
    axes[1].set_ylabel('Users (log scale)')
    axes[1].set_title(f'User Interaction Distribution\n(Mean: {user_counts.mean():.1f}, Median: {user_counts.median():.0f})')
    axes[1].axvline(user_counts.mean(), color='red', linestyle='--', label=f'Mean: {user_counts.mean():.1f}')
    axes[1].legend()

    # 3. 아이템당 상호작용 수
    item_counts = df.groupby('item_id').size()
    axes[2].hist(item_counts.values, bins=50, color='#2ecc71', edgecolor='white', log=True)
    axes[2].set_xlabel('Interactions')
    axes[2].set_ylabel('Items (log scale)')
    axes[2].set_title(f'Item Interaction Distribution\n(Mean: {item_counts.mean():.1f}, Median: {item_counts.median():.0f})')
    axes[2].axvline(item_counts.mean(), color='red', linestyle='--', label=f'Mean: {item_counts.mean():.1f}')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_basic_stats.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: 01_basic_stats.png")


def plot_time_patterns(df, output_dir):
    """시간 패턴 분석"""
    print("\n[2/5] 시간 패턴 분석...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 일별 상호작용
    daily = df.groupby(df['event_time'].dt.date).size()
    axes[0, 0].plot(daily.index, daily.values, color='#3498db', linewidth=1)
    axes[0, 0].fill_between(daily.index, daily.values, alpha=0.3, color='#3498db')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Interactions')
    axes[0, 0].set_title('Daily Interactions')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. 요일별 상호작용
    df['dayofweek'] = df['event_time'].dt.dayofweek
    dayofweek_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    df['dayofweek_name'] = df['dayofweek'].map(dayofweek_map)

    dow_counts = df.groupby('dayofweek').size()
    colors = ['#3498db'] * 5 + ['#e74c3c', '#e74c3c']  # 주말 강조
    axes[0, 1].bar([dayofweek_map[i] for i in range(7)], dow_counts.values, color=colors)
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Interactions')
    axes[0, 1].set_title('Interactions by Day of Week')

    # 주말 비율 표시
    weekend_ratio = (dow_counts[5] + dow_counts[6]) / dow_counts.sum() * 100
    axes[0, 1].annotate(f'Weekend: {weekend_ratio:.1f}%', xy=(5.5, dow_counts.max() * 0.9),
                        fontsize=12, color='#e74c3c', fontweight='bold')

    # 3. 시간대별 상호작용
    df['hour'] = df['event_time'].dt.hour
    hour_counts = df.groupby('hour').size()
    axes[1, 0].bar(hour_counts.index, hour_counts.values, color='#2ecc71')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Interactions')
    axes[1, 0].set_title('Interactions by Hour')
    axes[1, 0].set_xticks(range(0, 24, 2))

    # 피크 시간 표시
    peak_hour = hour_counts.idxmax()
    axes[1, 0].axvline(peak_hour, color='red', linestyle='--', label=f'Peak: {peak_hour}h')
    axes[1, 0].legend()

    # 4. 월별 상호작용
    df['month'] = df['event_time'].dt.to_period('M')
    month_counts = df.groupby('month').size()
    axes[1, 1].bar([str(m) for m in month_counts.index], month_counts.values, color='#9b59b6')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Interactions')
    axes[1, 1].set_title('Monthly Interactions')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_time_patterns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: 02_time_patterns.png")


def plot_long_tail(df, output_dir):
    """Long-tail 분석"""
    print("\n[3/5] Long-tail 분석...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 아이템 인기도 순위
    item_counts = df.groupby('item_id').size().sort_values(ascending=False).reset_index()
    item_counts.columns = ['item_id', 'count']
    item_counts['rank'] = range(1, len(item_counts) + 1)
    item_counts['cumsum'] = item_counts['count'].cumsum()
    item_counts['cumsum_pct'] = item_counts['cumsum'] / item_counts['count'].sum() * 100

    # 1. 아이템 인기도 분포 (log-log)
    axes[0].loglog(item_counts['rank'], item_counts['count'], 'b-', linewidth=1)
    axes[0].set_xlabel('Item Rank (log)')
    axes[0].set_ylabel('Interactions (log)')
    axes[0].set_title('Item Popularity Distribution (Long-tail)')
    axes[0].grid(True, alpha=0.3)

    # 2. 누적 분포
    axes[1].plot(item_counts['rank'] / len(item_counts) * 100, item_counts['cumsum_pct'],
                 'b-', linewidth=2)
    axes[1].set_xlabel('Item Percentile (%)')
    axes[1].set_ylabel('Cumulative Interactions (%)')
    axes[1].set_title('Cumulative Item Interactions')

    # 파레토 라인 (20-80)
    axes[1].axvline(20, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(80, color='red', linestyle='--', alpha=0.7)

    # 1%, 20% 지점 표시
    top1_pct = item_counts[item_counts['rank'] <= len(item_counts) * 0.01]['cumsum_pct'].max()
    top20_pct = item_counts[item_counts['rank'] <= len(item_counts) * 0.20]['cumsum_pct'].max()

    axes[1].annotate(f'Top 1%: {top1_pct:.1f}%', xy=(1, top1_pct),
                    xytext=(5, top1_pct + 5), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='green'))
    axes[1].annotate(f'Top 20%: {top20_pct:.1f}%', xy=(20, top20_pct),
                    xytext=(25, top20_pct - 10), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='red'))

    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_long_tail.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: 03_long_tail.png")


def plot_session_analysis(df, output_dir):
    """세션 분석"""
    print("\n[4/5] 세션 분석...")

    if 'user_session' not in df.columns:
        print("  세션 정보 없음 - 스킵")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 세션 길이 분포
    session_lengths = df.groupby('user_session').size()

    # 1. 세션 길이 히스토그램
    axes[0].hist(session_lengths.values, bins=50, color='#3498db', edgecolor='white', log=True)
    axes[0].set_xlabel('Items per Session')
    axes[0].set_ylabel('Sessions (log scale)')
    axes[0].set_title(f'Session Length Distribution\n(Mean: {session_lengths.mean():.1f}, Median: {session_lengths.median():.0f})')

    # 2. 세션 길이별 비율
    length_bins = [1, 2, 3, 5, 10, 20, float('inf')]
    length_labels = ['1', '2', '3-4', '5-9', '10-19', '20+']
    session_binned = pd.cut(session_lengths, bins=length_bins, labels=length_labels, right=False)
    session_dist = session_binned.value_counts().sort_index()

    axes[1].bar(session_dist.index, session_dist.values / session_dist.sum() * 100, color='#2ecc71')
    axes[1].set_xlabel('Session Length')
    axes[1].set_ylabel('Ratio (%)')
    axes[1].set_title('Session Length Distribution')

    # 비율 표시
    for i, (idx, val) in enumerate(session_dist.items()):
        pct = val / session_dist.sum() * 100
        axes[1].annotate(f'{pct:.1f}%', xy=(i, pct + 1), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_session_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: 04_session_analysis.png")


def plot_category_analysis(df, output_dir):
    """카테고리 분석"""
    print("\n[5/5] 카테고리 분석...")

    if 'category_code' not in df.columns:
        print("  카테고리 정보 없음 - 스킵")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 상위 카테고리 분포
    cat_counts = df['category_code'].value_counts().head(15)
    axes[0].barh(range(len(cat_counts)), cat_counts.values, color='#3498db')
    axes[0].set_yticks(range(len(cat_counts)))
    axes[0].set_yticklabels([str(c)[:30] for c in cat_counts.index])
    axes[0].set_xlabel('Interactions')
    axes[0].set_title('Top 15 Categories')
    axes[0].invert_yaxis()

    # 2. 브랜드 분포
    if 'brand' in df.columns:
        brand_counts = df['brand'].value_counts().head(15)
        axes[1].barh(range(len(brand_counts)), brand_counts.values, color='#e74c3c')
        axes[1].set_yticks(range(len(brand_counts)))
        axes[1].set_yticklabels([str(b)[:20] for b in brand_counts.index])
        axes[1].set_xlabel('Interactions')
        axes[1].set_title('Top 15 Brands')
        axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_category_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  저장: 05_category_analysis.png")


def print_summary(df):
    """요약 통계 출력"""
    print("\n" + "=" * 50)
    print("EDA 요약")
    print("=" * 50)

    # 기본 통계
    print("\n[기본 통계]")
    print(f"  총 상호작용: {len(df):,}")
    print(f"  고유 사용자: {df['user_id'].nunique():,}")
    print(f"  고유 아이템: {df['item_id'].nunique():,}")

    # Sparsity
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    sparsity = (1 - len(df) / (n_users * n_items)) * 100
    print(f"  Sparsity: {sparsity:.4f}%")

    # Event type
    print("\n[Event Type 분포]")
    for event, count in df['event_type'].value_counts().items():
        print(f"  {event}: {count:,} ({count/len(df)*100:.2f}%)")

    # 사용자 통계
    user_counts = df.groupby('user_id').size()
    print("\n[사용자 통계]")
    print(f"  평균 상호작용: {user_counts.mean():.1f}")
    print(f"  중앙값: {user_counts.median():.0f}")
    print(f"  최대: {user_counts.max():,}")

    # 아이템 통계
    item_counts = df.groupby('item_id').size()
    print("\n[아이템 통계]")
    print(f"  평균 상호작용: {item_counts.mean():.1f}")
    print(f"  중앙값: {item_counts.median():.0f}")
    print(f"  최대: {item_counts.max():,}")

    # Long-tail
    item_sorted = item_counts.sort_values(ascending=False)
    top1_items = int(len(item_sorted) * 0.01)
    top20_items = int(len(item_sorted) * 0.20)
    top1_ratio = item_sorted.head(top1_items).sum() / item_sorted.sum() * 100
    top20_ratio = item_sorted.head(top20_items).sum() / item_sorted.sum() * 100

    print("\n[Long-tail 분석]")
    print(f"  상위 1% 아이템: {top1_ratio:.1f}% 상호작용")
    print(f"  상위 20% 아이템: {top20_ratio:.1f}% 상호작용")


def main():
    parser = argparse.ArgumentParser(description='EDA 분석 및 시각화')
    parser.add_argument('--dir_path', type=str, default='../../data/', help='데이터 경로')
    parser.add_argument('--data_dir', type=str, default='train.parquet', help='데이터 파일')
    parser.add_argument('--output_dir', type=str, default='./', help='그래프 저장 경로')
    args = parser.parse_args()

    # 출력 디렉토리 확인
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 데이터 로드
    df = load_data(args.dir_path, args.data_dir)

    # 시각화
    plot_basic_stats(df, args.output_dir)
    plot_time_patterns(df, args.output_dir)
    plot_long_tail(df, args.output_dir)
    plot_session_analysis(df, args.output_dir)
    plot_category_analysis(df, args.output_dir)

    # 요약 출력
    print_summary(df)

    print("\n" + "=" * 50)
    print("EDA 완료!")
    print("=" * 50)
    print(f"  그래프 저장 위치: {args.output_dir}")


if __name__ == '__main__':
    main()
