"""
Detailed Data Analysis
- User interaction distribution
- Brand analysis for purchased items
- Brand distribution across all items
- Viewing patterns by brand
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Output directory
OUTPUT_DIR = "./analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Detailed Data Analysis")
print("=" * 70)

# Load data
print("\n[1] Loading data...")
df = pd.read_parquet("../train.parquet")
df['event_time'] = pd.to_datetime(df['event_time'])
print(f"Total records: {len(df):,}")
print(f"Users: {df['user_id'].nunique():,}")
print(f"Items: {df['item_id'].nunique():,}")

results = {}

# =============================================================================
# 1. User Interaction Distribution Analysis
# =============================================================================
print("\n[2] Analyzing user interaction distribution...")

# Count interactions per user by type
user_event_counts = df.groupby(['user_id', 'event_type']).size().unstack(fill_value=0)
user_event_counts['total'] = user_event_counts.sum(axis=1)

# User interaction statistics
user_stats = {
    'total_users': int(df['user_id'].nunique()),
    'interaction_per_user': {
        'mean': float(user_event_counts['total'].mean()),
        'median': float(user_event_counts['total'].median()),
        'std': float(user_event_counts['total'].std()),
        'min': int(user_event_counts['total'].min()),
        'max': int(user_event_counts['total'].max()),
        'percentiles': {
            '10%': float(user_event_counts['total'].quantile(0.1)),
            '25%': float(user_event_counts['total'].quantile(0.25)),
            '50%': float(user_event_counts['total'].quantile(0.5)),
            '75%': float(user_event_counts['total'].quantile(0.75)),
            '90%': float(user_event_counts['total'].quantile(0.9)),
            '95%': float(user_event_counts['total'].quantile(0.95)),
            '99%': float(user_event_counts['total'].quantile(0.99)),
        }
    },
    'users_by_interaction_count': {
        '1': int((user_event_counts['total'] == 1).sum()),
        '2-5': int(((user_event_counts['total'] >= 2) & (user_event_counts['total'] <= 5)).sum()),
        '6-10': int(((user_event_counts['total'] >= 6) & (user_event_counts['total'] <= 10)).sum()),
        '11-20': int(((user_event_counts['total'] >= 11) & (user_event_counts['total'] <= 20)).sum()),
        '21-50': int(((user_event_counts['total'] >= 21) & (user_event_counts['total'] <= 50)).sum()),
        '51-100': int(((user_event_counts['total'] >= 51) & (user_event_counts['total'] <= 100)).sum()),
        '100+': int((user_event_counts['total'] > 100).sum()),
    },
    'event_type_per_user': {}
}

for col in ['view', 'cart', 'purchase']:
    if col in user_event_counts.columns:
        user_stats['event_type_per_user'][col] = {
            'users_with_event': int((user_event_counts[col] > 0).sum()),
            'mean': float(user_event_counts[col].mean()),
            'median': float(user_event_counts[col].median()),
            'max': int(user_event_counts[col].max()),
        }

results['user_interaction_stats'] = user_stats

# Visualization 1: User interaction distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1-1: Distribution of total interactions per user (log scale)
ax1 = axes[0, 0]
interaction_counts = user_event_counts['total'].clip(upper=200)
sns.histplot(interaction_counts, bins=50, ax=ax1, color='steelblue')
ax1.set_xlabel('Interactions per User (capped at 200)')
ax1.set_ylabel('Number of Users')
ax1.set_title('Distribution of User Interactions')
ax1.axvline(x=user_stats['interaction_per_user']['median'], color='red', linestyle='--', label=f"Median: {user_stats['interaction_per_user']['median']:.1f}")
ax1.legend()

# 1-2: User segments by interaction count
ax2 = axes[0, 1]
segments = list(user_stats['users_by_interaction_count'].keys())
counts = list(user_stats['users_by_interaction_count'].values())
colors = sns.color_palette("Blues_r", len(segments))
bars = ax2.bar(segments, counts, color=colors)
ax2.set_xlabel('Interaction Count Range')
ax2.set_ylabel('Number of Users')
ax2.set_title('User Segments by Interaction Count')
ax2.tick_params(axis='x', rotation=45)
for bar, count in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
             f'{count:,}', ha='center', va='bottom', fontsize=8)

# 1-3: Event type breakdown per user
ax3 = axes[1, 0]
event_means = [user_stats['event_type_per_user'].get(e, {}).get('mean', 0) for e in ['view', 'cart', 'purchase']]
colors = ['#3498db', '#f39c12', '#27ae60']
bars = ax3.bar(['View', 'Cart', 'Purchase'], event_means, color=colors)
ax3.set_ylabel('Average Count per User')
ax3.set_title('Average Event Types per User')
for bar, val in zip(bars, event_means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.2f}', ha='center', va='bottom')

# 1-4: Users with each event type
ax4 = axes[1, 1]
users_with_events = [user_stats['event_type_per_user'].get(e, {}).get('users_with_event', 0) for e in ['view', 'cart', 'purchase']]
bars = ax4.bar(['View', 'Cart', 'Purchase'], users_with_events, color=colors)
ax4.set_ylabel('Number of Users')
ax4.set_title('Users with Each Event Type')
for bar, val in zip(bars, users_with_events):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
             f'{val:,}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_user_interaction_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_user_interaction_distribution.png")

# =============================================================================
# 2. Brand Analysis for Purchased Items
# =============================================================================
print("\n[3] Analyzing brands for purchased items...")

purchase_df = df[df['event_type'] == 'purchase'].copy()
purchase_df['brand'] = purchase_df['brand'].fillna('Unknown')

# Brand statistics for purchases
purchase_brand_counts = purchase_df['brand'].value_counts()
total_purchases = len(purchase_df)

purchase_brand_stats = {
    'total_purchases': int(total_purchases),
    'unique_brands_purchased': int(purchase_brand_counts.nunique()),
    'top_20_brands': {str(brand): int(count) for brand, count in purchase_brand_counts.head(20).items()},
    'brand_concentration': {
        'top_1_share': float(purchase_brand_counts.iloc[0] / total_purchases * 100) if len(purchase_brand_counts) > 0 else 0,
        'top_5_share': float(purchase_brand_counts.head(5).sum() / total_purchases * 100) if len(purchase_brand_counts) >= 5 else 0,
        'top_10_share': float(purchase_brand_counts.head(10).sum() / total_purchases * 100) if len(purchase_brand_counts) >= 10 else 0,
        'top_20_share': float(purchase_brand_counts.head(20).sum() / total_purchases * 100) if len(purchase_brand_counts) >= 20 else 0,
    },
    'brands_with_single_purchase': int((purchase_brand_counts == 1).sum()),
}

results['purchase_brand_stats'] = purchase_brand_stats

# Visualization 2: Purchase brand distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2-1: Top 20 brands by purchase count
ax1 = axes[0]
top20_brands = purchase_brand_counts.head(20)
colors = sns.color_palette("Greens_r", 20)
bars = ax1.barh(range(len(top20_brands)), top20_brands.values, color=colors)
ax1.set_yticks(range(len(top20_brands)))
ax1.set_yticklabels([str(b)[:20] for b in top20_brands.index])
ax1.invert_yaxis()
ax1.set_xlabel('Number of Purchases')
ax1.set_title('Top 20 Brands by Purchase Count')
for i, (bar, count) in enumerate(zip(bars, top20_brands.values)):
    ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{count}', ha='left', va='center', fontsize=8)

# 2-2: Brand concentration pie chart
ax2 = axes[1]
top5_sum = purchase_brand_counts.head(5).sum()
rest_sum = purchase_brand_counts.iloc[5:].sum()
sizes = [top5_sum, rest_sum]
labels = [f'Top 5 Brands\n({top5_sum:,})', f'Other Brands\n({rest_sum:,})']
colors = ['#27ae60', '#bdc3c7']
ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax2.set_title('Purchase Brand Concentration')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_purchase_brand_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_purchase_brand_distribution.png")

# =============================================================================
# 3. Overall Brand Distribution (All Items)
# =============================================================================
print("\n[4] Analyzing overall brand distribution...")

df['brand'] = df['brand'].fillna('Unknown')
all_brand_counts = df.groupby('brand').agg({
    'user_id': 'count',  # total interactions
    'item_id': 'nunique',  # unique items
}).rename(columns={'user_id': 'interactions', 'item_id': 'unique_items'})
all_brand_counts = all_brand_counts.sort_values('interactions', ascending=False)

# Brand event type breakdown
brand_events = df.groupby(['brand', 'event_type']).size().unstack(fill_value=0)
brand_events = brand_events.reindex(all_brand_counts.index)

overall_brand_stats = {
    'total_brands': int(df['brand'].nunique()),
    'top_20_brands_by_interaction': {},
    'brand_item_coverage': {
        'top_10_brands_items': int(all_brand_counts.head(10)['unique_items'].sum()),
        'top_20_brands_items': int(all_brand_counts.head(20)['unique_items'].sum()),
        'total_items': int(df['item_id'].nunique()),
    }
}

for brand in all_brand_counts.head(20).index:
    overall_brand_stats['top_20_brands_by_interaction'][str(brand)] = {
        'interactions': int(all_brand_counts.loc[brand, 'interactions']),
        'unique_items': int(all_brand_counts.loc[brand, 'unique_items']),
        'views': int(brand_events.loc[brand, 'view']) if 'view' in brand_events.columns else 0,
        'carts': int(brand_events.loc[brand, 'cart']) if 'cart' in brand_events.columns else 0,
        'purchases': int(brand_events.loc[brand, 'purchase']) if 'purchase' in brand_events.columns else 0,
    }

results['overall_brand_stats'] = overall_brand_stats

# Visualization 3: Overall brand distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 3-1: Top 20 brands by total interactions
ax1 = axes[0]
top20_all = all_brand_counts.head(20)
colors = sns.color_palette("Blues_r", 20)
bars = ax1.barh(range(len(top20_all)), top20_all['interactions'].values, color=colors)
ax1.set_yticks(range(len(top20_all)))
ax1.set_yticklabels([str(b)[:20] for b in top20_all.index])
ax1.invert_yaxis()
ax1.set_xlabel('Total Interactions')
ax1.set_title('Top 20 Brands by Total Interactions')

# 3-2: Event type breakdown for top 10 brands
ax2 = axes[1]
top10_brands = all_brand_counts.head(10).index.tolist()
brand_event_data = brand_events.loc[top10_brands][['view', 'cart', 'purchase']]
# Normalize to percentage for visibility
brand_event_pct = brand_event_data.div(brand_event_data.sum(axis=1), axis=0) * 100
brand_event_pct.plot(kind='barh', stacked=True, ax=ax2, color=['#3498db', '#f39c12', '#27ae60'])
ax2.set_xlabel('Percentage')
ax2.set_title('Event Type Breakdown for Top 10 Brands')
ax2.legend(title='Event Type', loc='lower right')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_overall_brand_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_overall_brand_distribution.png")

# =============================================================================
# 4. Brand Viewing Patterns Analysis
# =============================================================================
print("\n[5] Analyzing brand viewing patterns...")

# View duration proxy: count of views per session per brand
view_df = df[df['event_type'] == 'view'].copy()

# Extract hour from event_time
view_df['hour'] = view_df['event_time'].dt.hour
view_df['day_of_week'] = view_df['event_time'].dt.dayofweek
view_df['date'] = view_df['event_time'].dt.date

# Brand viewing patterns
brand_view_patterns = {}
top_brands = all_brand_counts.head(10).index.tolist()

for brand in top_brands:
    brand_views = view_df[view_df['brand'] == brand]

    if len(brand_views) == 0:
        continue

    # Views per user
    views_per_user = brand_views.groupby('user_id').size()

    # Hourly distribution
    hourly_dist = brand_views['hour'].value_counts().sort_index()

    # Day of week distribution
    dow_dist = brand_views['day_of_week'].value_counts().sort_index()

    brand_view_patterns[str(brand)] = {
        'total_views': int(len(brand_views)),
        'unique_viewers': int(brand_views['user_id'].nunique()),
        'views_per_user': {
            'mean': float(views_per_user.mean()),
            'median': float(views_per_user.median()),
            'max': int(views_per_user.max()),
        },
        'peak_hour': int(hourly_dist.idxmax()),
        'peak_day': int(dow_dist.idxmax()),  # 0=Monday, 6=Sunday
        'hourly_distribution': {str(h): int(c) for h, c in hourly_dist.items()},
    }

results['brand_view_patterns'] = brand_view_patterns

# Visualization 4: Brand viewing patterns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4-1: Views per user by brand
ax1 = axes[0, 0]
views_per_user_data = {b: brand_view_patterns[b]['views_per_user']['mean']
                       for b in brand_view_patterns.keys()}
brands = list(views_per_user_data.keys())
values = list(views_per_user_data.values())
colors = sns.color_palette("Oranges_r", len(brands))
bars = ax1.barh(brands, values, color=colors)
ax1.set_xlabel('Average Views per User')
ax1.set_title('Average Views per User by Brand (Top 10)')
ax1.invert_yaxis()

# 4-2: Hourly viewing pattern heatmap
ax2 = axes[0, 1]
hourly_data = pd.DataFrame({
    brand: pd.Series(brand_view_patterns[brand]['hourly_distribution']).astype(float)
    for brand in list(brand_view_patterns.keys())[:8]
}).T.fillna(0)
# Normalize rows
hourly_data_norm = hourly_data.div(hourly_data.sum(axis=1), axis=0)
sns.heatmap(hourly_data_norm, cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Proportion'})
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Brand')
ax2.set_title('Hourly View Distribution by Brand')

# 4-3: Unique viewers vs Total views
ax3 = axes[1, 0]
x_data = [brand_view_patterns[b]['unique_viewers'] for b in brand_view_patterns.keys()]
y_data = [brand_view_patterns[b]['total_views'] for b in brand_view_patterns.keys()]
labels = list(brand_view_patterns.keys())
ax3.scatter(x_data, y_data, s=100, alpha=0.7, color='steelblue')
for i, label in enumerate(labels):
    ax3.annotate(label[:15], (x_data[i], y_data[i]), fontsize=8, ha='left')
ax3.set_xlabel('Unique Viewers')
ax3.set_ylabel('Total Views')
ax3.set_title('Unique Viewers vs Total Views by Brand')

# 4-4: Day of week pattern
ax4 = axes[1, 1]
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_totals = view_df['day_of_week'].value_counts().sort_index()
colors = sns.color_palette("Purples_r", 7)
bars = ax4.bar(dow_names, dow_totals.values, color=colors)
ax4.set_xlabel('Day of Week')
ax4.set_ylabel('Total Views')
ax4.set_title('Overall View Distribution by Day of Week')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_brand_viewing_patterns.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_brand_viewing_patterns.png")

# =============================================================================
# 5. User Journey Analysis (View -> Cart -> Purchase)
# =============================================================================
print("\n[6] Analyzing user journey patterns...")

# Users who made purchases
purchase_users = df[df['event_type'] == 'purchase']['user_id'].unique()

journey_stats = {
    'total_purchasing_users': int(len(purchase_users)),
    'conversion_funnel': {
        'total_users': int(df['user_id'].nunique()),
        'users_with_view': int(df[df['event_type'] == 'view']['user_id'].nunique()),
        'users_with_cart': int(df[df['event_type'] == 'cart']['user_id'].nunique()),
        'users_with_purchase': int(len(purchase_users)),
    },
    'avg_events_before_purchase': {},
}

# For purchasing users, count events before purchase
for user in purchase_users[:1000]:  # Sample for performance
    user_df = df[df['user_id'] == user].sort_values('event_time')
    first_purchase_time = user_df[user_df['event_type'] == 'purchase']['event_time'].min()
    events_before = user_df[user_df['event_time'] < first_purchase_time]

    for event_type in ['view', 'cart']:
        if event_type not in journey_stats['avg_events_before_purchase']:
            journey_stats['avg_events_before_purchase'][event_type] = []
        journey_stats['avg_events_before_purchase'][event_type].append(
            len(events_before[events_before['event_type'] == event_type])
        )

# Calculate averages
for event_type in journey_stats['avg_events_before_purchase']:
    values = journey_stats['avg_events_before_purchase'][event_type]
    journey_stats['avg_events_before_purchase'][event_type] = {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
    }

# Conversion rates
funnel = journey_stats['conversion_funnel']
journey_stats['conversion_rates'] = {
    'view_to_cart': float(funnel['users_with_cart'] / funnel['users_with_view'] * 100) if funnel['users_with_view'] > 0 else 0,
    'cart_to_purchase': float(funnel['users_with_purchase'] / funnel['users_with_cart'] * 100) if funnel['users_with_cart'] > 0 else 0,
    'view_to_purchase': float(funnel['users_with_purchase'] / funnel['users_with_view'] * 100) if funnel['users_with_view'] > 0 else 0,
}

results['user_journey_stats'] = journey_stats

# Visualization 5: User journey
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5-1: Conversion funnel
ax1 = axes[0]
funnel_stages = ['View', 'Cart', 'Purchase']
funnel_values = [funnel['users_with_view'], funnel['users_with_cart'], funnel['users_with_purchase']]
colors = ['#3498db', '#f39c12', '#27ae60']
bars = ax1.bar(funnel_stages, funnel_values, color=colors)
ax1.set_ylabel('Number of Users')
ax1.set_title('Conversion Funnel')
ax1.set_yscale('log')
for bar, val in zip(bars, funnel_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
             f'{val:,}', ha='center', va='bottom', fontsize=10)

# 5-2: Conversion rates
ax2 = axes[1]
conv_labels = ['View→Cart', 'Cart→Purchase', 'View→Purchase']
conv_values = [journey_stats['conversion_rates']['view_to_cart'],
               journey_stats['conversion_rates']['cart_to_purchase'],
               journey_stats['conversion_rates']['view_to_purchase']]
colors = ['#9b59b6', '#e74c3c', '#1abc9c']
bars = ax2.bar(conv_labels, conv_values, color=colors)
ax2.set_ylabel('Conversion Rate (%)')
ax2.set_title('Conversion Rates')
for bar, val in zip(bars, conv_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_user_journey.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_user_journey.png")

# =============================================================================
# 6. Time-based Analysis
# =============================================================================
print("\n[7] Analyzing time-based patterns...")

df['date'] = df['event_time'].dt.date
df['hour'] = df['event_time'].dt.hour

# Daily interaction counts
daily_counts = df.groupby(['date', 'event_type']).size().unstack(fill_value=0)

# Hourly patterns by event type
hourly_by_event = df.groupby(['hour', 'event_type']).size().unstack(fill_value=0)

time_stats = {
    'date_range': {
        'start': str(df['event_time'].min()),
        'end': str(df['event_time'].max()),
        'total_days': int((df['event_time'].max() - df['event_time'].min()).days + 1),
    },
    'peak_hours': {
        'view': int(hourly_by_event['view'].idxmax()) if 'view' in hourly_by_event.columns else None,
        'cart': int(hourly_by_event['cart'].idxmax()) if 'cart' in hourly_by_event.columns else None,
        'purchase': int(hourly_by_event['purchase'].idxmax()) if 'purchase' in hourly_by_event.columns else None,
    },
    'avg_daily_events': {
        col: float(daily_counts[col].mean()) for col in daily_counts.columns
    }
}

results['time_stats'] = time_stats

# Visualization 6: Time patterns
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 6-1: Hourly pattern by event type
ax1 = axes[0]
for col, color in zip(['view', 'cart', 'purchase'], ['#3498db', '#f39c12', '#27ae60']):
    if col in hourly_by_event.columns:
        # Normalize for comparison
        normalized = hourly_by_event[col] / hourly_by_event[col].sum() * 100
        ax1.plot(hourly_by_event.index, normalized, label=col.capitalize(), color=color, linewidth=2, marker='o')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Percentage of Events')
ax1.set_title('Hourly Distribution by Event Type (Normalized)')
ax1.legend()
ax1.set_xticks(range(24))
ax1.grid(True, alpha=0.3)

# 6-2: Daily trend
ax2 = axes[1]
dates = daily_counts.index
ax2.fill_between(range(len(dates)), daily_counts['view'].values, alpha=0.3, label='View', color='#3498db')
ax2.plot(range(len(dates)), daily_counts['view'].values, color='#3498db', linewidth=1)
ax2.set_xlabel('Day')
ax2.set_ylabel('Number of Views')
ax2.set_title('Daily View Trend')
ax2.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_time_patterns.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_time_patterns.png")

# =============================================================================
# Save JSON Results
# =============================================================================
print("\n[8] Saving results to JSON...")

# Convert any remaining non-serializable types
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

results = convert_to_serializable(results)

with open(f"{OUTPUT_DIR}/analysis_results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"  Saved: analysis_results.json")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Analysis Complete!")
print("=" * 70)
print(f"\nOutput files saved to: {OUTPUT_DIR}/")
print("  - 01_user_interaction_distribution.png")
print("  - 02_purchase_brand_distribution.png")
print("  - 03_overall_brand_distribution.png")
print("  - 04_brand_viewing_patterns.png")
print("  - 05_user_journey.png")
print("  - 06_time_patterns.png")
print("  - analysis_results.json")

print("\n[Key Findings]")
print(f"  - Total Users: {results['user_interaction_stats']['total_users']:,}")
print(f"  - Avg Interactions/User: {results['user_interaction_stats']['interaction_per_user']['mean']:.1f}")
print(f"  - Users with Purchase: {results['user_interaction_stats']['event_type_per_user'].get('purchase', {}).get('users_with_event', 0):,}")
print(f"  - Top Purchase Brand Share: {results['purchase_brand_stats']['brand_concentration']['top_5_share']:.1f}%")
print(f"  - View→Purchase Rate: {results['user_journey_stats']['conversion_rates']['view_to_purchase']:.4f}%")
