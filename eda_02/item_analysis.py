"""
Item-Centric Analysis
- Total items and activity distribution
- Items with no views
- Viewing patterns for items (duration proxy, view counts, cart conversion)
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Output directory
OUTPUT_DIR = "./item_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Item-Centric Analysis")
print("=" * 70)

# Load data
print("\n[1] Loading data...")
df = pd.read_parquet("../train.parquet")
df['event_time'] = pd.to_datetime(df['event_time'])
print(f"Total records: {len(df):,}")
print(f"Total items: {df['item_id'].nunique():,}")

results = {}

# =============================================================================
# 1. Overall Item Activity Summary
# =============================================================================
print("\n[2] Analyzing overall item activity...")

# Count events per item by type
item_event_counts = df.groupby(['item_id', 'event_type']).size().unstack(fill_value=0)

# Ensure all columns exist
for col in ['view', 'cart', 'purchase']:
    if col not in item_event_counts.columns:
        item_event_counts[col] = 0

item_event_counts['total'] = item_event_counts.sum(axis=1)

# Basic statistics
total_items = df['item_id'].nunique()
items_with_view = (item_event_counts['view'] > 0).sum()
items_with_cart = (item_event_counts['cart'] > 0).sum()
items_with_purchase = (item_event_counts['purchase'] > 0).sum()
items_with_no_view = total_items - items_with_view

overall_stats = {
    'total_unique_items': int(total_items),
    'items_with_view': int(items_with_view),
    'items_with_cart': int(items_with_cart),
    'items_with_purchase': int(items_with_purchase),
    'items_with_no_view': int(items_with_no_view),
    'items_with_no_activity': 0,  # All items in data have at least one activity
    'activity_coverage': {
        'view_rate': float(items_with_view / total_items * 100),
        'cart_rate': float(items_with_cart / total_items * 100),
        'purchase_rate': float(items_with_purchase / total_items * 100),
    },
    'total_events': {
        'view': int(item_event_counts['view'].sum()),
        'cart': int(item_event_counts['cart'].sum()),
        'purchase': int(item_event_counts['purchase'].sum()),
        'total': int(item_event_counts['total'].sum()),
    },
    'avg_events_per_item': {
        'view': float(item_event_counts['view'].mean()),
        'cart': float(item_event_counts['cart'].mean()),
        'purchase': float(item_event_counts['purchase'].mean()),
        'total': float(item_event_counts['total'].mean()),
    }
}

results['overall_item_stats'] = overall_stats

print(f"  Total items: {total_items:,}")
print(f"  Items with view: {items_with_view:,} ({items_with_view/total_items*100:.1f}%)")
print(f"  Items with cart: {items_with_cart:,} ({items_with_cart/total_items*100:.1f}%)")
print(f"  Items with purchase: {items_with_purchase:,} ({items_with_purchase/total_items*100:.1f}%)")
print(f"  Items with NO view: {items_with_no_view:,}")

# Visualization 1: Item activity overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1-1: Items by activity type (Venn-like bar)
ax1 = axes[0, 0]
categories = ['With View', 'With Cart', 'With Purchase', 'No View']
values = [items_with_view, items_with_cart, items_with_purchase, items_with_no_view]
colors = ['#3498db', '#f39c12', '#27ae60', '#e74c3c']
bars = ax1.bar(categories, values, color=colors)
ax1.set_ylabel('Number of Items')
ax1.set_title('Items by Activity Type')
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f'{val:,}\n({val/total_items*100:.1f}%)', ha='center', va='bottom', fontsize=9)

# 1-2: Total events by type
ax2 = axes[0, 1]
event_types = ['View', 'Cart', 'Purchase']
event_counts = [overall_stats['total_events']['view'],
                overall_stats['total_events']['cart'],
                overall_stats['total_events']['purchase']]
colors = ['#3498db', '#f39c12', '#27ae60']
bars = ax2.bar(event_types, event_counts, color=colors)
ax2.set_ylabel('Total Events')
ax2.set_title('Total Events by Type')
ax2.set_yscale('log')
for bar, val in zip(bars, event_counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
             f'{val:,}', ha='center', va='bottom', fontsize=9)

# 1-3: Distribution of view counts per item
ax3 = axes[1, 0]
view_counts = item_event_counts['view'].clip(upper=500)
sns.histplot(view_counts[view_counts > 0], bins=50, ax=ax3, color='steelblue')
ax3.set_xlabel('View Count per Item (capped at 500)')
ax3.set_ylabel('Number of Items')
ax3.set_title('Distribution of View Counts per Item')
ax3.axvline(x=item_event_counts['view'].median(), color='red', linestyle='--',
            label=f"Median: {item_event_counts['view'].median():.0f}")
ax3.legend()

# 1-4: Item activity segments
ax4 = axes[1, 1]
segments = {
    'View Only': ((item_event_counts['view'] > 0) & (item_event_counts['cart'] == 0) & (item_event_counts['purchase'] == 0)).sum(),
    'View + Cart': ((item_event_counts['view'] > 0) & (item_event_counts['cart'] > 0) & (item_event_counts['purchase'] == 0)).sum(),
    'View + Cart + Purchase': ((item_event_counts['view'] > 0) & (item_event_counts['cart'] > 0) & (item_event_counts['purchase'] > 0)).sum(),
    'View + Purchase (no Cart)': ((item_event_counts['view'] > 0) & (item_event_counts['cart'] == 0) & (item_event_counts['purchase'] > 0)).sum(),
    'Cart Only': ((item_event_counts['view'] == 0) & (item_event_counts['cart'] > 0)).sum(),
    'Purchase Only': ((item_event_counts['view'] == 0) & (item_event_counts['purchase'] > 0)).sum(),
}
seg_labels = list(segments.keys())
seg_values = list(segments.values())
colors = sns.color_palette("Set2", len(seg_labels))
bars = ax4.barh(seg_labels, seg_values, color=colors)
ax4.set_xlabel('Number of Items')
ax4.set_title('Item Activity Segments')
for bar, val in zip(bars, seg_values):
    if val > 0:
        ax4.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', ha='left', va='center', fontsize=9)

results['item_activity_segments'] = {k: int(v) for k, v in segments.items()}

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_item_activity_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_item_activity_overview.png")

# =============================================================================
# 2. Items with No Views Analysis
# =============================================================================
print("\n[3] Analyzing items with no views...")

# Items that have cart or purchase but no view (unusual behavior)
no_view_items = item_event_counts[item_event_counts['view'] == 0]

no_view_stats = {
    'total_no_view_items': int(len(no_view_items)),
    'no_view_with_cart': int((no_view_items['cart'] > 0).sum()),
    'no_view_with_purchase': int((no_view_items['purchase'] > 0).sum()),
    'no_view_cart_events': int(no_view_items['cart'].sum()),
    'no_view_purchase_events': int(no_view_items['purchase'].sum()),
}

results['no_view_items_stats'] = no_view_stats

# Get details of no-view items
if len(no_view_items) > 0:
    no_view_item_ids = no_view_items.index.tolist()
    no_view_details = df[df['item_id'].isin(no_view_item_ids)].groupby('item_id').agg({
        'user_id': 'nunique',
        'brand': 'first',
        'price': 'first',
        'category_code': 'first'
    }).rename(columns={'user_id': 'unique_users'})

    no_view_stats['sample_no_view_items'] = []
    for item_id in no_view_item_ids[:10]:
        if item_id in no_view_details.index:
            no_view_stats['sample_no_view_items'].append({
                'item_id': str(item_id),
                'cart_count': int(no_view_items.loc[item_id, 'cart']),
                'purchase_count': int(no_view_items.loc[item_id, 'purchase']),
                'brand': str(no_view_details.loc[item_id, 'brand']),
            })

# Visualization 2: No-view items analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 2-1: No-view items breakdown
ax1 = axes[0]
categories = ['Total No-View', 'With Cart', 'With Purchase']
values = [no_view_stats['total_no_view_items'],
          no_view_stats['no_view_with_cart'],
          no_view_stats['no_view_with_purchase']]
colors = ['#e74c3c', '#f39c12', '#27ae60']
bars = ax1.bar(categories, values, color=colors)
ax1.set_ylabel('Number of Items')
ax1.set_title('Items with No View Events')
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val:,}', ha='center', va='bottom', fontsize=10)

# 2-2: Pie chart of view vs no-view
ax2 = axes[1]
sizes = [items_with_view, items_with_no_view]
labels = [f'With View\n({items_with_view:,})', f'No View\n({items_with_no_view:,})']
colors = ['#3498db', '#e74c3c']
ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax2.set_title('Items: View vs No View')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_no_view_items.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_no_view_items.png")

# =============================================================================
# 3. Viewing Behavior Analysis (for items with views)
# =============================================================================
print("\n[4] Analyzing viewing behavior for items with views...")

# Filter to items with at least one view
items_with_views = item_event_counts[item_event_counts['view'] > 0].index.tolist()
view_df = df[(df['item_id'].isin(items_with_views)) & (df['event_type'] == 'view')].copy()

# Per-item viewing statistics
item_view_stats = view_df.groupby('item_id').agg({
    'user_id': ['count', 'nunique'],  # total views, unique viewers
    'event_time': ['min', 'max'],
}).reset_index()
item_view_stats.columns = ['item_id', 'total_views', 'unique_viewers', 'first_view', 'last_view']

# Calculate viewing span (proxy for "how long" - time from first to last view of the item)
item_view_stats['view_span_hours'] = (
    (item_view_stats['last_view'] - item_view_stats['first_view']).dt.total_seconds() / 3600
)

# Views per user (how many times users viewed each item on average)
item_view_stats['views_per_user'] = item_view_stats['total_views'] / item_view_stats['unique_viewers']

# Merge with cart/purchase info
item_view_stats = item_view_stats.merge(
    item_event_counts[['cart', 'purchase']].reset_index(),
    on='item_id',
    how='left'
)

# Calculate conversion rates
item_view_stats['view_to_cart_rate'] = (item_view_stats['cart'] > 0).astype(int)
item_view_stats['view_to_purchase_rate'] = (item_view_stats['purchase'] > 0).astype(int)

# Statistics
viewing_behavior_stats = {
    'items_analyzed': int(len(item_view_stats)),
    'total_views': {
        'mean': float(item_view_stats['total_views'].mean()),
        'median': float(item_view_stats['total_views'].median()),
        'std': float(item_view_stats['total_views'].std()),
        'min': int(item_view_stats['total_views'].min()),
        'max': int(item_view_stats['total_views'].max()),
        'percentiles': {
            '25%': float(item_view_stats['total_views'].quantile(0.25)),
            '50%': float(item_view_stats['total_views'].quantile(0.50)),
            '75%': float(item_view_stats['total_views'].quantile(0.75)),
            '90%': float(item_view_stats['total_views'].quantile(0.90)),
            '99%': float(item_view_stats['total_views'].quantile(0.99)),
        }
    },
    'unique_viewers': {
        'mean': float(item_view_stats['unique_viewers'].mean()),
        'median': float(item_view_stats['unique_viewers'].median()),
        'max': int(item_view_stats['unique_viewers'].max()),
    },
    'views_per_user': {
        'mean': float(item_view_stats['views_per_user'].mean()),
        'median': float(item_view_stats['views_per_user'].median()),
        'max': float(item_view_stats['views_per_user'].max()),
    },
    'view_span_hours': {
        'mean': float(item_view_stats['view_span_hours'].mean()),
        'median': float(item_view_stats['view_span_hours'].median()),
        'max': float(item_view_stats['view_span_hours'].max()),
    },
    'conversion_rates': {
        'items_with_cart_after_view': int(item_view_stats['view_to_cart_rate'].sum()),
        'items_with_purchase_after_view': int(item_view_stats['view_to_purchase_rate'].sum()),
        'cart_conversion_rate': float(item_view_stats['view_to_cart_rate'].mean() * 100),
        'purchase_conversion_rate': float(item_view_stats['view_to_purchase_rate'].mean() * 100),
    }
}

results['viewing_behavior_stats'] = viewing_behavior_stats

# Visualization 3: Viewing behavior
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3-1: Distribution of total views per item
ax1 = axes[0, 0]
views_capped = item_view_stats['total_views'].clip(upper=1000)
sns.histplot(views_capped, bins=50, ax=ax1, color='steelblue')
ax1.set_xlabel('Total Views per Item (capped at 1000)')
ax1.set_ylabel('Number of Items')
ax1.set_title('Distribution of Total Views per Item')
ax1.axvline(x=viewing_behavior_stats['total_views']['median'], color='red', linestyle='--',
            label=f"Median: {viewing_behavior_stats['total_views']['median']:.0f}")
ax1.legend()

# 3-2: Distribution of unique viewers per item
ax2 = axes[0, 1]
viewers_capped = item_view_stats['unique_viewers'].clip(upper=500)
sns.histplot(viewers_capped, bins=50, ax=ax2, color='#27ae60')
ax2.set_xlabel('Unique Viewers per Item (capped at 500)')
ax2.set_ylabel('Number of Items')
ax2.set_title('Distribution of Unique Viewers per Item')
ax2.axvline(x=viewing_behavior_stats['unique_viewers']['median'], color='red', linestyle='--',
            label=f"Median: {viewing_behavior_stats['unique_viewers']['median']:.0f}")
ax2.legend()

# 3-3: Views per user distribution
ax3 = axes[1, 0]
vpu_capped = item_view_stats['views_per_user'].clip(upper=10)
sns.histplot(vpu_capped, bins=50, ax=ax3, color='#9b59b6')
ax3.set_xlabel('Average Views per User per Item (capped at 10)')
ax3.set_ylabel('Number of Items')
ax3.set_title('How Many Times Users View Each Item')
ax3.axvline(x=viewing_behavior_stats['views_per_user']['median'], color='red', linestyle='--',
            label=f"Median: {viewing_behavior_stats['views_per_user']['median']:.2f}")
ax3.legend()

# 3-4: View span distribution (proxy for engagement duration)
ax4 = axes[1, 1]
span_capped = item_view_stats['view_span_hours'].clip(upper=720)  # cap at 30 days
sns.histplot(span_capped[span_capped > 0], bins=50, ax=ax4, color='#e74c3c')
ax4.set_xlabel('View Span in Hours (capped at 720h/30days)')
ax4.set_ylabel('Number of Items')
ax4.set_title('Item View Span (First to Last View)')
ax4.axvline(x=viewing_behavior_stats['view_span_hours']['median'], color='blue', linestyle='--',
            label=f"Median: {viewing_behavior_stats['view_span_hours']['median']:.1f}h")
ax4.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_viewing_behavior.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_viewing_behavior.png")

# =============================================================================
# 4. Item Conversion Analysis (View -> Cart -> Purchase)
# =============================================================================
print("\n[5] Analyzing item conversion patterns...")

# Items segmented by conversion
item_conversion_segments = {
    'view_only': int(((item_view_stats['cart'] == 0) & (item_view_stats['purchase'] == 0)).sum()),
    'view_and_cart_only': int(((item_view_stats['cart'] > 0) & (item_view_stats['purchase'] == 0)).sum()),
    'view_and_purchase': int((item_view_stats['purchase'] > 0).sum()),
}

# Analyze relationship between views and conversion
view_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
view_labels = ['1-10', '11-50', '51-100', '101-500', '501-1000', '1000+']
item_view_stats['view_bin'] = pd.cut(item_view_stats['total_views'], bins=view_bins, labels=view_labels)

conversion_by_views = item_view_stats.groupby('view_bin').agg({
    'item_id': 'count',
    'view_to_cart_rate': 'mean',
    'view_to_purchase_rate': 'mean',
}).rename(columns={'item_id': 'item_count'})

conversion_analysis = {
    'segments': item_conversion_segments,
    'conversion_by_view_count': {
        str(idx): {
            'item_count': int(row['item_count']),
            'cart_conversion_rate': float(row['view_to_cart_rate'] * 100),
            'purchase_conversion_rate': float(row['view_to_purchase_rate'] * 100),
        }
        for idx, row in conversion_by_views.iterrows()
    }
}

results['item_conversion_analysis'] = conversion_analysis

# Visualization 4: Conversion analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4-1: Item conversion segments pie
ax1 = axes[0, 0]
seg_labels = ['View Only', 'View + Cart', 'View + Purchase']
seg_values = [item_conversion_segments['view_only'],
              item_conversion_segments['view_and_cart_only'],
              item_conversion_segments['view_and_purchase']]
colors = ['#3498db', '#f39c12', '#27ae60']
ax1.pie(seg_values, labels=[f'{l}\n({v:,})' for l, v in zip(seg_labels, seg_values)],
        colors=colors, autopct='%1.1f%%', startangle=90)
ax1.set_title('Items by Conversion Stage')

# 4-2: Conversion rate by view count
ax2 = axes[0, 1]
x_labels = list(conversion_analysis['conversion_by_view_count'].keys())
cart_rates = [conversion_analysis['conversion_by_view_count'][k]['cart_conversion_rate'] for k in x_labels]
purchase_rates = [conversion_analysis['conversion_by_view_count'][k]['purchase_conversion_rate'] for k in x_labels]

x = np.arange(len(x_labels))
width = 0.35
bars1 = ax2.bar(x - width/2, cart_rates, width, label='Cart Conversion', color='#f39c12')
bars2 = ax2.bar(x + width/2, purchase_rates, width, label='Purchase Conversion', color='#27ae60')
ax2.set_xlabel('View Count Range')
ax2.set_ylabel('Conversion Rate (%)')
ax2.set_title('Conversion Rate by View Count')
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, rotation=45)
ax2.legend()

# 4-3: Scatter plot - Views vs Cart conversion
ax3 = axes[1, 0]
sample_df = item_view_stats.sample(min(5000, len(item_view_stats)))
ax3.scatter(sample_df['total_views'].clip(upper=1000),
            sample_df['cart'],
            alpha=0.3, s=10, color='#f39c12')
ax3.set_xlabel('Total Views (capped at 1000)')
ax3.set_ylabel('Cart Events')
ax3.set_title('Views vs Cart Events (Sample)')

# 4-4: Top items by conversion (most carted/purchased relative to views)
ax4 = axes[1, 1]
# Items with at least 100 views
popular_items = item_view_stats[item_view_stats['total_views'] >= 100].copy()
popular_items['cart_per_100_views'] = (popular_items['cart'] / popular_items['total_views']) * 100
top_converting = popular_items.nlargest(15, 'cart_per_100_views')

y_pos = np.arange(len(top_converting))
ax4.barh(y_pos, top_converting['cart_per_100_views'].values, color='#f39c12')
ax4.set_yticks(y_pos)
ax4.set_yticklabels([f"Item {i+1}" for i in range(len(top_converting))])
ax4.set_xlabel('Cart Events per 100 Views')
ax4.set_title('Top 15 Items by Cart Conversion Rate\n(min 100 views)')
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_item_conversion.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_item_conversion.png")

# =============================================================================
# 5. Session-based Viewing Analysis
# =============================================================================
print("\n[6] Analyzing session-based viewing patterns...")

# Group by user_session to understand viewing depth
session_item_views = df[df['event_type'] == 'view'].groupby(['user_session', 'item_id']).agg({
    'event_time': ['count', 'min', 'max']
}).reset_index()
session_item_views.columns = ['user_session', 'item_id', 'view_count', 'first_view', 'last_view']
session_item_views['session_duration_sec'] = (
    (session_item_views['last_view'] - session_item_views['first_view']).dt.total_seconds()
)

# Per-item session statistics
item_session_stats = session_item_views.groupby('item_id').agg({
    'user_session': 'nunique',  # unique sessions
    'view_count': ['mean', 'max'],  # views per session
    'session_duration_sec': ['mean', 'median', 'max']
}).reset_index()
item_session_stats.columns = ['item_id', 'unique_sessions', 'avg_views_per_session',
                               'max_views_per_session', 'avg_session_duration',
                               'median_session_duration', 'max_session_duration']

session_viewing_stats = {
    'total_sessions_with_views': int(session_item_views['user_session'].nunique()),
    'avg_views_per_session_per_item': {
        'mean': float(item_session_stats['avg_views_per_session'].mean()),
        'median': float(item_session_stats['avg_views_per_session'].median()),
        'max': float(item_session_stats['max_views_per_session'].max()),
    },
    'session_duration_per_item_seconds': {
        'mean': float(item_session_stats['avg_session_duration'].mean()),
        'median': float(item_session_stats['median_session_duration'].median()),
    },
    'unique_sessions_per_item': {
        'mean': float(item_session_stats['unique_sessions'].mean()),
        'median': float(item_session_stats['unique_sessions'].median()),
        'max': int(item_session_stats['unique_sessions'].max()),
    }
}

results['session_viewing_stats'] = session_viewing_stats

# Visualization 5: Session-based viewing
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5-1: Views per session distribution
ax1 = axes[0, 0]
vps_capped = item_session_stats['avg_views_per_session'].clip(upper=5)
sns.histplot(vps_capped, bins=50, ax=ax1, color='#3498db')
ax1.set_xlabel('Average Views per Session per Item')
ax1.set_ylabel('Number of Items')
ax1.set_title('How Many Times Users View Item in One Session')
ax1.axvline(x=session_viewing_stats['avg_views_per_session_per_item']['median'],
            color='red', linestyle='--',
            label=f"Median: {session_viewing_stats['avg_views_per_session_per_item']['median']:.2f}")
ax1.legend()

# 5-2: Session duration distribution
ax2 = axes[0, 1]
duration_capped = item_session_stats['avg_session_duration'].clip(upper=300)  # cap at 5 min
sns.histplot(duration_capped[duration_capped > 0], bins=50, ax=ax2, color='#9b59b6')
ax2.set_xlabel('Average Session Duration (seconds)')
ax2.set_ylabel('Number of Items')
ax2.set_title('Average Time Spent Viewing Each Item per Session')

# 5-3: Unique sessions per item
ax3 = axes[1, 0]
sessions_capped = item_session_stats['unique_sessions'].clip(upper=500)
sns.histplot(sessions_capped, bins=50, ax=ax3, color='#27ae60')
ax3.set_xlabel('Unique Sessions per Item (capped at 500)')
ax3.set_ylabel('Number of Items')
ax3.set_title('Number of Sessions That Viewed Each Item')

# 5-4: Relationship between sessions and conversions
ax4 = axes[1, 1]
merged_stats = item_session_stats.merge(item_view_stats[['item_id', 'cart', 'purchase']], on='item_id', how='left')
sample_merged = merged_stats.sample(min(5000, len(merged_stats)))
ax4.scatter(sample_merged['unique_sessions'].clip(upper=500),
            sample_merged['cart'],
            alpha=0.3, s=10, color='#f39c12')
ax4.set_xlabel('Unique Sessions (capped at 500)')
ax4.set_ylabel('Cart Events')
ax4.set_title('Sessions vs Cart Events')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_session_viewing.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_session_viewing.png")

# =============================================================================
# 6. Item Popularity Tiers
# =============================================================================
print("\n[7] Analyzing item popularity tiers...")

# Define popularity tiers based on total interactions
item_event_counts_sorted = item_event_counts.sort_values('total', ascending=False).reset_index()
item_event_counts_sorted['rank'] = range(1, len(item_event_counts_sorted) + 1)
item_event_counts_sorted['percentile'] = item_event_counts_sorted['rank'] / len(item_event_counts_sorted) * 100

# Tier definitions
tiers = {
    'top_1%': item_event_counts_sorted[item_event_counts_sorted['percentile'] <= 1],
    'top_5%': item_event_counts_sorted[(item_event_counts_sorted['percentile'] > 1) & (item_event_counts_sorted['percentile'] <= 5)],
    'top_20%': item_event_counts_sorted[(item_event_counts_sorted['percentile'] > 5) & (item_event_counts_sorted['percentile'] <= 20)],
    'middle_60%': item_event_counts_sorted[(item_event_counts_sorted['percentile'] > 20) & (item_event_counts_sorted['percentile'] <= 80)],
    'bottom_20%': item_event_counts_sorted[item_event_counts_sorted['percentile'] > 80],
}

popularity_tiers = {}
for tier_name, tier_df in tiers.items():
    popularity_tiers[tier_name] = {
        'item_count': int(len(tier_df)),
        'total_interactions': int(tier_df['total'].sum()),
        'total_views': int(tier_df['view'].sum()),
        'total_carts': int(tier_df['cart'].sum()),
        'total_purchases': int(tier_df['purchase'].sum()),
        'avg_interactions': float(tier_df['total'].mean()),
        'interaction_share': float(tier_df['total'].sum() / item_event_counts['total'].sum() * 100),
    }

results['popularity_tiers'] = popularity_tiers

# Visualization 6: Popularity tiers
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 6-1: Interaction share by tier
ax1 = axes[0]
tier_labels = list(popularity_tiers.keys())
interaction_shares = [popularity_tiers[t]['interaction_share'] for t in tier_labels]
colors = sns.color_palette("RdYlGn_r", len(tier_labels))
bars = ax1.bar(tier_labels, interaction_shares, color=colors)
ax1.set_ylabel('Share of Total Interactions (%)')
ax1.set_title('Interaction Share by Popularity Tier')
ax1.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, interaction_shares):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# 6-2: Item count vs Interaction share
ax2 = axes[1]
item_counts = [popularity_tiers[t]['item_count'] for t in tier_labels]
x = np.arange(len(tier_labels))
width = 0.35

ax2_twin = ax2.twinx()
bars1 = ax2.bar(x - width/2, item_counts, width, label='Item Count', color='#3498db')
bars2 = ax2_twin.bar(x + width/2, interaction_shares, width, label='Interaction Share %', color='#e74c3c')

ax2.set_ylabel('Item Count', color='#3498db')
ax2_twin.set_ylabel('Interaction Share (%)', color='#e74c3c')
ax2.set_xticks(x)
ax2.set_xticklabels(tier_labels, rotation=45)
ax2.set_title('Items vs Interactions by Tier')

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_popularity_tiers.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_popularity_tiers.png")

# =============================================================================
# Save JSON Results
# =============================================================================
print("\n[8] Saving results to JSON...")

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

with open(f"{OUTPUT_DIR}/item_analysis_results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"  Saved: item_analysis_results.json")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Item Analysis Complete!")
print("=" * 70)
print(f"\nOutput files saved to: {OUTPUT_DIR}/")
print("  - 01_item_activity_overview.png")
print("  - 02_no_view_items.png")
print("  - 03_viewing_behavior.png")
print("  - 04_item_conversion.png")
print("  - 05_session_viewing.png")
print("  - 06_popularity_tiers.png")
print("  - item_analysis_results.json")

print("\n[Key Findings]")
print(f"  - Total Items: {overall_stats['total_unique_items']:,}")
print(f"  - Items with View: {overall_stats['items_with_view']:,} ({overall_stats['activity_coverage']['view_rate']:.1f}%)")
print(f"  - Items with NO View: {overall_stats['items_with_no_view']:,}")
print(f"  - Items with Cart: {overall_stats['items_with_cart']:,} ({overall_stats['activity_coverage']['cart_rate']:.1f}%)")
print(f"  - Items with Purchase: {overall_stats['items_with_purchase']:,} ({overall_stats['activity_coverage']['purchase_rate']:.1f}%)")
print(f"  - Avg Views per Item: {viewing_behavior_stats['total_views']['mean']:.1f}")
print(f"  - Avg Unique Viewers per Item: {viewing_behavior_stats['unique_viewers']['mean']:.1f}")
print(f"  - Cart Conversion Rate: {viewing_behavior_stats['conversion_rates']['cart_conversion_rate']:.2f}%")
print(f"  - Top 1% items have {popularity_tiers['top_1%']['interaction_share']:.1f}% of interactions")
