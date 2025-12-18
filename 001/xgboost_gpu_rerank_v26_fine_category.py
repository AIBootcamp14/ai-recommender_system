# xgboost_gpu_rerank_v26_fine_category.py
# XGBoost GPU Learning to Rank with:
# - GPU acceleration optimized for RTX 3090 24GB VRAM
# - Implicit weighted labels (purchase=5, cart=3, view=1)
# - Cold user dedicated recommendation (recent trending + low price + high conversion)
# - Slot-based final assembly (cold/active user separation)
# - Source features + v5_score blending
# - Item Behavior Profile Clustering (k=50, price included)
# - ✅ NEW v26: fine_category = real_category + cluster_id
#   (핸드폰 클러스터 vs 냉장고 클러스터 구분으로 더 정밀한 카테고리 매칭)
# - ✅ NEW v26: 저조회 유저(1-2 view)도 XGBoost 통과 (특별 처리 제거)

import os
import gc
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =========================
# Config
# =========================
DATA_DIR = "../data"
TRAIN_PATH = os.path.join(DATA_DIR, "new_train.parquet")  # real_category 포함된 파일
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

OUT_DIR = "./output_Rec_011"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = os.path.join(OUT_DIR, f"output_xgb_gpu_rerank_v26_{TIMESTAMP}.csv")
LOG_PATH = os.path.join(OUT_DIR, f"xgb_gpu_rerank_v26_log_{TIMESTAMP}.txt")
ANALYSIS_PATH = os.path.join(OUT_DIR, f"xgb_gpu_rerank_v26_analysis_{TIMESTAMP}.json")


def setup_logging():
    """콘솔 + 파일 동시 로깅 설정"""
    os.makedirs(OUT_DIR, exist_ok=True)

    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 포맷
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 파일 핸들러
    fh = logging.FileHandler(LOG_PATH, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

# 후보 생성 크기 (유저당)
CAND_CART = 50
CAND_REPEAT = 200
CAND_RECENT = 200
CAND_POP = 200
CAND_MAX = 600  # 중복 제거 후 최대 후보 수(대략)

# 학습을 위한 시간 split (마지막 N일을 "미래(라벨)"로 사용)
VAL_FUTURE_DAYS = 10  # 라벨 구간 길이
# 학습 규모 제한
MAX_TRAIN_USERS = 60000  # 3~10만 사이에서 조절 추천

# ✅ 클러스터링 설정
N_CLUSTERS = 50  # 아이템 행동 프로파일 클러스터 수

# ============================================
# ✅ XGBoost GPU Optimized Parameters for RTX 3090 24GB
# ============================================
XGB_GPU_PARAMS = {
    # GPU settings
    "device": "cuda",                    # GPU 사용
    "tree_method": "hist",               # GPU histogram-based tree construction

    # Learning parameters
    "objective": "rank:ndcg",            # Learning to Rank with NDCG
    "eval_metric": "ndcg@10",            # NDCG@10 evaluation
    "learning_rate": 0.05,               # Learning rate
    "n_estimators": 5000,                # Maximum number of trees (early stopping will determine actual)

    # Tree structure - optimized for 24GB VRAM
    "max_depth": 8,                      # 트리 깊이 (GPU에서는 더 깊은 트리가 효율적)
    "max_leaves": 127,                   # 리프 노드 수 (2^7-1, num_leaves=63보다 약간 큼)
    "min_child_weight": 50,              # min_data_in_leaf와 유사

    # Regularization
    "reg_alpha": 0.1,                    # L1 regularization
    "reg_lambda": 1.0,                   # L2 regularization
    "gamma": 0.1,                        # Minimum loss reduction for split

    # Subsampling - for regularization and speed
    "subsample": 0.8,                    # Row subsampling
    "colsample_bytree": 0.8,             # Feature subsampling per tree
    "colsample_bylevel": 0.8,            # Feature subsampling per level

    # GPU Memory optimization for RTX 3090 (24GB)
    "max_bin": 256,                      # GPU histogram bins (기본 256, 메모리 효율적)

    # Random seed
    "seed": 42,

    # Verbose
    "verbosity": 1,
}

# Early stopping rounds
EARLY_STOPPING_ROUNDS = 100


# =========================
# Utils
# =========================
def price_bucket(price):
    if pd.isna(price):
        return -1
    if price <= 50:
        return 0
    if price <= 100:
        return 1
    if price <= 200:
        return 2
    if price <= 500:
        return 3
    return 4

def get_price_bonus(price):
    # v5 heuristic (학습 피처로 넣기 위한 값)
    if pd.isna(price):
        return 1.0
    if price <= 50:
        return 1.5
    elif price <= 500:
        return 1.2 if price > 200 else 1.0
    else:
        return 0.8

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def split_users_hash(users, valid_ratio=0.2, seed=42):
    # 유저 단위로 train/val split (시간누수 없이, 동일 유저 후보 묶음 유지)
    # 안정적 재현성을 위해 해시 기반 분할
    users = np.array(users)
    h = pd.util.hash_pandas_object(pd.Series(users), index=False).values
    rng = np.random.RandomState(seed)
    # hash만으로도 충분하지만, seed 반영 위해 xor
    h = h ^ rng.randint(0, 2**32 - 1, size=h.shape, dtype=np.uint64)
    mask_val = (h % 100) < int(valid_ratio * 100)
    train_users = users[~mask_val].tolist()
    val_users = users[mask_val].tolist()
    return train_users, val_users


# =========================
# ✅ Item Behavior Profile & Clustering with fine_category
# =========================
def build_item_behavior_profile(df):
    """
    아이템별 유저 행동 패턴을 프로파일로 추출
    """
    logging.info("=" * 50)
    logging.info("[ItemProfile] Building item behavior profiles...")
    logging.info(f"  Input data: {len(df):,} rows, {df['item_id'].nunique():,} unique items")
    logging.info(f"  Event distribution: {df['event_type'].value_counts().to_dict()}")

    # 1. 기본 이벤트 카운트
    event_counts = df.groupby(["item_id", "event_type"]).size().unstack(fill_value=0)
    event_counts.columns = [f"cnt_{c}" for c in event_counts.columns]
    event_counts = event_counts.reset_index()

    # 컬럼이 없을 수 있으니 안전하게 처리
    for col in ["cnt_view", "cnt_cart", "cnt_purchase"]:
        if col not in event_counts.columns:
            event_counts[col] = 0

    # 2. 유저당 평균 조회수 (아이템별)
    view_df = df[df["event_type"] == "view"]
    user_view_per_item = view_df.groupby(["item_id", "user_id"]).size().reset_index(name="views")
    avg_views_per_user = user_view_per_item.groupby("item_id")["views"].mean().reset_index()
    avg_views_per_user.columns = ["item_id", "avg_views_per_user"]

    # 3. 유니크 유저 수
    unique_viewers = view_df.groupby("item_id")["user_id"].nunique().reset_index()
    unique_viewers.columns = ["item_id", "unique_viewers"]

    # 4. 조회 시간 범위 (첫 조회 ~ 마지막 조회)
    view_time_range = view_df.groupby("item_id")["event_time"].agg(["min", "max"]).reset_index()
    view_time_range["view_span_hours"] = (
        view_time_range["max"] - view_time_range["min"]
    ).dt.total_seconds() / 3600.0
    view_time_range = view_time_range[["item_id", "view_span_hours"]]

    # 5. 세션별 조회 (세션 집중도)
    if "user_session" in df.columns:
        session_views = view_df.groupby(["item_id", "user_session"]).size().reset_index(name="session_views")
        avg_session_views = session_views.groupby("item_id")["session_views"].mean().reset_index()
        avg_session_views.columns = ["item_id", "avg_session_views"]
        unique_sessions = session_views.groupby("item_id")["user_session"].nunique().reset_index()
        unique_sessions.columns = ["item_id", "unique_sessions"]
    else:
        avg_session_views = pd.DataFrame({"item_id": event_counts["item_id"], "avg_session_views": 1.0})
        unique_sessions = pd.DataFrame({"item_id": event_counts["item_id"], "unique_sessions": 1})

    # 6. 가격 정보
    item_price = df.groupby("item_id")["price"].mean().reset_index()
    item_price.columns = ["item_id", "avg_price"]

    # 7. 브랜드 정보 (최빈값)
    if "brand" in df.columns:
        item_brand = df.groupby("item_id")["brand"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown").reset_index()
        item_brand.columns = ["item_id", "brand"]
    else:
        item_brand = pd.DataFrame({"item_id": event_counts["item_id"], "brand": "unknown"})

    # 7-1. real_category 정보 (LLM 생성 카테고리)
    if "real_category" in df.columns:
        item_category = df.groupby("item_id")["real_category"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "other").reset_index()
        item_category.columns = ["item_id", "real_category"]
    else:
        item_category = pd.DataFrame({"item_id": event_counts["item_id"], "real_category": "other"})

    # 8. 피크 시간/요일
    view_df = view_df.copy()
    view_df["hour"] = view_df["event_time"].dt.hour
    view_df["dow"] = view_df["event_time"].dt.dayofweek

    peak_hour = view_df.groupby("item_id")["hour"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12).reset_index()
    peak_hour.columns = ["item_id", "peak_hour"]

    peak_dow = view_df.groupby("item_id")["dow"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 3).reset_index()
    peak_dow.columns = ["item_id", "peak_dow"]

    # Merge all
    profile = event_counts.copy()
    profile = profile.merge(avg_views_per_user, on="item_id", how="left")
    profile = profile.merge(unique_viewers, on="item_id", how="left")
    profile = profile.merge(view_time_range, on="item_id", how="left")
    profile = profile.merge(avg_session_views, on="item_id", how="left")
    profile = profile.merge(unique_sessions, on="item_id", how="left")
    profile = profile.merge(item_price, on="item_id", how="left")
    profile = profile.merge(item_brand, on="item_id", how="left")
    profile = profile.merge(item_category, on="item_id", how="left")
    profile = profile.merge(peak_hour, on="item_id", how="left")
    profile = profile.merge(peak_dow, on="item_id", how="left")

    # 전환율 계산 (smoothing 적용)
    profile["view_to_cart_rate"] = (profile["cnt_cart"] + 1) / (profile["cnt_view"] + 100)
    profile["cart_to_purchase_rate"] = (profile["cnt_purchase"] + 1) / (profile["cnt_cart"] + 10)
    profile["view_to_purchase_rate"] = (profile["cnt_purchase"] + 1) / (profile["cnt_view"] + 100)

    # 결측치 처리
    profile["avg_views_per_user"] = profile["avg_views_per_user"].fillna(1.0)
    profile["unique_viewers"] = profile["unique_viewers"].fillna(1)
    profile["view_span_hours"] = profile["view_span_hours"].fillna(0)
    profile["avg_session_views"] = profile["avg_session_views"].fillna(1.0)
    profile["unique_sessions"] = profile["unique_sessions"].fillna(1)
    profile["avg_price"] = profile["avg_price"].fillna(profile["avg_price"].median())
    profile["peak_hour"] = profile["peak_hour"].fillna(12)
    profile["peak_dow"] = profile["peak_dow"].fillna(3)

    # 상세 프로파일 통계 로깅
    logging.info(f"  Item profiles created: {len(profile):,} items")
    logging.info(f"  [Profile Stats]")
    logging.info(f"    avg_views_per_user: min={profile['avg_views_per_user'].min():.2f}, max={profile['avg_views_per_user'].max():.2f}, mean={profile['avg_views_per_user'].mean():.2f}")
    logging.info(f"    avg_price: min={profile['avg_price'].min():.2f}, max={profile['avg_price'].max():.2f}, mean={profile['avg_price'].mean():.2f}")

    # 브랜드 분포
    n_brands = profile["brand"].nunique()
    logging.info(f"    unique_brands: {n_brands}")

    # real_category 분포
    profile["real_category"] = profile["real_category"].fillna("other")
    n_categories = profile["real_category"].nunique()
    logging.info(f"    unique_categories: {n_categories}")

    return profile


def cluster_items_by_behavior(item_profile, n_clusters=50):
    """
    아이템 행동 프로파일 기반 K-Means 클러스터링
    가격 정보 포함
    """
    logging.info("=" * 50)
    logging.info(f"[Cluster] Clustering items into {n_clusters} groups...")

    # 클러스터링에 사용할 피처
    cluster_features = [
        "avg_views_per_user",
        "view_span_hours",
        "avg_session_views",
        "view_to_cart_rate",
        "view_to_purchase_rate",
        "avg_price",
        "peak_hour",
    ]
    logging.info(f"  Clustering features: {cluster_features}")

    # 피처 추출
    X = item_profile[cluster_features].copy()

    # 로그 변환 (skewed 분포 정규화)
    X["avg_views_per_user"] = np.log1p(X["avg_views_per_user"])
    X["view_span_hours"] = np.log1p(X["view_span_hours"])
    X["avg_session_views"] = np.log1p(X["avg_session_views"])
    X["avg_price"] = np.log1p(X["avg_price"])

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means 클러스터링
    logging.info(f"  Running K-Means (k={n_clusters}, n_init=10)...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    logging.info(f"  K-Means inertia: {kmeans.inertia_:.2f}")

    item_profile = item_profile.copy()
    item_profile["cluster_id"] = clusters

    # ✅ NEW v26: fine_category 생성 (real_category + cluster_id)
    item_profile["fine_category"] = (
        item_profile["real_category"].astype(str) + "_c" + item_profile["cluster_id"].astype(str)
    )
    n_fine_categories = item_profile["fine_category"].nunique()
    logging.info(f"  ✅ NEW: fine_category created: {n_fine_categories} unique fine categories")

    # 클러스터별 통계 로깅
    cluster_stats = item_profile.groupby("cluster_id").agg({
        "item_id": "count",
        "avg_price": "mean",
        "avg_views_per_user": "mean",
        "view_to_purchase_rate": "mean",
        "cnt_purchase": "sum"
    }).reset_index()
    cluster_stats.columns = ["cluster_id", "n_items", "avg_price", "avg_views_per_user", "avg_conversion", "total_purchases"]

    logging.info(f"  [Cluster Distribution]")
    logging.info(f"    Items per cluster: min={cluster_stats['n_items'].min()}, max={cluster_stats['n_items'].max()}, avg={cluster_stats['n_items'].mean():.1f}")

    # 가격대별 클러스터 분포
    price_bins = [0, 50, 100, 200, 500, float('inf')]
    price_labels = ['<50', '50-100', '100-200', '200-500', '>500']
    item_profile['price_tier'] = pd.cut(item_profile['avg_price'], bins=price_bins, labels=price_labels)

    return item_profile, cluster_stats, kmeans, scaler


def build_cluster_top_items(item_profile, item_stats, top_k=20):
    """
    각 클러스터별 top 아이템 (구매전환율 + 인기도 기준)
    """
    logging.info("=" * 50)
    logging.info("[ClusterTop] Building top items per cluster...")

    # item_stats와 merge (구매 정보)
    merged = item_profile.merge(
        item_stats[["item_id", "item_purchase_rate", "item_view_pop"]],
        on="item_id",
        how="left"
    )
    merged["item_purchase_rate"] = merged["item_purchase_rate"].fillna(0)
    merged["item_view_pop"] = merged["item_view_pop"].fillna(0)

    # 점수: 구매전환율 * (1 + log(인기도))
    merged["cluster_score"] = (
        merged["item_purchase_rate"] * (1 + np.log1p(merged["item_view_pop"]))
    )

    # 클러스터별 top_k 아이템
    cluster_top_items = {}
    for cluster_id in merged["cluster_id"].unique():
        cluster_items = merged[merged["cluster_id"] == cluster_id]
        scored_items = cluster_items[cluster_items["cluster_score"] > 0]

        if len(scored_items) >= top_k:
            top_items = scored_items.nlargest(top_k, "cluster_score")["item_id"].tolist()
        elif len(scored_items) > 0:
            top_items = scored_items.nlargest(len(scored_items), "cluster_score")["item_id"].tolist()
            remaining = top_k - len(top_items)
            filler_items = cluster_items[~cluster_items["item_id"].isin(top_items)].nlargest(remaining, "item_view_pop")["item_id"].tolist()
            top_items.extend(filler_items)
        else:
            top_items = cluster_items.nlargest(top_k, "item_view_pop")["item_id"].tolist()

        cluster_top_items[cluster_id] = top_items

    logging.info(f"  Built top-{top_k} items for {len(cluster_top_items)} clusters")
    return cluster_top_items


def get_user_cluster_affinity(user_id, user_view_history, item_profile):
    """
    유저가 본 아이템들의 클러스터 분포 계산
    """
    if user_view_history is None or len(user_view_history) == 0:
        return {}

    viewed_items = set(user_view_history)
    viewed_clusters = item_profile[item_profile["item_id"].isin(viewed_items)]["cluster_id"].value_counts()

    if len(viewed_clusters) == 0:
        return {}

    total = viewed_clusters.sum()
    affinity = {int(k): float(v / total) for k, v in viewed_clusters.items()}
    return affinity


def build_brand_cluster_map(item_profile):
    """
    브랜드별 주요 클러스터 매핑
    """
    logging.info("=" * 50)
    logging.info("[BrandCluster] Building brand-cluster mapping...")

    brand_cluster = item_profile.groupby(["brand", "cluster_id"]).size().reset_index(name="count")

    brand_main_cluster = brand_cluster.loc[
        brand_cluster.groupby("brand")["count"].idxmax()
    ][["brand", "cluster_id"]]
    brand_main_cluster.columns = ["brand", "main_cluster_id"]

    brand_cluster_map = dict(zip(brand_main_cluster["brand"], brand_main_cluster["main_cluster_id"]))

    logging.info(f"  Mapped {len(brand_cluster_map)} brands to clusters")
    return brand_cluster_map


# =========================
# 1) Load
# =========================
def load_data():
    logging.info("[Load] reading parquet/csv...")
    df = pd.read_parquet(TRAIN_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    target_users = sample["user_id"].unique().tolist()

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"])

    logging.info(f"  rows={len(df):,}, target_users={len(target_users):,}")
    return df, sample, target_users


# =========================
# 2) Global stats
# =========================
def build_global_stats(df):
    logging.info("[Global] building item stats...")
    view = df[df["event_type"] == "view"][["item_id"]]
    purchase = df[df["event_type"] == "purchase"][["item_id"]]

    item_view_pop = view["item_id"].value_counts()
    item_purchase_pop = purchase["item_id"].value_counts()

    item_view_pop = item_view_pop.rename("item_view_pop").reset_index().rename(columns={"index": "item_id"})
    item_purchase_pop = item_purchase_pop.rename("item_purchase_pop").reset_index().rename(columns={"index": "item_id"})

    item_price = df.groupby("item_id", as_index=False)["price"].mean().rename(columns={"price": "item_price"})

    stats = item_view_pop.merge(item_purchase_pop, on="item_id", how="left")
    stats = stats.merge(item_price, on="item_id", how="left")
    stats["item_purchase_pop"] = stats["item_purchase_pop"].fillna(0).astype(np.int64)

    stats["item_purchase_rate"] = (stats["item_purchase_pop"] + 1.0) / (stats["item_view_pop"] + 100.0)

    stats["price_bucket"] = stats["item_price"].apply(price_bucket).astype(np.int16)
    stats["price_bonus"] = stats["item_price"].apply(get_price_bonus).astype(np.float32)

    purchased_items = set(df[df["event_type"] == "purchase"]["item_id"].unique().tolist())
    purchased_view = stats[stats["item_id"].isin(purchased_items)].sort_values("item_view_pop", ascending=False)["item_id"].tolist()
    non_purchased_view = stats[~stats["item_id"].isin(purchased_items)].sort_values("item_view_pop", ascending=False)["item_id"].head(500).tolist()
    popular_items = purchased_view + non_purchased_view

    logging.info(f"  items with views={len(stats):,}, popular_pool={len(popular_items):,}")
    return stats, popular_items


# =========================
# 3) Build user-item aggregates (vectorized)
# =========================
def build_user_item_agg(df, recent_hours=40):
    logging.info("[Agg] building user-item aggregates...")
    view = df[df["event_type"] == "view"][["user_id", "item_id", "event_time"]].copy()
    view["dow"] = view["event_time"].dt.dayofweek.astype(np.int8)
    view["hour"] = view["event_time"].dt.hour.astype(np.int8)

    max_time = view["event_time"].max()
    cutoff = max_time - pd.Timedelta(hours=recent_hours)

    ui_cnt = view.groupby(["user_id", "item_id"], as_index=False).size().rename(columns={"size": "ui_view_cnt"})

    view_sorted = view.sort_values("event_time")
    ui_last = view_sorted.drop_duplicates(["user_id", "item_id"], keep="last")[["user_id", "item_id", "event_time", "dow", "hour"]]
    ui_last = ui_last.rename(columns={"event_time": "ui_last_ts", "dow": "ui_last_dow", "hour": "ui_last_hour"})

    view_40 = view[view["event_time"] >= cutoff]
    ui_cnt_40 = view_40.groupby(["user_id", "item_id"], as_index=False).size().rename(columns={"size": "ui_view_cnt_40h"})

    ui = ui_cnt.merge(ui_last, on=["user_id", "item_id"], how="left")
    ui = ui.merge(ui_cnt_40, on=["user_id", "item_id"], how="left")
    ui["ui_view_cnt_40h"] = ui["ui_view_cnt_40h"].fillna(0).astype(np.int16)

    ui["ui_last_hours_ago"] = (max_time - ui["ui_last_ts"]).dt.total_seconds() / 3600.0
    ui["ui_last_hours_ago"] = ui["ui_last_hours_ago"].fillna(9999.0).astype(np.float32)

    ui["is_peak"] = ((ui["ui_last_hour"] >= 14) & (ui["ui_last_hour"] <= 17)).astype(np.int8)
    ui["is_active"] = ((ui["ui_last_hour"] >= 10) & (ui["ui_last_hour"] <= 18)).astype(np.int8)
    ui["is_thu"] = (ui["ui_last_dow"] == 3).astype(np.int8)
    ui["is_fri"] = (ui["ui_last_dow"] == 4).astype(np.int8)
    ui["is_sat"] = (ui["ui_last_dow"] == 5).astype(np.int8)

    cart = df[df["event_type"] == "cart"][["user_id", "item_id", "event_time"]].copy()
    if len(cart) > 0:
        cart_sorted = cart.sort_values("event_time")
        ui_cart = cart_sorted.drop_duplicates(["user_id", "item_id"], keep="last")[["user_id", "item_id"]]
        ui_cart["ui_cart_flag"] = 1
        ui = ui.merge(ui_cart, on=["user_id", "item_id"], how="left")
        ui["ui_cart_flag"] = ui["ui_cart_flag"].fillna(0).astype(np.int8)
    else:
        ui["ui_cart_flag"] = 0

    logging.info(f"  ui rows={len(ui):,}, max_time={max_time}")
    return ui, max_time


# =========================
# 4) Candidate generation for target users
# =========================
def build_candidates(df, target_users, ui, popular_items):
    logging.info("[Cand] generating candidates for target users...")
    target_set = set(target_users)

    view = df[df["event_type"] == "view"][["user_id", "item_id", "event_time"]].copy()
    view = view[view["user_id"].isin(target_set)]
    view = view.sort_values("event_time", ascending=False)
    view = view.drop_duplicates(["user_id", "item_id"], keep="first")
    view["rank"] = view.groupby("user_id").cumcount()
    recent_unique = view[view["rank"] < (CAND_REPEAT + CAND_RECENT + 50)][["user_id", "item_id"]]

    cart = df[df["event_type"] == "cart"][["user_id", "item_id", "event_time"]].copy()
    cart = cart[cart["user_id"].isin(target_set)]
    cart = cart.sort_values("event_time", ascending=False)
    cart = cart.drop_duplicates(["user_id", "item_id"], keep="first")
    cart["rank"] = cart.groupby("user_id").cumcount()
    cart_top = cart[cart["rank"] < CAND_CART][["user_id", "item_id"]]

    ui_small = ui[ui["user_id"].isin(target_set)][["user_id", "item_id", "ui_view_cnt"]]
    repeat = recent_unique.merge(ui_small, on=["user_id", "item_id"], how="left")
    repeat = repeat[repeat["ui_view_cnt"].fillna(0) >= 2][["user_id", "item_id"]]
    repeat["rank"] = repeat.groupby("user_id").cumcount()
    repeat_top = repeat[repeat["rank"] < CAND_REPEAT][["user_id", "item_id"]]

    recent_unique["rank"] = recent_unique.groupby("user_id").cumcount()
    recent_top = recent_unique[recent_unique["rank"] < CAND_RECENT][["user_id", "item_id"]]

    cart_map = cart_top.groupby("user_id")["item_id"].apply(list).to_dict()
    repeat_map = repeat_top.groupby("user_id")["item_id"].apply(list).to_dict()
    recent_map = recent_top.groupby("user_id")["item_id"].apply(list).to_dict()

    cand_dict = {}
    src_dict = {}

    for u in target_users:
        seen = set()
        cand = []
        srcs = []

        for i in cart_map.get(u, []):
            if i in seen:
                continue
            cand.append(i); srcs.append(3); seen.add(i)
            if len(cand) >= CAND_MAX:
                break

        if len(cand) < CAND_MAX:
            for i in repeat_map.get(u, []):
                if i in seen:
                    continue
                cand.append(i); srcs.append(2); seen.add(i)
                if len(cand) >= CAND_MAX:
                    break

        if len(cand) < CAND_MAX:
            for i in recent_map.get(u, []):
                if i in seen:
                    continue
                cand.append(i); srcs.append(1); seen.add(i)
                if len(cand) >= CAND_MAX:
                    break

        if len(cand) < CAND_MAX:
            for i in popular_items:
                if i in seen:
                    continue
                cand.append(i); srcs.append(0); seen.add(i)
                if len(cand) >= CAND_MAX:
                    break

        cand_dict[u] = cand
        src_dict[u] = srcs

    logging.info("  done.")
    return cand_dict, src_dict


# =========================
# 5) Build feature matrix for (u,i) candidates
# =========================
def build_feature_df(cand_dict, src_dict, ui, item_stats, max_time, exclude_purchased_pairs=None,
                     item_profile=None, user_view_map=None):
    rows = []
    src_codes = []
    groups = []
    users = list(cand_dict.keys())

    for u in users:
        items = cand_dict[u]
        srcs = src_dict[u]
        groups.append(len(items))
        rows.extend([(u, i) for i in items])
        src_codes.extend(srcs)

    cand_df = pd.DataFrame(rows, columns=["user_id", "item_id"])
    cand_df["src_code"] = np.array(src_codes, dtype=np.int8)

    if exclude_purchased_pairs is not None and len(exclude_purchased_pairs) > 0:
        cand_df = cand_df.merge(exclude_purchased_pairs.assign(_p=1), on=["user_id", "item_id"], how="left")
        cand_df = cand_df[cand_df["_p"].isna()].drop(columns=["_p"])
        grp = cand_df.groupby("user_id").size()
        groups = [int(grp.get(u, 0)) for u in users]

    feats = cand_df.merge(ui, on=["user_id", "item_id"], how="left")

    feats["ui_view_cnt"] = feats["ui_view_cnt"].fillna(0).astype(np.int16)
    feats["ui_view_cnt_40h"] = feats["ui_view_cnt_40h"].fillna(0).astype(np.int16)
    feats["ui_cart_flag"] = feats["ui_cart_flag"].fillna(0).astype(np.int8)
    feats["ui_last_hours_ago"] = feats["ui_last_hours_ago"].fillna(9999.0).astype(np.float32)
    feats["ui_last_dow"] = feats["ui_last_dow"].fillna(-1).astype(np.int8)
    feats["ui_last_hour"] = feats["ui_last_hour"].fillna(-1).astype(np.int8)

    feats["repeat2"] = (feats["ui_view_cnt"] >= 2).astype(np.int8)
    feats["repeat3"] = (feats["ui_view_cnt"] >= 3).astype(np.int8)
    feats["repeat5"] = (feats["ui_view_cnt"] >= 5).astype(np.int8)

    feats = feats.merge(item_stats, on="item_id", how="left")
    feats["item_view_pop"] = feats["item_view_pop"].fillna(0).astype(np.int32)
    feats["item_purchase_pop"] = feats["item_purchase_pop"].fillna(0).astype(np.int32)
    feats["item_purchase_rate"] = feats["item_purchase_rate"].fillna(0.0).astype(np.float32)
    feats["item_price"] = feats["item_price"].fillna(-1.0).astype(np.float32)
    feats["price_bucket"] = feats["price_bucket"].fillna(-1).astype(np.int16)
    feats["price_bonus"] = feats["price_bonus"].fillna(1.0).astype(np.float32)

    for c in ["is_peak", "is_active", "is_thu", "is_fri", "is_sat"]:
        feats[c] = feats[c].fillna(0).astype(np.int8)

    feats["src_cart"] = (feats["src_code"] == 3).astype(np.int8)
    feats["src_repeat"] = (feats["src_code"] == 2).astype(np.int8)
    feats["src_recent"] = (feats["src_code"] == 1).astype(np.int8)
    feats["src_popular"] = (feats["src_code"] == 0).astype(np.int8)
    feats["src_priority"] = feats["src_code"].astype(np.float32)

    feats["v5_score"] = (
        feats["src_priority"]
        * feats["price_bonus"]
        * (1.0 + 0.5 * feats["item_purchase_rate"])
        + 0.05 * np.log1p(feats["item_view_pop"].astype(np.float32))
    ).astype(np.float32)

    # 클러스터 관련 피처
    if item_profile is not None and "cluster_id" in item_profile.columns:
        logging.info("[Feature] Adding cluster features...")
        cluster_map = item_profile.set_index("item_id")[["cluster_id", "avg_views_per_user", "view_to_purchase_rate", "fine_category"]]
        feats = feats.merge(
            cluster_map.reset_index()[["item_id", "cluster_id", "avg_views_per_user", "view_to_purchase_rate", "fine_category"]],
            on="item_id",
            how="left",
            suffixes=("", "_profile")
        )
        feats["item_cluster_id"] = feats["cluster_id"].fillna(-1).astype(np.int16)
        feats["item_avg_views_per_user"] = feats["avg_views_per_user"].fillna(1.0).astype(np.float32)
        feats["item_profile_conversion"] = feats["view_to_purchase_rate"].fillna(0.0).astype(np.float32)
        feats["item_fine_category"] = feats["fine_category"].fillna("other_c-1")

        n_missing_cluster = (feats["item_cluster_id"] == -1).sum()
        logging.info(f"  item_cluster_id: missing={n_missing_cluster} ({n_missing_cluster/len(feats)*100:.2f}%)")

        MIN_VIEWS_FOR_FULL_SCORE = 5

        if user_view_map is not None:
            logging.info("  Computing cluster_match_score...")

            item_to_cluster = item_profile.set_index("item_id")["cluster_id"].to_dict()

            unique_users = feats["user_id"].unique()
            user_cluster_affinity_local = {}
            user_view_count = {}

            for u in unique_users:
                user_views = user_view_map.get(u, [])
                n_views = len(user_views)
                user_view_count[u] = n_views

                if n_views == 0:
                    continue
                cluster_counts = defaultdict(int)
                for item in user_views:
                    c = item_to_cluster.get(item, -1)
                    if c != -1:
                        cluster_counts[c] += 1
                if cluster_counts:
                    total = sum(cluster_counts.values())
                    user_cluster_affinity_local[u] = {c: cnt / total for c, cnt in cluster_counts.items()}

            logging.info(f"    Pre-computed affinity for {len(user_cluster_affinity_local):,} users")

            affinity_records = []
            for u, aff_dict in user_cluster_affinity_local.items():
                n_views = user_view_count.get(u, 0)
                view_penalty = min(n_views / MIN_VIEWS_FOR_FULL_SCORE, 1.0)

                for c, score in aff_dict.items():
                    penalized_score = score * view_penalty
                    affinity_records.append((u, c, penalized_score))

            if affinity_records:
                affinity_df = pd.DataFrame(affinity_records, columns=["user_id", "cluster_id", "affinity_score"])
                feats = feats.merge(
                    affinity_df,
                    left_on=["user_id", "item_cluster_id"],
                    right_on=["user_id", "cluster_id"],
                    how="left"
                )
                feats["cluster_match_score"] = feats["affinity_score"].fillna(0.0).astype(np.float32)
                feats = feats.drop(columns=["cluster_id", "affinity_score"], errors="ignore")
            else:
                feats["cluster_match_score"] = 0.0

            logging.info(f"  cluster_match_score: mean={feats['cluster_match_score'].mean():.4f}")

            # fine_category_match_score
            if "fine_category" in item_profile.columns:
                logging.info("  Computing fine_category_match_score...")

                item_to_fine_category = item_profile.set_index("item_id")["fine_category"].to_dict()

                user_fine_category_affinity_local = {}
                for u in unique_users:
                    user_views = user_view_map.get(u, [])
                    if len(user_views) == 0:
                        continue
                    fine_cat_counts = defaultdict(int)
                    for item in user_views:
                        fine_cat = item_to_fine_category.get(item, "other_c-1")
                        if fine_cat and fine_cat != "other_c-1":
                            fine_cat_counts[fine_cat] += 1
                    if fine_cat_counts:
                        total = sum(fine_cat_counts.values())
                        user_fine_category_affinity_local[u] = {cat: cnt / total for cat, cnt in fine_cat_counts.items()}

                fine_cat_affinity_records = []
                for u, aff_dict in user_fine_category_affinity_local.items():
                    n_views = user_view_count.get(u, 0)
                    view_penalty = min(n_views / MIN_VIEWS_FOR_FULL_SCORE, 1.0)

                    for cat, score in aff_dict.items():
                        penalized_score = score * view_penalty
                        fine_cat_affinity_records.append((u, cat, penalized_score))

                if fine_cat_affinity_records:
                    fine_cat_affinity_df = pd.DataFrame(fine_cat_affinity_records, columns=["user_id", "fine_category", "fine_cat_affinity_score"])
                    feats = feats.merge(
                        fine_cat_affinity_df,
                        left_on=["user_id", "item_fine_category"],
                        right_on=["user_id", "fine_category"],
                        how="left"
                    )
                    feats["fine_category_match_score"] = feats["fine_cat_affinity_score"].fillna(0.0).astype(np.float32)
                    feats = feats.drop(columns=["fine_category", "fine_cat_affinity_score", "item_fine_category"], errors="ignore")
                else:
                    feats["fine_category_match_score"] = 0.0

                logging.info(f"  fine_category_match_score: mean={feats['fine_category_match_score'].mean():.4f}")
            else:
                feats["fine_category_match_score"] = 0.0

        else:
            feats["cluster_match_score"] = 0.0
            feats["fine_category_match_score"] = 0.0
    else:
        feats["item_cluster_id"] = -1
        feats["item_avg_views_per_user"] = 1.0
        feats["item_profile_conversion"] = 0.0
        feats["cluster_match_score"] = 0.0
        feats["fine_category_match_score"] = 0.0

    feature_cols = [
        "ui_view_cnt", "ui_view_cnt_40h",
        "repeat2", "repeat3", "repeat5",
        "ui_cart_flag",
        "ui_last_hours_ago",
        "ui_last_dow", "ui_last_hour",
        "is_peak", "is_active", "is_thu", "is_fri", "is_sat",
        "item_view_pop", "item_purchase_pop", "item_purchase_rate",
        "item_price", "price_bucket", "price_bonus",
        "src_cart", "src_repeat", "src_recent", "src_popular",
        "src_priority", "v5_score",
        "item_cluster_id", "item_avg_views_per_user", "item_profile_conversion",
        "cluster_match_score",
        "fine_category_match_score",
    ]

    X = feats[feature_cols]
    pairs = feats[["user_id", "item_id"]].copy()
    return X, pairs, groups


def build_feature_df_fast(cand_dict, src_dict, ui_mi, item_mi, purchased_mi=None,
                          item_cluster_map=None, user_cluster_affinity=None,
                          item_fine_category_map=None, user_fine_category_affinity=None):
    """
    빠른 피처 생성 (추론용)
    """
    users = list(cand_dict.keys())

    all_user_ids = []
    all_item_ids = []
    all_src = []
    groups = []

    for u in users:
        items = cand_dict[u]
        srcs = src_dict[u]
        groups.append(len(items))
        all_user_ids.extend([u] * len(items))
        all_item_ids.extend(items)
        all_src.extend(srcs)

    pairs = pd.DataFrame({"user_id": all_user_ids, "item_id": all_item_ids})
    src_code = np.array(all_src, dtype=np.int8)

    if len(pairs) == 0:
        X = pd.DataFrame()
        pairs["v5_score"] = []
        return X, pairs, groups

    mi = pd.MultiIndex.from_frame(pairs[["user_id", "item_id"]])
    ui_feat = ui_mi.reindex(mi)

    ui_defaults = {
        "ui_view_cnt": 0, "ui_view_cnt_40h": 0, "ui_cart_flag": 0,
        "ui_last_hours_ago": 9999.0, "ui_last_dow": -1, "ui_last_hour": -1,
        "is_peak": 0, "is_active": 0, "is_thu": 0, "is_fri": 0, "is_sat": 0,
    }
    for c, v in ui_defaults.items():
        if c in ui_feat.columns:
            ui_feat[c] = ui_feat[c].fillna(v)
        else:
            ui_feat[c] = v

    vc = ui_feat["ui_view_cnt"].to_numpy()
    repeat2 = (vc >= 2).astype(np.int8)
    repeat3 = (vc >= 3).astype(np.int8)
    repeat5 = (vc >= 5).astype(np.int8)

    item_feat = item_mi.reindex(pairs["item_id"].values)
    item_defaults = {
        "item_view_pop": 0, "item_purchase_pop": 0, "item_purchase_rate": 0.0,
        "item_price": -1.0, "price_bucket": -1, "price_bonus": 1.0,
    }
    for c, v in item_defaults.items():
        if c in item_feat.columns:
            item_feat[c] = item_feat[c].fillna(v)
        else:
            item_feat[c] = v

    src_cart = (src_code == 3).astype(np.int8)
    src_repeat = (src_code == 2).astype(np.int8)
    src_recent = (src_code == 1).astype(np.int8)
    src_popular = (src_code == 0).astype(np.int8)
    src_priority = src_code.astype(np.float32)

    v5_score = (
        src_priority
        * item_feat["price_bonus"].to_numpy(dtype=np.float32)
        * (1.0 + 0.5 * item_feat["item_purchase_rate"].to_numpy(dtype=np.float32))
        + 0.05 * np.log1p(item_feat["item_view_pop"].to_numpy(dtype=np.float32))
    ).astype(np.float32)

    item_ids = pairs["item_id"].values
    user_ids = pairs["user_id"].values

    if item_cluster_map is not None:
        item_cluster_ids = np.array([
            item_cluster_map.get(i, (-1, 1.0, 0.0))[0] for i in item_ids
        ], dtype=np.int16)
        item_avg_views = np.array([
            item_cluster_map.get(i, (-1, 1.0, 0.0))[1] for i in item_ids
        ], dtype=np.float32)
        item_profile_conv = np.array([
            item_cluster_map.get(i, (-1, 1.0, 0.0))[2] for i in item_ids
        ], dtype=np.float32)

        if user_cluster_affinity is not None:
            cluster_match = np.array([
                user_cluster_affinity.get(u, {}).get(c, 0.0)
                for u, c in zip(user_ids, item_cluster_ids)
            ], dtype=np.float32)
        else:
            cluster_match = np.zeros(len(pairs), dtype=np.float32)
    else:
        item_cluster_ids = np.full(len(pairs), -1, dtype=np.int16)
        item_avg_views = np.ones(len(pairs), dtype=np.float32)
        item_profile_conv = np.zeros(len(pairs), dtype=np.float32)
        cluster_match = np.zeros(len(pairs), dtype=np.float32)

    # fine_category 피처
    if item_fine_category_map is not None and user_fine_category_affinity is not None:
        item_fine_categories = [item_fine_category_map.get(i, "other_c-1") for i in item_ids]

        fine_category_match = np.array([
            user_fine_category_affinity.get(u, {}).get(fine_cat, 0.0)
            for u, fine_cat in zip(user_ids, item_fine_categories)
        ], dtype=np.float32)
    else:
        fine_category_match = np.zeros(len(pairs), dtype=np.float32)

    feature_cols = [
        "ui_view_cnt", "ui_view_cnt_40h",
        "repeat2", "repeat3", "repeat5",
        "ui_cart_flag",
        "ui_last_hours_ago",
        "ui_last_dow", "ui_last_hour",
        "is_peak", "is_active", "is_thu", "is_fri", "is_sat",
        "item_view_pop", "item_purchase_pop", "item_purchase_rate",
        "item_price", "price_bucket", "price_bonus",
        "src_cart", "src_repeat", "src_recent", "src_popular",
        "src_priority", "v5_score",
        "item_cluster_id", "item_avg_views_per_user", "item_profile_conversion",
        "cluster_match_score",
        "fine_category_match_score",
    ]

    X = pd.DataFrame({
        "ui_view_cnt": ui_feat["ui_view_cnt"].to_numpy(dtype=np.int16),
        "ui_view_cnt_40h": ui_feat["ui_view_cnt_40h"].to_numpy(dtype=np.int16),
        "repeat2": repeat2, "repeat3": repeat3, "repeat5": repeat5,
        "ui_cart_flag": ui_feat["ui_cart_flag"].to_numpy(dtype=np.int8),
        "ui_last_hours_ago": ui_feat["ui_last_hours_ago"].to_numpy(dtype=np.float32),
        "ui_last_dow": ui_feat["ui_last_dow"].to_numpy(dtype=np.int8),
        "ui_last_hour": ui_feat["ui_last_hour"].to_numpy(dtype=np.int8),
        "is_peak": ui_feat["is_peak"].to_numpy(dtype=np.int8),
        "is_active": ui_feat["is_active"].to_numpy(dtype=np.int8),
        "is_thu": ui_feat["is_thu"].to_numpy(dtype=np.int8),
        "is_fri": ui_feat["is_fri"].to_numpy(dtype=np.int8),
        "is_sat": ui_feat["is_sat"].to_numpy(dtype=np.int8),
        "item_view_pop": item_feat["item_view_pop"].to_numpy(dtype=np.int32),
        "item_purchase_pop": item_feat["item_purchase_pop"].to_numpy(dtype=np.int32),
        "item_purchase_rate": item_feat["item_purchase_rate"].to_numpy(dtype=np.float32),
        "item_price": item_feat["item_price"].to_numpy(dtype=np.float32),
        "price_bucket": item_feat["price_bucket"].to_numpy(dtype=np.int16),
        "price_bonus": item_feat["price_bonus"].to_numpy(dtype=np.float32),
        "src_cart": src_cart,
        "src_repeat": src_repeat,
        "src_recent": src_recent,
        "src_popular": src_popular,
        "src_priority": src_priority,
        "v5_score": v5_score,
        "item_cluster_id": item_cluster_ids,
        "item_avg_views_per_user": item_avg_views,
        "item_profile_conversion": item_profile_conv,
        "cluster_match_score": cluster_match,
        "fine_category_match_score": fine_category_match,
    })[feature_cols]

    pairs["v5_score"] = v5_score

    return X, pairs, groups



def prepare_indexed_lookups(ui, item_stats, purchased_pairs=None):
    """
    MultiIndex 기반 lookup 객체 생성
    """
    logging.info("  Preparing indexed lookups (MultiIndex).")

    ui_cols = [
        "ui_view_cnt", "ui_view_cnt_40h", "ui_cart_flag",
        "ui_last_hours_ago", "ui_last_dow", "ui_last_hour",
        "is_peak", "is_active", "is_thu", "is_fri", "is_sat",
    ]
    ui_cols = [c for c in ui_cols if c in ui.columns]

    ui_mi = ui.set_index(["user_id", "item_id"])[ui_cols]

    item_cols = [
        "item_view_pop", "item_purchase_pop", "item_purchase_rate",
        "item_price", "price_bucket", "price_bonus",
    ]
    item_cols = [c for c in item_cols if c in item_stats.columns]

    item_mi = item_stats.set_index("item_id")[item_cols]

    purchased_mi = None
    if purchased_pairs is not None and len(purchased_pairs) > 0:
        purchased_mi = pd.MultiIndex.from_frame(purchased_pairs[["user_id", "item_id"]])

    logging.info(f"  UI index: {len(ui_mi):,}, Item index: {len(item_mi):,}")
    return ui_mi, item_mi, purchased_mi



# =========================
# 6) Train ranker (XGBoost GPU)
# =========================
def train_ranker_xgb(X_train, y_train, group_train, X_val, y_val, group_val):
    """
    XGBoost GPU Learning to Rank 학습
    RTX 3090 24GB에 최적화된 파라미터 사용
    """
    logging.info("[Train] Starting XGBoost GPU training...")
    logging.info(f"  GPU parameters: device={XGB_GPU_PARAMS['device']}, tree_method={XGB_GPU_PARAMS['tree_method']}")
    logging.info(f"  Tree parameters: max_depth={XGB_GPU_PARAMS['max_depth']}, max_leaves={XGB_GPU_PARAMS['max_leaves']}")
    logging.info(f"  Learning parameters: lr={XGB_GPU_PARAMS['learning_rate']}, n_estimators={XGB_GPU_PARAMS['n_estimators']}")

    # XGBoost DMatrix 생성
    # Learning to Rank에서는 group 정보가 필요함
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(group_train)

    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group(group_val)

    # 파라미터 설정
    params = {
        "device": XGB_GPU_PARAMS["device"],
        "tree_method": XGB_GPU_PARAMS["tree_method"],
        "objective": XGB_GPU_PARAMS["objective"],
        "eval_metric": XGB_GPU_PARAMS["eval_metric"],
        "eta": XGB_GPU_PARAMS["learning_rate"],  # XGBoost에서는 eta
        "max_depth": XGB_GPU_PARAMS["max_depth"],
        "max_leaves": XGB_GPU_PARAMS["max_leaves"],
        "min_child_weight": XGB_GPU_PARAMS["min_child_weight"],
        "reg_alpha": XGB_GPU_PARAMS["reg_alpha"],
        "reg_lambda": XGB_GPU_PARAMS["reg_lambda"],
        "gamma": XGB_GPU_PARAMS["gamma"],
        "subsample": XGB_GPU_PARAMS["subsample"],
        "colsample_bytree": XGB_GPU_PARAMS["colsample_bytree"],
        "colsample_bylevel": XGB_GPU_PARAMS["colsample_bylevel"],
        "max_bin": XGB_GPU_PARAMS["max_bin"],
        "seed": XGB_GPU_PARAMS["seed"],
        "verbosity": XGB_GPU_PARAMS["verbosity"],
    }

    evals = [(dtrain, "train"), (dval, "eval")]

    # 학습
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=XGB_GPU_PARAMS["n_estimators"],
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=100  # 100 iteration마다 출력
    )

    best_iter = bst.best_iteration
    logging.info(f"[Train] best_iteration={best_iter}")

    return bst, best_iter


# =========================
# 7) Cold user recommendation
# =========================
def recommend_for_cold_user(
    user_id,
    popular_recent,
    item_stats,
    purchased_set,
    K=10
):
    """콜드 유저 전용 추천"""
    out = []
    seen = set()
    bought = purchased_set.get(user_id, set())

    for i in popular_recent:
        if i in bought or i in seen:
            continue
        out.append(i)
        seen.add(i)
        if len(out) >= K:
            return out

    cand = (
        item_stats
        .sort_values(["item_purchase_rate", "item_price"], ascending=[False, True])
        ["item_id"]
        .tolist()
    )

    for i in cand:
        if i in bought or i in seen:
            continue
        out.append(i)
        if len(out) >= K:
            break

    return out


# =========================
# Main
# =========================
def main():
    logger = setup_logging()
    analysis = {
        "timestamp": TIMESTAMP,
        "version": "v26_fine_category_xgboost_gpu",
        "model": "XGBoost GPU",
        "gpu_config": XGB_GPU_PARAMS,
        "config": {
            "CAND_CART": CAND_CART,
            "CAND_REPEAT": CAND_REPEAT,
            "CAND_RECENT": CAND_RECENT,
            "CAND_POP": CAND_POP,
            "CAND_MAX": CAND_MAX,
            "VAL_FUTURE_DAYS": VAL_FUTURE_DAYS,
            "MAX_TRAIN_USERS": MAX_TRAIN_USERS,
        },
        "data_stats": {},
        "training_stats": {},
        "inference_stats": {},
        "output_stats": {},
    }

    logging.info("=" * 60)
    logging.info("XGBoost GPU Rerank v26 - fine_category (real_category + cluster)")
    logging.info("=" * 60)
    logging.info("✅ Key features:")
    logging.info("  1. XGBoost GPU acceleration (optimized for RTX 3090 24GB)")
    logging.info("  2. fine_category = real_category + cluster_id")
    logging.info("  3. Learning to Rank with NDCG@10 objective")
    logging.info("  4. All users go through XGBoost ranking (no special handling for low-view users)")
    logging.info("=" * 60)
    logging.info(f"Output: {OUT_PATH}")
    logging.info(f"Log: {LOG_PATH}")
    logging.info(f"Analysis: {ANALYSIS_PATH}")

    safe_mkdir(OUT_DIR)
    df, sample, target_users = load_data()

    # 시간 split
    T_end = df["event_time"].max()
    T_cut = T_end - pd.Timedelta(days=VAL_FUTURE_DAYS)
    past_logs = df[df["event_time"] < T_cut].copy()
    future_logs = df[df["event_time"] >= T_cut].copy()
    logging.info(f"[Split] past_rows={len(past_logs):,}, future_rows={len(future_logs):,}, T_cut={T_cut}")

    analysis["data_stats"]["total_rows"] = len(df)
    analysis["data_stats"]["past_rows"] = len(past_logs)
    analysis["data_stats"]["future_rows"] = len(future_logs)
    analysis["data_stats"]["target_users"] = len(target_users)

    item_stats, popular_items = build_global_stats(df)
    analysis["data_stats"]["n_items_with_views"] = len(item_stats)

    # 아이템 행동 프로파일 생성 및 클러스터링
    logging.info("=" * 60)
    logging.info("Building Item Behavior Profiles & Clustering (with fine_category)")
    logging.info("=" * 60)
    item_profile = build_item_behavior_profile(df)
    item_profile, cluster_stats, kmeans_model, scaler = cluster_items_by_behavior(item_profile, n_clusters=N_CLUSTERS)

    cluster_top_items = build_cluster_top_items(item_profile, item_stats, top_k=20)
    brand_cluster_map = build_brand_cluster_map(item_profile)

    item_stats = item_stats.merge(
        item_profile[["item_id", "cluster_id", "avg_views_per_user", "view_to_purchase_rate"]],
        on="item_id",
        how="left"
    )
    item_stats["cluster_id"] = item_stats["cluster_id"].fillna(-1).astype(np.int16)

    # 아이템 클러스터 맵
    item_cluster_map = {}
    for _, row in item_profile.iterrows():
        item_cluster_map[row["item_id"]] = (
            int(row["cluster_id"]),
            float(row["avg_views_per_user"]),
            float(row["view_to_purchase_rate"])
        )

    # 아이템 fine_category 맵
    item_fine_category_map = item_profile.set_index("item_id")["fine_category"].to_dict()
    n_fine_categories = len(set(item_fine_category_map.values()))
    logging.info(f"  item_fine_category_map built: {len(item_fine_category_map):,} items, {n_fine_categories} unique fine categories")

    analysis["data_stats"]["n_clusters"] = N_CLUSTERS
    analysis["data_stats"]["n_fine_categories"] = n_fine_categories

    # ui 집계
    ui_past, max_time_past = build_user_item_agg(past_logs)

    # 학습 유저
    future_purchases = future_logs[future_logs["event_type"] == "purchase"][["user_id", "item_id"]].drop_duplicates()
    pos_users = future_purchases["user_id"].unique().tolist()

    if len(pos_users) > MAX_TRAIN_USERS:
        pos_users = pos_users[:MAX_TRAIN_USERS]

    active_users = past_logs[past_logs["event_type"] == "view"]["user_id"].dropna().unique().tolist()
    active_users = [u for u in active_users if u not in set(pos_users)]
    neg_add = max(0, min(len(active_users), MAX_TRAIN_USERS - len(pos_users)))
    if neg_add > 0:
        pos_users += active_users[:neg_add]

    train_users, val_users = split_users_hash(pos_users, valid_ratio=0.2, seed=42)
    logging.info(f"[Users] train_users={len(train_users):,}, val_users={len(val_users):,}")
    analysis["training_stats"]["train_users"] = len(train_users)
    analysis["training_stats"]["val_users"] = len(val_users)

    # 후보 생성
    cand_train, src_train = build_candidates(past_logs, train_users, ui_past, popular_items)
    cand_val, src_val = build_candidates(past_logs, val_users, ui_past, popular_items)

    past_purchases = past_logs[past_logs["event_type"] == "purchase"][["user_id", "item_id"]].drop_duplicates()

    # 유저별 view 히스토리
    logging.info("[Feature] Building user view history for cluster/fine_category matching...")
    user_view_map = past_logs[past_logs["event_type"] == "view"].groupby("user_id")["item_id"].apply(list).to_dict()
    logging.info(f"  User view map built for {len(user_view_map):,} users")

    # 피처 생성
    Xtr, ptr, gtr = build_feature_df(
        cand_train, src_train, ui_past, item_stats, max_time_past,
        exclude_purchased_pairs=past_purchases,
        item_profile=item_profile, user_view_map=user_view_map
    )
    Xva, pva, gva = build_feature_df(
        cand_val, src_val, ui_past, item_stats, max_time_past,
        exclude_purchased_pairs=past_purchases,
        item_profile=item_profile, user_view_map=user_view_map
    )

    # Implicit weighted labels
    event_weight = {
        "purchase": 5,
        "cart": 3,
        "view": 1,
    }

    future_ev = future_logs[["user_id", "item_id", "event_type"]].drop_duplicates()

    future_label_map = {}
    for u, i, e in future_ev.itertuples(index=False):
        w = event_weight.get(e, 0)
        if w == 0:
            continue
        future_label_map[(u, i)] = max(future_label_map.get((u, i), 0), w)

    def make_weighted_label(pairs):
        return np.array(
            [future_label_map.get((u, i), 0) for u, i in pairs[["user_id", "item_id"]].itertuples(index=False)],
            dtype=np.float32,
        )

    ytr = make_weighted_label(ptr)
    yva = make_weighted_label(pva)

    logging.info(f"[Label] train_nonzero={int((ytr>0).sum()):,}/{len(ytr):,}, val_nonzero={int((yva>0).sum()):,}/{len(yva):,}")
    analysis["training_stats"]["train_samples"] = len(ytr)
    analysis["training_stats"]["train_positives"] = int((ytr > 0).sum())
    analysis["training_stats"]["val_samples"] = len(yva)
    analysis["training_stats"]["val_positives"] = int((yva > 0).sum())

    # group에서 0인 유저 제거
    def filter_empty_groups(X, pairs, y, groups, users):
        keep_user_mask = [g > 0 for g in groups]
        kept_users = [u for u, k in zip(users, keep_user_mask) if k]
        m = pairs["user_id"].isin(set(kept_users))
        X2 = X[m].reset_index(drop=True)
        pairs2 = pairs[m].reset_index(drop=True)
        y2 = y[m.values]
        grp = pairs2.groupby("user_id").size()
        groups2 = [int(grp.get(u, 0)) for u in kept_users]
        return X2, pairs2, y2, groups2, kept_users

    Xtr, ptr, ytr, gtr, train_users_kept = filter_empty_groups(Xtr, ptr, ytr, gtr, list(cand_train.keys()))
    Xva, pva, yva, gva, val_users_kept = filter_empty_groups(Xva, pva, yva, gva, list(cand_val.keys()))

    logging.info(f"[Group] train_groups={len(gtr):,}, val_groups={len(gva):,}")
    analysis["training_stats"]["train_groups"] = len(gtr)
    analysis["training_stats"]["val_groups"] = len(gva)
    analysis["training_stats"]["feature_cols"] = list(Xtr.columns)

    # XGBoost GPU 학습
    ranker, best_iter = train_ranker_xgb(Xtr, ytr, gtr, Xva, yva, gva)
    analysis["training_stats"]["best_iteration"] = best_iter

    # Feature importance 저장
    importance = ranker.get_score(importance_type='gain')
    analysis["training_stats"]["feature_importance"] = importance
    logging.info("[Train] Feature Importance (Top 10):")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for fname, fval in sorted_imp[:10]:
        logging.info(f"  {fname}: {fval:.2f}")

    # 학습 데이터 메모리 해제
    del Xtr, Xva, ytr, yva, ptr, pva
    gc.collect()

    # =========================
    # Inference
    # =========================
    logging.info("[Infer] build agg on full logs for inference.")
    ui_full, max_time_full = build_user_item_agg(df)

    RECENT_DAYS = 14
    T_recent = df["event_time"].max() - pd.Timedelta(days=RECENT_DAYS)
    popular_recent = (
        df[df["event_time"] >= T_recent]
        .groupby("item_id")["user_id"]
        .count()
        .sort_values(ascending=False)
        .head(500)
        .index
        .tolist()
    )
    logging.info(f"[Infer] Recent trending items (last {RECENT_DAYS} days): {len(popular_recent)}")

    purchased_pairs_full = df[df["event_type"] == "purchase"][["user_id", "item_id"]].drop_duplicates()
    purchased_map = purchased_pairs_full.groupby("user_id")["item_id"].apply(set).to_dict()

    # 전체 데이터 기반 유저 view 히스토리
    logging.info("[Infer] Building user view history for inference...")
    user_view_map_full = df[df["event_type"] == "view"].groupby("user_id")["item_id"].apply(list).to_dict()
    logging.info(f"  User view map (full) built for {len(user_view_map_full):,} users")

    # 유저별 클러스터 affinity 사전 계산
    logging.info("[Infer] Pre-computing user cluster affinity (with view penalty)...")
    MIN_VIEWS_FOR_FULL_SCORE = 5
    user_cluster_affinity = {}

    for user_id, view_history in user_view_map_full.items():
        affinity = get_user_cluster_affinity(user_id, view_history, item_profile)
        if affinity:
            n_views = len(view_history)
            view_penalty = min(n_views / MIN_VIEWS_FOR_FULL_SCORE, 1.0)
            if view_penalty < 1.0:
                affinity = {c: score * view_penalty for c, score in affinity.items()}
            user_cluster_affinity[user_id] = affinity

    logging.info(f"  User cluster affinity computed for {len(user_cluster_affinity):,} users")

    # 유저별 fine_category affinity 사전 계산
    logging.info("[Infer] Pre-computing user fine_category affinity...")
    user_fine_category_affinity = {}

    for user_id, view_history in user_view_map_full.items():
        n_views = len(view_history)
        if n_views == 0:
            continue

        fine_cat_counts = defaultdict(int)
        for item in view_history:
            fine_cat = item_fine_category_map.get(item, "other_c-1")
            if fine_cat and fine_cat != "other_c-1":
                fine_cat_counts[fine_cat] += 1

        if fine_cat_counts:
            total = sum(fine_cat_counts.values())
            view_penalty = min(n_views / MIN_VIEWS_FOR_FULL_SCORE, 1.0)
            user_fine_category_affinity[user_id] = {
                cat: (cnt / total) * view_penalty for cat, cnt in fine_cat_counts.items()
            }

    logging.info(f"  User fine_category affinity computed for {len(user_fine_category_affinity):,} users")

    # 인덱스 lookup 준비
    logging.info("[Infer] preparing indexed lookups.")
    ui_indexed, item_stats_indexed, purchased_set = prepare_indexed_lookups(
        ui_full, item_stats, purchased_pairs_full
    )

    # 후보 전처리
    logging.info("[Infer] precomputing candidates ONCE for all target users...")
    df_min = df[df["event_type"].isin(["view", "cart"])][["user_id", "item_id", "event_time", "event_type"]].copy()
    cand_all, src_all = build_candidates(df_min, target_users, ui_full, popular_items)
    del df_min
    gc.collect()
    logging.info("[Infer] candidate precompute done.")

    total_cands = sum(len(v) for v in cand_all.values())
    avg_cands = total_cands / len(cand_all) if cand_all else 0
    logging.info(f"[Infer] Total candidates: {total_cands:,}, Avg per user: {avg_cands:.1f}")

    # 배치 단위로 추론
    BATCH_SIZE = 50000
    all_recs = []

    n_batches = (len(target_users) + BATCH_SIZE - 1) // BATCH_SIZE
    logging.info(f"[Infer] processing {len(target_users):,} users in {n_batches} batches.")

    popular_items_list = list(popular_items)

    # Blend 설정
    BLEND_ALPHA = 0.35
    USE_RANK_BLEND = True
    logging.info(f"[Infer] Blend settings: ALPHA={BLEND_ALPHA}, RANK_BLEND={USE_RANK_BLEND}")

    user_activity = ui_full.groupby("user_id")["ui_view_cnt"].sum()

    fallback_count = 0
    cold_user_count = 0

    for batch_start in range(0, len(target_users), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(target_users))
        batch_users = target_users[batch_start:batch_end]
        batch_num = batch_start // BATCH_SIZE + 1

        logging.info(f"  Batch {batch_num}/{n_batches}: users {batch_start:,}-{batch_end:,}")

        cand_batch = {u: cand_all.get(u, []) for u in batch_users}
        src_batch = {u: src_all.get(u, [0]*len(cand_batch[u])) for u in batch_users}

        # 배치 피처 생성
        Xte, pte, gte = build_feature_df_fast(
            cand_batch, src_batch, ui_indexed, item_stats_indexed, purchased_mi=None,
            item_cluster_map=item_cluster_map, user_cluster_affinity=user_cluster_affinity,
            item_fine_category_map=item_fine_category_map, user_fine_category_affinity=user_fine_category_affinity
        )

        # XGBoost 예측
        dtest = xgb.DMatrix(Xte)
        yscore = ranker.predict(dtest, iteration_range=(0, best_iter))

        # BLEND: XGBoost score + v5_score
        EPS = 1e-9

        pte = pte.copy()
        pte["xgb_score"] = yscore.astype(np.float32)

        if "v5_score" not in pte.columns:
            raise RuntimeError("pte has no 'v5_score'. Check build_feature_df_fast().")

        if USE_RANK_BLEND:
            pte["xgb_rank"] = pte.groupby("user_id")["xgb_score"].rank(method="average", pct=True)
            pte["v5_rank"]   = pte.groupby("user_id")["v5_score"].rank(method="average", pct=True)
            pte["score"] = (BLEND_ALPHA * pte["xgb_rank"] + (1.0 - BLEND_ALPHA) * pte["v5_rank"]).astype(np.float32)
            pte.drop(columns=["xgb_rank", "v5_rank"], inplace=True)
        else:
            l = pte["xgb_score"].to_numpy(dtype=np.float32)
            v = pte["v5_score"].to_numpy(dtype=np.float32)
            l = (l - l.mean()) / (l.std() + EPS)
            v = (v - v.mean()) / (v.std() + EPS)
            pte["score"] = (BLEND_ALPHA * l + (1.0 - BLEND_ALPHA) * v).astype(np.float32)

        # 구매 제외
        mi_pte = pd.MultiIndex.from_frame(pte[["user_id", "item_id"]])
        mask = ~mi_pte.isin(purchased_set)
        pte = pte.loc[mask].copy()

        pte = pte.drop_duplicates(["user_id", "item_id"], keep="first")

        pte_sorted = pte.sort_values(["user_id", "score"], ascending=[True, False])
        topk_df = pte_sorted.groupby("user_id", sort=False).head(10)

        topk_map = topk_df.groupby("user_id")["item_id"].apply(list).to_dict()

        # Slot-based final assembly
        for u in batch_users:
            recs = topk_map.get(u, [])
            activity = user_activity.get(u, 0)

            if activity <= 0:
                cold_user_count += 1
                cold_recs = recommend_for_cold_user(
                    u,
                    popular_recent,
                    item_stats,
                    purchased_map,
                    K=10
                )
                for i in cold_recs:
                    all_recs.append({"user_id": u, "item_id": i})
                continue

            seen = set()
            for i in recs:
                if i not in seen:
                    all_recs.append({"user_id": u, "item_id": i})
                    seen.add(i)

            if len(seen) < 10:
                fallback_count += 1
                for pi in popular_items_list:
                    if pi in purchased_map.get(u, set()) or pi in seen:
                        continue
                    all_recs.append({"user_id": u, "item_id": pi})
                    seen.add(pi)
                    if len(seen) >= 10:
                        break

        del Xte, pte, gte, pte_sorted, topk_df, cand_batch
        gc.collect()

    logging.info("[Submit] build submission df.")
    sub = pd.DataFrame(all_recs)

    n_users_in_sub = sub["user_id"].nunique()
    n_items_in_sub = sub["item_id"].nunique()
    recs_per_user = sub.groupby("user_id").size()

    analysis["output_stats"]["total_recs"] = len(sub)
    analysis["output_stats"]["n_users"] = n_users_in_sub
    analysis["output_stats"]["n_unique_items"] = n_items_in_sub
    analysis["output_stats"]["cold_users"] = cold_user_count
    analysis["output_stats"]["fallback_users"] = fallback_count
    analysis["output_stats"]["avg_recs_per_user"] = float(recs_per_user.mean())

    logging.info(f"[Output] Total recs: {len(sub):,}")
    logging.info(f"[Output] Users: {n_users_in_sub:,}")
    logging.info(f"[Output] Unique items: {n_items_in_sub:,}")
    logging.info(f"[Output] Cold users (activity<=0): {cold_user_count:,} ({cold_user_count/n_users_in_sub*100:.1f}%)")
    logging.info(f"[Output] Fallback users: {fallback_count:,}")
    logging.info(f"[Output] Recs per user: min={recs_per_user.min()}, max={recs_per_user.max()}, avg={recs_per_user.mean():.2f}")

    # Top 추천 아이템 분포
    item_counts = sub["item_id"].value_counts()
    logging.info("[Output] Top 10 recommended items:")
    for i, (item, cnt) in enumerate(item_counts.head(10).items()):
        pct = cnt / n_users_in_sub * 100
        logging.info(f"  {i+1}. {str(item)[:40]}: {cnt:,} ({pct:.2f}%)")

    # 결과 저장
    safe_mkdir(OUT_DIR)
    sub.to_csv(OUT_PATH, index=False)
    logging.info(f"[Done] Saved submission: {OUT_PATH}")

    # 분석 JSON 저장
    with open(ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    logging.info(f"[Done] Saved analysis: {ANALYSIS_PATH}")

    logging.info("=" * 60)
    logging.info("XGBoost GPU Rerank v26 Complete! (with fine_category)")
    logging.info("=" * 60)

    del df, past_logs, future_logs, ui_past, ui_full
    gc.collect()


if __name__ == "__main__":
    main()
