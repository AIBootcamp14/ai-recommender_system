"""
Neo4j Graph Data Loader
- User, Item, Category, Brand 노드 생성
- VIEWED, PURCHASED, BELONGS_TO, HAS_BRAND 관계 생성
"""
import os
import argparse
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm


class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """기존 데이터 삭제"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")

    def create_constraints(self):
        """인덱스 및 제약조건 생성"""
        with self.driver.session() as session:
            # Constraints for unique IDs
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Item) REQUIRE i.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE")
            print("Constraints created")

    def load_items(self, df):
        """아이템 노드 생성"""
        items = df[['item_id', 'category_code', 'brand']].drop_duplicates('item_id')

        with self.driver.session() as session:
            for _, row in tqdm(items.iterrows(), total=len(items), desc="Loading Items"):
                session.run("""
                    MERGE (i:Item {id: $item_id})
                    SET i.category = $category,
                        i.brand = $brand
                """, item_id=str(row['item_id']),
                    category=str(row['category_code']) if pd.notna(row['category_code']) else 'unknown',
                    brand=str(row['brand']) if pd.notna(row['brand']) else 'unknown')

    def load_categories(self, df):
        """카테고리 노드 생성 및 아이템 연결"""
        categories = df['category_code'].dropna().unique()

        with self.driver.session() as session:
            for cat in tqdm(categories, desc="Loading Categories"):
                # 상위 카테고리 추출 (첫 번째 레벨)
                cat_l1 = str(cat).split('.')[0] if '.' in str(cat) else str(cat)

                session.run("""
                    MERGE (c:Category {name: $category})
                    SET c.level1 = $level1
                """, category=str(cat), level1=cat_l1)

        # Item -> Category 관계
        items_cat = df[['item_id', 'category_code']].drop_duplicates()
        items_cat = items_cat[items_cat['category_code'].notna()]

        with self.driver.session() as session:
            for _, row in tqdm(items_cat.iterrows(), total=len(items_cat), desc="Linking Items to Categories"):
                session.run("""
                    MATCH (i:Item {id: $item_id})
                    MATCH (c:Category {name: $category})
                    MERGE (i)-[:BELONGS_TO]->(c)
                """, item_id=str(row['item_id']), category=str(row['category_code']))

    def load_brands(self, df):
        """브랜드 노드 생성 및 아이템 연결"""
        brands = df['brand'].dropna().unique()

        with self.driver.session() as session:
            for brand in tqdm(brands, desc="Loading Brands"):
                session.run("""
                    MERGE (b:Brand {name: $brand})
                """, brand=str(brand))

        # Item -> Brand 관계
        items_brand = df[['item_id', 'brand']].drop_duplicates()
        items_brand = items_brand[items_brand['brand'].notna()]

        with self.driver.session() as session:
            for _, row in tqdm(items_brand.iterrows(), total=len(items_brand), desc="Linking Items to Brands"):
                session.run("""
                    MATCH (i:Item {id: $item_id})
                    MATCH (b:Brand {name: $brand})
                    MERGE (i)-[:HAS_BRAND]->(b)
                """, item_id=str(row['item_id']), brand=str(row['brand']))

    def load_users_and_interactions(self, df, sample_users=10000):
        """사용자 노드 및 상호작용 생성 (샘플링)"""
        # 상호작용이 많은 사용자 샘플링
        user_counts = df.groupby('user_id').size().sort_values(ascending=False)
        sample_user_ids = user_counts.head(sample_users).index.tolist()

        df_sample = df[df['user_id'].isin(sample_user_ids)]
        print(f"Sampling {sample_users} users ({len(df_sample):,} interactions)")

        # 사용자 노드 생성
        users = df_sample['user_id'].unique()
        with self.driver.session() as session:
            for user_id in tqdm(users, desc="Loading Users"):
                session.run("""
                    MERGE (u:User {id: $user_id})
                """, user_id=str(user_id))

        # 상호작용 생성 (view, cart, purchase)
        interactions = df_sample.groupby(['user_id', 'item_id', 'event_type']).size().reset_index(name='count')

        with self.driver.session() as session:
            for _, row in tqdm(interactions.iterrows(), total=len(interactions), desc="Loading Interactions"):
                event_type = row['event_type'].upper()
                if event_type == 'VIEW':
                    rel_type = 'VIEWED'
                elif event_type == 'CART':
                    rel_type = 'ADDED_TO_CART'
                elif event_type == 'PURCHASE':
                    rel_type = 'PURCHASED'
                else:
                    continue

                session.run(f"""
                    MATCH (u:User {{id: $user_id}})
                    MATCH (i:Item {{id: $item_id}})
                    MERGE (u)-[r:{rel_type}]->(i)
                    SET r.count = $count
                """, user_id=str(row['user_id']), item_id=str(row['item_id']), count=int(row['count']))

    def load_co_viewed(self, df, min_count=50):
        """세션 내 함께 조회된 아이템 관계 생성"""
        print("\nBuilding co-viewed relationships...")

        if 'user_session' not in df.columns:
            df['session'] = df['user_id'].astype(str) + '_' + df['event_time'].dt.date.astype(str)
        else:
            df['session'] = df['user_session']

        # 세션별 아이템 쌍 계산
        from collections import defaultdict
        from itertools import combinations

        pair_counts = defaultdict(int)
        sessions = df.groupby('session')['item_id'].apply(list)

        for items in tqdm(sessions, desc="Counting co-views"):
            items = list(set(items))
            if len(items) >= 2:
                for i1, i2 in combinations(items[:20], 2):  # 최대 20개 아이템만
                    pair = tuple(sorted([str(i1), str(i2)]))
                    pair_counts[pair] += 1

        # 빈발 쌍만 저장
        frequent_pairs = [(p, c) for p, c in pair_counts.items() if c >= min_count]
        print(f"  Found {len(frequent_pairs):,} frequent pairs (count >= {min_count})")

        with self.driver.session() as session:
            for (item1, item2), count in tqdm(frequent_pairs, desc="Creating CO_VIEWED"):
                session.run("""
                    MATCH (i1:Item {id: $item1})
                    MATCH (i2:Item {id: $item2})
                    MERGE (i1)-[r:CO_VIEWED]-(i2)
                    SET r.count = $count
                """, item1=item1, item2=item2, count=count)

    def get_stats(self):
        """그래프 통계"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(*) as count
            """)
            print("\n=== Graph Statistics ===")
            for record in result:
                print(f"  {record['label']}: {record['count']:,}")

            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
            """)
            print("\n  Relationships:")
            for record in result:
                print(f"    {record['type']}: {record['count']:,}")


def main():
    parser = argparse.ArgumentParser(description='Load data to Neo4j')
    parser.add_argument('--dir_path', type=str, default='../data/', help='Data path')
    parser.add_argument('--data_dir', type=str, default='train.parquet', help='Data file')
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--user', type=str, default='neo4j', help='Neo4j user')
    parser.add_argument('--password', type=str, default='password123', help='Neo4j password')
    parser.add_argument('--sample_users', type=int, default=10000, help='Number of users to sample')
    parser.add_argument('--clear', action='store_true', help='Clear existing data')
    args = parser.parse_args()

    print("=" * 50)
    print("Neo4j Graph Data Loader")
    print("=" * 50)

    # 데이터 로드
    print("\n[1/6] Loading data...")
    df = pd.read_parquet(os.path.join(args.dir_path, args.data_dir))
    df['event_time'] = pd.to_datetime(df['event_time'])
    print(f"  Loaded {len(df):,} interactions")

    # Neo4j 연결
    print("\n[2/6] Connecting to Neo4j...")
    loader = Neo4jLoader(args.uri, args.user, args.password)

    try:
        if args.clear:
            print("\n[3/6] Clearing database...")
            loader.clear_database()

        print("\n[3/6] Creating constraints...")
        loader.create_constraints()

        print("\n[4/6] Loading nodes...")
        loader.load_items(df)
        loader.load_categories(df)
        loader.load_brands(df)

        print("\n[5/6] Loading users and interactions...")
        loader.load_users_and_interactions(df, args.sample_users)

        print("\n[6/6] Loading co-viewed relationships...")
        loader.load_co_viewed(df)

        # 통계
        loader.get_stats()

        print("\n" + "=" * 50)
        print("Graph loading complete!")
        print("=" * 50)
        print(f"  Neo4j Browser: http://localhost:7474")
        print(f"  User: {args.user}")
        print(f"  Password: {args.password}")

    finally:
        loader.close()


if __name__ == '__main__':
    main()
