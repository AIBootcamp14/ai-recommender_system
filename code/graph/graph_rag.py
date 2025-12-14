"""
GraphRAG for Recommendation
- 그래프 기반 추천
- LLM을 활용한 추천 이유 설명
"""
import os
import argparse
from neo4j import GraphDatabase
from typing import List, Dict, Optional


class GraphRAG:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # ==================== 그래프 쿼리 ====================

    def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """사용자의 상호작용 이력 조회"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {id: $user_id})-[r]->(i:Item)
                OPTIONAL MATCH (i)-[:BELONGS_TO]->(c:Category)
                OPTIONAL MATCH (i)-[:HAS_BRAND]->(b:Brand)
                RETURN i.id as item_id,
                       type(r) as action,
                       r.count as count,
                       c.name as category,
                       b.name as brand
                ORDER BY r.count DESC
                LIMIT $limit
            """, user_id=user_id, limit=limit)

            return [dict(record) for record in result]

    def get_similar_items(self, item_id: str, limit: int = 10) -> List[Dict]:
        """함께 조회된 아이템 (CO_VIEWED 관계 기반)"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Item {id: $item_id})-[r:CO_VIEWED]-(similar:Item)
                OPTIONAL MATCH (similar)-[:BELONGS_TO]->(c:Category)
                OPTIONAL MATCH (similar)-[:HAS_BRAND]->(b:Brand)
                RETURN similar.id as item_id,
                       r.count as co_view_count,
                       c.name as category,
                       b.name as brand
                ORDER BY r.count DESC
                LIMIT $limit
            """, item_id=item_id, limit=limit)

            return [dict(record) for record in result]

    def get_items_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """같은 카테고리 인기 아이템"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Item)-[:BELONGS_TO]->(c:Category {name: $category})
                MATCH (u:User)-[r:VIEWED]->(i)
                WITH i, sum(r.count) as view_count, c
                OPTIONAL MATCH (i)-[:HAS_BRAND]->(b:Brand)
                RETURN i.id as item_id,
                       view_count,
                       c.name as category,
                       b.name as brand
                ORDER BY view_count DESC
                LIMIT $limit
            """, category=category, limit=limit)

            return [dict(record) for record in result]

    def get_items_by_brand(self, brand: str, limit: int = 10) -> List[Dict]:
        """같은 브랜드 인기 아이템"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Item)-[:HAS_BRAND]->(b:Brand {name: $brand})
                MATCH (u:User)-[r:VIEWED]->(i)
                WITH i, sum(r.count) as view_count, b
                OPTIONAL MATCH (i)-[:BELONGS_TO]->(c:Category)
                RETURN i.id as item_id,
                       view_count,
                       c.name as category,
                       b.name as brand
                ORDER BY view_count DESC
                LIMIT $limit
            """, brand=brand, limit=limit)

            return [dict(record) for record in result]

    def get_collaborative_recommendations(self, user_id: str, limit: int = 10) -> List[Dict]:
        """협업 필터링 스타일 추천 (비슷한 사용자가 본 아이템)"""
        with self.driver.session() as session:
            result = session.run("""
                // 사용자가 본 아이템
                MATCH (u:User {id: $user_id})-[:VIEWED]->(i:Item)
                WITH u, collect(i) as viewed_items

                // 같은 아이템을 본 다른 사용자
                MATCH (other:User)-[:VIEWED]->(i:Item)
                WHERE i IN viewed_items AND other <> u

                // 그 사용자들이 본 다른 아이템
                MATCH (other)-[r:VIEWED]->(rec:Item)
                WHERE NOT rec IN viewed_items

                // 집계
                WITH rec, count(DISTINCT other) as user_count, sum(r.count) as total_views
                OPTIONAL MATCH (rec)-[:BELONGS_TO]->(c:Category)
                OPTIONAL MATCH (rec)-[:HAS_BRAND]->(b:Brand)

                RETURN rec.id as item_id,
                       user_count,
                       total_views,
                       c.name as category,
                       b.name as brand
                ORDER BY user_count DESC, total_views DESC
                LIMIT $limit
            """, user_id=user_id, limit=limit)

            return [dict(record) for record in result]

    def get_path_based_recommendations(self, user_id: str, limit: int = 10) -> List[Dict]:
        """경로 기반 추천 (User -> Item -> Category -> Item)"""
        with self.driver.session() as session:
            result = session.run("""
                // 사용자가 본 아이템의 카테고리
                MATCH (u:User {id: $user_id})-[:VIEWED]->(i:Item)-[:BELONGS_TO]->(c:Category)
                WITH u, collect(DISTINCT i) as viewed, collect(DISTINCT c) as categories

                // 같은 카테고리의 다른 인기 아이템
                MATCH (rec:Item)-[:BELONGS_TO]->(c:Category)
                WHERE c IN categories AND NOT rec IN viewed
                MATCH (:User)-[r:VIEWED]->(rec)

                WITH rec, c, sum(r.count) as popularity
                OPTIONAL MATCH (rec)-[:HAS_BRAND]->(b:Brand)

                RETURN rec.id as item_id,
                       c.name as category,
                       b.name as brand,
                       popularity
                ORDER BY popularity DESC
                LIMIT $limit
            """, user_id=user_id, limit=limit)

            return [dict(record) for record in result]

    # ==================== 추천 이유 생성 ====================

    def explain_recommendation(self, user_id: str, item_id: str) -> str:
        """추천 이유 생성 (그래프 경로 기반)"""
        explanations = []

        with self.driver.session() as session:
            # 1. 같은 카테고리 선호
            result = session.run("""
                MATCH (u:User {id: $user_id})-[:VIEWED]->(viewed:Item)-[:BELONGS_TO]->(c:Category)<-[:BELONGS_TO]-(rec:Item {id: $item_id})
                WITH c, count(DISTINCT viewed) as viewed_count
                RETURN c.name as category, viewed_count
                ORDER BY viewed_count DESC
                LIMIT 1
            """, user_id=user_id, item_id=item_id)

            record = result.single()
            if record:
                explanations.append(
                    f"You viewed {record['viewed_count']} items in '{record['category']}' category"
                )

            # 2. 같은 브랜드 선호
            result = session.run("""
                MATCH (u:User {id: $user_id})-[:VIEWED]->(viewed:Item)-[:HAS_BRAND]->(b:Brand)<-[:HAS_BRAND]-(rec:Item {id: $item_id})
                WITH b, count(DISTINCT viewed) as viewed_count
                RETURN b.name as brand, viewed_count
                ORDER BY viewed_count DESC
                LIMIT 1
            """, user_id=user_id, item_id=item_id)

            record = result.single()
            if record:
                explanations.append(
                    f"You viewed {record['viewed_count']} items from '{record['brand']}' brand"
                )

            # 3. 함께 조회된 아이템
            result = session.run("""
                MATCH (u:User {id: $user_id})-[:VIEWED]->(viewed:Item)-[r:CO_VIEWED]-(rec:Item {id: $item_id})
                RETURN viewed.id as viewed_item, r.count as co_view_count
                ORDER BY r.count DESC
                LIMIT 1
            """, user_id=user_id, item_id=item_id)

            record = result.single()
            if record:
                explanations.append(
                    f"Users who viewed '{record['viewed_item'][:20]}...' also viewed this item ({record['co_view_count']} times)"
                )

            # 4. 비슷한 사용자
            result = session.run("""
                MATCH (u:User {id: $user_id})-[:VIEWED]->(common:Item)<-[:VIEWED]-(other:User)
                MATCH (other)-[:VIEWED]->(rec:Item {id: $item_id})
                WITH count(DISTINCT other) as similar_users, count(DISTINCT common) as common_items
                RETURN similar_users, common_items
            """, user_id=user_id, item_id=item_id)

            record = result.single()
            if record and record['similar_users'] > 0:
                explanations.append(
                    f"{record['similar_users']} users with similar taste viewed this item"
                )

        if explanations:
            return " | ".join(explanations)
        return "Popular item in your interest area"

    # ==================== 통합 추천 ====================

    def recommend(self, user_id: str, top_k: int = 10) -> List[Dict]:
        """통합 추천 (여러 전략 결합)"""
        recommendations = {}

        # 1. 협업 필터링
        collab_recs = self.get_collaborative_recommendations(user_id, limit=top_k)
        for rec in collab_recs:
            item_id = rec['item_id']
            if item_id not in recommendations:
                recommendations[item_id] = {
                    'item_id': item_id,
                    'category': rec['category'],
                    'brand': rec['brand'],
                    'score': 0,
                    'sources': []
                }
            recommendations[item_id]['score'] += rec['user_count'] * 2
            recommendations[item_id]['sources'].append('collaborative')

        # 2. 경로 기반
        path_recs = self.get_path_based_recommendations(user_id, limit=top_k)
        for rec in path_recs:
            item_id = rec['item_id']
            if item_id not in recommendations:
                recommendations[item_id] = {
                    'item_id': item_id,
                    'category': rec['category'],
                    'brand': rec['brand'],
                    'score': 0,
                    'sources': []
                }
            recommendations[item_id]['score'] += rec['popularity'] / 100
            recommendations[item_id]['sources'].append('category_based')

        # 3. 사용자 히스토리의 CO_VIEWED
        history = self.get_user_history(user_id, limit=5)
        for item in history[:5]:
            similar = self.get_similar_items(item['item_id'], limit=5)
            for rec in similar:
                item_id = rec['item_id']
                if item_id not in recommendations:
                    recommendations[item_id] = {
                        'item_id': item_id,
                        'category': rec['category'],
                        'brand': rec['brand'],
                        'score': 0,
                        'sources': []
                    }
                recommendations[item_id]['score'] += rec['co_view_count']
                if 'co_viewed' not in recommendations[item_id]['sources']:
                    recommendations[item_id]['sources'].append('co_viewed')

        # 정렬 및 상위 K개 반환
        sorted_recs = sorted(recommendations.values(), key=lambda x: x['score'], reverse=True)

        # 추천 이유 추가
        for rec in sorted_recs[:top_k]:
            rec['explanation'] = self.explain_recommendation(user_id, rec['item_id'])

        return sorted_recs[:top_k]


def main():
    parser = argparse.ArgumentParser(description='GraphRAG Recommendations')
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--user', type=str, default='neo4j', help='Neo4j user')
    parser.add_argument('--password', type=str, default='password123', help='Neo4j password')
    parser.add_argument('--user_id', type=str, help='User ID for recommendations')
    parser.add_argument('--top_k', type=int, default=10, help='Number of recommendations')
    args = parser.parse_args()

    print("=" * 60)
    print("GraphRAG Recommendation System")
    print("=" * 60)

    rag = GraphRAG(args.uri, args.user, args.password)

    try:
        if args.user_id:
            # 특정 사용자에 대한 추천
            print(f"\nRecommendations for User: {args.user_id}")
            print("-" * 60)

            # 사용자 히스토리
            history = rag.get_user_history(args.user_id, limit=10)
            if history:
                print("\nUser History:")
                for item in history[:5]:
                    print(f"  [{item['action']}] {item['item_id'][:30]}... ({item['category']})")

            # 추천
            recommendations = rag.recommend(args.user_id, top_k=args.top_k)
            print(f"\nTop {args.top_k} Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['item_id'][:40]}...")
                print(f"   Category: {rec['category']}")
                print(f"   Brand: {rec['brand']}")
                print(f"   Score: {rec['score']:.2f}")
                print(f"   Sources: {', '.join(rec['sources'])}")
                print(f"   Why: {rec['explanation']}")

        else:
            # 샘플 사용자 조회
            with rag.driver.session() as session:
                result = session.run("""
                    MATCH (u:User)-[r:VIEWED]->(:Item)
                    WITH u, count(r) as view_count
                    WHERE view_count > 10
                    RETURN u.id as user_id, view_count
                    ORDER BY view_count DESC
                    LIMIT 5
                """)

                print("\nSample Users (with most views):")
                print("-" * 40)
                for record in result:
                    print(f"  User: {record['user_id'][:30]}... ({record['view_count']} views)")

                print("\nRun with --user_id <USER_ID> to get recommendations")

    finally:
        rag.close()


if __name__ == '__main__':
    main()
