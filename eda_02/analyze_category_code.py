"""
카테고리 코드(category_code) 분석 스크립트
- train.parquet 파일에서 category_code 유니크 값과 각각의 갯수를 분석
"""

import pandas as pd
from pathlib import Path

def main():
    # 데이터 경로
    data_path = Path(__file__).parent.parent / "train.parquet"

    print(f"데이터 파일 경로: {data_path}")
    print("=" * 60)

    # Parquet 파일 읽기 (category_code 컬럼만)
    print("train.parquet 파일 로딩 중...")
    df = pd.read_parquet(data_path, columns=['category_code'])

    print(f"전체 레코드 수: {len(df):,}")
    print("=" * 60)

    # 결측값 확인
    null_count = df['category_code'].isnull().sum()
    not_null_count = df['category_code'].notnull().sum()

    print(f"\n[결측값 현황]")
    print(f"  - NULL 값: {null_count:,} ({null_count/len(df)*100:.2f}%)")
    print(f"  - 유효 값: {not_null_count:,} ({not_null_count/len(df)*100:.2f}%)")
    print("=" * 60)

    # 유니크 값 분석 (NULL 제외)
    category_counts = df['category_code'].value_counts(dropna=True)

    print(f"\n[카테고리 코드 유니크 값 분석]")
    print(f"유니크 카테고리 수: {len(category_counts)}")
    print("=" * 60)

    # 전체 목록 출력 (갯수 내림차순)
    print(f"\n{'순위':<4} {'카테고리 코드':<50} {'갯수':>15} {'비율':>10}")
    print("-" * 85)

    for idx, (category, count) in enumerate(category_counts.items(), 1):
        pct = count / not_null_count * 100
        print(f"{idx:<4} {category:<50} {count:>15,} {pct:>9.2f}%")

    print("-" * 85)
    print(f"{'합계':<4} {'':<50} {category_counts.sum():>15,} {100.00:>9.2f}%")

    # 메인 카테고리 분석
    print("\n" + "=" * 60)
    print("[메인 카테고리 분석]")

    # 첫 번째 '.' 이전의 문자열 추출
    main_categories = df['category_code'].dropna().str.split('.').str[0]
    main_cat_counts = main_categories.value_counts()

    print(f"메인 카테고리 유니크 수: {len(main_cat_counts)}")
    print("-" * 40)
    for cat, count in main_cat_counts.items():
        print(f"  {cat}: {count:,} ({count/not_null_count*100:.2f}%)")

    # 서브 카테고리(2단계) 분석
    print("\n" + "=" * 60)
    print("[서브 카테고리(2단계) 분석]")

    # 첫 번째 '.' 이후의 문자열 추출 (2단계 카테고리)
    def get_sub_category(x):
        parts = x.split('.')
        return parts[1] if len(parts) > 1 else 'N/A'

    sub_categories = df['category_code'].dropna().apply(get_sub_category)
    sub_cat_counts = sub_categories.value_counts()

    print(f"서브 카테고리 유니크 수: {len(sub_cat_counts)}")
    print("-" * 40)
    for cat, count in sub_cat_counts.head(30).items():
        print(f"  {cat}: {count:,} ({count/not_null_count*100:.2f}%)")

if __name__ == "__main__":
    main()