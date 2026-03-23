"""
ADP 프로젝트 공유 유틸리티
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

# 시각화 기본 설정
def setup_plot_style():
    """한글 지원 matplotlib 설정"""
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    sns.set_theme(style="whitegrid", font="Malgun Gothic")

def seed_everything(seed: int = 42):
    """재현성을 위한 시드 설정"""
    np.random.seed(seed)

def generate_synthetic_data(
    n: int = 1000,
    p: int = 5,
    noise_std: float = 1.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """합성 회귀 데이터 생성"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + noise_std * rng.standard_normal(n)
    return X, y

def print_section(title: str):
    """섹션 구분 출력"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")
