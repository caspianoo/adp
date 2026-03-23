#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chapter 01: Advanced EDA - Generate all 20 notebooks (10 problem + 10 solution)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.notebook_generator import problem_notebook, solution_notebook

BASE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Topic 01: 다변량 데이터 시각화와 차원 해석
# ============================================================
def gen_topic_01():
    problem_notebook(
        chapter_num=1, section_num=1,
        title="다변량 데이터 시각화와 차원 해석",
        objectives=[
            "평행좌표(Parallel Coordinates)와 앤드류스 곡선(Andrews Curves)의 수학적 원리를 이해한다",
            "조건부 플롯(Conditional Plot)으로 다변량 관계를 시각적으로 분해한다",
            "다차원 스케일링(MDS)의 최적화 원리와 스트레스 함수를 유도한다",
            "t-SNE, UMAP과 MDS의 이론적 차이를 비교 분석한다",
        ],
        theory_md=r"""
### 1. 평행좌표 (Parallel Coordinates)

$p$차원 관측치 $\mathbf{x} = (x_1, \ldots, x_p)$를 $p$개의 수직 평행축 위의 점으로 매핑하고 직선으로 연결한다.

각 축은 정규화: $\tilde{x}_j = \frac{x_j - x_j^{\min}}{x_j^{\max} - x_j^{\min}}$

**핵심 성질**: 인접 축 간 교차 패턴은 변수 쌍의 **부호 상관**을 반영한다.

### 2. 앤드류스 곡선 (Andrews Curves)

$p$차원 벡터를 푸리에 기저로 1차원 함수로 변환:

$$f_{\mathbf{x}}(t) = \frac{x_1}{\sqrt{2}} + x_2\sin(t) + x_3\cos(t) + x_4\sin(2t) + x_5\cos(2t) + \cdots$$

등거리(isometric) 성질: $\int_{-\pi}^{\pi}[f_{\mathbf{x}}(t) - f_{\mathbf{y}}(t)]^2 dt = \pi\|\mathbf{x}-\mathbf{y}\|^2$

PCA 축 순서로 재배열하면 분산이 큰 성분이 저주파에 배치되어 해석력이 향상된다.

### 3. 다차원 스케일링 (MDS)

비유사도 행렬 $\boldsymbol{\Delta}=[\delta_{ij}]$로부터 저차원 좌표 $\mathbf{Y}$를 찾는다.

**Classical MDS**: 이중 중심화 + 고유값 분해

$$B = -\frac{1}{2}H\Delta^{(2)}H, \quad H = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^T$$

$$\mathbf{Y} = V_k\Lambda_k^{1/2}$$

**Stress (Kruskal)**:

$$\text{Stress}_1 = \sqrt{\frac{\sum_{i<j}(d_{ij}-\hat{d}_{ij})^2}{\sum_{i<j}d_{ij}^2}}$$

### 4. 조건부 플롯 (Coplot)

변수 $Z$를 겹치는 구간으로 분할하여 각 구간에서 $X$ vs $Y$ 관계를 패널로 시각화.
Simpson's paradox 탐지에 유용하다.
""",
        guided_code=r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates, andrews_curves

# 데이터 준비
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# 평행좌표
fig, ax = plt.subplots(figsize=(12, 5))
parallel_coordinates(df, 'species', ax=ax, colormap='viridis', alpha=0.4)
ax.set_title('평행좌표: Iris 데이터셋')
plt.tight_layout(); plt.show()

# 앤드류스 곡선
fig, ax = plt.subplots(figsize=(12, 5))
andrews_curves(df, 'species', ax=ax, colormap='viridis', alpha=0.3)
ax.set_title('앤드류스 곡선: Iris 데이터셋')
plt.tight_layout(); plt.show()

# Classical MDS 직접 구현
X_scaled = StandardScaler().fit_transform(iris.data)
from scipy.spatial.distance import pdist, squareform
D = squareform(pdist(X_scaled, 'euclidean'))
n = D.shape[0]
H = np.eye(n) - np.ones((n,n))/n
B = -0.5 * H @ (D**2) @ H
eigvals, eigvecs = np.linalg.eigh(B)
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
Y_mds = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))

fig, ax = plt.subplots(figsize=(8, 6))
for i, name in enumerate(iris.target_names):
    mask = iris.target == i
    ax.scatter(Y_mds[mask, 0], Y_mds[mask, 1], label=name, alpha=0.7)
ax.set_xlabel('MDS Dim 1'); ax.set_ylabel('MDS Dim 2')
ax.set_title('Classical MDS'); ax.legend()
plt.tight_layout(); plt.show()
""",
        exercises=[
            {
                "difficulty": "★",
                "description": "6차원 합성 데이터(3클러스터, 각 100개)를 생성하고 평행좌표와 앤드류스 곡선을 그려라. 변수 순서를 PCA 기여도 순으로 재배열했을 때 앤드류스 곡선의 분리도가 개선되는지 비교하라.",
                "hint": "np.random.multivariate_normal로 클러스터를 생성하고 PCA로 변수 중요도 순서를 결정하세요.",
                "skeleton": "np.random.seed(42)\n# TODO: 6차원 3클러스터 데이터\n# TODO: 평행좌표\n# TODO: 원래 vs PCA 순서 앤드류스 곡선 비교\n",
            },
            {
                "difficulty": "★★",
                "description": "Classical MDS를 직접 구현하라 (sklearn 금지).\n1) Swiss Roll에 적용\n2) Stress 값과 스크리 플롯\n3) 유클리드 거리 vs 측지선(Isomap) 비교",
                "hint": "측지선 거리는 k-NN 그래프에서 Dijkstra로 근사.",
                "skeleton": "from sklearn.datasets import make_swiss_roll\nX_sr, color = make_swiss_roll(1000, noise=0.5, random_state=42)\n\ndef classical_mds(D, n_components=2):\n    # TODO\n    pass\n",
            },
            {
                "difficulty": "★★",
                "description": "조건부 플롯(Coplot)을 구현하라. 변수 Z를 겹치는 구간으로 분할하고 각 구간에서 X vs Y 산점도를 격자형으로 그려라. 합성 데이터에서 Simpson's paradox 유사 현상이 관측되는지 확인하라.",
                "hint": "구간 겹침(overlap) 0.3~0.5로 설정.",
                "skeleton": "def coplot(df, x_col, y_col, z_col, n_panels=6, overlap=0.3):\n    # TODO\n    pass\n",
            },
            {
                "difficulty": "★★★",
                "description": "비선형 차원 축소(t-SNE, MDS)의 거리 보존 성능을 정량 비교하라.\n1) 10차원 5클러스터 합성 데이터\n2) Trustworthiness: $T(k)=1-\\frac{2}{nk(2n-3k-1)}\\sum_{i}\\sum_{j\\in U_i^k}(r(i,j)-k)$\n3) Continuity, Shepard diagram 상관계수\n4) k에 따른 변화 그래프",
                "skeleton": "from sklearn.manifold import TSNE, MDS\nfrom sklearn.metrics import trustworthiness\n# TODO: 데이터 생성, 임베딩, 지표 계산\n",
            },
        ],
        references=[
            "Inselberg, A. (2009). Parallel Coordinates: Visual Multidimensional Geometry.",
            "Borg, I. & Groenen, P. (2005). Modern Multidimensional Scaling.",
            "van der Maaten, L. & Hinton, G. (2008). Visualizing Data using t-SNE. JMLR.",
            "Andrews, D.F. (1972). Plots of High-Dimensional Data. Biometrics.",
        ],
        filepath=os.path.join(BASE, "ch01_01_multivariate_visualization.ipynb"),
    )

    solution_notebook(
        chapter_num=1, section_num=1,
        title="다변량 데이터 시각화와 차원 해석",
        solutions=[
            {
                "approach": "PCA 기반 변수 순서 재배열로 앤드류스 곡선 분리 개선 확인",
                "code": r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates, andrews_curves
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
means = [np.array([2,0,-1,3,1,-2]), np.array([-1,3,2,-1,0,4]), np.array([0,-2,1,0,3,-1])]
cov = np.eye(6)*0.8
for i in range(6):
    for j in range(6):
        if i!=j: cov[i,j] = 0.3*((-1)**(i+j))
data = np.vstack([np.random.multivariate_normal(m, cov, 100) for m in means])
labels = np.repeat(['A','B','C'], 100)
cols = [f'X{i}' for i in range(1,7)]
df = pd.DataFrame(data, columns=cols); df['cluster'] = labels

fig, ax = plt.subplots(figsize=(12,5))
parallel_coordinates(df, 'cluster', ax=ax, alpha=0.3)
ax.set_title('평행좌표: 6차원 3클러스터'); plt.tight_layout(); plt.show()

pca = PCA(); pca.fit(StandardScaler().fit_transform(data))
order = np.argsort(np.abs(pca.components_[0]))[::-1]
new_cols = [cols[i] for i in order]

fig, axes = plt.subplots(1,2,figsize=(16,5))
andrews_curves(df[cols+['cluster']], 'cluster', ax=axes[0], alpha=0.2)
axes[0].set_title('원래 순서')
andrews_curves(df[new_cols+['cluster']], 'cluster', ax=axes[1], alpha=0.2)
axes[1].set_title(f'PCA 순서: {new_cols}')
plt.tight_layout(); plt.show()
""",
                "interpretation": "PCA 주성분 기여도 순으로 변수를 재배열하면 분산이 큰 성분이 저주파 항에 배치되어 클러스터 분리가 시각적으로 향상된다.",
            },
            {
                "approach": "Classical MDS 직접 구현 + Swiss Roll에서 유클리드 vs Isomap 비교",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors

X_sr, color = make_swiss_roll(1000, noise=0.5, random_state=42)

def classical_mds(D, n_components=2):
    n = D.shape[0]
    H = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * H @ (D**2) @ H; B = (B+B.T)/2
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:,idx]
    pos = np.maximum(eigvals[:n_components], 0)
    return eigvecs[:,:n_components]*np.sqrt(pos), eigvals

D_euc = squareform(pdist(X_sr, 'euclidean'))
Y_euc, ev_euc = classical_mds(D_euc, 2)

nn = NearestNeighbors(n_neighbors=10).fit(X_sr)
G = nn.kneighbors_graph(mode='distance')
D_geo = shortest_path(G, method='D', directed=False)
Y_iso, ev_iso = classical_mds(D_geo, 2)

fig, axes = plt.subplots(1,3,figsize=(18,5))
axes[0].scatter(X_sr[:,0], X_sr[:,2], c=color, cmap='Spectral', s=5)
axes[0].set_title('Swiss Roll (원본)')
axes[1].scatter(Y_euc[:,0], Y_euc[:,1], c=color, cmap='Spectral', s=5)
axes[1].set_title('유클리드 MDS')
axes[2].scatter(Y_iso[:,0], Y_iso[:,1], c=color, cmap='Spectral', s=5)
axes[2].set_title('Isomap (측지선 MDS)')
plt.tight_layout(); plt.show()

def stress(D_orig, Y):
    d = squareform(pdist(Y))
    t = np.triu_indices_from(D_orig, k=1)
    return np.sqrt(np.sum((D_orig[t]-d[t])**2)/np.sum(D_orig[t]**2))
print(f"유클리드 MDS Stress: {stress(D_euc, Y_euc):.4f}")
print(f"Isomap Stress: {stress(D_geo, Y_iso):.4f}")
""",
                "interpretation": "Swiss Roll은 2차원 다양체이므로 유클리드 MDS는 구조를 보존 못 한다. 측지선 거리 기반 Isomap은 펼쳐진 구조를 복원한다.",
            },
            {
                "approach": "Coplot 구현 및 Simpson's paradox 탐색",
                "code": r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt
np.random.seed(42)

def coplot(df, x_col, y_col, z_col, n_panels=6, overlap=0.3):
    z = df[z_col].values; z_min, z_max = z.min(), z.max()
    step = (z_max - z_min) / (n_panels - (n_panels-1)*overlap)
    fig, axes = plt.subplots(2, (n_panels+1)//2, figsize=(16,8))
    axes = axes.flatten()
    for i in range(n_panels):
        lo = z_min + i*step*(1-overlap); hi = lo + step
        mask = (z>=lo)&(z<=hi); ax = axes[i]
        ax.scatter(df.loc[mask,x_col], df.loc[mask,y_col], s=15, alpha=0.6)
        if mask.sum()>2:
            c = np.polyfit(df.loc[mask,x_col], df.loc[mask,y_col], 1)
            xs = np.linspace(df[x_col].min(), df[x_col].max(), 50)
            ax.plot(xs, np.polyval(c, xs), 'r-', lw=2)
            ax.set_title(f'{z_col}:[{lo:.1f},{hi:.1f}] slope={c[0]:.2f}', fontsize=9)
    for j in range(n_panels, len(axes)): axes[j].set_visible(False)
    plt.suptitle(f'Coplot: {x_col} vs {y_col} | {z_col}'); plt.tight_layout(); plt.show()

n = 500
lstat = np.random.uniform(2,35,n)
rm = 4 + 0.1*(35-lstat) + np.random.normal(0,0.5,n)
medv = 5 + 3*rm - 0.5*lstat - 0.2*rm*(lstat>20) + np.random.normal(0,2,n)
medv = np.clip(medv, 5, 50)
df_h = pd.DataFrame({'RM':rm, 'MEDV':medv, 'LSTAT':lstat})
print(f"전체 RM-MEDV 상관: {df_h['RM'].corr(df_h['MEDV']):.3f}")
coplot(df_h, 'RM', 'MEDV', 'LSTAT', n_panels=6, overlap=0.3)
""",
                "interpretation": "Coplot으로 LSTAT 조건부에서 RM→MEDV 기울기 변화를 확인할 수 있다. 전체 상관만으로는 비선형적 조건부 관계를 포착할 수 없다.",
            },
            {
                "approach": "Trustworthiness/Continuity로 t-SNE vs MDS 정량 비교",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import trustworthiness
from scipy.spatial.distance import pdist, squareform

np.random.seed(42)
clusters = [np.random.multivariate_normal(np.random.randn(10)*5, np.eye(10)*(0.5+i*0.3), sz)
            for i, sz in enumerate([200,150,100,80,50])]
X = np.vstack(clusters); labels = np.repeat(range(5), [200,150,100,80,50])

Y_mds = MDS(n_components=2, random_state=42, normalized_stress='auto').fit_transform(X)
Y_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

def continuity_score(X_high, X_low, k=5):
    n = X_high.shape[0]
    D_h = squareform(pdist(X_high)); D_l = squareform(pdist(X_low))
    rank_h = np.argsort(np.argsort(D_h, axis=1), axis=1)
    total = 0
    for i in range(n):
        nn_low = set(np.argsort(D_l[i])[1:k+1])
        for j in nn_low:
            if rank_h[i,j] > k: total += rank_h[i,j] - k
    norm = n*k*(2*n-3*k-1)
    return 1 - 2/norm*total if norm>0 else 1.0

ks = [5,10,15,20,30,50]
t_mds = [trustworthiness(X, Y_mds, n_neighbors=k) for k in ks]
t_tsne = [trustworthiness(X, Y_tsne, n_neighbors=k) for k in ks]
c_mds = [continuity_score(X, Y_mds, k) for k in ks]
c_tsne = [continuity_score(X, Y_tsne, k) for k in ks]

D_o = squareform(pdist(X)); D_m = squareform(pdist(Y_mds)); D_t = squareform(pdist(Y_tsne))
tri = np.triu_indices_from(D_o, k=1)
r_mds = np.corrcoef(D_o[tri], D_m[tri])[0,1]
r_tsne = np.corrcoef(D_o[tri], D_t[tri])[0,1]

fig, axes = plt.subplots(2,2,figsize=(14,12))
axes[0,0].plot(ks,t_mds,'bo-',label='MDS'); axes[0,0].plot(ks,t_tsne,'rs-',label='t-SNE')
axes[0,0].set_title('Trustworthiness'); axes[0,0].legend(); axes[0,0].set_xlabel('k')
axes[0,1].plot(ks,c_mds,'bo-',label='MDS'); axes[0,1].plot(ks,c_tsne,'rs-',label='t-SNE')
axes[0,1].set_title('Continuity'); axes[0,1].legend(); axes[0,1].set_xlabel('k')
si = np.random.choice(len(tri[0]),5000,replace=False)
axes[1,0].scatter(D_o[tri][si],D_m[tri][si],s=1,alpha=0.3)
axes[1,0].set_title(f'Shepard: MDS (r={r_mds:.3f})')
axes[1,1].scatter(D_o[tri][si],D_t[tri][si],s=1,alpha=0.3)
axes[1,1].set_title(f'Shepard: t-SNE (r={r_tsne:.3f})')
plt.tight_layout(); plt.show()
""",
                "interpretation": "MDS는 전역 거리 보존(높은 Shepard r)에 우수하고, t-SNE는 지역 구조(작은 k에서 높은 Trustworthiness)에 우수하다. 목적에 맞는 방법 선택이 중요하다.",
            },
        ],
        discussion="### 시각화 선택 가이드\n\n| 목적 | 방법 |\n|------|------|\n| 변수별 패턴 | 평행좌표 |\n| 형태 비교 | 앤드류스 곡선 |\n| 전역 구조 | MDS |\n| 지역 클러스터 | t-SNE, UMAP |\n| 비선형 다양체 | Isomap |\n| 조건부 관계 | Coplot |",
        filepath=os.path.join(BASE, "ch01_01_multivariate_visualization_solution.ipynb"),
    )


# ============================================================
# Topic 02: 결측값 메커니즘과 다중 대치법
# ============================================================
def gen_topic_02():
    problem_notebook(
        chapter_num=1, section_num=2,
        title="결측값 메커니즘과 다중 대치법",
        objectives=[
            "MCAR, MAR, MNAR의 수학적 정의와 검정법을 이해한다",
            "Little's MCAR 검정을 구현하고 해석한다",
            "MICE 알고리즘의 수리적 기반과 Rubin's rule을 이해한다",
            "KNN 대치법의 거리 가중 방식을 구현하고 성능을 비교한다",
        ],
        theory_md=r"""
### 결측 메커니즘 (Rubin, 1976)

완전 데이터 $Y=(Y_{\text{obs}},Y_{\text{mis}})$, 결측 지시 행렬 $R$:

- **MCAR**: $P(R|Y_{\text{obs}},Y_{\text{mis}},\psi)=P(R|\psi)$
- **MAR**: $P(R|Y_{\text{obs}},Y_{\text{mis}},\psi)=P(R|Y_{\text{obs}},\psi)$
- **MNAR**: 위의 조건 불만족

### Little's MCAR 검정

$$d^2 = \sum_{j=1}^J n_j(\bar{y}_j-\hat{\mu}_j)^T\hat{\Sigma}_j^{-1}(\bar{y}_j-\hat{\mu}_j) \sim \chi^2\left(\sum_j p_j - p\right)$$

### MICE (Multiple Imputation by Chained Equations)

각 변수를 나머지의 조건부 모형으로 순환 대치. $m$개 대치 데이터에 Rubin's rule:

$$\bar{Q}=\frac{1}{m}\sum\hat{Q}_l, \quad T=\bar{U}+(1+\frac{1}{m})B$$

### KNN 대치

$$\hat{y}_{ij} = \frac{\sum_{l\in N_k(i)} w_{il}\cdot y_{lj}}{\sum_{l\in N_k(i)} w_{il}}, \quad w_{il}=\frac{1}{d(i,l)^2}$$
""",
        guided_code=r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import seaborn as sns

np.random.seed(42)
n = 500
X1 = np.random.normal(50, 10, n)
X2 = 0.6*X1 + np.random.normal(0, 8, n)
X3 = np.random.normal(30, 5, n)
df_full = pd.DataFrame({'X1':X1, 'X2':X2, 'X3':X3})

# MCAR
df_mcar = df_full.copy()
df_mcar[np.random.random((n,3))<0.2] = np.nan

# MAR (X1 클수록 X2 결측)
df_mar = df_full.copy()
prob = 1/(1+np.exp(-(X1-55)/5))
df_mar.loc[np.random.random(n)<prob, 'X2'] = np.nan

# 결측 패턴 시각화
fig, axes = plt.subplots(1,2,figsize=(14,5))
sns.heatmap(df_mcar.isnull(), cbar=False, ax=axes[0], yticklabels=False)
axes[0].set_title('MCAR'); sns.heatmap(df_mar.isnull(), cbar=False, ax=axes[1], yticklabels=False)
axes[1].set_title('MAR'); plt.tight_layout(); plt.show()

# MICE / KNN 대치
df_mice = pd.DataFrame(IterativeImputer(max_iter=20, random_state=42).fit_transform(df_mcar), columns=df_mcar.columns)
df_knn = pd.DataFrame(KNNImputer(n_neighbors=5, weights='distance').fit_transform(df_mcar), columns=df_mcar.columns)
""",
        exercises=[
            {
                "difficulty": "★",
                "description": "5변수 데이터에 MCAR/MAR/MNAR 결측 20%씩 생성하고, 결측 전후 분포/상관행렬 변화를 시각화하여 어떤 메커니즘이 편향을 유발하는지 분석하라.",
                "skeleton": "# TODO: 5변수 상관 데이터\n# TODO: 3종 결측 생성\n# TODO: 분포/상관행렬 비교\n",
            },
            {
                "difficulty": "★★",
                "description": "Little's MCAR 검정을 직접 구현하라 (외부 패키지 금지). EM으로 전체 mu/Sigma 추정, 카이제곱 통계량 계산. MCAR vs MAR 데이터에서 검정력 비교.",
                "hint": "EM의 E-step에서 조건부 기대값/공분산을 갱신한다.",
                "skeleton": "def littles_mcar_test(df):\n    # TODO\n    pass\n",
            },
            {
                "difficulty": "★★",
                "description": "MICE를 직접 구현하라 (sklearn 금지). Bayesian linear regression 사후 예측 분포에서 샘플링. m=5 대치 데이터에 Rubin's rule로 결합 추정치와 CI를 구하라.",
                "skeleton": "def mice_impute(df, m=5, max_iter=20):\n    # TODO\n    pass\n\ndef rubins_rule(estimates, variances):\n    # TODO\n    pass\n",
            },
            {
                "difficulty": "★★★",
                "description": "MNAR 민감도 분석 수행.\n1) Y=b0+b1*X1+b2*X2+e 적합\n2) MNAR로 Y 30% 결측\n3) 틸팅 모형 logit(P(R=0))=a0+a1*Y에서 a1을 변화시키며 b1 편향 분석\n4) 추정치 변화 그래프와 해석",
                "skeleton": "# TODO: 완전 데이터 + 참 회귀계수\n# TODO: 다양한 a1에서 대치 후 분석\n# TODO: 편향 그래프\n",
            },
        ],
        references=[
            "Rubin, D.B. (1976). Inference and Missing Data. Biometrika.",
            "Little, R.J.A. (1988). A Test of MCAR. JASA.",
            "van Buuren, S. (2018). Flexible Imputation of Missing Data. CRC Press.",
        ],
        filepath=os.path.join(BASE, "ch01_02_missing_data.ipynb"),
    )

    solution_notebook(
        chapter_num=1, section_num=2,
        title="결측값 메커니즘과 다중 대치법",
        solutions=[
            {
                "approach": "MCAR/MAR/MNAR 결측 생성 및 편향 분석",
                "code": r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
np.random.seed(42)
n = 1000
mean = [50,30,20,40,60]
cov_m = np.array([[100,30,-10,20,15],[30,64,5,-10,8],[-10,5,25,12,-5],[20,-10,12,49,10],[15,8,-5,10,81]])
data = np.random.multivariate_normal(mean, cov_m, n)
cols = ['V1','V2','V3','V4','V5']
df_full = pd.DataFrame(data, columns=cols)

df_mcar = df_full.copy()
for c in cols: df_mcar.loc[np.random.random(n)<0.2, c] = np.nan

df_mar = df_full.copy()
df_mar.loc[np.random.random(n)<1/(1+np.exp(-(data[:,0]-55)/8)), 'V2'] = np.nan

df_mnar = df_full.copy()
df_mnar.loc[np.random.random(n)<1/(1+np.exp(-(data[:,1]-35)/5)), 'V2'] = np.nan

# 상관행렬 비교
fig, axes = plt.subplots(1,4,figsize=(20,4))
for ax, t, d in zip(axes, ['원본','MCAR','MAR','MNAR'], [df_full,df_mcar,df_mar,df_mnar]):
    sns.heatmap(d.corr(), annot=True, fmt='.2f', ax=ax, vmin=-1, vmax=1, cmap='RdBu_r'); ax.set_title(t)
plt.tight_layout(); plt.show()

print("=== 평균 편향 ===")
for name, d in [('MCAR',df_mcar),('MAR',df_mar),('MNAR',df_mnar)]:
    bias = d.mean()-df_full.mean()
    print(f"{name}: {dict(zip(cols, bias.round(3)))}")
""",
                "interpretation": "MCAR은 편향 무시 가능. MAR은 조건 변수에 의존하여 일부 변수 편향. MNAR은 결측 변수 자체의 분포가 왜곡되어 가장 심각한 편향 발생.",
            },
            {
                "approach": "Little's MCAR 검정 EM 기반 구현",
                "code": r"""import numpy as np, pandas as pd
from scipy import stats

def littles_mcar_test(df):
    df = df.copy(); p = df.shape[1]; n = df.shape[0]
    R = (~df.isnull()).astype(int)
    patterns = R.drop_duplicates()
    pattern_keys = [tuple(row) for _, row in patterns.iterrows()]

    complete = df.dropna()
    mu = complete.mean().values if len(complete)>5 else df.mean().values
    sigma = (complete.cov().values if len(complete)>5 else df.cov().values)
    sigma = np.nan_to_num(sigma) + np.eye(p)*0.01

    for _ in range(50):
        mu_old = mu.copy()
        T1, T2 = np.zeros(p), np.zeros((p,p))
        for i in range(n):
            row = df.values[i]
            obs = np.where(~np.isnan(row))[0]; mis = np.where(np.isnan(row))[0]
            if len(mis)==0:
                T1 += row; T2 += np.outer(row,row); continue
            if len(obs)==0:
                T1 += mu; T2 += np.outer(mu,mu)+sigma; continue
            sig_oo_inv = np.linalg.pinv(sigma[np.ix_(obs,obs)])
            cm = mu[mis] + sigma[np.ix_(mis,obs)]@sig_oo_inv@(row[obs]-mu[obs])
            cc = sigma[np.ix_(mis,mis)] - sigma[np.ix_(mis,obs)]@sig_oo_inv@sigma[np.ix_(obs,mis)]
            xf = row.copy(); xf[mis] = cm; T1 += xf
            outer = np.outer(xf,xf)
            for a,ma in enumerate(mis):
                for b,mb in enumerate(mis): outer[ma,mb] += cc[a,b]
            T2 += outer
        mu = T1/n; sigma = T2/n - np.outer(mu,mu)
        if np.max(np.abs(mu-mu_old))<1e-6: break

    d2, df_test = 0, 0
    for pat in pattern_keys:
        obs_v = np.where(np.array(pat)==1)[0]
        if len(obs_v)==0: continue
        mask = np.all(R.values==np.array(pat), axis=1); nj = mask.sum()
        if nj<2: continue
        mean_j = np.nanmean(df.values[mask][:,obs_v], axis=0)
        diff = mean_j - mu[obs_v]
        d2 += nj * diff @ np.linalg.pinv(sigma[np.ix_(obs_v,obs_v)]) @ diff
        df_test += len(obs_v)
    df_test = max(df_test - p, 1)
    return {'statistic': d2, 'df': df_test, 'p_value': 1-stats.chi2.cdf(d2, df_test)}

np.random.seed(42)
data = np.random.multivariate_normal([50,30,20], [[100,30,-10],[30,64,5],[-10,5,25]], 500)
df_f = pd.DataFrame(data, columns=['A','B','C'])
df_mc = df_f.copy()
for c in ['A','B','C']: df_mc.loc[np.random.random(500)<0.2, c] = np.nan
df_ma = df_f.copy()
df_ma.loc[np.random.random(500)<1/(1+np.exp(-(data[:,0]-55)/5)), 'B'] = np.nan

r1 = littles_mcar_test(df_mc); r2 = littles_mcar_test(df_ma)
print(f"MCAR data: stat={r1['statistic']:.3f}, p={r1['p_value']:.4f}")
print(f"MAR data:  stat={r2['statistic']:.3f}, p={r2['p_value']:.4f}")
""",
                "interpretation": "MCAR 데이터에서는 p-value가 높아 MCAR 기각 못 함. MAR 데이터에서는 패턴별 평균 차이가 유의하여 MCAR 기각. Little's test는 MAR/MNAR 구분은 불가.",
            },
            {
                "approach": "Bayesian MICE 직접 구현 + Rubin's rule",
                "code": r"""import numpy as np, pandas as pd
np.random.seed(42)
n=500; X1=np.random.normal(0,1,n); X2=np.random.normal(0,1,n)
Y = 3+2*X1-1.5*X2+np.random.normal(0,1,n)
df_full = pd.DataFrame({'X1':X1,'X2':X2,'Y':Y})
df_miss = df_full.copy()
df_miss.loc[np.random.random(n)<1/(1+np.exp(-(X1-0.5)))*0.4, 'Y'] = np.nan

def mice_impute(df, m=5, max_iter=20):
    cols = df.columns.tolist(); imputed = []
    for imp in range(m):
        rng = np.random.RandomState(42+imp); di = df.copy()
        for c in cols:
            mask = di[c].isnull()
            if mask.any(): di.loc[mask,c] = di[c].mean()
        for _ in range(max_iter):
            for tc in cols:
                miss = df[tc].isnull()
                if not miss.any(): continue
                preds = [c for c in cols if c!=tc]; obs = ~df[tc].isnull()
                Xo = di.loc[obs, preds].values; yo = df.loc[obs, tc].values
                Xd = np.column_stack([np.ones(obs.sum()), Xo])
                XtXi = np.linalg.inv(Xd.T@Xd+np.eye(Xd.shape[1])*1e-6)
                bh = XtXi@Xd.T@yo; res = yo-Xd@bh; s2 = np.sum(res**2)/max(obs.sum()-Xd.shape[1],1)
                dof = max(obs.sum()-Xd.shape[1],1)
                s2d = dof*s2/rng.chisquare(dof)
                bd = rng.multivariate_normal(bh, s2d*XtXi)
                Xm = np.column_stack([np.ones(miss.sum()), di.loc[miss,preds].values])
                di.loc[miss,tc] = Xm@bd + rng.normal(0,np.sqrt(s2d),miss.sum())
        imputed.append(di.copy())
    return imputed

def rubins_rule(est, var):
    m=len(est); Qb=np.mean(est); Ub=np.mean(var); B=np.var(est,ddof=1)
    T=Ub+(1+1/m)*B; return {'estimate':Qb, 'se':np.sqrt(T), 'within':Ub, 'between':B}

dsets = mice_impute(df_miss, m=5)
b1s, v1s = [], []
for di in dsets:
    Xm = np.column_stack([np.ones(n), di['X1'].values, di['X2'].values])
    XtXi = np.linalg.inv(Xm.T@Xm); b=XtXi@Xm.T@di['Y'].values
    s2=np.sum((di['Y'].values-Xm@b)**2)/(n-3)
    b1s.append(b[1]); v1s.append(s2*XtXi[1,1])
r = rubins_rule(b1s, v1s)
print(f"결합 beta1: {r['estimate']:.4f} (참값: 2.0)")
print(f"95% CI: [{r['estimate']-1.96*r['se']:.4f}, {r['estimate']+1.96*r['se']:.4f}]")
""",
                "interpretation": "Bayesian MICE는 각 대치에 불확실성을 반영한다. Rubin's rule은 within/between variance를 결합하여 올바른 추론을 보장한다.",
            },
            {
                "approach": "MNAR 민감도 분석: 틸팅 매개변수에 따른 편향",
                "code": r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
np.random.seed(42)
n=1000; X1=np.random.normal(0,1,n); X2=np.random.normal(0,1,n)
Y = 3+2.0*X1-1.5*X2+np.random.normal(0,1,n)
Xm = np.column_stack([np.ones(n),X1,X2])
bt = np.linalg.lstsq(Xm, Y, rcond=None)[0]

alphas = [0,0.3,0.6,1.0,1.5,2.0]
res = {'a':[],'cc':[],'mice':[],'bias_cc':[],'bias_mice':[]}
for a1 in alphas:
    prob = 1/(1+np.exp(-(-1+a1*Y)))
    df_m = pd.DataFrame({'X1':X1,'X2':X2,'Y':Y})
    df_m.loc[np.random.random(n)<prob,'Y'] = np.nan
    cc = df_m.dropna(); Xc = np.column_stack([np.ones(len(cc)),cc['X1'],cc['X2']])
    bc = np.linalg.lstsq(Xc, cc['Y'].values, rcond=None)[0]
    di = pd.DataFrame(IterativeImputer(max_iter=20,random_state=42).fit_transform(df_m), columns=df_m.columns)
    Xi = np.column_stack([np.ones(n),di['X1'],di['X2']])
    bi = np.linalg.lstsq(Xi, di['Y'].values, rcond=None)[0]
    res['a'].append(a1); res['cc'].append(bc[1]); res['mice'].append(bi[1])
    res['bias_cc'].append(bc[1]-bt[1]); res['bias_mice'].append(bi[1]-bt[1])

fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].plot(res['a'],res['cc'],'bo-',lw=2,label='CC')
axes[0].plot(res['a'],res['mice'],'rs-',lw=2,label='MICE')
axes[0].axhline(bt[1],color='green',ls='--',lw=2,label=f'참값({bt[1]:.2f})')
axes[0].set_xlabel('α₁'); axes[0].set_ylabel('β̂₁'); axes[0].legend(); axes[0].set_title('추정치')
axes[1].plot(res['a'],res['bias_cc'],'bo-',lw=2,label='CC')
axes[1].plot(res['a'],res['bias_mice'],'rs-',lw=2,label='MICE')
axes[1].axhline(0,color='green',ls='--'); axes[1].set_xlabel('α₁'); axes[1].set_ylabel('편향')
axes[1].set_title('편향'); axes[1].legend()
plt.tight_layout(); plt.show()
""",
                "interpretation": "α₁=0(MCAR)에서는 편향 없음. α₁ 증가(MNAR 강도)에 따라 CC와 MICE 모두 편향 증가. MNAR에서는 결측 메커니즘 민감도 분석이 필수적이다.",
            },
        ],
        discussion="### 결측 처리 전략\n\n1. 결측률 < 5%: CC 수용 가능\n2. 5-30%, MCAR/MAR: MICE 권장\n3. > 30%: 민감도 분석 필수\n4. MNAR 의심: Selection model 또는 Pattern-Mixture model",
        filepath=os.path.join(BASE, "ch01_02_missing_data_solution.ipynb"),
    )


# ============================================================
# Topic 03: 이상치 탐지 - 통계적 방법
# ============================================================
def gen_topic_03():
    problem_notebook(
        chapter_num=1, section_num=3,
        title="이상치 탐지 - 통계적 방법",
        objectives=[
            "Mahalanobis 거리의 수학적 원리와 χ² 분포와의 관계를 이해한다",
            "Grubbs 검정의 가설 구조와 임계값 유도를 수행한다",
            "Generalized ESD 검정으로 복수 이상치를 탐지한다",
            "로버스트 추정량(MCD)을 활용한 다변량 이상치 탐지를 구현한다",
        ],
        theory_md=r"""
### 1. Mahalanobis 거리

$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}, \quad D_M^2 \sim \chi^2_p$$

### 2. Grubbs 검정

$$G = \frac{\max_i|x_i-\bar{x}|}{s}, \quad G_{\text{crit}} = \frac{n-1}{\sqrt{n}}\sqrt{\frac{t_{\alpha/(2n),n-2}^2}{n-2+t_{\alpha/(2n),n-2}^2}}$$

### 3. Generalized ESD (Rosner, 1983)

$$R_i = \frac{\max_j|x_j-\bar{x}_i|}{s_i}, \quad \lambda_i = \frac{(n-i)t_{p,n-i-1}}{\sqrt{(n-i-1+t_{p,n-i-1}^2)(n-i+1)}}$$

### 4. MCD (Minimum Covariance Determinant)

$$(\hat{\mu}_{\text{MCD}},\hat{\Sigma}_{\text{MCD}}) = \arg\min_{|J|=h}\det(\mathbf{S}_J)$$

붕괴점 ~50%. 이상치에 의한 평균/공분산 왜곡 방지.
""",
        guided_code=r"""import numpy as np, matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import mahalanobis

np.random.seed(42)
X_n = np.random.multivariate_normal([0,0], [[1,0.7],[0.7,1]], 200)
X_o = np.random.multivariate_normal([4,4], [[0.3,0],[0,0.3]], 10)
X = np.vstack([X_n, X_o]); labels = np.array([0]*200+[1]*10)

mu = X.mean(axis=0); cov_inv = np.linalg.inv(np.cov(X.T))
D = np.array([mahalanobis(x, mu, cov_inv) for x in X])
thresh = np.sqrt(stats.chi2.ppf(0.975, 2))
outliers = D > thresh

fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].scatter(X[~outliers,0], X[~outliers,1], c='blue', s=20, alpha=0.6, label='정상')
axes[0].scatter(X[outliers,0], X[outliers,1], c='red', s=40, marker='x', label='이상치')
axes[0].set_title('Mahalanobis 이상치 탐지'); axes[0].legend()
sorted_d2 = np.sort(D**2)
theo = stats.chi2.ppf(np.arange(1,len(sorted_d2)+1)/(len(sorted_d2)+1), 2)
axes[1].scatter(theo, sorted_d2, s=10, alpha=0.5)
axes[1].plot([0,max(theo)],[0,max(theo)],'r--')
axes[1].set_title('χ² QQ-Plot'); plt.tight_layout(); plt.show()
""",
        exercises=[
            {
                "difficulty": "★",
                "description": "5차원 데이터(n=500, 이상치 20개)에 Mahalanobis 이상치 탐지. χ² QQ-plot과 ROC 곡선으로 성능 평가.",
                "skeleton": "# TODO: 5차원 데이터 생성\n# TODO: Mahalanobis + QQ-plot + ROC\n",
            },
            {
                "difficulty": "★★",
                "description": "Generalized ESD 검정 직접 구현. 정상+오염 혼합 데이터에서 검정 수행.",
                "hint": "R_i > λ_i인 최대 i가 이상치 수.",
                "skeleton": "def generalized_esd(data, max_outliers=15, alpha=0.05):\n    # TODO\n    pass\n",
            },
            {
                "difficulty": "★★",
                "description": "MCD를 이용한 로버스트 Mahalanobis 구현. 이상치 비율 0~40%에서 표본공분산 vs MCD F1 비교.",
                "skeleton": "from sklearn.covariance import MinCovDet\n# TODO: 비율별 실험\n",
            },
            {
                "difficulty": "★★★",
                "description": "마스킹/스와핑 효과 시뮬레이션. 클러스터형 이상치에서 Grubbs 실패 예시. 이상치 비율별 Classical vs MCD Precision/Recall. 붕괴점 이론 논의.",
                "skeleton": "# TODO: 마스킹 효과\n# TODO: 비율별 성능\n# TODO: 붕괴점 분석\n",
            },
        ],
        references=[
            "Rousseeuw & Van Driessen (1999). Fast MCD. Technometrics.",
            "Grubbs (1950). Sample Criteria for Outlying Observations.",
            "Rosner (1983). Generalized ESD. Technometrics.",
        ],
        filepath=os.path.join(BASE, "ch01_03_outlier_statistical.ipynb"),
    )

    solution_notebook(
        chapter_num=1, section_num=3,
        title="이상치 탐지 - 통계적 방법",
        solutions=[
            {
                "approach": "5차원 Mahalanobis + QQ + ROC",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_curve, auc

np.random.seed(42); p=5
cov_n = np.eye(p);
for i in range(p):
    for j in range(p):
        if i!=j: cov_n[i,j]=0.5**abs(i-j)
X_n = np.random.multivariate_normal(np.zeros(p), cov_n, 500)
X_o = np.random.multivariate_normal(np.ones(p)*4, np.eye(p)*0.2, 20)
X = np.vstack([X_n, X_o]); y = np.array([0]*500+[1]*20)

mu = X.mean(0); Si = np.linalg.inv(np.cov(X.T))
D2 = np.array([mahalanobis(x,mu,Si)**2 for x in X])

fig, axes = plt.subplots(1,2,figsize=(14,5))
sd2 = np.sort(D2); theo = stats.chi2.ppf(np.arange(1,len(sd2)+1)/(len(sd2)+1), p)
axes[0].scatter(theo, sd2, s=8, alpha=0.5); axes[0].plot([0,max(theo)],[0,max(theo)],'r--')
axes[0].set_title('χ²(5) QQ-Plot')
fpr,tpr,_ = roc_curve(y, D2)
axes[1].plot(fpr,tpr,'b-',lw=2,label=f'AUC={auc(fpr,tpr):.3f}')
axes[1].plot([0,1],[0,1],'r--'); axes[1].set_title('ROC'); axes[1].legend()
plt.tight_layout(); plt.show()

thr = stats.chi2.ppf(0.975, p); det = D2>thr
tp=((det)&(y==1)).sum(); fp=((det)&(y==0)).sum()
print(f"Precision={tp/(tp+fp):.3f}, Recall={tp/(tp+(~det&(y==1)).sum()):.3f}")
""",
                "interpretation": "QQ-plot 상위 꼬리 이탈 점이 이상치 후보. AUC가 높을수록 Mahalanobis의 분리 능력이 우수함을 나타낸다.",
            },
            {
                "approach": "Generalized ESD 직접 구현",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from scipy import stats

def generalized_esd(data, max_outliers=15, alpha=0.05):
    data = np.array(data, dtype=float); n = len(data)
    remaining = data.copy(); rem_idx = np.arange(n)
    out_idx, Rs, lams = [], [], []
    for i in range(1, max_outliers+1):
        ni = len(remaining)
        if ni < 3: break
        m, s = np.mean(remaining), np.std(remaining, ddof=1)
        if s == 0: break
        devs = np.abs(remaining - m); mi = np.argmax(devs)
        Ri = devs[mi] / s
        p = 1 - alpha / (2*ni)
        tp = stats.t.ppf(p, ni-2)
        lam_i = (ni-1)*tp / np.sqrt((ni-2+tp**2)*ni)
        Rs.append(Ri); lams.append(lam_i); out_idx.append(rem_idx[mi])
        remaining = np.delete(remaining, mi); rem_idx = np.delete(rem_idx, mi)
    n_out = 0
    for i in range(len(Rs)):
        if Rs[i] > lams[i]: n_out = i+1
    return {'n_outliers': n_out, 'indices': out_idx[:n_out], 'R': Rs, 'lambda': lams}

np.random.seed(42)
data = np.concatenate([np.random.normal(50,5,200), np.array([90,95,85,10,5,88])])
r = generalized_esd(data, 15)
print(f"탐지: {r['n_outliers']}개, 값: {data[r['indices']]}")

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(range(1,len(r['R'])+1), r['R'], alpha=0.7, label='R_i')
ax.plot(range(1,len(r['lambda'])+1), r['lambda'], 'r-o', lw=2, label='λ_i')
ax.axvline(r['n_outliers']+0.5, color='green', ls='--'); ax.legend()
ax.set_title('Generalized ESD'); plt.tight_layout(); plt.show()
""",
                "interpretation": "ESD는 순차적으로 극단값을 제거하며 검정하여 마스킹 효과를 부분적으로 완화한다.",
            },
            {
                "approach": "이상치 비율별 Classical vs MCD 성능 비교",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.spatial.distance import mahalanobis
from scipy import stats
from sklearn.metrics import f1_score

np.random.seed(42)
def sim(rate, n=500, p=3, trials=20):
    f1c, f1m = [], []
    for t in range(trials):
        rng = np.random.RandomState(t)
        no = int(n*rate); nn = n-no
        cv = np.eye(p); cv[0,1]=cv[1,0]=0.5
        Xn = rng.multivariate_normal(np.zeros(p), cv, nn)
        Xo = rng.multivariate_normal(np.ones(p)*5, np.eye(p)*0.3, no)
        X = np.vstack([Xn,Xo]); y = np.array([0]*nn+[1]*no)
        Si = np.linalg.pinv(np.cov(X.T)); mu = X.mean(0)
        D2c = np.array([mahalanobis(x,mu,Si)**2 for x in X])
        pc = (D2c>stats.chi2.ppf(0.975,p)).astype(int)
        try:
            mcd = MinCovDet(random_state=t).fit(X)
            D2m = mcd.mahalanobis(X)
        except: D2m = np.zeros(n)
        pm = (D2m>stats.chi2.ppf(0.975,p)).astype(int)
        f1c.append(f1_score(y,pc,zero_division=0)); f1m.append(f1_score(y,pm,zero_division=0))
    return np.mean(f1c), np.mean(f1m)

rates = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]
fc, fm = zip(*[sim(r) for r in rates])
fig, ax = plt.subplots(figsize=(10,5))
ax.plot([r*100 for r in rates], fc, 'bo-', lw=2, label='Classical')
ax.plot([r*100 for r in rates], fm, 'rs-', lw=2, label='MCD')
ax.set_xlabel('이상치 비율 (%)'); ax.set_ylabel('F1'); ax.legend()
ax.set_title('이상치 비율별 F1 비교'); plt.tight_layout(); plt.show()
""",
                "interpretation": "이상치 비율 증가 시 Classical Mahalanobis는 마스킹으로 F1 급락. MCD는 붕괴점(~50%)까지 안정적 성능 유지.",
            },
            {
                "approach": "마스킹 효과 시각화",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy import stats
from sklearn.covariance import MinCovDet

np.random.seed(42)
X_n = np.random.multivariate_normal([0,0], [[1,0.5],[0.5,1]], 100)
X_o = np.random.multivariate_normal([6,6], [[0.2,0],[0,0.2]], 20)
X = np.vstack([X_n, X_o]); y = np.array([0]*100+[1]*20)

mu = X.mean(0); Si = np.linalg.inv(np.cov(X.T))
D2c = np.array([mahalanobis(x,mu,Si)**2 for x in X])
D2m = MinCovDet(random_state=42).fit(X).mahalanobis(X)
thr = stats.chi2.ppf(0.975,2)

fig, axes = plt.subplots(1,2,figsize=(14,5))
for ax, D2, t in [(axes[0],D2c,'Classical'),(axes[1],D2m,'MCD')]:
    det = D2>thr
    ax.scatter(X[~det,0],X[~det,1],c='blue',s=20,alpha=0.5,label='정상')
    ax.scatter(X[det,0],X[det,1],c='red',s=40,marker='x',label='이상치')
    tp=((det)&(y==1)).sum(); fp=((det)&(y==0)).sum(); fn=((~det)&(y==1)).sum()
    p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
    ax.set_title(f'{t}: P={p:.2f}, R={r:.2f}'); ax.legend()
plt.suptitle('마스킹 효과: 클러스터형 이상치'); plt.tight_layout(); plt.show()
""",
                "interpretation": "Classical 방법은 이상치 클러스터가 평균/공분산을 끌어당겨 마스킹 발생. MCD는 로버스트 추정으로 이를 방지한다.",
            },
        ],
        discussion="### 통계적 이상치 탐지 지침\n\n1. 일변량: Grubbs → ESD\n2. 다변량: MCD Mahalanobis 기본\n3. 분포 가정 위반: 비모수 또는 ML 방법\n4. 비율 > 50%: 붕괴점 초과, 문제 재정의 필요",
        filepath=os.path.join(BASE, "ch01_03_outlier_statistical_solution.ipynb"),
    )


# ============================================================
# Topic 04: 이상치 탐지 - 기계학습 기반
# ============================================================
def gen_topic_04():
    problem_notebook(
        chapter_num=1, section_num=4,
        title="이상치 탐지 - 기계학습 기반",
        objectives=[
            "Isolation Forest의 분리 깊이와 이상 점수의 관계를 이해한다",
            "LOF의 지역 밀도 비율 계산 원리를 파악한다",
            "One-Class SVM의 커널 기반 경계 학습을 이해한다",
            "앙상블 방법으로 이상치 탐지의 안정성을 높인다",
        ],
        theory_md=r"""
### 1. Isolation Forest (Liu et al., 2008)

이상 점수: $s(x,n) = 2^{-E[h(x)]/c(n)}$

$c(n) = 2H(n-1)-2(n-1)/n$, $H(i)=\ln(i)+\gamma$

### 2. LOF (Breunig et al., 2000)

$$\text{LOF}_k(x) = \frac{1}{|N_k(x)|}\sum_{o\in N_k(x)}\frac{\text{lrd}_k(o)}{\text{lrd}_k(x)}$$

### 3. One-Class SVM

$$\min_{w,\xi,\rho}\frac{1}{2}\|w\|^2+\frac{1}{\nu n}\sum\xi_i-\rho$$
$$\text{s.t.}\; w\cdot\Phi(x_i)\geq\rho-\xi_i$$
""",
        guided_code=r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X_n = np.random.multivariate_normal([0,0], [[1,0.5],[0.5,1]], 300)
X_o = np.vstack([np.random.uniform(-5,5,(15,2)), np.random.multivariate_normal([4,-3],[[0.1,0],[0,0.1]],5)])
X = np.vstack([X_n,X_o]); y = np.array([1]*300+[-1]*20)
Xs = StandardScaler().fit_transform(X)

models = [('IF', IsolationForest(n_estimators=100, contamination=0.06, random_state=42)),
          ('LOF', LocalOutlierFactor(n_neighbors=20, contamination=0.06)),
          ('OCSVM', OneClassSVM(kernel='rbf', gamma='scale', nu=0.06))]

fig, axes = plt.subplots(1,3,figsize=(18,5))
for ax, (name, m) in zip(axes, models):
    pred = m.fit_predict(Xs)
    ax.scatter(X[pred==1,0],X[pred==1,1],c='blue',s=15,alpha=0.5,label='정상')
    ax.scatter(X[pred==-1,0],X[pred==-1,1],c='red',s=40,marker='x',label='이상치')
    ax.set_title(name); ax.legend()
plt.tight_layout(); plt.show()
""",
        exercises=[
            {
                "difficulty": "★",
                "description": "IF/LOF/OCSVM의 하이퍼파라미터 민감도를 분석하라. 각 설정에서 F1을 계산하고 시각화.",
                "skeleton": "# TODO: 그리드 탐색 + F1 비교\n",
            },
            {
                "difficulty": "★★",
                "description": "Isolation Forest를 직접 구현하라 (sklearn 금지). iTree/iForest 클래스, 이상점수 계산, sklearn과 AUC 비교.",
                "hint": "서브샘플 ψ=256, 깊이 제한=ceil(log2(ψ)).",
                "skeleton": "class IsolationTree:\n    pass\nclass IsolationForestCustom:\n    pass\n",
            },
            {
                "difficulty": "★★",
                "description": "LOF 직접 구현. 균일분포에서 LOF≈1 확인. 밀도 변동 데이터(10:1)에서 LOF 장점 시연.",
                "skeleton": "def compute_lof(X, k=20):\n    # TODO\n    pass\n",
            },
            {
                "difficulty": "★★★",
                "description": "이상치 탐지 앙상블 설계.\n1) IF/LOF/OCSVM 점수 정규화\n2) 평균/최대/가중평균/다수결 결합\n3) 5가지 시나리오에서 단일 vs 앙상블 강건성 평가",
                "skeleton": "# TODO: 앙상블 프레임워크\n# TODO: 시나리오별 실험\n",
            },
        ],
        references=[
            "Liu, Ting & Zhou (2008). Isolation Forest. ICDM.",
            "Breunig et al. (2000). LOF. ACM SIGMOD.",
            "Scholkopf et al. (2001). Support of High-Dim Distribution.",
        ],
        filepath=os.path.join(BASE, "ch01_04_outlier_ml.ipynb"),
    )

    solution_notebook(
        chapter_num=1, section_num=4,
        title="이상치 탐지 - 기계학습 기반",
        solutions=[
            {
                "approach": "하이퍼파라미터 민감도 분석",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X_n = np.random.multivariate_normal([0,0],[[1,0.5],[0.5,1]],400)
X_o = np.vstack([np.random.uniform(-6,6,(20,2)),np.random.multivariate_normal([5,-4],0.1*np.eye(2),5)])
X = np.vstack([X_n,X_o]); y = np.array([1]*400+[-1]*25)
Xs = StandardScaler().fit_transform(X); c = 25/len(X)

ne = [50,100,200,500]; f1_if = [f1_score(y,IsolationForest(n_estimators=v,contamination=c,random_state=42).fit_predict(Xs),pos_label=-1) for v in ne]
nn = [5,10,20,50,100]; f1_lof = [f1_score(y,LocalOutlierFactor(n_neighbors=v,contamination=c).fit_predict(Xs),pos_label=-1) for v in nn]
gs = ['scale',0.01,0.1,1.0]; f1_svm = [f1_score(y,OneClassSVM(kernel='rbf',gamma=v,nu=c).fit_predict(Xs),pos_label=-1) for v in gs]

fig,axes = plt.subplots(1,3,figsize=(16,4))
axes[0].bar(range(len(ne)),f1_if); axes[0].set_xticks(range(len(ne))); axes[0].set_xticklabels(ne); axes[0].set_title('IF')
axes[1].bar(range(len(nn)),f1_lof); axes[1].set_xticks(range(len(nn))); axes[1].set_xticklabels(nn); axes[1].set_title('LOF')
axes[2].bar(range(len(gs)),f1_svm); axes[2].set_xticks(range(len(gs))); axes[2].set_xticklabels([str(g) for g in gs]); axes[2].set_title('OCSVM')
plt.tight_layout(); plt.show()
""",
                "interpretation": "IF는 n_estimators에 안정적(100+), LOF는 k에 중간 민감, OCSVM은 gamma에 매우 민감하여 교차검증 필수.",
            },
            {
                "approach": "Isolation Forest 직접 구현",
                "code": r"""import numpy as np
from sklearn.metrics import roc_auc_score

np.random.seed(42)

class ITreeNode:
    def __init__(self, left=None, right=None, feat=None, val=None, size=0, leaf=False):
        self.left, self.right, self.feat, self.val, self.size, self.leaf = left, right, feat, val, size, leaf

def _c(n):
    if n<=1: return 0
    return 2*(np.log(n-1)+0.5772)-2*(n-1)/n

def build_itree(X, hlim, h=0):
    n,p = X.shape
    if h>=hlim or n<=1: return ITreeNode(size=n, leaf=True)
    f = np.random.randint(0,p); xmin,xmax = X[:,f].min(), X[:,f].max()
    if xmin==xmax: return ITreeNode(size=n, leaf=True)
    sv = np.random.uniform(xmin,xmax); lm = X[:,f]<sv
    return ITreeNode(left=build_itree(X[lm],hlim,h+1), right=build_itree(X[~lm],hlim,h+1), feat=f, val=sv, size=n)

def path_len(x, node, cl=0):
    if node.leaf: return cl+_c(node.size)
    return path_len(x, node.left if x[node.feat]<node.val else node.right, cl+1)

class IForestCustom:
    def __init__(self, n_trees=100, psi=256):
        self.n_trees, self.psi = n_trees, psi
    def fit(self, X):
        n = X.shape[0]; psi = min(self.psi, n); hl = int(np.ceil(np.log2(psi)))
        self.trees = [build_itree(X[np.random.choice(n,psi,replace=False)], hl) for _ in range(self.n_trees)]
        self._cn = _c(psi); return self
    def score_samples(self, X):
        return np.array([2**(-np.mean([path_len(x,t) for t in self.trees])/self._cn) for x in X])

X_n = np.random.multivariate_normal([0,0,0], np.eye(3), 400)
X_o = np.random.uniform(-6,6,(20,3)); X=np.vstack([X_n,X_o]); y=np.array([0]*400+[1]*20)

ifc = IForestCustom(100,256).fit(X); sc = ifc.score_samples(X)
from sklearn.ensemble import IsolationForest as IF
ss = -IF(n_estimators=100,max_samples=256,random_state=42).fit(X).score_samples(X)
print(f"Custom AUC: {roc_auc_score(y,sc):.4f}, sklearn AUC: {roc_auc_score(y,ss):.4f}")
""",
                "interpretation": "직접 구현이 sklearn과 유사한 AUC 달성. 이상치는 정상보다 짧은 경로=높은 anomaly score.",
            },
            {
                "approach": "LOF 직접 구현 + 밀도 변동 분석",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(42)

def compute_lof(X, k=20):
    n=X.shape[0]; D=cdist(X,X); np.fill_diagonal(D,np.inf)
    kd = np.sort(D,axis=1)[:,k-1]; kn = np.argsort(D,axis=1)[:,:k]
    rd = np.zeros((n,k))
    for i in range(n):
        for j in range(k): rd[i,j]=max(kd[kn[i,j]], D[i,kn[i,j]])
    lrd = 1.0/(np.mean(rd,axis=1)+1e-10)
    return np.array([np.mean(lrd[kn[i]])/max(lrd[i],1e-10) for i in range(n)])

X_d = np.random.multivariate_normal([0,0],0.1*np.eye(2),200)
X_s = np.random.multivariate_normal([5,5],np.eye(2),50)
X_o = np.array([[3,3],[8,0],[-3,4]]); X=np.vstack([X_d,X_s,X_o]); y=np.array([0]*250+[1]*3)

lof = compute_lof(X,20)
fig,ax = plt.subplots(figsize=(8,6))
sc = ax.scatter(X[:,0],X[:,1],c=lof,cmap='Reds',s=20,vmin=0.8,vmax=3); plt.colorbar(sc,label='LOF')
ax.scatter(X_o[:,0],X_o[:,1],c='blue',marker='*',s=200,label='참 이상치')
ax.set_title('LOF (밀도 변동 데이터)'); ax.legend(); plt.tight_layout(); plt.show()

X_u = np.random.uniform(0,10,(500,2)); lu = compute_lof(X_u,20)
print(f"균일분포 LOF: mean={lu.mean():.4f}, std={lu.std():.4f}")
""",
                "interpretation": "LOF는 지역 밀도 비교로 밀도 변동에도 강건. 균일분포에서 LOF≈1은 이론적 기대와 일치.",
            },
            {
                "approach": "이상치 탐지 앙상블",
                "code": r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score

np.random.seed(42)

def gen_scenario(sid, n=400):
    rng = np.random.RandomState(sid)
    if sid==0: Xn,Xo = rng.multivariate_normal([0,0],np.eye(2),n), rng.uniform(-6,6,(20,2)); name='산발적'
    elif sid==1: Xn,Xo = rng.multivariate_normal([0,0],np.eye(2),n), rng.multivariate_normal([5,5],0.3*np.eye(2),15); name='클러스터형'
    elif sid==2: Xn = rng.multivariate_normal([0,0],np.eye(2),n); Xo = np.column_stack([rng.normal(0,0.1,20),rng.uniform(-5,5,20)]); name='축형'
    elif sid==3: Xn = np.vstack([rng.multivariate_normal([0,0],np.eye(2),200),rng.multivariate_normal([6,0],np.eye(2),200)]); Xo = rng.uniform(-3,9,(15,2)); name='다중클러스터'
    else: Xn = rng.exponential(2,(n,2)); Xo = rng.uniform(-2,15,(20,2)); name='비대칭'
    X = np.vstack([Xn,Xo]); y = np.array([0]*len(Xn)+[1]*len(Xo)); return X,y,name

results = []
for sid in range(5):
    X,y,name = gen_scenario(sid); c = y.sum()/len(y); Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(200,contamination=c,random_state=42); iso.fit(Xs); s1 = -iso.score_samples(Xs)
    lof = LocalOutlierFactor(20,contamination=c); lof.fit_predict(Xs); s2 = -lof.negative_outlier_factor_
    svm = OneClassSVM(kernel='rbf',gamma='scale',nu=max(c,0.01)); svm.fit(Xs); s3 = -svm.score_samples(Xs)
    S = MinMaxScaler().fit_transform(np.column_stack([s1,s2,s3]))
    aucs = [roc_auc_score(y,S[:,i]) for i in range(3)]
    w = np.array(aucs)/sum(aucs)
    a_avg = roc_auc_score(y,S.mean(1)); a_max = roc_auc_score(y,S.max(1))
    a_wt = roc_auc_score(y,(S*w).sum(1))
    results.append({'scenario':name,'IF':aucs[0],'LOF':aucs[1],'OCSVM':aucs[2],'Avg':a_avg,'Max':a_max,'Weighted':a_wt})

df_r = pd.DataFrame(results)
print(df_r.to_string(index=False, float_format='%.3f'))
""",
                "interpretation": "AUC 가중 평균 앙상블이 대부분 시나리오에서 안정적. 앙상블은 단일 모델의 '최악의 경우'를 방지한다.",
            },
        ],
        discussion="### ML 이상치 탐지 선택\n\n| 특성 | 추천 |\n|------|------|\n| 고차원 대규모 | IF |\n| 밀도 변동 | LOF |\n| 명확한 경계 | OCSVM |\n| 강건성 | 앙상블 |",
        filepath=os.path.join(BASE, "ch01_04_outlier_ml_solution.ipynb"),
    )


# ============================================================
# Topic 05: 데이터 변환과 정규화 이론
# ============================================================
def gen_topic_05():
    problem_notebook(
        chapter_num=1, section_num=5,
        title="데이터 변환과 정규화 이론",
        objectives=[
            "Box-Cox 변환의 MLE 기반 최적 λ 추정 원리를 유도한다",
            "Yeo-Johnson 변환의 음수 확장 수학을 이해한다",
            "분산 안정화 변환의 델타 방법 기반 이론을 이해한다",
            "Robust Scaling의 통계적 근거와 이상치 저항성을 분석한다",
        ],
        theory_md=r"""
### Box-Cox: $y^{(\lambda)}=\frac{y^\lambda-1}{\lambda}$ ($\lambda\neq0$), $\ln y$ ($\lambda=0$)

프로파일 우도: $\ell(\lambda)=-\frac{n}{2}\ln\hat{\sigma}^2(\lambda)+(\lambda-1)\sum\ln y_i$

### Yeo-Johnson: 음수 포함 확장

### 분산 안정화 (Delta Method)

$\text{Var}(Y)=g(\mu)$이면 $h'(\mu)\propto 1/\sqrt{g(\mu)}$

| 분포 | $g(\mu)$ | $h(y)$ |
|------|----------|--------|
| 포아송 | $\mu$ | $\sqrt{y}$ |
| 이항 | $\mu(1-\mu)$ | $\arcsin\sqrt{y}$ |

### Robust Scaling: $z=(x-\text{median})/\text{IQR}$, 붕괴점 25%
""",
        guided_code=r"""import numpy as np, matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import PowerTransformer, RobustScaler

np.random.seed(42)
data_ln = np.random.lognormal(2, 1, 500)

def boxcox_ll(lam, y):
    n=len(y); yt = np.log(y) if abs(lam)<1e-10 else (y**lam-1)/lam
    s2=np.var(yt); return -(- n/2*np.log(s2)+(lam-1)*np.sum(np.log(y))) if s2>0 else 1e10

res = minimize_scalar(boxcox_ll, bounds=(-2,2), method='bounded', args=(data_ln,))
print(f"최적 λ: {res.x:.4f}")

fig, axes = plt.subplots(1,3,figsize=(15,4))
axes[0].hist(data_ln, bins=40, density=True); axes[0].set_title(f'원본 (skew={stats.skew(data_ln):.2f})')
bc, lam = stats.boxcox(data_ln)
axes[1].hist(bc, bins=40, density=True, color='orange'); axes[1].set_title(f'Box-Cox (λ={lam:.2f})')
yj = PowerTransformer('yeo-johnson').fit_transform(data_ln.reshape(-1,1)).flatten()
axes[2].hist(yj, bins=40, density=True, color='green'); axes[2].set_title(f'Yeo-Johnson')
plt.tight_layout(); plt.show()
""",
        exercises=[
            {
                "difficulty": "★",
                "description": "Box-Cox 프로파일 우도를 λ∈[-3,3]에서 그리고, 95% CI를 구하라 (우도비 기준).",
                "skeleton": "# TODO: 프로파일 우도 + CI\n",
            },
            {
                "difficulty": "★★",
                "description": "분산 안정화 변환을 델타 방법으로 유도하고 시뮬레이션 검증. 포아송(√y), 이항(arcsin√p).",
                "skeleton": "# TODO: 시뮬레이션\n",
            },
            {
                "difficulty": "★★",
                "description": "이상치 존재 시 Standard/MinMax/Robust/Power 스케일러 비교. 오염율별 분류 정확도.",
                "skeleton": "# TODO: 오염율별 실험\n",
            },
            {
                "difficulty": "★★★",
                "description": "Box-Cox λ의 점근 분산을 Fisher 정보량으로 유도, 부트스트랩(percentile, BCa)과 비교.",
                "skeleton": "# TODO: Fisher info + bootstrap CI\n",
            },
        ],
        references=[
            "Box & Cox (1964). An Analysis of Transformations. JRSS-B.",
            "Yeo & Johnson (2000). A New Family of Power Transformations.",
        ],
        filepath=os.path.join(BASE, "ch01_05_transformations.ipynb"),
    )

    solution_notebook(
        chapter_num=1, section_num=5,
        title="데이터 변환과 정규화 이론",
        solutions=[
            {
                "approach": "Box-Cox 프로파일 우도 + 95% CI",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42); data = np.random.lognormal(2, 0.8, 500)

def pll(lam, y):
    n=len(y); yt=np.log(y) if abs(lam)<1e-10 else (y**lam-1)/lam
    s2=np.var(yt); return -n/2*np.log(s2)+(lam-1)*np.sum(np.log(y)) if s2>0 else -1e10

lams = np.linspace(-2,3,500); lls = np.array([pll(l,data) for l in lams])
bi = np.argmax(lls); lh = lams[bi]; llm = lls[bi]
thr = llm - stats.chi2.ppf(0.95,1)/2
ci = lams[lls>=thr]; ci_lo, ci_hi = ci[0], ci[-1]

fig,ax = plt.subplots(figsize=(10,5))
ax.plot(lams,lls,'b-',lw=2); ax.axhline(thr,color='red',ls='--')
ax.axvline(lh,color='green',lw=2,label=f'λ̂={lh:.3f}')
ax.axvspan(ci_lo,ci_hi,alpha=0.2,color='red',label=f'95% CI:[{ci_lo:.3f},{ci_hi:.3f}]')
ax.set_xlabel('λ'); ax.set_ylabel('프로파일 로그우도'); ax.legend()
plt.tight_layout(); plt.show()
""",
                "interpretation": "프로파일 우도 최대점이 최적 λ. CI에 0 포함 시 log 변환, 1 포함 시 변환 불필요 가능.",
            },
            {
                "approach": "분산 안정화 변환 검증",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
np.random.seed(42); nsim=10000
mus = np.arange(1,51)
vr = [np.var(np.random.poisson(m,nsim)) for m in mus]
vs = [np.var(np.sqrt(np.random.poisson(m,nsim))) for m in mus]
ps = np.linspace(0.05,0.95,30); nb=50
vp = [np.var(np.random.binomial(nb,p,nsim)/nb) for p in ps]
va = [np.var(np.arcsin(np.sqrt(np.random.binomial(nb,p,nsim)/nb))) for p in ps]

fig,axes = plt.subplots(2,2,figsize=(14,10))
axes[0,0].plot(mus,vr,'b-'); axes[0,0].plot(mus,mus,'r--'); axes[0,0].set_title('포아송: Var(Y)')
axes[0,1].plot(mus,vs,'g-'); axes[0,1].axhline(0.25,color='r',ls='--'); axes[0,1].set_title('포아송: Var(√Y)')
axes[1,0].plot(ps,vp,'b-'); axes[1,0].plot(ps,[p*(1-p)/nb for p in ps],'r--'); axes[1,0].set_title('이항: Var(p̂)')
axes[1,1].plot(ps,va,'g-'); axes[1,1].axhline(1/(4*nb),color='r',ls='--'); axes[1,1].set_title('이항: Var(arcsin√p̂)')
plt.tight_layout(); plt.show()
""",
                "interpretation": "포아송 √Y → 분산≈1/4, 이항 arcsin√p → 분산≈1/(4n). 델타 방법 예측과 시뮬레이션 일치.",
            },
            {
                "approach": "이상치 하 스케일러 비교",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

np.random.seed(42)
X_c, y = make_classification(500, 10, n_informative=5, random_state=42)

def add_out(X, rate):
    Xn = X.copy(); no = int(len(X)*rate)
    if no>0: idx=np.random.choice(len(X),no,replace=False); Xn[idx]+=np.random.uniform(10,20,(no,X.shape[1]))*np.random.choice([-1,1],(no,X.shape[1]))
    return Xn

rates = [0,0.05,0.10,0.20]
scalers = {'Standard':StandardScaler(),'MinMax':MinMaxScaler(),'Robust':RobustScaler(),'Power':PowerTransformer('yeo-johnson')}
res = {n:[] for n in scalers}
for r in rates:
    Xd = add_out(X_c, r)
    for n, s in scalers.items():
        res[n].append(cross_val_score(LogisticRegression(max_iter=1000), s.fit_transform(Xd), y, cv=5).mean())

fig,ax = plt.subplots(figsize=(10,5))
for n,accs in res.items(): ax.plot([r*100 for r in rates], accs, 'o-', lw=2, label=n)
ax.set_xlabel('이상치 비율(%)'); ax.set_ylabel('정확도'); ax.legend(); ax.set_title('스케일러 비교')
plt.tight_layout(); plt.show()
""",
                "interpretation": "RobustScaler가 이상치에 가장 강건. MinMaxScaler가 가장 민감.",
            },
            {
                "approach": "Fisher 정보량 vs 부트스트랩 CI",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar

np.random.seed(42); n=200; data=np.random.lognormal(2,0.5,n)

def pll(lam,y):
    n=len(y); yt=np.log(y) if abs(lam)<1e-10 else (y**lam-1)/lam
    s2=np.var(yt); return -n/2*np.log(s2)+(lam-1)*np.sum(np.log(y)) if s2>0 else -1e10

lh = minimize_scalar(lambda l:-pll(l,data), bounds=(-2,3), method='bounded').x
eps=1e-4; fi = -(pll(lh+eps,data)-2*pll(lh,data)+pll(lh-eps,data))/eps**2
ase = 1/np.sqrt(fi) if fi>0 else np.nan

bl = [minimize_scalar(lambda l:-pll(l,data[np.random.choice(n,n)]), bounds=(-2,3), method='bounded').x for _ in range(2000)]
bl = np.array(bl); bse = bl.std()

ci_a = (lh-1.96*ase, lh+1.96*ase)
ci_p = (np.percentile(bl,2.5), np.percentile(bl,97.5))

fig,ax = plt.subplots(figsize=(10,5))
ax.hist(bl,50,density=True,alpha=0.6,label='부트스트랩')
xs = np.linspace(lh-4*ase,lh+4*ase,200)
ax.plot(xs,stats.norm.pdf(xs,lh,ase),'r-',lw=2,label='점근 정규')
ax.axvline(lh,color='green',lw=2); ax.legend()
ax.set_title(f'λ̂={lh:.3f}, 점근SE={ase:.4f}, 부트SE={bse:.4f}')
plt.tight_layout(); plt.show()

print(f"점근 CI: [{ci_a[0]:.4f},{ci_a[1]:.4f}]")
print(f"백분위 CI: [{ci_p[0]:.4f},{ci_p[1]:.4f}]")
""",
                "interpretation": "Fisher SE와 부트스트랩 SE 유사 시 정규 근사 적절. BCa는 비대칭 분포에서도 정확한 커버리지 제공.",
            },
        ],
        discussion="### 변환 선택 가이드\n\n| 상황 | 변환 |\n|------|------|\n| 양수, 우편향 | Box-Cox |\n| 음수 포함 | Yeo-Johnson |\n| 카운트 | √y |\n| 비율 | arcsin√p |\n| 이상치 | Robust Scaling |",
        filepath=os.path.join(BASE, "ch01_05_transformations_solution.ipynb"),
    )


# ============================================================
# Topic 06: 범주형 인코딩 고급 기법
# ============================================================
def gen_topic_06():
    problem_notebook(
        chapter_num=1, section_num=6,
        title="범주형 인코딩 고급 기법",
        objectives=[
            "Target Encoding의 원리와 정보 누출 방지 기법(smoothing, CV)을 이해한다",
            "Weight of Evidence(WoE)와 Information Value(IV)의 수학적 정의와 활용법을 습득한다",
            "CatBoost Encoding의 순서 의존 정보 누출 방지를 이해한다",
            "고카디널리티 범주형 변수 처리 전략을 비교한다",
        ],
        theory_md=r"""
### 1. Target Encoding (Mean Encoding)

범주 $c$에 대해 타겟 평균으로 인코딩: $\hat{y}_c = \frac{\sum_{i:x_i=c} y_i}{n_c}$

**Smoothing** (정보 누출 방지):

$$\hat{y}_c^{\text{smooth}} = \frac{n_c \cdot \bar{y}_c + m \cdot \bar{y}_{\text{global}}}{n_c + m}$$

$m$은 smoothing factor. $n_c$가 작을수록 전역 평균에 가까워진다.

### 2. Weight of Evidence (WoE)

이진 분류에서 범주 $c$의 WoE:

$$\text{WoE}_c = \ln\frac{P(X=c|Y=1)}{P(X=c|Y=0)} = \ln\frac{n_c^+/N^+}{n_c^-/N^-}$$

**Information Value (IV)**: 변수의 전체 예측력

$$\text{IV} = \sum_c \left(\frac{n_c^+}{N^+} - \frac{n_c^-}{N^-}\right) \cdot \text{WoE}_c$$

| IV 범위 | 예측력 |
|---------|--------|
| < 0.02 | 없음 |
| 0.02-0.1 | 약함 |
| 0.1-0.3 | 보통 |
| 0.3-0.5 | 강함 |
| > 0.5 | 의심스러움 (과적합 가능) |

### 3. CatBoost Encoding (Ordered Target Statistics)

데이터를 랜덤 순열로 정렬 후, 각 관측치에 대해 **자기 자신 이전**의 동일 범주 타겟 통계만 사용:

$$\hat{y}_i^{\text{CB}} = \frac{\sum_{j<i, x_j=x_i} y_j + a \cdot p}{|\{j<i: x_j=x_i\}| + a}$$

$a$: prior weight, $p$: prior probability
""",
        guided_code=r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import KFold

np.random.seed(42)
n = 1000
categories = np.random.choice(['A','B','C','D','E','F','G','H'], n)
cat_effect = {'A':0.8,'B':0.6,'C':0.4,'D':0.3,'E':0.2,'F':0.1,'G':0.05,'H':0.5}
probs = np.array([cat_effect[c] for c in categories])
y = (np.random.random(n) < probs).astype(int)
df = pd.DataFrame({'cat': categories, 'y': y})

# 1. Naive Target Encoding (정보 누출!)
te_naive = df.groupby('cat')['y'].mean()
df['te_naive'] = df['cat'].map(te_naive)

# 2. Smoothed Target Encoding
global_mean = y.mean()
m = 10
te_smooth = {}
for c in df['cat'].unique():
    nc = (df['cat']==c).sum()
    yc = df.loc[df['cat']==c, 'y'].mean()
    te_smooth[c] = (nc*yc + m*global_mean) / (nc + m)
df['te_smooth'] = df['cat'].map(te_smooth)

# 3. CV Target Encoding (정보 누출 방지)
df['te_cv'] = np.nan
kf = KFold(5, shuffle=True, random_state=42)
for tr_idx, val_idx in kf.split(df):
    means = df.iloc[tr_idx].groupby('cat')['y'].mean()
    df.loc[df.index[val_idx], 'te_cv'] = df.iloc[val_idx]['cat'].map(means)
df['te_cv'].fillna(global_mean, inplace=True)

print(df.groupby('cat')[['y','te_naive','te_smooth','te_cv']].mean().round(3))
""",
        exercises=[
            {
                "difficulty": "★",
                "description": "WoE/IV를 구현하고, 합성 데이터에서 각 범주형 변수의 IV를 계산하여 예측력을 평가하라.",
                "skeleton": "def compute_woe_iv(df, feature, target):\n    # TODO\n    pass\n",
            },
            {
                "difficulty": "★★",
                "description": "CatBoost Encoding을 직접 구현하고, Naive/Smoothed/CV/CatBoost 인코딩의 과적합 정도를 비교하라. 학습-검증 AUC 차이로 과적합을 측정.",
                "skeleton": "def catboost_encoding(df, cat_col, target_col, a=1):\n    # TODO\n    pass\n",
            },
            {
                "difficulty": "★★",
                "description": "고카디널리티(1000개 범주) 변수에서 One-Hot, Frequency, Target(CV), Hash 인코딩의 분류 성능과 메모리를 비교하라.",
                "skeleton": "# TODO: 고카디널리티 실험\n",
            },
            {
                "difficulty": "★★★",
                "description": "Target Encoding의 이론적 편향-분산 분해.\n1) smoothing factor m에 따른 편향/분산 tradeoff를 시뮬레이션\n2) 최적 m을 교차검증으로 선택하는 알고리즘 구현\n3) 범주별 빈도 불균형 시 적응적 m 전략 설계",
                "skeleton": "# TODO: m에 따른 편향/분산\n# TODO: 적응적 m\n",
            },
        ],
        references=[
            "Micci-Barreca, D. (2001). A Preprocessing Scheme for High-Cardinality Categorical Attributes.",
            "Prokhorenkova et al. (2018). CatBoost: Unbiased Boosting with Categorical Features.",
            "Siddiqi, N. (2006). Credit Risk Scorecards: WoE/IV methodology.",
        ],
        filepath=os.path.join(BASE, "ch01_06_encoding.ipynb"),
    )

    solution_notebook(
        chapter_num=1, section_num=6,
        title="범주형 인코딩 고급 기법",
        solutions=[
            {
                "approach": "WoE/IV 구현",
                "code": r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt

np.random.seed(42)
n = 2000
cats = np.random.choice(['Low','Mid','High','VHigh'], n, p=[0.4,0.3,0.2,0.1])
effects = {'Low':0.1,'Mid':0.3,'High':0.6,'VHigh':0.9}
y = np.array([int(np.random.random()<effects[c]) for c in cats])
df = pd.DataFrame({'cat':cats,'y':y})

def compute_woe_iv(df, feature, target):
    Np = (df[target]==1).sum(); Nn = (df[target]==0).sum()
    result = []
    for c in df[feature].unique():
        mask = df[feature]==c
        np_c = (df.loc[mask,target]==1).sum(); nn_c = (df.loc[mask,target]==0).sum()
        dp = np_c/Np if Np>0 else 0; dn = nn_c/Nn if Nn>0 else 0
        woe = np.log((dp+1e-10)/(dn+1e-10))
        iv_c = (dp-dn)*woe
        result.append({'category':c, 'n':mask.sum(), 'event_rate':np_c/mask.sum(), 'woe':woe, 'iv':iv_c})
    res = pd.DataFrame(result); total_iv = res['iv'].sum()
    return res, total_iv

r, iv = compute_woe_iv(df, 'cat', 'y')
print(r.round(4)); print(f"\nTotal IV: {iv:.4f}")

fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].bar(r['category'], r['woe'], color='steelblue')
axes[0].set_title('WoE by Category'); axes[0].set_ylabel('WoE')
axes[1].bar(r['category'], r['iv'], color='coral')
axes[1].set_title(f'IV by Category (Total={iv:.3f})')
plt.tight_layout(); plt.show()
""",
                "interpretation": "IV가 0.1-0.3이면 보통 수준의 예측력. WoE는 각 범주의 타겟 관련 방향과 강도를 나타낸다.",
            },
            {
                "approach": "CatBoost Encoding 구현 + 과적합 비교",
                "code": r"""import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

np.random.seed(42)
n=1000; cats = np.random.choice([f'C{i}' for i in range(20)], n)
effects = {f'C{i}': 0.1+0.04*i for i in range(20)}
y = np.array([int(np.random.random()<effects[c]) for c in cats])

def catboost_enc(cats, y, a=1):
    p = y.mean(); encoded = np.zeros(len(cats))
    perm = np.random.permutation(len(cats))
    counts = {}; sums = {}
    for idx in perm:
        c = cats[idx]
        encoded[idx] = (sums.get(c,0)+a*p) / (counts.get(c,0)+a)
        counts[c] = counts.get(c,0)+1; sums[c] = sums.get(c,0)+y[idx]
    return encoded

def cv_target_enc(cats, y, n_folds=5):
    enc = np.full(len(cats), np.nan); gm = y.mean()
    kf = KFold(n_folds, shuffle=True, random_state=42)
    idx = np.arange(len(cats))
    for tr, val in kf.split(idx):
        means = {}
        for c in set(cats[tr]):
            mask = cats[tr]==c; means[c] = y[tr][mask].mean()
        enc[val] = [means.get(cats[v], gm) for v in val]
    return np.nan_to_num(enc, nan=gm)

# 비교
methods = {
    'Naive': pd.Series(cats).map(pd.DataFrame({'c':cats,'y':y}).groupby('c')['y'].mean()).values,
    'CV': cv_target_enc(cats, y),
    'CatBoost': catboost_enc(cats, y),
}

for name, X_enc in methods.items():
    train_score = LogisticRegression().fit(X_enc.reshape(-1,1), y).score(X_enc.reshape(-1,1), y)
    cv_score = cross_val_score(LogisticRegression(), X_enc.reshape(-1,1), y, cv=5).mean()
    print(f"{name:10s}: Train={train_score:.3f}, CV={cv_score:.3f}, Gap={train_score-cv_score:.3f}")
""",
                "interpretation": "Naive는 train-CV gap이 크고 과적합. CV와 CatBoost는 정보 누출을 방지하여 gap이 작다.",
            },
            {
                "approach": "고카디널리티 인코딩 비교",
                "code": r"""import numpy as np, pandas as pd, sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_extraction import FeatureHasher

np.random.seed(42)
n = 5000; n_cats = 500
cats = np.array([f'cat_{np.random.randint(0,n_cats)}' for _ in range(n)])
effects = {f'cat_{i}': np.random.beta(2,5) for i in range(n_cats)}
y = np.array([int(np.random.random()<effects[c]) for c in cats])

# Frequency encoding
freq = pd.Series(cats).value_counts(normalize=True)
X_freq = np.array([freq[c] for c in cats]).reshape(-1,1)

# Target encoding (CV)
def cv_te(cats, y):
    enc = np.full(len(cats), y.mean())
    kf = KFold(5, shuffle=True, random_state=42)
    for tr, val in kf.split(np.arange(len(cats))):
        means = {}
        for c in set(cats[tr]): mask=cats[tr]==c; means[c]=y[tr][mask].mean()
        enc[val] = [means.get(cats[v], y.mean()) for v in val]
    return enc.reshape(-1,1)
X_te = cv_te(cats, y)

# Hash encoding
fh = FeatureHasher(n_features=64, input_type='string')
X_hash = fh.fit_transform([[c] for c in cats]).toarray()

for name, X_enc in [('Frequency',X_freq),('TargetCV',X_te),('Hash64',X_hash)]:
    cv = cross_val_score(LogisticRegression(max_iter=500), X_enc, y, cv=5).mean()
    mem = sys.getsizeof(X_enc) if isinstance(X_enc, np.ndarray) else X_enc.nbytes
    print(f"{name:12s}: CV AUC-proxy={cv:.3f}, Shape={X_enc.shape}, Mem~{mem//1024}KB")
""",
                "interpretation": "Target Encoding(CV)이 가장 높은 예측 성능. Hash Encoding은 메모리 효율적이지만 충돌로 성능 저하 가능. Frequency는 간단하지만 타겟 정보 미반영.",
            },
            {
                "approach": "Target Encoding 편향-분산 분해",
                "code": r"""import numpy as np, matplotlib.pyplot as plt

np.random.seed(42)
true_rates = {'A':0.3,'B':0.7,'C':0.5}
cat_sizes = {'A':200,'B':50,'C':10}
n_sims = 500

ms = [0,1,3,5,10,20,50,100]
bias2_all, var_all = {c:[] for c in true_rates}, {c:[] for c in true_rates}

for m in ms:
    for cat, true_p in true_rates.items():
        nc = cat_sizes[cat]; estimates = []
        for _ in range(n_sims):
            y_sim = np.random.binomial(1, true_p, nc)
            ybar = y_sim.mean(); gm = 0.5
            te = (nc*ybar + m*gm)/(nc+m)
            estimates.append(te)
        estimates = np.array(estimates)
        bias2_all[cat].append((estimates.mean()-true_p)**2)
        var_all[cat].append(estimates.var())

fig, axes = plt.subplots(1,3,figsize=(18,5))
for i, (cat, tp) in enumerate(true_rates.items()):
    nc = cat_sizes[cat]
    mse = np.array(bias2_all[cat])+np.array(var_all[cat])
    axes[i].plot(ms, bias2_all[cat], 'b-o', label='Bias²')
    axes[i].plot(ms, var_all[cat], 'r-s', label='Variance')
    axes[i].plot(ms, mse, 'g-^', label='MSE')
    axes[i].set_xlabel('m (smoothing)'); axes[i].set_title(f'{cat} (n={nc}, p={tp})')
    axes[i].legend(); axes[i].set_yscale('log')
plt.suptitle('Target Encoding: Bias-Variance Tradeoff'); plt.tight_layout(); plt.show()
""",
                "interpretation": "m이 클수록 편향 증가+분산 감소. 소규모 범주(C, n=10)에서 m의 효과가 가장 크다. 최적 m은 범주 빈도에 따라 적응적으로 설정해야 한다.",
            },
        ],
        discussion="### 인코딩 선택 가이드\n\n| 상황 | 추천 |\n|------|------|\n| 저카디널리티 (<10) | One-Hot |\n| 순서형 | Ordinal |\n| 이진 타겟, 해석력 | WoE |\n| 트리 모델 | Target/CatBoost |\n| 초고카디널리티 | Hash + Target |",
        filepath=os.path.join(BASE, "ch01_06_encoding_solution.ipynb"),
    )


# ============================================================
# Topic 07: 피처 엔지니어링과 도메인 지식
# ============================================================
def gen_topic_07():
    problem_notebook(
        chapter_num=1, section_num=7,
        title="피처 엔지니어링과 도메인 지식",
        objectives=[
            "상호작용 항과 다항식 특성의 수학적 근거를 이해한다",
            "도메인 지식 기반 변환(비율, 차이, 로그 등)의 효과를 분석한다",
            "자동 피처 생성 기법(PolynomialFeatures, featuretools 원리)을 활용한다",
            "피처 선택과 엔지니어링의 상호작용을 최적화한다",
        ],
        theory_md=r"""
### 1. 상호작용 항 (Interaction Terms)

선형 모형에서 $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2 + \epsilon$

$\beta_3 \neq 0$이면 $x_1$의 효과가 $x_2$ 수준에 따라 달라진다:

$$\frac{\partial E[y]}{\partial x_1} = \beta_1 + \beta_3 x_2$$

### 2. 다항식 특성

$d$차 다항식 확장: $(x_1, x_2) \to (1, x_1, x_2, x_1^2, x_1 x_2, x_2^2)$

$p$개 변수, 차수 $d$일 때 특성 수: $\binom{p+d}{d}$

### 3. 도메인 기반 변환

- **비율**: $\text{BMI} = \text{weight}/\text{height}^2$
- **차이**: $\Delta \text{price} = \text{price}_t - \text{price}_{t-1}$
- **로그 비율**: $\log(x_1/x_2)$ — 스케일 불변

### 4. 피처 중요도 기반 선택

상호 정보량: $I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$
""",
        guided_code=r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

np.random.seed(42)
n = 500
x1 = np.random.uniform(1, 10, n)
x2 = np.random.uniform(1, 10, n)
# 진짜 관계: 상호작용 포함
y = 3 + 2*x1 + 1.5*x2 + 0.5*x1*x2 - 0.1*x1**2 + np.random.normal(0, 2, n)
X = np.column_stack([x1, x2])

# 다항식 특성 생성
for d in [1, 2, 3]:
    poly = PolynomialFeatures(d, include_bias=False)
    X_poly = poly.fit_transform(X)
    scores = cross_val_score(LinearRegression(), X_poly, y, cv=5, scoring='r2')
    print(f"Degree {d}: features={X_poly.shape[1]}, R²={scores.mean():.4f}")
""",
        exercises=[
            {
                "difficulty": "★",
                "description": "합성 데이터에서 상호작용 항의 유무에 따른 회귀 성능 차이를 분석하라. 진짜 상호작용과 가짜 상호작용을 포함한 모형의 AIC/BIC를 비교.",
                "skeleton": "# TODO: 상호작용 모형 비교\n",
            },
            {
                "difficulty": "★★",
                "description": "금융 데이터 시뮬레이션에서 도메인 기반 피처를 설계하라.\n- 이동평균 비율, 변동성, 모멘텀, RSI 유사 지표\n- 원시 피처 대비 예측 성능 향상을 정량 비교",
                "skeleton": "# TODO: 금융 피처 생성\n# TODO: 성능 비교\n",
            },
            {
                "difficulty": "★★",
                "description": "자동 피처 생성 파이프라인을 구현하라. 산술 연산(+,-,*,/)으로 2차 피처를 생성하고, 상호정보량 기반으로 상위 k개를 선택.",
                "skeleton": "def auto_feature_gen(X, k=20):\n    # TODO\n    pass\n",
            },
            {
                "difficulty": "★★★",
                "description": "피처 엔지니어링의 이론적 한계를 분석하라.\n1) 다항식 차수 증가에 따른 편향-분산 tradeoff 시뮬레이션\n2) 고차원에서 다항식 특성의 차원의 저주 현상 시연\n3) 정규화(Ridge/Lasso)가 과적합을 어떻게 완화하는지 이론적+실험적으로 분석",
                "skeleton": "# TODO: 다항식 차수 + 정규화 실험\n",
            },
        ],
        references=[
            "Kuhn, M. & Johnson, K. (2019). Feature Engineering and Selection. CRC Press.",
            "Hastie, T. et al. (2009). Elements of Statistical Learning, Ch. 5.",
        ],
        filepath=os.path.join(BASE, "ch01_07_feature_engineering.ipynb"),
    )

    solution_notebook(
        chapter_num=1, section_num=7,
        title="피처 엔지니어링과 도메인 지식",
        solutions=[
            {
                "approach": "상호작용 항 AIC/BIC 비교",
                "code": r"""import numpy as np
from sklearn.linear_model import LinearRegression
np.random.seed(42); n=500
x1,x2,x3 = np.random.uniform(1,10,n), np.random.uniform(1,10,n), np.random.uniform(1,10,n)
y = 3+2*x1+1.5*x2+0.5*x1*x2+np.random.normal(0,2,n)

models = {
    'x1+x2': np.column_stack([x1,x2]),
    'x1+x2+x1x2': np.column_stack([x1,x2,x1*x2]),
    'x1+x2+x1x3': np.column_stack([x1,x2,x1*x3]),
    'x1+x2+x1x2+x2x3': np.column_stack([x1,x2,x1*x2,x2*x3]),
}
for name, X in models.items():
    lr = LinearRegression().fit(X, y); yp = lr.predict(X)
    rss = np.sum((y-yp)**2); k=X.shape[1]+1
    aic = n*np.log(rss/n)+2*k; bic = n*np.log(rss/n)+k*np.log(n)
    print(f"{name:25s}: R²={lr.score(X,y):.4f}, AIC={aic:.1f}, BIC={bic:.1f}")
""",
                "interpretation": "진짜 상호작용(x1*x2)을 포함한 모형이 AIC/BIC 모두 최소. 가짜 상호작용(x1*x3)은 개선 없이 복잡도만 증가.",
            },
            {
                "approach": "금융 도메인 피처 설계",
                "code": r"""import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(42); n=1000
prices = np.cumsum(np.random.normal(0.001,0.02,n))+100
volumes = np.random.lognormal(10,0.5,n)
y = (np.diff(np.append(prices,prices[-1]+np.random.normal()))>0).astype(int)[:n]
df = pd.DataFrame({'price':prices,'volume':volumes,'y':y})

# 원시 피처
X_raw = df[['price','volume']].values

# 도메인 피처
df['ma5'] = df['price'].rolling(5,min_periods=1).mean()
df['ma20'] = df['price'].rolling(20,min_periods=1).mean()
df['ma_ratio'] = df['ma5']/df['ma20']
df['volatility'] = df['price'].rolling(10,min_periods=1).std()
df['momentum'] = df['price'].pct_change(5)
df['vol_ma'] = df['volume'].rolling(5,min_periods=1).mean()
df['vol_ratio'] = df['volume']/(df['vol_ma']+1)

# RSI-like
delta = df['price'].diff()
gain = delta.clip(lower=0).rolling(14,min_periods=1).mean()
loss = (-delta.clip(upper=0)).rolling(14,min_periods=1).mean()
df['rsi'] = 100 - 100/(1+gain/(loss+1e-10))
df = df.fillna(0)

feat_cols = ['ma_ratio','volatility','momentum','vol_ratio','rsi']
X_domain = df[feat_cols].values

raw_cv = cross_val_score(GradientBoostingClassifier(n_estimators=50,max_depth=3,random_state=42), X_raw, y, cv=5).mean()
dom_cv = cross_val_score(GradientBoostingClassifier(n_estimators=50,max_depth=3,random_state=42), X_domain, y, cv=5).mean()
print(f"Raw features CV: {raw_cv:.3f}")
print(f"Domain features CV: {dom_cv:.3f}")
""",
                "interpretation": "도메인 지식 기반 피처(이동평균 비율, RSI 등)가 원시 피처(가격, 거래량)보다 우수한 예측 성능 제공.",
            },
            {
                "approach": "자동 피처 생성 + 상호정보량 선택",
                "code": r"""import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from itertools import combinations

np.random.seed(42); n=500; p=5
X = np.random.randn(n,p)
y = 2*X[:,0]*X[:,1] + X[:,2]**2 - 3*X[:,3] + np.random.normal(0,0.5,n)

def auto_feature_gen(X, y, k=20):
    n,p = X.shape; new_feats = []; names = []
    for i,j in combinations(range(p),2):
        new_feats.append(X[:,i]*X[:,j]); names.append(f'x{i}*x{j}')
        new_feats.append(X[:,i]+X[:,j]); names.append(f'x{i}+x{j}')
        denom = X[:,j].copy(); denom[np.abs(denom)<0.01]=0.01
        new_feats.append(X[:,i]/denom); names.append(f'x{i}/x{j}')
    for i in range(p):
        new_feats.append(X[:,i]**2); names.append(f'x{i}^2')
    X_new = np.column_stack(new_feats)
    mi = mutual_info_regression(X_new, y, random_state=42)
    top_k = np.argsort(mi)[-k:]
    return X_new[:, top_k], [names[i] for i in top_k], mi[top_k]

X_auto, feat_names, mis = auto_feature_gen(X, y, k=10)
print("상위 피처 (상호정보량):")
for name, mi_val in sorted(zip(feat_names, mis), key=lambda x:-x[1]):
    print(f"  {name}: MI={mi_val:.4f}")

base_cv = cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2').mean()
aug_cv = cross_val_score(LinearRegression(), np.column_stack([X, X_auto]), y, cv=5, scoring='r2').mean()
print(f"\n원본 R²: {base_cv:.4f}, 확장 R²: {aug_cv:.4f}")
""",
                "interpretation": "상호정보량으로 선택된 피처(x0*x1, x2^2)가 실제 데이터 생성 메커니즘과 일치. 자동 생성+선택으로 R² 향상.",
            },
            {
                "approach": "다항식 차수 + 정규화 편향-분산",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score

np.random.seed(42); n=200
x = np.random.uniform(0,5,n); y = np.sin(x)+0.3*np.random.randn(n)
X = x.reshape(-1,1)

degrees = range(1,16)
results = {'OLS':[],'Ridge':[],'Lasso':[]}
for d in degrees:
    Xp = PolynomialFeatures(d,include_bias=False).fit_transform(X)
    for name, model in [('OLS',LinearRegression()),('Ridge',Ridge(alpha=1)),('Lasso',Lasso(alpha=0.01))]:
        cv = cross_val_score(model, Xp, y, cv=5, scoring='neg_mean_squared_error').mean()
        results[name].append(-cv)

fig,ax = plt.subplots(figsize=(10,5))
for name in results: ax.plot(list(degrees), results[name], 'o-', label=name)
ax.set_xlabel('다항식 차수'); ax.set_ylabel('CV MSE'); ax.legend()
ax.set_title('차수에 따른 정규화 효과'); plt.tight_layout(); plt.show()
""",
                "interpretation": "OLS는 고차에서 과적합(MSE 급증). Ridge/Lasso는 정규화로 고차 계수를 억제하여 과적합 완화. 최적 차수는 편향-분산 균형점.",
            },
        ],
        discussion="### 피처 엔지니어링 원칙\n\n1. 도메인 지식 우선 (물리적 의미 있는 변환)\n2. 상호작용은 AIC/BIC로 검증\n3. 자동 생성은 선택과 결합\n4. 정규화로 과적합 방지",
        filepath=os.path.join(BASE, "ch01_07_feature_engineering_solution.ipynb"),
    )


# ============================================================
# Topic 08: 데이터 품질 프레임워크
# ============================================================
def gen_topic_08():
    problem_notebook(
        chapter_num=1, section_num=8,
        title="데이터 품질 프레임워크",
        objectives=[
            "데이터 드리프트 탐지의 통계적 방법(PSI, KS, χ²)을 이해한다",
            "스키마 검증 파이프라인을 설계한다",
            "통계적 프로파일링으로 데이터 이상을 자동 탐지한다",
            "데이터 품질 모니터링 대시보드를 구현한다",
        ],
        theory_md=r"""
### 1. Population Stability Index (PSI)

분포 변화 측정:

$$\text{PSI} = \sum_{i=1}^B \left(p_i^{\text{new}} - p_i^{\text{ref}}\right)\ln\frac{p_i^{\text{new}}}{p_i^{\text{ref}}}$$

| PSI | 해석 |
|-----|------|
| < 0.1 | 안정 |
| 0.1-0.25 | 주의 |
| > 0.25 | 유의한 변화 |

### 2. Kolmogorov-Smirnov 검정

$$D = \sup_x |F_{\text{ref}}(x) - F_{\text{new}}(x)|$$

### 3. 스키마 검증

- 타입 일관성, 범위 검사, 유일성 제약
- 참조 무결성, 분포 제약

### 4. 통계적 프로파일링

각 변수의 요약 통계량 + 이상 탐지: Z-score, IQR, 엔트로피 변화
""",
        guided_code=r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# 참조 데이터 vs 새 데이터 (드리프트 시뮬레이션)
ref_data = np.random.normal(50, 10, 10000)
new_data = np.random.normal(52, 12, 10000)  # 약간의 드리프트

def compute_psi(ref, new, bins=10):
    breakpoints = np.percentile(ref, np.linspace(0,100,bins+1))
    breakpoints[0] = -np.inf; breakpoints[-1] = np.inf
    ref_counts = np.histogram(ref, breakpoints)[0] / len(ref)
    new_counts = np.histogram(new, breakpoints)[0] / len(new)
    ref_counts = np.clip(ref_counts, 1e-6, None)
    new_counts = np.clip(new_counts, 1e-6, None)
    return np.sum((new_counts-ref_counts)*np.log(new_counts/ref_counts))

psi = compute_psi(ref_data, new_data)
ks_stat, ks_p = stats.ks_2samp(ref_data, new_data)

print(f"PSI: {psi:.4f} ({'안정' if psi<0.1 else '주의' if psi<0.25 else '유의한 변화'})")
print(f"KS: stat={ks_stat:.4f}, p={ks_p:.6f}")
""",
        exercises=[
            {
                "difficulty": "★",
                "description": "다양한 드리프트 유형(평균 이동, 분산 변화, 분포 변화)에서 PSI, KS, χ² 검정의 민감도를 비교하라.",
                "skeleton": "# TODO: 드리프트 유형별 검정\n",
            },
            {
                "difficulty": "★★",
                "description": "범용 스키마 검증 클래스를 구현하라. 타입/범위/유일성/분포 제약을 선언적으로 정의하고 자동 검증.",
                "skeleton": "class SchemaValidator:\n    # TODO\n    pass\n",
            },
            {
                "difficulty": "★★",
                "description": "통계적 프로파일러를 구현하여 시계열 데이터에서 분포 이상을 자동 탐지하라. 윈도우별 통계량 변화를 모니터링.",
                "skeleton": "class DataProfiler:\n    # TODO\n    pass\n",
            },
            {
                "difficulty": "★★★",
                "description": "다변량 드리프트 탐지기를 설계하라. MMD(Maximum Mean Discrepancy) 또는 LSDD(Least-Squares Density Difference) 기반으로 고차원 분포 변화를 탐지하고, 부트스트랩 p-value를 계산.",
                "skeleton": "def mmd_test(X_ref, X_new, kernel='rbf', n_permutations=1000):\n    # TODO\n    pass\n",
            },
        ],
        references=[
            "Rabanser et al. (2019). Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.",
            "Gretton et al. (2012). A Kernel Two-Sample Test. JMLR.",
        ],
        filepath=os.path.join(BASE, "ch01_08_data_quality.ipynb"),
    )

    solution_notebook(
        chapter_num=1, section_num=8,
        title="데이터 품질 프레임워크",
        solutions=[
            {
                "approach": "드리프트 유형별 검정 비교",
                "code": r"""import numpy as np, matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

def psi(ref, new, bins=10):
    bp = np.percentile(ref, np.linspace(0,100,bins+1)); bp[0]=-np.inf; bp[-1]=np.inf
    rc = np.clip(np.histogram(ref,bp)[0]/len(ref),1e-6,None)
    nc = np.clip(np.histogram(new,bp)[0]/len(new),1e-6,None)
    return np.sum((nc-rc)*np.log(nc/rc))

ref = np.random.normal(50,10,5000)
drifts = {
    '평균+2': np.random.normal(52,10,5000),
    '평균+5': np.random.normal(55,10,5000),
    '분산x1.5': np.random.normal(50,15,5000),
    '분산x2': np.random.normal(50,20,5000),
    '혼합': np.concatenate([np.random.normal(50,10,4000), np.random.normal(70,5,1000)]),
    '균일': np.random.uniform(20,80,5000),
}

results = []
for name, new in drifts.items():
    p = psi(ref, new)
    ks_s, ks_p = stats.ks_2samp(ref, new)
    bp = np.percentile(ref, np.linspace(0,100,11)); bp[0]=-np.inf; bp[-1]=np.inf
    chi_s, chi_p = stats.chisquare(np.histogram(new,bp)[0], np.histogram(ref,bp)[0])
    results.append({'Drift':name, 'PSI':p, 'KS':ks_s, 'KS_p':ks_p, 'Chi2':chi_s, 'Chi2_p':chi_p})

import pandas as pd
print(pd.DataFrame(results).to_string(index=False, float_format='%.4f'))
""",
                "interpretation": "PSI는 평균 이동에 민감, KS는 모든 유형에 높은 검정력, χ²는 분포 형태 변화에 유용. 다각적 검정이 권장.",
            },
            {
                "approach": "스키마 검증 클래스",
                "code": r"""import numpy as np, pandas as pd

class SchemaValidator:
    def __init__(self):
        self.rules = []
    def add_type(self, col, dtype): self.rules.append(('type', col, dtype))
    def add_range(self, col, lo, hi): self.rules.append(('range', col, lo, hi))
    def add_unique(self, col): self.rules.append(('unique', col))
    def add_not_null(self, col): self.rules.append(('not_null', col))
    def validate(self, df):
        results = []
        for rule in self.rules:
            if rule[0]=='type':
                ok = df[rule[1]].dtype.kind in rule[2]
                results.append(f"Type({rule[1]}): {'PASS' if ok else 'FAIL'}")
            elif rule[0]=='range':
                lo_ok = df[rule[1]].min() >= rule[2]; hi_ok = df[rule[1]].max() <= rule[3]
                results.append(f"Range({rule[1]}): {'PASS' if lo_ok and hi_ok else 'FAIL'} [{df[rule[1]].min():.1f},{df[rule[1]].max():.1f}]")
            elif rule[0]=='unique':
                ok = df[rule[1]].nunique() == len(df)
                results.append(f"Unique({rule[1]}): {'PASS' if ok else 'FAIL'}")
            elif rule[0]=='not_null':
                nn = df[rule[1]].isnull().sum()
                results.append(f"NotNull({rule[1]}): {'PASS' if nn==0 else f'FAIL ({nn} nulls)'}")
        return results

sv = SchemaValidator()
sv.add_type('age', 'if'); sv.add_range('age', 0, 150)
sv.add_not_null('name'); sv.add_unique('id')

df_test = pd.DataFrame({'id':[1,2,3,3], 'name':['a','b',None,'d'], 'age':[25,200,30,-5]})
for r in sv.validate(df_test): print(r)
""",
                "interpretation": "선언적 스키마 검증으로 데이터 품질 문제를 자동 탐지. 파이프라인에 통합하여 조기 경보 가능.",
            },
            {
                "approach": "시계열 데이터 프로파일러",
                "code": r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt

np.random.seed(42)
# 시뮬레이션: 정상 + 이상 구간
n = 1000
data = np.random.normal(50, 10, n)
data[400:450] += 30  # 평균 이동
data[700:750] *= 3   # 분산 폭증

class DataProfiler:
    def __init__(self, window=50, stride=25):
        self.window, self.stride = window, stride
    def profile(self, data):
        stats_list = []
        for i in range(0, len(data)-self.window+1, self.stride):
            w = data[i:i+self.window]
            stats_list.append({'start':i, 'mean':w.mean(), 'std':w.std(), 'skew':float(pd.Series(w).skew()),
                              'null_rate':np.isnan(w).mean(), 'min':w.min(), 'max':w.max()})
        return pd.DataFrame(stats_list)
    def detect_anomalies(self, stats_df, z_thresh=2.5):
        alerts = []
        for col in ['mean','std']:
            mu, sig = stats_df[col].median(), stats_df[col].std()
            anomalous = np.abs(stats_df[col]-mu)/sig > z_thresh
            for idx in stats_df[anomalous].index:
                alerts.append(f"Window {stats_df.loc[idx,'start']}: {col} 이상 ({stats_df.loc[idx,col]:.1f})")
        return alerts

dp = DataProfiler(50, 25)
st = dp.profile(data)
alerts = dp.detect_anomalies(st)
print("탐지된 이상:")
for a in alerts: print(f"  {a}")

fig,axes = plt.subplots(3,1,figsize=(12,8),sharex=True)
axes[0].plot(data,lw=0.5); axes[0].set_title('원본')
axes[1].plot(st['start'],st['mean'],'b-o',ms=3); axes[1].set_title('윈도우 평균')
axes[2].plot(st['start'],st['std'],'r-o',ms=3); axes[2].set_title('윈도우 표준편차')
plt.tight_layout(); plt.show()
""",
                "interpretation": "윈도우 기반 프로파일링으로 평균 이동(구간 400-450)과 분산 폭증(구간 700-750)을 자동 탐지.",
            },
            {
                "approach": "MMD 기반 다변량 드리프트 탐지",
                "code": r"""import numpy as np
from scipy.spatial.distance import cdist

np.random.seed(42)

def mmd_rbf(X, Y, gamma=None):
    if gamma is None:
        gamma = 1.0/X.shape[1]
    Kxx = np.exp(-gamma*cdist(X,X,'sqeuclidean'))
    Kyy = np.exp(-gamma*cdist(Y,Y,'sqeuclidean'))
    Kxy = np.exp(-gamma*cdist(X,Y,'sqeuclidean'))
    n,m = len(X),len(Y)
    np.fill_diagonal(Kxx, 0); np.fill_diagonal(Kyy, 0)
    return Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2*Kxy.sum()/(n*m)

def mmd_test(X_ref, X_new, n_perm=500):
    mmd_obs = mmd_rbf(X_ref, X_new)
    combined = np.vstack([X_ref, X_new])
    n = len(X_ref); m = len(X_new)
    mmd_perms = []
    for _ in range(n_perm):
        perm = np.random.permutation(n+m)
        mmd_perms.append(mmd_rbf(combined[perm[:n]], combined[perm[n:]]))
    p_value = np.mean(np.array(mmd_perms)>=mmd_obs)
    return {'mmd': mmd_obs, 'p_value': p_value, 'permutation_mmds': mmd_perms}

# 테스트
X_ref = np.random.multivariate_normal([0,0,0], np.eye(3), 200)
X_same = np.random.multivariate_normal([0,0,0], np.eye(3), 200)
X_drift = np.random.multivariate_normal([0.5,0.3,0], 1.2*np.eye(3), 200)

r1 = mmd_test(X_ref, X_same)
r2 = mmd_test(X_ref, X_drift)
print(f"Same dist: MMD={r1['mmd']:.6f}, p={r1['p_value']:.4f}")
print(f"Drifted:   MMD={r2['mmd']:.6f}, p={r2['p_value']:.4f}")
""",
                "interpretation": "MMD는 커널 기반으로 고차원 분포 차이를 비모수적으로 탐지. 순열 검정으로 통계적 유의성 평가. 같은 분포에서는 p>0.05, 드리프트 시 p<0.05.",
            },
        ],
        discussion="### 데이터 품질 모니터링 체크리스트\n\n1. 스키마: 타입, 범위, null, 유일성\n2. 분포: PSI, KS (단변량), MMD (다변량)\n3. 프로파일: 윈도우 통계량 모니터링\n4. 자동화: 임계값 기반 알림 파이프라인",
        filepath=os.path.join(BASE, "ch01_08_data_quality_solution.ipynb"),
    )


# ============================================================
# Topic 09: 대용량 데이터 처리
# ============================================================
def gen_topic_09():
    problem_notebook(
        chapter_num=1, section_num=9,
        title="대용량 데이터 처리",
        objectives=[
            "Pandas vs Polars의 아키텍처 차이를 이해한다",
            "청크 처리(chunk processing)로 메모리 한계를 극복한다",
            "메모리 최적화 기법(다운캐스팅, 범주형 변환)을 적용한다",
            "지연 평가(lazy evaluation)의 원리와 장점을 이해한다",
        ],
        theory_md=r"""
### 1. Pandas vs Polars 아키텍처

**Pandas**: NumPy 기반, 단일 스레드, eager evaluation
- 메모리: 데이터의 2-5배 필요 (중간 결과 복사)

**Polars**: Arrow 기반, 멀티스레드, lazy evaluation
- 쿼리 최적화, 술어 밀어내기(predicate pushdown)
- 메모리 효율적: zero-copy, 참조 카운팅

### 2. 메모리 최적화

**다운캐스팅**: int64 → int8/16/32 (범위에 맞게)

| 타입 | 범위 | 메모리 |
|------|------|--------|
| int8 | -128~127 | 1 byte |
| int16 | -32768~32767 | 2 bytes |
| float32 | ±3.4e38 | 4 bytes |

**범주형 변환**: 반복 문자열 → category (메모리 절감 90%+)

### 3. 청크 처리

파일을 $k$ 행씩 읽어 순차 처리:
- 스트리밍 집계: 온라인 평균/분산 (Welford 알고리즘)
- 맵-리듀스 패턴

### 4. Welford의 온라인 알고리즘

$$M_k = M_{k-1} + \frac{x_k - M_{k-1}}{k}, \quad S_k = S_{k-1} + (x_k - M_{k-1})(x_k - M_k)$$

$$\text{Var} = S_n / (n-1)$$
""",
        guided_code=r"""import numpy as np, pandas as pd, time

# 대용량 합성 데이터 생성
np.random.seed(42)
n = 500_000
df = pd.DataFrame({
    'id': np.arange(n),
    'category': np.random.choice(['A','B','C','D','E']*200, n),
    'value1': np.random.normal(100, 20, n),
    'value2': np.random.uniform(0, 1000, n),
    'flag': np.random.randint(0, 2, n),
})

# 메모리 최적화
def optimize_memory(df):
    before = df.memory_usage(deep=True).sum()
    for col in df.select_dtypes(include=['int64']).columns:
        c_min, c_max = df[col].min(), df[col].max()
        if c_min >= 0 and c_max < 255: df[col] = df[col].astype('uint8')
        elif c_min > -128 and c_max < 127: df[col] = df[col].astype('int8')
        elif c_min > -32768 and c_max < 32767: df[col] = df[col].astype('int16')
        elif c_min > -2147483648 and c_max < 2147483647: df[col] = df[col].astype('int32')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    after = df.memory_usage(deep=True).sum()
    print(f"메모리: {before/1e6:.1f}MB → {after/1e6:.1f}MB ({100*(1-after/before):.1f}% 절감)")
    return df

df_opt = optimize_memory(df.copy())
""",
        exercises=[
            {
                "difficulty": "★",
                "description": "1000만 행 합성 데이터에서 메모리 최적화(다운캐스팅, 범주형 변환)를 적용하고 최적화 전후 메모리/연산 시간을 비교하라.",
                "skeleton": "# TODO: 대용량 데이터 생성\n# TODO: 최적화 전후 비교\n",
            },
            {
                "difficulty": "★★",
                "description": "Welford 온라인 알고리즘을 구현하고, 청크 처리로 메모리에 들어오지 않는 데이터의 평균/분산/백분위수를 정확하게 계산하라.",
                "hint": "백분위수는 t-digest 또는 Q-digest 근사 알고리즘을 사용.",
                "skeleton": "class OnlineStats:\n    # TODO: Welford algorithm\n    pass\n",
            },
            {
                "difficulty": "★★",
                "description": "Pandas vs Polars 벤치마크: groupby, join, filter, sort 연산의 속도를 비교하라. (Polars 미설치 시 pandas 최적화 기법으로 대체)",
                "skeleton": "# TODO: 벤치마크 프레임워크\n",
            },
            {
                "difficulty": "★★★",
                "description": "스트리밍 GroupBy 집계기를 구현하라. 파일을 청크로 읽으면서 그룹별 평균/분산/카운트를 정확히 계산하고, 결과가 전체 데이터 로드와 동일함을 검증.",
                "skeleton": "class StreamingGroupBy:\n    # TODO\n    pass\n",
            },
        ],
        references=[
            "Polars documentation: https://pola.rs/",
            "Welford, B.P. (1962). Note on a Method for Calculating Corrected Sums of Squares.",
            "McKinney, W. (2017). Python for Data Analysis, 2nd ed.",
        ],
        filepath=os.path.join(BASE, "ch01_09_large_scale.ipynb"),
    )

    solution_notebook(
        chapter_num=1, section_num=9,
        title="대용량 데이터 처리",
        solutions=[
            {
                "approach": "메모리 최적화 전후 비교",
                "code": r"""import numpy as np, pandas as pd, time

np.random.seed(42); n=2_000_000
df = pd.DataFrame({
    'id': np.arange(n), 'cat': np.random.choice([f'cat_{i}' for i in range(50)], n),
    'v1': np.random.normal(100,20,n), 'v2': np.random.randint(0,100,n), 'flag': np.random.randint(0,2,n),
})

before = df.memory_usage(deep=True).sum()
t0 = time.time(); _ = df.groupby('cat')['v1'].mean(); t_before = time.time()-t0

for c in df.select_dtypes('int64').columns:
    mn,mx = df[c].min(), df[c].max()
    if mn>=0 and mx<255: df[c]=df[c].astype('uint8')
    elif mn>-32768 and mx<32767: df[c]=df[c].astype('int16')
    else: df[c]=df[c].astype('int32')
for c in df.select_dtypes('float64').columns: df[c]=df[c].astype('float32')
for c in df.select_dtypes('object').columns: df[c]=df[c].astype('category')

after = df.memory_usage(deep=True).sum()
t0 = time.time(); _ = df.groupby('cat')['v1'].mean(); t_after = time.time()-t0

print(f"메모리: {before/1e6:.1f}MB → {after/1e6:.1f}MB ({100*(1-after/before):.1f}% 절감)")
print(f"GroupBy 시간: {t_before*1000:.1f}ms → {t_after*1000:.1f}ms")
""",
                "interpretation": "다운캐스팅과 범주형 변환으로 60-80% 메모리 절감. 캐시 효율 향상으로 연산 속도도 개선된다.",
            },
            {
                "approach": "Welford 온라인 통계량",
                "code": r"""import numpy as np

class OnlineStats:
    def __init__(self):
        self.n = 0; self.mean = 0; self.M2 = 0; self.min_v = np.inf; self.max_v = -np.inf
    def update(self, x):
        if np.isscalar(x): x = [x]
        for val in x:
            self.n += 1; delta = val-self.mean
            self.mean += delta/self.n; delta2 = val-self.mean
            self.M2 += delta*delta2
            self.min_v = min(self.min_v, val); self.max_v = max(self.max_v, val)
    @property
    def variance(self): return self.M2/(self.n-1) if self.n>1 else 0
    @property
    def std(self): return np.sqrt(self.variance)
    def summary(self): return f"n={self.n}, mean={self.mean:.4f}, std={self.std:.4f}, min={self.min_v:.4f}, max={self.max_v:.4f}"

# 검증: 청크 처리 vs 전체
np.random.seed(42); data = np.random.normal(42, 7, 100000)
os = OnlineStats()
chunk_size = 1000
for i in range(0, len(data), chunk_size):
    os.update(data[i:i+chunk_size])

print(f"Online:  {os.summary()}")
print(f"NumPy:   n={len(data)}, mean={data.mean():.4f}, std={data.std(ddof=1):.4f}, min={data.min():.4f}, max={data.max():.4f}")
print(f"일치: mean차이={abs(os.mean-data.mean()):.2e}, std차이={abs(os.std-data.std(ddof=1)):.2e}")
""",
                "interpretation": "Welford 알고리즘으로 단일 패스 + O(1) 메모리로 정확한 평균/분산 계산. 수치 안정성이 단순 합산 방식보다 우수.",
            },
            {
                "approach": "Pandas 최적화 벤치마크",
                "code": r"""import numpy as np, pandas as pd, time

np.random.seed(42); n=1_000_000
df = pd.DataFrame({
    'key1': np.random.choice([f'k{i}' for i in range(100)], n),
    'key2': np.random.choice([f'g{i}' for i in range(20)], n),
    'val': np.random.randn(n).astype('float32'),
    'id': np.arange(n, dtype='int32'),
})
df2 = pd.DataFrame({'key1': [f'k{i}' for i in range(100)], 'info': np.random.randn(100)})

ops = {}

t0=time.time(); _ = df.groupby('key1')['val'].agg(['mean','std','count']); ops['groupby']=time.time()-t0
t0=time.time(); _ = df.merge(df2, on='key1'); ops['join']=time.time()-t0
t0=time.time(); _ = df[df['val']>0]; ops['filter']=time.time()-t0
t0=time.time(); _ = df.sort_values('val'); ops['sort']=time.time()-t0

print(f"{'Operation':12s} {'Time(ms)':>10s}")
for op, t in ops.items(): print(f"{op:12s} {t*1000:10.1f}")

# 카테고리 최적화 효과
df_opt = df.copy(); df_opt['key1']=df_opt['key1'].astype('category')
t0=time.time(); _ = df_opt.groupby('key1')['val'].mean(); t_cat = time.time()-t0
t0=time.time(); _ = df.groupby('key1')['val'].mean(); t_str = time.time()-t0
print(f"\nGroupBy: string={t_str*1000:.1f}ms, category={t_cat*1000:.1f}ms")
""",
                "interpretation": "범주형 변환으로 GroupBy 속도 향상. Polars는 병렬 처리+쿼리 최적화로 추가 성능 이점.",
            },
            {
                "approach": "스트리밍 GroupBy 구현",
                "code": r"""import numpy as np, pandas as pd

class StreamingGroupBy:
    def __init__(self):
        self.groups = {}  # {key: {'n':0, 'mean':0, 'M2':0}}
    def update(self, chunk_df, key_col, val_col):
        for key, grp in chunk_df.groupby(key_col):
            vals = grp[val_col].values
            if key not in self.groups:
                self.groups[key] = {'n':0, 'mean':0.0, 'M2':0.0}
            g = self.groups[key]
            for v in vals:
                g['n'] += 1; d = v-g['mean']
                g['mean'] += d/g['n']; d2 = v-g['mean']
                g['M2'] += d*d2
    def result(self):
        rows = []
        for key, g in self.groups.items():
            var = g['M2']/(g['n']-1) if g['n']>1 else 0
            rows.append({'key':key, 'count':g['n'], 'mean':g['mean'], 'std':np.sqrt(var)})
        return pd.DataFrame(rows).sort_values('key').reset_index(drop=True)

# 검증
np.random.seed(42); n=500000
df = pd.DataFrame({'key':np.random.choice(['A','B','C','D','E'], n), 'val':np.random.randn(n)})

# 스트리밍
sgb = StreamingGroupBy()
chunk_size = 10000
for i in range(0, n, chunk_size):
    sgb.update(df.iloc[i:i+chunk_size], 'key', 'val')
res_stream = sgb.result()

# 전체
res_full = df.groupby('key')['val'].agg(['count','mean','std']).reset_index()
res_full.columns = ['key','count','mean','std']

print("Streaming vs Full:")
merged = res_stream.merge(res_full, on='key', suffixes=('_stream','_full'))
for _, r in merged.iterrows():
    print(f"  {r['key']}: mean diff={abs(r['mean_stream']-r['mean_full']):.2e}, std diff={abs(r['std_stream']-r['std_full']):.2e}")
""",
                "interpretation": "스트리밍 GroupBy가 전체 로드와 동일한 결과를 O(그룹 수) 메모리로 달성. Welford 기반으로 수치 안정성 보장.",
            },
        ],
        discussion="### 대용량 처리 전략\n\n1. 메모리 최적화 우선 (다운캐스팅, 범주형)\n2. 스트리밍/청크 처리 (Welford 기반)\n3. Polars/Dask 활용 (병렬+지연평가)\n4. 데이터베이스 연계 (SQL pushdown)",
        filepath=os.path.join(BASE, "ch01_09_large_scale_solution.ipynb"),
    )


# ============================================================
# Topic 10: 실전 - 금융 거래 데이터 종합 분석
# ============================================================
def gen_topic_10():
    problem_notebook(
        chapter_num=1, section_num=10,
        title="실전: 금융 거래 데이터 종합 분석",
        objectives=[
            "합성 금융 거래 데이터를 생성하고 EDA 파이프라인을 구축한다",
            "이상 거래 탐지를 다변량 통계+ML 방법으로 수행한다",
            "시간 패턴, 고객 세그먼트, 거래 네트워크를 분석한다",
            "종합 분석 리포트를 자동 생성한다",
        ],
        theory_md=r"""
### 금융 EDA의 핵심 관점

1. **시간 패턴**: 거래 빈도, 주기성, 추세
2. **이상 거래**: 금액, 빈도, 지리적 패턴
3. **고객 세그먼트**: RFM 분석 (Recency, Frequency, Monetary)
4. **Benford's Law**: 첫째 자릿수 분포 $P(d) = \log_{10}(1+1/d)$

### RFM 분석

- **Recency**: 마지막 거래 이후 경과일
- **Frequency**: 거래 횟수
- **Monetary**: 총 거래 금액

각 지표를 분위수로 점수화하여 고객을 세분화한다.

### Benford's Law

자연적으로 생성된 수치 데이터의 첫째 자릿수 분포:

$$P(\text{first digit}=d) = \log_{10}\left(1+\frac{1}{d}\right), \quad d=1,2,\ldots,9$$

거래 금액이 이 법칙을 심각하게 위반하면 조작/사기 의심.
""",
        guided_code=r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

# 합성 금융 거래 데이터 생성
n_customers = 500; n_transactions = 20000
base_date = datetime(2025, 1, 1)

customer_ids = np.random.randint(1, n_customers+1, n_transactions)
days_offset = np.random.exponential(5, n_transactions).astype(int)
dates = [base_date + timedelta(days=int(d)) for d in np.cumsum(np.random.geometric(0.3, n_transactions))]
dates = dates[:n_transactions]

# 정상 거래 금액 (로그정규)
amounts = np.random.lognormal(mean=4, sigma=1.5, size=n_transactions)
amounts = np.round(amounts, 2)

# 이상 거래 주입 (2%)
n_fraud = int(n_transactions * 0.02)
fraud_idx = np.random.choice(n_transactions, n_fraud, replace=False)
amounts[fraud_idx] *= np.random.uniform(5, 20, n_fraud)

categories = np.random.choice(['식료품','교통','쇼핑','외식','온라인','금융','의료'], n_transactions,
                                p=[0.25, 0.15, 0.20, 0.15, 0.10, 0.10, 0.05])

df = pd.DataFrame({
    'date': dates, 'customer_id': customer_ids, 'amount': amounts,
    'category': categories, 'is_fraud': 0
})
df.loc[fraud_idx, 'is_fraud'] = 1
df['date'] = pd.to_datetime(df['date'])

print(f"거래 수: {len(df)}, 이상 거래: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.1%})")
print(f"\n금액 통계:\n{df['amount'].describe()}")
""",
        exercises=[
            {
                "difficulty": "★",
                "description": "금융 거래 데이터에 대해 포괄적 EDA를 수행하라.\n- 시간별/요일별/월별 거래 패턴\n- 카테고리별 거래 분포\n- 금액 분포와 이상 거래 후보 식별\n- Benford's law 적합도 검정",
                "skeleton": "# TODO: 시간 패턴 분석\n# TODO: 카테고리 분석\n# TODO: Benford 검정\n",
            },
            {
                "difficulty": "★★",
                "description": "RFM 분석을 수행하고 K-means로 고객을 세분화하라. 세그먼트별 이상 거래 비율을 비교.",
                "skeleton": "# TODO: RFM 계산\n# TODO: K-means 세분화\n# TODO: 세그먼트별 분석\n",
            },
            {
                "difficulty": "★★",
                "description": "다변량 이상 거래 탐지 파이프라인을 구축하라.\n- 고객별 피처 엔지니어링 (거래 빈도, 평균/최대 금액, 카테고리 엔트로피 등)\n- Mahalanobis + IF + LOF 앙상블\n- Precision/Recall 평가",
                "skeleton": "# TODO: 피처 엔지니어링\n# TODO: 앙상블 탐지\n# TODO: 성능 평가\n",
            },
            {
                "difficulty": "★★★",
                "description": "종합 분석 리포트 자동 생성기를 구현하라.\n1) 모든 변수의 자동 프로파일링\n2) 이상치/결측/드리프트 자동 탐지\n3) 피처 중요도 기반 핵심 변수 식별\n4) 요약 통계 + 시각화를 포함한 HTML/텍스트 리포트 생성",
                "skeleton": "class AutoEDAReport:\n    # TODO\n    pass\n",
            },
        ],
        references=[
            "Benford, F. (1938). The Law of Anomalous Numbers. Proceedings APS.",
            "Bolton, R.J. & Hand, D.J. (2002). Statistical Fraud Detection: A Review.",
            "Fawcett, T. & Provost, F. (1997). Adaptive Fraud Detection. Data Mining and Knowledge Discovery.",
        ],
        filepath=os.path.join(BASE, "ch01_10_practice_financial_eda.ipynb"),
    )

    solution_notebook(
        chapter_num=1, section_num=10,
        title="실전: 금융 거래 데이터 종합 분석",
        solutions=[
            {
                "approach": "포괄적 EDA + Benford 검정",
                "code": r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta

np.random.seed(42)
n=20000; nc=500; base=datetime(2025,1,1)
cids = np.random.randint(1,nc+1,n)
dates = [base+timedelta(days=int(d)) for d in np.cumsum(np.random.geometric(0.3,n))][:n]
amounts = np.random.lognormal(4,1.5,n).round(2)
fi = np.random.choice(n,int(n*0.02),replace=False); amounts[fi]*=np.random.uniform(5,20,len(fi))
cats = np.random.choice(['식료품','교통','쇼핑','외식','온라인','금융','의료'],n,p=[.25,.15,.20,.15,.10,.10,.05])
df = pd.DataFrame({'date':pd.to_datetime(dates),'cid':cids,'amount':amounts,'cat':cats,'fraud':0})
df.loc[fi,'fraud']=1

# 시간 패턴
df['dow'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

fig,axes = plt.subplots(2,2,figsize=(14,10))
df.groupby('dow')['amount'].count().plot(kind='bar',ax=axes[0,0],title='요일별 거래 수')
df.groupby('cat')['amount'].mean().sort_values().plot(kind='barh',ax=axes[0,1],title='카테고리별 평균 금액')
axes[1,0].hist(np.log10(df['amount']+1), bins=50, alpha=0.7); axes[1,0].set_title('log10(금액) 분포')

# Benford
first_digits = df['amount'].apply(lambda x: int(str(abs(x)).replace('.','').lstrip('0')[0]) if x>0 else 0)
first_digits = first_digits[first_digits>0]
observed = first_digits.value_counts(normalize=True).sort_index()
expected = pd.Series({d: np.log10(1+1/d) for d in range(1,10)})
axes[1,1].bar(observed.index-0.15, observed.values, 0.3, label='관측', alpha=0.7)
axes[1,1].bar(expected.index+0.15, expected.values, 0.3, label='Benford', alpha=0.7)
axes[1,1].set_title("Benford's Law 검정"); axes[1,1].legend()
plt.tight_layout(); plt.show()

chi2, p = stats.chisquare(observed.values*len(first_digits), expected.values*len(first_digits))
print(f"Benford χ² 검정: stat={chi2:.2f}, p={p:.4f}")
""",
                "interpretation": "거래 금액이 Benford's law와 크게 괴리되면 조작/사기 가능성 시사. 이상 거래 주입으로 분포가 약간 변형된 것을 확인할 수 있다.",
            },
            {
                "approach": "RFM 분석 + K-means 세분화",
                "code": r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

np.random.seed(42)
n=20000; nc=500; base=datetime(2025,1,1)
cids = np.random.randint(1,nc+1,n)
dates = [base+timedelta(days=int(d)) for d in np.cumsum(np.random.geometric(0.3,n))][:n]
amounts = np.random.lognormal(4,1.5,n).round(2)
fi = np.random.choice(n,int(n*0.02),replace=False); amounts[fi]*=np.random.uniform(5,20,len(fi))
df = pd.DataFrame({'date':pd.to_datetime(dates),'cid':cids,'amount':amounts,'fraud':0})
df.loc[fi,'fraud']=1

ref_date = df['date'].max()
rfm = df.groupby('cid').agg(
    R=('date', lambda x: (ref_date-x.max()).days),
    F=('amount', 'count'),
    M=('amount', 'sum'),
    fraud_rate=('fraud', 'mean')
).reset_index()

X_rfm = StandardScaler().fit_transform(rfm[['R','F','M']])
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_rfm)
rfm['segment'] = km.labels_

fig,axes = plt.subplots(1,2,figsize=(14,5))
for seg in range(4):
    mask = rfm['segment']==seg
    axes[0].scatter(rfm.loc[mask,'F'], rfm.loc[mask,'M'], s=10, alpha=0.5, label=f'Seg {seg}')
axes[0].set_xlabel('Frequency'); axes[0].set_ylabel('Monetary'); axes[0].legend(); axes[0].set_title('RFM 세그먼트')

seg_stats = rfm.groupby('segment').agg({'R':'mean','F':'mean','M':'mean','fraud_rate':'mean','cid':'count'})
seg_stats.columns = ['R_mean','F_mean','M_mean','Fraud_Rate','Count']
print(seg_stats.round(3))
seg_stats['Fraud_Rate'].plot(kind='bar', ax=axes[1], title='세그먼트별 이상 거래 비율')
plt.tight_layout(); plt.show()
""",
                "interpretation": "RFM 세분화로 고빈도/고금액 고객(VIP)과 저활동 고객을 구분. 세그먼트별 이상 거래 비율 차이는 맞춤형 모니터링 전략 수립에 활용.",
            },
            {
                "approach": "다변량 이상 거래 탐지 앙상블",
                "code": r"""import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
from scipy.spatial.distance import mahalanobis
from scipy import stats as sp_stats
from datetime import datetime, timedelta

np.random.seed(42)
n=20000; nc=500; base=datetime(2025,1,1)
cids = np.random.randint(1,nc+1,n)
dates = [base+timedelta(days=int(d)) for d in np.cumsum(np.random.geometric(0.3,n))][:n]
amounts = np.random.lognormal(4,1.5,n).round(2)
cats = np.random.choice(['식료품','교통','쇼핑','외식','온라인','금융','의료'],n,p=[.25,.15,.20,.15,.10,.10,.05])
fi = np.random.choice(n,int(n*0.02),replace=False); amounts[fi]*=np.random.uniform(5,20,len(fi))
df = pd.DataFrame({'date':pd.to_datetime(dates),'cid':cids,'amount':amounts,'cat':cats,'fraud':0})
df.loc[fi,'fraud']=1

# 고객별 피처
feats = df.groupby('cid').agg(
    tx_count=('amount','count'), mean_amt=('amount','mean'), max_amt=('amount','max'),
    std_amt=('amount','std'), cat_nunique=('cat','nunique'), fraud_count=('fraud','sum')
).reset_index()
feats['std_amt'] = feats['std_amt'].fillna(0)
feats['cv'] = feats['std_amt']/(feats['mean_amt']+1)
feats['max_mean_ratio'] = feats['max_amt']/(feats['mean_amt']+1)

y_true = (feats['fraud_count']>0).astype(int)
X = feats[['tx_count','mean_amt','max_amt','std_amt','cv','max_mean_ratio','cat_nunique']].values
Xs = StandardScaler().fit_transform(X)

# Mahalanobis
mu = Xs.mean(0); Si = np.linalg.pinv(np.cov(Xs.T))
D2 = np.array([mahalanobis(x,mu,Si)**2 for x in Xs])
s_mah = MinMaxScaler().fit_transform(D2.reshape(-1,1)).flatten()

# IF
iso = IsolationForest(200, contamination=0.1, random_state=42).fit(Xs)
s_if = MinMaxScaler().fit_transform((-iso.score_samples(Xs)).reshape(-1,1)).flatten()

# LOF
lof = LocalOutlierFactor(20, contamination=0.1)
lof.fit_predict(Xs)
s_lof = MinMaxScaler().fit_transform((-lof.negative_outlier_factor_).reshape(-1,1)).flatten()

# 앙상블 (평균)
s_ens = (s_mah + s_if + s_lof) / 3
pred = (s_ens > np.percentile(s_ens, 90)).astype(int)

print(classification_report(y_true, pred, target_names=['정상','이상']))
""",
                "interpretation": "고객 수준 피처 엔지니어링과 앙상블 탐지로 이상 거래 패턴을 포착. CV(변동계수)와 max/mean 비율이 핵심 피처로 작용한다.",
            },
            {
                "approach": "자동 EDA 리포트 생성기",
                "code": r"""import numpy as np, pandas as pd
from scipy import stats

class AutoEDAReport:
    def __init__(self, df):
        self.df = df; self.report = []
    def profile(self):
        self.report.append("=" * 60)
        self.report.append("자동 EDA 리포트")
        self.report.append("=" * 60)
        self.report.append(f"\n데이터 크기: {self.df.shape[0]} rows x {self.df.shape[1]} cols")
        self.report.append(f"메모리: {self.df.memory_usage(deep=True).sum()/1e6:.1f} MB")
        # 결측
        nulls = self.df.isnull().sum()
        if nulls.sum()>0:
            self.report.append("\n[결측값]")
            for c in nulls[nulls>0].index:
                self.report.append(f"  {c}: {nulls[c]} ({nulls[c]/len(self.df):.1%})")
        else:
            self.report.append("\n[결측값] 없음")
        # 수치형 프로파일
        self.report.append("\n[수치형 변수 프로파일]")
        for c in self.df.select_dtypes(include=[np.number]).columns:
            s = self.df[c]
            sk = s.skew(); ku = s.kurtosis()
            _, norm_p = stats.normaltest(s.dropna()) if len(s.dropna())>20 else (0,1)
            outlier_iqr = ((s<s.quantile(0.25)-1.5*(s.quantile(0.75)-s.quantile(0.25))) |
                           (s>s.quantile(0.75)+1.5*(s.quantile(0.75)-s.quantile(0.25)))).sum()
            self.report.append(f"  {c}: mean={s.mean():.2f}, std={s.std():.2f}, skew={sk:.2f}, "
                              f"kurtosis={ku:.2f}, outliers(IQR)={outlier_iqr}, normal_p={norm_p:.4f}")
        # 범주형
        self.report.append("\n[범주형 변수]")
        for c in self.df.select_dtypes(include=['object','category']).columns:
            n_unique = self.df[c].nunique()
            top = self.df[c].value_counts().head(3)
            self.report.append(f"  {c}: {n_unique} categories, top3={dict(top)}")
        return '\n'.join(self.report)

# 테스트
np.random.seed(42)
df_test = pd.DataFrame({
    'amount': np.random.lognormal(4,1.5,1000),
    'count': np.random.poisson(5,1000),
    'cat': np.random.choice(['A','B','C'],1000),
    'score': np.random.normal(50,10,1000),
})
df_test.loc[np.random.choice(1000,50), 'score'] = np.nan

report = AutoEDAReport(df_test)
print(report.profile())
""",
                "interpretation": "자동 EDA 리포트는 수동 탐색 시간을 줄이고 데이터 품질 문제를 체계적으로 파악하는 출발점이다. 프로덕션에서는 이를 스케줄링하여 정기 모니터링에 활용한다.",
            },
        ],
        discussion="### 금융 EDA 체크리스트\n\n1. 시간 패턴: 주기성, 추세, 이상 시점\n2. 금액 분포: Benford, 이상치, 계절성\n3. 고객 세분화: RFM + 행동 패턴\n4. 이상 탐지: 다변량 앙상블\n5. 자동화: 리포트 + 알림 파이프라인",
        filepath=os.path.join(BASE, "ch01_10_practice_financial_eda_solution.ipynb"),
    )


# ============================================================
# Main: Generate all 20 notebooks
# ============================================================
if __name__ == "__main__":
    generators = [
        gen_topic_01, gen_topic_02, gen_topic_03, gen_topic_04, gen_topic_05,
        gen_topic_06, gen_topic_07, gen_topic_08, gen_topic_09, gen_topic_10,
    ]
    topics = [
        "다변량 데이터 시각화와 차원 해석",
        "결측값 메커니즘과 다중 대치법",
        "이상치 탐지 - 통계적 방법",
        "이상치 탐지 - 기계학습 기반",
        "데이터 변환과 정규화 이론",
        "범주형 인코딩 고급 기법",
        "피처 엔지니어링과 도메인 지식",
        "데이터 품질 프레임워크",
        "대용량 데이터 처리",
        "실전: 금융 거래 데이터 종합 분석",
    ]

    print("=" * 60)
    print("Chapter 01: 고급 탐색적 데이터 분석 - 노트북 생성")
    print("=" * 60)

    for i, (gen_fn, topic) in enumerate(zip(generators, topics), 1):
        print(f"\n[{i:02d}/10] {topic}...")
        gen_fn()
        print(f"  -> 문제 + 모범답안 노트북 생성 완료")

    print("\n" + "=" * 60)
    print(f"총 20개 노트북 생성 완료!")
    print("=" * 60)

    # 생성된 파일 목록 출력
    import glob
    files = sorted(glob.glob(os.path.join(BASE, "*.ipynb")))
    print(f"\n생성된 파일 ({len(files)}개):")
    for f in files:
        size = os.path.getsize(f)
        print(f"  {os.path.basename(f):55s} ({size/1024:.1f} KB)")
