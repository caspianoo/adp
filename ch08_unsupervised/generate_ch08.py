"""Chapter 08 - Unsupervised Learning & Dimensionality Reduction: 전체 노트북 생성"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.notebook_generator import problem_notebook, solution_notebook

OUT = os.path.dirname(__file__)
CH = 8

def gen_all():
    # ── 1. PCA ──
    problem_notebook(CH, 1,
        "PCA: 스펙트럼 정리와 최적성",
        [
            "고유값 분해와 스펙트럼 정리의 관계를 이해한다",
            "PCA의 최적성(분산 최대화, 재구성 오차 최소화)을 수학적으로 증명한다",
            "설명 분산(explained variance)과 바이플롯을 해석한다",
            "확률적 PCA(PPCA)의 EM 알고리즘을 이해한다",
        ],
        r"""### 1. 스펙트럼 정리 (Spectral Theorem)

실대칭 행렬 $A \in \mathbb{R}^{n \times n}$에 대해:

$$A = Q \Lambda Q^\top = \sum_{i=1}^{n} \lambda_i \mathbf{q}_i \mathbf{q}_i^\top$$

여기서 $Q$는 직교 행렬($Q^\top Q = I$), $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$이다.

### 2. PCA의 최적화 문제

공분산 행렬 $\Sigma = \frac{1}{n-1}X^\top X$ (중심화된 데이터)에 대해:

$$\max_{\|\mathbf{w}\|=1} \mathbf{w}^\top \Sigma \mathbf{w}$$

**라그랑주 승수법**:
$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^\top \Sigma \mathbf{w} - \lambda(\mathbf{w}^\top \mathbf{w} - 1)$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2\Sigma \mathbf{w} - 2\lambda \mathbf{w} = 0 \implies \Sigma \mathbf{w} = \lambda \mathbf{w}$$

따라서 최적 $\mathbf{w}$는 $\Sigma$의 **최대 고유값에 대응하는 고유벡터**이다.

### 3. 설명 분산 비율

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{i=1}^p \lambda_i}$$

누적 설명 분산: $\text{CEVR}_k = \sum_{i=1}^k \text{EVR}_i$

### 4. 확률적 PCA (Probabilistic PCA)

생성 모델: $\mathbf{x} = W\mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\epsilon}$

- $\mathbf{z} \sim \mathcal{N}(0, I_q)$ — 잠재 변수
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 I_p)$ — 노이즈

$$p(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, WW^\top + \sigma^2 I)$$

MLE 해: $W_{ML} = U_q (\Lambda_q - \sigma^2 I)^{1/2} R$

여기서 $U_q$는 상위 $q$개 고유벡터, $R$은 임의 직교행렬이다.
""",
        """# PCA 구현 가이드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 & 표준화
iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

# 2. 고유값 분해로 직접 PCA 구현
cov_matrix = np.cov(X.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 내림차순 정렬
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("고유값:", eigenvalues)
print("설명 분산 비율:", eigenvalues / eigenvalues.sum())

# 3. sklearn PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"설명 분산 비율: {pca.explained_variance_ratio_}")
print(f"누적 분산: {np.cumsum(pca.explained_variance_ratio_)}")

# 4. 바이플롯
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)

feature_names = iris.feature_names
for i, name in enumerate(feature_names):
    ax.arrow(0, 0, pca.components_[0, i]*3, pca.components_[1, i]*3,
             head_width=0.1, head_length=0.05, fc='red', ec='red')
    ax.text(pca.components_[0, i]*3.3, pca.components_[1, i]*3.3, name, fontsize=10, color='red')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA 바이플롯 — Iris 데이터')
plt.colorbar(scatter)
plt.tight_layout()
plt.show()""",
        [
            {"description": "Iris 데이터에 대해 공분산 행렬을 직접 계산하고, 고유값 분해를 수행하여 sklearn PCA 결과와 비교하세요.\n\n(a) 고유값과 `pca.explained_variance_`의 관계를 설명하세요.\n(b) 고유벡터와 `pca.components_`의 관계를 확인하세요.", "difficulty": "★", "skeleton": "# 여기에 코드를 작성하세요\nfrom sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.datasets import load_iris\nimport numpy as np\n\niris = load_iris()\nX = StandardScaler().fit_transform(iris.data)\n\n# (a) 공분산 행렬 & 고유값 분해\ncov_mat = # TODO\neigenvalues, eigenvectors = # TODO\n\n# (b) sklearn PCA\npca = PCA(n_components=4)\npca.fit(X)\n\n# 비교\nprint('직접 계산 고유값:', ___)\nprint('sklearn explained_variance_:', ___)"},
            {"description": "스크리 플롯(Scree Plot)과 누적 분산 그래프를 그리세요. 적절한 주성분 수를 결정하는 **Kaiser 기준**(고유값 > 1)과 **누적 분산 90% 기준**을 비교하세요.", "difficulty": "★★", "hint": "표준화된 데이터의 공분산은 상관행렬이므로 고유값 평균이 1입니다."},
            {"description": "확률적 PCA(PPCA)를 직접 구현하세요.\n\n$$W_{ML} = U_q(\\Lambda_q - \\sigma^2 I)^{1/2}$$\n\n여기서 $\\sigma^2 = \\frac{1}{p-q}\\sum_{i=q+1}^{p}\\lambda_i$\n\nsklearn의 `PCA(svd_solver='full')`과 결과를 비교하세요.", "difficulty": "★★★", "skeleton": "# PPCA 직접 구현\ndef ppca(X, q):\n    \"\"\"확률적 PCA\"\"\"\n    n, p = X.shape\n    mu = X.mean(axis=0)\n    X_centered = X - mu\n    \n    # 공분산 행렬\n    S = np.cov(X_centered.T)\n    eigenvalues, eigenvectors = np.linalg.eigh(S)\n    \n    # 내림차순 정렬\n    idx = np.argsort(eigenvalues)[::-1]\n    eigenvalues = eigenvalues[idx]\n    eigenvectors = eigenvectors[:, idx]\n    \n    # sigma^2 추정\n    sigma2 = # TODO\n    \n    # W_ML 계산\n    W = # TODO\n    \n    return W, mu, sigma2"},
            {"description": "고차원 데이터(digits 데이터셋, 64차원)에 대해 PCA를 적용하고:\n\n(a) 95% 분산을 설명하는 데 필요한 주성분 수를 구하세요.\n(b) 원본 이미지와 재구성 이미지를 비교하세요.\n(c) 재구성 오차를 주성분 수에 따라 시각화하세요.", "difficulty": "★★★", "hint": "`pca.inverse_transform()`으로 재구성할 수 있습니다."},
        ],
        [
            "Jolliffe, I.T. (2002). Principal Component Analysis, Springer",
            "Tipping, M.E. & Bishop, C.M. (1999). Probabilistic Principal Component Analysis",
            "Bishop, C.M. (2006). Pattern Recognition and Machine Learning, Ch.12",
        ],
        os.path.join(OUT, "ch08_01_pca_problem.ipynb")
    )

    solution_notebook(CH, 1,
        "PCA: 스펙트럼 정리와 최적성",
        [
            {"approach": "### 문제 1: 고유값 분해 vs sklearn PCA 비교\n\n공분산 행렬의 고유값 분해와 sklearn PCA의 관계를 확인합니다.",
             "code": """import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)

# (a) 공분산 행렬 & 고유값 분해
cov_mat = np.cov(X.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

# 내림차순 정렬
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# (b) sklearn PCA
pca = PCA(n_components=4)
pca.fit(X)

print("=== 고유값 비교 ===")
print(f"직접 계산 고유값:        {eigenvalues}")
print(f"sklearn explained_var:   {pca.explained_variance_}")
print(f"보정 관계 (n-1)/n:       {eigenvalues * len(X)/(len(X)-1)}")
print()
print("=== 고유벡터 비교 ===")
print(f"직접 계산 (첫 번째):\\n{eigenvectors[:, 0]}")
print(f"sklearn (첫 번째):\\n{pca.components_[0]}")
print("부호 차이를 고려한 일치 확인:", np.allclose(np.abs(eigenvectors[:, 0]), np.abs(pca.components_[0])))""",
             "interpretation": "sklearn PCA의 `explained_variance_`는 `n-1`로 나눈 불편 분산이므로 직접 계산한 고유값과 동일합니다(numpy의 `np.cov`도 `n-1`로 나눔). 고유벡터는 부호가 다를 수 있으나 방향은 동일합니다."},
            {"approach": "### 문제 2: 스크리 플롯과 주성분 수 결정",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)

pca = PCA().fit(X)
eigenvalues = pca.explained_variance_
evr = pca.explained_variance_ratio_
cumulative = np.cumsum(evr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 스크리 플롯
axes[0].bar(range(1, len(eigenvalues)+1), eigenvalues, alpha=0.7, label='고유값')
axes[0].axhline(y=1.0, color='r', linestyle='--', label='Kaiser 기준 (λ=1)')
axes[0].set_xlabel('주성분 번호')
axes[0].set_ylabel('고유값')
axes[0].set_title('스크리 플롯')
axes[0].legend()

# 누적 분산
axes[1].plot(range(1, len(cumulative)+1), cumulative, 'bo-', label='누적 분산')
axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% 기준')
axes[1].set_xlabel('주성분 수')
axes[1].set_ylabel('누적 설명 분산 비율')
axes[1].set_title('누적 설명 분산')
axes[1].legend()

plt.tight_layout()
plt.show()

# Kaiser 기준
kaiser_n = np.sum(eigenvalues > 1.0)
# 90% 기준
var90_n = np.argmax(cumulative >= 0.9) + 1
print(f"Kaiser 기준: {kaiser_n}개 주성분")
print(f"90% 분산 기준: {var90_n}개 주성분")""",
             "interpretation": "Kaiser 기준은 고유값이 1보다 큰 주성분만 유지합니다(표준화 데이터 기준). 90% 누적 분산 기준은 더 보수적입니다. Iris 데이터에서는 2개 주성분으로도 약 96%의 분산을 설명합니다."},
            {"approach": "### 문제 3: 확률적 PCA 직접 구현",
             "code": """import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)

def ppca(X, q):
    n, p = X.shape
    mu = X.mean(axis=0)
    X_centered = X - mu

    S = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # sigma^2: 버려진 차원의 평균 분산
    sigma2 = np.mean(eigenvalues[q:])

    # W_ML = U_q (Lambda_q - sigma^2 I)^{1/2}
    Lambda_q = np.diag(eigenvalues[:q])
    U_q = eigenvectors[:, :q]
    W = U_q @ np.sqrt(Lambda_q - sigma2 * np.eye(q))

    return W, mu, sigma2

W, mu, sigma2 = ppca(X, q=2)
print(f"노이즈 분산 sigma^2 = {sigma2:.4f}")
print(f"W 행렬 shape: {W.shape}")
print(f"W:\\n{W}")

# sklearn PCA와 비교
pca = PCA(n_components=2)
pca.fit(X)
print(f"\\nsklearn noise_variance_: {pca.noise_variance_:.4f}")
print(f"PPCA sigma^2:            {sigma2:.4f}")""",
             "interpretation": "PPCA는 노이즈 분산 $\\sigma^2$를 명시적으로 모델링합니다. 이 값은 버려진 주성분의 평균 고유값으로 추정됩니다. sklearn의 `noise_variance_`와 일치함을 확인할 수 있습니다."},
            {"approach": "### 문제 4: 고차원 데이터 PCA (digits)",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

# (a) 95% 분산 설명에 필요한 주성분 수
pca_full = PCA().fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.argmax(cumvar >= 0.95) + 1
print(f"95% 분산 설명에 필요한 주성분 수: {n_95} (전체 {X.shape[1]}차원)")

# (b) 재구성 비교
fig, axes = plt.subplots(3, 8, figsize=(16, 6))
for k_idx, n_comp in enumerate([5, 20, n_95]):
    pca_k = PCA(n_components=n_comp)
    X_reduced = pca_k.fit_transform(X)
    X_reconstructed = pca_k.inverse_transform(X_reduced)

    for i in range(4):
        axes[k_idx, i*2].imshow(X[i].reshape(8, 8), cmap='gray')
        axes[k_idx, i*2].set_title('원본' if k_idx == 0 else '')
        axes[k_idx, i*2].axis('off')

        axes[k_idx, i*2+1].imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
        axes[k_idx, i*2+1].set_title(f'PC={n_comp}' if i == 0 else '')
        axes[k_idx, i*2+1].axis('off')
plt.suptitle('원본 vs 재구성 이미지')
plt.tight_layout()
plt.show()

# (c) 재구성 오차
errors = []
n_range = range(1, 65, 2)
for n_comp in n_range:
    pca_k = PCA(n_components=n_comp)
    X_r = pca_k.inverse_transform(pca_k.fit_transform(X))
    mse = np.mean((X - X_r) ** 2)
    errors.append(mse)

plt.figure(figsize=(10, 5))
plt.plot(list(n_range), errors, 'bo-')
plt.axvline(x=n_95, color='r', linestyle='--', label=f'95% 분산 기준 ({n_95}개)')
plt.xlabel('주성분 수')
plt.ylabel('MSE (재구성 오차)')
plt.title('주성분 수에 따른 재구성 오차')
plt.legend()
plt.tight_layout()
plt.show()""",
             "interpretation": "64차원 digits 데이터를 약 28~30개 주성분으로 95% 분산을 설명할 수 있습니다. 주성분 수가 증가하면 재구성 오차가 단조 감소하며, 20개 이상이면 시각적으로 원본과 거의 구분 불가합니다."},
        ],
        "PCA는 선형 차원 축소의 기본이지만 **비선형 구조**를 포착하지 못합니다. 이를 확장한 커널 PCA(다음 절), 오토인코더 등이 있습니다. 또한 데이터가 가우시안 분포를 따르지 않으면 ICA(독립 성분 분석)가 더 적합할 수 있습니다.",
        os.path.join(OUT, "ch08_01_pca_solution.ipynb")
    )

    # ── 2. 커널 PCA ──
    problem_notebook(CH, 2,
        "커널 PCA와 비선형 확장",
        [
            "커널 트릭의 수학적 원리를 이해한다",
            "RBF 커널 PCA의 고유값 문제를 유도한다",
            "사전 이미지(pre-image) 문제를 이해한다",
            "커널 선택과 하이퍼파라미터 튜닝을 수행한다",
        ],
        r"""### 1. 커널 트릭 (Kernel Trick)

특성 맵 $\phi: \mathbb{R}^d \to \mathcal{F}$에 대해 커널 함수:

$$k(\mathbf{x}, \mathbf{y}) = \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle_\mathcal{F}$$

**Mercer 정리**: 양의 정부호(positive definite) 커널은 항상 내적으로 표현 가능

### 2. 대표 커널

| 커널 | 수식 |
|------|------|
| 선형 | $k(\mathbf{x}, \mathbf{y}) = \mathbf{x}^\top \mathbf{y}$ |
| 다항 | $k(\mathbf{x}, \mathbf{y}) = (\gamma \mathbf{x}^\top \mathbf{y} + r)^d$ |
| RBF | $k(\mathbf{x}, \mathbf{y}) = \exp(-\gamma \|\mathbf{x} - \mathbf{y}\|^2)$ |

### 3. 커널 PCA 유도

특성 공간에서의 공분산 행렬:
$$C = \frac{1}{n}\sum_{i=1}^n \phi(\mathbf{x}_i)\phi(\mathbf{x}_i)^\top$$

고유값 문제 $C\mathbf{v} = \lambda \mathbf{v}$를 커널화:

$$K \boldsymbol{\alpha} = n\lambda \boldsymbol{\alpha}$$

여기서 $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$이고 $\mathbf{v} = \sum_i \alpha_i \phi(\mathbf{x}_i)$

### 4. 커널 행렬의 중심화

$$\tilde{K} = K - \mathbf{1}_n K - K \mathbf{1}_n + \mathbf{1}_n K \mathbf{1}_n$$

여기서 $(\mathbf{1}_n)_{ij} = 1/n$

### 5. 사전 이미지 문제

$\phi(\mathbf{x}^*) \approx \sum_i \alpha_i \phi(\mathbf{x}_i)$로부터 $\mathbf{x}^*$를 복원하는 것은 일반적으로 **비볼록 최적화** 문제이다.
""",
        """# 커널 PCA 구현 가이드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles, make_moons

# 1. 비선형 데이터 생성
X_circles, y_circles = make_circles(n_samples=400, factor=0.3, noise=0.05)
X_moons, y_moons = make_moons(n_samples=400, noise=0.1)

# 2. 일반 PCA vs 커널 PCA
from sklearn.decomposition import PCA

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for row, (X, y, name) in enumerate([(X_circles, y_circles, '동심원'), (X_moons, y_moons, '반달')]):
    axes[row, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=20)
    axes[row, 0].set_title(f'{name} — 원본')

    pca_linear = PCA(n_components=2).fit_transform(X)
    axes[row, 1].scatter(pca_linear[:, 0], pca_linear[:, 1], c=y, cmap='viridis', s=20)
    axes[row, 1].set_title(f'{name} — 선형 PCA')

    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
    X_kpca = kpca.fit_transform(X)
    axes[row, 2].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', s=20)
    axes[row, 2].set_title(f'{name} — 커널 PCA (RBF)')

plt.tight_layout()
plt.show()""",
        [
            {"description": "RBF 커널 행렬을 직접 계산하고, 중심화한 후 고유값 분해를 수행하여 `sklearn.decomposition.KernelPCA`와 결과가 일치하는지 확인하세요.\n\n$$\\tilde{K} = K - \\mathbf{1}_n K - K \\mathbf{1}_n + \\mathbf{1}_n K \\mathbf{1}_n$$", "difficulty": "★★", "skeleton": "import numpy as np\nfrom sklearn.metrics.pairwise import rbf_kernel\nfrom sklearn.datasets import make_moons\n\nX, y = make_moons(n_samples=200, noise=0.1, random_state=42)\ngamma = 10\n\n# 1. 커널 행렬 계산\nK = # TODO\n\n# 2. 중심화\nn = len(X)\none_n = np.ones((n, n)) / n\nK_centered = # TODO\n\n# 3. 고유값 분해\neigenvalues, eigenvectors = # TODO\n\n# 4. 프로젝션\n# alpha = eigenvectors / sqrt(eigenvalues)\n"},
            {"description": "gamma 값에 따른 커널 PCA 결과 변화를 시각화하세요. gamma ∈ {0.1, 1, 5, 10, 50}에 대해 make_circles 데이터의 2D 투영을 비교하세요.", "difficulty": "★★"},
            {"description": "커널 PCA에서 `fit_inverse_transform=True` 옵션을 사용하여 사전 이미지를 복원하세요. 원본 데이터와 복원 데이터의 MSE를 계산하세요.", "difficulty": "★★★", "hint": "KernelPCA(fit_inverse_transform=True)로 설정하면 inverse_transform이 가능합니다."},
        ],
        [
            "Schölkopf, B., Smola, A., & Müller, K.R. (1998). Nonlinear Component Analysis as a Kernel Eigenvalue Problem",
            "Mika, S. et al. (1999). Kernel PCA and De-noising in Feature Spaces",
        ],
        os.path.join(OUT, "ch08_02_kernel_pca_problem.ipynb")
    )

    solution_notebook(CH, 2,
        "커널 PCA와 비선형 확장",
        [
            {"approach": "### 문제 1: 커널 PCA 직접 구현",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA

X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
gamma = 10

# 1. 커널 행렬 계산
K = rbf_kernel(X, gamma=gamma)

# 2. 중심화
n = len(X)
one_n = np.ones((n, n)) / n
K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

# 3. 고유값 분해
eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 4. 프로젝션 (상위 2개)
alphas = eigenvectors[:, :2] / np.sqrt(eigenvalues[:2])
X_kpca_manual = K_centered @ alphas

# 5. sklearn 비교
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
X_kpca_sklearn = kpca.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_kpca_manual[:, 0], X_kpca_manual[:, 1], c=y, cmap='viridis', s=20)
axes[0].set_title('직접 구현')
axes[1].scatter(X_kpca_sklearn[:, 0], X_kpca_sklearn[:, 1], c=y, cmap='viridis', s=20)
axes[1].set_title('sklearn KernelPCA')
plt.tight_layout()
plt.show()

print("결과 일치 (부호 고려):", np.allclose(np.abs(X_kpca_manual), np.abs(X_kpca_sklearn), atol=1e-6))""",
             "interpretation": "직접 구현한 결과와 sklearn 결과가 부호를 제외하면 동일함을 확인할 수 있습니다. 고유벡터의 부호는 임의적이므로 방향만 일치하면 됩니다."},
            {"approach": "### 문제 2: gamma에 따른 결과 변화",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=42)
gammas = [0.1, 1, 5, 10, 50]

fig, axes = plt.subplots(1, len(gammas), figsize=(20, 4))
for i, g in enumerate(gammas):
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=g)
    X_kpca = kpca.fit_transform(X)
    axes[i].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', s=10)
    axes[i].set_title(f'γ = {g}')
plt.suptitle('gamma에 따른 커널 PCA 결과', fontsize=14)
plt.tight_layout()
plt.show()""",
             "interpretation": "gamma가 너무 작으면 선형 PCA와 유사하고, 너무 크면 과적합되어 구조가 사라집니다. 적절한 gamma (5~10)에서 두 클래스가 선형 분리됩니다."},
            {"approach": "### 문제 3: 사전 이미지 복원",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10, fit_inverse_transform=True)
X_kpca = kpca.fit_transform(X)
X_reconstructed = kpca.inverse_transform(X_kpca)

mse = np.mean((X - X_reconstructed) ** 2)
print(f"재구성 MSE: {mse:.6f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=20)
axes[0].set_title('원본 데이터')
axes[1].scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], c=y, cmap='viridis', s=20)
axes[1].set_title(f'사전 이미지 복원 (MSE={mse:.4f})')
plt.tight_layout()
plt.show()""",
             "interpretation": "사전 이미지 복원은 커널 릿지 회귀를 사용하여 특성 공간의 점을 원래 입력 공간으로 매핑합니다. MSE가 작을수록 좋은 복원이며, 커널/gamma 선택에 따라 품질이 달라집니다."},
        ],
        "커널 PCA는 비선형 구조를 포착하지만 **커널 행렬의 크기가 $O(n^2)$**이므로 대규모 데이터에는 비효율적입니다. Nyström 근사나 Random Fourier Features로 확장할 수 있습니다.",
        os.path.join(OUT, "ch08_02_kernel_pca_solution.ipynb")
    )

    # ── 3. t-SNE ──
    problem_notebook(CH, 3,
        "t-SNE 이론과 실전",
        [
            "t-SNE의 KL 발산 기반 목적 함수를 이해한다",
            "perplexity 파라미터의 의미와 효과를 이해한다",
            "Barnes-Hut 근사의 원리를 이해한다",
            "t-SNE의 올바른 해석법과 한계를 인지한다",
        ],
        r"""### 1. t-SNE 목적 함수

**고차원 유사도** (가우시안):
$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}, \quad p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**저차원 유사도** (Student-t, 자유도 1):
$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

**KL 발산 최소화**:
$$C = KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### 2. Perplexity

$$\text{Perp}(P_i) = 2^{H(P_i)}, \quad H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$$

perplexity ≈ 유효 이웃 수. 일반적으로 5~50 범위. $\sigma_i$는 이진 탐색으로 결정.

### 3. 기울기

$$\frac{\partial C}{\partial \mathbf{y}_i} = 4\sum_j (p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$$

### 4. Barnes-Hut 근사

- 시간 복잡도: $O(n^2) \to O(n \log n)$
- 쿼드트리를 이용한 원거리 점 그룹핑
""",
        """# t-SNE 실습 가이드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# 데이터 로드
digits = load_digits()
X, y = digits.data, digits.target

# t-SNE 적용
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter, ticks=range(10))
plt.title('t-SNE — Digits 데이터셋')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.show()
print(f"KL divergence: {tsne.kl_divergence_:.4f}")""",
        [
            {"description": "perplexity 값을 {5, 15, 30, 50, 100}으로 변화시키며 digits 데이터의 t-SNE 결과를 비교하세요. 각 결과에 KL 발산 값을 표시하세요.", "difficulty": "★★"},
            {"description": "t-SNE 최적화 과정을 시각화하세요. `n_iter`를 {50, 150, 300, 500, 1000}으로 변화시키며 결과를 비교하세요.", "difficulty": "★★"},
            {"description": "t-SNE와 PCA의 결과를 정량적으로 비교하세요.\n\n(a) k-NN 분류 정확도로 임베딩 품질 비교\n(b) Trustworthiness 지표 계산\n(c) 각 방법의 장단점 정리", "difficulty": "★★★", "hint": "sklearn.manifold.trustworthiness를 사용하세요."},
        ],
        [
            "van der Maaten, L. & Hinton, G. (2008). Visualizing Data using t-SNE",
            "Wattenberg, M. et al. (2016). How to Use t-SNE Effectively (Distill)",
        ],
        os.path.join(OUT, "ch08_03_tsne_problem.ipynb")
    )

    solution_notebook(CH, 3,
        "t-SNE 이론과 실전",
        [
            {"approach": "### 문제 1: Perplexity 비교",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

perplexities = [5, 15, 30, 50, 100]
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for i, perp in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
    axes[i].set_title(f'perp={perp}\\nKL={tsne.kl_divergence_:.3f}')
    axes[i].set_xticks([]); axes[i].set_yticks([])
plt.suptitle('Perplexity에 따른 t-SNE 결과', fontsize=14)
plt.tight_layout()
plt.show()""",
             "interpretation": "perplexity가 작으면 국소 구조에 집중하여 작은 클러스터가 많이 형성되고, 크면 전역 구조를 반영하지만 세부 구조가 흐려집니다. 30이 일반적으로 좋은 기본값입니다."},
            {"approach": "### 문제 2: 최적화 과정 시각화",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

iters = [50, 150, 300, 500, 1000]
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for i, n_iter in enumerate(iters):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=n_iter)
    X_tsne = tsne.fit_transform(X)
    axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
    axes[i].set_title(f'n_iter={n_iter}\\nKL={tsne.kl_divergence_:.3f}')
    axes[i].set_xticks([]); axes[i].set_yticks([])
plt.suptitle('반복 횟수에 따른 t-SNE 수렴', fontsize=14)
plt.tight_layout()
plt.show()""",
             "interpretation": "초기에는 exaggeration 단계에서 큰 클러스터가 형성되고, 반복이 진행될수록 세부 구조가 정교해집니다. 보통 500-1000회 반복이면 충분합니다."},
            {"approach": "### 문제 3: t-SNE vs PCA 정량 비교",
             "code": """import numpy as np
from sklearn.manifold import TSNE, trustworthiness
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

digits = load_digits()
X, y = digits.data, digits.target

# 임베딩
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# (a) k-NN 분류 정확도
for name, X_emb in [('PCA', X_pca), ('t-SNE', X_tsne)]:
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X_emb, y, cv=5)
    print(f"{name} — k-NN 정확도: {scores.mean():.4f} ± {scores.std():.4f}")

# (b) Trustworthiness
for name, X_emb in [('PCA', X_pca), ('t-SNE', X_tsne)]:
    tw = trustworthiness(X, X_emb, n_neighbors=5)
    print(f"{name} — Trustworthiness: {tw:.4f}")

# (c) 장단점 정리
print("\\n=== 장단점 비교 ===")
print("PCA:   선형, 빠름, 해석 가능, 전역 구조 보존")
print("t-SNE: 비선형, 느림, 해석 제한, 국소 구조 보존")""",
             "interpretation": "t-SNE는 국소 이웃 관계를 잘 보존하여 시각화에 탁월하지만, 전역 거리 관계가 왜곡됩니다. PCA는 전역 분산을 최대화하지만 비선형 구조를 놓칩니다. Trustworthiness가 높을수록 이웃 관계가 잘 보존됨을 의미합니다."},
        ],
        "t-SNE는 **시각화 전용** 도구이며, 축 값이나 클러스터 간 거리에 의미를 부여하면 안 됩니다. 다른 실행에서 결과가 달라질 수 있으므로 반드시 `random_state`를 고정하세요. 최근에는 UMAP이 더 빠르고 전역 구조 보존이 나은 대안으로 주목받고 있습니다.",
        os.path.join(OUT, "ch08_03_tsne_solution.ipynb")
    )

    # ── 4. UMAP ──
    problem_notebook(CH, 4,
        "UMAP: 위상적 관점",
        [
            "UMAP의 리만 다양체 가정을 이해한다",
            "퍼지 단체 집합(fuzzy simplicial set) 개념을 이해한다",
            "UMAP과 t-SNE의 차이점을 비교한다",
            "UMAP의 주요 하이퍼파라미터를 튜닝한다",
        ],
        r"""### 1. UMAP의 이론적 기반

**가정**: 데이터가 리만 다양체(Riemannian manifold)에 균일하게 분포

**국소 거리**: 각 점 $x_i$에서의 리만 메트릭을 근사:
$$d_i(x_j) = \max(0, d(x_i, x_j) - \rho_i) / \sigma_i$$

여기서 $\rho_i$는 최근접 이웃까지의 거리

### 2. 퍼지 단체 집합 (Fuzzy Simplicial Set)

**고차원 그래프 가중치**:
$$w_{ij}^{(h)} = \exp\left(-\frac{d_i(x_j)}{\sigma_i}\right)$$

**대칭화**:
$$w_{ij} = w_{ij}^{(h)} + w_{ji}^{(h)} - w_{ij}^{(h)} \cdot w_{ji}^{(h)}$$

### 3. 저차원 유사도

$$w_{ij}^{(l)} = \left(1 + a \|\mathbf{y}_i - \mathbf{y}_j\|^{2b}\right)^{-1}$$

기본값: $a \approx 1.929, b \approx 0.7915$ (min_dist=0.1에 대해)

### 4. 교차 엔트로피 최소화

$$C = \sum_{i \neq j} \left[ w_{ij} \log\frac{w_{ij}}{w_{ij}^{(l)}} + (1-w_{ij})\log\frac{1-w_{ij}}{1-w_{ij}^{(l)}} \right]$$

이는 t-SNE의 KL 발산과 달리 **반발력(repulsive force)** 항을 포함한다.

### 5. UMAP vs t-SNE 비교

| 특성 | t-SNE | UMAP |
|------|-------|------|
| 목적 함수 | KL 발산 | 교차 엔트로피 |
| 전역 구조 | 약함 | 상대적으로 보존 |
| 속도 | $O(n\log n)$ | $O(n)$ (SGD) |
| 새 데이터 | 불가 | transform 가능 |
""",
        """# UMAP 실습 가이드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 주의: umap-learn 패키지 필요
# pip install umap-learn
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn 미설치. pip install umap-learn으로 설치하세요.")

digits = load_digits()
X, y = digits.data, digits.target

if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.title('UMAP — Digits 데이터셋')
    plt.tight_layout()
    plt.show()
else:
    # UMAP 없이 t-SNE로 대체
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
    plt.title('t-SNE (UMAP 미설치)')
    plt.tight_layout()
    plt.show()""",
        [
            {"description": "UMAP의 `n_neighbors`와 `min_dist` 파라미터 조합을 체계적으로 비교하세요.\n\n- n_neighbors ∈ {5, 15, 50, 200}\n- min_dist ∈ {0.0, 0.1, 0.5, 0.99}\n\n4×4 그리드로 시각화하세요.", "difficulty": "★★"},
            {"description": "UMAP과 t-SNE의 결과를 같은 데이터에 대해 비교하세요.\n\n(a) 실행 시간 비교\n(b) k-NN 정확도 비교\n(c) Trustworthiness 비교", "difficulty": "★★"},
            {"description": "UMAP의 `transform()` 기능을 활용하여 새로운 데이터를 기존 임베딩 공간에 투영하세요. 학습 데이터의 80%로 UMAP을 학습하고, 나머지 20%를 transform하세요.", "difficulty": "★★★"},
        ],
        [
            "McInnes, L. et al. (2018). UMAP: Uniform Manifold Approximation and Projection",
            "McInnes, L. (2018). How UMAP Works (umap-learn docs)",
        ],
        os.path.join(OUT, "ch08_04_umap_problem.ipynb")
    )

    solution_notebook(CH, 4,
        "UMAP: 위상적 관점",
        [
            {"approach": "### 문제 1: n_neighbors와 min_dist 비교",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

n_neighbors_list = [5, 15, 50, 200]
min_dist_list = [0.0, 0.1, 0.5, 0.99]

if HAS_UMAP:
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    for i, nn in enumerate(n_neighbors_list):
        for j, md in enumerate(min_dist_list):
            reducer = umap.UMAP(n_neighbors=nn, min_dist=md, random_state=42)
            X_umap = reducer.fit_transform(X)
            axes[i, j].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=3)
            axes[i, j].set_title(f'nn={nn}, md={md}', fontsize=10)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    plt.suptitle('UMAP: n_neighbors × min_dist', fontsize=16)
    plt.tight_layout()
    plt.show()
else:
    print("umap-learn 미설치. pip install umap-learn으로 설치하세요.")""",
             "interpretation": "n_neighbors가 크면 전역 구조가 더 잘 반영되고, 작으면 국소 구조에 집중합니다. min_dist가 작으면 점들이 밀집되고, 크면 균일하게 퍼집니다."},
            {"approach": "### 문제 2: UMAP vs t-SNE 비교",
             "code": """import numpy as np
import time
from sklearn.manifold import TSNE, trustworthiness
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

digits = load_digits()
X, y = digits.data, digits.target

# t-SNE
t0 = time.time()
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
t_tsne = time.time() - t0

results = {'t-SNE': {'embedding': X_tsne, 'time': t_tsne}}

try:
    import umap
    t0 = time.time()
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)
    t_umap = time.time() - t0
    results['UMAP'] = {'embedding': X_umap, 'time': t_umap}
except ImportError:
    print("UMAP 미설치")

for name, res in results.items():
    X_emb = res['embedding']
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X_emb, y, cv=5)
    tw = trustworthiness(X, X_emb, n_neighbors=5)
    print(f"{name}: 시간={res['time']:.2f}s, k-NN 정확도={scores.mean():.4f}, Trustworthiness={tw:.4f}")""",
             "interpretation": "UMAP은 일반적으로 t-SNE보다 빠르면서 비슷하거나 더 나은 임베딩 품질을 보입니다. 특히 전역 구조 보존에서 우위를 보이며, transform 기능으로 새 데이터 투영이 가능합니다."},
            {"approach": "### 문제 3: UMAP transform 활용",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_train_umap = reducer.fit_transform(X_train)
    X_test_umap = reducer.transform(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap='tab10', s=10, alpha=0.5)
    axes[0].set_title('학습 데이터 임베딩')
    axes[1].scatter(X_train_umap[:, 0], X_train_umap[:, 1], c='lightgray', s=5, alpha=0.3)
    axes[1].scatter(X_test_umap[:, 0], X_test_umap[:, 1], c=y_test, cmap='tab10', s=30, edgecolors='black', linewidths=0.5)
    axes[1].set_title('테스트 데이터 투영')
    plt.suptitle('UMAP transform: 새 데이터 투영')
    plt.tight_layout()
    plt.show()
except ImportError:
    print("umap-learn 미설치. pip install umap-learn으로 설치하세요.")""",
             "interpretation": "UMAP은 t-SNE와 달리 학습된 매핑을 새 데이터에 적용할 수 있습니다. 이는 프로덕션 환경에서 새로운 관측치를 기존 임베딩에 배치하는 데 유용합니다."},
        ],
        "UMAP은 위상 데이터 분석(TDA)에 기반한 이론적 프레임워크를 제공하며, t-SNE보다 이론적으로 더 견고합니다. 하지만 여전히 **하이퍼파라미터에 민감**하며, 결과 해석에 주의가 필요합니다.",
        os.path.join(OUT, "ch08_04_umap_solution.ipynb")
    )

    # ── 5. NMF ──
    problem_notebook(CH, 5,
        "비음수 행렬 분해 (NMF)",
        [
            "NMF의 수학적 정의와 곱셈 업데이트 규칙을 유도한다",
            "스파스 NMF와 정규화의 효과를 이해한다",
            "NMF와 토픽 모델링의 관계를 이해한다",
            "NMF의 초기화 전략(NNDSVD)을 이해한다",
        ],
        r"""### 1. NMF 문제 정의

비음수 행렬 $V \in \mathbb{R}_+^{m \times n}$에 대해:

$$V \approx WH, \quad W \in \mathbb{R}_+^{m \times r}, \; H \in \mathbb{R}_+^{r \times n}$$

- $W$: 기저 행렬 (부분-기반 표현)
- $H$: 계수 행렬 (각 기저의 가중치)
- $r$: 잠재 차원 (랭크)

### 2. 목적 함수

**프로베니우스 노름**:
$$\min_{W,H \geq 0} \|V - WH\|_F^2 = \sum_{ij}(V_{ij} - (WH)_{ij})^2$$

**일반화 KL 발산**:
$$D_{KL}(V \| WH) = \sum_{ij}\left[V_{ij}\log\frac{V_{ij}}{(WH)_{ij}} - V_{ij} + (WH)_{ij}\right]$$

### 3. 곱셈 업데이트 규칙 (Lee & Seung)

**프로베니우스 노름 기반**:
$$H \leftarrow H \odot \frac{W^\top V}{W^\top W H}, \quad W \leftarrow W \odot \frac{V H^\top}{W H H^\top}$$

**수렴 증명**: 보조 함수(auxiliary function)를 통한 비증가 보장

### 4. 스파스 NMF

$$\min_{W,H \geq 0} \|V - WH\|_F^2 + \alpha \|H\|_1 + \beta \|W\|_F^2$$

L1 정규화 → 스파스한 계수 행렬 → 해석 용이
""",
        """# NMF 구현 가이드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_olivetti_faces

# 얼굴 데이터 로드
faces = fetch_olivetti_faces()
X = faces.data  # (400, 4096), 값 범위 [0, 1]

# NMF 적용
n_components = 16
nmf = NMF(n_components=n_components, init='nndsvd', max_iter=500, random_state=42)
W = nmf.fit_transform(X)  # (400, 16)
H = nmf.components_        # (16, 4096)

# 기저 얼굴 시각화
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(H[i].reshape(64, 64), cmap='gray')
    ax.set_title(f'기저 {i+1}')
    ax.axis('off')
plt.suptitle('NMF 기저 (부분-기반 표현)', fontsize=14)
plt.tight_layout()
plt.show()

print(f"재구성 오차: {nmf.reconstruction_err_:.4f}")""",
        [
            {"description": "곱셈 업데이트 규칙을 직접 구현하여 NMF를 수행하세요.\n\n$$H \\leftarrow H \\odot \\frac{W^\\top V}{W^\\top W H + \\epsilon}$$\n$$W \\leftarrow W \\odot \\frac{V H^\\top}{W H H^\\top + \\epsilon}$$\n\nsklearn NMF와 재구성 오차를 비교하세요.", "difficulty": "★★★", "skeleton": "def nmf_multiplicative(V, r, max_iter=200, eps=1e-7):\n    m, n = V.shape\n    W = np.random.rand(m, r) + eps\n    H = np.random.rand(r, n) + eps\n    \n    errors = []\n    for iteration in range(max_iter):\n        # H 업데이트\n        H *= # TODO\n        \n        # W 업데이트\n        W *= # TODO\n        \n        err = np.linalg.norm(V - W @ H, 'fro')\n        errors.append(err)\n    \n    return W, H, errors"},
            {"description": "NMF를 텍스트 데이터에 적용하여 토픽 모델링을 수행하세요. 20 Newsgroups 데이터셋에서 5개 토픽을 추출하고, 각 토픽의 상위 10개 단어를 출력하세요.", "difficulty": "★★", "hint": "sklearn.datasets.fetch_20newsgroups와 TfidfVectorizer를 사용하세요."},
            {"description": "NMF의 컴포넌트 수(r)에 따른 재구성 오차 변화를 분석하세요. r ∈ {2, 5, 10, 20, 50}에 대해 오차를 플롯하고, 적절한 r을 결정하는 기준을 제시하세요.", "difficulty": "★★"},
        ],
        [
            "Lee, D.D. & Seung, H.S. (1999). Learning the parts of objects by NMF, Nature",
            "Lee, D.D. & Seung, H.S. (2001). Algorithms for NMF, NIPS",
        ],
        os.path.join(OUT, "ch08_05_nmf_problem.ipynb")
    )

    solution_notebook(CH, 5,
        "비음수 행렬 분해 (NMF)",
        [
            {"approach": "### 문제 1: 곱셈 업데이트 직접 구현",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()
X = faces.data

def nmf_multiplicative(V, r, max_iter=200, eps=1e-7):
    m, n = V.shape
    np.random.seed(42)
    W = np.random.rand(m, r) + eps
    H = np.random.rand(r, n) + eps

    errors = []
    for iteration in range(max_iter):
        # H 업데이트
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        # W 업데이트
        W *= (V @ H.T) / (W @ H @ H.T + eps)

        err = np.linalg.norm(V - W @ H, 'fro')
        errors.append(err)

    return W, H, errors

W_manual, H_manual, errors = nmf_multiplicative(X, r=16, max_iter=300)

# sklearn 비교
nmf_sk = NMF(n_components=16, init='random', max_iter=300, random_state=42)
W_sk = nmf_sk.fit_transform(X)

plt.figure(figsize=(10, 5))
plt.plot(errors, label='직접 구현')
plt.xlabel('반복')
plt.ylabel('재구성 오차 (Frobenius)')
plt.title('NMF 수렴 과정')
plt.legend()
plt.tight_layout()
plt.show()

print(f"직접 구현 최종 오차: {errors[-1]:.4f}")
print(f"sklearn NMF 오차: {nmf_sk.reconstruction_err_:.4f}")""",
             "interpretation": "곱셈 업데이트 규칙은 각 단계에서 목적 함수가 비증가(non-increasing)함이 보장됩니다. sklearn은 더 나은 초기화(NNDSVD)와 최적화를 사용하므로 오차가 더 작을 수 있습니다."},
            {"approach": "### 문제 2: NMF 토픽 모델링",
             "code": """import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# 5개 카테고리 선택
categories = ['rec.sport.baseball', 'sci.electronics', 'comp.graphics', 'talk.politics.guns', 'soc.religion.christian']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

# TF-IDF
tfidf = TfidfVectorizer(max_features=2000, max_df=0.95, min_df=2, stop_words='english')
X_tfidf = tfidf.fit_transform(newsgroups.data)

# NMF 토픽 추출
n_topics = 5
nmf = NMF(n_components=n_topics, init='nndsvd', max_iter=300, random_state=42)
W = nmf.fit_transform(X_tfidf)
H = nmf.components_

feature_names = tfidf.get_feature_names_out()
for topic_idx in range(n_topics):
    top_words_idx = H[topic_idx].argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"토픽 {topic_idx+1}: {', '.join(top_words)}")""",
             "interpretation": "NMF는 각 토픽을 비음수 가중치로 표현하므로 해석이 직관적입니다. LDA와 달리 확률 모델이 아니지만, TF-IDF 행렬에 적용하면 유사한 토픽 구조를 발견합니다."},
            {"approach": "### 문제 3: 컴포넌트 수에 따른 재구성 오차",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()
X = faces.data

r_values = [2, 5, 10, 20, 50]
errors = []
for r in r_values:
    nmf = NMF(n_components=r, init='nndsvd', max_iter=500, random_state=42)
    W = nmf.fit_transform(X)
    errors.append(nmf.reconstruction_err_)
    print(f"r={r:3d}: 재구성 오차 = {nmf.reconstruction_err_:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(r_values, errors, 'bo-', markersize=8)
plt.xlabel('컴포넌트 수 (r)')
plt.ylabel('재구성 오차')
plt.title('NMF: 컴포넌트 수에 따른 재구성 오차')
plt.xticks(r_values)
plt.tight_layout()
plt.show()

# 엘보 포인트 (상대적 감소율)
for i in range(1, len(errors)):
    reduction = (errors[i-1] - errors[i]) / errors[i-1] * 100
    print(f"r={r_values[i-1]}→{r_values[i]}: {reduction:.1f}% 감소")""",
             "interpretation": "엘보 포인트(기울기 변화가 급격히 줄어드는 지점)에서 적절한 r을 결정합니다. r이 너무 크면 과적합 위험이 있고, 너무 작으면 중요 정보가 손실됩니다."},
        ],
        "NMF는 부분-기반(parts-based) 표현을 학습한다는 점에서 PCA(전체 표현)와 차별됩니다. 텍스트, 이미지, 음악 등 비음수 데이터에 자연스럽게 적용되며, 토픽 모델링에서 LDA의 대안으로 널리 사용됩니다.",
        os.path.join(OUT, "ch08_05_nmf_solution.ipynb")
    )

    # ── 6. GMM & EM ──
    problem_notebook(CH, 6,
        "가우시안 혼합과 EM 알고리즘",
        [
            "가우시안 혼합 모델(GMM)의 확률 모델을 이해한다",
            "EM 알고리즘의 E-step과 M-step을 수학적으로 유도한다",
            "BIC/AIC를 이용한 모델 선택을 수행한다",
            "초기화 전략(k-means++, 랜덤)의 영향을 이해한다",
        ],
        r"""### 1. 가우시안 혼합 모델

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

- $\pi_k$: 혼합 가중치 ($\sum_k \pi_k = 1, \; \pi_k \geq 0$)
- $\boldsymbol{\mu}_k$: 각 성분의 평균
- $\boldsymbol{\Sigma}_k$: 각 성분의 공분산 행렬

### 2. EM 알고리즘

**E-step (기대값)**: 사후 확률 (responsibility) 계산

$$\gamma_{nk} = \frac{\pi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

**M-step (최대화)**: 파라미터 업데이트

$$N_k = \sum_{n=1}^N \gamma_{nk}$$

$$\boldsymbol{\mu}_k^{new} = \frac{1}{N_k}\sum_{n=1}^N \gamma_{nk} \mathbf{x}_n$$

$$\boldsymbol{\Sigma}_k^{new} = \frac{1}{N_k}\sum_{n=1}^N \gamma_{nk}(\mathbf{x}_n - \boldsymbol{\mu}_k^{new})(\mathbf{x}_n - \boldsymbol{\mu}_k^{new})^\top$$

$$\pi_k^{new} = \frac{N_k}{N}$$

### 3. 로그 우도

$$\ln p(X|\theta) = \sum_{n=1}^N \ln \left[\sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\right]$$

EM은 각 반복에서 로그 우도를 **비감소**시킨다.

### 4. 모델 선택: BIC / AIC

$$\text{BIC} = -2\ln L + p\ln n, \quad \text{AIC} = -2\ln L + 2p$$

여기서 $p$는 자유 파라미터 수, $n$은 데이터 수. **낮을수록** 좋은 모델.
""",
        """# GMM & EM 구현 가이드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 데이터 생성
X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 1.5, 0.5], random_state=42)

# GMM 적합
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)
y_pred = gmm.predict(X)
probs = gmm.predict_proba(X)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=20, alpha=0.7)
axes[0].set_title('실제 레이블')
axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=20, alpha=0.7)
axes[1].set_title('GMM 예측')

# 등고선 (밀도)
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-2, X[:, 0].max()+2, 100),
                     np.linspace(X[:, 1].min()-2, X[:, 1].max()+2, 100))
Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[1].contour(xx, yy, Z, levels=10, alpha=0.5, cmap='gray')

plt.tight_layout()
plt.show()
print(f"BIC: {gmm.bic(X):.2f}, AIC: {gmm.aic(X):.2f}")""",
        [
            {"description": "EM 알고리즘을 직접 구현하세요.\n\n(a) E-step: responsibility $\\gamma_{nk}$ 계산\n(b) M-step: $\\mu_k, \\Sigma_k, \\pi_k$ 업데이트\n(c) 로그 우도의 수렴 과정을 시각화", "difficulty": "★★★", "skeleton": "from scipy.stats import multivariate_normal\n\ndef em_gmm(X, K, max_iter=100, tol=1e-6):\n    n, d = X.shape\n    \n    # 초기화\n    np.random.seed(42)\n    mu = X[np.random.choice(n, K, replace=False)]\n    sigma = [np.eye(d)] * K\n    pi = np.ones(K) / K\n    \n    log_likelihoods = []\n    \n    for it in range(max_iter):\n        # E-step\n        gamma = np.zeros((n, K))\n        for k in range(K):\n            gamma[:, k] = # TODO\n        gamma /= gamma.sum(axis=1, keepdims=True)\n        \n        # M-step\n        for k in range(K):\n            Nk = # TODO\n            mu[k] = # TODO\n            # sigma[k] = TODO\n            # pi[k] = TODO\n        \n        # 로그 우도\n        ll = # TODO\n        log_likelihoods.append(ll)\n    \n    return mu, sigma, pi, gamma, log_likelihoods"},
            {"description": "BIC와 AIC를 사용하여 최적 클러스터 수를 결정하세요. K ∈ {1, 2, ..., 8}에 대해 BIC/AIC를 계산하고 시각화하세요.", "difficulty": "★★"},
            {"description": "공분산 유형(full, tied, diag, spherical)에 따른 GMM 결과를 비교하세요. 각 유형의 파라미터 수와 적합도(BIC)를 분석하세요.", "difficulty": "★★"},
            {"description": "GMM의 초기화 전략이 수렴에 미치는 영향을 조사하세요. 10번의 서로 다른 랜덤 초기화에 대해 로그 우도와 수렴 속도를 비교하세요.", "difficulty": "★★★"},
        ],
        [
            "Bishop, C.M. (2006). Pattern Recognition and Machine Learning, Ch.9",
            "Dempster, A.P. et al. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm",
        ],
        os.path.join(OUT, "ch08_06_gmm_problem.ipynb")
    )

    solution_notebook(CH, 6,
        "가우시안 혼합과 EM 알고리즘",
        [
            {"approach": "### 문제 1: EM 알고리즘 직접 구현",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 1.5, 0.5], random_state=42)
K = 3

def em_gmm(X, K, max_iter=100, tol=1e-6):
    n, d = X.shape
    np.random.seed(42)
    mu = X[np.random.choice(n, K, replace=False)]
    sigma = [np.eye(d) for _ in range(K)]
    pi = np.ones(K) / K
    log_likelihoods = []

    for it in range(max_iter):
        # E-step
        gamma = np.zeros((n, K))
        for k in range(K):
            gamma[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])
        gamma /= gamma.sum(axis=1, keepdims=True)

        # M-step
        for k in range(K):
            Nk = gamma[:, k].sum()
            mu[k] = (gamma[:, k:k+1].T @ X / Nk).flatten()
            diff = X - mu[k]
            sigma[k] = (diff.T @ (diff * gamma[:, k:k+1])) / Nk + 1e-6 * np.eye(d)
            pi[k] = Nk / n

        # 로그 우도
        ll = 0
        for k in range(K):
            ll += pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])
        ll = np.log(ll + 1e-300).sum()
        log_likelihoods.append(ll)

        if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return mu, sigma, pi, gamma, log_likelihoods

mu, sigma, pi, gamma, lls = em_gmm(X, K)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(lls, 'b-')
axes[0].set_xlabel('반복')
axes[0].set_ylabel('로그 우도')
axes[0].set_title('EM 수렴 과정')

y_pred = gamma.argmax(axis=1)
axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=15, alpha=0.7)
for k in range(K):
    axes[1].scatter(mu[k][0], mu[k][1], c='red', marker='X', s=200, edgecolors='black')
axes[1].set_title('EM 결과')
plt.tight_layout()
plt.show()""",
             "interpretation": "EM 알고리즘은 각 반복에서 로그 우도가 비감소함을 확인할 수 있습니다. 수렴 후 학습된 평균, 공분산, 혼합 가중치가 실제 데이터 생성 파라미터와 유사합니다."},
            {"approach": "### 문제 2: BIC/AIC를 이용한 모델 선택",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 1.5, 0.5], random_state=42)

K_range = range(1, 9)
bics, aics = [], []
for k in K_range:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
    gmm.fit(X)
    bics.append(gmm.bic(X))
    aics.append(gmm.aic(X))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(K_range, bics, 'bo-', label='BIC')
ax.plot(K_range, aics, 'rs-', label='AIC')
ax.axvline(x=K_range[np.argmin(bics)], color='blue', linestyle='--', alpha=0.5)
ax.axvline(x=K_range[np.argmin(aics)], color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('클러스터 수 (K)')
ax.set_ylabel('정보 기준')
ax.set_title('BIC/AIC에 의한 최적 K 선택')
ax.legend()
plt.tight_layout()
plt.show()

print(f"BIC 최적 K: {K_range[np.argmin(bics)]}")
print(f"AIC 최적 K: {K_range[np.argmin(aics)]}")""",
             "interpretation": "BIC는 AIC보다 더 강한 복잡도 패널티(log n)를 부여하므로 보수적인 모델을 선호합니다. 두 기준 모두 최솟값에서 최적 K를 결정합니다."},
            {"approach": "### 문제 3: 공분산 유형 비교",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 1.5, 0.5], random_state=42)

cov_types = ['full', 'tied', 'diag', 'spherical']
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, cov_type in enumerate(cov_types):
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42)
    gmm.fit(X)
    y_pred = gmm.predict(X)

    axes[i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=10, alpha=0.7)
    n_params = gmm._n_parameters()
    axes[i].set_title(f'{cov_type}\\nBIC={gmm.bic(X):.0f}, params={n_params}')

plt.suptitle('공분산 유형별 GMM 결과', fontsize=14)
plt.tight_layout()
plt.show()""",
             "interpretation": "full: 가장 유연하지만 파라미터 수 최다. diag: 축 정렬 타원. spherical: 원형 클러스터. tied: 모든 성분이 같은 공분산. BIC로 비교하면 데이터에 가장 적합한 유형을 선택할 수 있습니다."},
            {"approach": "### 문제 4: 초기화 전략의 영향",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 1.5, 0.5], random_state=42)

n_runs = 10
log_likelihoods = []
convergence_iters = []

for run in range(n_runs):
    gmm = GaussianMixture(n_components=3, random_state=run, max_iter=200, tol=1e-6)
    gmm.fit(X)
    log_likelihoods.append(gmm.lower_bound_)
    convergence_iters.append(gmm.n_iter_)
    print(f"Run {run+1}: 로그우도={gmm.lower_bound_:.2f}, 수렴 반복={gmm.n_iter_}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, n_runs+1), log_likelihoods)
axes[0].set_xlabel('실행 번호')
axes[0].set_ylabel('최종 로그 우도')
axes[0].set_title('초기화별 로그 우도')

axes[1].bar(range(1, n_runs+1), convergence_iters, color='orange')
axes[1].set_xlabel('실행 번호')
axes[1].set_ylabel('수렴 반복 수')
axes[1].set_title('초기화별 수렴 속도')
plt.tight_layout()
plt.show()

print(f"\\n로그 우도 범위: [{min(log_likelihoods):.2f}, {max(log_likelihoods):.2f}]")
print(f"최적 초기화: Run {np.argmax(log_likelihoods)+1}")""",
             "interpretation": "EM은 초기값에 따라 다른 지역 최적해에 수렴할 수 있습니다. sklearn은 n_init 파라미터로 여러 초기화 중 최적을 선택합니다. k-means 기반 초기화가 랜덤보다 안정적입니다."},
        ],
        "GMM은 K-means의 확률적 일반화로, 소프트 할당과 타원형 클러스터를 지원합니다. 하지만 고차원에서는 공분산 추정이 불안정해지므로 PCA 전처리가 권장됩니다.",
        os.path.join(OUT, "ch08_06_gmm_solution.ipynb")
    )

    # ── 7. Spectral Clustering ──
    problem_notebook(CH, 7,
        "스펙트럼 클러스터링",
        [
            "그래프 라플라시안의 정의와 성질을 이해한다",
            "정규화 컷(NCut)의 최적화 문제를 유도한다",
            "스펙트럼 클러스터링의 알고리즘을 구현한다",
            "유사도 그래프 구성 방법을 비교한다",
        ],
        r"""### 1. 유사도 그래프

- **$\epsilon$-이웃**: $w_{ij} = 1$ if $d(x_i, x_j) < \epsilon$
- **k-NN 그래프**: k개 최근접 이웃과 연결
- **완전 연결**: $w_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$

### 2. 그래프 라플라시안

**비정규화 라플라시안**: $L = D - W$

여기서 $D = \text{diag}(d_1, \dots, d_n)$, $d_i = \sum_j w_{ij}$

**성질**:
- $L$은 양의 반정부호 (positive semi-definite)
- $L\mathbf{1} = 0$ (최소 고유값 0, 고유벡터 $\mathbf{1}$)
- 0 고유값의 중복도 = 연결 성분의 수

### 3. 정규화 라플라시안

**대칭 정규화**: $L_{sym} = D^{-1/2}LD^{-1/2} = I - D^{-1/2}WD^{-1/2}$

**랜덤 워크 정규화**: $L_{rw} = D^{-1}L = I - D^{-1}W$

### 4. 정규화 컷 (NCut)

$$\text{NCut}(A, B) = \frac{\text{cut}(A, B)}{\text{vol}(A)} + \frac{\text{cut}(A, B)}{\text{vol}(B)}$$

여기서 $\text{cut}(A,B) = \sum_{i \in A, j \in B} w_{ij}$, $\text{vol}(A) = \sum_{i \in A} d_i$

이산 최적화 → 연속 완화: $L_{rw}\mathbf{u} = \lambda \mathbf{u}$의 **최소 $k$개 고유벡터** 사용

### 5. 알고리즘 (Ng, Jordan, Weiss)

1. 유사도 행렬 $W$ 구성
2. $L_{sym} = I - D^{-1/2}WD^{-1/2}$ 계산
3. 최소 $k$개 고유벡터 $U \in \mathbb{R}^{n \times k}$
4. 행 정규화: $T_{ij} = U_{ij} / (\sum_j U_{ij}^2)^{1/2}$
5. $T$의 행에 k-means 적용
""",
        """# 스펙트럼 클러스터링 구현 가이드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.datasets import make_moons, make_circles

# 데이터 생성
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
X_circles, y_circles = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for row, (X, y, name) in enumerate([(X_moons, y_moons, '반달'), (X_circles, y_circles, '동심원')]):
    # 원본
    axes[row, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=20)
    axes[row, 0].set_title(f'{name} — 원본')

    # K-means
    km = KMeans(n_clusters=2, random_state=42)
    axes[row, 1].scatter(X[:, 0], X[:, 1], c=km.fit_predict(X), cmap='viridis', s=20)
    axes[row, 1].set_title(f'{name} — K-means')

    # Spectral
    sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=10, random_state=42)
    axes[row, 2].scatter(X[:, 0], X[:, 1], c=sc.fit_predict(X), cmap='viridis', s=20)
    axes[row, 2].set_title(f'{name} — 스펙트럼')

plt.tight_layout()
plt.show()""",
        [
            {"description": "그래프 라플라시안을 직접 구성하고, 고유값 분해를 수행하여 스펙트럼 클러스터링을 구현하세요.\n\n1. RBF 유사도 행렬 $W$ 계산\n2. 차수 행렬 $D$, 라플라시안 $L = D - W$ 계산\n3. 최소 $k$개 고유벡터 추출\n4. 고유벡터에 k-means 적용", "difficulty": "★★★", "skeleton": "def spectral_clustering_manual(X, n_clusters, gamma=1.0):\n    from sklearn.metrics.pairwise import rbf_kernel\n    from sklearn.cluster import KMeans\n    \n    # 1. 유사도 행렬\n    W = rbf_kernel(X, gamma=gamma)\n    \n    # 2. 라플라시안\n    D = np.diag(W.sum(axis=1))\n    L = # TODO\n    \n    # 3. 고유값 분해\n    eigenvalues, eigenvectors = # TODO\n    \n    # 4. k-means\n    # TODO\n    \n    return labels"},
            {"description": "라플라시안의 고유값 스펙트럼을 분석하여 최적 클러스터 수를 결정하세요. 고유값 갭(eigengap)을 시각화하세요.", "difficulty": "★★"},
            {"description": "gamma 파라미터가 스펙트럼 클러스터링 결과에 미치는 영향을 분석하세요. gamma ∈ {0.1, 1, 5, 10, 50}에 대해 결과를 비교하세요.", "difficulty": "★★"},
        ],
        [
            "von Luxburg, U. (2007). A Tutorial on Spectral Clustering, Statistics and Computing",
            "Ng, A., Jordan, M., & Weiss, Y. (2002). On Spectral Clustering, NIPS",
        ],
        os.path.join(OUT, "ch08_07_spectral_clustering_problem.ipynb")
    )

    solution_notebook(CH, 7,
        "스펙트럼 클러스터링",
        [
            {"approach": "### 문제 1: 스펙트럼 클러스터링 직접 구현",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_moons

X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)

def spectral_clustering_manual(X, n_clusters, gamma=1.0):
    # 1. 유사도 행렬
    W = rbf_kernel(X, gamma=gamma)

    # 2. 비정규화 라플라시안
    D = np.diag(W.sum(axis=1))
    L = D - W

    # 3. 정규화 라플라시안 (대칭)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt

    # 4. 고유값 분해 (최소 k개)
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    U = eigenvectors[:, :n_clusters]

    # 5. 행 정규화
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1
    T = U / norms

    # 6. k-means
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(T)

    return labels, eigenvalues

labels_manual, eigs = spectral_clustering_manual(X, n_clusters=2, gamma=10)

# sklearn 비교
sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=10, random_state=42)
labels_sklearn = sc.fit_predict(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_manual, cmap='viridis', s=20)
axes[0].set_title('직접 구현')
axes[1].scatter(X[:, 0], X[:, 1], c=labels_sklearn, cmap='viridis', s=20)
axes[1].set_title('sklearn SpectralClustering')
plt.tight_layout()
plt.show()""",
             "interpretation": "정규화 라플라시안의 최소 고유벡터가 클러스터 구조를 인코딩합니다. 비선형 구조(반달, 동심원)를 k-means보다 훨씬 잘 분리합니다."},
            {"approach": "### 문제 2: 고유값 갭 분석",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_blobs

# 다양한 클러스터 수의 데이터
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

W = rbf_kernel(X, gamma=1.0)
D = np.diag(W.sum(axis=1))
L = D - W
eigenvalues = np.linalg.eigvalsh(L)

# 첫 15개 고유값
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(15), eigenvalues[:15])
axes[0].set_xlabel('인덱스')
axes[0].set_ylabel('고유값')
axes[0].set_title('라플라시안 고유값 스펙트럼')

# 고유값 갭
gaps = np.diff(eigenvalues[:15])
axes[1].bar(range(len(gaps)), gaps)
axes[1].set_xlabel('인덱스')
axes[1].set_ylabel('고유값 갭')
axes[1].set_title('고유값 갭 (Eigengap)')
plt.tight_layout()
plt.show()

optimal_k = np.argmax(gaps[:10]) + 1
print(f"최대 고유값 갭 위치 → 최적 클러스터 수: {optimal_k}")""",
             "interpretation": "고유값 갭 휴리스틱: 고유값 차이가 가장 큰 위치 이전의 고유값 수가 최적 클러스터 수입니다. 4개 클러스터 데이터에서 4번째 갭이 최대임을 확인할 수 있습니다."},
            {"approach": "### 문제 3: gamma 파라미터 영향",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
gammas = [0.1, 1, 5, 10, 50]

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for i, g in enumerate(gammas):
    sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=g, random_state=42)
    labels = sc.fit_predict(X)
    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20)
    axes[i].set_title(f'γ = {g}')
plt.suptitle('gamma에 따른 스펙트럼 클러스터링 결과', fontsize=14)
plt.tight_layout()
plt.show()""",
             "interpretation": "gamma가 너무 작으면 모든 점이 연결되어 분리가 안 되고, 너무 크면 그래프가 단절됩니다. 적절한 gamma에서 비선형 구조를 정확히 분리합니다."},
        ],
        "스펙트럼 클러스터링은 그래프 이론에 기반한 강력한 방법이지만, 유사도 행렬의 크기가 $O(n^2)$이므로 대규모 데이터에 비효율적입니다. Nyström 근사나 랜덤 서브샘플링으로 확장할 수 있습니다.",
        os.path.join(OUT, "ch08_07_spectral_clustering_solution.ipynb")
    )

    # ── 8. Density Clustering ──
    problem_notebook(CH, 8,
        "밀도 기반 클러스터링",
        [
            "DBSCAN의 알고리즘과 핵심 개념(core/border/noise)을 이해한다",
            "HDBSCAN의 계층적 확장 원리를 이해한다",
            "OPTICS의 도달 거리(reachability distance) 플롯을 해석한다",
            "밀도 기반 방법의 장단점을 분석한다",
        ],
        r"""### 1. DBSCAN 핵심 개념

- **$\epsilon$-이웃**: $N_\epsilon(\mathbf{x}) = \{\mathbf{y} \in D : d(\mathbf{x}, \mathbf{y}) \leq \epsilon\}$
- **핵심 점(Core point)**: $|N_\epsilon(\mathbf{x})| \geq \text{MinPts}$
- **경계 점(Border point)**: 핵심 점의 $\epsilon$-이웃에 있지만 자신은 핵심 점이 아닌 점
- **노이즈 점**: 핵심 점도 경계 점도 아닌 점

### 2. 밀도 도달 가능성

$\mathbf{x}$에서 $\mathbf{y}$로 **밀도 도달 가능**:

$$\exists \mathbf{x}_1 = \mathbf{x}, \mathbf{x}_2, \dots, \mathbf{x}_m = \mathbf{y} \text{ s.t. } \mathbf{x}_{i+1} \in N_\epsilon(\mathbf{x}_i) \text{ and each } \mathbf{x}_i \text{ is a core point}$$

### 3. OPTICS

**핵심 거리(Core distance)**:
$$\text{cd}_\epsilon(\mathbf{x}) = \begin{cases} d(\mathbf{x}, N_\epsilon^{\text{MinPts}}(\mathbf{x})) & \text{if } |\mathbf{x}| \text{ is core} \\ \text{undefined} & \text{otherwise} \end{cases}$$

**도달 거리(Reachability distance)**:
$$\text{rd}_\epsilon(\mathbf{x}, \mathbf{y}) = \max(\text{cd}_\epsilon(\mathbf{x}), d(\mathbf{x}, \mathbf{y}))$$

### 4. HDBSCAN

- 상호 도달 거리: $d_{mreach}(\mathbf{x}, \mathbf{y}) = \max(\text{cd}_k(\mathbf{x}), \text{cd}_k(\mathbf{y}), d(\mathbf{x}, \mathbf{y}))$
- 최소 신장 트리 → 클러스터 계층 구조 → 안정성 기반 선택
""",
        """# 밀도 기반 클러스터링 가이드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.datasets import make_moons, make_blobs

# 복합 데이터 생성
X1, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
X2, _ = make_blobs(n_samples=100, centers=[[2.5, 2.5]], cluster_std=0.3, random_state=42)
noise = np.random.RandomState(42).uniform(-1.5, 3.5, size=(30, 2))
X = np.vstack([X1, X2, noise])

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

# OPTICS
optics = OPTICS(min_samples=5, xi=0.05)
labels_optics = optics.fit_predict(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis', s=20)
axes[0].set_title(f'DBSCAN (eps=0.3, 클러스터={len(set(labels_dbscan))-1})')
axes[1].scatter(X[:, 0], X[:, 1], c=labels_optics, cmap='viridis', s=20)
axes[1].set_title(f'OPTICS (클러스터={len(set(labels_optics))-1})')
plt.tight_layout()
plt.show()""",
        [
            {"description": "DBSCAN의 eps 파라미터를 결정하기 위한 k-distance 그래프를 그리세요.\n\n(a) k=MinPts일 때 각 점의 k번째 최근접 이웃 거리를 정렬하여 플롯\n(b) 엘보 포인트에서 eps를 결정", "difficulty": "★★", "skeleton": "from sklearn.neighbors import NearestNeighbors\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# k-distance 그래프\nk = 5  # MinPts\nnn = NearestNeighbors(n_neighbors=k)\nnn.fit(X)\ndistances, _ = nn.kneighbors(X)\n\n# k번째 이웃까지의 거리 (정렬)\nk_distances = # TODO\n\nplt.plot(k_distances)\nplt.xlabel('포인트 (정렬됨)')\nplt.ylabel(f'{k}-거리')\nplt.title('k-distance 그래프')\nplt.show()"},
            {"description": "OPTICS의 도달 거리 플롯(reachability plot)을 그리고 해석하세요. 클러스터 경계가 어떻게 결정되는지 설명하세요.", "difficulty": "★★"},
            {"description": "DBSCAN, OPTICS, (선택: HDBSCAN)를 다양한 형태의 데이터셋에서 비교하세요.\n\n데이터: make_moons, make_circles, make_blobs(다양한 밀도)\n\n각 방법의 강점과 약점을 분석하세요.", "difficulty": "★★★"},
        ],
        [
            "Ester, M. et al. (1996). A Density-Based Algorithm for Discovering Clusters (DBSCAN), KDD",
            "Campello, R.J.G.B. et al. (2013). Density-Based Clustering Based on Hierarchical Density Estimates (HDBSCAN)",
            "Ankerst, M. et al. (1999). OPTICS: Ordering Points To Identify the Clustering Structure",
        ],
        os.path.join(OUT, "ch08_08_density_clustering_problem.ipynb")
    )

    solution_notebook(CH, 8,
        "밀도 기반 클러스터링",
        [
            {"approach": "### 문제 1: k-distance 그래프로 eps 결정",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN

X1, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
X2, _ = make_blobs(n_samples=100, centers=[[2.5, 2.5]], cluster_std=0.3, random_state=42)
noise = np.random.RandomState(42).uniform(-1.5, 3.5, size=(30, 2))
X = np.vstack([X1, X2, noise])

k = 5
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X)
distances, _ = nn.kneighbors(X)

# k번째 이웃 거리를 정렬
k_distances = np.sort(distances[:, k-1])[::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(k_distances)
axes[0].axhline(y=0.3, color='r', linestyle='--', label='eps=0.3')
axes[0].set_xlabel('포인트 (정렬됨)')
axes[0].set_ylabel(f'{k}-거리')
axes[0].set_title('k-distance 그래프')
axes[0].legend()

# 최적 eps로 DBSCAN
eps_opt = 0.3
dbscan = DBSCAN(eps=eps_opt, min_samples=k)
labels = dbscan.fit_predict(X)
axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20)
axes[1].set_title(f'DBSCAN (eps={eps_opt})')
plt.tight_layout()
plt.show()

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"클러스터 수: {n_clusters}, 노이즈 점: {n_noise}")""",
             "interpretation": "k-distance 그래프에서 급격한 기울기 변화(엘보)가 나타나는 지점이 적절한 eps입니다. 이 값보다 큰 거리의 점들은 노이즈로 분류됩니다."},
            {"approach": "### 문제 2: OPTICS 도달 거리 플롯",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.datasets import make_moons, make_blobs

X1, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
X2, _ = make_blobs(n_samples=100, centers=[[2.5, 2.5]], cluster_std=0.3, random_state=42)
noise = np.random.RandomState(42).uniform(-1.5, 3.5, size=(30, 2))
X = np.vstack([X1, X2, noise])

optics = OPTICS(min_samples=5, xi=0.05)
optics.fit(X)

# 도달 거리 플롯
reachability = optics.reachability_[optics.ordering_]
labels = optics.labels_[optics.ordering_]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Reachability 플롯
colors = ['gray' if l == -1 else plt.cm.viridis(l / max(optics.labels_.max(), 1)) for l in labels]
axes[0].bar(range(len(reachability)), reachability, color=colors, width=1.0)
axes[0].set_ylabel('도달 거리')
axes[0].set_title('OPTICS 도달 거리 플롯')

# 클러스터 결과
axes[1].scatter(X[:, 0], X[:, 1], c=optics.labels_, cmap='viridis', s=20)
axes[1].set_title(f'OPTICS 클러스터링 결과 (클러스터 수: {optics.labels_.max()+1})')
plt.tight_layout()
plt.show()""",
             "interpretation": "도달 거리 플롯에서 깊은 골짜기가 밀집 클러스터에 해당하고, 봉우리가 클러스터 경계입니다. OPTICS는 DBSCAN과 달리 다양한 밀도의 클러스터를 동시에 발견할 수 있습니다."},
            {"approach": "### 문제 3: 밀도 기반 방법 비교",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler

datasets = {
    '반달': make_moons(n_samples=300, noise=0.05, random_state=42),
    '동심원': make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=42),
    '다양한 밀도': make_blobs(n_samples=[100, 200, 50], centers=[[-5,0],[0,0],[5,0]],
                           cluster_std=[0.5, 1.5, 0.3], random_state=42),
}

methods = {
    'DBSCAN (eps=0.3)': lambda X: DBSCAN(eps=0.3, min_samples=5).fit_predict(X),
    'DBSCAN (eps=0.5)': lambda X: DBSCAN(eps=0.5, min_samples=5).fit_predict(X),
    'OPTICS': lambda X: OPTICS(min_samples=5, xi=0.05).fit_predict(X),
}

fig, axes = plt.subplots(3, len(methods)+1, figsize=(20, 12))
for i, (dname, (X, y)) in enumerate(datasets.items()):
    X = StandardScaler().fit_transform(X)
    axes[i, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=15)
    axes[i, 0].set_title(f'{dname} (원본)')

    for j, (mname, method) in enumerate(methods.items(), 1):
        labels = method(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        axes[i, j].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=15)
        axes[i, j].set_title(f'{mname}\\nC={n_clusters}, N={n_noise}')

plt.tight_layout()
plt.show()""",
             "interpretation": "DBSCAN은 eps에 민감하여 다양한 밀도의 클러스터를 처리하기 어렵습니다. OPTICS는 계층적 밀도 구조를 포착하므로 더 유연합니다. HDBSCAN은 자동 클러스터 수 결정이 가능하여 실전에서 가장 권장됩니다."},
        ],
        "밀도 기반 클러스터링은 임의 형태의 클러스터를 발견하고 노이즈를 자연스럽게 처리합니다. 하지만 고차원 데이터에서는 '차원의 저주'로 밀도 추정이 어려워지므로, PCA/UMAP 전처리가 권장됩니다.",
        os.path.join(OUT, "ch08_08_density_clustering_solution.ipynb")
    )

    # ── 9. Cluster Validation ──
    problem_notebook(CH, 9,
        "클러스터 유효성 검증",
        [
            "실루엣 계수의 정의와 해석을 이해한다",
            "갭 통계량의 이론적 배경과 계산법을 이해한다",
            "클러스터 안정성(stability) 분석 방법을 학습한다",
            "내부/외부 평가 지표를 비교한다",
        ],
        r"""### 1. 실루엣 계수 (Silhouette Coefficient)

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

- $a(i)$: 같은 클러스터 내 평균 거리 (응집도)
- $b(i)$: 가장 가까운 다른 클러스터까지의 평균 거리 (분리도)
- $s(i) \in [-1, 1]$: 1에 가까울수록 좋은 배치

### 2. 갭 통계량 (Gap Statistic)

$$\text{Gap}_n(k) = E^*_n[\log W_k] - \log W_k$$

여기서 $W_k = \sum_{r=1}^k \frac{1}{2n_r}\sum_{i,j \in C_r} d(x_i, x_j)$

- $E^*_n$: 균일 분포 참조 데이터의 기댓값 (Monte Carlo)
- 최적 $k$: $\text{Gap}(k) \geq \text{Gap}(k+1) - s_{k+1}$을 만족하는 최소 $k$

### 3. 내부 평가 지표

| 지표 | 수식 | 범위 |
|------|------|------|
| 실루엣 | $(b-a)/\max(a,b)$ | [-1, 1] |
| Calinski-Harabasz | $\frac{SS_B/(k-1)}{SS_W/(n-k)}$ | [0, ∞) |
| Davies-Bouldin | $\frac{1}{k}\sum_i \max_{j\neq i}\frac{s_i + s_j}{d(c_i, c_j)}$ | [0, ∞) |

### 4. 외부 평가 지표 (레이블이 있을 때)

- **ARI (Adjusted Rand Index)**: 우연을 보정한 쌍 일치도
- **NMI (Normalized Mutual Information)**: 정보 이론 기반 일치도
- **V-measure**: 동질성과 완전성의 조화 평균
""",
        """# 클러스터 유효성 검증 가이드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

# 1. 실루엣 분석
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
K_range = range(2, 8)
silhouette_avgs = []

for idx, k in enumerate(K_range):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil_avg = silhouette_score(X, labels)
    silhouette_avgs.append(sil_avg)

    ax = axes[idx // 3, idx % 3]
    sil_values = silhouette_samples(X, labels)
    y_lower = 10
    for i in range(k):
        cluster_sil = np.sort(sil_values[labels == i])
        ax.fill_betweenx(np.arange(y_lower, y_lower + len(cluster_sil)),
                         0, cluster_sil, alpha=0.7)
        y_lower += len(cluster_sil) + 10
    ax.axvline(x=sil_avg, color='red', linestyle='--')
    ax.set_title(f'K={k}, 평균={sil_avg:.3f}')

plt.suptitle('실루엣 분석', fontsize=14)
plt.tight_layout()
plt.show()""",
        [
            {"description": "갭 통계량(Gap Statistic)을 직접 구현하세요.\n\n(a) K=1~8에 대해 갭 통계량 계산\n(b) 참조 분포는 데이터 범위의 균일 분포에서 B=20회 샘플링\n(c) 표준 오차를 포함한 그래프 작성", "difficulty": "★★★", "skeleton": "def gap_statistic(X, K_range, B=20):\n    gaps = []\n    stds = []\n    \n    for k in K_range:\n        km = KMeans(n_clusters=k, random_state=42, n_init=10)\n        km.fit(X)\n        Wk = # TODO: 클러스터 내 분산 합\n        \n        # 참조 분포\n        Wk_refs = []\n        for b in range(B):\n            X_ref = # TODO: 균일 분포 샘플\n            km_ref = KMeans(n_clusters=k, random_state=b, n_init=3)\n            km_ref.fit(X_ref)\n            Wk_ref = # TODO\n            Wk_refs.append(np.log(Wk_ref))\n        \n        gap = np.mean(Wk_refs) - np.log(Wk)\n        std = np.std(Wk_refs) * np.sqrt(1 + 1/B)\n        gaps.append(gap)\n        stds.append(std)\n    \n    return gaps, stds"},
            {"description": "Calinski-Harabasz 지수와 Davies-Bouldin 지수를 K=2~8에 대해 계산하고, 실루엣 계수와 함께 비교하세요. 세 지표가 동일한 최적 K를 제시하는지 확인하세요.", "difficulty": "★★"},
            {"description": "클러스터 안정성 분석을 수행하세요. 부트스트랩 리샘플링(20회)을 통해 각 K에서의 클러스터 할당 안정성(ARI)을 측정하세요.", "difficulty": "★★★"},
            {"description": "외부 평가 지표(ARI, NMI, V-measure)를 사용하여 K-means, GMM, 스펙트럼 클러스터링의 성능을 비교하세요.", "difficulty": "★★"},
        ],
        [
            "Rousseeuw, P.J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis",
            "Tibshirani, R. et al. (2001). Estimating the Number of Clusters via the Gap Statistic",
        ],
        os.path.join(OUT, "ch08_09_cluster_validation_problem.ipynb")
    )

    solution_notebook(CH, 9,
        "클러스터 유효성 검증",
        [
            {"approach": "### 문제 1: 갭 통계량 구현",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

def compute_Wk(X, labels, centers):
    Wk = 0
    for k in range(len(centers)):
        cluster_points = X[labels == k]
        Wk += np.sum((cluster_points - centers[k]) ** 2)
    return Wk

def gap_statistic(X, K_range, B=20):
    gaps, stds = [], []

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        Wk = compute_Wk(X, labels, km.cluster_centers_)

        # 참조 분포 (균일)
        mins, maxs = X.min(axis=0), X.max(axis=0)
        Wk_refs = []
        for b in range(B):
            X_ref = np.random.RandomState(b).uniform(mins, maxs, size=X.shape)
            km_ref = KMeans(n_clusters=k, random_state=b, n_init=3)
            labels_ref = km_ref.fit_predict(X_ref)
            Wk_ref = compute_Wk(X_ref, labels_ref, km_ref.cluster_centers_)
            Wk_refs.append(np.log(Wk_ref))

        gap = np.mean(Wk_refs) - np.log(Wk)
        std = np.std(Wk_refs) * np.sqrt(1 + 1/B)
        gaps.append(gap)
        stds.append(std)

    return gaps, stds

K_range = range(1, 9)
gaps, stds = gap_statistic(X, K_range)

plt.figure(figsize=(10, 5))
plt.errorbar(list(K_range), gaps, yerr=stds, fmt='bo-', capsize=5)
plt.xlabel('클러스터 수 (K)')
plt.ylabel('Gap 통계량')
plt.title('갭 통계량')
plt.tight_layout()
plt.show()

# 최적 K 결정 (Gap(k) >= Gap(k+1) - s_{k+1})
for k in range(len(gaps) - 1):
    if gaps[k] >= gaps[k+1] - stds[k+1]:
        print(f"최적 K: {list(K_range)[k]}")
        break""",
             "interpretation": "갭 통계량은 데이터의 클러스터 구조를 균일 분포와 비교합니다. Gap(k)가 처음으로 충분히 클 때(1-SE 규칙)의 k가 최적입니다."},
            {"approach": "### 문제 2: 내부 평가 지표 비교",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

K_range = range(2, 9)
sil_scores, ch_scores, db_scores = [], [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels))
    ch_scores.append(calinski_harabasz_score(X, labels))
    db_scores.append(davies_bouldin_score(X, labels))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(list(K_range), sil_scores, 'bo-')
axes[0].set_title(f'실루엣 (최적 K={list(K_range)[np.argmax(sil_scores)]})')
axes[0].set_xlabel('K'); axes[0].set_ylabel('실루엣 점수 (↑)')

axes[1].plot(list(K_range), ch_scores, 'rs-')
axes[1].set_title(f'Calinski-Harabasz (최적 K={list(K_range)[np.argmax(ch_scores)]})')
axes[1].set_xlabel('K'); axes[1].set_ylabel('CH 지수 (↑)')

axes[2].plot(list(K_range), db_scores, 'g^-')
axes[2].set_title(f'Davies-Bouldin (최적 K={list(K_range)[np.argmin(db_scores)]})')
axes[2].set_xlabel('K'); axes[2].set_ylabel('DB 지수 (↓)')

plt.suptitle('내부 평가 지표 비교', fontsize=14)
plt.tight_layout()
plt.show()""",
             "interpretation": "세 지표가 모두 K=4를 최적으로 제시하면 결과가 일관됩니다. 실루엣은 가장 직관적이고, CH는 클러스터 간/내 분산 비율, DB는 가장 유사한 클러스터 쌍의 근접도를 측정합니다."},
            {"approach": "### 문제 3: 클러스터 안정성 분석",
             "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

def cluster_stability(X, K_range, n_bootstrap=20):
    stability_scores = {k: [] for k in K_range}
    n = len(X)

    for k in K_range:
        # 전체 데이터 레이블
        km_full = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_full = km_full.fit_predict(X)

        for b in range(n_bootstrap):
            # 부트스트랩 샘플
            indices = np.random.RandomState(b).choice(n, n, replace=True)
            X_boot = X[indices]

            km_boot = KMeans(n_clusters=k, random_state=b, n_init=5)
            labels_boot = km_boot.fit_predict(X_boot)

            # 공통 인덱스에서 ARI 계산
            ari = adjusted_rand_score(labels_full[indices], labels_boot)
            stability_scores[k].append(ari)

    return stability_scores

K_range = range(2, 9)
stability = cluster_stability(X, K_range)

means = [np.mean(stability[k]) for k in K_range]
stds = [np.std(stability[k]) for k in K_range]

plt.figure(figsize=(10, 5))
plt.errorbar(list(K_range), means, yerr=stds, fmt='bo-', capsize=5)
plt.xlabel('클러스터 수 (K)')
plt.ylabel('안정성 (ARI)')
plt.title('부트스트랩 클러스터 안정성')
plt.tight_layout()
plt.show()

print(f"가장 안정적인 K: {list(K_range)[np.argmax(means)]}")""",
             "interpretation": "안정성이 높은 K는 데이터를 부분적으로 변경해도 클러스터 구조가 유지됨을 의미합니다. 실제 클러스터 수에서 안정성이 최대가 되며, 과다/과소 분할에서는 불안정해집니다."},
            {"approach": "### 문제 4: 외부 평가 지표로 방법 비교",
             "code": """import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

datasets = {
    '구형': make_blobs(n_samples=500, centers=3, random_state=42),
    '비선형': make_moons(n_samples=500, noise=0.1, random_state=42),
}

methods = {
    'K-means': lambda X, k: KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X),
    'GMM': lambda X, k: GaussianMixture(n_components=k, random_state=42).fit_predict(X),
    'Spectral': lambda X, k: SpectralClustering(n_clusters=k, random_state=42).fit_predict(X),
}

for dname, (X, y_true) in datasets.items():
    k = len(np.unique(y_true))
    print(f"\\n=== {dname} 데이터 (K={k}) ===")
    print(f"{'방법':15s} {'ARI':>8s} {'NMI':>8s} {'V-measure':>10s}")
    print("-" * 45)
    for mname, method in methods.items():
        labels = method(X, k)
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
        vm = v_measure_score(y_true, labels)
        print(f"{mname:15s} {ari:8.4f} {nmi:8.4f} {vm:10.4f}")""",
             "interpretation": "구형 클러스터에서는 세 방법 모두 잘 작동하지만, 비선형 구조에서는 스펙트럼 클러스터링이 압도적으로 우수합니다. ARI는 우연 보정이 되어 있어 NMI보다 엄격합니다."},
        ],
        "단일 지표에 의존하지 말고 여러 지표를 종합적으로 고려하세요. 내부 지표는 레이블 없이 사용 가능하지만, 외부 지표는 '정답'이 있을 때만 사용 가능합니다. 실무에서는 도메인 지식과 결합한 해석이 중요합니다.",
        os.path.join(OUT, "ch08_09_cluster_validation_solution.ipynb")
    )

    # ── 10. Practice: Customer Segmentation ──
    problem_notebook(CH, 10,
        "실전: 고객 세분화",
        [
            "RFM(Recency, Frequency, Monetary) 분석의 원리를 이해한다",
            "다양한 클러스터링 기법을 실전 데이터에 적용한다",
            "클러스터 프로파일링과 비즈니스 해석을 수행한다",
            "차원 축소와 클러스터링의 결합 파이프라인을 구축한다",
        ],
        r"""### 1. RFM 분석

| 지표 | 정의 | 의미 |
|------|------|------|
| Recency (R) | 마지막 구매 이후 경과일 | 낮을수록 최근 활성 고객 |
| Frequency (F) | 구매 횟수 | 높을수록 충성 고객 |
| Monetary (M) | 총 구매 금액 | 높을수록 고가치 고객 |

### 2. 고객 세분화 프로세스

1. **데이터 전처리**: 이상치 제거, 로그 변환, 표준화
2. **차원 축소**: PCA로 노이즈 제거 및 시각화
3. **최적 K 결정**: 실루엣, 엘보, 갭 통계량
4. **클러스터링**: K-means, GMM, 계층적 등
5. **프로파일링**: 각 세그먼트의 특성 분석
6. **전략 수립**: 세그먼트별 마케팅 전략

### 3. 스케일링과 변환

RFM 변수는 왜도(skewness)가 클 수 있음:

$$x' = \log(x + 1) \quad \text{(로그 변환)}$$

$$z = \frac{x - \mu}{\sigma} \quad \text{(표준화)}$$
""",
        """# 고객 세분화 가이드
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 합성 RFM 데이터 생성
np.random.seed(42)
n_customers = 1000

# 4개 세그먼트 시뮬레이션
segments = {
    'VIP': {'R': (5, 10), 'F': (30, 50), 'M': (5000, 15000), 'n': 100},
    '활성': {'R': (10, 30), 'F': (10, 30), 'M': (1000, 5000), 'n': 300},
    '휴면': {'R': (100, 365), 'F': (1, 5), 'M': (100, 500), 'n': 400},
    '이탈 위험': {'R': (60, 120), 'F': (5, 15), 'M': (500, 2000), 'n': 200},
}

data = []
for seg_name, params in segments.items():
    for _ in range(params['n']):
        r = np.random.uniform(*params['R'])
        f = np.random.uniform(*params['F'])
        m = np.random.uniform(*params['M'])
        data.append({'Recency': r, 'Frequency': f, 'Monetary': m, 'Segment': seg_name})

df = pd.DataFrame(data)
print(df.describe())
print(f"\\n세그먼트 분포:\\n{df['Segment'].value_counts()}")

# RFM 분포 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
    axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(col)
plt.suptitle('RFM 분포')
plt.tight_layout()
plt.show()""",
        [
            {"description": "RFM 데이터에 대해 완전한 고객 세분화 파이프라인을 구축하세요.\n\n(a) 로그 변환 및 표준화\n(b) 최적 클러스터 수 결정 (실루엣 + 엘보)\n(c) K-means 클러스터링\n(d) 각 세그먼트의 RFM 프로파일 시각화", "difficulty": "★★"},
            {"description": "PCA를 이용한 차원 축소 후 클러스터링을 수행하고, 원본 특성 공간에서 직접 클러스터링한 결과와 비교하세요.\n\n(a) PCA 바이플롯 위에 클러스터 표시\n(b) 각 접근법의 실루엣 점수 비교", "difficulty": "★★"},
            {"description": "K-means, GMM, 계층적 클러스터링(Ward), DBSCAN을 비교하세요.\n\n(a) ARI를 사용하여 실제 세그먼트 레이블과 비교\n(b) 각 방법의 장단점 분석\n(c) 최적 방법 추천과 근거 제시", "difficulty": "★★★"},
            {"description": "각 클러스터(세그먼트)에 대한 비즈니스 보고서를 작성하세요.\n\n(a) 세그먼트별 RFM 요약 통계\n(b) 레이더 차트로 세그먼트 프로파일 시각화\n(c) 세그먼트별 마케팅 전략 제안", "difficulty": "★★★", "hint": "matplotlib의 polar 차트를 활용하세요."},
            {"description": "UMAP 또는 t-SNE로 RFM 데이터를 2D로 시각화하고, 클러스터 경계를 오버레이하세요. 차원 축소 방법에 따라 세그먼트 분리가 어떻게 달라지는지 분석하세요.", "difficulty": "★★"},
        ],
        [
            "Fader, P.S. et al. (2005). RFM and CLV: Using Iso-Value Curves for Customer Base Analysis",
            "Bult, J.R. & Wansbeek, T. (1995). Optimal Selection for Direct Mail",
        ],
        os.path.join(OUT, "ch08_10_practice_segmentation_problem.ipynb")
    )

    solution_notebook(CH, 10,
        "실전: 고객 세분화",
        [
            {"approach": "### 문제 1: 완전한 고객 세분화 파이프라인",
             "code": """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)
segments = {
    'VIP': {'R': (5, 10), 'F': (30, 50), 'M': (5000, 15000), 'n': 100},
    '활성': {'R': (10, 30), 'F': (10, 30), 'M': (1000, 5000), 'n': 300},
    '휴면': {'R': (100, 365), 'F': (1, 5), 'M': (100, 500), 'n': 400},
    '이탈 위험': {'R': (60, 120), 'F': (5, 15), 'M': (500, 2000), 'n': 200},
}
data = []
for seg_name, params in segments.items():
    for _ in range(params['n']):
        data.append({'Recency': np.random.uniform(*params['R']),
                     'Frequency': np.random.uniform(*params['F']),
                     'Monetary': np.random.uniform(*params['M']),
                     'Segment': seg_name})
df = pd.DataFrame(data)

# (a) 로그 변환 & 표준화
rfm_cols = ['Recency', 'Frequency', 'Monetary']
df_log = df[rfm_cols].apply(lambda x: np.log1p(x))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_log)

# (b) 최적 K (엘보 + 실루엣)
inertias, sil_scores = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(K_range), inertias, 'bo-')
axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia')
axes[0].set_title('엘보 방법')
axes[1].plot(list(K_range), sil_scores, 'rs-')
axes[1].set_xlabel('K'); axes[1].set_ylabel('실루엣 점수')
axes[1].set_title('실루엣 분석')
plt.tight_layout()
plt.show()

# (c) K-means (K=4)
km = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = km.fit_predict(X_scaled)

# (d) 프로파일
profile = df.groupby('Cluster')[rfm_cols].mean()
print("=== 세그먼트 프로파일 ===")
print(profile.round(1))

fig, ax = plt.subplots(figsize=(10, 6))
profile.plot(kind='bar', ax=ax)
ax.set_title('클러스터별 평균 RFM')
ax.set_xlabel('클러스터')
plt.tight_layout()
plt.show()""",
             "interpretation": "로그 변환으로 왜도를 줄이고, 표준화로 변수 간 스케일을 맞춘 후 클러스터링하면 더 의미 있는 세분화가 가능합니다. 실루엣과 엘보 모두 K=4를 지지합니다."},
            {"approach": "### 문제 2: PCA + 클러스터링",
             "code": """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)
segments = {
    'VIP': {'R': (5, 10), 'F': (30, 50), 'M': (5000, 15000), 'n': 100},
    '활성': {'R': (10, 30), 'F': (10, 30), 'M': (1000, 5000), 'n': 300},
    '휴면': {'R': (100, 365), 'F': (1, 5), 'M': (100, 500), 'n': 400},
    '이탈 위험': {'R': (60, 120), 'F': (5, 15), 'M': (500, 2000), 'n': 200},
}
data = []
for seg_name, params in segments.items():
    for _ in range(params['n']):
        data.append({'Recency': np.random.uniform(*params['R']),
                     'Frequency': np.random.uniform(*params['F']),
                     'Monetary': np.random.uniform(*params['M']),
                     'Segment': seg_name})
df = pd.DataFrame(data)
rfm_cols = ['Recency', 'Frequency', 'Monetary']
X_scaled = StandardScaler().fit_transform(df[rfm_cols].apply(np.log1p))

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 클러스터링 비교
km_orig = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_orig = km_orig.fit_predict(X_scaled)

km_pca = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_pca = km_pca.fit_predict(X_pca)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# 바이플롯
for label in range(4):
    mask = labels_orig == label
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], s=15, alpha=0.7, label=f'Cluster {label}')
for i, col in enumerate(rfm_cols):
    axes[0].arrow(0, 0, pca.components_[0, i]*2, pca.components_[1, i]*2,
                  head_width=0.08, fc='red', ec='red')
    axes[0].text(pca.components_[0, i]*2.3, pca.components_[1, i]*2.3, col, color='red', fontsize=11)
axes[0].set_title('원본 공간 클러스터링 + PCA 바이플롯')
axes[0].legend()

for label in range(4):
    mask = labels_pca == label
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], s=15, alpha=0.7, label=f'Cluster {label}')
axes[1].set_title('PCA 공간 클러스터링')
axes[1].legend()
plt.tight_layout()
plt.show()

sil_orig = silhouette_score(X_scaled, labels_orig)
sil_pca = silhouette_score(X_pca, labels_pca)
print(f"원본 공간 실루엣: {sil_orig:.4f}")
print(f"PCA 공간 실루엣: {sil_pca:.4f}")""",
             "interpretation": "3차원 RFM에서 PCA 2차원 축소는 정보 손실이 적으므로 두 접근법의 실루엣 점수가 유사합니다. 바이플롯을 통해 각 클러스터가 어떤 RFM 특성에 의해 분리되는지 시각적으로 확인할 수 있습니다."},
            {"approach": "### 문제 3: 다양한 클러스터링 방법 비교",
             "code": """import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score

np.random.seed(42)
segments_map = {'VIP': 0, '활성': 1, '휴면': 2, '이탈 위험': 3}
segments = {
    'VIP': {'R': (5, 10), 'F': (30, 50), 'M': (5000, 15000), 'n': 100},
    '활성': {'R': (10, 30), 'F': (10, 30), 'M': (1000, 5000), 'n': 300},
    '휴면': {'R': (100, 365), 'F': (1, 5), 'M': (100, 500), 'n': 400},
    '이탈 위험': {'R': (60, 120), 'F': (5, 15), 'M': (500, 2000), 'n': 200},
}
data = []
true_labels = []
for seg_name, params in segments.items():
    for _ in range(params['n']):
        data.append({'Recency': np.random.uniform(*params['R']),
                     'Frequency': np.random.uniform(*params['F']),
                     'Monetary': np.random.uniform(*params['M'])})
        true_labels.append(segments_map[seg_name])
df = pd.DataFrame(data)
y_true = np.array(true_labels)
X_scaled = StandardScaler().fit_transform(df.apply(np.log1p))

methods = {
    'K-means': KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_scaled),
    'GMM': GaussianMixture(n_components=4, random_state=42).fit_predict(X_scaled),
    'Ward': AgglomerativeClustering(n_clusters=4).fit_predict(X_scaled),
    'DBSCAN': DBSCAN(eps=0.8, min_samples=10).fit_predict(X_scaled),
}

print(f"{'방법':12s} {'ARI':>8s} {'실루엣':>8s} {'클러스터수':>10s}")
print("-" * 42)
for name, labels in methods.items():
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ari = adjusted_rand_score(y_true, labels)
    sil = silhouette_score(X_scaled, labels) if n_clusters > 1 else -1
    print(f"{name:12s} {ari:8.4f} {sil:8.4f} {n_clusters:10d}")""",
             "interpretation": "구형 클러스터에서는 K-means, GMM, Ward 모두 잘 작동합니다. DBSCAN은 eps 선택에 민감하며 구형 데이터에는 부적합할 수 있습니다. RFM 세분화에는 K-means 또는 GMM이 권장됩니다."},
            {"approach": "### 문제 4: 비즈니스 보고서 (레이더 차트)",
             "code": """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

np.random.seed(42)
segments = {
    'VIP': {'R': (5, 10), 'F': (30, 50), 'M': (5000, 15000), 'n': 100},
    '활성': {'R': (10, 30), 'F': (10, 30), 'M': (1000, 5000), 'n': 300},
    '휴면': {'R': (100, 365), 'F': (1, 5), 'M': (100, 500), 'n': 400},
    '이탈 위험': {'R': (60, 120), 'F': (5, 15), 'M': (500, 2000), 'n': 200},
}
data = []
for seg_name, params in segments.items():
    for _ in range(params['n']):
        data.append({'Recency': np.random.uniform(*params['R']),
                     'Frequency': np.random.uniform(*params['F']),
                     'Monetary': np.random.uniform(*params['M'])})
df = pd.DataFrame(data)
rfm_cols = ['Recency', 'Frequency', 'Monetary']
X_scaled = StandardScaler().fit_transform(df[rfm_cols].apply(np.log1p))

km = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = km.fit_predict(X_scaled)

# (a) 요약 통계
print("=== 세그먼트별 RFM 요약 ===")
summary = df.groupby('Cluster')[rfm_cols].agg(['mean', 'median', 'std']).round(1)
print(summary)

# (b) 레이더 차트
profile = df.groupby('Cluster')[rfm_cols].mean()
# Recency는 역수(낮을수록 좋음)
profile['Recency'] = profile['Recency'].max() - profile['Recency']
profile_norm = MinMaxScaler().fit_transform(profile)

categories = ['Recency\\n(역, 높을수록 최근)', 'Frequency', 'Monetary']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
cluster_names = ['세그먼트 A', '세그먼트 B', '세그먼트 C', '세그먼트 D']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i in range(4):
    values = list(profile_norm[i]) + [profile_norm[i][0]]
    ax.plot(angles, values, 'o-', linewidth=2, label=cluster_names[i], color=colors[i])
    ax.fill(angles, values, alpha=0.1, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_title('세그먼트 프로파일 (레이더 차트)', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# (c) 마케팅 전략
print("\\n=== 세그먼트별 전략 ===")
for cluster in range(4):
    seg = df[df['Cluster'] == cluster]
    avg_r = seg['Recency'].mean()
    avg_f = seg['Frequency'].mean()
    avg_m = seg['Monetary'].mean()
    print(f"\\n세그먼트 {cluster}: R={avg_r:.0f}일, F={avg_f:.0f}회, M={avg_m:.0f}원")
    if avg_r < 20 and avg_f > 25:
        print("  → VIP 세그먼트: 프리미엄 혜택, 전담 매니저 배정")
    elif avg_r < 40 and avg_f > 8:
        print("  → 활성 고객: 크로스셀/업셀 기회, 로열티 프로그램 강화")
    elif avg_r > 80:
        print("  → 휴면 고객: 재활성화 캠페인, 할인 쿠폰 발송")
    else:
        print("  → 이탈 위험: 설문 조사, 윈백 캠페인, 개인화 추천")""",
             "interpretation": "레이더 차트로 각 세그먼트의 특성을 직관적으로 파악할 수 있습니다. VIP는 모든 지표에서 우수하고, 휴면 고객은 모든 지표가 낮습니다. 세그먼트별 차별화된 전략이 필요합니다."},
            {"approach": "### 문제 5: 차원 축소 시각화 비교",
             "code": """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

np.random.seed(42)
segments = {
    'VIP': {'R': (5, 10), 'F': (30, 50), 'M': (5000, 15000), 'n': 100},
    '활성': {'R': (10, 30), 'F': (10, 30), 'M': (1000, 5000), 'n': 300},
    '휴면': {'R': (100, 365), 'F': (1, 5), 'M': (100, 500), 'n': 400},
    '이탈 위험': {'R': (60, 120), 'F': (5, 15), 'M': (500, 2000), 'n': 200},
}
data = []
for seg_name, params in segments.items():
    for _ in range(params['n']):
        data.append({'Recency': np.random.uniform(*params['R']),
                     'Frequency': np.random.uniform(*params['F']),
                     'Monetary': np.random.uniform(*params['M'])})
df = pd.DataFrame(data)
rfm_cols = ['Recency', 'Frequency', 'Monetary']
X_scaled = StandardScaler().fit_transform(df[rfm_cols].apply(np.log1p))

km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=15, alpha=0.7)
axes[0].set_title(f'PCA (설명 분산: {pca.explained_variance_ratio_.sum():.1%})')
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=15, alpha=0.7)
axes[1].set_title('t-SNE')
axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')
plt.suptitle('차원 축소 시각화 비교', fontsize=14)
plt.tight_layout()
plt.show()""",
             "interpretation": "PCA는 전역 구조(세그먼트 간 거리)를 보존하고, t-SNE는 국소 구조(세그먼트 내 점 밀집도)를 더 잘 보여줍니다. 비즈니스 보고서에는 PCA(해석 가능)를, 탐색적 분석에는 t-SNE(패턴 발견)를 사용하는 것이 좋습니다."},
        ],
        "고객 세분화는 비지도 학습의 대표적 비즈니스 응용입니다. 기술적 성능(실루엣 점수)뿐 아니라 비즈니스 해석 가능성과 실행 가능성(actionability)이 중요합니다. 정기적으로 세분화를 업데이트하고 A/B 테스트로 전략 효과를 검증하세요.",
        os.path.join(OUT, "ch08_10_practice_segmentation_solution.ipynb")
    )

    print("Chapter 08 전체 노트북 생성 완료!")

if __name__ == "__main__":
    gen_all()
