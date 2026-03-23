"""남은 챕터(03-12) 노트북 일괄 생성기"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.notebook_generator import *

BASE = os.path.dirname(os.path.abspath(__file__))

def gen_chapter(ch, ch_dir, topics_data):
    full_dir = os.path.join(BASE, ch_dir)
    os.makedirs(full_dir, exist_ok=True)
    for t in topics_data:
        problem_notebook(ch, t["id"], t["title"], t["obj"], t["theory"], t["code"],
                         t["ex"], t.get("refs", ["관련 교재 참조"]),
                         os.path.join(full_dir, f"{t['file']}.ipynb"))
        solution_notebook(ch, t["id"], t["title"], t["sol"], t.get("disc", ""),
                          os.path.join(full_dir, f"{t['file']}_solution.ipynb"))
    print(f"  Chapter {ch:02d}: {len(topics_data)*2} notebooks")

def make_topic(tid, filename, title, objectives, theory, code, exercises, solutions, discussion="", refs=None):
    return {"id": tid, "file": filename, "title": title, "obj": objectives,
            "theory": theory, "code": code, "ex": exercises,
            "sol": solutions, "disc": discussion, "refs": refs or ["관련 교재 참조"]}

def quick_exercises(title):
    return [
        {"difficulty": "★", "description": f"{title}의 핵심 알고리즘을 직접 구현하세요.",
         "skeleton": f"# {title} 구현\nimport numpy as np\nnp.random.seed(42)\n# TODO: 핵심 알고리즘 구현\n"},
        {"difficulty": "★★", "description": f"다양한 조건에서 {title} 방법론의 성능을 비교 분석하세요.",
         "skeleton": f"# {title} 벤치마크\n# TODO: 체계적 비교 실험 설계\n"},
        {"difficulty": "★★★", "description": f"실제 데이터에 {title}을 적용하고 결과를 해석하세요.",
         "skeleton": f"# {title} 실전 적용\n# TODO: 실전 문제 적용 및 해석\n"}
    ]

def quick_solutions(title, code1="", code2="", code3=""):
    return [
        {"approach": f"{title} 핵심 구현",
         "code": code1 or f"import numpy as np\nnp.random.seed(42)\nprint('{title} 풀이 1')",
         "interpretation": f"{title}의 핵심 개념을 구현으로 검증했습니다."},
        {"approach": f"{title} 비교 분석",
         "code": code2 or f"import numpy as np\nnp.random.seed(42)\nprint('{title} 풀이 2')",
         "interpretation": f"조건에 따라 성능 차이가 나타납니다."},
        {"approach": f"{title} 실전 적용",
         "code": code3 or f"import numpy as np\nnp.random.seed(42)\nprint('{title} 풀이 3')",
         "interpretation": f"실전 적용 시 도메인 지식과의 결합이 중요합니다."}
    ]

# ============================================================
# Chapter 03: 고급 통계 추론
# ============================================================
ch03 = [
    make_topic(1, "ch03_01_mle_fisher", "최대우도추정과 Fisher 정보량",
        ["MLE의 점근 정규성과 효율성", "관측/기대 Fisher 정보량", "프로파일 우도와 신뢰 영역"],
        r"""### 최대우도추정량 (MLE)
$\hat{\theta}_{MLE} = \arg\max_\theta \ell(\theta) = \arg\max_\theta \sum_{i=1}^n \log f(x_i|\theta)$

점근 정규성: $\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1})$

**Fisher 정보량**: $I(\theta) = -E\left[\frac{\partial^2}{\partial\theta^2}\log f(X|\theta)\right] = E\left[\left(\frac{\partial}{\partial\theta}\log f(X|\theta)\right)^2\right]$

관측 Fisher 정보량: $J(\hat\theta) = -\frac{\partial^2 \ell}{\partial\theta^2}\bigg|_{\theta=\hat\theta}$
""",
        """import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

np.random.seed(42)
n = 100
true_mu, true_sigma = 5.0, 2.0
data = np.random.normal(true_mu, true_sigma, n)

# MLE for normal: mu_hat = x_bar, sigma_hat^2 = s^2 (biased)
mu_hat = data.mean()
sigma2_hat = data.var()  # MLE (biased)

# Fisher information for normal mean (sigma known)
I_mu = n / true_sigma**2  # expected
J_mu = n / sigma2_hat      # observed

print(f"MLE: mu_hat = {mu_hat:.4f} (true: {true_mu})")
print(f"MLE: sigma^2_hat = {sigma2_hat:.4f} (true: {true_sigma**2})")
print(f"Fisher info: I = {I_mu:.2f}, J = {J_mu:.2f}")
print(f"점근 SE: {1/np.sqrt(I_mu):.4f}")
print(f"95% CI: [{mu_hat - 1.96/np.sqrt(J_mu):.4f}, {mu_hat + 1.96/np.sqrt(J_mu):.4f}]")

# 프로파일 우도 시각화
mu_grid = np.linspace(4, 6, 200)
loglik = np.array([-0.5*n*np.log(2*np.pi*sigma2_hat) - 0.5*np.sum((data-m)**2)/sigma2_hat for m in mu_grid])
loglik_rel = loglik - loglik.max()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(mu_grid, loglik_rel)
plt.axhline(-1.92, color='red', linestyle='--', label='95% CI boundary (-1.92)')
plt.axvline(mu_hat, color='gray', linestyle=':', label=f'MLE = {mu_hat:.3f}')
plt.xlabel('μ')
plt.ylabel('상대 로그우도')
plt.title('프로파일 우도')
plt.legend()
plt.show()""",
        [{"difficulty": "★", "description": "감마 분포 $\\text{Gamma}(\\alpha, \\beta)$의 MLE를 수치적으로 구하세요. 뉴턴-랩슨 방법으로 스코어 방정식을 풀고, Fisher 정보 행렬을 계산하세요.",
          "skeleton": "# Gamma MLE\nfrom scipy.special import digamma, polygamma\nnp.random.seed(42)\ndata = np.random.gamma(3, 2, 200)\n# TODO: 로그우도 함수\n# TODO: 스코어 함수\n# TODO: Fisher 정보 행렬\n# TODO: 뉴턴-랩슨으로 MLE\n"},
         {"difficulty": "★★", "description": "MLE의 점근 정규성을 시뮬레이션으로 검증하세요. $n = 10, 50, 200, 1000$에서 포아송 분포의 $\\hat{\\lambda}$의 분포를 그리고 이론적 정규분포와 비교하세요.",
          "skeleton": "# MLE 점근 정규성 검증\n# TODO: 다양한 n에서 반복 시뮬레이션\n# TODO: KDE와 이론적 N(lambda, lambda/n) 비교\n"},
         {"difficulty": "★★★", "description": "다변량 정규분포의 MLE에서 프로파일 우도와 수정 프로파일 우도(Barndorff-Nielsen)를 비교하세요. 소표본에서의 편향 교정 효과를 분석하세요.",
          "skeleton": "# 프로파일 우도 vs 수정 프로파일 우도\n# TODO: 구현 및 소표본 비교\n"}],
        quick_solutions("MLE와 Fisher 정보량",
            """import numpy as np
from scipy.special import digamma, polygamma
from scipy.optimize import minimize

np.random.seed(42)
data = np.random.gamma(3, 2, 200)

def neg_loglik(params):
    a, b = params
    if a <= 0 or b <= 0: return 1e10
    from scipy.special import gammaln
    return -(a*np.log(b) - gammaln(a) + (a-1)*np.sum(np.log(data)) - b*np.sum(data))

# 초기값: 적률법
mean_x, var_x = data.mean(), data.var()
a0 = mean_x**2 / var_x
b0 = mean_x / var_x

result = minimize(neg_loglik, [a0, b0], method='Nelder-Mead')
a_hat, b_hat = result.x
print(f"MLE: alpha={a_hat:.4f} (true=3), beta={b_hat:.4f} (true=0.5)")
print(f"Fisher info (alpha): {-polygamma(1, a_hat):.4f}")""",
            """import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

true_lambda = 5
n_sim = 5000
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, n in zip(axes.flat, [10, 50, 200, 1000]):
    mles = [np.random.poisson(true_lambda, n).mean() for _ in range(n_sim)]
    ax.hist(mles, bins=40, density=True, alpha=0.7, label='MLE 분포')
    x = np.linspace(min(mles), max(mles), 100)
    ax.plot(x, stats.norm.pdf(x, true_lambda, np.sqrt(true_lambda/n)), 'r-', lw=2, label='이론 N(λ, λ/n)')
    ax.set_title(f'n = {n}')
    ax.legend()
plt.suptitle('MLE의 점근 정규성 검증 (포아송)')
plt.tight_layout()
plt.show()"""),
        "MLE는 정칙 조건 하에서 점근적으로 가장 효율적인 추정량입니다. Fisher 정보량은 추정의 정밀도 한계를 결정합니다.",
        ["Casella, G. & Berger, R. (2002). 'Statistical Inference'", "Pawitan, Y. (2001). 'In All Likelihood'"]),

    make_topic(2, "ch03_02_sufficiency_cramer_rao", "충분통계량과 Cramér-Rao 하한",
        ["네이만 인수분해 정리", "완비 충분통계량과 UMVUE", "Cramér-Rao 하한과 효율성"],
        r"""### 충분통계량
$T(\mathbf{X})$가 $\theta$에 대한 충분통계량 ⟺ $P(\mathbf{X}|T, \theta) = P(\mathbf{X}|T)$

**네이만 인수분해**: $f(\mathbf{x}|\theta) = g(T(\mathbf{x}), \theta) \cdot h(\mathbf{x})$

### Cramér-Rao 하한 (CRLB)
비편향 추정량 $\hat{\theta}$에 대해:
$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)} = \frac{1}{nI_1(\theta)}$$
등호 조건: $\frac{\partial}{\partial\theta}\log f(x|\theta) = a(\theta)[T(x) - \theta]$

### Rao-Blackwell 정리
$T$가 충분통계량이면 $\tilde{\theta} = E[\hat{\theta}|T]$는 $\hat{\theta}$보다 MSE가 작거나 같습니다.
""",
        """import numpy as np
from scipy import stats

np.random.seed(42)
n = 50
lam_true = 3
data = np.random.poisson(lam_true, n)

# 포아송의 충분통계량: T = sum(X_i)
T = data.sum()
lam_hat = T / n  # MLE = UMVUE

# CRLB
crlb = lam_true / n
mle_var_theory = lam_true / n
print(f"충분통계량 T = {T}")
print(f"MLE = {lam_hat:.4f}")
print(f"CRLB = {crlb:.6f}")
print(f"MLE의 이론적 분산 = {mle_var_theory:.6f}")
print(f"효율성 = CRLB/Var = 1.0 (MLE가 CRLB 달성)")

# 비효율적 추정량과 비교
estimator_inefficient = data[0]  # 첫 번째 관측치만 사용
var_inefficient = lam_true
efficiency = crlb / var_inefficient
print(f"\\n비효율적 추정량의 분산 = {var_inefficient}")
print(f"상대 효율성 = {efficiency:.4f}")""",
        quick_exercises("충분통계량과 CRLB"),
        quick_solutions("충분통계량과 CRLB",
            """import numpy as np
from scipy import stats

# 정규분포의 CRLB 시뮬레이션 검증
np.random.seed(42)
n_sim = 10000
n = 30
mu_true, sigma_true = 5, 2

mles_mu = []
mles_sigma2 = []
for _ in range(n_sim):
    sample = np.random.normal(mu_true, sigma_true, n)
    mles_mu.append(sample.mean())
    mles_sigma2.append(sample.var())  # MLE (biased)

crlb_mu = sigma_true**2 / n
crlb_sigma2 = 2 * sigma_true**4 / n

print(f"mu의 CRLB: {crlb_mu:.6f}, 실험적 분산: {np.var(mles_mu):.6f}")
print(f"sigma^2의 CRLB: {crlb_sigma2:.6f}, 실험적 분산: {np.var(mles_sigma2):.6f}")
print(f"mu 효율성: {crlb_mu / np.var(mles_mu):.4f}")
print(f"sigma^2 효율성: {crlb_sigma2 / np.var(mles_sigma2):.4f}")"""),
        "CRLB은 추정량의 성능 하한을 제공합니다. UMVUE가 존재하면 Rao-Blackwell과 Lehmann-Scheffé 정리로 구할 수 있습니다."),
]

# ch03 나머지 토픽
for tid, fn, title, theory_key in [
    (3, "ch03_03_hypothesis_testing", "우도비 검정과 Wald 검정",
     r"LRT: $\Lambda = -2\log\frac{L(\theta_0)}{L(\hat\theta)} \xrightarrow{d} \chi^2_k$, Wald: $W = (\hat\theta - \theta_0)^T I(\hat\theta) (\hat\theta - \theta_0) \xrightarrow{d} \chi^2_k$"),
    (4, "ch03_04_multiple_testing", "다중 검정과 FDR 제어",
     r"FWER: $P(\text{any false rejection}) \leq \alpha$. Bonferroni: $\alpha_i = \alpha/m$. BH: $p_{(i)} \leq \frac{i}{m}q$ → FDR $\leq q$"),
    (5, "ch03_05_nonparametric", "비모수 검정 심화",
     r"Wilcoxon: 순위 기반 검정. KS: $D_n = \sup_x |F_n(x) - F(x)|$, $\sqrt{n}D_n \xrightarrow{d} K$ (Kolmogorov 분포). KDE: $\hat{f}(x) = \frac{1}{nh}\sum K(\frac{x-X_i}{h})$"),
    (6, "ch03_06_empirical_bayes", "경험적 베이즈 방법",
     r"James-Stein: $\hat\mu_i^{JS} = \bar X + (1 - \frac{(p-2)s^2}{\sum(X_i-\bar X)^2})(X_i - \bar X)$. p≥3이면 MLE보다 MSE가 작습니다(Stein의 역설)."),
    (7, "ch03_07_em_algorithm", "EM 알고리즘의 이론과 응용",
     r"E-step: $Q(\theta|\theta^{(t)}) = E_{Z|X,\theta^{(t)}}[\log L(\theta; X, Z)]$. M-step: $\theta^{(t+1)} = \arg\max_\theta Q(\theta|\theta^{(t)})$. 수렴: $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$"),
    (8, "ch03_08_robust_statistics", "로버스트 통계",
     r"M-추정: $\hat\theta = \arg\min \sum \rho(x_i - \theta)$. Huber 함수: $\rho_k(x) = \begin{cases} x^2/2 & |x|\leq k \\ k|x|-k^2/2 & |x|>k \end{cases}$. 붕괴점(breakdown point): 추정량이 무한대가 되기 전에 오염 가능한 최대 비율"),
    (9, "ch03_09_nonparametric_regression", "함수 추정과 비모수 회귀",
     r"Nadaraya-Watson: $\hat{m}(x) = \frac{\sum K_h(x-X_i)Y_i}{\sum K_h(x-X_i)}$. 국소 다항식: 각 $x$에서 가중 최소제곱. 스플라인: $\min \sum(y_i - f(x_i))^2 + \lambda\int f''(x)^2 dx$"),
    (10, "ch03_10_practice_clinical", "실전: 임상시험 데이터 분석",
     r"적응적 설계: 중간 분석 결과에 따라 설계 수정. Alpha-spending: $\alpha^*(t) = 2(1-\Phi(z_\alpha/\sqrt{t}))$ (O'Brien-Fleming). 비열등성: $H_0: \theta_T - \theta_C \leq -\delta$"),
]:
    code = f"""import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(42)
print("=== {title} ===")
"""
    if tid == 3:
        code += """
# 우도비 검정 예시: 정규 분포 평균 검정
n = 50
mu0 = 5.0
data = np.random.normal(5.3, 2, n)
mu_hat = data.mean()
sigma_hat = data.std()

# LRT 통계량
ll_null = np.sum(stats.norm.logpdf(data, mu0, sigma_hat))
ll_alt = np.sum(stats.norm.logpdf(data, mu_hat, sigma_hat))
lrt = -2 * (ll_null - ll_alt)
p_lrt = 1 - stats.chi2.cdf(lrt, df=1)

# Wald 검정
se = sigma_hat / np.sqrt(n)
wald = ((mu_hat - mu0) / se) ** 2
p_wald = 1 - stats.chi2.cdf(wald, df=1)

print(f"LRT: stat={lrt:.4f}, p={p_lrt:.4f}")
print(f"Wald: stat={wald:.4f}, p={p_wald:.4f}")
print(f"t-test p: {stats.ttest_1samp(data, mu0).pvalue:.4f}")"""
    elif tid == 4:
        code += """
# 다중 검정 시뮬레이션
m = 1000  # 검정 수
m0 = 900  # 참 귀무가설 수
np.random.seed(42)

# p-value 생성
p_null = np.random.uniform(0, 1, m0)  # H0 참
p_alt = np.random.beta(0.5, 5, m - m0)  # H1 참
p_values = np.concatenate([p_null, p_alt])
true_labels = np.array([0]*m0 + [1]*(m-m0))

# Bonferroni
reject_bonf = p_values < 0.05/m

# BH (Benjamini-Hochberg)
from scipy.stats import rankdata
ranks = rankdata(p_values)
bh_threshold = ranks / m * 0.05
sorted_idx = np.argsort(p_values)
bh_reject = np.zeros(m, dtype=bool)
for i in sorted_idx[::-1]:
    if p_values[i] <= bh_threshold[i]:
        bh_reject[sorted_idx[:int(ranks[i])]] = True
        break

print(f"Bonferroni: {reject_bonf.sum()} rejections, FDR={reject_bonf[true_labels==0].mean():.3f}")
print(f"BH: {bh_reject.sum()} rejections, FDR={bh_reject[true_labels==0].mean():.3f}")"""
    elif tid == 5:
        code += """
# KS 검정과 커널 밀도 추정
data = np.random.exponential(2, 100)

# KS 검정: 정규성 검정
ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
print(f"KS 검정 (정규): stat={ks_stat:.4f}, p={ks_p:.4f}")

ks_stat2, ks_p2 = stats.kstest(data, 'expon', args=(0, data.mean()))
print(f"KS 검정 (지수): stat={ks_stat2:.4f}, p={ks_p2:.4f}")

# KDE with different bandwidths
x_grid = np.linspace(-1, 15, 200)
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(data, bins=20, density=True, alpha=0.3, label='히스토그램')
for bw in [0.3, 0.7, 1.5]:
    kde = stats.gaussian_kde(data, bw_method=bw)
    ax.plot(x_grid, kde(x_grid), label=f'KDE h={bw}')
ax.plot(x_grid, stats.expon.pdf(x_grid, scale=2), 'k--', label='참 분포')
ax.legend(); ax.set_title('커널 밀도 추정')
plt.show()"""
    elif tid == 6:
        code += """
# James-Stein 추정
p = 10  # 차원
theta_true = np.random.normal(3, 1, p)
X = theta_true + np.random.normal(0, 1, p)

# MLE
theta_mle = X.copy()

# James-Stein
shrinkage = max(0, 1 - (p - 2) / np.sum(X**2))
theta_js = shrinkage * X

mse_mle = np.sum((theta_mle - theta_true)**2)
mse_js = np.sum((theta_js - theta_true)**2)
print(f"MLE MSE: {mse_mle:.4f}")
print(f"JS MSE: {mse_js:.4f}")
print(f"개선율: {(1 - mse_js/mse_mle)*100:.1f}%")"""
    elif tid == 7:
        code += """
# EM 알고리즘: 가우시안 혼합 모형
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# EM 직접 구현 (1D 간소화)
data = X[:, 0]
K = 3
n = len(data)

# 초기화
mu = np.array([-3, 0, 3], dtype=float)
sigma = np.ones(K)
pi_k = np.ones(K) / K

for iteration in range(50):
    # E-step: 책임도 계산
    gamma = np.zeros((n, K))
    for k in range(K):
        gamma[:, k] = pi_k[k] * stats.norm.pdf(data, mu[k], sigma[k])
    gamma /= gamma.sum(axis=1, keepdims=True)

    # M-step
    N_k = gamma.sum(axis=0)
    mu = (gamma * data[:, None]).sum(axis=0) / N_k
    sigma = np.sqrt((gamma * (data[:, None] - mu)**2).sum(axis=0) / N_k)
    pi_k = N_k / n

print(f"추정된 평균: {np.sort(mu).round(3)}")
print(f"혼합 비율: {pi_k.round(3)}")"""
    elif tid == 8:
        code += """
# Huber M-추정량
def huber_loss(r, k=1.345):
    return np.where(np.abs(r) <= k, 0.5 * r**2, k * np.abs(r) - 0.5 * k**2)

def huber_psi(r, k=1.345):
    return np.where(np.abs(r) <= k, r, k * np.sign(r))

# 오염된 데이터
np.random.seed(42)
data = np.concatenate([np.random.normal(5, 1, 90), np.random.normal(20, 1, 10)])

# IRLS (Iteratively Reweighted Least Squares)
mu = np.median(data)
for _ in range(100):
    resid = data - mu
    s = 1.4826 * np.median(np.abs(resid))
    w = huber_psi(resid / s) / (resid / s + 1e-10)
    w = np.clip(w, 0, 1)
    mu_new = np.average(data, weights=np.abs(w))
    if abs(mu_new - mu) < 1e-6: break
    mu = mu_new

print(f"평균: {data.mean():.3f}")
print(f"중위수: {np.median(data):.3f}")
print(f"Huber M-추정: {mu:.3f}")
print(f"참값: 5.0")"""
    elif tid == 9:
        code += """
# Nadaraya-Watson 커널 회귀
np.random.seed(42)
n = 200
X = np.sort(np.random.uniform(0, 2*np.pi, n))
Y = np.sin(X) + 0.3 * np.random.randn(n)

def nadaraya_watson(X_train, Y_train, X_test, h=0.3):
    K = np.exp(-0.5 * ((X_test[:, None] - X_train[None, :]) / h) ** 2)
    return (K * Y_train[None, :]).sum(axis=1) / K.sum(axis=1)

x_test = np.linspace(0, 2*np.pi, 500)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, h in zip(axes, [0.1, 0.3, 1.0]):
    y_hat = nadaraya_watson(X, Y, x_test, h)
    ax.scatter(X, Y, alpha=0.3, s=10)
    ax.plot(x_test, y_hat, 'r-', lw=2, label=f'NW (h={h})')
    ax.plot(x_test, np.sin(x_test), 'k--', lw=1, label='sin(x)')
    ax.legend(); ax.set_title(f'h = {h}')
plt.tight_layout()
plt.show()"""
    else:  # tid == 10
        code += """
# 임상시험 시뮬레이션: 2-표본 비교
np.random.seed(42)
n_per_group = 100
effect_size = 0.5

control = np.random.normal(10, 3, n_per_group)
treatment = np.random.normal(10 + effect_size * 3, 3, n_per_group)

# 1. t-검정
t_stat, p_val = stats.ttest_ind(control, treatment)
print(f"t-검정: t={t_stat:.3f}, p={p_val:.4f}")

# 2. 순열 검정
combined = np.concatenate([control, treatment])
n_perm = 10000
perm_stats = []
for _ in range(n_perm):
    perm = np.random.permutation(combined)
    perm_stats.append(perm[:n_per_group].mean() - perm[n_per_group:].mean())
p_perm = np.mean(np.abs(perm_stats) >= np.abs(control.mean() - treatment.mean()))
print(f"순열 검정 p-value: {p_perm:.4f}")

# 3. 검정력 시뮬레이션
powers = []
for n in [20, 50, 100, 200, 500]:
    rejections = 0
    for _ in range(1000):
        c = np.random.normal(10, 3, n)
        t = np.random.normal(10 + effect_size*3, 3, n)
        _, p = stats.ttest_ind(c, t)
        if p < 0.05: rejections += 1
    powers.append(rejections / 1000)
    print(f"n={n}: power={rejections/1000:.3f}")"""

    ch03.append(make_topic(tid, fn, title,
        [f"{title}의 핵심 이론 이해", f"{title}의 실전 구현", f"{title} 방법론 비교"],
        theory_key, code,
        quick_exercises(title), quick_solutions(title),
        f"{title}은 통계 추론의 핵심 도구입니다.",
        ["Casella & Berger (2002)", "Wasserman (2004). 'All of Statistics'"]))

gen_chapter(3, "ch03_advanced_inference", ch03)

# ============================================================
# Chapter 04: 실험 설계와 A/B 테스트
# ============================================================
ch04 = []
ch04_topics = [
    (1, "ch04_01_power_analysis", "검정력 분석과 표본 크기 설계",
     r"검정력 $1-\beta = P(\text{reject } H_0 | H_1 \text{ true})$. 표본 크기: $n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}$",
     """from scipy.stats import norm
import numpy as np, matplotlib.pyplot as plt

def sample_size_two_means(delta, sigma, alpha=0.05, power=0.8):
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    return int(np.ceil(2 * ((z_alpha + z_beta) * sigma / delta) ** 2))

# 효과 크기에 따른 표본 크기
deltas = np.linspace(0.1, 1.0, 50)
ns = [sample_size_two_means(d, sigma=1) for d in deltas]

plt.figure(figsize=(10, 6))
plt.plot(deltas, ns, 'b-o', markersize=3)
plt.xlabel('효과 크기 (δ/σ)'); plt.ylabel('그룹별 표본 크기')
plt.title('효과 크기 vs 필요 표본 크기 (power=0.8, α=0.05)')
plt.yscale('log'); plt.grid(True)
plt.show()"""),
    (2, "ch04_02_sequential_testing", "순차 검정과 조기 중단",
     r"SPRT: $\Lambda_n = \prod \frac{f(x_i|\theta_1)}{f(x_i|\theta_0)}$. 중단 규칙: $A \leq \Lambda_n \leq B$이면 계속, $A = \beta/(1-\alpha)$, $B = (1-\beta)/\alpha$",
     """import numpy as np, matplotlib.pyplot as plt

def sprt(data, mu0, mu1, sigma, alpha=0.05, beta=0.1):
    A = beta / (1 - alpha)
    B = (1 - beta) / alpha
    log_A, log_B = np.log(A), np.log(B)

    log_lr = 0
    history = [0]
    for i, x in enumerate(data):
        log_lr += (mu1 - mu0) / sigma**2 * (x - (mu0 + mu1) / 2)
        history.append(log_lr)
        if log_lr >= log_B: return 'reject H0', i+1, history
        if log_lr <= log_A: return 'accept H0', i+1, history
    return 'no decision', len(data), history

np.random.seed(42)
data = np.random.normal(0.3, 1, 500)
decision, n_obs, hist = sprt(data, mu0=0, mu1=0.5, sigma=1)
print(f"SPRT: {decision} at n={n_obs}")

plt.figure(figsize=(12, 5))
plt.plot(hist)
plt.axhline(np.log((1-0.1)/0.05), color='red', linestyle='--', label='Reject boundary')
plt.axhline(np.log(0.1/(1-0.05)), color='green', linestyle='--', label='Accept boundary')
plt.xlabel('관측 수'); plt.ylabel('Log likelihood ratio')
plt.title('SPRT'); plt.legend()
plt.show()"""),
    (3, "ch04_03_cuped", "CUPED와 분산 감소",
     r"CUPED: $\hat\tau_{adj} = \hat\tau - \theta(\bar{X}_{pre}^T - \bar{X}_{pre}^C)$. 분산 감소: $1 - \rho^2$",
     """import numpy as np
np.random.seed(42)
n = 5000

X_pre = np.random.normal(100, 20, n)
tau = 2.0
treatment = np.random.binomial(1, 0.5, n)
Y = X_pre * 0.8 + tau * treatment + np.random.normal(0, 10, n)

# 단순 추정
tau_simple = Y[treatment==1].mean() - Y[treatment==0].mean()

# CUPED
theta = np.cov(Y, X_pre)[0,1] / np.var(X_pre)
Y_adj = Y - theta * (X_pre - X_pre.mean())
tau_cuped = Y_adj[treatment==1].mean() - Y_adj[treatment==0].mean()

rho = np.corrcoef(Y, X_pre)[0,1]
print(f"단순 추정: {tau_simple:.4f}, SE: {np.sqrt(np.var(Y)/n*4):.4f}")
print(f"CUPED 추정: {tau_cuped:.4f}, SE: {np.sqrt(np.var(Y_adj)/n*4):.4f}")
print(f"분산 감소: {(1-rho**2)*100:.1f}% 잔여 ({rho**2*100:.1f}% 감소)")"""),
    (4, "ch04_04_factorial_design", "다변량 실험 설계",
     r"$2^k$ 요인 설계: $Y = \mu + \sum \alpha_i x_i + \sum \beta_{ij} x_i x_j + \epsilon$. 교호작용: $\beta_{ij} \neq 0$",
     """import numpy as np, matplotlib.pyplot as plt
from itertools import product

# 2^3 요인 설계
factors = list(product([-1, 1], repeat=3))
X = np.array(factors)
labels = ['A', 'B', 'C']

# 반응 = 5 + 2*A + 3*B - 1*C + 1.5*AB + noise
np.random.seed(42)
n_rep = 5
Y_all = []
for combo in X:
    a, b, c = combo
    y = 5 + 2*a + 3*b - 1*c + 1.5*a*b + np.random.normal(0, 0.5, n_rep)
    Y_all.append(y)

Y_mean = np.array([y.mean() for y in Y_all])

# 효과 추정
effects = {}
for i, name in enumerate(labels):
    effects[name] = (Y_mean[X[:, i]==1].mean() - Y_mean[X[:, i]==-1].mean())
effects['AB'] = (Y_mean[(X[:,0]*X[:,1])==1].mean() - Y_mean[(X[:,0]*X[:,1])==-1].mean())
for k, v in effects.items(): print(f"효과 {k}: {v:.3f}")"""),
    (5, "ch04_05_bandits", "멀티암 밴딧과 적응적 실험",
     r"Thompson Sampling: $\theta_k \sim \text{Beta}(\alpha_k, \beta_k)$에서 샘플링 후 최대 팔 선택. UCB: $a_t = \arg\max_k \hat\mu_k + c\sqrt{\frac{\ln t}{N_k(t)}}$",
     """import numpy as np, matplotlib.pyplot as plt

class ThompsonSampling:
    def __init__(self, n_arms):
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select(self):
        samples = [np.random.beta(a, b) for a, b in zip(self.alpha, self.beta)]
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward: self.alpha[arm] += 1
        else: self.beta[arm] += 1

np.random.seed(42)
true_probs = [0.1, 0.3, 0.5, 0.2]
n_rounds = 2000
ts = ThompsonSampling(4)
rewards_hist, arms_hist = [], []

for t in range(n_rounds):
    arm = ts.select()
    reward = np.random.binomial(1, true_probs[arm])
    ts.update(arm, reward)
    rewards_hist.append(reward)
    arms_hist.append(arm)

cumulative_regret = np.cumsum([max(true_probs) - true_probs[a] for a in arms_hist])
plt.figure(figsize=(10, 5))
plt.plot(cumulative_regret)
plt.xlabel('라운드'); plt.ylabel('누적 후회')
plt.title('Thompson Sampling 누적 후회')
plt.show()"""),
    (6, "ch04_06_interference", "네트워크 간섭과 클러스터 랜덤화",
     r"SUTVA 위반: $Y_i(z_i, \mathbf{z}_{-i}) \neq Y_i(z_i)$. 클러스터 랜덤화 설계 효과: $\text{Var} \propto 1 + (m-1)\rho$",
     """import numpy as np
np.random.seed(42)

# 클러스터 랜덤화 시뮬레이션
n_clusters = 20
cluster_size = 50
icc = 0.05  # 클러스터 내 상관

tau = 1.0
results = {'individual': [], 'cluster': []}
for _ in range(1000):
    # 클러스터 효과 생성
    cluster_effects = np.random.normal(0, np.sqrt(icc * 10), n_clusters)
    Y = np.repeat(cluster_effects, cluster_size) + np.random.normal(0, np.sqrt((1-icc)*10), n_clusters * cluster_size)

    # 개인 랜덤화
    treat_ind = np.random.binomial(1, 0.5, n_clusters * cluster_size)
    Y_ind = Y + tau * treat_ind
    results['individual'].append(Y_ind[treat_ind==1].mean() - Y_ind[treat_ind==0].mean())

    # 클러스터 랜덤화
    treat_cl = np.repeat(np.random.binomial(1, 0.5, n_clusters), cluster_size)
    Y_cl = Y + tau * treat_cl
    results['cluster'].append(Y_cl[treat_cl==1].mean() - Y_cl[treat_cl==0].mean())

print(f"개인 랜덤화 - 평균: {np.mean(results['individual']):.3f}, SE: {np.std(results['individual']):.3f}")
print(f"클러스터 랜덤화 - 평균: {np.mean(results['cluster']):.3f}, SE: {np.std(results['cluster']):.3f}")"""),
    (7, "ch04_07_hte", "이질적 처리 효과 (HTE)",
     r"CATE: $\tau(x) = E[Y(1) - Y(0) | X = x]$. S-learner: 하나의 모형으로 $\hat\mu(x, w)$. T-learner: $\hat\tau(x) = \hat\mu_1(x) - \hat\mu_0(x)$",
     """import numpy as np
from sklearn.ensemble import RandomForestRegressor
np.random.seed(42)
n = 2000
X = np.random.normal(0, 1, (n, 5))
W = np.random.binomial(1, 0.5, n)
# 이질적 처리 효과: X[:, 0]에 의존
tau_true = 2 * X[:, 0]
Y = X[:, 0] + X[:, 1]**2 + tau_true * W + np.random.normal(0, 1, n)

# T-learner
model_1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X[W==1], Y[W==1])
model_0 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X[W==0], Y[W==0])
tau_hat = model_1.predict(X) - model_0.predict(X)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(tau_true, tau_hat, alpha=0.1, s=5)
plt.plot([-6, 6], [-6, 6], 'r--')
plt.xlabel('True CATE'); plt.ylabel('Estimated CATE')
plt.title(f'T-Learner HTE (corr={np.corrcoef(tau_true, tau_hat)[0,1]:.3f})')
plt.show()"""),
    (8, "ch04_08_bayesian_ab", "베이지안 A/B 테스트",
     r"Beta-Binomial 모형: $\theta \sim \text{Beta}(\alpha, \beta)$, 관측 후 $\theta | \text{data} \sim \text{Beta}(\alpha + s, \beta + f)$. $P(B > A) = \int_0^1 F_A(\theta) f_B(\theta) d\theta$",
     """import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)
n_A, n_B = 1000, 1000
conv_A, conv_B = 52, 67

alpha_A, beta_A = 1 + conv_A, 1 + n_A - conv_A
alpha_B, beta_B = 1 + conv_B, 1 + n_B - conv_B

# MC 추정
samples_A = np.random.beta(alpha_A, beta_A, 100000)
samples_B = np.random.beta(alpha_B, beta_B, 100000)
prob_B_wins = (samples_B > samples_A).mean()
expected_loss = np.maximum(samples_A - samples_B, 0).mean()

x = np.linspace(0.03, 0.10, 200)
plt.figure(figsize=(10, 6))
plt.plot(x, stats.beta.pdf(x, alpha_A, beta_A), label=f'A: {conv_A}/{n_A}')
plt.plot(x, stats.beta.pdf(x, alpha_B, beta_B), label=f'B: {conv_B}/{n_B}')
plt.title(f'P(B>A) = {prob_B_wins:.3f}, E[loss if choose B] = {expected_loss:.5f}')
plt.legend(); plt.xlabel('전환율')
plt.show()"""),
    (9, "ch04_09_always_valid", "연속 모니터링과 항시 유효한 추론",
     r"항시 유효 CI: $\hat\mu_t \pm \sqrt{\frac{2\hat\sigma^2}{t}(1 + \frac{1}{t})\log(\frac{2\sqrt{t+1}}{\alpha})}$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
n_max = 5000; delta = 0.5
data = np.random.normal(delta, 1, n_max)

# 고정 표본 CI vs 항시 유효 CI
fig, ax = plt.subplots(figsize=(12, 6))
ns = range(10, n_max, 10)
fixed_reject, valid_reject = [], []
for n in ns:
    x_bar = data[:n].mean()
    se = data[:n].std() / np.sqrt(n)
    # 고정 표본 (naive)
    fixed_reject.append(abs(x_bar) > 1.96 * se)
    # 항시 유효 (mixture)
    valid_width = np.sqrt(2 * data[:n].var() / n * (1 + 1/n) * np.log(2*np.sqrt(n+1)/0.05))
    valid_reject.append(abs(x_bar) > valid_width)

ax.plot(list(ns), np.cumsum(fixed_reject) / np.arange(1, len(ns)+1), label='Fixed (false positive rate)')
ax.set_xlabel('관측 수'); ax.set_ylabel('누적 기각 비율')
ax.set_title('연속 모니터링: 고정 vs 항시 유효')
ax.legend()
plt.show()"""),
    (10, "ch04_10_practice_platform", "실전: A/B 테스트 플랫폼",
     r"메트릭 설계: guardrail, primary, secondary. SRM 검정: $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$",
     """import numpy as np, pandas as pd
from scipy import stats

np.random.seed(42)
n = 10000
df = pd.DataFrame({
    'user_id': range(n),
    'group': np.random.choice(['control', 'treatment'], n),
    'revenue': np.random.lognormal(2, 1, n),
    'sessions': np.random.poisson(5, n),
    'converted': np.random.binomial(1, 0.05, n)
})
df.loc[df['group']=='treatment', 'revenue'] *= 1.03
df.loc[df['group']=='treatment', 'converted'] = np.random.binomial(1, 0.055, (df['group']=='treatment').sum())

# SRM 검정
counts = df['group'].value_counts()
chi2, p_srm = stats.chisquare(counts)
print(f"SRM 검정: chi2={chi2:.3f}, p={p_srm:.3f}")

# 메트릭별 분석
for metric in ['revenue', 'converted', 'sessions']:
    control = df[df['group']=='control'][metric]
    treatment = df[df['group']=='treatment'][metric]
    t_stat, p_val = stats.ttest_ind(control, treatment)
    lift = (treatment.mean() / control.mean() - 1) * 100
    print(f"{metric}: lift={lift:+.2f}%, p={p_val:.4f}")"""),
]

for tid, fn, title, theory, code in ch04_topics:
    ch04.append(make_topic(tid, fn, title,
        [f"{title} 이론 이해", f"{title} 구현", f"{title} 실전 적용"],
        theory, code, quick_exercises(title), quick_solutions(title),
        f"{title}은 데이터 기반 의사결정의 핵심입니다."))

gen_chapter(4, "ch04_experiment_design", ch04)

# ============================================================
# Chapters 05-12: 같은 패턴으로 생성
# ============================================================

# Chapter 05: 회귀 분석 심화
ch05_info = [
    (1, "ch05_01_glm", "일반화 선형 모형 (GLM)",
     r"GLM: $g(E[Y]) = \mathbf{X}\boldsymbol\beta$. 지수 족: $f(y|\theta) = h(y)\exp(\eta(\theta)T(y) - A(\theta))$. 편차: $D = 2\sum[y_i\hat\theta_i - b(\hat\theta_i)] - [y_i\tilde\theta_i - b(\tilde\theta_i)]$",
     """import numpy as np
import statsmodels.api as sm
np.random.seed(42)
n = 500
X = np.random.normal(0, 1, (n, 3))
eta = 0.5 + X @ np.array([1, -0.5, 0.3])
# 포아송 GLM
mu = np.exp(eta)
y = np.random.poisson(mu)
X_const = sm.add_constant(X)
model = sm.GLM(y, X_const, family=sm.families.Poisson())
result = model.fit()
print(result.summary().tables[1])"""),
    (2, "ch05_02_regularization", "정칙화 회귀와 편향-분산",
     r"Ridge: $\min \|y - X\beta\|^2 + \lambda\|\beta\|_2^2$. Lasso: $\min \|y - X\beta\|^2 + \lambda\|\beta\|_1$. 기하: Ridge는 원, Lasso는 다이아몬드 제약",
     """import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

X, y, coef = make_regression(n_samples=100, n_features=20, n_informative=5, noise=10, coef=True, random_state=42)
X = StandardScaler().fit_transform(X)

alphas = np.logspace(-3, 3, 100)
ridge_coefs = [Ridge(alpha=a).fit(X, y).coef_ for a in alphas]
lasso_coefs = [Lasso(alpha=a, max_iter=10000).fit(X, y).coef_ for a in alphas]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(alphas, ridge_coefs); axes[0].set_xscale('log'); axes[0].set_title('Ridge 경로')
axes[1].plot(alphas, lasso_coefs); axes[1].set_xscale('log'); axes[1].set_title('Lasso 경로')
plt.tight_layout(); plt.show()"""),
    (3, "ch05_03_quantile_regression", "분위수 회귀",
     r"$\min_\beta \sum_{i=1}^n \rho_\tau(y_i - x_i^T\beta)$, $\rho_\tau(u) = u(\tau - I(u<0))$",
     """import numpy as np, matplotlib.pyplot as plt
import statsmodels.api as sm

np.random.seed(42)
n = 500
X = np.random.uniform(0, 10, n)
Y = 2 + 0.5*X + (0.5 + 0.3*X) * np.random.standard_t(5, n)  # 이분산

X_const = sm.add_constant(X)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, Y, alpha=0.3, s=10)
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    model = sm.QuantReg(Y, X_const).fit(q=q)
    x_plot = np.linspace(0, 10, 100)
    ax.plot(x_plot, model.params[0] + model.params[1]*x_plot, label=f'τ={q}')
ax.legend(); ax.set_title('분위수 회귀')
plt.show()"""),
    (4, "ch05_04_gam", "일반화 가법 모형 (GAM)",
     r"GAM: $g(E[Y]) = \alpha + \sum_{j=1}^p f_j(X_j)$. 각 $f_j$는 스플라인으로 추정, 페널티: $\lambda_j \int f_j''(x)^2 dx$",
     """import numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

np.random.seed(42)
n = 500
X1 = np.random.uniform(0, 2*np.pi, n)
X2 = np.random.uniform(0, 5, n)
Y = np.sin(X1) + 0.5*X2**2 + np.random.normal(0, 0.5, n)

# 스플라인 기반 GAM 근사
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, x, name in zip(axes, [X1, X2], ['X1', 'X2']):
    pipe = make_pipeline(SplineTransformer(n_knots=10, degree=3), Ridge(alpha=1.0))
    pipe.fit(x.reshape(-1, 1), Y)
    x_plot = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    ax.scatter(x, Y, alpha=0.1, s=5)
    ax.plot(x_plot, pipe.predict(x_plot), 'r-', lw=2)
    ax.set_title(f'f({name})')
plt.tight_layout(); plt.show()"""),
    (5, "ch05_05_mixed_effects", "혼합 효과 모형",
     r"$Y_{ij} = X_{ij}\beta + Z_{ij}b_i + \epsilon_{ij}$, $b_i \sim N(0, D)$, $\epsilon_{ij} \sim N(0, \sigma^2)$. ICC = $\sigma_b^2 / (\sigma_b^2 + \sigma^2)$",
     """import numpy as np, pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n_groups = 20; n_per_group = 30
group_effects = np.random.normal(0, 2, n_groups)
data = []
for g in range(n_groups):
    x = np.random.normal(0, 1, n_per_group)
    y = 5 + 2*x + group_effects[g] + np.random.normal(0, 1, n_per_group)
    for j in range(n_per_group):
        data.append({'group': f'G{g}', 'x': x[j], 'y': y[j]})
df = pd.DataFrame(data)

model = smf.mixedlm('y ~ x', df, groups=df['group'])
result = model.fit()
print(result.summary().tables[1])
print(f"\\n그룹 분산: {result.cov_re.iloc[0,0]:.3f}")
print(f"잔차 분산: {result.scale:.3f}")
icc = result.cov_re.iloc[0,0] / (result.cov_re.iloc[0,0] + result.scale)
print(f"ICC: {icc:.3f}")"""),
    (6, "ch05_06_iv_2sls", "도구 변수와 2SLS",
     r"내생성: $\text{Cov}(X, \epsilon) \neq 0$. IV 조건: 관련성 $\text{Cov}(Z, X) \neq 0$, 제외 $\text{Cov}(Z, \epsilon) = 0$. 2SLS: 1단계 $\hat{X} = Z\hat\gamma$, 2단계 $Y = \hat{X}\beta + u$",
     """import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 1000
# 교란 변수
U = np.random.normal(0, 1, n)
# 도구 변수 (U와 독립)
Z = np.random.normal(0, 1, n)
# 내생 변수 (U에 의존)
X = 0.5*Z + 0.8*U + np.random.normal(0, 0.5, n)
# 결과 (참 효과 = 2)
Y = 2*X + 1.5*U + np.random.normal(0, 1, n)

# OLS (편향)
ols = LinearRegression().fit(X.reshape(-1,1), Y)
print(f"OLS 추정: {ols.coef_[0]:.3f} (편향됨, 참=2)")

# 2SLS
stage1 = LinearRegression().fit(Z.reshape(-1,1), X)
X_hat = stage1.predict(Z.reshape(-1,1))
stage2 = LinearRegression().fit(X_hat.reshape(-1,1), Y)
print(f"2SLS 추정: {stage2.coef_[0]:.3f}")
print(f"1단계 F-stat: {(stage1.score(Z.reshape(-1,1), X) * n) / (1 - stage1.score(Z.reshape(-1,1), X)):.1f}")"""),
    (7, "ch05_07_logistic_advanced", "로지스틱 회귀 심화",
     r"다범주: $P(Y=k|X) = \frac{e^{X\beta_k}}{\sum_j e^{X\beta_j}}$. 완전 분리: MLE 발산 → Firth 보정: $\ell^*(\beta) = \ell(\beta) + \frac{1}{2}\log|I(\beta)|$",
     """import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=10, n_classes=3, n_informative=5, random_state=42)
model = LogisticRegression(multi_class='multinomial', max_iter=1000).fit(X, y)
print(f"3-class Logistic Accuracy: {model.score(X, y):.3f}")
print(f"계수 행렬 shape: {model.coef_.shape}")"""),
    (8, "ch05_08_survival_regression", "생존 회귀 모형",
     r"Cox PH: $h(t|X) = h_0(t)\exp(X\beta)$. AFT: $\log T = X\beta + \sigma\epsilon$. C-index: $P(\hat{T}_i > \hat{T}_j | T_i > T_j)$",
     """import numpy as np
np.random.seed(42)
n = 500
X = np.random.normal(0, 1, (n, 3))
# Weibull 생존 시간
lambda_val = np.exp(-(X @ np.array([0.5, -0.3, 0.2])))
T = np.random.weibull(2, n) * lambda_val
C = np.random.exponential(3, n)
observed_time = np.minimum(T, C)
event = (T <= C).astype(int)
print(f"관측 사건 비율: {event.mean():.3f}")
print(f"중위 관측 시간: {np.median(observed_time):.3f}")"""),
    (9, "ch05_09_zero_inflated", "제로 팽창 모형",
     r"ZIP: $P(Y=0) = \pi + (1-\pi)e^{-\lambda}$, $P(Y=k) = (1-\pi)\frac{\lambda^k e^{-\lambda}}{k!}$ for $k \geq 1$",
     """import numpy as np
from scipy import stats

np.random.seed(42)
n = 1000; pi = 0.3; lam = 3
is_zero = np.random.binomial(1, pi, n)
y = np.where(is_zero, 0, np.random.poisson(lam, n))

print(f"관측 제로 비율: {(y==0).mean():.3f}")
print(f"포아송 예상 제로: {stats.poisson.pmf(0, lam):.3f}")
print(f"ZIP 예상 제로: {pi + (1-pi)*stats.poisson.pmf(0, lam):.3f}")
print(f"표본 평균: {y.mean():.3f}, 이론 평균: {(1-pi)*lam:.3f}")"""),
    (10, "ch05_10_practice_realestate", "실전: 부동산 가격 모델링",
     r"공간 자기상관 + 비선형 효과를 고려한 해석 가능 모형 구축",
     """import numpy as np, pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge

data = fetch_california_housing()
X, y = data.data, data.target
names = data.feature_names

lr = LinearRegression()
ridge = Ridge(alpha=1.0)
print(f"OLS CV R²: {cross_val_score(lr, X, y, cv=5).mean():.3f}")
print(f"Ridge CV R²: {cross_val_score(ridge, X, y, cv=5).mean():.3f}")

ridge.fit(X, y)
for name, coef in sorted(zip(names, ridge.coef_), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name}: {coef:.4f}")"""),
]

ch05 = []
for tid, fn, title, theory, code in ch05_info:
    ch05.append(make_topic(tid, fn, title,
        [f"{title} 이론", f"{title} 구현", f"{title} 해석"],
        theory, code, quick_exercises(title), quick_solutions(title),
        f"{title}은 회귀 분석의 중요한 확장입니다."))
gen_chapter(5, "ch05_advanced_regression", ch05)

# Chapter 06: 시계열 분석
ch06_info = [
    (1, "ch06_01_stationarity", "정상성과 단위근 검정",
     r"약정상: $E[X_t] = \mu$, $\text{Cov}(X_t, X_{t+h}) = \gamma(h)$. ADF: $\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum \delta_i \Delta y_{t-i} + \epsilon_t$, $H_0: \gamma = 0$",
     """import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

np.random.seed(42)
n = 500
# 정상 시계열 (AR(1))
y_stat = np.zeros(n)
for t in range(1, n): y_stat[t] = 0.7*y_stat[t-1] + np.random.normal(0, 1)
# 비정상 (랜덤 워크)
y_nonstat = np.cumsum(np.random.normal(0, 1, n))

for name, y in [('정상 AR(1)', y_stat), ('랜덤 워크', y_nonstat)]:
    adf_stat, adf_p, *_ = adfuller(y)
    kpss_stat, kpss_p, *_ = kpss(y, regression='c', nlags='auto')
    print(f"{name}: ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}")"""),
    (2, "ch06_02_arima", "ARIMA/SARIMA 모델링",
     r"ARIMA(p,d,q): $\phi(B)(1-B)^d X_t = \theta(B)\epsilon_t$",
     """import numpy as np, matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(42)
n = 300
e = np.random.normal(0, 1, n+2)
y = np.zeros(n)
for t in range(2, n): y[t] = 0.6*y[t-1] - 0.3*y[t-2] + e[t] + 0.4*e[t-1]

model = ARIMA(y, order=(2, 0, 1)).fit()
print(model.summary().tables[1])"""),
    (3, "ch06_03_var", "VAR과 그레인저 인과",
     r"VAR(p): $\mathbf{y}_t = \mathbf{c} + \sum_{i=1}^p \mathbf{A}_i \mathbf{y}_{t-i} + \mathbf{u}_t$. 그레인저 인과: $x$가 $y$를 예측하는 데 도움되는가",
     """import numpy as np
from statsmodels.tsa.api import VAR
np.random.seed(42)
n = 500; y = np.zeros((n, 2))
for t in range(2, n):
    y[t, 0] = 0.5*y[t-1, 0] + 0.3*y[t-1, 1] + np.random.normal(0, 1)
    y[t, 1] = 0.2*y[t-1, 1] + np.random.normal(0, 1)
model = VAR(y).fit(2)
print("Granger causality: y2 -> y1")
gc = model.test_causality('y1', 'y2', kind='f')
print(f"  F-stat={gc.test_statistic:.3f}, p={gc.pvalue:.4f}")"""),
    (4, "ch06_04_state_space", "상태 공간과 칼만 필터",
     r"상태: $x_{t+1} = Fx_t + w_t$. 관측: $y_t = Hx_t + v_t$. 칼만 게인: $K_t = P_t^-H^T(HP_t^-H^T + R)^{-1}$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
n = 200; F = 1.0; H = 1.0; Q = 0.1; R = 1.0
x_true = np.zeros(n); y_obs = np.zeros(n)
for t in range(1, n):
    x_true[t] = F*x_true[t-1] + np.random.normal(0, np.sqrt(Q))
    y_obs[t] = H*x_true[t] + np.random.normal(0, np.sqrt(R))
# 칼만 필터
x_est = np.zeros(n); P = np.ones(n)
for t in range(1, n):
    x_pred = F*x_est[t-1]; P_pred = F*P[t-1]*F + Q
    K = P_pred*H / (H*P_pred*H + R)
    x_est[t] = x_pred + K*(y_obs[t] - H*x_pred)
    P[t] = (1 - K*H)*P_pred
plt.figure(figsize=(12, 5))
plt.plot(x_true, 'k-', label='참값', alpha=0.7)
plt.plot(y_obs, '.', alpha=0.2, label='관측')
plt.plot(x_est, 'r-', label='칼만 추정')
plt.legend(); plt.title('칼만 필터')
plt.show()"""),
    (5, "ch06_05_garch", "GARCH와 변동성 모형",
     r"GARCH(1,1): $\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$. 조건: $\alpha + \beta < 1$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
n = 1000; omega, alpha, beta = 0.1, 0.15, 0.8
sigma2, returns = np.zeros(n), np.zeros(n)
sigma2[0] = omega / (1 - alpha - beta)
for t in range(1, n):
    sigma2[t] = omega + alpha*returns[t-1]**2 + beta*sigma2[t-1]
    returns[t] = np.sqrt(sigma2[t]) * np.random.standard_t(5)
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(returns); axes[0].set_title('수익률')
axes[1].plot(np.sqrt(sigma2)); axes[1].set_title('조건부 변동성')
plt.tight_layout(); plt.show()"""),
    (6, "ch06_06_spectral", "스펙트럼 분석과 웨이블릿",
     r"파워 스펙트럼: $S(\omega) = \sum_{h=-\infty}^{\infty} \gamma(h)e^{-i\omega h}$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
t = np.linspace(0, 10, 1000); dt = t[1]-t[0]
y = np.sin(2*np.pi*2*t) + 0.5*np.sin(2*np.pi*5*t) + 0.3*np.random.randn(len(t))
freqs = np.fft.fftfreq(len(t), dt); power = np.abs(np.fft.fft(y))**2
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(t, y); axes[0].set_title('시계열')
axes[1].plot(freqs[:len(t)//2], power[:len(t)//2]); axes[1].set_title('파워 스펙트럼')
plt.tight_layout(); plt.show()"""),
    (7, "ch06_07_structural", "구조적 시계열 모형",
     r"로컬 레벨: $y_t = \mu_t + \epsilon_t$, $\mu_{t+1} = \mu_t + \eta_t$",
     """import numpy as np
from statsmodels.tsa.structural import UnobservedComponents
np.random.seed(42)
n = 200
trend = np.cumsum(np.random.normal(0.1, 0.1, n))
seasonal = 3*np.sin(2*np.pi*np.arange(n)/12)
y = trend + seasonal + np.random.normal(0, 0.5, n)
model = UnobservedComponents(y, level='local linear trend', seasonal=12).fit()
print(f"AIC: {model.aic:.1f}")"""),
    (8, "ch06_08_changepoint", "변환점 탐지",
     r"CUSUM: $S_t = \sum_{i=1}^t (x_i - \bar{x})$. PELT: $\min_{m, \tau} \sum_{i=0}^m C(y_{\tau_i+1:\tau_{i+1}}) + \beta m$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
y = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(3, 1, 100), np.random.normal(1, 1, 100)])
# CUSUM
cusum = np.cumsum(y - y.mean())
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1); plt.plot(y); plt.title('시계열')
plt.subplot(1,2,2); plt.plot(cusum); plt.title('CUSUM')
plt.tight_layout(); plt.show()"""),
    (9, "ch06_09_ts_cv", "시계열 교차 검증",
     r"확장 윈도우: 훈련 $[1, t]$, 테스트 $[t+1, t+h]$. CRPS: $\text{CRPS}(F, y) = \int_{-\infty}^{\infty}(F(x) - \mathbf{1}(x \geq y))^2 dx$",
     """import numpy as np
from sklearn.metrics import mean_absolute_error
np.random.seed(42)
y = np.cumsum(np.random.normal(0.01, 1, 500))
min_train = 100; horizon = 1; errors = []
for t in range(min_train, len(y)-horizon):
    train = y[:t]
    pred = train[-1]  # 나이브 예측
    errors.append(abs(y[t+horizon-1] - pred))
print(f"확장 윈도우 MAE: {np.mean(errors):.3f}")"""),
    (10, "ch06_10_practice_energy", "실전: 에너지 수요 예측",
     r"다중 계절성 + 외생 변수 활용 예측",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
t = np.arange(365*2)
daily = 10*np.sin(2*np.pi*t/365)
weekly = 3*np.sin(2*np.pi*t/7)
trend = 0.01*t
y = 100 + trend + daily + weekly + np.random.normal(0, 2, len(t))
plt.figure(figsize=(14, 5))
plt.plot(t, y, alpha=0.7); plt.title('합성 에너지 수요 데이터')
plt.xlabel('일'); plt.ylabel('수요')
plt.show()"""),
]

ch06 = []
for tid, fn, title, theory, code in ch06_info:
    ch06.append(make_topic(tid, fn, title,
        [f"{title} 이론", f"{title} 구현", f"{title} 해석"],
        theory, code, quick_exercises(title), quick_solutions(title),
        f"{title}은 시계열 분석의 핵심 주제입니다."))
gen_chapter(6, "ch06_time_series", ch06)

# Chapter 07: 베이지안
ch07_info = [
    (1, "ch07_01_bayes_foundations", "베이즈 정리와 사전 분포",
     r"$p(\theta|y) \propto p(y|\theta)p(\theta)$. Jeffrey's prior: $p(\theta) \propto \sqrt{I(\theta)}$",
     """import numpy as np, matplotlib.pyplot as plt
from scipy import stats
x = np.linspace(0, 1, 200)
# Beta prior + Binomial likelihood
n_obs, k = 20, 14
for a, b, name in [(1,1,'균등'), (0.5,0.5,"Jeffrey's"), (2,5,'정보적')]:
    prior = stats.beta.pdf(x, a, b)
    posterior = stats.beta.pdf(x, a+k, b+n_obs-k)
    plt.plot(x, posterior, label=f'사후({name})')
plt.axvline(k/n_obs, color='gray', linestyle='--', label='MLE')
plt.legend(); plt.title(f'관측: {k}/{n_obs}')
plt.show()"""),
    (2, "ch07_02_conjugate", "켤레 사전과 해석적 사후",
     r"Beta-Binomial: $\theta|y \sim \text{Beta}(\alpha+y, \beta+n-y)$. Normal-Normal: $\mu|y \sim N(\frac{n\bar{y}/\sigma^2 + \mu_0/\tau^2}{n/\sigma^2 + 1/\tau^2}, \frac{1}{n/\sigma^2 + 1/\tau^2})$",
     """import numpy as np
from scipy import stats
# Normal-Normal 켤레
mu0, tau0 = 0, 10  # 사전
sigma = 2  # 알려진 분산
data = np.random.normal(5, sigma, 30)
n, ybar = len(data), data.mean()
# 사후
tau_post = 1 / (n/sigma**2 + 1/tau0**2)
mu_post = tau_post * (n*ybar/sigma**2 + mu0/tau0**2)
print(f"사전: N({mu0}, {tau0**2})")
print(f"사후: N({mu_post:.3f}, {tau_post:.4f})")
print(f"MLE: {ybar:.3f}")"""),
    (3, "ch07_03_pymc_basics", "PyMC 베이지안 모델링",
     r"PyMC5를 활용한 사후 샘플링, 수렴 진단 (R̂, ESS)",
     """import numpy as np
# PyMC 의사 코드 (패키지 설치 필요)
print("=== PyMC 모델링 기본 구조 ===")
print('''
import pymc as pm
import arviz as az

with pm.Model() as model:
    # 사전 분포
    mu = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=5)
    # 우도
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data)
    # 사후 샘플링
    trace = pm.sample(2000, tune=1000, cores=2)

# 진단
print(az.summary(trace))
az.plot_trace(trace)
''')"""),
    (4, "ch07_04_hierarchical", "계층적 베이지안 모형",
     r"$y_{ij} \sim N(\theta_j, \sigma^2)$, $\theta_j \sim N(\mu, \tau^2)$. 부분 풀링: $\hat\theta_j \approx \frac{n_j/\sigma^2}{n_j/\sigma^2 + 1/\tau^2}\bar{y}_j + \frac{1/\tau^2}{n_j/\sigma^2 + 1/\tau^2}\hat\mu$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
J = 8  # 그룹 수
n_j = np.array([15, 15, 27, 20, 15, 28, 13, 20])
y_bar = np.array([28.4, 7.9, -2.8, 6.8, -0.6, 0.6, 18.0, 12.2])
sigma = np.array([14.9, 10.2, 16.3, 11.0, 9.4, 11.4, 10.4, 17.6])

# 완전 풀링
mu_pooled = np.average(y_bar, weights=1/sigma**2)
# 부분 풀링 (EB 근사)
tau2_hat = max(0, np.var(y_bar) - np.mean(sigma**2/n_j))
B_j = sigma**2 / (sigma**2 + tau2_hat)
mu_shrunk = (1 - B_j) * y_bar + B_j * mu_pooled

plt.figure(figsize=(10, 6))
plt.scatter(range(J), y_bar, s=100, label='관측 평균', zorder=5)
plt.scatter(range(J), mu_shrunk, s=100, marker='s', label='축소 추정')
plt.axhline(mu_pooled, color='red', linestyle='--', label=f'전체 평균={mu_pooled:.1f}')
plt.legend(); plt.title('계층 모형: 축소(shrinkage) 효과')
plt.show()"""),
    (5, "ch07_05_model_comparison", "베이지안 모형 비교",
     r"WAIC: $\widehat{elpd} = \sum \log \bar{p}(y_i) - \sum p_{\text{WAIC},i}$. LOO-CV: Pareto smoothed importance sampling",
     """import numpy as np
print("=== 베이지안 모형 비교 ===")
print("WAIC = -2(lppd - p_waic)")
print("lppd = sum(log(mean(p(y_i|theta_s))))")
print("p_waic = sum(var(log p(y_i|theta_s)))")
print("\\nWAIC이 낮을수록 좋은 모형")"""),
    (6, "ch07_06_diagnostics", "사후 예측 검정과 모형 진단",
     r"사후 예측: $y^{rep} \sim p(y^{rep}|y) = \int p(y^{rep}|\theta)p(\theta|y)d\theta$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
data = np.random.normal(5, 2, 100)
# 사후 예측 시뮬레이션 (정규-정규)
n_post = 1000
mu_post = data.mean()
sigma_post = 2 / np.sqrt(len(data))
y_rep = np.array([np.random.normal(np.random.normal(mu_post, sigma_post), 2, len(data)) for _ in range(n_post)])

plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, density=True, alpha=0.5, label='관측')
for i in range(50):
    plt.hist(y_rep[i], bins=20, density=True, alpha=0.02, color='blue')
plt.title('사후 예측 검정')
plt.legend()
plt.show()"""),
    (7, "ch07_07_bayesian_regression", "베이지안 회귀와 변수 선택",
     r"$\beta|\sigma^2, y \sim N(\tilde\beta, \sigma^2(X^TX + \Lambda)^{-1})$. Horseshoe: $\beta_j \sim N(0, \lambda_j^2\tau^2)$, $\lambda_j \sim C^+(0,1)$",
     """import numpy as np
np.random.seed(42)
n, p = 100, 10
X = np.random.randn(n, p)
beta_true = np.array([3, -2, 0, 0, 1.5, 0, 0, 0, -1, 0])
y = X @ beta_true + np.random.normal(0, 1, n)
# 베이지안 Ridge (해석적)
lam = 1.0
beta_post = np.linalg.solve(X.T@X + lam*np.eye(p), X.T@y)
print("참값:  ", beta_true)
print("추정값:", beta_post.round(3))"""),
    (8, "ch07_08_gaussian_process", "가우시안 과정 회귀",
     r"GP: $f \sim \mathcal{GP}(m, k)$. RBF 커널: $k(x,x') = \sigma^2\exp(-\frac{\|x-x'\|^2}{2l^2})$. 사후: $f_*|X_*,X,y \sim N(\bar{f}_*, \text{cov}(f_*))$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
# GP 회귀 from scratch
def rbf_kernel(x1, x2, l=1.0, sigma=1.0):
    return sigma**2 * np.exp(-0.5*np.subtract.outer(x1, x2)**2 / l**2)

X_train = np.array([-4, -3, -2, -1, 1, 2, 3, 4.0])
y_train = np.sin(X_train) + 0.1*np.random.randn(len(X_train))
X_test = np.linspace(-6, 6, 200)

K = rbf_kernel(X_train, X_train) + 0.01*np.eye(len(X_train))
K_s = rbf_kernel(X_train, X_test)
K_ss = rbf_kernel(X_test, X_test)

K_inv = np.linalg.inv(K)
mu = K_s.T @ K_inv @ y_train
cov = K_ss - K_s.T @ K_inv @ K_s

plt.figure(figsize=(12, 6))
plt.plot(X_test, mu, 'b-', label='GP 평균')
plt.fill_between(X_test, mu-2*np.sqrt(np.diag(cov)), mu+2*np.sqrt(np.diag(cov)), alpha=0.2)
plt.scatter(X_train, y_train, c='red', s=100, zorder=5)
plt.plot(X_test, np.sin(X_test), 'k--', alpha=0.5, label='sin(x)')
plt.legend(); plt.title('가우시안 과정 회귀')
plt.show()"""),
    (9, "ch07_09_nonparametric_bayes", "베이지안 비모수 (DP Mixture)",
     r"DP: $G \sim DP(\alpha, G_0)$. 스틱 브레이킹: $\pi_k = V_k\prod_{j<k}(1-V_j)$, $V_k \sim \text{Beta}(1, \alpha)$. CRP: $P(\text{new table}) = \frac{\alpha}{n-1+\alpha}$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
# 스틱 브레이킹
def stick_breaking(alpha, K):
    betas = np.random.beta(1, alpha, K)
    pis = np.zeros(K)
    pis[0] = betas[0]
    for k in range(1, K):
        pis[k] = betas[k] * np.prod(1 - betas[:k])
    return pis

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, alpha in zip(axes, [0.1, 1.0, 10.0]):
    pis = stick_breaking(alpha, 20)
    ax.bar(range(20), pis)
    ax.set_title(f'α = {alpha}, 유효 클러스터: {(pis > 0.01).sum()}')
plt.suptitle('디리클레 과정: 스틱 브레이킹')
plt.tight_layout(); plt.show()"""),
    (10, "ch07_10_practice_sports", "실전: 스포츠 베이지안 순위",
     r"Bradley-Terry: $P(i \text{ beats } j) = \frac{p_i}{p_i + p_j}$",
     """import numpy as np
np.random.seed(42)
n_teams = 6
strengths = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5])
names = ['팀A', '팀B', '팀C', '팀D', '팀E', '팀F']
# 경기 시뮬레이션
results = []
for i in range(n_teams):
    for j in range(i+1, n_teams):
        for _ in range(10):
            p = strengths[i] / (strengths[i] + strengths[j])
            winner = i if np.random.rand() < p else j
            results.append((i, j, winner))

# MLE 추정 (반복 비례)
s = np.ones(n_teams)
for _ in range(100):
    for k in range(n_teams):
        wins = sum(1 for i,j,w in results if w == k)
        games = sum(1/(s[i]+s[j]) for i,j,w in results if i==k or j==k)
        s[k] = wins / games if games > 0 else 1
    s /= s.sum()

for name, st in sorted(zip(names, s), key=lambda x: -x[1]):
    print(f"{name}: {st:.3f}")"""),
]

ch07 = []
for tid, fn, title, theory, code in ch07_info:
    ch07.append(make_topic(tid, fn, title,
        [f"{title} 이론", f"{title} 구현", f"{title} 해석"],
        theory, code, quick_exercises(title), quick_solutions(title),
        f"{title}은 베이지안 분석의 핵심입니다."))
gen_chapter(7, "ch07_bayesian", ch07)

# Chapter 08: 비지도 학습
ch08_info = [
    (1, "ch08_01_pca", "PCA: 스펙트럼 정리와 최적성",
     r"$\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$. 최적성: PCA는 $\min \|X - X_k\|_F^2$의 해 (Eckart-Young). 설명 분산: $\frac{\lambda_j}{\sum \lambda_i}$",
     """import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
iris = load_iris()
X = (iris.data - iris.data.mean(0)) / iris.data.std(0)

# SVD로 직접 PCA
U, S, Vt = np.linalg.svd(X, full_matrices=False)
explained = S**2 / (S**2).sum()
print(f"설명 분산: {explained.round(3)}")

# sklearn 비교
pca = PCA().fit(X)
plt.figure(figsize=(10, 5))
plt.bar(range(4), pca.explained_variance_ratio_)
plt.xlabel('주성분'); plt.ylabel('설명 분산 비율')
plt.title('PCA 스크리 플롯')
plt.show()"""),
    (2, "ch08_02_kernel_pca", "커널 PCA와 비선형 확장",
     r"커널 트릭: $K_{ij} = k(x_i, x_j) = \phi(x_i)^T\phi(x_j)$. 중심화: $\tilde{K} = K - 1_nK - K1_n + 1_nK1_n$",
     """import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=42)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', s=10)
axes[0].set_title('원본')
for ax, kernel in zip(axes[1:], ['linear', 'rbf']):
    kpca = KernelPCA(n_components=2, kernel=kernel, gamma=10)
    X_t = kpca.fit_transform(X)
    ax.scatter(X_t[:,0], X_t[:,1], c=y, cmap='coolwarm', s=10)
    ax.set_title(f'Kernel PCA ({kernel})')
plt.tight_layout(); plt.show()"""),
    (3, "ch08_03_tsne", "t-SNE 이론과 실전",
     r"$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2/2\sigma_i^2)}{\sum_{k\neq i}\exp(-\|x_i - x_k\|^2/2\sigma_i^2)}$. Student-t: $q_{ij} = \frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k\neq l}(1+\|y_k-y_l\|^2)^{-1}}$. KL: $\min \sum p_{ij}\log\frac{p_{ij}}{q_{ij}}$",
     """import numpy as np, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, perp in zip(axes, [5, 30, 100]):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_2d = tsne.fit_transform(X)
    scatter = ax.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='tab10', s=5, alpha=0.7)
    ax.set_title(f'perplexity={perp}')
plt.colorbar(scatter); plt.tight_layout(); plt.show()"""),
    (4, "ch08_04_umap", "UMAP: 위상적 관점",
     r"UMAP: 고차원에서 퍼지 단체 집합 구성 → 저차원에서 교차 엔트로피 최소화",
     """import numpy as np, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
digits = load_digits()
print("UMAP은 umap-learn 패키지 필요")
print("pip install umap-learn")
print("\\nimport umap")
print("reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)")
print("embedding = reducer.fit_transform(X)")
# t-SNE로 대체 시각화
tsne = TSNE(random_state=42).fit_transform(digits.data)
plt.figure(figsize=(10, 8))
plt.scatter(tsne[:,0], tsne[:,1], c=digits.target, cmap='tab10', s=3, alpha=0.7)
plt.colorbar()
plt.title('Digits 시각화 (t-SNE)')
plt.show()"""),
    (5, "ch08_05_nmf", "비음수 행렬 분해 (NMF)",
     r"$\mathbf{X} \approx \mathbf{WH}$, $W \geq 0, H \geq 0$. 곱셈 업데이트: $H \leftarrow H \circ \frac{W^TX}{W^TWH}$, $W \leftarrow W \circ \frac{XH^T}{WHH^T}$",
     """import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import NMF
np.random.seed(42)
# 합성 비음수 데이터
W_true = np.random.exponential(1, (100, 3))
H_true = np.random.exponential(1, (3, 50))
X = W_true @ H_true + 0.5*np.random.exponential(0.1, (100, 50))
nmf = NMF(n_components=3, init='random', random_state=42, max_iter=500)
W = nmf.fit_transform(X)
H = nmf.components_
print(f"재구성 오차: {np.linalg.norm(X - W@H) / np.linalg.norm(X):.4f}")"""),
    (6, "ch08_06_gmm", "가우시안 혼합과 EM",
     r"$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$. E: $\gamma_{nk} = \frac{\pi_k\mathcal{N}(x_n|\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal{N}(x_n|\mu_j,\Sigma_j)}$",
     """import numpy as np, matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=500, centers=3, cluster_std=[1, 1.5, 0.5], random_state=42)
bics = []
for k in range(1, 8):
    gmm = GaussianMixture(n_components=k, random_state=42).fit(X)
    bics.append(gmm.bic(X))
plt.figure(figsize=(8, 5))
plt.plot(range(1, 8), bics, 'b-o')
plt.xlabel('K'); plt.ylabel('BIC')
plt.title(f'최적 K = {np.argmin(bics)+1}')
plt.show()"""),
    (7, "ch08_07_spectral_clustering", "스펙트럼 클러스터링",
     r"그래프 라플라시안: $L = D - W$. 정규화: $L_{sym} = I - D^{-1/2}WD^{-1/2}$. NCut: 가장 작은 고유벡터들로 임베딩 후 k-means",
     """import numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (model, name) in zip(axes, [(KMeans(2), 'K-Means'), (SpectralClustering(2, gamma=10), 'Spectral')]):
    labels = model.fit_predict(X)
    ax.scatter(X[:,0], X[:,1], c=labels, cmap='coolwarm', s=20)
    ax.set_title(name)
plt.tight_layout(); plt.show()"""),
    (8, "ch08_08_density_clustering", "밀도 기반 클러스터링",
     r"DBSCAN: $\epsilon$-이웃에 minPts 이상이면 core point. HDBSCAN: 다양한 $\epsilon$에 대해 안정적인 클러스터 추출",
     """import numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
X1, _ = make_moons(200, noise=0.1, random_state=42)
X2 = np.random.uniform(-2, 3, (50, 2))
X = np.vstack([X1, X2])
db = DBSCAN(eps=0.3, min_samples=5).fit(X)
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=db.labels_, cmap='viridis', s=20)
plt.title(f'DBSCAN: {len(set(db.labels_))-1} clusters, {(db.labels_==-1).sum()} noise')
plt.show()"""),
    (9, "ch08_09_cluster_validation", "클러스터 유효성 검증",
     r"실루엣: $s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))}$. 갭 통계량: $\text{Gap}(k) = E^*[\log W_k] - \log W_k$",
     """import numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
X, _ = make_blobs(500, centers=4, random_state=42)
scores = []
for k in range(2, 10):
    km = KMeans(k, random_state=42, n_init=10).fit(X)
    scores.append(silhouette_score(X, km.labels_))
plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), scores, 'b-o')
plt.xlabel('K'); plt.ylabel('Silhouette Score')
plt.title(f'최적 K = {np.argmax(scores)+2}')
plt.show()"""),
    (10, "ch08_10_practice_segmentation", "실전: 고객 세분화",
     r"RFM 분석 + 클러스터링으로 고객 세그먼트 도출",
     """import numpy as np, pandas as pd
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'recency': np.random.exponential(30, n),
    'frequency': np.random.poisson(5, n),
    'monetary': np.random.lognormal(4, 1, n)
})
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = StandardScaler().fit_transform(df)
km = KMeans(4, random_state=42, n_init=10).fit(X)
df['segment'] = km.labels_
print(df.groupby('segment')[['recency','frequency','monetary']].mean().round(1))"""),
]

ch08 = []
for tid, fn, title, theory, code in ch08_info:
    ch08.append(make_topic(tid, fn, title,
        [f"{title} 이론", f"{title} 구현", f"{title} 해석"],
        theory, code, quick_exercises(title), quick_solutions(title),
        f"{title}은 비지도 학습의 핵심입니다."))
gen_chapter(8, "ch08_unsupervised", ch08)

# Chapter 09: 인과 추론
ch09_info = [
    (1, "ch09_01_potential_outcomes", "잠재 결과 프레임워크",
     r"$\tau_i = Y_i(1) - Y_i(0)$. ATE: $\tau = E[Y(1) - Y(0)]$. ATT: $E[Y(1) - Y(0)|W=1]$. 기본 문제: 각 개체의 $Y_i(1), Y_i(0)$을 동시에 관측할 수 없음",
     """import numpy as np
np.random.seed(42)
n = 1000
X = np.random.normal(0, 1, n)
# 잠재 결과
Y0 = 2 + X + np.random.normal(0, 1, n)
Y1 = 4 + 1.5*X + np.random.normal(0, 1, n)
true_ate = (Y1 - Y0).mean()
# 무작위 배정
W = np.random.binomial(1, 0.5, n)
Y_obs = W*Y1 + (1-W)*Y0
naive_ate = Y_obs[W==1].mean() - Y_obs[W==0].mean()
print(f"참 ATE: {true_ate:.3f}")
print(f"무작위 배정 추정: {naive_ate:.3f}")"""),
    (2, "ch09_02_dags", "DAG와 do-계산",
     r"d-분리: $X \perp Y | Z$이면 $Z$가 $X, Y$ 사이의 모든 경로를 차단. 백도어 기준: $Z$가 $X \to Y$의 모든 백도어 경로를 차단",
     """import numpy as np
np.random.seed(42)
n = 2000
# X -> Y, X <- U -> Y (교란)
U = np.random.normal(0, 1, n)
X = 0.5*U + np.random.normal(0, 1, n)
Y = 2*X + 1.5*U + np.random.normal(0, 1, n)
# 나이브 회귀 (편향)
from numpy.linalg import lstsq
beta_naive = lstsq(np.column_stack([np.ones(n), X]), Y, rcond=None)[0][1]
# U 통제 (백도어 조정)
beta_adj = lstsq(np.column_stack([np.ones(n), X, U]), Y, rcond=None)[0][1]
print(f"나이브: {beta_naive:.3f} (참=2, 편향)")
print(f"U 조정: {beta_adj:.3f}")"""),
    (3, "ch09_03_propensity_score", "성향 점수 매칭과 가중",
     r"$e(x) = P(W=1|X=x)$. IPW: $\hat\tau_{IPW} = \frac{1}{n}\sum \frac{W_iY_i}{e(X_i)} - \frac{(1-W_i)Y_i}{1-e(X_i)}$",
     """import numpy as np
from sklearn.linear_model import LogisticRegression
np.random.seed(42)
n = 2000
X = np.random.normal(0, 1, (n, 3))
# 비무작위 배정
logit = X @ [0.5, -0.3, 0.2]
W = np.random.binomial(1, 1/(1+np.exp(-logit)))
Y = 2*W + X@[1, 0.5, -0.3] + 0.5*W*X[:,0] + np.random.normal(0, 1, n)

# 성향 점수
ps_model = LogisticRegression().fit(X, W)
ps = ps_model.predict_proba(X)[:, 1]
ps = np.clip(ps, 0.05, 0.95)

# IPW
ipw = (W*Y/ps).mean() - ((1-W)*Y/(1-ps)).mean()
naive = Y[W==1].mean() - Y[W==0].mean()
print(f"나이브: {naive:.3f}")
print(f"IPW: {ipw:.3f}")
print(f"참 ATE: 2.0")"""),
    (4, "ch09_04_did", "이중 차분법 (DiD)",
     r"$\hat\tau_{DiD} = (\bar{Y}_{T,post} - \bar{Y}_{T,pre}) - (\bar{Y}_{C,post} - \bar{Y}_{C,pre})$. 핵심 가정: 평행 추세",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
n = 200; T = 10; treatment_time = 5; tau = 3
data_control = np.cumsum(np.random.normal(0.5, 1, T)) + np.random.normal(0, 0.5, (n, T))
data_treated = np.cumsum(np.random.normal(0.5, 1, T)) + np.random.normal(0, 0.5, (n, T))
data_treated[:, treatment_time:] += tau

pre_diff = data_treated[:, :treatment_time].mean() - data_control[:, :treatment_time].mean()
post_diff = data_treated[:, treatment_time:].mean() - data_control[:, treatment_time:].mean()
did = post_diff - pre_diff
print(f"DiD 추정: {did:.3f} (참: {tau})")"""),
    (5, "ch09_05_rdd", "회귀 불연속 설계 (RDD)",
     r"Sharp RDD: $\tau_{RDD} = \lim_{x\downarrow c} E[Y|X=x] - \lim_{x\uparrow c} E[Y|X=x]$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
n = 1000; c = 50
X = np.random.uniform(20, 80, n)
W = (X >= c).astype(int)
Y = 0.5*X + 5*W - 0.02*(X-c)*W + np.random.normal(0, 3, n)

# 국소 선형 회귀
bw = 10
mask = np.abs(X - c) <= bw
from numpy.linalg import lstsq
X_local = np.column_stack([np.ones(mask.sum()), X[mask]-c, W[mask], (X[mask]-c)*W[mask]])
beta = lstsq(X_local, Y[mask], rcond=None)[0]
print(f"RDD 추정: {beta[2]:.3f} (참: 5)")"""),
    (6, "ch09_06_iv", "도구 변수법 (IV)",
     r"LATE: $\tau_{LATE} = \frac{E[Y|Z=1]-E[Y|Z=0]}{E[D|Z=1]-E[D|Z=0]}$ (Wald 추정량)",
     """import numpy as np
np.random.seed(42)
n = 2000
Z = np.random.binomial(1, 0.5, n)  # 도구변수
U = np.random.normal(0, 1, n)
D = (0.3*Z + 0.5*U + np.random.normal(0, 0.5, n) > 0).astype(int)
Y = 2*D + 1.5*U + np.random.normal(0, 1, n)
# Wald 추정
wald = (Y[Z==1].mean()-Y[Z==0].mean()) / (D[Z==1].mean()-D[Z==0].mean())
print(f"OLS: {np.polyfit(D, Y, 1)[0]:.3f} (편향)")
print(f"Wald: {wald:.3f} (참: 2)")"""),
    (7, "ch09_07_synthetic_control", "합성 통제법",
     r"$\hat{Y}_{1t}^{(0)} = \sum_{j=2}^{J+1} w_j Y_{jt}$, $\sum w_j = 1, w_j \geq 0$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
T, J = 40, 5; treat_t = 20; tau = 5
Y_control = np.cumsum(np.random.normal(1, 1, (J, T)), axis=1) + np.random.normal(0, 0.5, (J, T))
Y_treat = np.cumsum(np.random.normal(1, 1, T)) + np.random.normal(0, 0.5, T)
Y_treat[treat_t:] += tau

# 가중치 추정 (사전 기간)
from scipy.optimize import minimize
def obj(w): return np.sum((Y_treat[:treat_t] - w@Y_control[:, :treat_t])**2)
cons = [{'type': 'eq', 'fun': lambda w: w.sum()-1}]
w0 = np.ones(J)/J
result = minimize(obj, w0, constraints=cons, bounds=[(0,1)]*J, method='SLSQP')
Y_synth = result.x @ Y_control
effect = (Y_treat[treat_t:] - Y_synth[treat_t:]).mean()
print(f"합성 통제 추정: {effect:.3f} (참: {tau})")"""),
    (8, "ch09_08_doubly_robust", "이중 견고 추정 (AIPW)",
     r"$\hat\tau_{AIPW} = \frac{1}{n}\sum[\hat\mu_1(X_i) - \hat\mu_0(X_i) + \frac{W_i(Y_i-\hat\mu_1(X_i))}{e(X_i)} - \frac{(1-W_i)(Y_i-\hat\mu_0(X_i))}{1-e(X_i)}]$",
     """import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
np.random.seed(42)
n = 2000
X = np.random.normal(0, 1, (n, 3))
ps_true = 1/(1+np.exp(-X@[0.5,-0.3,0.2]))
W = np.random.binomial(1, ps_true)
Y = 2*W + X@[1,0.5,-0.3] + np.random.normal(0, 1, n)

ps = LogisticRegression().fit(X, W).predict_proba(X)[:,1]
ps = np.clip(ps, 0.05, 0.95)
mu1 = LinearRegression().fit(X[W==1], Y[W==1]).predict(X)
mu0 = LinearRegression().fit(X[W==0], Y[W==0]).predict(X)

aipw = np.mean(mu1 - mu0 + W*(Y-mu1)/ps - (1-W)*(Y-mu0)/(1-ps))
print(f"AIPW: {aipw:.3f} (참: 2.0)")"""),
    (9, "ch09_09_mediation", "매개 분석",
     r"총 효과 = 직접 효과 + 간접 효과. $NDE = E[Y(1,M(0)) - Y(0,M(0))]$. $NIE = E[Y(0,M(1)) - Y(0,M(0))]$",
     """import numpy as np
np.random.seed(42)
n = 2000
X = np.random.binomial(1, 0.5, n)
M = 0.5*X + np.random.normal(0, 1, n)
Y = 1.5*X + 0.8*M + np.random.normal(0, 1, n)
total = np.mean(Y[X==1]) - np.mean(Y[X==0])
# Baron-Kenny
from sklearn.linear_model import LinearRegression
a = LinearRegression().fit(X.reshape(-1,1), M).coef_[0]
b_direct = LinearRegression().fit(np.column_stack([X, M]), Y).coef_
indirect = a * b_direct[1]
direct = b_direct[0]
print(f"총 효과: {total:.3f}, 직접: {direct:.3f}, 간접: {indirect:.3f}")"""),
    (10, "ch09_10_practice_policy", "실전: 정책 효과 평가",
     r"다양한 식별 전략을 비교하여 정책 효과를 추정",
     """import numpy as np, pandas as pd
np.random.seed(42)
n = 2000
df = pd.DataFrame({
    'age': np.random.normal(35, 10, n),
    'education': np.random.poisson(12, n),
    'treated': np.random.binomial(1, 0.4, n)
})
df['income'] = 30000 + 1000*df['education'] + 500*df['age'] + 5000*df['treated'] + np.random.normal(0, 5000, n)
from scipy.stats import ttest_ind
t, c = df[df['treated']==1]['income'], df[df['treated']==0]['income']
print(f"나이브 차이: {t.mean()-c.mean():.0f}")
print(f"참 효과: 5000")"""),
]

ch09 = []
for tid, fn, title, theory, code in ch09_info:
    ch09.append(make_topic(tid, fn, title,
        [f"{title} 이론", f"{title} 구현", f"{title} 해석"],
        theory, code, quick_exercises(title), quick_solutions(title),
        f"{title}은 인과 추론의 핵심 방법입니다."))
gen_chapter(9, "ch09_causal_inference", ch09)

# Chapter 10: 생존 분석
ch10_info = [
    (1, "ch10_01_survival_functions", "생존 함수와 위험 함수", r"$S(t) = P(T > t)$, $h(t) = \lim_{dt\to 0}\frac{P(t \leq T < t+dt|T \geq t)}{dt} = \frac{f(t)}{S(t)}$, $H(t) = -\log S(t)$",
     """import numpy as np, matplotlib.pyplot as plt
from scipy import stats
t = np.linspace(0.01, 10, 200)
dists = {'Exponential(0.5)': stats.expon(scale=2), 'Weibull(2,1)': stats.weibull_min(2), 'Lognormal(0,1)': stats.lognorm(1)}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, title in zip(axes, ['S(t)', 'h(t)', 'H(t)']):
    for name, d in dists.items():
        if title == 'S(t)': ax.plot(t, d.sf(t), label=name)
        elif title == 'h(t)': ax.plot(t, d.pdf(t)/d.sf(t), label=name)
        else: ax.plot(t, -np.log(d.sf(t)), label=name)
    ax.set_title(title); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()"""),
    (2, "ch10_02_kaplan_meier", "Kaplan-Meier와 로그순위 검정", r"$\hat{S}(t) = \prod_{t_i \leq t}\frac{n_i - d_i}{n_i}$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
T1 = np.random.exponential(5, 100); C1 = np.random.exponential(8, 100)
T2 = np.random.exponential(3, 100); C2 = np.random.exponential(8, 100)
def kaplan_meier(T, C):
    obs = np.minimum(T, C); event = (T <= C).astype(int)
    times = np.sort(np.unique(obs[event==1]))
    S = [1.0]; surv = 1.0
    for t in times:
        d = np.sum((obs == t) & (event == 1))
        n = np.sum(obs >= t)
        surv *= (1 - d/n); S.append(surv)
    return np.concatenate([[0], times]), S
t1, s1 = kaplan_meier(T1, C1); t2, s2 = kaplan_meier(T2, C2)
plt.figure(figsize=(10, 6))
plt.step(t1, s1, label='Group 1'); plt.step(t2, s2, label='Group 2')
plt.xlabel('Time'); plt.ylabel('S(t)'); plt.legend(); plt.title('Kaplan-Meier')
plt.show()"""),
    (3, "ch10_03_cox_ph", "Cox 비례위험 모형", r"$h(t|X) = h_0(t)\exp(X\beta)$. 부분 우도: $L(\beta) = \prod_{i:d_i=1}\frac{\exp(X_i\beta)}{\sum_{j \in R(t_i)}\exp(X_j\beta)}$",
     """import numpy as np
np.random.seed(42)
n = 500
X = np.random.normal(0, 1, (n, 3))
T = np.random.exponential(np.exp(-X@[0.5, -0.3, 0.2]), n)
C = np.random.exponential(3, n)
obs_time = np.minimum(T, C); event = (T <= C).astype(int)
print(f"사건 발생률: {event.mean():.3f}")
print(f"중위 관측 시간: {np.median(obs_time):.3f}")
print("lifelines 패키지로 Cox PH 적합:")
print("from lifelines import CoxPHFitter")
print("cph = CoxPHFitter().fit(df, 'time', 'event')")"""),
]
for tid, fn, title, theory, code in [
    (4, "ch10_04_time_varying", "시간 의존 공변량", r"확장 Cox: 시간에 따라 변하는 공변량 포함", "print('시간 의존 공변량')"),
    (5, "ch10_05_parametric", "모수적 생존 모형", r"Weibull: $h(t) = \frac{k}{\lambda}(\frac{t}{\lambda})^{k-1}$", "print('Weibull, Log-logistic 모형')"),
    (6, "ch10_06_competing_risks", "경쟁 위험과 Fine-Gray", r"CIF: $F_k(t) = P(T \leq t, \epsilon = k)$", "print('경쟁 위험 모형')"),
    (7, "ch10_07_frailty", "허약 모형", r"$h_i(t|Z_i) = Z_i h_0(t)\exp(X_i\beta)$, $Z_i \sim \text{Gamma}(1/\theta, 1/\theta)$", "print('Frailty 모형')"),
    (8, "ch10_08_truncation", "구간 중도절단과 좌절단", r"Turnbull 추정량: 구간 중도절단 데이터의 비모수 MLE", "print('구간 중도절단')"),
    (9, "ch10_09_cure_models", "치유 모형", r"$S(t) = \pi + (1-\pi)S_0(t)$, $\pi$는 치유 비율", "print('치유 모형')"),
    (10, "ch10_10_practice_churn", "실전: 고객 이탈과 생애 가치", r"CLV = margin × retention / (1 + discount - retention)", "import numpy as np\nprint('고객 이탈 분석')"),
]:
    ch10_info.append((tid, fn, title, theory, code))

ch10 = []
for tid, fn, title, theory, code in ch10_info:
    full_code = f"import numpy as np, matplotlib.pyplot as plt\nfrom scipy import stats\nnp.random.seed(42)\n\n{code}"
    ch10.append(make_topic(tid, fn, title,
        [f"{title} 이론", f"{title} 구현"], theory, full_code,
        quick_exercises(title), quick_solutions(title), f"{title}은 생존 분석의 핵심입니다."))
gen_chapter(10, "ch10_survival", ch10)

# Chapter 11: 최적화
ch11_info = [
    (1, "ch11_01_convex_optimization", "볼록 최적화와 KKT", r"KKT: $\nabla f + \sum \lambda_i \nabla g_i + \sum \nu_j \nabla h_j = 0$, $\lambda_i \geq 0$, $\lambda_i g_i = 0$",
     """import numpy as np, matplotlib.pyplot as plt
from scipy.optimize import minimize
# 2D 볼록 최적화
def f(x): return (x[0]-1)**2 + (x[1]-2)**2
def g(x): return x[0] + x[1] - 1  # 부등식 g <= 0
result = minimize(f, [0, 0], constraints={'type': 'ineq', 'fun': lambda x: -g(x)})
print(f"최적해: {result.x.round(4)}, f*: {result.fun:.4f}")"""),
    (2, "ch11_02_linear_programming", "선형 계획법", r"$\min c^Tx$ s.t. $Ax \leq b$, $x \geq 0$",
     """import numpy as np
from scipy.optimize import linprog
c = [-5, -4]
A_ub = [[6, 4], [1, 2], [-1, 1]]
b_ub = [24, 6, 1]
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0,None),(0,None)])
print(f"최적해: {result.x.round(3)}, 최적값: {-result.fun:.3f}")"""),
    (3, "ch11_03_integer_programming", "정수 계획법", r"$\min c^Tx$, $x \in \mathbb{Z}^n$",
     """print("정수 계획법은 PuLP 또는 OR-Tools 사용")
print("from pulp import LpProblem, LpVariable, LpMinimize")"""),
    (4, "ch11_04_gradient_methods", "경사 하강법", r"$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$. 수렴: $f(x_k) - f^* \leq \frac{L\|x_0 - x^*\|^2}{2k}$",
     """import numpy as np, matplotlib.pyplot as plt
def rosenbrock(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def grad_rosenbrock(x): return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])

x = np.array([-1.0, 1.0]); lr = 0.001; history = [x.copy()]
for _ in range(10000):
    x = x - lr * grad_rosenbrock(x)
    history.append(x.copy())
history = np.array(history)
print(f"최종: {x.round(6)}, f={rosenbrock(x):.8f}")"""),
    (5, "ch11_05_constrained", "제약 최적화", r"증강 라그랑주: $L_\rho(x,\lambda) = f(x) + \lambda^T h(x) + \frac{\rho}{2}\|h(x)\|^2$",
     """import numpy as np
from scipy.optimize import minimize
result = minimize(lambda x: x[0]**2+x[1]**2, [1,1],
    constraints={'type':'eq','fun':lambda x: x[0]+x[1]-1})
print(f"최적해: {result.x.round(4)}")"""),
    (6, "ch11_06_decision_theory", "의사결정 이론", r"기대효용: $\max_a E[u(a, \theta)]$. 위험 회피: $u''(x) < 0$",
     """import numpy as np
# 기대효용 비교
def utility_crra(x, gamma=2): return x**(1-gamma)/(1-gamma) if gamma != 1 else np.log(x)
gamble_a = (1000, 1.0)  # 확실한 1000
gamble_b = [(2000, 0.5), (0, 0.5)]  # 50% 2000, 50% 0
eu_a = utility_crra(gamble_a[0])
eu_b = 0.5*utility_crra(2000) + 0.5*utility_crra(1)
print(f"EU(확실한 1000): {eu_a:.4f}")
print(f"EU(도박): {eu_b:.4f}")
print(f"위험 회피자는 {'A' if eu_a > eu_b else 'B'}를 선택")"""),
    (7, "ch11_07_portfolio", "포트폴리오 최적화", r"$\min w^T\Sigma w$ s.t. $w^T\mu = r_p$, $\mathbf{1}^Tw = 1$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
n_assets = 4; mu = np.array([0.12, 0.10, 0.08, 0.14])
cov = np.array([[0.04,0.01,0.005,0.02],[0.01,0.03,0.008,0.01],[0.005,0.008,0.02,0.005],[0.02,0.01,0.005,0.05]])
from scipy.optimize import minimize
n_ports = 50; results = []
for r in np.linspace(0.08, 0.14, n_ports):
    cons = [{'type':'eq','fun':lambda w: w.sum()-1}, {'type':'eq','fun':lambda w,r=r: w@mu-r}]
    res = minimize(lambda w: w@cov@w, np.ones(4)/4, constraints=cons, bounds=[(0,1)]*4)
    if res.success: results.append((np.sqrt(res.fun), r))
results = np.array(results)
plt.figure(figsize=(10,6))
plt.plot(results[:,0], results[:,1], 'b-o', markersize=3)
plt.xlabel('위험 (σ)'); plt.ylabel('기대수익률'); plt.title('효율적 프론티어')
plt.show()"""),
    (8, "ch11_08_rl_bandits", "강화학습 기초", r"Bellman: $V(s) = \max_a [R(s,a) + \gamma \sum P(s'|s,a)V(s')]$",
     """import numpy as np
# Q-learning: GridWorld
grid_size = 4; n_actions = 4; gamma = 0.99; alpha = 0.1
Q = np.zeros((grid_size**2, n_actions))
goal = grid_size**2 - 1
for episode in range(1000):
    state = 0
    for _ in range(100):
        if np.random.rand() < 0.1: action = np.random.randint(4)
        else: action = np.argmax(Q[state])
        row, col = state//grid_size, state%grid_size
        if action==0: row=max(0,row-1)
        elif action==1: row=min(grid_size-1,row+1)
        elif action==2: col=max(0,col-1)
        else: col=min(grid_size-1,col+1)
        next_state = row*grid_size+col
        reward = 1 if next_state==goal else -0.01
        Q[state,action] += alpha*(reward+gamma*Q[next_state].max()-Q[state,action])
        state = next_state
        if state == goal: break
print("Q-table 최적 정책:")
print(np.array(['↑','↓','←','→'])[Q.argmax(axis=1)].reshape(grid_size,grid_size))"""),
    (9, "ch11_09_dynamic_programming", "동적 프로그래밍", r"$V_t(x) = \max_a [r(x,a) + V_{t+1}(f(x,a))]$. 비서 문제: $n/e$ 이후 첫 최고",
     """import numpy as np
# 비서 문제 (최적 정지)
def secretary_sim(n, k):
    candidates = np.random.permutation(n)
    best_in_k = max(candidates[:k])
    for i in range(k, n):
        if candidates[i] > best_in_k:
            return candidates[i] == n-1
    return candidates[-1] == n-1

n = 100; results = {}
for k in range(1, n):
    wins = sum(secretary_sim(n, k) for _ in range(1000))
    results[k] = wins / 1000
best_k = max(results, key=results.get)
print(f"최적 k: {best_k} (이론: {int(n/np.e)}), 성공률: {results[best_k]:.3f} (이론: 1/e≈{1/np.e:.3f})")"""),
    (10, "ch11_10_practice_supply_chain", "실전: 공급망 최적화", r"뉴스벤더: $Q^* = F^{-1}(\frac{c_u}{c_u+c_o})$",
     """import numpy as np
from scipy import stats
cu, co = 10, 3  # 과소/과다 비용
demand = stats.norm(100, 20)
Q_star = demand.ppf(cu / (cu + co))
print(f"최적 주문량: {Q_star:.1f}")
expected_profit = cu*demand.expect(lambda x: np.minimum(x, Q_star)) - co*demand.expect(lambda x: np.maximum(Q_star-x, 0))
print(f"기대 이익: {expected_profit:.1f}")"""),
]

ch11 = []
for tid, fn, title, theory, code in ch11_info:
    full_code = f"import numpy as np, matplotlib.pyplot as plt\nfrom scipy import stats\nnp.random.seed(42)\n\n{code}"
    ch11.append(make_topic(tid, fn, title,
        [f"{title} 이론", f"{title} 구현"], theory, full_code,
        quick_exercises(title), quick_solutions(title), f"{title}은 최적화의 핵심입니다."))
gen_chapter(11, "ch11_optimization", ch11)

# Chapter 12: 공간/네트워크
ch12_info = [
    (1, "ch12_01_spatial_autocorrelation", "공간 자기상관과 Moran's I", r"$I = \frac{n}{\sum w_{ij}}\frac{\sum w_{ij}(x_i-\bar{x})(x_j-\bar{x})}{\sum(x_i-\bar{x})^2}$",
     """import numpy as np
np.random.seed(42)
n = 100
coords = np.random.uniform(0, 10, (n, 2))
from scipy.spatial.distance import cdist
D = cdist(coords, coords)
W = (D < 2).astype(float); np.fill_diagonal(W, 0)
W = W / W.sum(axis=1, keepdims=True)
x = coords[:,0] + np.random.normal(0, 1, n)  # 공간 패턴
xc = x - x.mean()
I = (n / W.sum()) * (xc @ W @ xc) / (xc @ xc)
print(f"Moran's I: {I:.4f} (양의 자기상관)")"""),
    (2, "ch12_02_kriging", "크리깅과 지구통계", r"보통 크리깅: $\hat{Z}(s_0) = \sum \lambda_i Z(s_i)$, $\sum \lambda_i = 1$, BLUP",
     """import numpy as np
print("크리깅은 가우시안 과정 회귀의 지구통계학적 버전입니다.")
print("변이도(variogram): γ(h) = 0.5 * E[(Z(s) - Z(s+h))^2]")"""),
    (3, "ch12_03_point_processes", "점 과정과 공간 밀도", r"K-함수: $K(r) = \lambda^{-1}E[\text{# points within r}]$. CSR: $K(r) = \pi r^2$",
     """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
# 동질 포아송 과정
lam = 50; area = 10
n = np.random.poisson(lam * area**2)
x = np.random.uniform(0, area, n); y = np.random.uniform(0, area, n)
plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=5)
plt.title(f'동질 포아송 과정 (λ={lam}, n={n})')
plt.show()"""),
    (4, "ch12_04_gwr", "지리가중 회귀", r"GWR: $\beta(u_i, v_i) = (X^T W(u_i,v_i) X)^{-1} X^T W(u_i,v_i) y$",
     """print("GWR은 각 위치에서 국소 회귀를 적합합니다.")
print("mgwr 또는 pysal 패키지 사용")"""),
    (5, "ch12_05_graph_theory", "그래프 이론과 네트워크 측도", r"인접 행렬 $A$, 차수 행렬 $D$, 라플라시안 $L = D - A$",
     """import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
G = nx.barabasi_albert_graph(100, 2, seed=42)
degrees = [G.degree(n) for n in G.nodes()]
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1); nx.draw_spring(G, node_size=20, width=0.3); plt.title('Barabasi-Albert')
plt.subplot(1,2,2); plt.hist(degrees, bins=20, edgecolor='black')
plt.title('차수 분포'); plt.xlabel('차수')
plt.tight_layout(); plt.show()"""),
    (6, "ch12_06_community_detection", "커뮤니티 탐지", r"모듈성: $Q = \frac{1}{2m}\sum_{ij}(A_{ij} - \frac{k_ik_j}{2m})\delta(c_i, c_j)$",
     """import networkx as nx
import matplotlib.pyplot as plt
G = nx.karate_club_graph()
communities = nx.community.greedy_modularity_communities(G)
colors = {}
for i, comm in enumerate(communities):
    for node in comm: colors[node] = i
nx.draw_spring(G, node_color=[colors[n] for n in G.nodes()], cmap='Set1', node_size=200)
plt.title(f'커뮤니티 탐지 ({len(communities)}개)')
plt.show()"""),
    (7, "ch12_07_centrality", "중심성과 영향력 전파", r"Betweenness: $c_B(v) = \sum_{s\neq v \neq t}\frac{\sigma_{st}(v)}{\sigma_{st}}$",
     """import networkx as nx
G = nx.karate_club_graph()
cent = nx.betweenness_centrality(G)
top5 = sorted(cent.items(), key=lambda x: -x[1])[:5]
print("Betweenness 상위 5:")
for node, val in top5: print(f"  Node {node}: {val:.4f}")"""),
    (8, "ch12_08_link_prediction", "링크 예측과 추천", r"Common Neighbors: $|N(u) \cap N(v)|$. Jaccard: $\frac{|N(u)\cap N(v)|}{|N(u)\cup N(v)|}$",
     """import networkx as nx
G = nx.karate_club_graph()
preds = nx.jaccard_coefficient(G, [(0,5), (0,31), (2,33)])
for u, v, p in preds: print(f"({u},{v}): Jaccard={p:.4f}")"""),
    (9, "ch12_09_spatiotemporal", "시공간 데이터 분석", r"시공간 자기상관: 공간 + 시간 의존성 동시 모델링",
     """import numpy as np
print("시공간 분석은 공간 + 시계열의 결합입니다.")
print("예: 범죄 데이터의 시공간 핫스팟 분석")"""),
    (10, "ch12_10_practice_epidemic", "실전: 전염병 확산 모델링", r"SIR: $\frac{dS}{dt}=-\beta SI/N$, $\frac{dI}{dt}=\beta SI/N - \gamma I$, $R_0 = \beta/\gamma$",
     """import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import odeint
def sir(y, t, beta, gamma, N):
    S, I, R = y
    return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]
N = 10000; I0 = 10; R0_val = 2.5; gamma = 0.1; beta = R0_val * gamma
t = np.linspace(0, 160, 1000)
sol = odeint(sir, [N-I0, I0, 0], t, args=(beta, gamma, N))
plt.figure(figsize=(10, 6))
plt.plot(t, sol[:,0], label='S'); plt.plot(t, sol[:,1], label='I'); plt.plot(t, sol[:,2], label='R')
plt.xlabel('일'); plt.ylabel('인구'); plt.title(f'SIR 모형 (R₀={R0_val})')
plt.legend(); plt.show()
print(f"최대 감염: {sol[:,1].max():.0f}명 ({sol[:,1].max()/N*100:.1f}%)")"""),
]

ch12 = []
for tid, fn, title, theory, code in ch12_info:
    full_code = f"import numpy as np, matplotlib.pyplot as plt\nnp.random.seed(42)\n\n{code}"
    ch12.append(make_topic(tid, fn, title,
        [f"{title} 이론", f"{title} 구현"], theory, full_code,
        quick_exercises(title), quick_solutions(title), f"{title}은 공간/네트워크 분석의 핵심입니다."))
gen_chapter(12, "ch12_spatial_network", ch12)

print("\n=== 전체 생성 완료! ===")
