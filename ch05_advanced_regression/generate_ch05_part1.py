"""Ch05 Advanced Regression - Topics 1-5 generator"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.notebook_generator import problem_notebook, solution_notebook

OUT = os.path.dirname(__file__)
CH = 5

# ============================================================
# Topic 1: GLM
# ============================================================
problem_notebook(
    chapter_num=CH, section_num=1,
    title="일반화 선형 모형 GLM",
    objectives=[
        "지수족(Exponential Family) 분포의 정의와 성질 이해",
        "링크 함수(Link Function)의 역할과 정준 링크 도출",
        "편차(Deviance)와 모형 적합도 평가",
        "준우도(Quasi-Likelihood) 추정의 원리와 활용",
    ],
    theory_md=r"""
### 1. 지수족 분포 (Exponential Family)

GLM의 반응변수 $Y$는 지수족에 속하는 분포를 따른다고 가정합니다:

$$f(y;\theta,\phi) = \exp\!\left[\frac{y\theta - b(\theta)}{a(\phi)} + c(y,\phi)\right]$$

여기서:
- $\theta$: **정준 모수** (canonical parameter)
- $\phi$: **산포 모수** (dispersion parameter)
- $b(\theta)$: 누적량 생성함수 — $E[Y] = b'(\theta)$, $\text{Var}(Y) = b''(\theta)\,a(\phi)$

| 분포 | $\theta$ | $b(\theta)$ | $a(\phi)$ | 정준 링크 |
|------|----------|-------------|-----------|-----------|
| 정규 | $\mu$ | $\theta^2/2$ | $\sigma^2$ | 항등 |
| 이항 | $\log\frac{p}{1-p}$ | $\log(1+e^\theta)$ | $1/n$ | 로짓 |
| 포아송 | $\log\mu$ | $e^\theta$ | $1$ | 로그 |
| 감마 | $-1/\mu$ | $-\log(-\theta)$ | $\phi$ | 역수 |

### 2. 링크 함수 (Link Function)

링크 함수 $g(\cdot)$는 평균 $\mu = E[Y]$를 선형 예측자 $\eta = X\beta$에 연결합니다:

$$g(\mu_i) = \eta_i = \mathbf{x}_i^\top \boldsymbol{\beta}$$

**정준 링크**: $g(\mu) = \theta$일 때, 즉 $\eta = \theta$일 때의 링크 함수

장점: 충분통계량이 존재하고, Fisher 정보행렬이 관측 정보행렬과 일치

### 3. 모수 추정: IRLS

GLM 모수는 **반복 재가중 최소제곱법**(IRLS)으로 추정합니다:

$$\boldsymbol{\beta}^{(t+1)} = (X^\top W^{(t)} X)^{-1} X^\top W^{(t)} \mathbf{z}^{(t)}$$

여기서:
- $W = \text{diag}\!\left[\frac{1}{V(\mu_i)(g'(\mu_i))^2}\right]$: 가중행렬
- $z_i = \eta_i + (y_i - \mu_i)\,g'(\mu_i)$: 작업 반응변수 (working response)

### 4. 편차 (Deviance)

포화모형 대비 현재 모형의 적합도를 측정합니다:

$$D(y;\hat{\mu}) = 2\sum_{i=1}^{n}\left[\ell(\tilde{\mu}_i; y_i) - \ell(\hat{\mu}_i; y_i)\right] \cdot a(\phi)$$

- 정규분포: $D = \sum(y_i - \hat{\mu}_i)^2$ (잔차제곱합과 동일)
- 포아송: $D = 2\sum\left[y_i\log\frac{y_i}{\hat{\mu}_i} - (y_i - \hat{\mu}_i)\right]$
- 이항: $D = 2\sum\left[y_i\log\frac{y_i}{\hat{\mu}_i} + (1-y_i)\log\frac{1-y_i}{1-\hat{\mu}_i}\right]$

### 5. 준우도 (Quasi-Likelihood)

분포를 완전히 지정하지 않고, 평균-분산 관계만 가정:

$$V(\mu_i) = \phi \cdot v(\mu_i)$$

준우도 함수:
$$Q(\mu; y) = \int_{y}^{\mu} \frac{y - t}{V(t)} dt$$

**과분산(Overdispersion)** 처리에 유용: $\hat{\phi} = \frac{1}{n-p}\sum \frac{(y_i - \hat{\mu}_i)^2}{V(\hat{\mu}_i)}$
""",
    guided_code="""# GLM 구현 가이드
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, Binomial, Gamma, Gaussian
from statsmodels.genmod.families.links import Log, Logit, Identity, InversePower
import matplotlib.pyplot as plt

# --- 1. 포아송 GLM 예제 ---
np.random.seed(42)
n = 200
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
eta = 0.5 + 0.8*x1 - 0.3*x2
mu = np.exp(eta)
y_pois = np.random.poisson(mu)

X = pd.DataFrame({'x1': x1, 'x2': x2})
X_const = sm.add_constant(X)

# 포아송 GLM 적합
model_pois = sm.GLM(y_pois, X_const, family=Poisson(link=Log()))
result_pois = model_pois.fit()
print(result_pois.summary())

# --- 2. 편차 및 적합도 ---
print(f"\\nDeviance: {result_pois.deviance:.4f}")
print(f"Pearson chi2: {result_pois.pearson_chi2:.4f}")
print(f"자유도: {result_pois.df_resid}")

# 편차 / 자유도 ≈ 1 이면 적합도 양호
print(f"Deviance/df: {result_pois.deviance/result_pois.df_resid:.4f}")

# --- 3. IRLS 수동 구현 ---
def irls_poisson(X, y, max_iter=25, tol=1e-8):
    \"\"\"포아송 GLM의 IRLS 수동 구현\"\"\"
    beta = np.zeros(X.shape[1])
    for iteration in range(max_iter):
        eta = X @ beta
        mu = np.exp(eta)
        # 가중치: W = diag(mu) (포아송 + log link)
        W = np.diag(mu)
        # 작업 반응변수
        z = eta + (y - mu) / mu
        # 가중 최소제곱
        XtWX = X.T @ W @ X
        XtWz = X.T @ W @ z
        beta_new = np.linalg.solve(XtWX, XtWz)
        if np.max(np.abs(beta_new - beta)) < tol:
            print(f"IRLS 수렴: {iteration+1}회 반복")
            break
        beta = beta_new
    return beta

beta_manual = irls_poisson(X_const.values, y_pois)
print(f"\\n수동 IRLS: {beta_manual}")
print(f"statsmodels: {result_pois.params.values}")

# --- 4. 다양한 링크 함수 비교 ---
# 감마 GLM (양수 연속 반응변수)
y_gamma = np.random.gamma(shape=2, scale=np.exp(0.3 + 0.5*x1), size=n)

for link_name, link_fn in [('log', Log()), ('inverse_power', InversePower())]:
    model = sm.GLM(y_gamma, X_const, family=Gamma(link=link_fn))
    res = model.fit()
    print(f"\\nGamma GLM ({link_name} link) - AIC: {res.aic:.2f}, Deviance: {res.deviance:.4f}")

# --- 5. 준우도: 과분산 포아송 ---
# 과분산 데이터 생성
y_overdisp = np.random.negative_binomial(n=3, p=3/(3+mu), size=n)

model_quasi = sm.GLM(y_overdisp, X_const, family=Poisson())
res_quasi = model_quasi.fit(scale='X2')  # Pearson chi2로 스케일 추정
print(f"\\n과분산 추정 phi: {res_quasi.scale:.4f}")
print(res_quasi.summary())""",
    exercises=[
        {
            "difficulty": "★",
            "description": "주어진 데이터에서 포아송 GLM을 적합하고, 회귀계수의 해석(승수적 효과)을 서술하세요.\n\n$e^{\\beta_j}$는 $x_j$가 1단위 증가할 때 $\\mu$의 **배수** 변화입니다.",
            "hint": "np.exp(coef)로 IRR(Incidence Rate Ratio)을 계산하세요.",
            "skeleton": "# 데이터 생성\nnp.random.seed(123)\nn = 300\nage = np.random.uniform(20, 70, n)\nexposure = np.random.uniform(0.5, 5, n)\neta = -2 + 0.03*age + np.log(exposure)\ny = np.random.poisson(np.exp(eta))\n\ndf = pd.DataFrame({'age': age, 'exposure': exposure, 'count': y})\n\n# TODO: 포아송 GLM 적합 (offset으로 log(exposure) 사용)\n# TODO: IRR 계산 및 해석\n"
        },
        {
            "difficulty": "★★",
            "description": "이항 GLM에서 로짓, 프로빗, cloglog 링크 함수를 비교하고, AIC 기준 최적 모형을 선택하세요.\n\n각 링크 함수의 수학적 정의:\n- 로짓: $g(p) = \\log\\frac{p}{1-p}$\n- 프로빗: $g(p) = \\Phi^{-1}(p)$\n- cloglog: $g(p) = \\log(-\\log(1-p))$",
            "hint": "statsmodels의 Binomial family에 각 링크를 지정하세요.",
            "skeleton": "# 데이터 생성\nfrom statsmodels.genmod.families.links import Logit, Probit, CLogLog\n\nnp.random.seed(42)\nn = 500\nx = np.random.normal(0, 1, n)\np_true = 1 / (1 + np.exp(-(0.5 + 1.5*x)))\ny = np.random.binomial(1, p_true)\n\n# TODO: 세 가지 링크 함수로 GLM 적합\n# TODO: AIC 비교 표 작성\n# TODO: 적합값 곡선 시각화\n"
        },
        {
            "difficulty": "★★",
            "description": "포아송 GLM의 편차(Deviance)를 수동으로 계산하고, `statsmodels` 결과와 비교하세요.\n\n포아송 편차 공식:\n$$D = 2\\sum_{i=1}^{n}\\left[y_i\\log\\frac{y_i}{\\hat{\\mu}_i} - (y_i - \\hat{\\mu}_i)\\right]$$",
            "skeleton": "# 데이터 및 모형 적합\nnp.random.seed(42)\nn = 200\nx = np.random.normal(0, 1, n)\ny = np.random.poisson(np.exp(1 + 0.5*x))\n\nX = sm.add_constant(x)\nresult = sm.GLM(y, X, family=sm.families.Poisson()).fit()\n\n# TODO: 편차 수동 계산\n# TODO: result.deviance와 비교\n# TODO: 편차 잔차 계산 및 정규성 확인 (QQ plot)\n"
        },
        {
            "difficulty": "★★★",
            "description": "IRLS 알고리즘을 **감마 GLM (로그 링크)**에 대해 직접 구현하세요.\n\n감마 분포 + 로그 링크:\n- $\\mu_i = \\exp(\\mathbf{x}_i^\\top\\boldsymbol{\\beta})$\n- $V(\\mu_i) = \\mu_i^2$\n- $g'(\\mu_i) = 1/\\mu_i$\n- $W_{ii} = \\mu_i^2 / (\\mu_i^2 \\cdot (1/\\mu_i)^2) = 1$ (상수!)",
            "skeleton": "def irls_gamma_log(X, y, max_iter=50, tol=1e-8):\n    \"\"\"감마 GLM (로그 링크) IRLS 구현\"\"\"\n    n, p = X.shape\n    beta = np.zeros(p)\n    \n    for it in range(max_iter):\n        # TODO: eta, mu 계산\n        # TODO: 가중치 W 계산 (감마 + 로그 링크)\n        # TODO: 작업 반응변수 z 계산\n        # TODO: beta 업데이트\n        # TODO: 수렴 확인\n        pass\n    \n    return beta\n\n# 테스트\nnp.random.seed(42)\nn = 300\nx = np.random.normal(0, 1, n)\nmu_true = np.exp(1 + 0.5*x)\ny = np.random.gamma(shape=5, scale=mu_true/5, size=n)\n\nX = sm.add_constant(x)\n# beta_irls = irls_gamma_log(X, y)\n# statsmodels 결과와 비교\n"
        },
        {
            "difficulty": "★★★",
            "description": "준우도(Quasi-Likelihood)를 사용하여 과분산 카운트 데이터를 분석하세요.\n\n1. 일반 포아송 vs 준포아송(Quasi-Poisson) 모형 비교\n2. 과분산 모수 $\\hat{\\phi}$ 추정\n3. 표준오차 변화 분석",
            "skeleton": "# 과분산 카운트 데이터\nnp.random.seed(42)\nn = 400\nx1 = np.random.normal(0, 1, n)\nx2 = np.random.binomial(1, 0.5, n)\nmu = np.exp(1 + 0.5*x1 + 0.3*x2)\n# 음이항으로 과분산 생성\ny = np.random.negative_binomial(n=2, p=2/(2+mu))\n\nX = sm.add_constant(pd.DataFrame({'x1': x1, 'x2': x2}))\n\n# TODO: 포아송 GLM 적합\n# TODO: 과분산 검정 (Deviance/df, Pearson/df)\n# TODO: 준포아송 모형 적합 (scale='X2')\n# TODO: 표준오차 비교 표 작성\n"
        }
    ],
    references=[
        "McCullagh & Nelder (1989). Generalized Linear Models, 2nd ed.",
        "Dobson & Barnett (2018). An Introduction to Generalized Linear Models, 4th ed.",
        "statsmodels GLM documentation: https://www.statsmodels.org/stable/glm.html",
    ],
    filepath=os.path.join(OUT, "ch05_01_glm_problems.ipynb")
)

solution_notebook(
    chapter_num=CH, section_num=1,
    title="일반화 선형 모형 GLM",
    solutions=[
        {
            "approach": "포아송 GLM 적합 및 IRR 해석. offset 항으로 노출(exposure)을 포함합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(123)
n = 300
age = np.random.uniform(20, 70, n)
exposure = np.random.uniform(0.5, 5, n)
eta = -2 + 0.03*age + np.log(exposure)
y = np.random.poisson(np.exp(eta))

df = pd.DataFrame({'age': age, 'exposure': exposure, 'count': y})

# 포아송 GLM with offset
X = sm.add_constant(df[['age']])
model = sm.GLM(df['count'], X, family=sm.families.Poisson(),
               offset=np.log(df['exposure']))
result = model.fit()
print(result.summary())

# IRR 계산
irr = np.exp(result.params)
irr_ci = np.exp(result.conf_int())
print("\\n=== IRR (Incidence Rate Ratio) ===")
print(f"age: IRR = {irr['age']:.4f}")
print(f"  → 나이 1세 증가 시 발생률 {(irr['age']-1)*100:.2f}% 증가")
print(f"  95% CI: ({irr_ci.loc['age', 0]:.4f}, {irr_ci.loc['age', 1]:.4f})")""",
            "interpretation": "IRR(age) ≈ 1.03은 나이가 1세 증가할 때 사건 발생률이 약 3% 증가함을 의미합니다. offset으로 log(exposure)를 포함하여 노출 기간을 보정했습니다."
        },
        {
            "approach": "세 가지 링크 함수(로짓, 프로빗, cloglog)로 이항 GLM을 적합하고 AIC를 비교합니다.",
            "code": """import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit, Probit, CLogLog
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500
x = np.random.normal(0, 1, n)
p_true = 1 / (1 + np.exp(-(0.5 + 1.5*x)))
y = np.random.binomial(1, p_true)
X = sm.add_constant(x)

links = {'Logit': Logit(), 'Probit': Probit(), 'CLogLog': CLogLog()}
results = {}
for name, link in links.items():
    model = sm.GLM(y, X, family=Binomial(link=link))
    results[name] = model.fit()

# AIC 비교
print("=== 링크 함수 비교 ===")
print(f"{'Link':<10} {'AIC':<10} {'Deviance':<12} {'beta0':<10} {'beta1':<10}")
for name, res in results.items():
    print(f"{name:<10} {res.aic:<10.2f} {res.deviance:<12.4f} "
          f"{res.params[0]:<10.4f} {res.params[1]:<10.4f}")

# 적합값 곡선 시각화
x_plot = np.linspace(-3, 3, 200)
X_plot = sm.add_constant(x_plot)

plt.figure(figsize=(10, 6))
for name, res in results.items():
    p_hat = res.predict(X_plot)
    plt.plot(x_plot, p_hat, label=f'{name} (AIC={res.aic:.1f})')

plt.scatter(x, y, alpha=0.1, color='gray', label='관측값')
plt.xlabel('x'); plt.ylabel('P(Y=1)')
plt.title('링크 함수 비교')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()""",
            "interpretation": "로짓 링크는 대칭적 S-커브, 프로빗은 약간 더 가파른 전이, cloglog는 비대칭 커브를 보입니다. 데이터가 로짓으로 생성되었으므로 로짓 링크의 AIC가 가장 낮을 것입니다."
        },
        {
            "approach": "포아송 편차를 수식으로 직접 계산하고 statsmodels 결과와 대조합니다.",
            "code": """import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
n = 200
x = np.random.normal(0, 1, n)
y = np.random.poisson(np.exp(1 + 0.5*x))

X = sm.add_constant(x)
result = sm.GLM(y, X, family=sm.families.Poisson()).fit()
mu_hat = result.fittedvalues

# 수동 편차 계산
# D = 2 * sum[ y*log(y/mu) - (y - mu) ]
# y=0인 경우 y*log(y/mu) = 0으로 처리
mask = y > 0
deviance_terms = np.zeros(n)
deviance_terms[mask] = y[mask] * np.log(y[mask] / mu_hat[mask])
deviance_terms -= (y - mu_hat)
D_manual = 2 * np.sum(deviance_terms)

print(f"수동 계산 Deviance: {D_manual:.6f}")
print(f"statsmodels Deviance: {result.deviance:.6f}")
print(f"차이: {abs(D_manual - result.deviance):.2e}")

# 편차 잔차
d_resid = np.sign(y - mu_hat) * np.sqrt(2 * np.abs(deviance_terms))
print(f"\\n편차 잔차 요약:")
print(f"  평균: {np.mean(d_resid):.4f}")
print(f"  표준편차: {np.std(d_resid):.4f}")

# QQ Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
stats.probplot(d_resid, dist="norm", plot=axes[0])
axes[0].set_title('편차 잔차 QQ Plot')
axes[1].scatter(mu_hat, d_resid, alpha=0.5)
axes[1].axhline(0, color='r', linestyle='--')
axes[1].set_xlabel('적합값'); axes[1].set_ylabel('편차 잔차')
axes[1].set_title('편차 잔차 vs 적합값')
plt.tight_layout(); plt.show()""",
            "interpretation": "수동 계산한 편차와 statsmodels 결과가 기계 정밀도 수준에서 일치합니다. QQ plot에서 편차 잔차가 표준정규분포에 근사하면 모형 적합이 양호한 것입니다."
        },
        {
            "approach": "감마 GLM + 로그 링크의 IRLS를 직접 구현합니다. 로그 링크 + 감마에서 가중치가 상수(=1)가 되는 것이 핵심입니다.",
            "code": """import numpy as np
import statsmodels.api as sm

def irls_gamma_log(X, y, max_iter=50, tol=1e-8):
    \"\"\"감마 GLM (로그 링크) IRLS 구현\"\"\"
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = np.log(np.mean(y))  # 초기값

    for it in range(max_iter):
        eta = X @ beta
        mu = np.exp(eta)

        # 감마 + 로그 링크:
        # V(mu) = mu^2, g'(mu) = 1/mu
        # W_ii = 1 / (V(mu) * g'(mu)^2) = 1 / (mu^2 * 1/mu^2) = 1
        W = np.ones(n)  # 가중치가 상수!

        # 작업 반응변수: z = eta + (y - mu) * g'(mu) = eta + (y - mu)/mu
        z = eta + (y - mu) / mu

        # 가중 최소제곱 (W = I 이므로 일반 OLS)
        XtWX = X.T @ np.diag(W) @ X
        XtWz = X.T @ np.diag(W) @ z
        beta_new = np.linalg.solve(XtWX, XtWz)

        diff = np.max(np.abs(beta_new - beta))
        beta = beta_new
        if diff < tol:
            print(f"IRLS 수렴: {it+1}회 반복, max|delta| = {diff:.2e}")
            break

    return beta

# 테스트
np.random.seed(42)
n = 300
x = np.random.normal(0, 1, n)
mu_true = np.exp(1 + 0.5*x)
y = np.random.gamma(shape=5, scale=mu_true/5, size=n)

X = sm.add_constant(x)

beta_irls = irls_gamma_log(X, y)
result_sm = sm.GLM(y, X, family=sm.families.Gamma(
    link=sm.families.links.Log())).fit()

print(f"\\n{'':>15} {'Intercept':>12} {'x':>12}")
print(f"{'IRLS 수동':>15} {beta_irls[0]:>12.6f} {beta_irls[1]:>12.6f}")
print(f"{'statsmodels':>15} {result_sm.params[0]:>12.6f} {result_sm.params[1]:>12.6f}")
print(f"\\n감마 + 로그 링크의 핵심: W = I (상수 가중치)")
print(f"→ 작업 반응변수에 대한 단순 OLS와 동일!")""",
            "interpretation": "감마 분포 + 로그 링크 조합에서는 IRLS의 가중치가 상수 1이 되어, 각 반복이 OLS 문제로 축소됩니다. 수동 구현과 statsmodels 결과가 일치하는 것을 확인할 수 있습니다."
        },
        {
            "approach": "준포아송(Quasi-Poisson) 모형으로 과분산 데이터를 분석하고 표준오차 변화를 비교합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)
n = 400
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.5, n)
mu = np.exp(1 + 0.5*x1 + 0.3*x2)
y = np.random.negative_binomial(n=2, p=2/(2+mu))

X = sm.add_constant(pd.DataFrame({'x1': x1, 'x2': x2}))

# 1) 일반 포아송
res_pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()

# 2) 과분산 검정
dev_ratio = res_pois.deviance / res_pois.df_resid
pear_ratio = res_pois.pearson_chi2 / res_pois.df_resid
print("=== 과분산 진단 ===")
print(f"Deviance/df = {dev_ratio:.4f} (1보다 훨씬 크면 과분산)")
print(f"Pearson/df  = {pear_ratio:.4f}")

# 3) 준포아송
res_quasi = sm.GLM(y, X, family=sm.families.Poisson()).fit(scale='X2')
phi_hat = res_quasi.scale
print(f"\\n추정 과분산 모수 phi = {phi_hat:.4f}")

# 4) 표준오차 비교
se_pois = res_pois.bse
se_quasi = res_quasi.bse
ratio = se_quasi / se_pois

print(f"\\n{'변수':<12} {'SE(Poisson)':<14} {'SE(Quasi)':<14} {'비율':<8}")
for var in X.columns:
    print(f"{var:<12} {se_pois[var]:<14.4f} {se_quasi[var]:<14.4f} {ratio[var]:<8.4f}")

print(f"\\n이론적 비율 = sqrt(phi) = {np.sqrt(phi_hat):.4f}")
print("→ 준포아송 SE = 포아송 SE × sqrt(phi)")

# 5) p-value 비교
print(f"\\n{'변수':<12} {'p(Poisson)':<14} {'p(Quasi)':<14}")
for var in X.columns:
    print(f"{var:<12} {res_pois.pvalues[var]:<14.4f} {res_quasi.pvalues[var]:<14.4f}")""",
            "interpretation": "과분산이 존재할 때 일반 포아송 모형은 표준오차를 과소추정합니다. 준포아송 모형의 표준오차는 sqrt(phi)배 증가하여 더 보수적인(현실적인) 추론을 제공합니다. 회귀계수 점추정치는 동일하지만, p-value가 달라져 통계적 유의성 판단이 변할 수 있습니다."
        }
    ],
    discussion="""
### GLM 실무 가이드라인

1. **분포 선택**: 반응변수의 특성 (연속/이산, 범위, 분산구조)에 따라 결정
2. **링크 함수 선택**: 정준 링크가 기본이나, 해석의 편의를 위해 다른 링크도 가능
3. **적합도 진단**: 편차/df ≈ 1 확인, 잔차 분석, 영향력 진단
4. **과분산 처리**: 준우도 또는 음이항 모형 사용
5. **모형 비교**: AIC, BIC, 우도비 검정 (내포 모형인 경우)

### ADP 시험 포인트
- 지수족 분포의 성질 ($E[Y] = b'(\\theta)$, $\\text{Var}(Y) = b''(\\theta)a(\\phi)$)
- 편차 공식과 잔차 검정
- 과분산 판단 기준과 준우도의 역할
""",
    filepath=os.path.join(OUT, "ch05_01_glm_solutions.ipynb")
)

# ============================================================
# Topic 2: Regularization
# ============================================================
problem_notebook(
    chapter_num=CH, section_num=2,
    title="정칙화 회귀 (Ridge/Lasso/Elastic Net)",
    objectives=[
        "Ridge, Lasso, Elastic Net의 목적함수와 기하학적 해석",
        "정칙화 경로(Regularization Path) 시각화 및 해석",
        "최적 하이퍼파라미터 선택 (교차검증)",
        "변수 선택과 다중공선성 처리에서의 활용",
    ],
    theory_md=r"""
### 1. 정칙화 회귀의 목적함수

**OLS**: $\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \|y - X\boldsymbol{\beta}\|_2^2$

**Ridge** ($L_2$ 페널티):
$$\hat{\boldsymbol{\beta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\beta}} \left[\|y - X\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_2^2\right]$$

닫힌 해: $\hat{\boldsymbol{\beta}}_{\text{ridge}} = (X^\top X + \lambda I)^{-1}X^\top y$

**Lasso** ($L_1$ 페널티):
$$\hat{\boldsymbol{\beta}}_{\text{lasso}} = \arg\min_{\boldsymbol{\beta}} \left[\frac{1}{2n}\|y - X\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_1\right]$$

닫힌 해 없음 → 좌표 하강법(Coordinate Descent) 사용

**Elastic Net** ($L_1 + L_2$):
$$\hat{\boldsymbol{\beta}}_{\text{enet}} = \arg\min_{\boldsymbol{\beta}} \left[\frac{1}{2n}\|y - X\boldsymbol{\beta}\|_2^2 + \lambda\left(\alpha\|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2\right)\right]$$

### 2. 기하학적 해석

- **Ridge**: $\|\boldsymbol{\beta}\|_2^2 \le t$ → 구(sphere) 형태의 제약 영역
  - 모든 계수가 0을 향해 **축소**되지만 정확히 0이 되지 않음

- **Lasso**: $\|\boldsymbol{\beta}\|_1 \le t$ → 다이아몬드 형태의 제약 영역
  - 꼭짓점에서 최적해를 가질 확률 높음 → **변수 선택** 효과

- **Elastic Net**: Ridge와 Lasso의 중간 형태
  - 그룹 효과: 상관된 변수들을 함께 선택/제외

### 3. 소프트 임계값 연산자 (Soft Thresholding)

Lasso의 좌표 하강법에서 핵심 연산:

$$S(\hat{\beta}_j, \lambda) = \text{sign}(\hat{\beta}_j)\max(|\hat{\beta}_j| - \lambda, 0)$$

### 4. 편향-분산 트레이드오프

$$\text{MSE} = \text{Bias}^2 + \text{Variance}$$

$\lambda$ 증가 → 편향 증가, 분산 감소

### 5. 최적 $\lambda$ 선택

$k$-fold 교차검증:
$$\text{CV}(\lambda) = \frac{1}{k}\sum_{j=1}^{k}\frac{1}{n_j}\sum_{i \in \text{fold}_j}(y_i - \hat{y}_i^{(-j)})^2$$

- `lambda.min`: CV 오차가 최소인 $\lambda$
- `lambda.1se`: 최소 + 1SE 규칙 → 더 간결한 모형
""",
    guided_code="""# 정칙화 회귀 구현 가이드
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# --- 1. 데이터 생성 (다중공선성 포함) ---
np.random.seed(42)
n, p = 200, 20
X = np.random.randn(n, p)
# 상관된 변수 추가
X[:, 5] = X[:, 0] + np.random.normal(0, 0.1, n)
X[:, 6] = X[:, 1] + np.random.normal(0, 0.1, n)

true_beta = np.zeros(p)
true_beta[:5] = [3, -2, 1.5, -1, 0.5]  # 5개만 비영
y = X @ true_beta + np.random.normal(0, 1, n)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. Ridge 경로 ---
alphas = np.logspace(-2, 6, 100)
ridge_coefs = []
for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_scaled, y)
    ridge_coefs.append(model.coef_)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(p):
    plt.plot(np.log10(alphas), [c[i] for c in ridge_coefs])
plt.xlabel('log10(alpha)'); plt.ylabel('계수')
plt.title('Ridge 경로'); plt.grid(True, alpha=0.3)

# --- 3. Lasso 경로 ---
from sklearn.linear_model import lasso_path
alphas_lasso, lasso_coefs, _ = lasso_path(X_scaled, y, alphas=np.logspace(-4, 1, 100))

plt.subplot(1, 2, 2)
for i in range(p):
    plt.plot(np.log10(alphas_lasso), lasso_coefs[i])
plt.xlabel('log10(alpha)'); plt.ylabel('계수')
plt.title('Lasso 경로'); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# --- 4. 교차검증으로 최적 alpha 선택 ---
ridge_cv = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5)
ridge_cv.fit(X_scaled, y)
print(f"Ridge 최적 alpha: {ridge_cv.alpha_:.4f}")

lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_scaled, y)
print(f"Lasso 최적 alpha: {lasso_cv.alpha_:.4f}")
print(f"Lasso 선택된 변수 수: {np.sum(lasso_cv.coef_ != 0)}")

enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95], cv=5, random_state=42)
enet_cv.fit(X_scaled, y)
print(f"ElasticNet 최적: alpha={enet_cv.alpha_:.4f}, l1_ratio={enet_cv.l1_ratio_:.2f}")

# --- 5. 변수 선택 결과 비교 ---
print("\\n=== 계수 비교 ===")
print(f"{'변수':<6} {'True':<8} {'Ridge':<10} {'Lasso':<10} {'ENet':<10}")
for i in range(p):
    print(f"X{i:<5d} {true_beta[i]:<8.2f} {ridge_cv.coef_[i]:<10.4f} "
          f"{lasso_cv.coef_[i]:<10.4f} {enet_cv.coef_[i]:<10.4f}")""",
    exercises=[
        {
            "difficulty": "★",
            "description": "주어진 데이터에 Ridge, Lasso, Elastic Net을 적용하고, 교차검증 MSE를 비교하세요.",
            "hint": "sklearn의 cross_val_score를 사용하세요.",
            "skeleton": "from sklearn.datasets import make_regression\nfrom sklearn.model_selection import cross_val_score\n\nX, y = make_regression(n_samples=200, n_features=30, n_informative=10,\n                       noise=10, random_state=42)\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\n# TODO: 각 모형 적합 및 CV MSE 비교\n"
        },
        {
            "difficulty": "★★",
            "description": "Lasso의 좌표 하강법(Coordinate Descent)을 직접 구현하세요.\n\n소프트 임계값 연산자를 사용한 업데이트 규칙:\n$$\\hat{\\beta}_j \\leftarrow S\\!\\left(\\frac{1}{n}\\sum_{i=1}^n x_{ij}r_{ij}, \\lambda\\right)$$\n\n여기서 $r_{ij} = y_i - \\sum_{k \\neq j} x_{ik}\\hat{\\beta}_k$",
            "hint": "soft_threshold(z, lam) = sign(z) * max(|z| - lam, 0)",
            "skeleton": "def soft_threshold(z, lam):\n    \"\"\"소프트 임계값 연산자\"\"\"\n    pass\n\ndef lasso_cd(X, y, lam, max_iter=1000, tol=1e-6):\n    \"\"\"Lasso 좌표 하강법\"\"\"\n    n, p = X.shape\n    beta = np.zeros(p)\n    # TODO: 구현\n    return beta\n\n# sklearn LassoCV 결과와 비교\n"
        },
        {
            "difficulty": "★★",
            "description": "Ridge 회귀의 기하학적 해석을 2D에서 시각화하세요.\n\n등고선(OLS 목적함수)과 제약 영역($L_1$, $L_2$ ball)을 함께 그려 최적해의 위치를 보이세요.",
            "skeleton": "# 2변수 문제\nnp.random.seed(42)\nX = np.random.randn(100, 2)\ny = 3*X[:, 0] + 2*X[:, 1] + np.random.normal(0, 0.5, 100)\n\n# OLS 해\nbeta_ols = np.linalg.lstsq(X, y, rcond=None)[0]\n\n# TODO: 등고선 + L1/L2 제약 영역 시각화\n# TODO: Ridge, Lasso 해 표시\n"
        },
        {
            "difficulty": "★★★",
            "description": "Ridge 회귀의 SVD 기반 해석을 구현하세요.\n\n$X = U\\Sigma V^\\top$일 때:\n$$\\hat{\\boldsymbol{\\beta}}_\\text{ridge} = V \\text{diag}\\!\\left(\\frac{\\sigma_j^2}{\\sigma_j^2 + \\lambda}\\right) V^\\top \\hat{\\boldsymbol{\\beta}}_\\text{OLS}$$\n\n축소 인자(shrinkage factor) $d_j = \\frac{\\sigma_j^2}{\\sigma_j^2 + \\lambda}$를 시각화하세요.",
            "skeleton": "# SVD 기반 Ridge 해석\nnp.random.seed(42)\nX = np.random.randn(100, 10)\ny = X[:, :3] @ [2, -1, 0.5] + np.random.normal(0, 0.5, 100)\n\n# TODO: SVD 분해\n# TODO: 축소 인자 계산 및 시각화\n# TODO: 유효 자유도 계산: df(lambda) = sum(d_j)\n"
        }
    ],
    references=[
        "Hastie, Tibshirani & Friedman (2009). Elements of Statistical Learning, Ch.3",
        "Tibshirani (1996). Regression Shrinkage and Selection via the Lasso. JRSSB.",
        "Zou & Hastie (2005). Regularization and Variable Selection via the Elastic Net.",
    ],
    filepath=os.path.join(OUT, "ch05_02_regularization_problems.ipynb")
)

solution_notebook(
    chapter_num=CH, section_num=2,
    title="정칙화 회귀 (Ridge/Lasso/Elastic Net)",
    solutions=[
        {
            "approach": "세 가지 정칙화 모형을 교차검증으로 비교합니다.",
            "code": """import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

X, y = make_regression(n_samples=200, n_features=30, n_informative=10,
                       noise=10, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    'Ridge': RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5),
    'Lasso': LassoCV(cv=5, random_state=42),
    'ElasticNet': ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=42)
}

print(f"{'모형':<15} {'CV MSE':<12} {'최적 alpha':<14} {'비영 계수':<10}")
for name, model in models.items():
    model.fit(X_scaled, y)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    n_nonzero = np.sum(model.coef_ != 0)
    alpha = model.alpha_
    print(f"{name:<15} {-scores.mean():<12.4f} {alpha:<14.6f} {n_nonzero:<10}")""",
            "interpretation": "Lasso는 변수 선택을 수행하여 간결한 모형을 생성합니다. Ridge는 모든 변수를 포함하되 축소합니다. Elastic Net은 둘의 장점을 결합합니다."
        },
        {
            "approach": "소프트 임계값 연산자와 좌표 하강법으로 Lasso를 직접 구현합니다.",
            "code": """import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)

def lasso_cd(X, y, lam, max_iter=1000, tol=1e-6):
    n, p = X.shape
    beta = np.zeros(p)
    r = y.copy()  # 잔차

    for iteration in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            # j번째 변수 제외 잔차
            r += X[:, j] * beta[j]
            # 업데이트
            z_j = X[:, j].T @ r / n
            beta[j] = soft_threshold(z_j, lam)
            r -= X[:, j] * beta[j]

        if np.max(np.abs(beta - beta_old)) < tol:
            print(f"수렴: {iteration+1}회 반복")
            break
    return beta

# 테스트
np.random.seed(42)
n, p = 100, 20
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:5] = [3, -2, 1.5, -1, 0.5]
y = X @ true_beta + np.random.normal(0, 0.5, n)

scaler = StandardScaler()
X_s = scaler.fit_transform(X)
y_c = y - y.mean()

lam = 0.1
beta_cd = lasso_cd(X_s, y_c, lam)

sk_model = Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
sk_model.fit(X_s, y_c)

print(f"\\n{'변수':<6} {'True':<8} {'수동CD':<10} {'sklearn':<10}")
for i in range(min(10, p)):
    print(f"X{i:<5d} {true_beta[i]:<8.2f} {beta_cd[i]:<10.4f} {sk_model.coef_[i]:<10.4f}")""",
            "interpretation": "좌표 하강법은 각 변수를 순환하며 소프트 임계값을 적용합니다. 수동 구현이 sklearn과 일치하는 것을 확인할 수 있으며, 실제로 sklearn 내부도 좌표 하강법을 사용합니다."
        },
        {
            "approach": "2D에서 OLS 등고선과 L1/L2 제약 영역을 시각화합니다.",
            "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso

np.random.seed(42)
X = np.random.randn(100, 2)
y = 3*X[:, 0] + 2*X[:, 1] + np.random.normal(0, 0.5, 100)

beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]

# Ridge/Lasso 해
ridge = Ridge(alpha=5); ridge.fit(X, y)
lasso = Lasso(alpha=0.5); lasso.fit(X, y)

# 등고선 그리드
b0 = np.linspace(-1, 5, 200)
b1 = np.linspace(-1, 4, 200)
B0, B1 = np.meshgrid(b0, b1)
Z = np.zeros_like(B0)
for i in range(len(b0)):
    for j in range(len(b1)):
        beta_try = np.array([B0[j, i], B1[j, i]])
        Z[j, i] = np.sum((y - X @ beta_try)**2)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# L2 (Ridge)
axes[0].contour(B0, B1, Z, levels=30, alpha=0.6, cmap='RdYlBu_r')
theta = np.linspace(0, 2*np.pi, 100)
t_ridge = np.sqrt(np.sum(ridge.coef_**2))
axes[0].plot(t_ridge*np.cos(theta), t_ridge*np.sin(theta), 'g-', lw=2, label='L2 ball')
axes[0].plot(*beta_ols, 'r*', ms=15, label=f'OLS ({beta_ols[0]:.2f}, {beta_ols[1]:.2f})')
axes[0].plot(*ridge.coef_, 'go', ms=10, label=f'Ridge ({ridge.coef_[0]:.2f}, {ridge.coef_[1]:.2f})')
axes[0].set_title('Ridge (L2 제약)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('β₁'); axes[0].set_ylabel('β₂')

# L1 (Lasso)
axes[1].contour(B0, B1, Z, levels=30, alpha=0.6, cmap='RdYlBu_r')
t_lasso = np.sum(np.abs(lasso.coef_))
# L1 다이아몬드
diamond_x = [t_lasso, 0, -t_lasso, 0, t_lasso]
diamond_y = [0, t_lasso, 0, -t_lasso, 0]
axes[1].plot(diamond_x, diamond_y, 'b-', lw=2, label='L1 ball')
axes[1].plot(*beta_ols, 'r*', ms=15, label=f'OLS ({beta_ols[0]:.2f}, {beta_ols[1]:.2f})')
axes[1].plot(*lasso.coef_, 'bs', ms=10, label=f'Lasso ({lasso.coef_[0]:.2f}, {lasso.coef_[1]:.2f})')
axes[1].set_title('Lasso (L1 제약)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('β₁'); axes[1].set_ylabel('β₂')

plt.tight_layout(); plt.show()""",
            "interpretation": "L2 제약은 원형이므로 등고선과 접하는 점이 축 위에 있을 가능성이 낮습니다(축소만). L1 제약은 다이아몬드의 꼭짓점에서 등고선과 접할 가능성이 높아 변수 선택이 발생합니다."
        },
        {
            "approach": "SVD를 통해 Ridge 회귀의 축소 인자와 유효 자유도를 분석합니다.",
            "code": """import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.randn(100, 10)
y = X[:, :3] @ [2, -1, 0.5] + np.random.normal(0, 0.5, 100)

# SVD 분해
U, s, Vt = np.linalg.svd(X, full_matrices=False)
V = Vt.T

# OLS 해
beta_ols = V @ np.diag(1/s) @ U.T @ y

# Ridge 해 (SVD 기반)
lambdas = [0.01, 0.1, 1, 10, 100]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 축소 인자
for lam in lambdas:
    d = s**2 / (s**2 + lam)
    axes[0].plot(range(1, 11), d, 'o-', label=f'λ={lam}')
axes[0].set_xlabel('특이값 번호 j')
axes[0].set_ylabel('축소 인자 d_j')
axes[0].set_title('Ridge 축소 인자: $d_j = \\sigma_j^2 / (\\sigma_j^2 + \\lambda)$')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 유효 자유도
lam_range = np.logspace(-3, 4, 200)
df_eff = [np.sum(s**2 / (s**2 + l)) for l in lam_range]
axes[1].plot(np.log10(lam_range), df_eff)
axes[1].set_xlabel('log10(λ)')
axes[1].set_ylabel('유효 자유도 df(λ)')
axes[1].set_title('Ridge 유효 자유도')
axes[1].axhline(10, color='r', linestyle='--', alpha=0.5, label='p=10 (OLS)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout(); plt.show()

# 수치 검증
lam = 1.0
d = s**2 / (s**2 + lam)
beta_ridge_svd = V @ np.diag(d) @ V.T @ beta_ols

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=lam, fit_intercept=False)
ridge.fit(X, y)

print(f"SVD 기반: {beta_ridge_svd[:5]}")
print(f"sklearn:  {ridge.coef_[:5]}")
print(f"유효 자유도: {np.sum(d):.4f} (p = {X.shape[1]})")""",
            "interpretation": "작은 특이값에 대응하는 방향은 더 많이 축소됩니다. 이는 데이터에서 분산이 작은 방향의 계수를 강하게 정칙화하는 효과입니다. 유효 자유도는 lambda가 커질수록 p에서 0으로 감소합니다."
        }
    ],
    discussion="""
### 정칙화 선택 가이드

| 상황 | 추천 방법 |
|------|-----------|
| 모든 변수가 중요할 것으로 예상 | Ridge |
| 소수 변수만 중요 (sparse) | Lasso |
| 그룹화된 상관 변수 존재 | Elastic Net |
| 다중공선성 심각 | Ridge 또는 Elastic Net |
| 변수 선택이 필요 | Lasso 또는 Elastic Net |

### ADP 시험 포인트
- $L_1$, $L_2$ 페널티의 기하학적 차이와 변수 선택 원리
- 편향-분산 트레이드오프에서 $\\lambda$의 역할
- 교차검증 기반 $\\lambda$ 선택 절차
""",
    filepath=os.path.join(OUT, "ch05_02_regularization_solutions.ipynb")
)

# ============================================================
# Topic 3: Quantile Regression
# ============================================================
problem_notebook(
    chapter_num=CH, section_num=3,
    title="분위수 회귀 (Quantile Regression)",
    objectives=[
        "체크 함수(Check Function)의 정의와 성질",
        "분위수 회귀의 선형 계획(LP) 정식화",
        "조건부 분포의 전체 형태 추정",
        "이분산성 하에서 분위수 회귀의 장점",
    ],
    theory_md=r"""
### 1. 체크 함수 (Check Function / Pinball Loss)

$\tau$-분위수에 대한 비대칭 손실함수:

$$\rho_\tau(u) = u(\tau - \mathbb{1}(u < 0)) = \begin{cases} \tau|u| & u \ge 0 \\ (1-\tau)|u| & u < 0 \end{cases}$$

**분위수 회귀**의 목적함수:

$$\hat{\boldsymbol{\beta}}(\tau) = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} \rho_\tau(y_i - \mathbf{x}_i^\top\boldsymbol{\beta})$$

### 2. 특수한 경우

- $\tau = 0.5$: **중앙값 회귀** (LAD, Least Absolute Deviations)
  - OLS보다 이상치에 강건

- OLS와의 비교:
  - OLS: $E[Y|X]$ 추정 (조건부 평균)
  - QR: $Q_\tau(Y|X)$ 추정 (조건부 분위수)

### 3. 선형 계획법(LP) 정식화

분위수 회귀는 LP로 변환 가능:

$$\min_{\boldsymbol{\beta}, \mathbf{u}, \mathbf{v}} \tau \mathbf{1}^\top\mathbf{u} + (1-\tau)\mathbf{1}^\top\mathbf{v}$$
$$\text{s.t.} \quad y = X\boldsymbol{\beta} + \mathbf{u} - \mathbf{v}, \quad \mathbf{u}, \mathbf{v} \ge 0$$

### 4. 점근 분포

$$\sqrt{n}(\hat{\boldsymbol{\beta}}(\tau) - \boldsymbol{\beta}(\tau)) \xrightarrow{d} N\!\left(0, \frac{\tau(1-\tau)}{f_{Y|X}^2(Q_\tau)}\,(X^\top X)^{-1}\right)$$

여기서 $f_{Y|X}(Q_\tau)$는 조건부 밀도의 분위수에서의 값 (sparsity function의 역수)

### 5. 교차 분위수 문제 (Crossing Quantiles)

서로 다른 $\tau$에서 추정한 분위수 함수가 교차할 수 있음:
- 해결: 동시 추정 (simultaneous quantile regression) 또는 단조 재배열
""",
    guided_code="""# 분위수 회귀 구현 가이드
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt

# --- 1. 이분산 데이터 생성 ---
np.random.seed(42)
n = 500
x = np.random.uniform(0, 10, n)
# 분산이 x에 따라 증가하는 데이터
y = 2 + 1.5*x + (0.5 + 0.3*x) * np.random.normal(0, 1, n)

X = sm.add_constant(x)

# --- 2. OLS vs 분위수 회귀 ---
ols_result = sm.OLS(y, X).fit()

taus = [0.1, 0.25, 0.5, 0.75, 0.9]
qr_results = {}
for tau in taus:
    qr = QuantReg(y, X)
    qr_results[tau] = qr.fit(q=tau)

# --- 3. 시각화 ---
x_plot = np.linspace(0, 10, 100)
X_plot = sm.add_constant(x_plot)

plt.figure(figsize=(12, 6))
plt.scatter(x, y, alpha=0.2, s=10, color='gray')

# OLS
plt.plot(x_plot, ols_result.predict(X_plot), 'k-', lw=2, label='OLS (평균)')

# 분위수 회귀선
colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(taus)))
for tau, color in zip(taus, colors):
    y_pred = qr_results[tau].predict(X_plot)
    plt.plot(x_plot, y_pred, color=color, lw=1.5, label=f'τ={tau}')

plt.xlabel('x'); plt.ylabel('y')
plt.title('분위수 회귀: 이분산 데이터')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# --- 4. 계수 비교 ---
print(f"{'τ':<6} {'절편':<10} {'기울기':<10} {'SE(기울기)':<12}")
print("-" * 40)
for tau in taus:
    r = qr_results[tau]
    print(f"{tau:<6.2f} {r.params[0]:<10.4f} {r.params[1]:<10.4f} {r.bse[1]:<12.4f}")
print(f"{'OLS':<6} {ols_result.params[0]:<10.4f} {ols_result.params[1]:<10.4f} "
      f"{ols_result.bse[1]:<12.4f}")

# --- 5. 체크 함수 시각화 ---
fig, ax = plt.subplots(figsize=(8, 5))
u = np.linspace(-3, 3, 200)
for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
    rho = np.where(u >= 0, tau * u, (tau - 1) * u)
    ax.plot(u, rho, label=f'τ={tau}')
ax.set_xlabel('u = y - Xβ'); ax.set_ylabel('ρ_τ(u)')
ax.set_title('체크 함수 (Check Function)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()""",
    exercises=[
        {
            "difficulty": "★",
            "description": "이분산 데이터에서 OLS와 중앙값 회귀(τ=0.5)를 비교하고, 이상치에 대한 강건성을 확인하세요.",
            "hint": "이상치를 추가한 후 두 모형의 계수 변화를 비교하세요.",
            "skeleton": "# 데이터 + 이상치\nnp.random.seed(42)\nn = 200\nx = np.random.uniform(0, 10, n)\ny = 2 + x + np.random.normal(0, 1, n)\n# 이상치 추가\ny[0:5] = 50\n\n# TODO: OLS vs 중앙값 회귀 비교\n# TODO: 이상치 유무에 따른 계수 변화\n"
        },
        {
            "difficulty": "★★",
            "description": "체크 함수를 이용하여 분위수 회귀를 `scipy.optimize.minimize`로 직접 구현하세요.",
            "skeleton": "from scipy.optimize import minimize\n\ndef check_loss(beta, X, y, tau):\n    \"\"\"체크 함수 손실\"\"\"\n    pass\n\n# TODO: 수동 구현과 QuantReg 비교\n"
        },
        {
            "difficulty": "★★★",
            "description": "다수의 분위수(τ=0.05, 0.10, ..., 0.95)에서 분위수 회귀를 실행하여 **조건부 분포**의 전체 형태를 추정하세요.\n\n특정 $x$ 값에서 추정된 조건부 밀도를 시각화하세요.",
            "hint": "분위수 함수의 차분으로 조건부 밀도를 근사할 수 있습니다.",
            "skeleton": "# 조건부 분포 추정\ntaus = np.arange(0.05, 0.96, 0.05)\n\n# TODO: 모든 tau에서 분위수 회귀 적합\n# TODO: x=3, x=7에서 조건부 분위수 함수 및 밀도 추정\n"
        }
    ],
    references=[
        "Koenker & Bassett (1978). Regression Quantiles. Econometrica.",
        "Koenker (2005). Quantile Regression. Cambridge University Press.",
        "statsmodels QuantReg: https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html",
    ],
    filepath=os.path.join(OUT, "ch05_03_quantile_regression_problems.ipynb")
)

solution_notebook(
    chapter_num=CH, section_num=3,
    title="분위수 회귀 (Quantile Regression)",
    solutions=[
        {
            "approach": "이상치 추가 전후로 OLS와 중앙값 회귀(LAD)의 강건성을 비교합니다.",
            "code": """import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200
x = np.random.uniform(0, 10, n)
y_clean = 2 + x + np.random.normal(0, 1, n)
y_outlier = y_clean.copy()
y_outlier[0:5] = 50

X = sm.add_constant(x)

results = {}
for label, y_data in [('Clean', y_clean), ('Outlier', y_outlier)]:
    results[f'OLS_{label}'] = sm.OLS(y_data, X).fit()
    results[f'LAD_{label}'] = QuantReg(y_data, X).fit(q=0.5)

print(f"{'모형':<20} {'절편':<10} {'기울기':<10}")
for name, r in results.items():
    print(f"{name:<20} {r.params[0]:<10.4f} {r.params[1]:<10.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x_plot = np.linspace(0, 10, 100)
X_plot = sm.add_constant(x_plot)

for ax, (label, y_data) in zip(axes, [('Clean', y_clean), ('Outlier', y_outlier)]):
    ax.scatter(x, y_data, alpha=0.3, s=10)
    ax.plot(x_plot, results[f'OLS_{label}'].predict(X_plot), 'r-', label='OLS')
    ax.plot(x_plot, results[f'LAD_{label}'].predict(X_plot), 'b--', label='LAD')
    ax.set_title(f'{label} 데이터'); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()""",
            "interpretation": "이상치 존재 시 OLS의 기울기는 크게 변하지만, 중앙값 회귀(LAD)는 거의 변하지 않습니다. 이는 L1 손실의 강건성 때문입니다."
        },
        {
            "approach": "scipy.optimize로 체크 함수를 직접 최소화하여 분위수 회귀를 구현합니다.",
            "code": """import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from scipy.optimize import minimize

def check_loss(beta, X, y, tau):
    residuals = y - X @ beta
    return np.sum(np.where(residuals >= 0, tau * residuals, (tau - 1) * residuals))

np.random.seed(42)
n = 300
x = np.random.uniform(0, 10, n)
y = 2 + 1.5*x + (0.5 + 0.3*x) * np.random.normal(0, 1, n)
X = sm.add_constant(x)

taus = [0.1, 0.25, 0.5, 0.75, 0.9]
print(f"{'τ':<6} {'수동 β0':<10} {'SM β0':<10} {'수동 β1':<10} {'SM β1':<10}")
for tau in taus:
    # 수동 구현
    beta0 = np.array([0.0, 0.0])
    res = minimize(check_loss, beta0, args=(X, y, tau), method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-8})
    beta_manual = res.x

    # statsmodels
    qr = QuantReg(y, X).fit(q=tau)

    print(f"{tau:<6.2f} {beta_manual[0]:<10.4f} {qr.params[0]:<10.4f} "
          f"{beta_manual[1]:<10.4f} {qr.params[1]:<10.4f}")""",
            "interpretation": "수동 구현과 statsmodels 결과가 잘 일치합니다. 실제로는 LP 알고리즘이 수치적으로 더 안정적이고 빠릅니다."
        },
        {
            "approach": "다수 분위수에서 회귀를 실행하여 조건부 분포를 추정하고, 특정 x 값에서 밀도를 시각화합니다.",
            "code": """import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500
x = np.random.uniform(0, 10, n)
y = 2 + 1.5*x + (0.5 + 0.3*x) * np.random.normal(0, 1, n)
X = sm.add_constant(x)

taus = np.arange(0.05, 0.96, 0.05)
qr_fits = {}
for tau in taus:
    qr_fits[tau] = QuantReg(y, X).fit(q=tau)

# 특정 x 값에서 조건부 분위수 & 밀도
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, x_val in enumerate([2, 5, 7, 9]):
    ax = axes[idx // 2, idx % 2]
    X_pred = np.array([1, x_val])
    quantiles = [qr_fits[tau].predict(X_pred)[0] for tau in taus]

    ax2 = ax.twinx()
    ax.plot(taus, quantiles, 'b-o', ms=3, label='분위수 함수')
    ax.set_xlabel('τ'); ax.set_ylabel('Q_τ(Y|x)', color='b')

    # 밀도 근사: f(q) ≈ Δτ / ΔQ
    for i in range(len(taus) - 1):
        dq = quantiles[i+1] - quantiles[i]
        if dq > 0:
            density = (taus[i+1] - taus[i]) / dq
            mid_q = (quantiles[i] + quantiles[i+1]) / 2
            ax2.bar(mid_q, density, width=dq*0.8, alpha=0.3, color='orange')

    ax2.set_ylabel('f(y|x)', color='orange')
    ax.set_title(f'x = {x_val}: 조건부 분포')
    ax.grid(True, alpha=0.3)

plt.suptitle('조건부 분위수 함수 & 밀도 추정', fontsize=14)
plt.tight_layout(); plt.show()

print("x가 커질수록 조건부 분포의 분산이 증가 → 이분산성 포착")""",
            "interpretation": "분위수 회귀는 조건부 분포의 전체 형태를 추정할 수 있습니다. x가 커질수록 분위수 간 간격이 넓어지는 것은 이분산성(분산 증가)을 반영합니다."
        }
    ],
    discussion="""
### 분위수 회귀 활용 분야

1. **임금 분석**: 교육의 효과가 임금 분포의 상위/하위에서 다름
2. **금융 위험 관리**: VaR (Value at Risk) = 조건부 하위 분위수
3. **환경 과학**: 극단값 분석
4. **성장 곡선**: 소아 성장 백분위수 차트

### ADP 시험 포인트
- 체크 함수의 비대칭성과 분위수 추정 원리
- OLS(조건부 평균) vs QR(조건부 분위수)의 차이
- 이분산성 하에서 QR의 이점
""",
    filepath=os.path.join(OUT, "ch05_03_quantile_regression_solutions.ipynb")
)

# ============================================================
# Topic 4: GAM
# ============================================================
problem_notebook(
    chapter_num=CH, section_num=4,
    title="일반화 가법 모형 GAM",
    objectives=[
        "GAM의 구조와 비모수적 함수 추정의 원리",
        "스플라인(Spline) 기저 함수의 종류와 매듭(knot) 선택",
        "평활화 페널티와 일반화 교차검증(GCV)",
        "텐서 곱(Tensor Product) 스플라인과 교호작용",
    ],
    theory_md=r"""
### 1. GAM 구조

$$g(E[Y|X]) = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots + f_p(x_p)$$

- 각 $f_j$는 **비모수적 평활 함수**
- 가법적(additive) 구조: 해석 용이
- GLM의 확장: $g$는 링크 함수

### 2. 스플라인 기저 함수

**B-스플라인 기저**: $f(x) = \sum_{k=1}^{K} \beta_k B_k(x)$

**자연 3차 스플라인 (Natural Cubic Spline)**:
- 매듭 사이에서 3차 다항식
- 매듭에서 2차 도함수까지 연속
- 양 끝에서 선형 (경계 조건)

**박판 회귀 스플라인 (Thin-Plate Regression Spline)**:
- 매듭 위치를 데이터에서 자동 결정
- 고유분해 기반 기저 축소

### 3. 페널티 회귀 스플라인

목적함수 (단일 변수):

$$\min_{\boldsymbol{\beta}} \|y - B\boldsymbol{\beta}\|^2 + \lambda \int [f''(x)]^2 dx$$

행렬 형태: $\lambda \boldsymbol{\beta}^\top S \boldsymbol{\beta}$ (여기서 $S$는 페널티 행렬)

해: $\hat{\boldsymbol{\beta}} = (B^\top B + \lambda S)^{-1} B^\top y$

### 4. 평활도 선택

**일반화 교차검증 (GCV)**:

$$\text{GCV}(\lambda) = \frac{n \cdot \text{RSS}}{(n - \text{edf})^2}$$

여기서 **유효 자유도**: $\text{edf} = \text{tr}[B(B^\top B + \lambda S)^{-1}B^\top]$

### 5. 텐서 곱 스플라인

두 변수의 교호작용:

$$f(x_1, x_2) = \sum_j \sum_k \beta_{jk} B_j(x_1) B_k(x_2)$$

- 텐서 곱: $B_j(x_1) \otimes B_k(x_2)$
- 차원이 다른 변수 간의 교호작용 모형화에 유용
""",
    guided_code="""# GAM 구현 가이드
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygam import LinearGAM, LogisticGAM, s, f, te
from sklearn.preprocessing import SplineTransformer

# --- 1. 비선형 데이터 생성 ---
np.random.seed(42)
n = 500
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 10, n)
x3 = np.random.choice([0, 1, 2], n)  # 범주형

y = (np.sin(x1) + 0.5*x2**0.5 +
     np.array([0, 1, -0.5])[x3] +
     np.random.normal(0, 0.5, n))

X = np.column_stack([x1, x2, x3])

# --- 2. GAM 적합 (pyGAM) ---
# s(): 스플라인, f(): 범주형
gam = LinearGAM(s(0) + s(1) + f(2))
gam.gridsearch(X, y)

print(f"GCV: {gam.statistics_['GCV']:.4f}")
print(f"AIC: {gam.statistics_['AIC']:.4f}")

# --- 3. 부분 의존성 도표 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ['f1(x1): sin 패턴', 'f2(x2): sqrt 패턴', 'f3(x3): 범주']

for i, (ax, title) in enumerate(zip(axes, titles)):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=0.95)[1],
            c='r', ls='--')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()

# --- 4. sklearn SplineTransformer로 수동 구현 ---
from sklearn.linear_model import Ridge

spline_x1 = SplineTransformer(n_knots=10, degree=3, include_bias=False)
spline_x2 = SplineTransformer(n_knots=10, degree=3, include_bias=False)

B1 = spline_x1.fit_transform(x1.reshape(-1, 1))
B2 = spline_x2.fit_transform(x2.reshape(-1, 1))
X_spline = np.column_stack([B1, B2, pd.get_dummies(x3).values])

ridge = Ridge(alpha=1.0)
ridge.fit(X_spline, y)
print(f"\\n수동 스플라인 R²: {ridge.score(X_spline, y):.4f}")

# --- 5. EDF 확인 ---
print("\\n유효 자유도 (EDF):")
for i, term in enumerate(gam.terms):
    if hasattr(term, 'n_coefs'):
        edf = gam.statistics_['edof_per_coef'][i] if i < len(gam.statistics_.get('edof_per_coef', [])) else 'N/A'
print(gam.summary())""",
    exercises=[
        {
            "difficulty": "★",
            "description": "비선형 관계가 있는 데이터에 선형 회귀와 GAM을 적합하고 R²를 비교하세요.",
            "hint": "pyGAM의 LinearGAM을 사용하세요.",
            "skeleton": "np.random.seed(42)\nn = 300\nx = np.random.uniform(0, 6, n)\ny = np.sin(x) + 0.3*x + np.random.normal(0, 0.3, n)\n\n# TODO: OLS vs GAM 비교\n# TODO: 적합 곡선 시각화\n"
        },
        {
            "difficulty": "★★",
            "description": "스플라인 기저 함수의 매듭(knot) 수에 따른 과적합/과소적합 변화를 시각화하세요.\n\n매듭 수: 3, 5, 10, 20, 50",
            "skeleton": "# 다양한 매듭 수에서 스플라인 적합\nfrom sklearn.preprocessing import SplineTransformer\nfrom sklearn.linear_model import LinearRegression\n\n# TODO: 매듭 수별 적합 및 시각화\n# TODO: GCV 또는 CV 오차 비교\n"
        },
        {
            "difficulty": "★★",
            "description": "페널티 스플라인의 평활도 매개변수 $\\lambda$에 따른 유효 자유도(EDF) 변화를 분석하세요.",
            "hint": "hat matrix H = B(B'B + λS)^{-1}B'의 trace가 EDF입니다.",
            "skeleton": "# 페널티 스플라인의 EDF\nfrom sklearn.preprocessing import SplineTransformer\n\n# TODO: lambda별 EDF 계산\n# TODO: EDF vs lambda 그래프\n"
        },
        {
            "difficulty": "★★★",
            "description": "텐서 곱 스플라인으로 두 변수의 교호작용을 포함한 GAM을 적합하고, 3D 표면으로 시각화하세요.",
            "skeleton": "# 교호작용이 있는 데이터\nnp.random.seed(42)\nn = 500\nx1 = np.random.uniform(0, 5, n)\nx2 = np.random.uniform(0, 5, n)\ny = np.sin(x1) * np.cos(x2) + np.random.normal(0, 0.2, n)\n\n# TODO: te() 항을 포함한 GAM 적합\n# TODO: 3D 표면 시각화\n"
        }
    ],
    references=[
        "Hastie & Tibshirani (1990). Generalized Additive Models.",
        "Wood (2017). Generalized Additive Models: An Introduction with R, 2nd ed.",
        "pyGAM documentation: https://pygam.readthedocs.io/",
    ],
    filepath=os.path.join(OUT, "ch05_04_gam_problems.ipynb")
)

solution_notebook(
    chapter_num=CH, section_num=4,
    title="일반화 가법 모형 GAM",
    solutions=[
        {
            "approach": "선형 회귀와 GAM의 적합력을 비교합니다.",
            "code": """import numpy as np
import statsmodels.api as sm
from pygam import LinearGAM, s
import matplotlib.pyplot as plt

np.random.seed(42)
n = 300
x = np.random.uniform(0, 6, n)
y = np.sin(x) + 0.3*x + np.random.normal(0, 0.3, n)

# OLS
X_ols = sm.add_constant(x)
ols = sm.OLS(y, X_ols).fit()

# GAM
X_gam = x.reshape(-1, 1)
gam = LinearGAM(s(0, n_splines=15))
gam.gridsearch(X_gam, y)

x_plot = np.linspace(0, 6, 200)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.3, s=10, label='데이터')
plt.plot(x_plot, ols.predict(sm.add_constant(x_plot)), 'r-', lw=2, label=f'OLS (R²={ols.rsquared:.4f})')

y_gam = gam.predict(x_plot.reshape(-1, 1))
ci = gam.confidence_intervals(x_plot.reshape(-1, 1), width=0.95)
plt.plot(x_plot, y_gam, 'b-', lw=2, label=f'GAM (R²={gam.statistics_["pseudo_r2"]["explained_deviance"]:.4f})')
plt.fill_between(x_plot, ci[:, 0], ci[:, 1], alpha=0.2, color='blue')

plt.xlabel('x'); plt.ylabel('y')
plt.title('OLS vs GAM'); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()""",
            "interpretation": "GAM은 sin(x) 패턴을 정확히 포착하여 훨씬 높은 R²를 보입니다. 선형 회귀는 비선형 관계를 포착하지 못합니다."
        },
        {
            "approach": "스플라인 매듭 수에 따른 적합 변화를 시각화합니다.",
            "code": """import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

np.random.seed(42)
n = 300
x = np.random.uniform(0, 6, n)
y = np.sin(x) + 0.3*x + np.random.normal(0, 0.3, n)

n_knots_list = [3, 5, 10, 20, 50]
x_plot = np.linspace(0, 6, 200)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
cv_scores = []

for idx, n_knots in enumerate(n_knots_list):
    ax = axes.flat[idx]
    st = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
    X_sp = st.fit_transform(x.reshape(-1, 1))
    X_sp_plot = st.transform(x_plot.reshape(-1, 1))

    lr = LinearRegression()
    lr.fit(X_sp, y)

    cv = cross_val_score(lr, X_sp, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-cv.mean())

    ax.scatter(x, y, alpha=0.2, s=5)
    ax.plot(x_plot, lr.predict(X_sp_plot), 'r-', lw=2)
    ax.set_title(f'knots={n_knots}, CV-MSE={-cv.mean():.4f}')
    ax.grid(True, alpha=0.3)

# CV-MSE 비교
axes.flat[5].bar(range(len(n_knots_list)), cv_scores)
axes.flat[5].set_xticks(range(len(n_knots_list)))
axes.flat[5].set_xticklabels(n_knots_list)
axes.flat[5].set_xlabel('매듭 수'); axes.flat[5].set_ylabel('CV-MSE')
axes.flat[5].set_title('교차검증 MSE')
plt.tight_layout(); plt.show()""",
            "interpretation": "매듭이 너무 적으면 과소적합, 너무 많으면 과적합이 발생합니다. CV-MSE가 최소인 매듭 수가 최적입니다. 페널티 스플라인은 매듭을 충분히 많이 넣고 페널티로 평활도를 조절합니다."
        },
        {
            "approach": "페널티 행렬을 구성하고 lambda에 따른 EDF 변화를 분석합니다.",
            "code": """import numpy as np
from sklearn.preprocessing import SplineTransformer
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200
x = np.sort(np.random.uniform(0, 6, n))
y = np.sin(x) + 0.3*x + np.random.normal(0, 0.3, n)

# 스플라인 기저
st = SplineTransformer(n_knots=20, degree=3, include_bias=False)
B = st.fit_transform(x.reshape(-1, 1))
K = B.shape[1]

# 2차 차분 페널티 행렬 (근사적 곡률 페널티)
D2 = np.diff(np.eye(K), n=2, axis=0)
S = D2.T @ D2

# lambda별 EDF 계산
lambdas = np.logspace(-4, 6, 100)
edfs = []
gcvs = []

for lam in lambdas:
    H = B @ np.linalg.solve(B.T @ B + lam * S, B.T)
    edf = np.trace(H)
    edfs.append(edf)

    y_hat = H @ y
    rss = np.sum((y - y_hat)**2)
    gcv = n * rss / (n - edf)**2
    gcvs.append(gcv)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# EDF vs lambda
axes[0].plot(np.log10(lambdas), edfs)
axes[0].set_xlabel('log10(λ)'); axes[0].set_ylabel('EDF')
axes[0].set_title('유효 자유도 vs 평활도')
axes[0].axhline(K, color='r', ls='--', label=f'K={K}')
axes[0].axhline(2, color='g', ls='--', label='직선(2)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# GCV vs lambda
axes[1].plot(np.log10(lambdas), gcvs)
opt_idx = np.argmin(gcvs)
axes[1].axvline(np.log10(lambdas[opt_idx]), color='r', ls='--',
                label=f'최적: λ={lambdas[opt_idx]:.4f}')
axes[1].set_xlabel('log10(λ)'); axes[1].set_ylabel('GCV')
axes[1].set_title('GCV 기준 평활도 선택')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 최적 lambda에서의 적합
lam_opt = lambdas[opt_idx]
beta_opt = np.linalg.solve(B.T @ B + lam_opt * S, B.T @ y)
x_plot = np.linspace(0, 6, 200)
B_plot = st.transform(x_plot.reshape(-1, 1))

axes[2].scatter(x, y, alpha=0.3, s=10)
axes[2].plot(x_plot, B_plot @ beta_opt, 'r-', lw=2,
             label=f'EDF={edfs[opt_idx]:.1f}')
axes[2].set_title(f'최적 적합 (λ={lam_opt:.4f})')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout(); plt.show()""",
            "interpretation": "lambda가 0에 가까우면 EDF→K (보간), lambda가 커지면 EDF→2 (직선). GCV가 최소인 lambda에서 편향-분산 균형이 달성됩니다."
        },
        {
            "approach": "텐서 곱 스플라인으로 교호작용을 모형화하고 3D 표면으로 시각화합니다.",
            "code": """import numpy as np
from pygam import LinearGAM, s, te
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
n = 500
x1 = np.random.uniform(0, 5, n)
x2 = np.random.uniform(0, 5, n)
y = np.sin(x1) * np.cos(x2) + np.random.normal(0, 0.2, n)
X = np.column_stack([x1, x2])

# 가법 모형 vs 텐서 곱 모형
gam_add = LinearGAM(s(0) + s(1))
gam_add.gridsearch(X, y)

gam_te = LinearGAM(te(0, 1))
gam_te.gridsearch(X, y)

print(f"가법 모형 GCV: {gam_add.statistics_['GCV']:.4f}")
print(f"텐서 곱 모형 GCV: {gam_te.statistics_['GCV']:.4f}")

# 3D 표면
grid_x1 = np.linspace(0, 5, 50)
grid_x2 = np.linspace(0, 5, 50)
X1, X2 = np.meshgrid(grid_x1, grid_x2)
X_grid = np.column_stack([X1.ravel(), X2.ravel()])

fig = plt.figure(figsize=(18, 5))

# 참값
ax1 = fig.add_subplot(131, projection='3d')
Z_true = np.sin(X1) * np.cos(X2)
ax1.plot_surface(X1, X2, Z_true, cmap='viridis', alpha=0.7)
ax1.set_title('참값: sin(x1)cos(x2)')

# 가법 모형
ax2 = fig.add_subplot(132, projection='3d')
Z_add = gam_add.predict(X_grid).reshape(X1.shape)
ax2.plot_surface(X1, X2, Z_add, cmap='viridis', alpha=0.7)
ax2.set_title(f'가법 모형 (GCV={gam_add.statistics_["GCV"]:.4f})')

# 텐서 곱
ax3 = fig.add_subplot(133, projection='3d')
Z_te = gam_te.predict(X_grid).reshape(X1.shape)
ax3.plot_surface(X1, X2, Z_te, cmap='viridis', alpha=0.7)
ax3.set_title(f'텐서 곱 (GCV={gam_te.statistics_["GCV"]:.4f})')

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('x1'); ax.set_ylabel('x2')

plt.tight_layout(); plt.show()""",
            "interpretation": "sin(x1)*cos(x2)는 비가법적(교호작용) 함수이므로, 가법 모형으로는 정확히 포착할 수 없습니다. 텐서 곱 스플라인은 이러한 교호작용을 포착하여 더 낮은 GCV를 보입니다."
        }
    ],
    discussion="""
### GAM 실무 가이드라인

1. **변수 유형별 항 선택**: 연속형 → s(), 범주형 → f(), 교호작용 → te()
2. **평활도 선택**: GCV 또는 REML 기반 자동 선택
3. **모형 진단**: 잔차 분석, concurvity (다중공선성의 비모수 버전) 확인
4. **해석**: 부분 의존성 도표로 각 변수의 비선형 효과 시각화

### ADP 시험 포인트
- 스플라인 기저 함수의 원리 (매듭, 차수, 경계 조건)
- 페널티 항과 유효 자유도의 관계
- GCV의 정의와 역할
""",
    filepath=os.path.join(OUT, "ch05_04_gam_solutions.ipynb")
)

# ============================================================
# Topic 5: Mixed Effects
# ============================================================
problem_notebook(
    chapter_num=CH, section_num=5,
    title="혼합 효과 모형 (Mixed Effects Model)",
    objectives=[
        "고정 효과(Fixed Effect)와 랜덤 효과(Random Effect)의 구분",
        "REML vs ML 추정의 차이 이해",
        "급내 상관 계수(ICC)의 계산과 해석",
        "랜덤 기울기 모형의 구조와 적합",
    ],
    theory_md=r"""
### 1. 혼합 효과 모형의 구조

$$\mathbf{y}_i = X_i\boldsymbol{\beta} + Z_i\mathbf{b}_i + \boldsymbol{\varepsilon}_i$$

여기서:
- $\boldsymbol{\beta}$: **고정 효과** (Fixed Effects) — 모집단 수준 효과
- $\mathbf{b}_i$: **랜덤 효과** (Random Effects) — 그룹별 변동
- $\mathbf{b}_i \sim N(\mathbf{0}, G)$, $\boldsymbol{\varepsilon}_i \sim N(\mathbf{0}, R_i)$

### 2. 랜덤 절편 모형

$$y_{ij} = \beta_0 + \beta_1 x_{ij} + b_{0i} + \varepsilon_{ij}$$

- $b_{0i} \sim N(0, \sigma_b^2)$: 그룹별 절편 변동
- $\varepsilon_{ij} \sim N(0, \sigma_e^2)$: 개체 내 오차

### 3. ICC (Intraclass Correlation Coefficient)

$$\text{ICC} = \frac{\sigma_b^2}{\sigma_b^2 + \sigma_e^2}$$

- ICC = 0: 그룹 간 차이 없음 → 일반 회귀로 충분
- ICC = 1: 모든 변동이 그룹 간 차이에 의해 설명
- ICC > 0.05: 혼합 모형 고려 필요

### 4. REML vs ML

**ML (Maximum Likelihood)**:
$$\ell(\boldsymbol{\beta}, \theta) = -\frac{1}{2}\left[n\log(2\pi) + \log|V| + (y-X\boldsymbol{\beta})^\top V^{-1}(y-X\boldsymbol{\beta})\right]$$

**REML (Restricted ML)**:
$$\ell_R(\theta) = \ell(\hat{\boldsymbol{\beta}}, \theta) - \frac{1}{2}\log|X^\top V^{-1}X|$$

- REML: 분산 성분의 **비편향** 추정 (자유도 보정과 유사)
- ML: 고정 효과 수가 다른 모형 비교에 사용
- REML: 동일 고정 효과 구조 하에서 분산 구조 비교에 사용

### 5. BLUP (Best Linear Unbiased Prediction)

랜덤 효과의 예측:
$$\hat{\mathbf{b}}_i = G Z_i^\top V_i^{-1}(\mathbf{y}_i - X_i\hat{\boldsymbol{\beta}})$$

- "축소 추정량": 그룹 평균을 전체 평균 쪽으로 축소
- 표본 크기가 작은 그룹일수록 더 많이 축소 → 부분 풀링(Partial Pooling)
""",
    guided_code="""# 혼합 효과 모형 구현 가이드
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# --- 1. 계층적 데이터 생성 ---
np.random.seed(42)
n_groups = 20
n_per_group = 30
n_total = n_groups * n_per_group

# 랜덤 효과
b0 = np.random.normal(0, 2, n_groups)  # 랜덤 절편
b1 = np.random.normal(0, 0.5, n_groups)  # 랜덤 기울기

group = np.repeat(range(n_groups), n_per_group)
x = np.random.normal(0, 1, n_total)

# y = (5 + b0[group]) + (2 + b1[group]) * x + eps
y = (5 + b0[group]) + (2 + b1[group]) * x + np.random.normal(0, 1, n_total)

df = pd.DataFrame({'y': y, 'x': x, 'group': group})

# --- 2. 랜덤 절편 모형 ---
model_ri = smf.mixedlm("y ~ x", df, groups=df["group"])
result_ri = model_ri.fit(reml=True)
print("=== 랜덤 절편 모형 ===")
print(result_ri.summary())

# --- 3. 랜덤 절편 + 기울기 모형 ---
model_rs = smf.mixedlm("y ~ x", df, groups=df["group"],
                        re_formula="~x")
result_rs = model_rs.fit(reml=True)
print("\\n=== 랜덤 절편 + 기울기 모형 ===")
print(result_rs.summary())

# --- 4. ICC 계산 ---
# 영모형 (null model)
model_null = smf.mixedlm("y ~ 1", df, groups=df["group"])
result_null = model_null.fit(reml=True)

var_b = float(result_null.cov_re.iloc[0, 0])
var_e = result_null.scale
icc = var_b / (var_b + var_e)
print(f"\\n=== ICC ===")
print(f"σ²_b = {var_b:.4f}")
print(f"σ²_e = {var_e:.4f}")
print(f"ICC = {icc:.4f}")

# --- 5. 축소 추정 시각화 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 그룹별 OLS vs 혼합 모형
group_coefs_ols = []
for g in range(n_groups):
    mask = df['group'] == g
    X_g = sm.add_constant(df.loc[mask, 'x'])
    ols_g = sm.OLS(df.loc[mask, 'y'], X_g).fit()
    group_coefs_ols.append(ols_g.params.values)

group_coefs_ols = np.array(group_coefs_ols)
blup = result_rs.random_effects

# 절편
axes[0].scatter(range(n_groups), group_coefs_ols[:, 0], label='그룹별 OLS', alpha=0.7)
axes[0].scatter(range(n_groups),
               [result_rs.fe_params['Intercept'] + blup[g]['Group'] for g in range(n_groups)],
               label='BLUP (혼합)', marker='x', alpha=0.7)
axes[0].axhline(result_rs.fe_params['Intercept'], color='r', ls='--', label='고정 효과')
axes[0].set_xlabel('그룹'); axes[0].set_ylabel('절편')
axes[0].set_title('절편: OLS vs BLUP (축소 추정)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 기울기
axes[1].scatter(range(n_groups), group_coefs_ols[:, 1], label='그룹별 OLS', alpha=0.7)
axes[1].scatter(range(n_groups),
               [result_rs.fe_params['x'] + blup[g]['x'] for g in range(n_groups)],
               label='BLUP (혼합)', marker='x', alpha=0.7)
axes[1].axhline(result_rs.fe_params['x'], color='r', ls='--', label='고정 효과')
axes[1].set_xlabel('그룹'); axes[1].set_ylabel('기울기')
axes[1].set_title('기울기: OLS vs BLUP (축소 추정)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout(); plt.show()""",
    exercises=[
        {
            "difficulty": "★",
            "description": "주어진 계층적 데이터에서 ICC를 계산하고, 혼합 모형의 필요성을 판단하세요.",
            "hint": "영모형(y ~ 1, random=group)에서 분산 성분을 추출하세요.",
            "skeleton": "# 학생 성적 데이터 (학교 내 학생)\nnp.random.seed(42)\nn_schools = 15\nn_students = np.random.randint(20, 50, n_schools)\nschool_effect = np.random.normal(0, 3, n_schools)\n\ndata = []\nfor i in range(n_schools):\n    scores = 70 + school_effect[i] + np.random.normal(0, 5, n_students[i])\n    for s in scores:\n        data.append({'school': i, 'score': s})\ndf = pd.DataFrame(data)\n\n# TODO: ICC 계산\n# TODO: 혼합 모형 필요성 판단\n"
        },
        {
            "difficulty": "★★",
            "description": "REML과 ML 추정을 비교하고, 분산 성분 추정치의 차이를 확인하세요.\n\n두 방법의 고정 효과 추정치와 분산 성분 추정치를 비교 표로 정리하세요.",
            "skeleton": "# 동일 데이터에 REML vs ML 적합\n# TODO: 두 방법으로 적합\n# TODO: 고정 효과, 분산 성분 비교\n"
        },
        {
            "difficulty": "★★★",
            "description": "부분 풀링(Partial Pooling)의 효과를 시각화하세요.\n\n세 가지 추정 방식을 비교합니다:\n1. **완전 풀링**: 모든 그룹을 무시하고 하나의 회귀\n2. **비풀링**: 그룹별 독립 회귀\n3. **부분 풀링**: 혼합 효과 모형 (BLUP)",
            "hint": "표본 크기가 작은 그룹에서 축소 효과가 더 강하게 나타납니다.",
            "skeleton": "# 그룹별 표본 크기를 다르게 설정\nnp.random.seed(42)\nn_groups = 10\ngroup_sizes = [5, 5, 10, 10, 20, 20, 50, 50, 100, 100]\n\n# TODO: 데이터 생성 (그룹별 크기 다르게)\n# TODO: 완전 풀링, 비풀링, 부분 풀링 비교\n# TODO: 축소 효과 시각화\n"
        }
    ],
    references=[
        "Gelman & Hill (2007). Data Analysis Using Regression and Multilevel/Hierarchical Models.",
        "Bates et al. (2015). Fitting Linear Mixed-Effects Models Using lme4. JSS.",
        "statsmodels MixedLM: https://www.statsmodels.org/stable/mixed_linear.html",
    ],
    filepath=os.path.join(OUT, "ch05_05_mixed_effects_problems.ipynb")
)

solution_notebook(
    chapter_num=CH, section_num=5,
    title="혼합 효과 모형 (Mixed Effects Model)",
    solutions=[
        {
            "approach": "영모형에서 분산 성분을 추출하여 ICC를 계산합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n_schools = 15
n_students = np.random.randint(20, 50, n_schools)
school_effect = np.random.normal(0, 3, n_schools)

data = []
for i in range(n_schools):
    scores = 70 + school_effect[i] + np.random.normal(0, 5, n_students[i])
    for s in scores:
        data.append({'school': i, 'score': s})
df = pd.DataFrame(data)

# 영모형
model_null = smf.mixedlm("score ~ 1", df, groups=df["school"])
result_null = model_null.fit(reml=True)

var_b = float(result_null.cov_re.iloc[0, 0])  # 학교 간 분산
var_e = result_null.scale  # 학교 내 분산
icc = var_b / (var_b + var_e)

print(f"학교 간 분산 (σ²_b): {var_b:.4f}")
print(f"학교 내 분산 (σ²_e): {var_e:.4f}")
print(f"ICC = {icc:.4f}")
print(f"\\n해석: 전체 분산의 {icc*100:.1f}%가 학교 간 차이에 의해 설명됩니다.")
print(f"ICC > 0.05이므로 혼합 효과 모형이 필요합니다." if icc > 0.05
      else "ICC ≤ 0.05이므로 일반 회귀로 충분합니다.")""",
            "interpretation": "ICC가 높을수록 그룹 간 변동이 크며, 혼합 효과 모형이 필요합니다. 일반적으로 ICC > 0.05이면 혼합 모형을 고려합니다."
        },
        {
            "approach": "REML과 ML 추정을 비교하여 분산 성분 추정치의 차이를 확인합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n_groups = 20
n_per_group = 30
n_total = n_groups * n_per_group

b0 = np.random.normal(0, 2, n_groups)
group = np.repeat(range(n_groups), n_per_group)
x = np.random.normal(0, 1, n_total)
y = (5 + b0[group]) + 2*x + np.random.normal(0, 1, n_total)
df = pd.DataFrame({'y': y, 'x': x, 'group': group})

# REML
result_reml = smf.mixedlm("y ~ x", df, groups=df["group"]).fit(reml=True)
# ML
result_ml = smf.mixedlm("y ~ x", df, groups=df["group"]).fit(reml=False)

print(f"{'항목':<20} {'REML':<12} {'ML':<12}")
print("-" * 44)
print(f"{'Intercept':<20} {result_reml.fe_params['Intercept']:<12.4f} {result_ml.fe_params['Intercept']:<12.4f}")
print(f"{'x':<20} {result_reml.fe_params['x']:<12.4f} {result_ml.fe_params['x']:<12.4f}")
print(f"{'σ²_b (그룹)':<20} {float(result_reml.cov_re.iloc[0,0]):<12.4f} {float(result_ml.cov_re.iloc[0,0]):<12.4f}")
print(f"{'σ²_e (잔차)':<20} {result_reml.scale:<12.4f} {result_ml.scale:<12.4f}")
print(f"{'Log-Likelihood':<20} {result_reml.llf:<12.2f} {result_ml.llf:<12.2f}")

print(f"\\n참값: σ²_b = 4.0, σ²_e = 1.0")
print("→ REML이 분산 성분을 덜 편향되게 추정 (ML은 과소추정 경향)")""",
            "interpretation": "REML은 고정 효과의 자유도를 보정하여 분산 성분을 비편향적으로 추정합니다. ML은 분산을 과소추정하는 경향이 있습니다. 고정 효과 추정치는 거의 동일합니다."
        },
        {
            "approach": "표본 크기가 다른 그룹에서 완전 풀링, 비풀링, 부분 풀링을 비교합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

np.random.seed(42)
n_groups = 10
group_sizes = [5, 5, 10, 10, 20, 20, 50, 50, 100, 100]
true_intercepts = np.random.normal(5, 2, n_groups)

data = []
for i, (sz, b0) in enumerate(zip(group_sizes, true_intercepts)):
    x = np.random.normal(0, 1, sz)
    y = b0 + 2*x + np.random.normal(0, 1, sz)
    for j in range(sz):
        data.append({'group': i, 'x': x[j], 'y': y[j], 'size': sz})
df = pd.DataFrame(data)

# 1) 완전 풀링
pooled = sm.OLS.from_formula("y ~ x", df).fit()
pooled_int = pooled.params['Intercept']

# 2) 비풀링 (그룹별 OLS)
no_pool_int = []
for g in range(n_groups):
    sub = df[df['group'] == g]
    ols_g = sm.OLS.from_formula("y ~ x", sub).fit()
    no_pool_int.append(ols_g.params['Intercept'])

# 3) 부분 풀링 (혼합 효과)
mixed = smf.mixedlm("y ~ x", df, groups=df["group"]).fit(reml=True)
blup_int = [mixed.fe_params['Intercept'] + mixed.random_effects[g]['Group']
            for g in range(n_groups)]

# 시각화
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(n_groups)
width = 0.2

ax.bar(x_pos - width, true_intercepts, width, label='참값', alpha=0.8)
ax.bar(x_pos, no_pool_int, width, label='비풀링 (OLS)', alpha=0.8)
ax.bar(x_pos + width, blup_int, width, label='부분 풀링 (BLUP)', alpha=0.8)
ax.axhline(pooled_int, color='red', ls='--', label=f'완전 풀링 ({pooled_int:.2f})')

ax.set_xticks(x_pos)
ax.set_xticklabels([f'G{i}\\n(n={s})' for i, s in enumerate(group_sizes)])
ax.set_ylabel('절편'); ax.set_title('풀링 전략 비교')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

# 축소 정도
print(f"{'그룹':<8} {'n':<6} {'참값':<8} {'비풀링':<8} {'부분풀링':<8} {'축소율':<8}")
for g in range(n_groups):
    shrink = 1 - abs(blup_int[g] - pooled_int) / (abs(no_pool_int[g] - pooled_int) + 1e-8)
    print(f"G{g:<7d} {group_sizes[g]:<6d} {true_intercepts[g]:<8.2f} "
          f"{no_pool_int[g]:<8.2f} {blup_int[g]:<8.2f} {shrink:<8.2f}")""",
            "interpretation": "표본 크기가 작은 그룹(n=5)에서 BLUP는 전체 평균 쪽으로 강하게 축소됩니다(부분 풀링). 표본 크기가 큰 그룹(n=100)은 그룹 내 정보가 충분하여 거의 축소되지 않습니다."
        }
    ],
    discussion="""
### 혼합 효과 모형 실무 가이드

1. **랜덤 효과 구조 선택**: 이론적 근거 기반, 수렴 고려
2. **ICC 확인**: 혼합 모형 필요성 판단
3. **REML vs ML**: 분산 비교 → REML, 고정효과 비교 → ML (또는 동일 랜덤 구조에서)
4. **모형 비교**: AIC/BIC (ML 기반), 우도비 검정

### ADP 시험 포인트
- 고정 효과 vs 랜덤 효과의 구분 기준
- ICC의 공식과 해석
- REML의 편향 보정 원리
- 부분 풀링(BLUP)의 축소 효과
""",
    filepath=os.path.join(OUT, "ch05_05_mixed_effects_solutions.ipynb")
)

print("Part 1 (Topics 1-5) generated successfully!")
