"""Ch05 Advanced Regression - Topics 6-10 generator"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.notebook_generator import problem_notebook, solution_notebook

OUT = os.path.dirname(__file__)
CH = 5

# ============================================================
# Topic 6: IV / 2SLS
# ============================================================
problem_notebook(
    chapter_num=CH, section_num=6,
    title="도구 변수와 2SLS",
    objectives=[
        "내생성(Endogeneity) 문제의 원인과 식별",
        "도구 변수(IV)의 조건: 관련성과 외생성",
        "2SLS(Two-Stage Least Squares) 추정량의 도출",
        "약한 IV 검정과 과대식별 검정(Sargan test)",
    ],
    theory_md=r"""
### 1. 내생성 문제

$$y = X\boldsymbol{\beta} + \varepsilon, \quad E[X^\top\varepsilon] \neq 0$$

원인: 누락 변수, 동시성, 측정 오차

→ OLS 추정량이 **일치성(consistency)**을 잃음

### 2. 도구 변수 (Instrumental Variable)

도구 변수 $Z$의 조건:
1. **관련성**: $\text{Cov}(Z, X) \neq 0$ → 1단계에서 $X$를 잘 설명
2. **외생성**: $\text{Cov}(Z, \varepsilon) = 0$ → 결과에 직접 영향 없음

### 3. 2SLS 추정

**1단계**: $\hat{X} = Z(Z^\top Z)^{-1}Z^\top X = P_Z X$

**2단계**: $\hat{\boldsymbol{\beta}}_{2SLS} = (\hat{X}^\top \hat{X})^{-1}\hat{X}^\top y$

행렬 형태:
$$\hat{\boldsymbol{\beta}}_{2SLS} = (X^\top P_Z X)^{-1} X^\top P_Z y$$

### 4. Wald 추정량 (단순 IV)

도구변수와 내생변수가 각각 1개일 때:

$$\hat{\beta}_{IV} = \frac{\text{Cov}(Z, Y)}{\text{Cov}(Z, X)} = \frac{\sum(z_i - \bar{z})(y_i - \bar{y})}{\sum(z_i - \bar{z})(x_i - \bar{x})}$$

### 5. 진단 검정

**약한 IV 검정**: 1단계 F-통계량 > 10 (Stock & Yogo 기준)

**Hausman 검정**: OLS와 2SLS의 차이가 유의한지

$$H = (\hat{\boldsymbol{\beta}}_{2SLS} - \hat{\boldsymbol{\beta}}_{OLS})^\top [\hat{V}_{2SLS} - \hat{V}_{OLS}]^{-1} (\hat{\boldsymbol{\beta}}_{2SLS} - \hat{\boldsymbol{\beta}}_{OLS}) \sim \chi^2_p$$

**Sargan 과대식별 검정** (도구 수 > 내생변수 수):
$$nR^2 \sim \chi^2_{q-p}$ (여기서 $q$는 도구 수, $p$는 내생변수 수)
""",
    guided_code="""# 도구 변수 / 2SLS 구현 가이드
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt

# --- 1. 내생성 있는 데이터 생성 ---
np.random.seed(42)
n = 1000

# 누락 변수 (ability)
ability = np.random.normal(0, 1, n)

# 도구 변수: 부모 교육 수준 (education과 관련, wage에 직접 영향 없음)
parent_edu = np.random.normal(12, 3, n)

# 내생 변수: 교육 연수
education = 8 + 0.4*parent_edu + 0.5*ability + np.random.normal(0, 1, n)

# 결과 변수: 임금 (log)
log_wage = 1 + 0.1*education + 0.3*ability + np.random.normal(0, 0.5, n)

df = pd.DataFrame({
    'log_wage': log_wage,
    'education': education,
    'parent_edu': parent_edu,
    'ability': ability  # 실제로는 관측 불가
})

# --- 2. OLS (편향) ---
X_ols = sm.add_constant(df['education'])
ols = sm.OLS(df['log_wage'], X_ols).fit()
print(f"OLS 교육 계수: {ols.params['education']:.4f} (참값: 0.1)")
print(f"→ 누락변수(ability) 때문에 상향 편향")

# --- 3. 2SLS ---
# linearmodels 사용
iv_model = IV2SLS.from_formula(
    "log_wage ~ 1 + [education ~ parent_edu]", data=df
)
iv_result = iv_model.fit(cov_type='unadjusted')
print(f"\\n2SLS 교육 계수: {iv_result.params['education']:.4f}")
print(iv_result.summary)

# --- 4. 수동 2SLS ---
Z = sm.add_constant(df['parent_edu'])
X = sm.add_constant(df['education'])
y = df['log_wage']

# 1단계: education ~ parent_edu
stage1 = sm.OLS(df['education'], Z).fit()
print(f"\\n1단계 F-stat: {stage1.fvalue:.2f} (>10이면 강한 IV)")
edu_hat = stage1.fittedvalues

# 2단계: log_wage ~ edu_hat
X_hat = sm.add_constant(edu_hat)
stage2 = sm.OLS(y, X_hat).fit()
print(f"수동 2SLS 교육 계수: {stage2.params.iloc[1]:.4f}")

# --- 5. 시각화 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df['education'], df['log_wage'], alpha=0.1, s=5)
x_range = np.linspace(df['education'].min(), df['education'].max(), 100)
axes[0].plot(x_range, ols.params['const'] + ols.params['education']*x_range,
            'r-', label=f'OLS (β={ols.params["education"]:.3f})', lw=2)
axes[0].plot(x_range, iv_result.params['Intercept'] + iv_result.params['education']*x_range,
            'b--', label=f'2SLS (β={iv_result.params["education"]:.3f})', lw=2)
axes[0].set_xlabel('교육 연수'); axes[0].set_ylabel('log(임금)')
axes[0].legend(); axes[0].set_title('OLS vs 2SLS')

# 1단계 관계
axes[1].scatter(df['parent_edu'], df['education'], alpha=0.1, s=5)
axes[1].plot(df['parent_edu'], stage1.fittedvalues, 'r-', lw=2)
axes[1].set_xlabel('부모 교육 수준'); axes[1].set_ylabel('교육 연수')
axes[1].set_title(f'1단계: F={stage1.fvalue:.1f}')

plt.tight_layout(); plt.show()""",
    exercises=[
        {
            "difficulty": "★",
            "description": "주어진 데이터에서 OLS와 2SLS 추정량을 비교하고, 내생성 편향의 방향과 크기를 분석하세요.",
            "hint": "linearmodels의 IV2SLS를 사용하세요.",
            "skeleton": "# 공급-수요 동시성 예제\nnp.random.seed(42)\nn = 500\ndemand_shock = np.random.normal(0, 1, n)\nsupply_shock = np.random.normal(0, 1, n)\nweather = np.random.normal(0, 1, n)  # 도구변수\n\nprice = 10 + supply_shock - 0.5*weather + demand_shock\nquantity = 20 - 2*price + demand_shock + np.random.normal(0, 0.5, n)\n\ndf = pd.DataFrame({'price': price, 'quantity': quantity, 'weather': weather})\n\n# TODO: OLS vs 2SLS 비교\n"
        },
        {
            "difficulty": "★★",
            "description": "2SLS를 행렬 연산으로 직접 구현하고, 표준오차를 올바르게 계산하세요.\n\n주의: 2단계의 표준오차는 $\\hat{X}$가 아닌 원래 $X$를 사용하여 계산해야 합니다.",
            "skeleton": "def two_stage_ls(y, X, Z):\n    \"\"\"2SLS 수동 구현\n    y: 반응변수\n    X: 내생변수 포함 설계행렬\n    Z: 도구변수 행렬\n    \"\"\"\n    # TODO: 1단계 (X를 Z에 회귀)\n    # TODO: 2단계 (y를 X_hat에 회귀)\n    # TODO: 올바른 표준오차 계산\n    pass\n"
        },
        {
            "difficulty": "★★★",
            "description": "약한 IV 검정과 Sargan 과대식별 검정을 수동으로 구현하세요.\n\n도구 변수가 2개인 경우를 다루세요.",
            "hint": "Sargan: 2SLS 잔차를 모든 IV에 회귀하여 nR²를 계산합니다.",
            "skeleton": "# 과대식별 (IV 2개, 내생변수 1개)\nnp.random.seed(42)\nn = 500\nz1 = np.random.normal(0, 1, n)\nz2 = np.random.normal(0, 1, n)\nu = np.random.normal(0, 1, n)\nx = 1 + 0.5*z1 + 0.3*z2 + 0.8*u\ny = 2 + 3*x + u + np.random.normal(0, 0.5, n)\n\n# TODO: 약한 IV 검정 (1단계 F통계량)\n# TODO: 2SLS 추정\n# TODO: Sargan 검정\n"
        }
    ],
    references=[
        "Angrist & Pischke (2009). Mostly Harmless Econometrics, Ch.4.",
        "Wooldridge (2010). Econometric Analysis of Cross Section and Panel Data, Ch.5.",
        "linearmodels documentation: https://bashtage.github.io/linearmodels/",
    ],
    filepath=os.path.join(OUT, "ch05_06_iv_2sls_problems.ipynb")
)

solution_notebook(
    chapter_num=CH, section_num=6,
    title="도구 변수와 2SLS",
    solutions=[
        {
            "approach": "공급-수요 동시성 문제에서 OLS 편향을 확인하고 2SLS로 교정합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

np.random.seed(42)
n = 500
demand_shock = np.random.normal(0, 1, n)
supply_shock = np.random.normal(0, 1, n)
weather = np.random.normal(0, 1, n)

price = 10 + supply_shock - 0.5*weather + demand_shock
quantity = 20 - 2*price + demand_shock + np.random.normal(0, 0.5, n)

df = pd.DataFrame({'price': price, 'quantity': quantity, 'weather': weather})

# OLS (편향)
ols = sm.OLS(df['quantity'], sm.add_constant(df['price'])).fit()
print(f"OLS 가격 계수: {ols.params.iloc[1]:.4f} (참값: -2.0)")

# 2SLS
iv = IV2SLS.from_formula("quantity ~ 1 + [price ~ weather]", data=df)
iv_result = iv.fit(cov_type='unadjusted')
print(f"2SLS 가격 계수: {iv_result.params['price']:.4f}")

# 1단계 진단
stage1 = sm.OLS(df['price'], sm.add_constant(df['weather'])).fit()
print(f"\\n1단계 F-stat: {stage1.fvalue:.2f}")
print(f"편향 = OLS - 참값 = {ols.params.iloc[1] - (-2):.4f}")
print("→ demand_shock이 price와 quantity에 동시 영향 → OLS 상향 편향")""",
            "interpretation": "수요 충격이 가격과 수량 모두에 영향을 미치므로 OLS는 가격 계수를 과소추정(절대값 기준)합니다. weather는 공급에만 영향하는 외생 도구이므로 2SLS가 일치적 추정량을 제공합니다."
        },
        {
            "approach": "2SLS를 행렬 연산으로 직접 구현하고 올바른 표준오차를 계산합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

def two_stage_ls(y, X, Z):
    n = len(y)
    # 1단계: X_hat = P_Z X
    P_Z = Z @ np.linalg.solve(Z.T @ Z, Z.T)
    X_hat = P_Z @ X

    # 2단계: beta = (X_hat' X_hat)^{-1} X_hat' y
    beta = np.linalg.solve(X_hat.T @ X_hat, X_hat.T @ y)

    # 표준오차: 원래 X 사용!
    residuals = y - X @ beta  # 원래 X로 잔차
    sigma2 = np.sum(residuals**2) / (n - X.shape[1])

    # Var(beta) = sigma^2 (X_hat' X_hat)^{-1}
    var_beta = sigma2 * np.linalg.inv(X_hat.T @ X_hat)
    se = np.sqrt(np.diag(var_beta))

    return beta, se, residuals

# 데이터
np.random.seed(42)
n = 1000
ability = np.random.normal(0, 1, n)
parent_edu = np.random.normal(12, 3, n)
education = 8 + 0.4*parent_edu + 0.5*ability + np.random.normal(0, 1, n)
log_wage = 1 + 0.1*education + 0.3*ability + np.random.normal(0, 0.5, n)

y = log_wage
X = np.column_stack([np.ones(n), education])
Z = np.column_stack([np.ones(n), parent_edu])

beta_manual, se_manual, _ = two_stage_ls(y, X, Z)

# linearmodels 비교
df = pd.DataFrame({'log_wage': log_wage, 'education': education, 'parent_edu': parent_edu})
iv = IV2SLS.from_formula("log_wage ~ 1 + [education ~ parent_edu]", data=df)
iv_result = iv.fit(cov_type='unadjusted')

print(f"{'':>15} {'수동 β':<12} {'LM β':<12} {'수동 SE':<12} {'LM SE':<12}")
print(f"{'Intercept':>15} {beta_manual[0]:<12.6f} {iv_result.params['Intercept']:<12.6f} "
      f"{se_manual[0]:<12.6f} {iv_result.std_errors['Intercept']:<12.6f}")
print(f"{'education':>15} {beta_manual[1]:<12.6f} {iv_result.params['education']:<12.6f} "
      f"{se_manual[1]:<12.6f} {iv_result.std_errors['education']:<12.6f}")""",
            "interpretation": "2SLS의 표준오차를 계산할 때 반드시 원래 X로 잔차를 계산해야 합니다. X_hat으로 잔차를 구하면 분산이 과소추정됩니다."
        },
        {
            "approach": "1단계 F검정으로 약한 IV를 진단하고, Sargan 검정으로 과대식별 제약의 타당성을 검정합니다.",
            "code": """import numpy as np
import statsmodels.api as sm
from scipy import stats

np.random.seed(42)
n = 500
z1 = np.random.normal(0, 1, n)
z2 = np.random.normal(0, 1, n)
u = np.random.normal(0, 1, n)
x = 1 + 0.5*z1 + 0.3*z2 + 0.8*u
y = 2 + 3*x + u + np.random.normal(0, 0.5, n)

# 1) 약한 IV 검정
Z = sm.add_constant(np.column_stack([z1, z2]))
stage1 = sm.OLS(x, Z).fit()
print("=== 약한 IV 검정 ===")
print(f"1단계 F-stat: {stage1.fvalue:.4f}")
print(f"p-value: {stage1.f_pvalue:.6f}")
print(f"판정: {'강한 IV (F > 10)' if stage1.fvalue > 10 else '약한 IV (F ≤ 10)'}")

# 2) 2SLS
X_full = sm.add_constant(x)
P_Z = Z @ np.linalg.solve(Z.T @ Z, Z.T)
X_hat = P_Z @ X_full
beta_2sls = np.linalg.solve(X_hat.T @ X_hat, X_hat.T @ y)
resid_2sls = y - X_full @ beta_2sls
print(f"\\n2SLS 계수: intercept={beta_2sls[0]:.4f}, x={beta_2sls[1]:.4f} (참값: 2, 3)")

# 3) Sargan 과대식별 검정
# 2SLS 잔차를 모든 IV에 회귀
sargan_reg = sm.OLS(resid_2sls, Z).fit()
sargan_stat = n * sargan_reg.rsquared
df_sargan = Z.shape[1] - X_full.shape[1]  # q - p (도구수 - 내생변수수)
sargan_pval = 1 - stats.chi2.cdf(sargan_stat, df_sargan)

print(f"\\n=== Sargan 과대식별 검정 ===")
print(f"검정통계량: nR² = {sargan_stat:.4f}")
print(f"자유도: {df_sargan}")
print(f"p-value: {sargan_pval:.4f}")
print(f"판정: {'도구 유효 (귀무 기각 못함)' if sargan_pval > 0.05 else '도구 타당성 의심'}")

# 4) Hausman 검정
ols = sm.OLS(y, X_full).fit()
diff = beta_2sls - ols.params.values
sigma2_2sls = np.sum(resid_2sls**2) / (n - 2)
V_2sls = sigma2_2sls * np.linalg.inv(X_hat.T @ X_hat)
V_ols = np.array(ols.cov_params())
V_diff = V_2sls - V_ols

try:
    hausman_stat = diff @ np.linalg.solve(V_diff, diff)
    hausman_pval = 1 - stats.chi2.cdf(hausman_stat, 1)
    print(f"\\n=== Hausman 검정 ===")
    print(f"검정통계량: {hausman_stat:.4f}")
    print(f"p-value: {hausman_pval:.4f}")
    print(f"판정: {'내생성 존재 (OLS 편향)' if hausman_pval < 0.05 else '내생성 없음'}")
except:
    print("\\nHausman 검정: 분산 차이 행렬이 양정치가 아님")""",
            "interpretation": "1단계 F > 10이면 도구가 충분히 강합니다. Sargan 검정의 귀무가설은 '모든 IV가 외생적'이므로, 기각하지 못하면 IV의 타당성을 지지합니다. Hausman 검정으로 내생성 존재를 확인합니다."
        }
    ],
    discussion="""
### IV/2SLS 실무 가이드

1. **좋은 도구 변수 찾기**: 이론적 정당화가 필수 (데이터만으로는 외생성 검증 불가)
2. **약한 IV 문제**: F < 10이면 2SLS가 OLS보다 나쁠 수 있음
3. **과대식별**: 도구가 많을수록 유한 표본 편향 증가
4. **강건한 표준오차**: 이분산성 하에서는 HAC 표준오차 사용

### ADP 시험 포인트
- 내생성의 세 가지 원인 (누락변수, 동시성, 측정오차)
- IV의 두 조건 (관련성, 외생성)
- 2SLS의 2단계 절차
- Sargan 검정의 귀무가설과 해석
""",
    filepath=os.path.join(OUT, "ch05_06_iv_2sls_solutions.ipynb")
)

# ============================================================
# Topic 7: Advanced Logistic
# ============================================================
problem_notebook(
    chapter_num=CH, section_num=7,
    title="로지스틱 회귀 심화",
    objectives=[
        "다범주 로지스틱 회귀 (Multinomial / Ordinal)",
        "완전 분리(Complete Separation) 문제의 진단과 처리",
        "Firth 편향 보정 로지스틱 회귀",
        "ROC/AUC 분석과 최적 절단점 선택",
    ],
    theory_md=r"""
### 1. 다범주 로지스틱 회귀 (Multinomial Logit)

$K$개 범주, 기준 범주 $K$에 대해:

$$\log\frac{P(Y=k|X)}{P(Y=K|X)} = \mathbf{x}^\top\boldsymbol{\beta}_k, \quad k=1,\ldots,K-1$$

확률:
$$P(Y=k|X) = \frac{\exp(\mathbf{x}^\top\boldsymbol{\beta}_k)}{1 + \sum_{j=1}^{K-1}\exp(\mathbf{x}^\top\boldsymbol{\beta}_j)}$$

### 2. 순서형 로지스틱 회귀 (Ordinal Logit)

비례 오즈 모형 (Proportional Odds):

$$\text{logit}[P(Y \le k)] = \alpha_k - \mathbf{x}^\top\boldsymbol{\beta}$$

- $\alpha_1 < \alpha_2 < \cdots < \alpha_{K-1}$: 절단점 (cut points)
- $\boldsymbol{\beta}$: 모든 범주에 공통 (비례 오즈 가정)

### 3. 완전 분리 (Complete Separation)

선형 조합 $\mathbf{x}^\top\boldsymbol{\beta}$로 $Y=0$과 $Y=1$을 완벽히 분류 가능할 때:
- MLE가 존재하지 않음 ($\hat{\beta} \to \pm\infty$)
- 표준오차가 매우 커짐
- 수렴하지 않거나 경고 발생

### 4. Firth 보정 (Penalized Likelihood)

Jeffreys 사전분포를 이용한 페널티:

$$\ell^*(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) + \frac{1}{2}\log|I(\boldsymbol{\beta})|$$

여기서 $I(\boldsymbol{\beta})$는 Fisher 정보행렬

효과:
- 유한한 추정치 보장 (완전 분리 해결)
- 소표본에서 편향 감소
- 일반적으로 계수가 0 쪽으로 축소

### 5. 모형 평가

**ROC 곡선**: 모든 절단점에서 (FPR, TPR) 쌍

$$\text{AUC} = P(\hat{p}_1 > \hat{p}_0)$$

**최적 절단점**:
- Youden's J: $\max(TPR - FPR)$
- 비용 기반: $\min(c_{FP}\cdot FPR + c_{FN}\cdot FNR)$
""",
    guided_code="""# 로지스틱 회귀 심화 가이드
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# --- 1. 다범주 로지스틱 ---
np.random.seed(42)
n = 500
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)

# 3범주 생성
logit1 = 0.5 + 1.0*x1 - 0.5*x2
logit2 = -0.3 + 0.2*x1 + 1.0*x2
p1 = np.exp(logit1) / (1 + np.exp(logit1) + np.exp(logit2))
p2 = np.exp(logit2) / (1 + np.exp(logit1) + np.exp(logit2))
p3 = 1 - p1 - p2

y_multi = np.array([np.random.choice([0,1,2], p=[p1[i], p2[i], p3[i]]) for i in range(n)])

X = np.column_stack([x1, x2])
multi_lr = LogisticRegression(multi_class='multinomial', max_iter=1000)
multi_lr.fit(X, y_multi)
print("다범주 로지스틱 계수:")
print(pd.DataFrame(multi_lr.coef_, columns=['x1', 'x2'],
                    index=[f'class {i}' for i in range(3)]))

# --- 2. 완전 분리 시연 ---
np.random.seed(42)
x_sep = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y_sep = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

X_sep = sm.add_constant(x_sep)
try:
    model_sep = sm.Logit(y_sep, X_sep)
    result_sep = model_sep.fit(maxiter=100, disp=0)
    print(f"\\n완전 분리 - 계수: {result_sep.params[1]:.2f}, SE: {result_sep.bse[1]:.2f}")
    print("→ 계수가 매우 크고 SE도 매우 큼!")
except Exception as e:
    print(f"완전 분리 오류: {e}")

# --- 3. ROC / AUC ---
np.random.seed(42)
n = 1000
X_roc = np.random.randn(n, 3)
p_true = 1 / (1 + np.exp(-(0.5 + X_roc @ [1, -0.5, 0.3])))
y_roc = np.random.binomial(1, p_true)

lr = LogisticRegression()
lr.fit(X_roc, y_roc)
y_prob = lr.predict_proba(X_roc)[:, 1]

fpr, tpr, thresholds = roc_curve(y_roc, y_prob)
roc_auc = auc(fpr, tpr)

# Youden's J
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC={roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.scatter(fpr[best_idx], tpr[best_idx], c='red', s=100, zorder=5,
           label=f'최적 절단점={best_threshold:.3f}')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC 곡선'); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

print(f"\\n최적 절단점 (Youden): {best_threshold:.4f}")
y_pred_opt = (y_prob >= best_threshold).astype(int)
print(classification_report(y_roc, y_pred_opt, target_names=['Class 0', 'Class 1']))""",
    exercises=[
        {
            "difficulty": "★",
            "description": "3범주 데이터에 다범주 로지스틱 회귀를 적합하고, 각 범주에 대한 확률 맵을 시각화하세요.",
            "hint": "predict_proba로 각 범주 확률을 얻고 contourf로 시각화하세요.",
            "skeleton": "from sklearn.linear_model import LogisticRegression\n\nnp.random.seed(42)\n# 3범주 데이터 생성\nfrom sklearn.datasets import make_classification\nX, y = make_classification(n_samples=300, n_features=2, n_informative=2,\n                           n_redundant=0, n_classes=3, n_clusters_per_class=1,\n                           random_state=42)\n\n# TODO: 다범주 로지스틱 적합\n# TODO: 확률 맵 시각화 (2D)\n"
        },
        {
            "difficulty": "★★",
            "description": "완전 분리 상황을 만들고, Firth 보정 로지스틱 회귀로 해결하세요.\n\n`firthlogist` 패키지 또는 수동 구현을 사용하세요.",
            "skeleton": "# 완전 분리 데이터\nnp.random.seed(42)\nn = 20\nx = np.sort(np.random.normal(0, 1, n))\ny = (x > 0).astype(int)  # 완전 분리\n\n# TODO: 일반 MLE 시도 (경고 확인)\n# TODO: Firth 보정 (L2 페널티로 근사 가능)\n# TODO: 계수 비교\n"
        },
        {
            "difficulty": "★★★",
            "description": "ROC 곡선에서 다양한 최적 절단점 기준(Youden, 비용 기반, F1 최대화)을 비교하세요.",
            "skeleton": "# 불균형 이진 분류\nnp.random.seed(42)\nn = 1000\nX = np.random.randn(n, 3)\np = 1 / (1 + np.exp(-(X @ [1, -0.5, 0.3] - 1.5)))  # 불균형\ny = np.random.binomial(1, p)\nprint(f'양성 비율: {y.mean():.3f}')\n\n# TODO: ROC 곡선\n# TODO: Youden, 비용 기반, F1 기준 최적 절단점 비교\n"
        }
    ],
    references=[
        "Agresti (2013). Categorical Data Analysis, 3rd ed.",
        "Heinze & Schemper (2002). A solution to the problem of separation in logistic regression. Statistics in Medicine.",
        "Firth (1993). Bias reduction of maximum likelihood estimates. Biometrika.",
    ],
    filepath=os.path.join(OUT, "ch05_07_logistic_advanced_problems.ipynb")
)

solution_notebook(
    chapter_num=CH, section_num=7,
    title="로지스틱 회귀 심화",
    solutions=[
        {
            "approach": "다범주 로지스틱 회귀의 결정 경계와 확률 맵을 시각화합니다.",
            "code": """import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

np.random.seed(42)
X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=3, n_clusters_per_class=1,
                           random_state=42)

model = LogisticRegression(multi_class='multinomial', max_iter=1000)
model.fit(X, y)

# 확률 맵
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200),
                      np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 200))
Z_prob = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z_pred = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for k in range(3):
    ax = axes[k]
    prob_k = Z_prob[:, k].reshape(xx.shape)
    c = ax.contourf(xx, yy, prob_k, levels=20, cmap='RdYlBu_r', alpha=0.7)
    plt.colorbar(c, ax=ax)
    for cls in range(3):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], s=15, alpha=0.5, label=f'Class {cls}')
    ax.set_title(f'P(Y={k}|X)')
    ax.legend(fontsize=8)

plt.suptitle('다범주 로지스틱 회귀 확률 맵')
plt.tight_layout(); plt.show()

print(f"정확도: {model.score(X, y):.4f}")""",
            "interpretation": "각 클래스의 확률 맵은 공간의 각 위치에서 해당 클래스에 속할 확률을 보여줍니다. 세 확률의 합은 항상 1이며, 결정 경계는 확률이 같은 등고선입니다."
        },
        {
            "approach": "완전 분리 문제를 L2 페널티(Firth 근사)로 해결합니다.",
            "code": """import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

np.random.seed(42)
n = 20
x = np.sort(np.random.normal(0, 1, n))
y = (x > 0).astype(int)

X = sm.add_constant(x)

# 1) 일반 MLE
try:
    mle = sm.Logit(y, X).fit(disp=0, maxiter=100)
    print(f"MLE: β0={mle.params[0]:.2f}, β1={mle.params[1]:.2f}")
    print(f"SE:  {mle.bse[0]:.2f}, {mle.bse[1]:.2f}")
    print(f"→ 계수와 SE가 매우 큼 (발산 중)")
except:
    print("MLE 수렴 실패")

# 2) L2 페널티 (Firth 근사) - 다양한 C값
C_values = [100, 10, 1, 0.1]
print(f"\\n{'C':<8} {'β0':<10} {'β1':<10}")
for C in C_values:
    lr = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=1000)
    lr.fit(x.reshape(-1, 1), y)
    print(f"{C:<8} {lr.intercept_[0]:<10.4f} {lr.coef_[0, 0]:<10.4f}")

# 시각화
x_plot = np.linspace(-3, 3, 200)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, s=50, zorder=5, label='데이터')

for C, ls in zip([100, 1, 0.1], ['-', '--', ':']):
    lr = LogisticRegression(penalty='l2', C=C, max_iter=1000)
    lr.fit(x.reshape(-1, 1), y)
    p_hat = lr.predict_proba(x_plot.reshape(-1, 1))[:, 1]
    ax.plot(x_plot, p_hat, ls, label=f'C={C}', lw=2)

ax.set_xlabel('x'); ax.set_ylabel('P(Y=1)')
ax.set_title('완전 분리: 페널티 효과')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()""",
            "interpretation": "C가 클수록(페널티 약함) 시그모이드가 급격해지며 완전 분리 문제가 나타납니다. C를 줄이면(페널티 강화) 유한한 계수를 얻을 수 있으며, 이는 Firth 보정의 효과와 유사합니다."
        },
        {
            "approach": "불균형 데이터에서 Youden, 비용 기반, F1 기준 최적 절단점을 비교합니다.",
            "code": """import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000
X = np.random.randn(n, 3)
p = 1 / (1 + np.exp(-(X @ [1, -0.5, 0.3] - 1.5)))
y = np.random.binomial(1, p)
print(f'양성 비율: {y.mean():.3f}')

lr = LogisticRegression(max_iter=1000)
lr.fit(X, y)
y_prob = lr.predict_proba(X)[:, 1]

fpr, tpr, thresholds_roc = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

# 1) Youden's J
j_scores = tpr - fpr
youden_idx = np.argmax(j_scores)
youden_thresh = thresholds_roc[youden_idx]

# 2) 비용 기반 (FN이 FP보다 3배 비싸다고 가정)
cost_fn, cost_fp = 3, 1
costs = cost_fp * fpr + cost_fn * (1 - tpr)
cost_idx = np.argmin(costs)
cost_thresh = thresholds_roc[cost_idx]

# 3) F1 최대화
f1_scores = []
thresh_list = np.linspace(0.01, 0.99, 200)
for t in thresh_list:
    y_pred_t = (y_prob >= t).astype(int)
    f1_scores.append(f1_score(y, y_pred_t, zero_division=0))
f1_idx = np.argmax(f1_scores)
f1_thresh = thresh_list[f1_idx]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ROC
axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'AUC={roc_auc:.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].scatter(fpr[youden_idx], tpr[youden_idx], c='red', s=100, zorder=5,
               label=f'Youden={youden_thresh:.3f}')
axes[0].scatter(fpr[cost_idx], tpr[cost_idx], c='green', s=100, zorder=5,
               marker='s', label=f'Cost={cost_thresh:.3f}')
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('ROC 곡선'); axes[0].legend()

# F1
axes[1].plot(thresh_list, f1_scores, 'b-')
axes[1].axvline(f1_thresh, color='r', ls='--', label=f'최적={f1_thresh:.3f}')
axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('F1 Score')
axes[1].set_title('F1 vs 절단점'); axes[1].legend()

# 비교 요약
names = ['Youden', 'Cost-based', 'F1-max', 'Default(0.5)']
threshs = [youden_thresh, cost_thresh, f1_thresh, 0.5]
for i, (name, t) in enumerate(zip(names, threshs)):
    y_pred = (y_prob >= t).astype(int)
    tp = np.sum((y_pred == 1) & (y == 1))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    axes[2].bar(i, f1, label=f'{name}({t:.3f})')

axes[2].set_xticks(range(4)); axes[2].set_xticklabels(names, rotation=45)
axes[2].set_ylabel('F1 Score'); axes[2].set_title('절단점별 F1')

plt.tight_layout(); plt.show()

print(f"\\n{'기준':<15} {'절단점':<10} {'Sensitivity':<12} {'Specificity':<12}")
for name, t in zip(names, threshs):
    y_pred = (y_prob >= t).astype(int)
    sens = np.sum((y_pred == 1) & (y == 1)) / np.sum(y == 1)
    spec = np.sum((y_pred == 0) & (y == 0)) / np.sum(y == 0)
    print(f"{name:<15} {t:<10.4f} {sens:<12.4f} {spec:<12.4f}")""",
            "interpretation": "비용 기반 절단점은 FN 비용이 높을 때 더 낮은 절단점(높은 민감도)을 선택합니다. 불균형 데이터에서는 기본 0.5보다 F1 또는 Youden 기준이 더 적절합니다."
        }
    ],
    discussion="""
### 로지스틱 회귀 심화 요점

1. **다범주**: Multinomial(명목) vs Ordinal(순서) 선택
2. **완전 분리**: 경고 메시지 확인, Firth 또는 페널티 사용
3. **절단점**: 목적에 따라 다른 기준 적용 (진단 → 민감도, 선별 → 특이도)

### ADP 시험 포인트
- Multinomial logit의 확률 공식
- 완전 분리의 정의와 해결법
- ROC/AUC의 해석 (AUC = 양성이 음성보다 높은 확률을 가질 확률)
""",
    filepath=os.path.join(OUT, "ch05_07_logistic_advanced_solutions.ipynb")
)

# ============================================================
# Topic 8: Survival Regression
# ============================================================
problem_notebook(
    chapter_num=CH, section_num=8,
    title="생존 회귀 (Survival Regression)",
    objectives=[
        "AFT(Accelerated Failure Time) 모형의 구조와 해석",
        "Cox PH(Proportional Hazards) 모형의 부분우도",
        "Concordance Index(C-index)의 정의와 계산",
        "비례 위험 가정의 진단",
    ],
    theory_md=r"""
### 1. 생존 분석 기본 개념

**생존 함수**: $S(t) = P(T > t) = 1 - F(t)$

**위험 함수**: $h(t) = \lim_{\Delta t \to 0}\frac{P(t \le T < t+\Delta t|T \ge t)}{\Delta t} = \frac{f(t)}{S(t)}$

**누적 위험**: $H(t) = \int_0^t h(u)du = -\log S(t)$

### 2. Cox 비례 위험 모형

$$h(t|\mathbf{x}) = h_0(t)\exp(\mathbf{x}^\top\boldsymbol{\beta})$$

- $h_0(t)$: 기저 위험함수 (비모수적)
- $\exp(\beta_j)$: **위험비(Hazard Ratio)** — $x_j$가 1 증가 시 위험의 배수 변화

**부분우도(Partial Likelihood)**:

$$L(\boldsymbol{\beta}) = \prod_{i: \delta_i=1} \frac{\exp(\mathbf{x}_i^\top\boldsymbol{\beta})}{\sum_{j \in R(t_i)} \exp(\mathbf{x}_j^\top\boldsymbol{\beta})}$$

여기서 $R(t_i)$는 시각 $t_i$에서의 위험 집합 (risk set)

### 3. AFT 모형

$$\log T = \mathbf{x}^\top\boldsymbol{\beta} + \sigma\varepsilon$$

- $\varepsilon$의 분포에 따라: Weibull, Log-normal, Log-logistic 등
- 해석: $\exp(\beta_j)$는 시간 가속 인자 (time ratio)
- $\exp(\beta_j) > 1$: 생존 시간 연장

### 4. Concordance Index (C-index)

$$C = P(\hat{\eta}_i > \hat{\eta}_j | T_i < T_j)$$

- 순서가 일치하는 쌍의 비율
- C = 0.5: 무작위, C = 1.0: 완벽한 판별
- 생존 분석에서의 AUC에 해당

### 5. 비례 위험 가정 진단

**Schoenfeld 잔차 검정**:
$$\text{corr}(\text{Schoenfeld residual}_j, \text{rank}(t)) = 0$$

귀무가설: 비례 위험 가정 성립

시각적 진단: $\log(-\log S(t))$ vs $\log t$ → 평행한 곡선이면 PH 성립
""",
    guided_code="""# 생존 회귀 구현 가이드
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

# --- 1. 생존 데이터 생성 ---
np.random.seed(42)
n = 500
age = np.random.normal(60, 10, n)
treatment = np.random.binomial(1, 0.5, n)

# Weibull 생존 시간
shape = 2
scale = np.exp(3 - 0.02*age + 0.5*treatment)
time = np.random.weibull(shape, n) * scale

# 중도절단 (20% 정도)
censor_time = np.random.exponential(np.median(time)*1.5, n)
observed_time = np.minimum(time, censor_time)
event = (time <= censor_time).astype(int)

df = pd.DataFrame({
    'time': observed_time,
    'event': event,
    'age': age,
    'treatment': treatment
})

print(f"중도절단율: {1-event.mean():.1%}")
print(f"관측 건수: {n}, 사건 발생: {event.sum()}")

# --- 2. Kaplan-Meier ---
kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(10, 6))

for trt, label in [(0, '대조군'), (1, '치료군')]:
    mask = df['treatment'] == trt
    kmf.fit(df.loc[mask, 'time'], df.loc[mask, 'event'], label=label)
    kmf.plot_survival_function(ax=ax)

ax.set_xlabel('시간'); ax.set_ylabel('생존 확률')
ax.set_title('Kaplan-Meier 생존 곡선'); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# --- 3. Cox PH 모형 ---
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='event')
cph.print_summary()

# 위험비 해석
print("\\n=== 위험비 (Hazard Ratio) ===")
for var in ['age', 'treatment']:
    hr = np.exp(cph.params_[var])
    ci = np.exp(cph.confidence_intervals_.loc[var])
    print(f"{var}: HR={hr:.4f}, 95% CI=({ci.iloc[0]:.4f}, {ci.iloc[1]:.4f})")

# --- 4. AFT 모형 ---
aft = WeibullAFTFitter()
aft.fit(df, duration_col='time', event_col='event')
aft.print_summary()

# --- 5. C-index ---
c_cox = concordance_index(df['time'], -cph.predict_partial_hazard(df), df['event'])
print(f"\\nCox C-index: {c_cox:.4f}")

# --- 6. PH 가정 검정 ---
cph.check_assumptions(df, show_plots=True)""",
    exercises=[
        {
            "difficulty": "★",
            "description": "주어진 생존 데이터에 Cox PH 모형을 적합하고, 위험비(HR)를 해석하세요.",
            "hint": "exp(coef)가 HR입니다. HR>1은 위험 증가, HR<1은 위험 감소.",
            "skeleton": "from lifelines import CoxPHFitter\n\n# 폐암 환자 데이터 생성\nnp.random.seed(42)\nn = 400\ndf = pd.DataFrame({\n    'time': np.random.weibull(1.5, n) * 365,\n    'event': np.random.binomial(1, 0.7, n),\n    'age': np.random.normal(65, 8, n),\n    'stage': np.random.choice([1,2,3,4], n, p=[0.1,0.3,0.35,0.25]),\n    'chemo': np.random.binomial(1, 0.5, n)\n})\n\n# TODO: Cox PH 적합 및 HR 해석\n"
        },
        {
            "difficulty": "★★",
            "description": "Cox PH와 Weibull AFT 모형을 비교하고, C-index로 예측 성능을 평가하세요.",
            "skeleton": "from lifelines import CoxPHFitter, WeibullAFTFitter\nfrom lifelines.utils import concordance_index\n\n# TODO: 두 모형 적합\n# TODO: C-index 비교\n# TODO: AIC 비교\n"
        },
        {
            "difficulty": "★★★",
            "description": "비례 위험 가정을 Schoenfeld 잔차로 검정하고, 가정이 위배되는 변수에 대한 처리 방안을 제시하세요.",
            "hint": "시간 변환(time-varying coefficients) 또는 층화(stratification)를 고려하세요.",
            "skeleton": "# PH 가정 위배 데이터 (치료 효과가 시간에 따라 감소)\nnp.random.seed(42)\nn = 500\ntreatment = np.random.binomial(1, 0.5, n)\nage = np.random.normal(60, 10, n)\n\n# 시간에 따라 치료 효과 감소 → PH 위배\ntime = np.random.weibull(2, n) * np.exp(2 - 0.01*age)\ntime[treatment == 1] *= np.exp(0.5 * np.random.uniform(0.5, 1.5, treatment.sum()))\n\nevent = np.random.binomial(1, 0.75, n)\ndf = pd.DataFrame({'time': time, 'event': event, 'age': age, 'treatment': treatment})\n\n# TODO: PH 가정 검정\n# TODO: 위배 시 대안 모형\n"
        }
    ],
    references=[
        "Collett (2015). Modelling Survival Data in Medical Research, 3rd ed.",
        "Klein & Moeschberger (2003). Survival Analysis, 2nd ed.",
        "lifelines documentation: https://lifelines.readthedocs.io/",
    ],
    filepath=os.path.join(OUT, "ch05_08_survival_regression_problems.ipynb")
)

solution_notebook(
    chapter_num=CH, section_num=8,
    title="생존 회귀 (Survival Regression)",
    solutions=[
        {
            "approach": "Cox PH 모형을 적합하고 위험비를 해석합니다.",
            "code": """import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

np.random.seed(42)
n = 400
df = pd.DataFrame({
    'time': np.random.weibull(1.5, n) * 365,
    'event': np.random.binomial(1, 0.7, n),
    'age': np.random.normal(65, 8, n),
    'stage': np.random.choice([1,2,3,4], n, p=[0.1,0.3,0.35,0.25]),
    'chemo': np.random.binomial(1, 0.5, n)
})

cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='event')
cph.print_summary()

# HR 해석
print("\\n=== 위험비 (Hazard Ratio) 해석 ===")
for var in ['age', 'stage', 'chemo']:
    hr = np.exp(cph.params_[var])
    ci = np.exp(cph.confidence_intervals_.loc[var])
    pval = cph.summary['p'][var]
    print(f"{var}: HR={hr:.4f}, 95% CI=({ci.iloc[0]:.4f}, {ci.iloc[1]:.4f}), p={pval:.4f}")

# 생존 곡선 시각화
fig, ax = plt.subplots(figsize=(10, 6))
cph.plot_partial_effects_on_outcome(covariates='chemo', values=[0, 1], ax=ax)
ax.set_title('치료 여부에 따른 생존 곡선 (Cox PH)')
plt.tight_layout(); plt.show()""",
            "interpretation": "HR > 1은 해당 변수가 증가할 때 사건 발생 위험이 증가함을 의미합니다. 예를 들어 stage의 HR이 1.2이면 병기가 1단계 올라갈 때 위험이 20% 증가합니다."
        },
        {
            "approach": "Cox PH와 Weibull AFT를 비교합니다.",
            "code": """import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index

np.random.seed(42)
n = 400
df = pd.DataFrame({
    'time': np.random.weibull(1.5, n) * 365,
    'event': np.random.binomial(1, 0.7, n),
    'age': np.random.normal(65, 8, n),
    'stage': np.random.choice([1,2,3,4], n, p=[0.1,0.3,0.35,0.25]),
    'chemo': np.random.binomial(1, 0.5, n)
})

# Cox PH
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='event')

# Weibull AFT
aft = WeibullAFTFitter()
aft.fit(df, duration_col='time', event_col='event')

# C-index
c_cox = concordance_index(df['time'], -cph.predict_partial_hazard(df), df['event'])
c_aft = concordance_index(df['time'], aft.predict_median(df), df['event'])

print(f"{'모형':<15} {'C-index':<10} {'AIC':<12}")
print(f"{'Cox PH':<15} {c_cox:<10.4f} {cph.AIC_partial_:<12.2f}")
print(f"{'Weibull AFT':<15} {c_aft:<10.4f} {aft.AIC_:<12.2f}")

# 계수 비교
print("\\n=== 계수 비교 ===")
print(f"{'변수':<10} {'Cox HR':<10} {'AFT TR':<10}")
for var in ['age', 'stage', 'chemo']:
    hr = np.exp(cph.params_[var])
    tr = np.exp(aft.params_.get(('mu_', var), 0))
    print(f"{var:<10} {hr:<10.4f} {tr:<10.4f}")
print("\\nHR > 1 → 위험 증가 (나쁨)")
print("TR > 1 → 생존 시간 연장 (좋음)")""",
            "interpretation": "Cox PH의 HR과 AFT의 Time Ratio는 역의 관계입니다. HR > 1이면 TR < 1 (위험 증가 = 생존 시간 단축). C-index가 높을수록 예측 판별력이 좋습니다."
        },
        {
            "approach": "Schoenfeld 잔차로 PH 가정을 검정하고, 위배 시 층화 모형을 적용합니다.",
            "code": """import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500
treatment = np.random.binomial(1, 0.5, n)
age = np.random.normal(60, 10, n)

time = np.random.weibull(2, n) * np.exp(2 - 0.01*age)
time[treatment == 1] *= np.exp(0.5 * np.random.uniform(0.5, 1.5, treatment.sum()))

event = np.random.binomial(1, 0.75, n)
df = pd.DataFrame({'time': time, 'event': event, 'age': age, 'treatment': treatment})

# 일반 Cox PH
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='event')
print("=== 일반 Cox PH ===")
cph.print_summary()

# PH 가정 검정
print("\\n=== PH 가정 검정 ===")
try:
    results = cph.check_assumptions(df, p_value_threshold=0.05, show_plots=False)
    print(results)
except Exception as e:
    print(f"검정 결과: {e}")

# 대안 1: 층화 (treatment로 층화)
cph_strat = CoxPHFitter()
cph_strat.fit(df, duration_col='time', event_col='event',
              strata=['treatment'])
print("\\n=== 층화 Cox 모형 (treatment 층화) ===")
cph_strat.print_summary()

# 대안 2: 시간 분할 (piecewise)
median_time = df['time'].median()
df['time_group'] = (df['time'] > median_time).astype(int)
cph_early = CoxPHFitter()
cph_late = CoxPHFitter()

df_early = df[df['time'] <= median_time].copy()
df_late = df[df['time'] > median_time].copy()

if len(df_early[df_early['event']==1]) > 5 and len(df_late[df_late['event']==1]) > 5:
    cph_early.fit(df_early[['time','event','age','treatment']],
                  duration_col='time', event_col='event')
    cph_late.fit(df_late[['time','event','age','treatment']],
                 duration_col='time', event_col='event')
    print(f"\\n초기 치료 HR: {np.exp(cph_early.params_['treatment']):.4f}")
    print(f"후기 치료 HR: {np.exp(cph_late.params_['treatment']):.4f}")
    print("→ 시간에 따른 치료 효과 변화 확인")""",
            "interpretation": "PH 가정이 위배되면 (1) 해당 변수로 층화, (2) 시간 구간별 모형, (3) 시간 의존 계수 모형 등을 고려합니다. 층화 모형은 해당 변수의 기저 위험함수를 각 층별로 따로 추정합니다."
        }
    ],
    discussion="""
### 생존 회귀 모형 선택 가이드

| 상황 | 추천 모형 |
|------|-----------|
| PH 가정 성립, 기저 위험 관심 없음 | Cox PH |
| 특정 분포 가정 가능 | AFT (Weibull, Log-normal) |
| PH 가정 위배 | 층화 Cox, AFT, 시간 의존 계수 |
| 경쟁 위험 존재 | Fine-Gray, 원인별 위험 모형 |

### ADP 시험 포인트
- Cox PH의 부분우도 정의와 기저 위험의 비모수적 처리
- HR의 해석 (배수적 효과)
- C-index의 정의 (concordant pairs 비율)
- PH 가정 진단 방법
""",
    filepath=os.path.join(OUT, "ch05_08_survival_regression_solutions.ipynb")
)

# ============================================================
# Topic 9: Zero-Inflated Models
# ============================================================
problem_notebook(
    chapter_num=CH, section_num=9,
    title="제로 팽창 모형 (Zero-Inflated Models)",
    objectives=[
        "과분산과 영과잉(zero-inflation)의 구분",
        "ZIP/ZINB 모형의 구조와 추정",
        "Hurdle 모형과 ZIP의 차이",
        "모형 비교와 선택 (Vuong 검정)",
    ],
    theory_md=r"""
### 1. 영과잉 문제

카운트 데이터에서 관측된 0의 비율이 포아송/음이항이 예측하는 것보다 많은 현상

예: 의료 서비스 이용 횟수, 보험 청구 건수, 결함 수

### 2. ZIP (Zero-Inflated Poisson)

두 가지 과정의 혼합:
1. 확률 $\pi$로 **구조적 0** 생성 (항상 0)
2. 확률 $(1-\pi)$로 포아송 과정 (0 포함 가능)

$$P(Y=0) = \pi + (1-\pi)e^{-\lambda}$$
$$P(Y=k) = (1-\pi)\frac{e^{-\lambda}\lambda^k}{k!}, \quad k=1,2,\ldots$$

모형:
- **영팽창 모형** (inflate): $\text{logit}(\pi_i) = \mathbf{z}_i^\top\boldsymbol{\gamma}$
- **카운트 모형** (count): $\log(\lambda_i) = \mathbf{x}_i^\top\boldsymbol{\beta}$

### 3. ZINB (Zero-Inflated Negative Binomial)

ZIP + 과분산 → 카운트 부분을 음이항으로 대체:

$$P(Y=k|Y \text{ from NB}) = \frac{\Gamma(k+r)}{\Gamma(r)k!}\left(\frac{r}{r+\lambda}\right)^r\left(\frac{\lambda}{r+\lambda}\right)^k$$

### 4. Hurdle 모형

"장벽" 모형: 0과 양수를 별개의 과정으로 모형화

$$P(Y=0) = f_1(0)$$
$$P(Y=k) = \frac{1-f_1(0)}{1-f_2(0)} f_2(k), \quad k=1,2,\ldots$$

**ZIP vs Hurdle 차이**:
- ZIP: 0은 두 출처 (구조적 0 + 포아송 0)
- Hurdle: 0은 오직 하나의 과정, 양수는 절단 분포

### 5. 모형 비교

**Vuong 검정**: 비내포(non-nested) 모형 비교

$$V = \frac{\sqrt{n} \cdot \bar{m}}{s_m}, \quad m_i = \log\frac{f_1(y_i)}{f_2(y_i)}$$

- $V > 1.96$: 모형 1 선호
- $V < -1.96$: 모형 2 선호
- $|V| \le 1.96$: 구분 불가
""",
    guided_code="""# 제로 팽창 모형 구현 가이드
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
import matplotlib.pyplot as plt

# --- 1. 영과잉 데이터 생성 ---
np.random.seed(42)
n = 1000
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.4, n)

# 영팽창 확률
pi = 1 / (1 + np.exp(-(0.5 - 0.8*x1)))
# 카운트 과정
lam = np.exp(0.5 + 0.3*x1 + 0.5*x2)

# ZIP 데이터 생성
is_zero = np.random.binomial(1, pi)
count_part = np.random.poisson(lam)
y = np.where(is_zero, 0, count_part)

X = pd.DataFrame({'x1': x1, 'x2': x2})
print(f"영의 비율: {(y==0).mean():.3f}")
print(f"포아송이 예측하는 영 비율: {np.exp(-lam.mean()):.3f}")
print(f"→ 영과잉 존재!")

# --- 2. 분포 비교 ---
fig, ax = plt.subplots(figsize=(10, 6))
max_val = min(int(np.percentile(y, 99)), 15)
bins = np.arange(0, max_val+2) - 0.5
ax.hist(y, bins=bins, density=True, alpha=0.5, label='관측', edgecolor='black')

# 포아송 기대
from scipy.stats import poisson
x_vals = np.arange(0, max_val+1)
pois_pmf = poisson.pmf(x_vals, y.mean())
ax.plot(x_vals, pois_pmf, 'ro-', label=f'Poisson(λ={y.mean():.2f})')

ax.set_xlabel('카운트'); ax.set_ylabel('확률')
ax.set_title('영과잉 데이터 vs 포아송'); ax.legend()
plt.tight_layout(); plt.show()

# --- 3. ZIP 모형 적합 ---
X_const = sm.add_constant(X)
zip_model = ZeroInflatedPoisson(y, X_const, exog_infl=X_const, inflation='logit')
zip_result = zip_model.fit(disp=0)
print("\\n=== ZIP 모형 ===")
print(zip_result.summary())

# --- 4. 일반 포아송과 비교 ---
pois_model = sm.GLM(y, X_const, family=sm.families.Poisson())
pois_result = pois_model.fit()

print(f"\\n{'모형':<10} {'AIC':<12} {'BIC':<12}")
print(f"{'Poisson':<10} {pois_result.aic:<12.2f} {pois_result.bic:<12.2f}")
print(f"{'ZIP':<10} {zip_result.aic:<12.2f} {zip_result.bic:<12.2f}")

# --- 5. ZINB ---
zinb_model = ZeroInflatedNegativeBinomialP(y, X_const, exog_infl=X_const, inflation='logit')
zinb_result = zinb_model.fit(disp=0, maxiter=100)
print(f"{'ZINB':<10} {zinb_result.aic:<12.2f} {zinb_result.bic:<12.2f}")""",
    exercises=[
        {
            "difficulty": "★",
            "description": "주어진 카운트 데이터에서 영과잉 여부를 진단하세요.\n\n관측 영 비율과 포아송이 예측하는 영 비율을 비교하세요.",
            "hint": "포아송 영 비율 = exp(-lambda_hat)",
            "skeleton": "# 보험 청구 데이터\nnp.random.seed(42)\nn = 500\nage = np.random.normal(40, 10, n)\ny = np.zeros(n, dtype=int)\nmask = np.random.binomial(1, 0.6, n).astype(bool)\ny[mask] = np.random.poisson(1 + 0.02*age[mask])\n\n# TODO: 영과잉 진단\n# TODO: 히스토그램으로 시각화\n"
        },
        {
            "difficulty": "★★",
            "description": "ZIP과 Hurdle 모형을 적합하고, AIC/BIC로 비교하세요.\n\nHurdle 모형은 로지스틱 (0 vs 양수) + 절단 포아송으로 구성됩니다.",
            "skeleton": "# TODO: ZIP 적합 (statsmodels)\n# TODO: Hurdle 수동 구현 (logistic + truncated Poisson)\n# TODO: AIC 비교\n"
        },
        {
            "difficulty": "★★★",
            "description": "Vuong 검정을 수동으로 구현하여 포아송, ZIP, ZINB 모형을 비교하세요.",
            "hint": "V = sqrt(n) * mean(m) / std(m), 여기서 m_i = log(f1(yi)/f2(yi))",
            "skeleton": "from scipy import stats\n\ndef vuong_test(llf1, llf2):\n    \"\"\"Vuong 검정 (로그우도 벡터 비교)\"\"\"\n    # TODO: 구현\n    pass\n\n# TODO: 세 모형의 개별 관측치 로그우도 계산\n# TODO: Vuong 검정 수행\n"
        }
    ],
    references=[
        "Lambert (1992). Zero-Inflated Poisson Regression. Technometrics.",
        "Cameron & Trivedi (2013). Regression Analysis of Count Data, 2nd ed.",
        "Vuong (1989). Likelihood Ratio Tests for Model Selection. Econometrica.",
    ],
    filepath=os.path.join(OUT, "ch05_09_zero_inflated_problems.ipynb")
)

solution_notebook(
    chapter_num=CH, section_num=9,
    title="제로 팽창 모형 (Zero-Inflated Models)",
    solutions=[
        {
            "approach": "관측 영 비율과 포아송 예측 영 비율을 비교하여 영과잉을 진단합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500
age = np.random.normal(40, 10, n)
y = np.zeros(n, dtype=int)
mask = np.random.binomial(1, 0.6, n).astype(bool)
y[mask] = np.random.poisson(1 + 0.02*age[mask])

# 포아송 적합
X = sm.add_constant(age)
pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
lam_hat = pois.fittedvalues

# 영과잉 진단
obs_zero = (y == 0).mean()
exp_zero = np.mean(np.exp(-lam_hat))
print(f"관측 영 비율: {obs_zero:.4f}")
print(f"포아송 예측 영 비율: {exp_zero:.4f}")
print(f"초과 영: {obs_zero - exp_zero:.4f}")
print(f"→ {'영과잉 존재!' if obs_zero > exp_zero * 1.1 else '영과잉 미미'}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))
max_val = min(int(y.max()), 15)
vals = np.arange(0, max_val+1)

obs_freq = np.array([(y == v).mean() for v in vals])
exp_freq = np.array([np.mean(poisson.pmf(v, lam_hat)) for v in vals])

width = 0.35
ax.bar(vals - width/2, obs_freq, width, label='관측', alpha=0.7)
ax.bar(vals + width/2, exp_freq, width, label='포아송 기대', alpha=0.7)
ax.set_xlabel('카운트'); ax.set_ylabel('비율')
ax.set_title('영과잉 진단'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()""",
            "interpretation": "관측 영 비율이 포아송 모형이 예측하는 것보다 유의하게 높으면 영과잉이 존재합니다. 이 경우 ZIP 또는 ZINB 모형을 고려해야 합니다."
        },
        {
            "approach": "ZIP과 수동 Hurdle 모형을 비교합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from sklearn.linear_model import LogisticRegression
from scipy.stats import poisson
from scipy.optimize import minimize

np.random.seed(42)
n = 500
age = np.random.normal(40, 10, n)
y = np.zeros(n, dtype=int)
mask = np.random.binomial(1, 0.6, n).astype(bool)
y[mask] = np.random.poisson(1 + 0.02*age[mask])

X = sm.add_constant(age)

# ZIP
zip_model = ZeroInflatedPoisson(y, X, exog_infl=X, inflation='logit')
zip_result = zip_model.fit(disp=0)

# Hurdle 모형 수동 구현
# Part 1: P(Y > 0) - 로지스틱
y_bin = (y > 0).astype(int)
logit_model = sm.Logit(y_bin, X).fit(disp=0)

# Part 2: 양수 카운트 - 절단 포아송 (Y | Y > 0)
y_pos = y[y > 0]
X_pos = X[y > 0]

# 절단 포아송: P(Y=k|Y>0) = P(Y=k) / (1 - P(Y=0))
def truncated_poisson_nll(beta, X, y):
    lam = np.exp(X @ beta)
    log_pmf = y * np.log(lam) - lam - np.array([np.sum(np.log(np.arange(1, yi+1))) for yi in y])
    log_trunc = np.log(1 - np.exp(-lam))
    return -np.sum(log_pmf - log_trunc)

beta0 = np.zeros(X_pos.shape[1])
result_tp = minimize(truncated_poisson_nll, beta0, args=(X_pos, y_pos), method='Nelder-Mead')

# AIC 계산
k_zip = len(zip_result.params)
k_hurdle = len(logit_model.params) + len(result_tp.x)

aic_zip = zip_result.aic
aic_hurdle = 2*k_hurdle + 2*(logit_model.llf * (-1) + result_tp.fun)

# 일반 포아송
pois_result = sm.GLM(y, X, family=sm.families.Poisson()).fit()

print(f"{'모형':<15} {'AIC':<12} {'파라미터 수':<12}")
print(f"{'Poisson':<15} {pois_result.aic:<12.2f} {2:<12}")
print(f"{'ZIP':<15} {aic_zip:<12.2f} {k_zip:<12}")
print(f"{'Hurdle':<15} {aic_hurdle:<12.2f} {k_hurdle:<12}")""",
            "interpretation": "ZIP은 0이 구조적 0과 포아송 0의 혼합인 반면, Hurdle은 0과 양수를 완전히 분리합니다. 영의 발생 원인에 대한 이론적 근거에 따라 선택합니다."
        },
        {
            "approach": "Vuong 검정을 구현하여 비내포 모형을 비교합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from scipy import stats
from scipy.stats import poisson, norm

def vuong_test(model1_name, model2_name, llf1, llf2):
    \"\"\"Vuong 검정\"\"\"
    m = llf1 - llf2
    n = len(m)
    v_stat = np.sqrt(n) * np.mean(m) / np.std(m, ddof=1)
    p_value = 2 * (1 - norm.cdf(np.abs(v_stat)))

    print(f"\\nVuong 검정: {model1_name} vs {model2_name}")
    print(f"V = {v_stat:.4f}, p = {p_value:.4f}")
    if v_stat > 1.96:
        print(f"→ {model1_name} 선호")
    elif v_stat < -1.96:
        print(f"→ {model2_name} 선호")
    else:
        print(f"→ 구분 불가")
    return v_stat, p_value

# 데이터
np.random.seed(42)
n = 1000
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.4, n)
pi = 1 / (1 + np.exp(-(0.5 - 0.8*x1)))
lam = np.exp(0.5 + 0.3*x1 + 0.5*x2)
is_zero = np.random.binomial(1, pi)
y = np.where(is_zero, 0, np.random.poisson(lam))

X = sm.add_constant(pd.DataFrame({'x1': x1, 'x2': x2}))

# 모형 적합
pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
zip_mod = ZeroInflatedPoisson(y, X, exog_infl=X, inflation='logit').fit(disp=0)

# 개별 로그우도 계산
# Poisson
lam_pois = pois.fittedvalues
llf_pois = poisson.logpmf(y, lam_pois)

# ZIP
pi_zip = zip_mod.predict(X, exog_infl=X, which='prob-inflate')
lam_zip = zip_mod.predict(X, exog_infl=X, which='mean-nonzero')
llf_zip = np.where(y == 0,
    np.log(pi_zip + (1 - pi_zip) * np.exp(-lam_zip)),
    np.log(1 - pi_zip) + poisson.logpmf(y, lam_zip))

# Vuong 검정
vuong_test("ZIP", "Poisson", llf_zip, llf_pois)

print(f"\\n{'모형':<10} {'AIC':<12} {'BIC':<12}")
print(f"{'Poisson':<10} {pois.aic:<12.2f} {pois.bic:<12.2f}")
print(f"{'ZIP':<10} {zip_mod.aic:<12.2f} {zip_mod.bic:<12.2f}")""",
            "interpretation": "Vuong 검정통계량 V > 1.96이면 ZIP이 포아송보다 유의하게 우수합니다. 이는 데이터에 구조적 영과잉이 존재하여 단순 포아송으로는 부적절함을 의미합니다."
        }
    ],
    discussion="""
### 제로 팽창 모형 선택 흐름

1. 영과잉 진단 (관측 vs 예측 영 비율)
2. 과분산 확인 (분산 > 평균이면 NB 계열)
3. 모형 선택:
   - 영과잉만: ZIP
   - 영과잉 + 과분산: ZINB
   - 0/양수 별개 과정: Hurdle
4. Vuong 검정으로 비내포 모형 비교

### ADP 시험 포인트
- ZIP의 혼합 구조 (구조적 0 + 카운트 0)
- Hurdle과 ZIP의 차이
- Vuong 검정의 검정통계량 계산과 해석
""",
    filepath=os.path.join(OUT, "ch05_09_zero_inflated_solutions.ipynb")
)

# ============================================================
# Topic 10: Practice - Real Estate
# ============================================================
problem_notebook(
    chapter_num=CH, section_num=10,
    title="실전: 부동산 가격 모델링",
    objectives=[
        "다양한 회귀 기법을 종합적으로 활용하여 부동산 가격 예측",
        "변수 선택, 비선형 효과, 공간 효과를 고려한 모형 구축",
        "모형 진단 및 해석 능력 배양",
        "ADP 실전 문제 유형 대비",
    ],
    theory_md=r"""
### 실전 분석 프레임워크

1. **데이터 탐색**: 분포, 이상치, 결측, 다중공선성
2. **기본 모형**: OLS → 잔차 진단
3. **비선형**: GAM, 다항식, 스플라인
4. **정칙화**: Ridge/Lasso → 변수 선택
5. **특수 구조**: 혼합 효과 (지역 랜덤 효과), GLM (로그 변환 vs 감마)
6. **모형 비교**: CV-RMSE, AIC/BIC, 잔차 분석
7. **해석**: 계수 해석, 변수 중요도, 부분 의존성

### 부동산 가격 모형의 주요 고려사항

- **로그 변환**: 가격은 양수 + 우편향 → $\log(\text{price})$
- **공간 효과**: 위치(구/동) → 랜덤 효과 또는 더미 변수
- **비선형 관계**: 면적, 층수, 건축연도 등
- **교호작용**: 면적×위치, 층수×건물유형
- **과분산**: 감마 GLM이 적합할 수 있음
""",
    guided_code="""# 실전: 부동산 가격 분석 가이드
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- 1. 데이터 생성 (서울 아파트 유사) ---
np.random.seed(42)
n = 1000

districts = np.random.choice(['강남', '서초', '송파', '마포', '용산',
                               '강서', '노원', '관악', '성북', '동대문'], n)
district_effect = {'강남': 2.0, '서초': 1.8, '송파': 1.2, '용산': 1.5,
                   '마포': 0.8, '강서': -0.3, '노원': -0.5,
                   '관악': -0.7, '성북': -0.4, '동대문': -0.6}

area = np.random.uniform(30, 150, n)  # 전용면적 (m²)
floor_num = np.random.randint(1, 30, n)  # 층수
build_year = np.random.randint(1985, 2023, n)  # 건축연도
rooms = np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.3, 0.4, 0.2])
near_subway = np.random.binomial(1, 0.6, n)  # 역세권

# 가격 생성 (억 원)
log_price = (1.5
             + np.array([district_effect[d] for d in districts])
             + 0.015 * area
             + 0.003 * area * np.array([1 if d in ['강남','서초','송파'] else 0 for d in districts])
             + 0.005 * np.log(floor_num + 1)
             + 0.01 * (build_year - 2000)
             + 0.05 * rooms
             + 0.1 * near_subway
             + np.random.normal(0, 0.15, n))

price = np.exp(log_price)  # 억 원

df = pd.DataFrame({
    'price': price, 'log_price': log_price,
    'area': area, 'floor': floor_num, 'build_year': build_year,
    'rooms': rooms, 'near_subway': near_subway, 'district': districts
})

print(df.describe())
print(f"\\n구별 평균 가격 (억원):")
print(df.groupby('district')['price'].mean().sort_values(ascending=False))

# --- 2. 기본 EDA ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0,0].hist(df['price'], bins=50, edgecolor='black')
axes[0,0].set_title('가격 분포')

axes[0,1].hist(df['log_price'], bins=50, edgecolor='black')
axes[0,1].set_title('로그 가격 분포')

axes[0,2].scatter(df['area'], df['price'], alpha=0.3, s=5)
axes[0,2].set_title('면적 vs 가격')

axes[1,0].boxplot([df.loc[df['district']==d, 'price'] for d in sorted(district_effect.keys())],
                  labels=sorted(district_effect.keys()))
axes[1,0].set_title('구별 가격'); axes[1,0].tick_params(axis='x', rotation=45)

axes[1,1].scatter(df['build_year'], df['price'], alpha=0.3, s=5)
axes[1,1].set_title('건축연도 vs 가격')

axes[1,2].scatter(df['floor'], df['price'], alpha=0.3, s=5)
axes[1,2].set_title('층수 vs 가격')

plt.tight_layout(); plt.show()

# --- 3. OLS 기본 모형 ---
formula = 'log_price ~ area + floor + build_year + rooms + near_subway + C(district)'
ols = smf.ols(formula, df).fit()
print("\\n=== OLS 기본 모형 ===")
print(f"R² = {ols.rsquared:.4f}, Adj R² = {ols.rsquared_adj:.4f}")
print(f"AIC = {ols.aic:.2f}")""",
    exercises=[
        {
            "difficulty": "★",
            "description": "주어진 부동산 데이터에서 OLS 회귀를 수행하고, 잔차 진단(정규성, 등분산성, 영향력)을 실시하세요.\n\n1. 잔차 QQ Plot\n2. 잔차 vs 적합값\n3. Cook's Distance",
            "hint": "statsmodels의 OLSResults.get_influence()를 사용하세요.",
            "skeleton": "# TODO: OLS 적합 (log_price ~ ...)\n# TODO: 잔차 진단 (4가지 플롯)\n# TODO: 영향력 있는 관측치 식별\n"
        },
        {
            "difficulty": "★★",
            "description": "Lasso를 사용하여 변수 선택을 수행하고, 교호작용 항(면적×구, 층수×건물연도)을 포함한 확장 모형과 비교하세요.\n\nCV-RMSE로 모형을 비교하세요.",
            "skeleton": "from sklearn.preprocessing import PolynomialFeatures\n\n# TODO: 교호작용 항 생성\n# TODO: Lasso로 변수 선택\n# TODO: 선택된 변수로 최종 모형\n# TODO: CV-RMSE 비교\n"
        },
        {
            "difficulty": "★★",
            "description": "혼합 효과 모형을 적용하여 구(district)를 랜덤 효과로 모형화하세요.\n\n고정 효과 모형(더미 변수)과 혼합 효과 모형을 비교하세요.",
            "skeleton": "import statsmodels.formula.api as smf\n\n# TODO: 고정 효과 모형 (C(district))\n# TODO: 랜덤 효과 모형 (groups=district)\n# TODO: ICC 계산\n# TODO: 두 모형 비교\n"
        },
        {
            "difficulty": "★★★",
            "description": "GAM을 적용하여 면적과 건축연도의 비선형 효과를 포착하고, 감마 GLM(로그 링크)과 로그 변환 OLS를 비교하세요.\n\n최종 모형 선택 보고서를 작성하세요.",
            "skeleton": "from pygam import LinearGAM, s, f\n\n# TODO: GAM 적합 (비선형 + 범주형)\n# TODO: 감마 GLM 적합\n# TODO: 로그 변환 OLS 적합\n# TODO: CV-RMSE로 최종 비교\n# TODO: 결과 해석 및 보고서\n"
        },
        {
            "difficulty": "★★★",
            "description": "분위수 회귀를 적용하여 가격대별(τ=0.1, 0.5, 0.9) 면적 효과의 차이를 분석하세요.\n\n고가 아파트와 저가 아파트에서 면적 프리미엄이 다른지 확인하세요.",
            "skeleton": "from statsmodels.regression.quantile_regression import QuantReg\n\n# TODO: 분위수별 회귀 적합\n# TODO: 면적 계수의 분위수별 변화 시각화\n# TODO: 해석\n"
        }
    ],
    references=[
        "부동산 빅데이터 분석 (국토교통부 실거래가 데이터)",
        "Hastie, Tibshirani & Friedman (2009). Elements of Statistical Learning.",
        "ADP 고급 분석 기법 가이드",
    ],
    filepath=os.path.join(OUT, "ch05_10_practice_realestate_problems.ipynb")
)

solution_notebook(
    chapter_num=CH, section_num=10,
    title="실전: 부동산 가격 모델링",
    solutions=[
        {
            "approach": "OLS 잔차 진단을 체계적으로 수행합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000
districts = np.random.choice(['강남','서초','송파','마포','용산',
                               '강서','노원','관악','성북','동대문'], n)
district_effect = {'강남':2.0,'서초':1.8,'송파':1.2,'용산':1.5,'마포':0.8,
                   '강서':-0.3,'노원':-0.5,'관악':-0.7,'성북':-0.4,'동대문':-0.6}
area = np.random.uniform(30, 150, n)
floor_num = np.random.randint(1, 30, n)
build_year = np.random.randint(1985, 2023, n)
rooms = np.random.choice([1,2,3,4], n, p=[0.1,0.3,0.4,0.2])
near_subway = np.random.binomial(1, 0.6, n)

log_price = (1.5 + np.array([district_effect[d] for d in districts])
             + 0.015*area + 0.005*np.log(floor_num+1) + 0.01*(build_year-2000)
             + 0.05*rooms + 0.1*near_subway + np.random.normal(0, 0.15, n))
df = pd.DataFrame({'log_price': log_price, 'area': area, 'floor': floor_num,
                    'build_year': build_year, 'rooms': rooms,
                    'near_subway': near_subway, 'district': districts})

formula = 'log_price ~ area + floor + build_year + rooms + near_subway + C(district)'
ols = smf.ols(formula, df).fit()

# 잔차 진단
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# QQ Plot
stats.probplot(ols.resid, dist="norm", plot=axes[0, 0])
axes[0, 0].set_title('잔차 QQ Plot')

# 잔차 vs 적합값
axes[0, 1].scatter(ols.fittedvalues, ols.resid, alpha=0.3, s=5)
axes[0, 1].axhline(0, color='r', ls='--')
axes[0, 1].set_xlabel('적합값'); axes[0, 1].set_ylabel('잔차')
axes[0, 1].set_title('잔차 vs 적합값')

# Cook's Distance
influence = ols.get_influence()
cooks_d = influence.cooks_distance[0]
axes[1, 0].stem(range(n), cooks_d, markerfmt=',', linefmt='b-')
axes[1, 0].axhline(4/n, color='r', ls='--', label=f'4/n={4/n:.4f}')
axes[1, 0].set_title("Cook's Distance"); axes[1, 0].legend()

# 잔차 히스토그램
axes[1, 1].hist(ols.resid, bins=50, density=True, alpha=0.7)
x_norm = np.linspace(-0.6, 0.6, 100)
axes[1, 1].plot(x_norm, stats.norm.pdf(x_norm, 0, ols.resid.std()), 'r-')
axes[1, 1].set_title('잔차 분포')

plt.suptitle(f'OLS 잔차 진단 (R²={ols.rsquared:.4f})', fontsize=14)
plt.tight_layout(); plt.show()

# 영향력 관측치
high_cook = np.where(cooks_d > 4/n)[0]
print(f"영향력 높은 관측치 수: {len(high_cook)}/{n}")

# 정규성 검정
_, shapiro_p = stats.shapiro(ols.resid[:500])
print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")

# 등분산성 검정
_, bp_pval, _, _ = sm.stats.diagnostic.het_breuschpagan(ols.resid, ols.model.exog)
print(f"Breusch-Pagan p-value: {bp_pval:.4f}")""",
            "interpretation": "QQ Plot이 직선에 가까우면 잔차 정규성이 양호합니다. 잔차 vs 적합값에서 패턴이 없어야 합니다. Cook's Distance > 4/n인 관측치는 영향력이 높습니다."
        },
        {
            "approach": "교호작용 항을 포함한 확장 모형에 Lasso를 적용하여 변수 선택을 수행합니다.",
            "code": """import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000
districts = np.random.choice(['강남','서초','송파','마포','용산',
                               '강서','노원','관악','성북','동대문'], n)
district_effect = {'강남':2.0,'서초':1.8,'송파':1.2,'용산':1.5,'마포':0.8,
                   '강서':-0.3,'노원':-0.5,'관악':-0.7,'성북':-0.4,'동대문':-0.6}
area = np.random.uniform(30, 150, n)
floor_num = np.random.randint(1, 30, n)
build_year = np.random.randint(1985, 2023, n)
rooms = np.random.choice([1,2,3,4], n, p=[0.1,0.3,0.4,0.2])
near_subway = np.random.binomial(1, 0.6, n)

log_price = (1.5 + np.array([district_effect[d] for d in districts])
             + 0.015*area + 0.005*np.log(floor_num+1) + 0.01*(build_year-2000)
             + 0.05*rooms + 0.1*near_subway + np.random.normal(0, 0.15, n))

df = pd.DataFrame({'log_price': log_price, 'area': area, 'floor': floor_num,
                    'build_year': build_year, 'rooms': rooms,
                    'near_subway': near_subway, 'district': districts})

# 특성 행렬 구성
district_dummies = pd.get_dummies(df['district'], prefix='d', drop_first=True)
X_base = pd.concat([df[['area', 'floor', 'build_year', 'rooms', 'near_subway']],
                     district_dummies], axis=1)

# 교호작용 추가
for d_col in district_dummies.columns:
    X_base[f'area_x_{d_col}'] = df['area'] * district_dummies[d_col]
X_base['area_x_floor'] = df['area'] * df['floor']
X_base['area_sq'] = df['area']**2
X_base['floor_x_year'] = df['floor'] * df['build_year']

y = df['log_price']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_base)

# Lasso CV
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_scaled, y)

selected = X_base.columns[lasso_cv.coef_ != 0]
print(f"전체 변수: {X_base.shape[1]}")
print(f"Lasso 선택 변수: {len(selected)}")
print(f"선택된 변수: {list(selected)}")

# 비교
lr_full = LinearRegression()
lr_selected = LinearRegression()

cv_full = -cross_val_score(lr_full, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
cv_lasso = -cross_val_score(lasso_cv, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')

X_sel = X_scaled[:, lasso_cv.coef_ != 0]
cv_sel = -cross_val_score(lr_selected, X_sel, y, cv=5, scoring='neg_root_mean_squared_error')

print(f"\\nCV-RMSE 비교:")
print(f"OLS 전체: {cv_full.mean():.4f}")
print(f"Lasso: {cv_lasso.mean():.4f}")
print(f"OLS 선택: {cv_sel.mean():.4f}")""",
            "interpretation": "Lasso가 교호작용 항 중 유의미한 것만 선택합니다. 강남/서초/송파의 면적 교호작용이 선택되면 이 지역에서 면적 프리미엄이 더 크다는 것을 의미합니다."
        },
        {
            "approach": "구(district)를 랜덤 효과로 모형화하는 혼합 효과 모형을 적합합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

np.random.seed(42)
n = 1000
districts = np.random.choice(['강남','서초','송파','마포','용산',
                               '강서','노원','관악','성북','동대문'], n)
district_effect = {'강남':2.0,'서초':1.8,'송파':1.2,'용산':1.5,'마포':0.8,
                   '강서':-0.3,'노원':-0.5,'관악':-0.7,'성북':-0.4,'동대문':-0.6}
area = np.random.uniform(30, 150, n)
floor_num = np.random.randint(1, 30, n)
build_year = np.random.randint(1985, 2023, n)
rooms = np.random.choice([1,2,3,4], n, p=[0.1,0.3,0.4,0.2])
near_subway = np.random.binomial(1, 0.6, n)

log_price = (1.5 + np.array([district_effect[d] for d in districts])
             + 0.015*area + 0.005*np.log(floor_num+1) + 0.01*(build_year-2000)
             + 0.05*rooms + 0.1*near_subway + np.random.normal(0, 0.15, n))

df = pd.DataFrame({'log_price': log_price, 'area': area, 'floor': floor_num,
                    'build_year': build_year, 'rooms': rooms,
                    'near_subway': near_subway, 'district': districts})

# 고정 효과 모형
fe = smf.ols('log_price ~ area + floor + build_year + rooms + near_subway + C(district)', df).fit()

# 혼합 효과 모형
me = smf.mixedlm('log_price ~ area + floor + build_year + rooms + near_subway',
                  df, groups=df['district']).fit(reml=True)

# ICC
null_me = smf.mixedlm('log_price ~ 1', df, groups=df['district']).fit(reml=True)
var_b = float(null_me.cov_re.iloc[0, 0])
var_e = null_me.scale
icc = var_b / (var_b + var_e)

print(f"ICC = {icc:.4f} → 전체 분산의 {icc*100:.1f}%가 구 간 차이")

# 비교
print(f"\\n{'항목':<20} {'고정효과(FE)':<15} {'혼합효과(ME)':<15}")
print(f"{'R² / pseudo-R²':<20} {fe.rsquared:<15.4f} {'N/A':<15}")
print(f"{'AIC':<20} {fe.aic:<15.2f} {me.aic:<15.2f}")
print(f"{'BIC':<20} {fe.bic:<15.2f} {me.bic:<15.2f}")

print(f"\\n고정 효과 비교:")
for var in ['area', 'floor', 'build_year', 'rooms', 'near_subway']:
    print(f"  {var}: FE={fe.params[var]:.6f}, ME={me.fe_params[var]:.6f}")

# 구별 효과 비교
print(f"\\n구별 효과:")
blup = me.random_effects
for d in sorted(district_effect.keys()):
    fe_val = fe.params.get(f'C(district)[T.{d}]', 0)
    me_val = blup.get(d, {}).get('Group', 0)
    true_val = district_effect[d] - district_effect.get(sorted(district_effect.keys())[0], 0)
    print(f"  {d}: FE={fe_val:.4f}, ME(BLUP)={me_val:.4f}")""",
            "interpretation": "혼합 효과 모형은 구를 랜덤 효과로 처리하여 새로운 구에도 예측이 가능하고, 데이터가 적은 구에서는 전체 평균 쪽으로 축소(shrinkage)합니다."
        },
        {
            "approach": "GAM, 감마 GLM, 로그 변환 OLS를 비교하여 최종 모형을 선택합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000
districts = np.random.choice(['강남','서초','송파','마포','용산',
                               '강서','노원','관악','성북','동대문'], n)
district_effect = {'강남':2.0,'서초':1.8,'송파':1.2,'용산':1.5,'마포':0.8,
                   '강서':-0.3,'노원':-0.5,'관악':-0.7,'성북':-0.4,'동대문':-0.6}
area = np.random.uniform(30, 150, n)
floor_num = np.random.randint(1, 30, n)
build_year = np.random.randint(1985, 2023, n)
rooms = np.random.choice([1,2,3,4], n, p=[0.1,0.3,0.4,0.2])
near_subway = np.random.binomial(1, 0.6, n)

log_price = (1.5 + np.array([district_effect[d] for d in districts])
             + 0.015*area + 0.005*np.log(floor_num+1) + 0.01*(build_year-2000)
             + 0.05*rooms + 0.1*near_subway + np.random.normal(0, 0.15, n))
price = np.exp(log_price)

df = pd.DataFrame({'price': price, 'log_price': log_price, 'area': area,
                    'floor': floor_num, 'build_year': build_year,
                    'rooms': rooms, 'near_subway': near_subway, 'district': districts})

# 교차검증 비교
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {'OLS(log)': [], 'Gamma GLM': [], 'OLS+poly': []}

for train_idx, test_idx in kf.split(df):
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    # 1) OLS (로그 변환)
    ols = smf.ols('log_price ~ area + floor + build_year + rooms + near_subway + C(district)',
                  train).fit()
    pred_log = ols.predict(test)
    pred_price = np.exp(pred_log)
    results['OLS(log)'].append(np.sqrt(mean_squared_error(test['price'], pred_price)))

    # 2) Gamma GLM
    try:
        X_train = pd.get_dummies(train[['area','floor','build_year','rooms','near_subway','district']],
                                  columns=['district'], drop_first=True)
        X_test = pd.get_dummies(test[['area','floor','build_year','rooms','near_subway','district']],
                                 columns=['district'], drop_first=True)
        # 열 맞추기
        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[X_train.columns]

        X_train_c = sm.add_constant(X_train)
        X_test_c = sm.add_constant(X_test)
        gamma = sm.GLM(train['price'], X_train_c,
                       family=sm.families.Gamma(link=sm.families.links.Log())).fit()
        pred_gamma = gamma.predict(X_test_c)
        results['Gamma GLM'].append(np.sqrt(mean_squared_error(test['price'], pred_gamma)))
    except:
        results['Gamma GLM'].append(np.nan)

    # 3) OLS + 다항식
    train2 = train.copy()
    test2 = test.copy()
    train2['area_sq'] = train2['area']**2
    test2['area_sq'] = test2['area']**2
    ols_poly = smf.ols('log_price ~ area + area_sq + floor + build_year + rooms + near_subway + C(district)',
                       train2).fit()
    pred_poly = np.exp(ols_poly.predict(test2))
    results['OLS+poly'].append(np.sqrt(mean_squared_error(test['price'], pred_poly)))

print("=== CV-RMSE (억원) ===")
for name, scores in results.items():
    scores_clean = [s for s in scores if not np.isnan(s)]
    if scores_clean:
        print(f"{name:<15} {np.mean(scores_clean):.4f} ± {np.std(scores_clean):.4f}")""",
            "interpretation": "로그 변환 OLS는 가격의 우편향 분포를 효과적으로 처리합니다. 감마 GLM은 로그 링크로 유사한 효과를 내지만 역변환 없이 직접 원래 스케일에서 예측합니다."
        },
        {
            "approach": "분위수 회귀로 가격대별 면적 효과 차이를 분석합니다.",
            "code": """import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000
districts = np.random.choice(['강남','서초','송파','마포','용산',
                               '강서','노원','관악','성북','동대문'], n)
district_effect = {'강남':2.0,'서초':1.8,'송파':1.2,'용산':1.5,'마포':0.8,
                   '강서':-0.3,'노원':-0.5,'관악':-0.7,'성북':-0.4,'동대문':-0.6}
area = np.random.uniform(30, 150, n)
floor_num = np.random.randint(1, 30, n)
build_year = np.random.randint(1985, 2023, n)

log_price = (1.5 + np.array([district_effect[d] for d in districts])
             + 0.015*area + 0.003*area*np.array([1 if d in ['강남','서초','송파'] else 0 for d in districts])
             + 0.005*np.log(floor_num+1) + 0.01*(build_year-2000)
             + np.random.normal(0, 0.15, n))

df = pd.DataFrame({'log_price': log_price, 'area': area, 'floor': floor_num,
                    'build_year': build_year})

X = sm.add_constant(df[['area', 'floor', 'build_year']])

# 분위수별 회귀
taus = [0.1, 0.25, 0.5, 0.75, 0.9]
qr_results = {}
for tau in taus:
    qr = QuantReg(df['log_price'], X)
    qr_results[tau] = qr.fit(q=tau)

# OLS 비교
ols = sm.OLS(df['log_price'], X).fit()

# 계수 비교
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
vars_to_plot = ['area', 'floor', 'build_year']

for ax, var in zip(axes, vars_to_plot):
    coefs = [qr_results[tau].params[var] for tau in taus]
    cis_lo = [qr_results[tau].conf_int().loc[var, 0] for tau in taus]
    cis_hi = [qr_results[tau].conf_int().loc[var, 1] for tau in taus]

    ax.plot(taus, coefs, 'b-o', label='분위수 회귀')
    ax.fill_between(taus, cis_lo, cis_hi, alpha=0.2)
    ax.axhline(ols.params[var], color='r', ls='--', label=f'OLS ({ols.params[var]:.6f})')
    ax.set_xlabel('τ'); ax.set_ylabel(f'{var} 계수')
    ax.set_title(f'{var} 효과의 분위수별 변화')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()

# 표
print(f"{'τ':<6} {'area':<12} {'floor':<12} {'build_year':<12}")
for tau in taus:
    r = qr_results[tau]
    print(f"{tau:<6.2f} {r.params['area']:<12.6f} {r.params['floor']:<12.6f} "
          f"{r.params['build_year']:<12.6f}")
print(f"{'OLS':<6} {ols.params['area']:<12.6f} {ols.params['floor']:<12.6f} "
      f"{ols.params['build_year']:<12.6f}")""",
            "interpretation": "면적 계수가 상위 분위수(고가)에서 더 크다면, 고가 아파트일수록 면적 프리미엄이 크다는 의미입니다. 이는 강남 등 고가 지역에서 대형 아파트의 면적당 단가가 더 높은 현상을 반영합니다."
        }
    ],
    discussion="""
### 부동산 가격 모델링 종합 정리

| 기법 | 장점 | 단점 |
|------|------|------|
| OLS (로그) | 해석 용이, 계산 빠름 | 비선형 제한 |
| Lasso | 변수 선택 자동화 | 편향 존재 |
| 혼합 효과 | 지역 효과 모형화 | 수렴 어려움 |
| GAM | 비선형 포착 | 해석 복잡 |
| 분위수 회귀 | 분포 전체 파악 | 교차 분위수 문제 |
| 감마 GLM | 양수 반응, 자연스러운 스케일 | 분포 가정 필요 |

### ADP 시험 전략
1. 데이터 특성 파악 → 적절한 방법 선택
2. 기본 모형 → 진단 → 개선의 순환
3. 모형 비교 시 CV 기반 지표 사용
4. 결과 해석에서 실무적 의미 부여
""",
    filepath=os.path.join(OUT, "ch05_10_practice_realestate_solutions.ipynb")
)

print("Part 2 (Topics 6-10) generated successfully!")
