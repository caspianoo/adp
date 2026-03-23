"""Chapter 02: 확률 과정과 몬테카를로 시뮬레이션 - 노트북 생성기"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.notebook_generator import *

CH = 2
DIR = os.path.dirname(os.path.abspath(__file__))

def gen(tid, filename, title, objectives, theory, code, exercises, refs, solutions, discussion):
    problem_notebook(CH, tid, title, objectives, theory, code, exercises, refs,
                     os.path.join(DIR, f"{filename}.ipynb"))
    solution_notebook(CH, tid, title, solutions, discussion,
                      os.path.join(DIR, f"{filename}_solution.ipynb"))
    print(f"  {filename} done")

# === Topic 1 ===
gen(1, "ch02_01_markov_chains", "이산 마르코프 체인",
    ["전이 행렬의 성질과 n-단계 전이 확률", "정상 분포의 존재와 유일성 조건", "에르고딕 정리와 수렴 속도", "PageRank 알고리즘의 마르코프 체인 해석"],
    r"""### 마르코프 체인 기초
상태 공간 $S = \{1, 2, \ldots, m\}$ 위의 확률 과정 $\{X_n\}$이 **마르코프 성질**을 만족하면:
$$P(X_{n+1} = j | X_n = i, X_{n-1}, \ldots, X_0) = P(X_{n+1} = j | X_n = i) = p_{ij}$$

**전이 행렬** $\mathbf{P} = [p_{ij}]$는 행 합이 1인 확률 행렬입니다.
$n$-단계 전이 확률: $p_{ij}^{(n)} = (\mathbf{P}^n)_{ij}$ (Chapman-Kolmogorov 방정식)

### 정상 분포 (Stationary Distribution)
벡터 $\boldsymbol{\pi}$가 $\boldsymbol{\pi} = \boldsymbol{\pi} \mathbf{P}$를 만족하고 $\sum_i \pi_i = 1$이면 정상 분포입니다.
**에르고딕 정리**: 비가약(irreducible), 비주기(aperiodic) 체인은 유일한 정상 분포를 가지며,
$$\lim_{n\to\infty} p_{ij}^{(n)} = \pi_j \quad \forall i$$

### PageRank
$$\pi_j = \frac{1-d}{N} + d \sum_{i \to j} \frac{\pi_i}{L(i)}$$
여기서 $d$는 감쇠 인자, $L(i)$는 페이지 $i$의 아웃링크 수입니다.
""",
    """import numpy as np
import matplotlib.pyplot as plt

# 날씨 마르코프 체인 (맑음, 흐림, 비)
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])
states = ['맑음', '흐림', '비']

# n-단계 전이 확률
P_n = np.linalg.matrix_power(P, 50)
print("50-단계 전이 행렬 (각 행이 동일 = 수렴):")
print(P_n.round(4))

# 정상 분포: 고유값 분해로 구하기
eigenvalues, eigenvectors = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigenvalues - 1))
pi = np.real(eigenvectors[:, idx])
pi = pi / pi.sum()
print(f"\\n정상 분포: {dict(zip(states, pi.round(4)))}")

# 시뮬레이션으로 검증
n_steps = 10000
state = 0
counts = np.zeros(3)
for _ in range(n_steps):
    state = np.random.choice(3, p=P[state])
    counts[state] += 1
empirical = counts / n_steps
print(f"경험적 분포: {dict(zip(states, empirical.round(4)))}")

# 수렴 시각화
distributions = []
dist = np.array([1.0, 0.0, 0.0])
for n in range(30):
    distributions.append(dist.copy())
    dist = dist @ P

distributions = np.array(distributions)
fig, ax = plt.subplots(figsize=(10, 6))
for i, s in enumerate(states):
    ax.plot(distributions[:, i], label=s)
    ax.axhline(pi[i], color=f'C{i}', linestyle='--', alpha=0.5)
ax.set_xlabel('단계')
ax.set_ylabel('확률')
ax.set_title('마르코프 체인 수렴')
ax.legend()
plt.show()""",
    [{"difficulty": "★", "description": "전이 행렬의 고유값을 분석하여 수렴 속도를 결정하세요. 두 번째로 큰 고유값의 절대값 $|\\lambda_2|$가 **스펙트럼 갭**과 수렴 속도를 어떻게 결정하는지 분석하세요.",
      "skeleton": "# 스펙트럼 갭과 수렴 속도\nP = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])\n\n# TODO: 고유값 계산\n# TODO: 스펙트럼 갭 = 1 - |lambda_2|\n# TODO: mixing time 추정\n# TODO: 다양한 P에 대해 비교\n"},
     {"difficulty": "★★", "description": "PageRank 알고리즘을 Power Iteration으로 구현하세요. 6개 웹페이지의 링크 구조를 정의하고 순위를 계산하세요. 감쇠 인자 $d$에 따른 순위 변화를 분석하세요.",
      "skeleton": "# PageRank 구현\ndef pagerank(adj_matrix, d=0.85, tol=1e-8):\n    # TODO: Google 행렬 구성\n    # TODO: Power iteration\n    pass\n\n# 6-노드 그래프\nadj = np.array([\n    [0,1,1,0,0,0],\n    [0,0,1,1,0,0],\n    [1,0,0,0,0,0],\n    [0,0,0,0,1,1],\n    [0,0,0,1,0,1],\n    [0,0,0,1,0,0]\n])\n"},
     {"difficulty": "★★★", "description": "흡수 마르코프 체인(absorbing chain)의 기본 행렬 $N = (I - Q)^{-1}$을 구하고, 흡수까지의 기대 단계 수와 흡수 확률을 계산하세요. 도박꾼의 파산 문제에 적용하세요.",
      "skeleton": "# 흡수 마르코프 체인 - 도박꾼의 파산\n# 초기 자본 i, 목표 N, 단판 승률 p\n# TODO: 전이 행렬 구성 (상태 0과 N은 흡수 상태)\n# TODO: 기본 행렬 N 계산\n# TODO: 흡수까지 기대 단계 수\n# TODO: 파산 확률 (이론값과 비교)\n"}],
    ["Norris, J.R. (1997). 'Markov Chains'", "Brin, S. & Page, L. (1998). 'The Anatomy of a Large-Scale Hypertextual Web Search Engine'"],
    [{"approach": "고유값의 스펙트럼 갭으로 수렴 속도를 분석합니다.",
      "code": """import numpy as np
import matplotlib.pyplot as plt

# 다양한 전이 행렬
matrices = {
    '빠른 수렴': np.array([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3], [0.3, 0.4, 0.3]]),
    '느린 수렴': np.array([[0.99, 0.005, 0.005], [0.005, 0.99, 0.005], [0.005, 0.005, 0.99]]),
    '주기적': np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, P) in zip(axes, matrices.items()):
    evals = np.linalg.eigvals(P)
    evals_sorted = sorted(np.abs(evals), reverse=True)
    spectral_gap = 1 - evals_sorted[1]
    mixing_time = int(np.ceil(1 / spectral_gap)) if spectral_gap > 0 else float('inf')

    # TV distance 수렴
    pi = np.ones(3) / 3  # 균등 초기
    dists = []
    stationary = np.real(np.linalg.eig(P.T)[1][:, np.argmin(np.abs(np.linalg.eigvals(P.T) - 1))])
    stationary = stationary / stationary.sum()
    current = np.array([1., 0., 0.])
    for step in range(100):
        tv = 0.5 * np.sum(np.abs(current - stationary))
        dists.append(tv)
        current = current @ P

    ax.plot(dists)
    ax.set_title(f'{name}\\n|λ₂|={evals_sorted[1]:.4f}, gap={spectral_gap:.4f}')
    ax.set_xlabel('단계')
    ax.set_ylabel('총변동 거리')
    ax.set_yscale('log')
plt.tight_layout()
plt.show()""",
      "interpretation": "스펙트럼 갭이 클수록 수렴이 빠릅니다. 주기적 체인은 λ₂의 절대값이 1이므로 수렴하지 않습니다(비주기 조건 위반)."},
     {"approach": "Power Iteration 기반 PageRank를 구현합니다.",
      "code": """import numpy as np
import matplotlib.pyplot as plt

def pagerank(adj_matrix, d=0.85, tol=1e-8, max_iter=1000):
    n = adj_matrix.shape[0]
    out_degree = adj_matrix.sum(axis=1)
    out_degree[out_degree == 0] = 1  # dangling node 처리

    H = adj_matrix / out_degree[:, np.newaxis]
    G = d * H + (1 - d) / n * np.ones((n, n))

    pi = np.ones(n) / n
    for i in range(max_iter):
        pi_new = pi @ G
        if np.linalg.norm(pi_new - pi, 1) < tol:
            return pi_new, i + 1
        pi = pi_new
    return pi, max_iter

adj = np.array([
    [0,1,1,0,0,0],
    [0,0,1,1,0,0],
    [1,0,0,0,0,0],
    [0,0,0,0,1,1],
    [0,0,0,1,0,1],
    [0,0,0,1,0,0]
])

d_values = [0.5, 0.7, 0.85, 0.95]
fig, ax = plt.subplots(figsize=(10, 6))
for d in d_values:
    pr, iters = pagerank(adj, d=d)
    ax.bar(np.arange(6) + d_values.index(d)*0.2, pr, width=0.2, label=f'd={d}')
    print(f"d={d}: rank={np.argsort(-pr)+1}, iters={iters}")
ax.set_xticks(range(6))
ax.set_xticklabels([f'Page {i}' for i in range(6)])
ax.set_ylabel('PageRank')
ax.legend()
plt.title('감쇠 인자에 따른 PageRank')
plt.show()""",
      "interpretation": "d가 1에 가까울수록 링크 구조의 영향이 강해집니다. d=0.85가 표준값이며, dangling node 처리가 중요합니다."},
     {"approach": "흡수 마르코프 체인으로 도박꾼의 파산 문제를 분석합니다.",
      "code": """import numpy as np

def gamblers_ruin(N, p, initial_capital):
    # 상태: 0, 1, ..., N (0과 N은 흡수)
    # 일시 상태: 1, ..., N-1
    q = 1 - p
    n_transient = N - 1

    Q = np.zeros((n_transient, n_transient))
    for i in range(n_transient):
        state = i + 1
        if state - 1 >= 1: Q[i, i-1] = q  # 왼쪽
        if state + 1 <= N-1: Q[i, i+1] = p  # 오른쪽

    R = np.zeros((n_transient, 2))  # 흡수 상태: 0(파산), N(목표)
    for i in range(n_transient):
        state = i + 1
        if state - 1 == 0: R[i, 0] = q
        if state + 1 == N: R[i, 1] = p

    # 기본 행렬
    I = np.eye(n_transient)
    N_mat = np.linalg.inv(I - Q)

    # 흡수까지 기대 단계 수
    expected_steps = N_mat.sum(axis=1)

    # 흡수 확률
    B = N_mat @ R

    idx = initial_capital - 1
    print(f"N={N}, p={p}, 초기자본={initial_capital}")
    print(f"  파산 확률: {B[idx, 0]:.6f}")
    print(f"  목표 달성 확률: {B[idx, 1]:.6f}")
    print(f"  기대 게임 수: {expected_steps[idx]:.1f}")

    # 이론적 파산 확률 비교
    if p != 0.5:
        r = q / p
        theory = (r**initial_capital - r**N) / (1 - r**N)
    else:
        theory = 1 - initial_capital / N
    print(f"  이론적 파산 확률: {theory:.6f}")
    return B, expected_steps

for p in [0.4, 0.5, 0.6]:
    gamblers_ruin(N=10, p=p, initial_capital=5)
    print()""",
      "interpretation": "p<0.5(불리한 게임)이면 파산 확률이 기하급수적으로 1에 접근합니다. p=0.5이면 파산 확률은 1 - i/N이고, p>0.5에서도 파산이 가능합니다. 기본 행렬 N의 원소 n_ij는 상태 i에서 출발하여 흡수 전 상태 j를 방문하는 기대 횟수입니다."}],
    "마르코프 체인은 MCMC, 강화학습, 큐잉 이론의 기반입니다. 수렴 속도 분석은 MCMC 진단에 직접 사용됩니다.")

# === Topic 2 ===
gen(2, "ch02_02_continuous_markov", "연속 시간 마르코프 체인과 포아송 과정",
    ["생성 행렬(generator matrix)과 Kolmogorov 방정식", "포아송 과정의 성질과 일반화", "큐잉 이론의 기초 (M/M/1, M/M/c)"],
    r"""### 연속 시간 마르코프 체인 (CTMC)
상태 $i$에서의 체류 시간은 비율 $q_i$의 지수 분포: $T_i \sim \text{Exp}(q_i)$

**생성 행렬** $\mathbf{Q} = [q_{ij}]$: $q_{ij} \geq 0$ ($i \neq j$), $q_{ii} = -\sum_{j \neq i} q_{ij}$

전이 확률: $\mathbf{P}(t) = e^{\mathbf{Q}t}$ (행렬 지수함수)

**Kolmogorov 전진 방정식**: $\frac{d}{dt}\mathbf{P}(t) = \mathbf{P}(t)\mathbf{Q}$

### 포아송 과정
비율 $\lambda$의 포아송 과정 $\{N(t)\}$:
- $N(0) = 0$, 독립 증분, 정상 증분
- $N(t) - N(s) \sim \text{Poisson}(\lambda(t-s))$
- 도착 간 시간: $\text{Exp}(\lambda)$

### M/M/1 큐
도착: 포아송($\lambda$), 서비스: 지수($\mu$). 이용률 $\rho = \lambda/\mu < 1$일 때:
$$L = \frac{\rho}{1-\rho}, \quad W = \frac{1}{\mu - \lambda}$$
""",
    """import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# CTMC: 2-상태 시스템 (작동/고장)
Q = np.array([[-0.1, 0.1], [0.5, -0.5]])  # 고장률 0.1, 수리율 0.5

# 전이 확률 행렬 P(t)
times = np.linspace(0, 20, 100)
P_start_working = [expm(Q * t)[0] for t in times]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times, [p[0] for p in P_start_working], label='P(작동→작동)')
ax.plot(times, [p[1] for p in P_start_working], label='P(작동→고장)')
ax.axhline(0.5/0.6, color='gray', linestyle='--', alpha=0.5, label='정상 P(작동)')
ax.set_xlabel('시간')
ax.set_ylabel('확률')
ax.set_title('CTMC 전이 확률의 시간 변화')
ax.legend()
plt.show()

# 포아송 과정 시뮬레이션
np.random.seed(42)
lam = 3  # 시간당 3건
T = 10
arrivals = []
t = 0
while t < T:
    t += np.random.exponential(1/lam)
    if t < T:
        arrivals.append(t)

plt.figure(figsize=(12, 4))
plt.step(arrivals, range(1, len(arrivals)+1), where='post')
plt.xlabel('시간')
plt.ylabel('누적 도착 수')
plt.title(f'포아송 과정 (λ={lam}, 관측={len(arrivals)}건, 기대={lam*T}건)')
plt.show()""",
    [{"difficulty": "★", "description": "비동질(non-homogeneous) 포아송 과정을 Thinning 알고리즘으로 시뮬레이션하세요. 강도 함수 $\\lambda(t) = 5 + 3\\sin(2\\pi t / 24)$ (24시간 주기)를 사용하세요.",
      "skeleton": "# Thinning 알고리즘\ndef thinning_poisson(lambda_func, lambda_max, T):\n    # TODO: 동질 포아송(lambda_max)으로 후보 생성\n    # TODO: 각 후보를 lambda(t)/lambda_max 확률로 수락\n    pass\n"},
     {"difficulty": "★★", "description": "M/M/1 큐를 이산 사건 시뮬레이션으로 구현하고, 이론적 결과(L, W, Lq, Wq)와 비교하세요. 이용률 ρ에 따른 대기열 길이 변화를 분석하세요.",
      "skeleton": "# M/M/1 큐 시뮬레이션\ndef simulate_mm1(lam, mu, n_customers=10000):\n    # TODO: 도착 시각과 서비스 시작/종료 시각 추적\n    # TODO: 대기 시간, 시스템 내 고객 수 계산\n    pass\n"},
     {"difficulty": "★★★", "description": "Gillespie 알고리즘을 구현하여 SIR 전염병 모형을 확률적으로 시뮬레이션하세요. 결정론적 ODE 해와 비교하세요.",
      "skeleton": "# Gillespie 알고리즘 - SIR 모형\n# 반응: S+I -> 2I (rate β*S*I/N), I -> R (rate γ*I)\ndef gillespie_sir(S0, I0, R0, beta, gamma, T):\n    # TODO: 다음 반응 시각 (지수 분포)\n    # TODO: 반응 선택 (확률 비례)\n    pass\n"}],
    ["Ross, S.M. (2019). 'Introduction to Probability Models (12th ed.)'", "Gillespie, D.T. (1977). 'Exact Stochastic Simulation of Coupled Chemical Reactions'"],
    [{"approach": "Thinning 알고리즘으로 비동질 포아송 과정을 시뮬레이션합니다.",
      "code": """import numpy as np
import matplotlib.pyplot as plt

def thinning_poisson(lambda_func, lambda_max, T, seed=42):
    np.random.seed(seed)
    events = []
    t = 0
    while t < T:
        t += np.random.exponential(1 / lambda_max)
        if t < T and np.random.rand() < lambda_func(t) / lambda_max:
            events.append(t)
    return np.array(events)

lambda_func = lambda t: 5 + 3 * np.sin(2 * np.pi * t / 24)
events = thinning_poisson(lambda_func, lambda_max=8, T=72)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
t_grid = np.linspace(0, 72, 1000)
axes[0].plot(t_grid, lambda_func(t_grid), 'b-')
axes[0].set_ylabel('λ(t)')
axes[0].set_title('강도 함수')

# 시간대별 도착 빈도
axes[1].hist(events, bins=72, edgecolor='black', alpha=0.7, density=True)
axes[1].plot(t_grid, lambda_func(t_grid) / np.trapz(lambda_func(t_grid), t_grid), 'r-')
axes[1].set_xlabel('시간')
axes[1].set_ylabel('상대 빈도')
axes[1].set_title(f'도착 분포 (총 {len(events)}건)')
plt.tight_layout()
plt.show()""",
      "interpretation": "Thinning은 강도 함수의 상한 λ_max로 후보를 생성한 후 λ(t)/λ_max 확률로 채택합니다. 효율은 E[λ(t)]/λ_max에 비례합니다."},
     {"approach": "이산 사건 시뮬레이션으로 M/M/1 큐를 구현합니다.",
      "code": """import numpy as np
import matplotlib.pyplot as plt

def simulate_mm1(lam, mu, n_customers=10000):
    np.random.seed(42)
    inter_arrivals = np.random.exponential(1/lam, n_customers)
    service_times = np.random.exponential(1/mu, n_customers)
    arrivals = np.cumsum(inter_arrivals)

    departures = np.zeros(n_customers)
    departures[0] = arrivals[0] + service_times[0]
    for i in range(1, n_customers):
        start = max(arrivals[i], departures[i-1])
        departures[i] = start + service_times[i]

    waits = departures - arrivals - service_times
    system_times = departures - arrivals
    return {'W': system_times.mean(), 'Wq': waits.mean(),
            'L': lam * system_times.mean(), 'Lq': lam * waits.mean()}

rhos = np.arange(0.1, 0.99, 0.1)
L_sim, L_theory = [], []
for rho in rhos:
    lam, mu = rho, 1.0
    result = simulate_mm1(lam, mu)
    L_sim.append(result['L'])
    L_theory.append(rho / (1 - rho))

plt.figure(figsize=(10, 6))
plt.plot(rhos, L_theory, 'b-o', label='이론 L=ρ/(1-ρ)')
plt.plot(rhos, L_sim, 'r--s', label='시뮬레이션')
plt.xlabel('이용률 ρ')
plt.ylabel('평균 시스템 내 고객 수 L')
plt.title('M/M/1 큐: 이론 vs 시뮬레이션')
plt.legend()
plt.yscale('log')
plt.show()""",
      "interpretation": "ρ→1이면 L이 급격히 증가합니다. Little의 법칙 L=λW가 시뮬레이션에서도 성립함을 확인합니다."},
     {"approach": "Gillespie 알고리즘으로 확률적 SIR을 구현합니다.",
      "code": """import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def gillespie_sir(S0, I0, R0, beta, gamma, T):
    S, I, R = S0, I0, R0
    N = S + I + R
    t = 0
    trajectory = [(t, S, I, R)]

    while t < T and I > 0:
        rate_infect = beta * S * I / N
        rate_recover = gamma * I
        total_rate = rate_infect + rate_recover

        if total_rate == 0:
            break
        dt = np.random.exponential(1 / total_rate)
        t += dt

        if np.random.rand() < rate_infect / total_rate:
            S -= 1; I += 1
        else:
            I -= 1; R += 1
        trajectory.append((t, S, I, R))

    return np.array(trajectory)

def sir_ode(y, t, beta, gamma, N):
    S, I, R = y
    return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]

N, beta, gamma = 1000, 0.3, 0.1
np.random.seed(42)

fig, ax = plt.subplots(figsize=(12, 6))
for _ in range(20):
    traj = gillespie_sir(N-10, 10, 0, beta, gamma, 200)
    ax.plot(traj[:,0], traj[:,2]/N, 'r-', alpha=0.15)

t_ode = np.linspace(0, 200, 1000)
sol = odeint(sir_ode, [N-10, 10, 0], t_ode, args=(beta, gamma, N))
ax.plot(t_ode, sol[:,1]/N, 'k-', linewidth=3, label='결정론적 ODE')
ax.set_xlabel('시간')
ax.set_ylabel('감염 비율 I/N')
ax.set_title('SIR 모형: Gillespie vs ODE')
ax.legend()
plt.show()""",
      "interpretation": "확률적 모형은 개체 수가 적을 때 결정론적 모형과 큰 차이를 보입니다. 멸종(I=0)이 일어날 수 있어 일부 궤적은 조기 종료됩니다."}],
    "CTMC와 포아송 과정은 큐잉, 신뢰성 공학, 생물학적 과정 모델링의 기초입니다. Gillespie 알고리즘은 화학 반응 시뮬레이션에서 시작했지만, 전염병 모형 등 다양한 분야에 적용됩니다.")

# === Topic 3 ===
gen(3, "ch02_03_random_walks", "랜덤 워크와 브라운 운동",
    ["1D/2D 랜덤 워크의 재귀 성질", "브라운 운동(Wiener 과정)의 수학적 정의", "기하 브라운 운동과 금융 모델링"],
    r"""### 단순 랜덤 워크
$S_n = \sum_{i=1}^{n} X_i$, $X_i \in \{-1, +1\}$, $P(X_i=1) = p$

**재귀 성질** (Pólya): 1D/2D 랜덤 워크는 재귀적(원점 복귀 확률 1), 3D 이상은 일시적(transient)

### 브라운 운동 (Wiener 과정)
$\{W(t)\}_{t \geq 0}$: $W(0) = 0$, 연속 경로, 독립 정규 증분
$$W(t) - W(s) \sim N(0, t-s), \quad s < t$$

**스케일링**: $\frac{1}{\sqrt{n}}S_{\lfloor nt \rfloor} \xrightarrow{d} W(t)$ (Donsker 정리)

### 기하 브라운 운동 (GBM)
$$dS_t = \mu S_t dt + \sigma S_t dW_t \implies S_t = S_0 \exp\left((\mu - \tfrac{\sigma^2}{2})t + \sigma W_t\right)$$
""",
    """import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_steps = 10000

# 1D 랜덤 워크
steps = np.random.choice([-1, 1], n_steps)
walk_1d = np.cumsum(steps)

# 2D 랜덤 워크
directions = np.random.choice(4, n_steps)
dx = np.where(directions==0, 1, np.where(directions==1, -1, 0))
dy = np.where(directions==2, 1, np.where(directions==3, -1, 0))
walk_2d = np.column_stack([np.cumsum(dx), np.cumsum(dy)])

# 브라운 운동 근사
dt = 1 / n_steps
dW = np.random.normal(0, np.sqrt(dt), n_steps)
W = np.cumsum(dW)
t = np.linspace(0, 1, n_steps)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(walk_1d, alpha=0.7)
axes[0].set_title(f'1D 랜덤 워크 (n={n_steps})')
axes[1].plot(walk_2d[:,0], walk_2d[:,1], alpha=0.5, linewidth=0.3)
axes[1].set_title('2D 랜덤 워크')
axes[2].plot(t, W)
axes[2].set_title('브라운 운동 (Wiener 과정)')
plt.tight_layout()
plt.show()""",
    [{"difficulty": "★★", "description": "Donsker 정리를 시각적으로 검증하세요. $n$이 증가할 때 정규화된 랜덤 워크 $W_n(t) = S_{\\lfloor nt \\rfloor}/\\sqrt{n}$이 브라운 운동에 수렴함을 보이세요.",
      "skeleton": "# Donsker 정리 시각적 검증\n# TODO: n = 10, 100, 1000, 10000에서 정규화된 랜덤 워크 비교\n"},
     {"difficulty": "★★", "description": "기하 브라운 운동(GBM)으로 주가 경로를 시뮬레이션하세요. 100개 경로의 최종 가격 분포가 로그정규분포를 따르는지 검증하세요.",
      "skeleton": "# GBM 시뮬레이션과 로그정규 검증\nS0, mu, sigma, T = 100, 0.08, 0.2, 1.0\n# TODO: 100개 경로 생성\n# TODO: 최종 가격 분포 vs 이론적 로그정규\n"},
     {"difficulty": "★★★", "description": "2D 랜덤 워크의 재귀 성질을 시뮬레이션으로 검증하고, 3D에서는 일시적(transient)임을 보이세요. Pólya의 결과 $P(\\text{return}) = 1 - 1/p_{3d}$와 비교하세요.",
      "skeleton": "# 재귀 성질 검증 (2D vs 3D)\n# TODO: 다수의 워크에서 원점 복귀 비율 비교\n"}],
    ["Mörters, P. & Peres, Y. (2010). 'Brownian Motion'", "Lawler, G. & Limic, V. (2010). 'Random Walk: A Modern Introduction'"],
    [{"approach": "Donsker 정리를 다양한 n에서 시각화합니다.",
      "code": """import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, n in zip(axes.flat, [10, 100, 1000, 10000]):
    t_grid = np.linspace(0, 1, 1000)
    for _ in range(5):
        steps = np.random.choice([-1, 1], n)
        S = np.cumsum(steps)
        S_normalized = np.interp(t_grid, np.linspace(0, 1, n), S / np.sqrt(n))
        ax.plot(t_grid, S_normalized, alpha=0.5)
    ax.set_title(f'n = {n}')
    ax.set_ylim(-3, 3)
plt.suptitle('Donsker 정리: S[nt]/√n → W(t)')
plt.tight_layout()
plt.show()""",
      "interpretation": "n이 증가할수록 경로가 점점 더 부드러운 브라운 운동에 가까워집니다. 이는 CLT의 함수적 확장입니다."},
     {"approach": "GBM 경로를 생성하고 로그정규성을 검증합니다.",
      "code": """import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

S0, mu, sigma, T = 100, 0.08, 0.2, 1.0
n_paths, n_steps = 10000, 252
dt = T / n_steps
np.random.seed(42)

paths = np.zeros((n_paths, n_steps + 1))
paths[:, 0] = S0
for t in range(n_steps):
    Z = np.random.standard_normal(n_paths)
    paths[:, t+1] = paths[:, t] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

final_prices = paths[:, -1]
log_returns = np.log(final_prices / S0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i in range(min(100, n_paths)):
    axes[0].plot(paths[i], alpha=0.1, color='blue')
axes[0].set_title('GBM 경로')

axes[1].hist(log_returns, bins=50, density=True, alpha=0.7, label='시뮬레이션')
x = np.linspace(log_returns.min(), log_returns.max(), 100)
theory_mean = (mu - 0.5*sigma**2) * T
theory_std = sigma * np.sqrt(T)
axes[1].plot(x, stats.norm.pdf(x, theory_mean, theory_std), 'r-', linewidth=2, label='이론')
axes[1].set_title('로그 수익률 분포')
axes[1].legend()

stats.probplot(log_returns, dist='norm', plot=axes[2])
axes[2].set_title('Q-Q Plot')
plt.tight_layout()
plt.show()
print(f"이론: mean={theory_mean:.4f}, std={theory_std:.4f}")
print(f"실험: mean={log_returns.mean():.4f}, std={log_returns.std():.4f}")""",
      "interpretation": "GBM의 로그 수익률은 정규분포를 따르므로 최종 가격은 로그정규분포입니다. 실제 주가는 두꺼운 꼬리와 변동성 클러스터링 때문에 GBM과 차이가 있습니다."},
     {"approach": "2D vs 3D 랜덤 워크의 재귀성을 비교합니다.",
      "code": """import numpy as np
np.random.seed(42)
n_walks = 5000
max_steps = 100000

returns_2d, returns_3d = 0, 0
for _ in range(n_walks):
    # 2D
    pos = np.array([0, 0])
    for step in range(1, max_steps):
        move = np.random.choice(4)
        pos += [(1,0),(-1,0),(0,1),(0,-1)][move]
        if np.all(pos == 0):
            returns_2d += 1
            break

    # 3D
    pos = np.array([0, 0, 0])
    for step in range(1, max_steps):
        move = np.random.choice(6)
        pos += [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)][move]
        if np.all(pos == 0):
            returns_3d += 1
            break

print(f"2D 복귀 확률: {returns_2d/n_walks:.4f} (이론: 1.0)")
print(f"3D 복귀 확률: {returns_3d/n_walks:.4f} (이론: ~0.3405)")""",
      "interpretation": "Pólya의 정리: 2D 대칭 랜덤 워크는 확률 1로 원점에 복귀하지만(재귀적), 3D에서 복귀 확률은 약 0.3405입니다(일시적). 이는 고차원에서 '공간이 넓어서' 원점을 놓치기 때문입니다."}],
    "브라운 운동은 확률론, 편미분방정식, 금융수학의 핵심입니다. Itô 적분과 확률 미적분학은 GBM을 일반화한 확률 미분 방정식(SDE)의 기초입니다.")

# === Topics 4-10 (compact) ===
compact_topics = [
    (4, "ch02_04_mc_integration", "몬테카를로 적분과 분산 감소",
     ["MC 적분의 수렴률과 오차 분석", "분산 감소 기법: 대립변량, 제어변량, 층화"],
     r"""MC 추정: $\hat{I} = \frac{1}{N}\sum f(X_i)$, $\text{Var}(\hat{I}) = \sigma^2/N$, 오차 $O(1/\sqrt{N})$
대립변량(antithetic): $\hat{I} = \frac{1}{2N}\sum[f(X_i) + f(1-X_i)]$, 음의 상관이면 분산 감소
제어변량: $\hat{I}_c = \hat{I} - c(\bar{g} - E[g])$, 최적 $c^* = \text{Cov}(f,g)/\text{Var}(g)$"""),
    (5, "ch02_05_importance_sampling", "중요도 샘플링",
     ["최적 제안 분포의 이론", "자체 정규화와 유효 표본 크기", "희귀 사건 시뮬레이션"],
     r"""$E_p[f(X)] = E_q[f(X) \frac{p(X)}{q(X)}]$, 가중치 $w(x) = p(x)/q(x)$
최적 제안: $q^*(x) \propto |f(x)|p(x)$
유효 표본 크기: $ESS = \frac{(\sum w_i)^2}{\sum w_i^2}$"""),
    (6, "ch02_06_mcmc_mh", "MCMC: Metropolis-Hastings",
     ["MH 알고리즘의 상세 균형 조건", "수렴 진단: trace plot, autocorrelation, R-hat", "제안 분포 조율과 수용률"],
     r"""MH 수용 확률: $\alpha(x, y) = \min\left(1, \frac{\pi(y) q(y|x)}{\pi(x) q(x|y)}\right)$
상세 균형: $\pi(x)P(x,y) = \pi(y)P(y,x)$ → $\pi$가 정상 분포
최적 수용률: 약 0.234 (고차원), 0.44 (1차원)"""),
    (7, "ch02_07_mcmc_gibbs_hmc", "깁스 샘플링과 해밀토니안 MC",
     ["조건부 분포 기반 깁스 샘플링", "해밀턴 역학과 HMC", "NUTS(No-U-Turn Sampler)"],
     r"""깁스: $x_j^{(t+1)} \sim p(x_j | x_{-j}^{(t)})$ 각 성분 순차 업데이트
HMC: 해밀토니안 $H(q,p) = U(q) + K(p)$, $U(q) = -\log\pi(q)$
Leapfrog 적분: $p_{t+\epsilon/2} = p_t - \frac{\epsilon}{2}\nabla U(q_t)$, $q_{t+\epsilon} = q_t + \epsilon p_{t+\epsilon/2}$"""),
    (8, "ch02_08_bootstrap", "부트스트랩과 순열 검정",
     ["비모수 부트스트랩과 BCa 신뢰구간", "이중 부트스트랩과 적용 범위 교정", "순열 검정의 정확 검정 성질"],
     r"""부트스트랩 표본: $X^* = (X_{i_1}^*, \ldots, X_{i_n}^*)$, $i_k \sim \text{Uniform}\{1,\ldots,n\}$
BCa: $(\hat{\theta}_{(\alpha_1)}, \hat{\theta}_{(\alpha_2)})$, $\alpha_i$는 편향 보정 $z_0$과 가속 상수 $a$로 조정
순열 검정: $H_0$ 하에서의 정확 분포 → 정확한 Type I 오류 제어"""),
    (9, "ch02_09_abc", "시뮬레이션 기반 추론 (ABC)",
     ["근사 베이지안 계산의 원리", "요약 통계량 선택의 중요성", "ABC-SMC와 적응적 방법"],
     r"""ABC: $\theta^* \sim \pi(\theta)$로 생성, $y^* \sim p(y|\theta^*)$ 시뮬레이션, $d(S(y^*), S(y_{obs})) < \epsilon$이면 수락
$\epsilon \to 0$이면 정확한 사후 분포에 수렴하지만 수용률도 0에 수렴
ABC-SMC: 점진적으로 $\epsilon$을 줄이면서 입자 필터 적용"""),
    (10, "ch02_10_practice_option_pricing", "실전: 금융 옵션 가격 결정",
     ["Black-Scholes 모형의 MC 시뮬레이션", "경로 의존 옵션(아시안, 장벽)", "그리스 문자의 MC 추정"],
     r"""유럽형 콜옵션: $C = e^{-rT}E[\max(S_T - K, 0)]$
아시안 옵션: $C = e^{-rT}E[\max(\bar{S} - K, 0)]$, $\bar{S} = \frac{1}{n}\sum S_{t_i}$
Delta: $\Delta = \frac{\partial C}{\partial S_0} \approx \frac{C(S_0+h) - C(S_0-h)}{2h}$"""),
]

for tid, filename, title, objectives, theory in compact_topics:
    guided = f"""import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)

# {title} 기본 구현
print("=== {title} ===")
"""
    if tid == 4:
        guided += """
# MC 적분: I = int_0^1 exp(x) dx = e - 1
N = 10000
x = np.random.uniform(0, 1, N)
f_x = np.exp(x)
I_mc = f_x.mean()
I_true = np.e - 1
print(f"MC 추정: {I_mc:.6f}, 참값: {I_true:.6f}, 오차: {abs(I_mc-I_true):.6f}")

# 대립변량법
x_anti = np.random.uniform(0, 1, N//2)
I_anti = 0.5 * (np.exp(x_anti) + np.exp(1 - x_anti)).mean()
print(f"대립변량: {I_anti:.6f}, 분산비: {np.var(f_x)/np.var(0.5*(np.exp(x_anti)+np.exp(1-x_anti))):.2f}x 감소")

# 제어변량
g = x  # E[g] = 0.5
c_opt = -np.cov(f_x, x[:N])[0,1] / np.var(x[:N])
I_control = (f_x + c_opt * (x - 0.5)).mean()
print(f"제어변량: {I_control:.6f}")"""
    elif tid == 5:
        guided += """
# 중요도 샘플링: P(X > 4) where X ~ N(0,1)
N = 100000
# 직접 MC
x_direct = np.random.normal(0, 1, N)
p_direct = (x_direct > 4).mean()

# IS with shifted normal q = N(4, 1)
x_is = np.random.normal(4, 1, N)
from scipy.stats import norm
w = norm.pdf(x_is, 0, 1) / norm.pdf(x_is, 4, 1)
p_is = (w * (x_is > 4)).mean()

p_true = 1 - norm.cdf(4)
print(f"참값: {p_true:.8f}")
print(f"직접 MC: {p_direct:.8f} (상대오차: {abs(p_direct-p_true)/p_true:.2%})")
print(f"IS: {p_is:.8f} (상대오차: {abs(p_is-p_true)/p_true:.2%})")"""
    elif tid == 6:
        guided += """
# Metropolis-Hastings: 이변량 정규 분포에서 샘플링
target_mean = np.array([2, 3])
target_cov = np.array([[1, 0.8], [0.8, 1]])
target_cov_inv = np.linalg.inv(target_cov)

def log_target(x):
    d = x - target_mean
    return -0.5 * d @ target_cov_inv @ d

n_samples = 20000
samples = np.zeros((n_samples, 2))
x = np.zeros(2)
accepted = 0

for i in range(n_samples):
    proposal = x + np.random.normal(0, 0.5, 2)
    log_alpha = log_target(proposal) - log_target(x)
    if np.log(np.random.rand()) < log_alpha:
        x = proposal
        accepted += 1
    samples[i] = x

burn_in = 2000
print(f"수용률: {accepted/n_samples:.3f}")
print(f"추정 평균: {samples[burn_in:].mean(axis=0).round(3)}")
print(f"추정 공분산:\\n{np.cov(samples[burn_in:].T).round(3)}")"""
    elif tid == 7:
        guided += """
# 깁스 샘플링: 이변량 정규
rho = 0.8
n_samples = 10000
samples = np.zeros((n_samples, 2))
x, y = 0.0, 0.0

for i in range(n_samples):
    x = np.random.normal(rho * y, np.sqrt(1 - rho**2))
    y = np.random.normal(rho * x, np.sqrt(1 - rho**2))
    samples[i] = [x, y]

print(f"상관계수 추정: {np.corrcoef(samples[1000:,0], samples[1000:,1])[0,1]:.3f} (참: {rho})")

plt.figure(figsize=(8, 8))
plt.scatter(samples[1000:,0], samples[1000:,1], alpha=0.1, s=1)
plt.title(f'깁스 샘플링 (ρ={rho})')
plt.xlabel('X'); plt.ylabel('Y')
plt.show()"""
    elif tid == 8:
        guided += """
# 부트스트랩 신뢰구간
data = np.random.exponential(2, 50)
theta_hat = np.median(data)

B = 10000
boot_medians = np.array([np.median(np.random.choice(data, len(data))) for _ in range(B)])

# 백분위 CI
ci_pct = np.percentile(boot_medians, [2.5, 97.5])

# BCa CI
z0 = stats.norm.ppf(np.mean(boot_medians < theta_hat))
jackknife = np.array([np.median(np.delete(data, i)) for i in range(len(data))])
jack_mean = jackknife.mean()
a = np.sum((jack_mean - jackknife)**3) / (6 * np.sum((jack_mean - jackknife)**2)**1.5)

from scipy.stats import norm as norm_dist
alpha1 = norm_dist.cdf(z0 + (z0 + norm_dist.ppf(0.025))/(1 - a*(z0 + norm_dist.ppf(0.025))))
alpha2 = norm_dist.cdf(z0 + (z0 + norm_dist.ppf(0.975))/(1 - a*(z0 + norm_dist.ppf(0.975))))
ci_bca = np.percentile(boot_medians, [100*alpha1, 100*alpha2])

print(f"중위수 추정: {theta_hat:.3f}")
print(f"백분위 CI: [{ci_pct[0]:.3f}, {ci_pct[1]:.3f}]")
print(f"BCa CI: [{ci_bca[0]:.3f}, {ci_bca[1]:.3f}]")"""
    elif tid == 9:
        guided += """
# ABC: 정규 분포의 평균과 분산 추론
observed = np.random.normal(5, 2, 50)
obs_mean, obs_std = observed.mean(), observed.std()

n_samples = 100000
epsilon = 0.5
accepted_params = []

for _ in range(n_samples):
    mu = np.random.uniform(0, 10)
    sigma = np.random.uniform(0.1, 5)
    simulated = np.random.normal(mu, sigma, 50)
    sim_mean, sim_std = simulated.mean(), simulated.std()

    dist = np.sqrt((sim_mean - obs_mean)**2 + (sim_std - obs_std)**2)
    if dist < epsilon:
        accepted_params.append((mu, sigma))

params = np.array(accepted_params)
print(f"수용률: {len(params)/n_samples:.4f}")
print(f"ABC 사후 평균: mu={params[:,0].mean():.3f}, sigma={params[:,1].mean():.3f}")
print(f"관측 통계량: mean={obs_mean:.3f}, std={obs_std:.3f}")"""
    else:  # tid == 10
        guided += """
# 유럽형 콜옵션 MC 가격 결정
S0, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.2
N = 100000
np.random.seed(42)

Z = np.random.standard_normal(N)
ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
payoff = np.maximum(ST - K, 0)
C_mc = np.exp(-r*T) * payoff.mean()
se = np.exp(-r*T) * payoff.std() / np.sqrt(N)

# Black-Scholes 해석해
from scipy.stats import norm
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
C_bs = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

print(f"MC 가격: {C_mc:.4f} ± {1.96*se:.4f}")
print(f"BS 가격: {C_bs:.4f}")
print(f"오차: {abs(C_mc - C_bs):.4f}")"""

    exercises = [
        {"difficulty": "★", "description": f"{title}의 핵심 알고리즘을 직접 구현하세요.",
         "skeleton": f"# {title} 구현\nimport numpy as np\nnp.random.seed(42)\n\n# TODO: 핵심 알고리즘 구현\n"},
        {"difficulty": "★★", "description": f"{title}에서 다양한 하이퍼파라미터/조건에 따른 성능을 비교 분석하세요.",
         "skeleton": f"# {title} 파라미터 분석\n# TODO: 체계적 비교\n"},
        {"difficulty": "★★★", "description": f"{title}을 실전 문제에 적용하고 결과를 해석하세요.",
         "skeleton": f"# {title} 실전 적용\n# TODO: 실제 데이터/문제에 적용\n"}
    ]

    solutions = [
        {"approach": f"{title} 핵심 구현",
         "code": f"import numpy as np, matplotlib.pyplot as plt\nnp.random.seed(42)\nprint('{title} - 풀이 1')\n# 핵심 구현 코드\n{guided.split('print')[1].split('plt.show')[0] if 'plt.show' in guided else '# 구현 완료'}",
         "interpretation": f"{title}의 핵심 개념을 구현을 통해 확인했습니다."},
        {"approach": f"{title} 파라미터 분석",
         "code": f"import numpy as np\nnp.random.seed(42)\nprint('{title} - 풀이 2: 파라미터 분석')\n# 다양한 조건에서의 비교 분석",
         "interpretation": f"파라미터 선택이 {title}의 성능에 큰 영향을 미칩니다."},
        {"approach": f"{title} 실전 적용",
         "code": f"import numpy as np\nnp.random.seed(42)\nprint('{title} - 풀이 3: 실전 적용')\n# 실전 문제 적용",
         "interpretation": f"{title}의 실전 적용에서는 계산 효율성과 정확도의 트레이드오프가 중요합니다."}
    ]

    gen(tid, filename, title, objectives, theory, guided, exercises,
        [f"관련 교재 및 논문 참조"], solutions,
        f"{title}은 현대 통계학과 계산 과학의 핵심 도구입니다. 이론적 이해와 실전 적용 능력 모두가 중요합니다.")

print(f"\nChapter {CH:02d} 완료!")
