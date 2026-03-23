# ADP (Advanced Data Practice) - 데이터 분석 심화 학습 프로젝트

## 프로젝트 개요

수학·물리학 배경과 통계적 기반을 갖춘 학습자를 위한 **도전적 데이터 분석 학습 프로젝트**.
기초를 건너뛰고, 이론의 수리적 기반부터 실전 분석까지를 아우르는 12개 챕터 × 10+ 세부 노트북으로 구성.

- **대상**: 수학/물리 전공자, 통계 기초 보유자
- **난이도**: 중급~고급 (수식 유도, 알고리즘 구현, 실전 데이터 분석)
- **형식**: 각 세부 항목마다 `문제 노트북(.ipynb)` + `모범답안 노트북(_solution.ipynb)` 쌍으로 제공

---

## 디렉토리 구조

```
adp/
├── AGENTS.md                    # 이 파일 (프로젝트 계획)
├── README.md                    # 프로젝트 소개
├── requirements.txt             # 공통 패키지
├── data/                        # 공유 데이터셋
│   ├── README.md
│   └── ...
├── utils/                       # 공유 유틸리티
│   └── helpers.py
│
├── ch01_advanced_eda/           # 챕터 01
│   ├── README.md                # 챕터 개요
│   ├── ch01_01_multivariate_visualization.ipynb
│   ├── ch01_01_multivariate_visualization_solution.ipynb
│   ├── ch01_02_missing_data.ipynb
│   ├── ch01_02_missing_data_solution.ipynb
│   └── ...
│
├── ch02_stochastic_montecarlo/  # 챕터 02
│   └── ...
│
└── ... (ch03 ~ ch12)
```

---

## 챕터 구성 상세

---

### Chapter 01: 고급 탐색적 데이터 분석과 데이터 전처리
**디렉토리**: `ch01_advanced_eda/`
**핵심 키워드**: 다변량 분석, 결측값 이론, 이상치 탐지, 피처 엔지니어링, 대용량 처리

| # | 파일명 | 주제 | 핵심 내용 |
|---|--------|------|-----------|
| 01 | `ch01_01_multivariate_visualization` | 다변량 데이터 시각화와 차원 해석 | 평행좌표, 앤드류스 곡선, 조건부 플롯, 다차원 스케일링 시각화 |
| 02 | `ch01_02_missing_data` | 결측값 메커니즘과 다중 대치법 | MCAR/MAR/MNAR 판별, Little's test, MICE, KNN 대치, 민감도 분석 |
| 03 | `ch01_03_outlier_statistical` | 이상치 탐지 - 통계적 방법 | Mahalanobis 거리, Grubbs 검정, ESD, 다변량 이상치 |
| 04 | `ch01_04_outlier_ml` | 이상치 탐지 - 기계학습 기반 | Isolation Forest, LOF, One-Class SVM, Autoencoder 기반 탐지 |
| 05 | `ch01_05_transformations` | 데이터 변환과 정규화 이론 | Box-Cox, Yeo-Johnson, Robust Scaling, 안정화 변환의 수학 |
| 06 | `ch01_06_encoding` | 범주형 인코딩 고급 기법 | Target Encoding, WoE/IV, CatBoost Encoding, 정보 누출 방지 |
| 07 | `ch01_07_feature_engineering` | 피처 엔지니어링과 도메인 지식 | 상호작용 항, 다항식 특성, 도메인 기반 변환, 자동 피처 생성 |
| 08 | `ch01_08_data_quality` | 데이터 품질 프레임워크 | 데이터 드리프트 탐지, 스키마 검증, 통계적 프로파일링, 자동 검증 파이프라인 |
| 09 | `ch01_09_large_scale` | 대용량 데이터 처리 | Polars vs Pandas 벤치마크, 청크 처리, 메모리 최적화, 지연 평가 |
| 10 | `ch01_10_practice_financial_eda` | 실전: 금융 거래 데이터 종합 분석 | 종합 EDA 파이프라인, 이상 거래 탐지, 리포트 자동 생성 |

---

### Chapter 02: 확률 과정과 몬테카를로 시뮬레이션
**디렉토리**: `ch02_stochastic_montecarlo/`
**핵심 키워드**: 마르코프 체인, 브라운 운동, MC 적분, MCMC, 시뮬레이션 기반 추론

| # | 파일명 | 주제 | 핵심 내용 |
|---|--------|------|-----------|
| 01 | `ch02_01_markov_chains` | 이산 마르코프 체인 | 전이 행렬, 정상 분포, 에르고딕 정리, 흡수 체인, PageRank |
| 02 | `ch02_02_continuous_markov` | 연속 시간 마르코프 체인과 포아송 과정 | 생성 행렬, Kolmogorov 방정식, 비동질 포아송, 큐잉 이론 |
| 03 | `ch02_03_random_walks` | 랜덤 워크와 브라운 운동 | 1D/2D 랜덤 워크, 재귀 성질, Wiener 과정, 기하 브라운 운동 |
| 04 | `ch02_04_mc_integration` | 몬테카를로 적분과 분산 감소 | 층화 샘플링, 대립 변량법, 제어 변량법, 수렴 속도 분석 |
| 05 | `ch02_05_importance_sampling` | 중요도 샘플링 | 최적 제안 분포, 자체 정규화, 유효 표본 크기, 희귀 사건 시뮬레이션 |
| 06 | `ch02_06_mcmc_mh` | MCMC: Metropolis-Hastings | 알고리즘 구현, 수렴 진단, 제안 분포 조율, 적응적 MCMC |
| 07 | `ch02_07_mcmc_gibbs_hmc` | MCMC: 깁스 샘플링과 해밀토니안 MC | 조건부 샘플링, 해밀턴 역학, NUTS, 수렴 비교 |
| 08 | `ch02_08_bootstrap` | 부트스트랩과 순열 검정 | 비모수 부트스트랩, 신뢰구간 (BCa), 이중 부트스트랩, 순열의 정확 검정 |
| 09 | `ch02_09_abc` | 시뮬레이션 기반 추론 (ABC) | 근사 베이지안 계산, 요약 통계량 선택, ABC-SMC |
| 10 | `ch02_10_practice_option_pricing` | 실전: 금융 옵션 가격 결정 | Black-Scholes MC, 아시안 옵션, 그리스 문자 추정, 분산 감소 적용 |

---

### Chapter 03: 고급 통계 추론
**디렉토리**: `ch03_advanced_inference/`
**핵심 키워드**: MLE, 충분통계량, 검정 이론, 다중 비교, EM 알고리즘, 로버스트 통계

| # | 파일명 | 주제 | 핵심 내용 |
|---|--------|------|-----------|
| 01 | `ch03_01_mle_fisher` | 최대우도추정과 Fisher 정보량 | MLE 점근 이론, 관측/기대 정보량, 프로파일 우도, 다변량 MLE |
| 02 | `ch03_02_sufficiency_cramer_rao` | 충분통계량과 Cramér-Rao 하한 | 네이만 인수분해, 완비 충분통계량, UMVUE, 효율성 |
| 03 | `ch03_03_hypothesis_testing` | 우도비 검정과 Wald 검정 | 네이만-피어슨 보조정리, 스코어 검정, 점근 동치성, 합집합-교집합 검정 |
| 04 | `ch03_04_multiple_testing` | 다중 검정과 FDR 제어 | FWER (Bonferroni, Holm), FDR (BH, BY), q-value, 적응적 절차 |
| 05 | `ch03_05_nonparametric` | 비모수 검정 심화 | 순위 검정, Kolmogorov-Smirnov, 커널 밀도 추정, 순위 기반 회귀 |
| 06 | `ch03_06_empirical_bayes` | 경험적 베이즈 방법 | James-Stein 추정, 축소 추정, 대규모 동시 추론, Robbins' formula |
| 07 | `ch03_07_em_algorithm` | EM 알고리즘의 이론과 응용 | 수렴 증명, 혼합 모형, 불완전 데이터, 변분 EM |
| 08 | `ch03_08_robust_statistics` | 로버스트 통계 | M-추정, 영향 함수, 붕괴점, Huber 손실, 중위수 기반 방법 |
| 09 | `ch03_09_nonparametric_regression` | 함수 추정과 비모수 회귀 | Nadaraya-Watson, 국소 다항식, 스플라인, 평활 매개변수 선택 |
| 10 | `ch03_10_practice_clinical` | 실전: 임상시험 데이터 분석 | 적응적 설계, 중간 분석, 생존 엔드포인트, 비열등성 검정 |

---

### Chapter 04: 실험 설계와 A/B 테스트
**디렉토리**: `ch04_experiment_design/`
**핵심 키워드**: 검정력, 순차 검정, CUPED, 밴딧, 네트워크 효과, HTE

| # | 파일명 | 주제 | 핵심 내용 |
|---|--------|------|-----------|
| 01 | `ch04_01_power_analysis` | 검정력 분석과 표본 크기 설계 | 효과 크기, 비율/평균 검정, 비열등성 설계, 다변량 검정력 |
| 02 | `ch04_02_sequential_testing` | 순차 검정과 조기 중단 | SPRT, alpha-spending, O'Brien-Fleming, 그룹 순차 설계 |
| 03 | `ch04_03_cuped` | CUPED와 분산 감소 기법 | 공변량 조정, 사전-사후 설계, 층화, CUPAC |
| 04 | `ch04_04_factorial_design` | 다변량 실험 설계 | 완전/분할 요인 설계, 교호작용, 반응 표면법, 최적 설계 |
| 05 | `ch04_05_bandits` | 멀티암 밴딧과 적응적 실험 | Thompson Sampling, UCB, 문맥 밴딧, 베이지안 최적화 |
| 06 | `ch04_06_interference` | 네트워크 간섭과 클러스터 랜덤화 | SUTVA 위반, 유출 효과, 클러스터 설계, 이분 설계 |
| 07 | `ch04_07_hte` | 이질적 처리 효과 추정 | CATE, Causal Forest, Meta-learners (S/T/X), 개인화 |
| 08 | `ch04_08_bayesian_ab` | 베이지안 A/B 테스트 | 승률 계산, 기대 손실, 사전 분포 설계, 조기 종료 |
| 09 | `ch04_09_always_valid` | 연속 모니터링과 항시 유효한 추론 | 항시 유효 p-value, 혼합 순차비, e-value, 신뢰 시퀀스 |
| 10 | `ch04_10_practice_platform` | 실전: 플랫폼 A/B 테스트 파이프라인 | 메트릭 설계, 자동화, 의사결정 프레임워크, 과두 제어 |

---

### Chapter 05: 회귀 분석 심화
**디렉토리**: `ch05_advanced_regression/`
**핵심 키워드**: GLM, 정칙화, 분위수 회귀, GAM, 혼합 효과, 제로 팽창

| # | 파일명 | 주제 | 핵심 내용 |
|---|--------|------|-----------|
| 01 | `ch05_01_glm` | 일반화 선형 모형 이론 | 지수 족, 링크 함수, 편차, 과분산, 준우도 |
| 02 | `ch05_02_regularization` | 정칙화 회귀와 편향-분산 | Ridge/Lasso/Elastic Net, 기하학적 해석, 경로 알고리즘, 선택 일관성 |
| 03 | `ch05_03_quantile_regression` | 분위수 회귀와 기대 손실 | 체크 함수, 선형 계획 풀이, 조건부 분포 추정, 비대칭 손실 |
| 04 | `ch05_04_gam` | 일반화 가법 모형 | 스플라인 기저, 페널티, 텐서 곱, 모형 선택, pyGAM 활용 |
| 05 | `ch05_05_mixed_effects` | 혼합 효과 모형 | 고정/랜덤 효과, REML, 교차 랜덤 효과, ICC, 설계 효과 |
| 06 | `ch05_06_iv_2sls` | 도구 변수와 2SLS | 내생성, 약한 도구 변수, Sargan 검정, 이분산 하에서의 추론 |
| 07 | `ch05_07_logistic_advanced` | 로지스틱 회귀 심화 | 다범주 (Multinomial/Ordinal), 완전 분리, Firth 보정, 교정 곡선 |
| 08 | `ch05_08_survival_regression` | 생존 회귀 모형 | AFT 모형, Cox PH, 시간 의존 공변량, concordance index |
| 09 | `ch05_09_zero_inflated` | 제로 팽창 모형과 과분산 | ZIP/ZINB, Hurdle 모형, 과분산 검정, 모형 비교 |
| 10 | `ch05_10_practice_realestate` | 실전: 부동산 가격 모델링 | 공간 자기상관 처리, 비선형 효과, 해석 가능한 모형 구축 |

---

### Chapter 06: 시계열 분석과 예측
**디렉토리**: `ch06_time_series/`
**핵심 키워드**: ARIMA, VAR, 상태 공간, GARCH, 스펙트럼, 변환점 탐지

| # | 파일명 | 주제 | 핵심 내용 |
|---|--------|------|-----------|
| 01 | `ch06_01_stationarity` | 정상성과 단위근 검정 | ADF, PP, KPSS, 구조적 단절 하에서의 검정, 계절 단위근 |
| 02 | `ch06_02_arima` | ARIMA/SARIMA 모델링 | ACF/PACF, 모형 식별, 자동 선택 (auto_arima), 진단 검정 |
| 03 | `ch06_03_var` | 벡터 자기회귀와 그레인저 인과 | VAR 추정, 충격 반응 함수, 분산 분해, 공적분 (Johansen) |
| 04 | `ch06_04_state_space` | 상태 공간 모형과 칼만 필터 | 관측/상태 방정식, 칼만 재귀, 평활, 누락 관측 처리 |
| 05 | `ch06_05_garch` | GARCH와 변동성 모형 | ARCH 효과 검정, GARCH(1,1), EGARCH, GJR-GARCH, 다변량 GARCH |
| 06 | `ch06_06_spectral` | 스펙트럼 분석과 웨이블릿 | 주기도, Welch 방법, 코히어런스, 이산 웨이블릿 변환, 시간-주파수 분석 |
| 07 | `ch06_07_structural` | 구조적 시계열 모형 | 로컬 레벨/추세, 계절 분해, Harvey 모형, 회귀 성분, Prophet 내부 |
| 08 | `ch06_08_changepoint` | 변환점 탐지 | CUSUM, PELT, 베이지안 온라인 변환점, 구조적 변화 검정 (Bai-Perron) |
| 09 | `ch06_09_ts_cv` | 시계열 교차 검증과 평가 | 확장 윈도우, 슬라이딩 윈도우, 확률적 예측 평가 (CRPS), 예측 조합 |
| 10 | `ch06_10_practice_energy` | 실전: 에너지 수요 예측 | 다중 계절성, 외생 변수, 앙상블 예측, 확률적 예측 |

---

### Chapter 07: 베이지안 데이터 분석
**디렉토리**: `ch07_bayesian/`
**핵심 키워드**: 사전 분포, 계층 모형, PyMC, 모형 비교, 가우시안 과정

| # | 파일명 | 주제 | 핵심 내용 |
|---|--------|------|-----------|
| 01 | `ch07_01_bayes_foundations` | 베이즈 정리와 사전 분포 | 주관적/객관적 사전, Jeffrey's prior, 약한 정보 사전, 사전 예측 검정 |
| 02 | `ch07_02_conjugate` | 켤레 사전과 해석적 사후 | Beta-Binomial, Normal-Normal, 충분통계량, 순차 갱신 |
| 03 | `ch07_03_pymc_basics` | PyMC 베이지안 모델링 | 모형 명세, 사후 샘플링, 수렴 진단 (R̂, ESS), 사후 요약 |
| 04 | `ch07_04_hierarchical` | 계층적 베이지안 모형 | 부분 풀링, 초사전분포, 비중심화 매개변수화, 다수준 모형 |
| 05 | `ch07_05_model_comparison` | 베이지안 모형 비교 | WAIC, LOO-CV, 베이즈 팩터, 사후 예측 검정 |
| 06 | `ch07_06_diagnostics` | 사후 예측 검정과 모형 진단 | 적합도 검정, 잔차 분석, 예측 점검, 교정, ArviZ 활용 |
| 07 | `ch07_07_bayesian_regression` | 베이지안 회귀와 변수 선택 | 스파이크-앤-슬래브, Horseshoe prior, 베이지안 LASSO, 모형 평균 |
| 08 | `ch07_08_gaussian_process` | 가우시안 과정 회귀 | 공분산 함수, 초매개변수 최적화, 근사 GP, 분류 확장 |
| 09 | `ch07_09_nonparametric_bayes` | 베이지안 비모수 (DP Mixture) | 디리클레 과정, 스틱-브레이킹, 중국 식당 과정, DP 혼합 모형 |
| 10 | `ch07_10_practice_sports` | 실전: 스포츠 베이지안 순위 | Bradley-Terry, Elo의 베이지안 확장, 동적 모형, 예측 시장 |

---

### Chapter 08: 비지도 학습과 차원 축소
**디렉토리**: `ch08_unsupervised/`
**핵심 키워드**: PCA, t-SNE, UMAP, NMF, GMM, 스펙트럼 클러스터링, 유효성 검증

| # | 파일명 | 주제 | 핵심 내용 |
|---|--------|------|-----------|
| 01 | `ch08_01_pca` | PCA: 스펙트럼 정리와 최적성 | 고유값 분해, 설명 분산, 스크리 플롯, 바이플롯, 확률적 PCA |
| 02 | `ch08_02_kernel_pca` | 커널 PCA와 비선형 확장 | 커널 트릭, RBF/다항식 커널, 사전 이미지 문제, MDS와의 관계 |
| 03 | `ch08_03_tsne` | t-SNE 이론과 실전 | KL 발산, perplexity, 과밀/과소, Barnes-Hut 근사, 해석 주의점 |
| 04 | `ch08_04_umap` | UMAP: 위상적 관점 | 리만 다양체, 퍼지 단체 집합, 최적화, t-SNE 대비 장단점 |
| 05 | `ch08_05_nmf` | 비음수 행렬 분해 | 곱셈 업데이트, 스파스 NMF, 토픽 모델링 응용, 이미지 분해 |
| 06 | `ch08_06_gmm` | 가우시안 혼합 모형과 EM | EM 유도, 초기화 전략, 모형 선택 (BIC/AIC), 정규화 |
| 07 | `ch08_07_spectral_clustering` | 스펙트럼 클러스터링 | 그래프 라플라시안, 정규화, NCut, 근사 알고리즘 |
| 08 | `ch08_08_density_clustering` | 밀도 기반 클러스터링 | DBSCAN, HDBSCAN, OPTICS, 상호 도달 가능 거리 |
| 09 | `ch08_09_cluster_validation` | 클러스터 유효성 검증 | 실루엣, 칼린스키-하라바즈, 안정성 기반, 갭 통계량 |
| 10 | `ch08_10_practice_segmentation` | 실전: 고객 세분화 | RFM 분석, 행동 클러스터링, 세그먼트 프로파일링, 시각화 대시보드 |

---

### Chapter 09: 인과 추론
**디렉토리**: `ch09_causal_inference/`
**핵심 키워드**: 잠재 결과, DAG, 성향 점수, DiD, RDD, IV, 합성 통제

| # | 파일명 | 주제 | 핵심 내용 |
|---|--------|------|-----------|
| 01 | `ch09_01_potential_outcomes` | 잠재 결과 프레임워크 | Rubin 인과 모형, ATE/ATT/ATU, 무작위 배정, 식별 가정 |
| 02 | `ch09_02_dags` | DAG와 do-계산 | 구조적 인과 모형, d-분리, 백도어/프론트도어 기준, do-연산 |
| 03 | `ch09_03_propensity_score` | 성향 점수 매칭과 가중 | PS 추정, 매칭 (NN, 캘리퍼), IPW, 균형 검정, 겹침 가중치 |
| 04 | `ch09_04_did` | 이중 차분법 (DiD) | 평행 추세 가정, 시차 DiD, 이질적 처리 시점, Callaway-Sant'Anna |
| 05 | `ch09_05_rdd` | 회귀 불연속 설계 | Sharp/Fuzzy RDD, 대역폭 선택, 국소 다항식, McCrary 검정 |
| 06 | `ch09_06_iv` | 도구 변수법 | 제외 제한, 약한 도구 변수, 2SLS, LATE 해석, Anderson-Rubin 검정 |
| 07 | `ch09_07_synthetic_control` | 합성 통제법 | 가중치 추정, 추론 (placebo), 일반화 합성 통제, 행렬 완성 |
| 08 | `ch09_08_doubly_robust` | 이중 견고 추정 | AIPW, TMLE, 교차 적합, 기계학습과 인과 추론의 결합 |
| 09 | `ch09_09_mediation` | 매개 분석과 경로 분석 | 직접/간접 효과, Baron-Kenny, 인과적 매개 분석, 민감도 |
| 10 | `ch09_10_practice_policy` | 실전: 정책 효과 평가 | 최저임금과 고용, 다양한 식별 전략 비교, 로버스트 분석 |

---

### Chapter 10: 생존 분석과 이벤트 히스토리
**디렉토리**: `ch10_survival/`
**핵심 키워드**: 위험 함수, Kaplan-Meier, Cox PH, 경쟁 위험, 허약 모형, 치유 모형

| # | 파일명 | 주제 | 핵심 내용 |
|---|--------|------|-----------|
| 01 | `ch10_01_survival_functions` | 생존 함수와 위험 함수 | S(t), h(t), H(t) 관계, 우측/좌측/구간 중도절단, 절단 |
| 02 | `ch10_02_kaplan_meier` | Kaplan-Meier와 로그순위 검정 | 곱-극한 추정, 그린우드 공식, 로그순위/Wilcoxon, 층화 검정 |
| 03 | `ch10_03_cox_ph` | Cox 비례위험 모형 | 부분 우도, Breslow 추정, 비례위험 가정 검정, Schoenfeld 잔차 |
| 04 | `ch10_04_time_varying` | 시간 의존 공변량 | 확장 Cox, 계수 기간 형식, 시간 변환 변수, 랜드마크 분석 |
| 05 | `ch10_05_parametric` | 모수적 생존 모형 | Exponential, Weibull, Log-logistic, Log-normal, AIC 비교 |
| 06 | `ch10_06_competing_risks` | 경쟁 위험과 Fine-Gray | 원인별 위험, 부분분포 위험, 누적 발생 함수, 다상태 모형 |
| 07 | `ch10_07_frailty` | 허약 모형 | Gamma/Log-normal frailty, 공유 허약, 이질성, EM 추정 |
| 08 | `ch10_08_truncation` | 구간 중도절단과 좌절단 | 구간 중도절단 MLE, Turnbull 추정, 좌절단 보정, 유병/발생 |
| 09 | `ch10_09_cure_models` | 치유 모형 | 혼합 치유 모형, 비혼합 치유 모형, EM 알고리즘, 충분 추적 |
| 10 | `ch10_10_practice_churn` | 실전: 고객 이탈과 생애 가치 | CLV 추정, 이탈 예측, 동적 생존 모형, 비즈니스 의사결정 |

---

### Chapter 11: 최적화와 의사결정 분석
**디렉토리**: `ch11_optimization/`
**핵심 키워드**: 볼록 최적화, LP/IP, 경사 하강, 포트폴리오, 강화학습, 동적 프로그래밍

| # | 파일명 | 주제 | 핵심 내용 |
|---|--------|------|-----------|
| 01 | `ch11_01_convex_optimization` | 볼록 최적화와 KKT 조건 | 볼록 집합/함수, Slater 조건, 쌍대 이론, KKT 필요충분 조건 |
| 02 | `ch11_02_linear_programming` | 선형 계획법과 쌍대성 | 심플렉스, 내점법, 쌍대 문제, 감도 분석, SciPy/PuLP 활용 |
| 03 | `ch11_03_integer_programming` | 정수 계획법과 조합 최적화 | Branch-and-Bound, 이동 세일즈맨, 스케줄링, 근사 알고리즘 |
| 04 | `ch11_04_gradient_methods` | 경사 하강법과 변형 | GD, SGD, Momentum, Adam, L-BFGS, 수렴률 분석, 학습률 스케줄 |
| 05 | `ch11_05_constrained` | 제약 최적화와 라그랑주 | 증강 라그랑주, 벌칙법, 장벽법, ADMM |
| 06 | `ch11_06_decision_theory` | 의사결정 이론과 효용 함수 | 기대효용, 위험 회피, 전망 이론, 정보의 가치, 의사결정 나무 |
| 07 | `ch11_07_portfolio` | 포트폴리오 최적화 | Markowitz, 효율적 프론티어, Black-Litterman, 리스크 패리티 |
| 08 | `ch11_08_rl_bandits` | 강화학습 기초: 밴딧과 MDP | Bellman 방정식, 가치/정책 반복, Q-learning, 탐색-이용 |
| 09 | `ch11_09_dynamic_programming` | 동적 프로그래밍과 최적 정지 | 비서 문제, 최적 정지 정리, 미국식 옵션, 후진 유도법 |
| 10 | `ch11_10_practice_supply_chain` | 실전: 공급망 최적화 | 재고 관리, 수송 문제, 수요 불확실성 하 최적화, 시뮬레이션 최적화 |

---

### Chapter 12: 공간 통계와 네트워크 분석
**디렉토리**: `ch12_spatial_network/`
**핵심 키워드**: 공간 자기상관, 크리깅, 점 과정, 그래프, 커뮤니티, 시공간

| # | 파일명 | 주제 | 핵심 내용 |
|---|--------|------|-----------|
| 01 | `ch12_01_spatial_autocorrelation` | 공간 자기상관과 Moran's I | 공간 가중 행렬, 전역/국소 Moran's I, Geary's C, LISA |
| 02 | `ch12_02_kriging` | 크리깅과 지구통계 | 변이도, 보통/범용 크리깅, 크리깅 분산, 교차 검증 |
| 03 | `ch12_03_point_processes` | 점 과정과 공간 밀도 | 동질/비동질 포아송, K-함수, 커널 밀도 추정, 모형 적합 |
| 04 | `ch12_04_gwr` | 지리가중 회귀 | 대역폭 선택, 계수 표면, 다중공선성, MGWR |
| 05 | `ch12_05_graph_theory` | 그래프 이론과 네트워크 측도 | 인접 행렬, 경로, 차수 분포, 소세계 네트워크, 무척도 네트워크 |
| 06 | `ch12_06_community_detection` | 커뮤니티 탐지 | 모듈성, Louvain/Leiden, 라벨 전파, 확률적 블록 모형 |
| 07 | `ch12_07_centrality` | 중심성과 영향력 전파 | Degree, Betweenness, Eigenvector, PageRank, 확산 모형 (SIR/SIS) |
| 08 | `ch12_08_link_prediction` | 링크 예측과 추천 | 유사도 기반, 행렬 인수분해, GNN 기초, 협업 필터링 |
| 09 | `ch12_09_spatiotemporal` | 시공간 데이터 분석 | 시공간 자기상관, 궤적 분석, 시공간 모형, 동적 네트워크 |
| 10 | `ch12_10_practice_epidemic` | 실전: 전염병 확산 모델링 | SIR/SEIR 모형, 네트워크 기반 확산, 공간 확산, 개입 효과 |

---

## 구현 전략

### 노트북 구조 (문제 파일)
각 `.ipynb` 파일은 다음 구조를 따릅니다:
1. **제목과 학습 목표** - 이 노트북에서 다루는 내용과 목표
2. **이론적 배경** - 핵심 수학적 기반 (수식 포함, 간결하게)
3. **구현 가이드** - 핵심 알고리즘의 스켈레톤 코드
4. **연습 문제** (3~5개) - 난이도별 분류 (★~★★★)
   - 구현 문제: 알고리즘을 직접 구현
   - 분석 문제: 실제 데이터에 적용하여 인사이트 도출
   - 도전 문제: 확장/변형/비교 과제
5. **참고 자료** - 논문, 교재, 온라인 리소스 링크

### 노트북 구조 (모범답안 파일)
각 `_solution.ipynb` 파일은 다음 구조를 따릅니다:
1. **풀이 요약** - 접근법 개요
2. **상세 풀이** - 모든 문제에 대한 완전한 코드와 해설
3. **결과 해석** - 분석 결과의 의미와 주의점
4. **확장 토론** - 더 알아볼 주제, 한계점

### 데이터 전략
- 공개 데이터셋 활용 (UCI, Kaggle, scikit-learn)
- 시뮬레이션 데이터 생성 코드 포함
- `data/` 디렉토리에 공유 데이터 배치

### 패키지 의존성
```
numpy, scipy, pandas, polars
matplotlib, seaborn, plotly
scikit-learn, statsmodels
pymc, arviz
lifelines, causalinference
networkx, geopandas
cvxpy, pulp
```

---

## 구현 순서

1. **Phase 1**: 프로젝트 골격 생성 (디렉토리, README, requirements.txt, utils)
2. **Phase 2**: Chapter 01~04 노트북 생성 (기초→추론→실험)
3. **Phase 3**: Chapter 05~08 노트북 생성 (회귀→시계열→베이지안→비지도)
4. **Phase 4**: Chapter 09~12 노트북 생성 (인과→생존→최적화→공간/네트워크)
5. **Phase 5**: 모범답안 생성 (각 Phase와 병행)
6. **Phase 6**: 최종 검수 및 README 업데이트

---

## 품질 기준

- [ ] 모든 노트북이 수학 표기(LaTeX)를 올바르게 사용
- [ ] 코드가 실행 가능한 상태 (import, 데이터 로드 포함)
- [ ] 문제 난이도가 ★~★★★으로 명확히 분류
- [ ] 모범답안이 충분한 해설을 포함
- [ ] 각 챕터의 실전 문제가 현실적 시나리오 기반
