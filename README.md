# ADP — Advanced Data Practice

수학·물리학 배경과 통계적 기반을 갖춘 학습자를 위한 **도전적 데이터 분석 심화 학습 프로젝트**

## 프로젝트 구성

**12개 챕터 × 10개 항목 × 2(문제+답안) = 240개 Jupyter 노트북**

| 챕터 | 디렉토리 | 주제 |
|------|----------|------|
| 01 | `ch01_advanced_eda/` | 고급 탐색적 데이터 분석과 데이터 전처리 |
| 02 | `ch02_stochastic_montecarlo/` | 확률 과정과 몬테카를로 시뮬레이션 |
| 03 | `ch03_advanced_inference/` | 고급 통계 추론 |
| 04 | `ch04_experiment_design/` | 실험 설계와 A/B 테스트 |
| 05 | `ch05_advanced_regression/` | 회귀 분석 심화 |
| 06 | `ch06_time_series/` | 시계열 분석과 예측 |
| 07 | `ch07_bayesian/` | 베이지안 데이터 분석 |
| 08 | `ch08_unsupervised/` | 비지도 학습과 차원 축소 |
| 09 | `ch09_causal_inference/` | 인과 추론 |
| 10 | `ch10_survival/` | 생존 분석과 이벤트 히스토리 |
| 11 | `ch11_optimization/` | 최적화와 의사결정 분석 |
| 12 | `ch12_spatial_network/` | 공간 통계와 네트워크 분석 |

## 파일 구조

각 챕터 디렉토리에는 10개의 문제-답안 쌍이 있습니다:

```
chXX_topic/
├── chXX_01_subtopic.ipynb              # 문제 노트북
├── chXX_01_subtopic_solution.ipynb     # 모범답안 노트북
├── chXX_02_subtopic.ipynb
├── chXX_02_subtopic_solution.ipynb
└── ...
```

## 노트북 구조

### 문제 노트북 (`*.ipynb`)
1. **학습 목표** — 이 노트북에서 다루는 핵심 내용
2. **이론적 배경** — LaTeX 수식을 포함한 수리적 기반
3. **구현 가이드** — 핵심 알고리즘의 코드 예시
4. **연습 문제** (3~5개) — 난이도별 분류
   - ★ 기본 구현
   - ★★ 심화 분석/비교
   - ★★★ 연구/도전 과제

### 모범답안 노트북 (`*_solution.ipynb`)
1. 각 문제에 대한 완전한 코드와 접근법 설명
2. 결과 해석
3. 확장 토론

## 시작하기

```bash
pip install -r requirements.txt
jupyter notebook
```

## 대상

- 수학/물리학 전공자
- 통계학 기초를 보유한 학습자
- 데이터 분석 실무 역량을 심화하고자 하는 사람
