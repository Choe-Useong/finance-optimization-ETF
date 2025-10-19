# ETF 포트폴리오 최적화

본 문서에서는 개별 종목과 ETF의 기대수익률을 계량적으로 추정하여 투자 포트폴리오를 최적화하는 전체 과정에 대한 설명이다.

### 1. 요구수익률 개념

요구수익률은 현재 시장가격을 정당화하기 위해 자산(기업, ETF 등)이 최소한으로 달성해야 하는 수익률이다. 이를 통해 포트폴리오 구성이 이루어진다.

### 2. 개별 종목의 요구수익률(r) 추정

#### 2.1 RIM(Residual Income Model) 기반 요구수익률 추정

Residual Income Model을 사용해 요구수익률을 역산한다.

$$V_{0} = B_{0} + \sum_{t = 1}^{40}\frac{(ROE_{t} - r) \cdot B_{t - 1}}{(1 + r)^{t}} + \frac{(ROE_{40} - r) \cdot B_{39}}{(r - g)(1 + r)^{40}}$$

- $V_{0}$: 기업 현재 시가총액
- $B_{0}$: 최근 장부가치(지배주주지분)
- $ROE_{t}$: ARIMA 모델로 예측한 미래 40분기(10년)의 ROE(지배주주순이익)
- $r$: 요구수익률 (역산 대상)
- $g$: 영구 성장률 (0으로가정)

#### 2.2 ARIMA 모델로 ROE 예측 {#arima-모델로-roe-예측-1}

ARIMA(1,0,1)(1,0,1,4) 모델로 계절성을 포함한 ROE를 예측한다. 본 분석에서 사용된 ROE는 2011년부터 분기 기준으로 제공되는 데이터를 기반으로 한다. 단 ROE 시계열 길이가 20분기 이하일 경우 예측치를 역사적 평균으로 대체한다.

$$ROE_{t} = c + \phi ROE_{t - 1} + \theta\epsilon_{t - 1} + \Phi ROE_{t - 4} + \Theta\epsilon_{t - 4} + \epsilon_{t}$$

- $\phi,\Phi$: 자기회귀(AR) 및 계절성 AR 계수
- $\theta,\Theta$: 이동평균(MA) 및 계절성 MA 계수
- $\epsilon_{t}$: 잔차항

#### 2.3 RIM 실패 시 CAPM 방식으로 요구수익률 대체

수치적으로 해를 구할 수 없을 경우 CAPM을 사용하여 요구수익률을 계산한다.

$$r_{i} = \beta_{i} \cdot r_{market} + (1 - \beta_{i}) \cdot r_{f}$$

- $\beta_{i}$: 시장 대비 종목의 민감도(베타)
- $r_{market}$: 시장 기대수익률(RIM 기반 지수 추정)
- $r_{f}$: 무위험 수익률

베타(β) 계산법:

$$\beta_{i} = \frac{\text{Cov}(r_{i},r_{m})}{\text{Var}(r_{m})}$$

### 3. ETF 기대수익률 계산

개별 ETF는 구성 종목의 요구수익률을 각 종목의 ETF 내 편입 비중($w_{i}$)에 따라 가중평균하여 기대수익률을 계산한다.

$$r_{ETF} = \sum_{i}^{}w_{i} \cdot r_{i}$$

편입 비중 정의:

$$w_{i} = \frac{\text{종목 i의 편입 비율}}$$

### 4. 공분산 추정: EWMA 방식

Exponentially Weighted Moving Average(EWMA)를 사용하여 최근 수익률에 더 높은 가중치를 부여하여 공분산을 계산한다. $\lambda$ 는 0.94로 설정한다.

로그 수익률:

$$r_{t} = \ln\left( \frac{P_{t}}{P_{t - 1}} \right)$$

EWMA 공분산:

$$\Sigma_{t} = \lambda\Sigma_{t - 1} + (1 - \lambda)r_{t}r_{t}^{T}$$

연율화:

$$\Sigma_{annual} = 252 \times \Sigma_{t}$$

### 5. 초과수익률 계산

무위험 수익률 대비 ETF가 달성할 수 있는 추가적인 수익률을 계산한다. 무위험수익률은 국고채3년 수익률로 한다.

$$\mu_{excess} = r_{ETF} - r_{f}$$

### 6. 해외 시장 기대수익률 {#해외-시장-기대수익률-1}

해외 시장 지수는Damodaran의 Implied Equity Risk Premium(ERP) 방식으로 추정된 r을 사용한다. 해당 방식은 실현된 현금흐름(FCFE)을 바탕으로 기대수익률을 역산한다.

$$\text{Cash Yield}_{TTM} = \frac{\text{배당금}_{12M} + \text{자사주매입}_{12M}}{\text{시가총액}}$$

이 값은 Trailing 12개월 기준으로, 시장 전체 주주에게 실제 지급된 현금 흐름의 수익률을 의미하며, 이를 첫 해 현금흐름($FCFE_{1}$)로 설정하고, 이후 일정 성장률을 가정하여 DCF를 구성한다.

DCF 수식 구조는 다음과 같다:

$$\text{Index Level} = \sum_{t = 1}^{N}\frac{FCFE_{t}}{(1 + r)^{t}} + \frac{FCFE_{N + 1}}{(r - g)(1 + r)^{N}}$$

여기서 r은 시장 기대수익률이며, r을 역산한 후,

$$\text{Implied ERP} = r - r_{f}$$

으로 계산한다. 여기서 $r_{f}$는 해당 국가의 무위험 수익률(미국은 10년 국채 수익률 기준)을 사용하였다.

미국 외 해외 국가 위험 프리미엄 계산:

다모다란은 미국 외 국가의 Implied ERP를 추정할 때, 미국의 ERP에 추가로 국가 위험 프리미엄(Country Risk Premium, CRP)을 더하는 방식 사용하였다.

$$ERP_{해외} = ERP_{미국} + CRP,\quad CRP = CDS\text{ 스프레드(또는 Default Spread)} \times \lambda$$

최종 해외 ETF 기대수익률(CAPM 방식):

$$r_{ETF(해외)} = r_{f} + \beta_{m}(r_{\text{Market}} - r_{f})$$

### 해외 환노출 ETF의 기대수익률 계산

해외 ETF가 완전 환헤지가 아닐 경우, 환율효과를 통제하여 기대수익률을 구한다.

$$r_{\text{ETF}} = r_{f} + \beta_{m}\, \mathbb{E}\left[r_{\text{Market}} - r_{f}\right] + \beta_{f} \cdot r_{\text{FX}}$$

#### 구성 요소 설명

- $r_{\text{ETF}}$: 해외 ETF의 기대수익률
- $r_{f}$: 무위험 수익률
- $r_{\text{Market}}$: 해외 시장지수의 수익률
- $r_{\text{FX}}$: 환율 수익률 (예: 원-달러 환율 수익률)
- $\beta_{m}$: 해외시장에 대한 민감도 (시장 베타)
- $\beta_{f}$: 환율에 대한 민감도 (환율 베타)

금의 기대수익률 추정

본 분석에서는 금 ETF의 기대수익률을 추정하기 위해, 미국 금 ETF인 GLD와 원/달러 환율 데이터를 활용하였다. KRX 금현물 ETF(A411060)는 상장된 지 오래되지 않아 충분한 과거 수익률 데이터를 제공하지 못하기 때문에, 더 긴 기간의 데이터를 활용하는 방식으로 대체하였다.

### 1. 데이터 수집 및 처리

- **GLD 가격**과 **KRW/USD 환율**을 월간 단위로 15년치 데이터 수집
- 두 데이터를 병합하여, 실제 원화 기준 금 가격 시계열을 생성:

$$\mathrm{KRW}_{\mathrm{GOLD},\, t} = \mathrm{GLD}_{t} \times \mathrm{FX}_{t}$$

### 2. 로그 수익률 및 예측

- 로그 수익률은 다음과 같이 계산하였다:

$$r_{t} = \ln\left( \frac{\mathrm{KRW}_{\mathrm{GOLD},\, t}}{\mathrm{KRW}_{\mathrm{GOLD},\, t-1}} \right)$$

- 생성된 월간 수익률 시계열에 대해 ARMA(1,1) 모델을 적합시켜, 다음 달 수익률을 예측하였다.
- 예측된 월간 기대수익률을 연율화한 후, 무위험수익률을 차감하여 초과 기대수익률을 추정하였다:

$$\mathbb{E}\left[r_{\text{gold}}\right] = 12 \cdot {\widehat{r}}_{\text{next month}} - r_{f}$$

### 7. Kelly 기준 포트폴리오 최적화

Kelly 최적화로 기대 로그 효용을 최대화하는 포트폴리오 비중을 결정한다. 비중의 과도한 편중을 막기 위해 릿지 패널티를 적용한다. 패널티는 기본 0.1로 설정한다.

$$\max_{\mathbf{w}}\left[ \mathbf{w}^{\top}\mu^{\text{excess}} - \frac{1}{2}\mathbf{w}^{\top}\Sigma\mathbf{w} \right] - \lambda \parallel \mathbf{w} \parallel^{2}$$

- 제약 조건

  - $\mathbf{w}_{i} \geq 0$

- $\lambda$: 정칙화 계수 (Ridge penalty)

기대수익률·공분산 추정치의 불확실성과 자산의 꼬리 위험을 고려하여 최종적으로는 비중에 1/2을 곱한 하프켈리를 사용한다.

$$\mathbf{w}^{\text{half}} = \tfrac{1}{2}\mathbf{w}$$

산출된 비중의 합의 나머지 비중은 국고채3년을 할당한다.
