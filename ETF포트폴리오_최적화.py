import pandas as pd
import yfinance as yf
import statsmodels.tsa.arima.model
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import root_scalar
from pykrx import stock, bond
from datetime import datetime
from scipy.optimize import minimize
import re

us_etfs = ['133690.KS',  # TIGER 미국나스닥100
           '381180.KS',  # TIGER 미국필라델피아반도체나스닥
           '182480.KS',  # TIGER 미국MSCI리츠(합성 H)
           ]
cn_etfs = ['192090.KS']
etf_list = ['462010.KS',  # TIGER 2차전지소재Fn
            '396500.KS',  # KODEX 반도체 탑10
            '091180.KS',  # KODEX 자동차
            '227560.KS',  # TIGER 200 생활소비재
            ]
comm_etf = ['411060.KS',  # ACE KRX금현물
            ]
ktickers = etf_list
all_ticker = ktickers + us_etfs + cn_etfs + comm_etf




etf_name_map = {
    '133690.KS': 'TIGER 미국나스닥100',
    '381180.KS': 'TIGER 미국필라델피아반도체',
    '182480.KS': 'TIGER 미국MSCI리츠(합성 H)',
    '192090.KS': 'KODEX 차이나CSI300',
    '462010.KS': 'ACE 2차전지 테마포커스',
    '396500.KS': 'KODEX 반도체 탑10',
    '091180.KS': 'KODEX 자동차',
    '227560.KS': 'TIGER 200 생활소비재',
    '411060.KS': 'ACE KRX금현물',
}




data = pd.read_excel(r"C:\Users\admin\Desktop\갭스.xlsx", header= None)
data = data.drop(index = [0,1,2,3,4,5,6,7,10,11,13])
data = data.set_index(0)
data= data.T

data = data.melt(
    id_vars=data.columns[0:3],          # 그대로 유지할 열
    value_vars=data.columns[3:],  # 세로로 변환할 열
    var_name='날짜',                  # 새로 생길 '변수 이름' 열
    value_name='값'                  # 값이 들어갈 열 이름
)


data['날짜'] = pd.to_datetime(data['날짜'])
data.set_index(data['날짜'], inplace = True)
data.drop(columns = '날짜', inplace = True)
data = data.reset_index()
data = data.pivot(index = ['날짜','Symbol', 'Symbol Name'], columns = 'Item Name', values = '값')
data = data.apply(pd.to_numeric, errors='coerce').reset_index()
data['날짜'] = pd.to_datetime(data['날짜'])
data = data.set_index('날짜')




dataI = pd.read_excel(r"C:\Users\admin\Desktop\갭스지수.xlsx", header= None)
dataI = dataI.drop(index = [0,1,2,3,4,5,6,7,10,11,13])
dataI = dataI.set_index(0)
dataI= dataI.T

dataI = dataI.melt(
    id_vars=dataI.columns[0:3],          # 그대로 유지할 열
    value_vars=dataI.columns[3:],  # 세로로 변환할 열
    var_name='날짜',                  # 새로 생길 '변수 이름' 열
    value_name='값'                  # 값이 들어갈 열 이름
)


dataI['날짜'] = pd.to_datetime(dataI['날짜'])
dataI.set_index(dataI['날짜'], inplace = True)
dataI.drop(columns = '날짜', inplace = True)
dataI = dataI.reset_index()
dataI = dataI.pivot(index = ['날짜','Symbol', 'Symbol Name'], columns = 'Item Name', values = '값')
dataI = dataI.apply(pd.to_numeric, errors='coerce').reset_index()
dataI['날짜'] = pd.to_datetime(dataI['날짜'])
dataI = dataI.set_index('날짜')

#코스피 2,538,235,151 * 백만
#코스닥 414,507,549*백만



def RIM_value_with_terminal(r_annual, B0, roe_forecast):
    """
    r_annual: 연율 기준 할인율 (예: 0.10은 10%)
    B0: 현재 북밸류 (예: 주당 BPS * 주식 수)
    roe_forecast: 연율화된 ROE 예측값 (% 단위, 예: 15.0이면 15%)
    """
    # 연율 → 분기율 변환
    r_q = (1 + r_annual) ** (1 / 4) - 1
    
    Bt = B0
    value = B0
    
    for h in range(1, len(roe_forecast) + 1):
        roe_annual = roe_forecast.iloc[h - 1] / 100
        roe_q = (1 + roe_annual) ** (1/4) - 1  # 연율 ROE → 분기 ROE
        RI = (roe_q - r_q) * Bt
        value += RI / (1 + r_q)**h
        Bt *= (1 + roe_q)
    
    # Terminal value
    terminal_roe_annual = roe_forecast.iloc[-1] / 100
    terminal_roe_q = (1 + terminal_roe_annual) ** (1/4) - 1
    terminal_RI = (terminal_roe_q - r_q) * Bt
    TV = terminal_RI / (r_q * (1 + r_q)**len(roe_forecast))
    
    return value + TV


from pandas.tseries.offsets import MonthEnd
import numpy as np



def compute_daily_beta_1y(sym_code, market_code):
    try:
        from datetime import datetime
        import yfinance as yf
        import numpy as np
        import pandas as pd

        print(f"\n⏳ {sym_code} vs {market_code} 베타 계산 시작")

        end_date = datetime.today()
        start_date = end_date - pd.DateOffset(years=1)
        print(f" 기간: {start_date.date()} ~ {end_date.date()}")

        # DataFrame으로 명시적 다운로드
        stock_df = yf.download(sym_code, start=start_date, end=end_date, interval='1d')[['Close']]
        market_df = yf.download(market_code, start=start_date, end=end_date, interval='1d')[['Close']]
        print(" 다운로드 완료")

        # 열 이름 명시적 변경
        stock_df = stock_df.rename(columns={'Close': 'stock'})
        market_df = market_df.rename(columns={'Close': 'market'})

        # 날짜 기준 병합
        merged = pd.merge(stock_df, market_df, left_index=True, right_index=True).dropna()
        print(" 병합 후 데이터 수:", len(merged))

        if len(merged) < 30:
            print(f" 데이터 너무 짧음 ({len(merged)}일)")
            return None

        # 로그 수익률 계산
        merged['stock_ret'] = np.log(merged['stock'] / merged['stock'].shift(1))
        merged['market_ret'] = np.log(merged['market'] / merged['market'].shift(1))
        merged = merged.dropna()
        print(f" 수익률 계산 완료. 관측치 수: {len(merged)}")

        # numpy로 변환 후 베타 계산
        x = merged['market_ret'].to_numpy()
        y = merged['stock_ret'].to_numpy()
        beta = np.cov(y, x)[0, 1] / np.var(x)
        print(f" 최종 베타: β = {beta:.4f}")
        return beta

    except Exception as e:
        print(f" {sym_code} 베타 계산 실패: {e}")
        return None







# 기준일 값 (수동 한 번만)
base_date = '2025-06-26'
base_index = {
    '코스피': 3079.56,
    '코스닥': 787.95
}
base_market_caps = {
    '코스피': 2519593952 * 1_000_000,
    '코스닥': 407828275 * 1_000_000
}

# 오늘 시총 계산 함수 (정상작동 버전)
def get_current_market_caps():
    ticker_map = {'코스피': '^KS11', '코스닥': '^KQ11'}
    market_caps_today = {}

    for market, ticker in ticker_map.items():
        hist = yf.download(ticker, start=base_date, progress=False, auto_adjust=False)[['Close']]
        current_price = hist['Close'].iloc[-1].item()
        ratio = current_price / base_index[market]
        market_caps_today[market] = base_market_caps[market] * ratio

    return market_caps_today

# 사용 예시
market_caps = get_current_market_caps()
print(market_caps)




















result_market_list = []

for market in ['코스피', '코스닥']:
    try:
        market_data = dataI[dataI['Symbol Name'].str.contains(market)]
        
        # ROE 시계열
        roe_series = market_data['ROE(지배주주순이익)(%)'].dropna()
        
        # 북밸류 B0
        B0 = market_data['지배주주지분(원)'].dropna().tail(1).values[0]
        
        if len(roe_series) <= 20:
            forecast = pd.Series([roe_series.mean()] * 40)
        else:
            model = ARIMA(roe_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 4))
            result = model.fit()
            forecast = result.get_forecast(steps=40).predicted_mean
        
        market_cap = market_caps[market]

        # 내재 r 찾기
        def residual_r(r_annual):
            return RIM_value_with_terminal(r_annual, B0, forecast) - market_cap

        r_result = root_scalar(residual_r, bracket=[0.001, 0.9], method='brentq')

        result_market_list.append({
            '시장명': market,
            'r 추정치': r_result.root
        })
    
    except Exception as e:
        print(f"{market} 처리 중 오류 발생: {e}")



market_df = pd.DataFrame(result_market_list)
print(market_df)



# 오늘 날짜를 'YYYYMMDD' 문자열로 변환
today = datetime.today().strftime('%Y%m%d')

# 오늘 날짜로 데이터 요청
rf = bond.get_otc_treasury_yields(today).loc['국고채 3년','수익률'] / 100



# 결과 저장용
merged_dict = {}
etf_returns = {}

for etf_code in etf_list:
    print(f"\n ETF {etf_code} 처리 시작")

    try:
        # 1. ETF 구성 정보 불러오기
        df = stock.get_etf_portfolio_deposit_file(etf_code.split('.')[0]).reset_index()
        df['심볼A'] = 'A' + df['티커'].str.replace('.KS', '', regex=False)
        df['비중'] = pd.to_numeric(df['비중']) / 100
        df = df[['심볼A', '비중']]

        # 2. 종목 이름 매칭
        matched_names = data[data['Symbol'].isin(df['심볼A'])]['Symbol Name'].unique().tolist()
        result_list = []

        for name in matched_names:
            try:
                firm_data = data[data['Symbol Name'] == name]
                roe_series = firm_data['ROE(지배주주순이익)(%)'].dropna()
                B0 = firm_data['지배주주지분(원)'].dropna().tail(1).values[0]
                symbol = firm_data['Symbol'].astype(str).iloc[-1]
                base_code = symbol[1:]

                # sym_code 및 시장 구분
                try:
                    sym_code = base_code + '.KS'
                    ticker = yf.Ticker(sym_code)
                    market_cap = ticker.info.get('marketCap')
                    if market_cap is None:
                        raise ValueError()
                    market_name = '코스피'
                    market_code = '^KS11'
                except:
                    sym_code = base_code + '.KQ'
                    ticker = yf.Ticker(sym_code)
                    market_cap = ticker.info.get('marketCap')
                    if market_cap is None:
                        raise ValueError()
                    market_name = '코스닥'
                    market_code = '^KQ11'

                # ROE forecast
                if len(roe_series) <= 20:
                    forecast = pd.Series([roe_series.mean()] * 40)
                else:
                    model = ARIMA(roe_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 4))
                    result = model.fit()
                    forecast = result.get_forecast(steps=40).predicted_mean

                # RIM 할인율 추정
                def residual_r(r_annual):
                    return RIM_value_with_terminal(r_annual, B0, forecast) - market_cap

                try:
                    r_result = root_scalar(residual_r, bracket=[0.001, 0.9], method='brentq')
                    r_final = r_result.root
                except ValueError:
                    # 대체 방식
                    r_market = market_df[market_df['시장명'] == market_name]['r 추정치'].values[0]
                    beta = compute_daily_beta_1y(sym_code, market_code)
                    if beta is None:
                        print(f"{name} → 베타 계산 실패, 제외")
                        continue
                    r_final = beta * r_market + (1 - beta) * rf
                    print(f"{name}: r 대체됨 → β={beta:.3f}, r_market={r_market:.4f}, rf={rf:.4f} → r={r_final:.4f}")

                result_list.append({
                    '종목명': name,
                    '심볼': sym_code,
                    'r 추정치': r_final
                })

            except Exception as e:
                print(f"{name} 처리 중 오류: {e}")
                continue

        # 병합
        rim_df = pd.DataFrame(result_list)
        rim_df['심볼A'] = 'A' + rim_df['심볼'].str[:6]
        merged = pd.merge(rim_df, df, on='심볼A', how='left')

        # 기대수익률 계산
        etf_r = (merged['r 추정치'] * merged['비중']).sum()
        etf_returns[etf_code] = etf_r

        # 결과 저장
        merged_dict[etf_code] = merged
        print(f" ETF {etf_code} 기대수익률: {etf_r:.4%}")

    except Exception as e:
        print(f" ETF {etf_code} 처리 실패: {e}")







def optimize_weights(mu, cov, objective='sharpe', ridge=1e-3, sum_to_one=True):
    mu_arr = mu.values
    cov_mat = cov.values
    N = len(mu_arr)

    if objective == 'sharpe':
        def obj(w):
            ret = w @ mu_arr
            vol = np.sqrt(w @ cov_mat @ w)
            ratio = -ret / vol if vol > 0 else np.inf
            penalty = ridge * np.sum(w ** 2)
            return ratio + penalty

    elif objective == 'kelly':
        def obj(w):
            utility = -(w @ mu_arr - 0.5 * w @ cov_mat @ w)
            penalty = ridge * np.sum(w ** 2)
            return utility + penalty

    else:
        raise ValueError("objective는 'sharpe' 또는 'kelly'만 가능합니다.")

    bounds = [(0, 1)] * N if sum_to_one else [(0, None)] * N

    #  비중합 = 1 제약 여부
    if sum_to_one:
        cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},)
    else:
        cons = ()

    w0 = np.ones(N) / N

    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
    return pd.Series(res.x if res.success else np.full(N, np.nan), index=mu.index)














def get_annualized_cov_matrix(ticker_list, start="2023-01-01", end=None, lambda_=0.94):
    """
    감쇠계수 기반 EWMA 연율화 공분산 행렬 계산
    - ticker_list: 티커 리스트
    - lambda_: 감쇠계수 λ (예: 0.94)
    """
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')

    price_df = pd.DataFrame()

    for ticker in ticker_list:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)[['Close']]
            price_df[ticker] = df['Close']
        except Exception as e:
            print(f" {ticker} 다운로드 실패: {e}")

    returns = np.log(price_df / price_df.shift(1)).dropna()

    alpha = 1 - lambda_  # EWMA의 α = 1 - λ

    # 감쇠계수 기반 EWMA 공분산
    ewm_cov = returns.ewm(alpha=alpha).cov(pairwise=True)

    # 마지막 날짜의 공분산 행렬만 추출
    last_date = ewm_cov.index.get_level_values(0).max()
    cov_matrix = ewm_cov.loc[last_date].copy()

    # 연율화
    annualized_cov = cov_matrix * 252

    return annualized_cov.loc[ticker_list, ticker_list]







cov_matrix = get_annualized_cov_matrix(all_ticker, start="2023-01-01")

print(" 연율화 공분산 행렬 (EWMA 기반):")
print(cov_matrix)
















import pandas_datareader.data as web
import datetime

# 오늘 날짜 지정 (혹시 데이터가 없을 경우 대비해서 하루 전까지 fallback 가능)
endus = datetime.datetime.today()
startus = endus - datetime.timedelta(days=7)  # 최근 일주일

# FRED에서 10년 국채 수익률 'DGS10' 불러오기
frd = web.DataReader('DGS10', 'fred', startus, endus)

# 결측 제거 + 가장 최근 값 추출
frd = frd.dropna()
usrf = frd.iloc[-1, 0] / 100  # 퍼센트 → 소수로 변환




text = """Implied ERP in previous month = 4.41% (Trailing 12 month, with adjusted payout); \
4.58% (Trailing 12 month cash yield); 5.86% (Average CF yield last 10 years); \
4.34% (Net cash yield); 4.03% (Normalized Earnings & Payout)"""

# 수치만 추출 후 소수로 변환
pattern = r"([\d.]+)%\s*\(Trailing 12 month cash yield\)"
match = re.search(pattern, text)

if match:
    userp = float(match.group(1)) / 100
    print(userp)
else:
    print("값을 찾을 수 없습니다.")




chinaerp = 5.27 / 100




from datetime import datetime, timedelta
import pandas_datareader.data as web
import statsmodels.api as sm

exticker = us_etfs + cn_etfs

# 벤치마크 및 환율, ERP 매핑 (사용자 명칭 유지)
benchmark_map = {etf: '449180.KS' for etf in us_etfs}
benchmark_map.update({etf: '000300.SS' for etf in cn_etfs})

fx_ticker_map = {etf: 'KRW=X' for etf in us_etfs}
fx_ticker_map.update({etf: 'CNY=X' for etf in cn_etfs})

erp_map = {etf: 4.58 / 100 for etf in us_etfs}
erp_map.update({etf: 5.27 / 100 for etf in cn_etfs})

# 기간 설정
ex_end = datetime.today()
ex_start = ex_end - timedelta(days=365)

# 종목 전체 수집 대상
combined_ticker_universe = list(set(exticker + list(benchmark_map.values()) + list(fx_ticker_map.values())))

# 가격 데이터 수집
price_matrix_for_beta_estimation = yf.download(combined_ticker_universe, start=ex_start, end=ex_end)['Close']

# 로그수익률 계산
log_return_for_exmu_beta = np.log(price_matrix_for_beta_estimation).diff().dropna()

# 무위험수익률
try:
    t10y_price_table = web.DataReader('DGS10', 'fred', ex_end - timedelta(days=7), ex_end)
    t10y_price_table = t10y_price_table.dropna()
    usrf = t10y_price_table.iloc[-1, 0] / 100
except:
    usrf = 0.045  # fallback

# 결과 저장 테이블
exmu_result = []

# ETF별 베타 계산
for etf in exticker:
    try:
        market_index_symbol = benchmark_map[etf]
        fx_ticker = fx_ticker_map[etf]
        erp_value = erp_map[etf]

        regression_input_frame = pd.concat([
            log_return_for_exmu_beta[etf],
            log_return_for_exmu_beta[market_index_symbol],
            log_return_for_exmu_beta[fx_ticker]
        ], axis=1).dropna()
        regression_input_frame.columns = ['etf_return', 'market_return', 'fx_return']

        regression_design_matrix = sm.add_constant(regression_input_frame[['market_return', 'fx_return']])
        regression_target_vector = regression_input_frame['etf_return']

        fitted_ols_model = sm.OLS(regression_target_vector, regression_design_matrix).fit()
        market_beta_fx_adjusted = fitted_ols_model.params['market_return']

        expected_excess_return = market_beta_fx_adjusted * erp_value + (1 - market_beta_fx_adjusted) * usrf

        exmu_result.append({
            'ETF': etf,
            'Market': market_index_symbol,
            'Beta (FX-controlled)': round(market_beta_fx_adjusted, 4),
            'Expected Return (%)': round(expected_excess_return * 100, 2)
        })

    except Exception as e:
        print(f" {etf} 처리 실패: {e}")

# 결과 출력
exmu_df = pd.DataFrame(exmu_result)
exmu_df['Expected Return (%)'] = exmu_df['Expected Return (%)'] + (usrf*100) - (rf*100)

print(exmu_df)



# 1. 한국 ETF 기대수익률 (직접 수익률 - rf)
kr_mu = pd.Series(etf_returns)  # {'091160': val, ...}
kr_mu = kr_mu - rf            # 동일한 무위험수익률 사용

# 2. 외국 ETF 기대수익률 (exmu_df의 값, 이미 rf 포함됨)
foreign_mu = exmu_df.set_index('ETF')['Expected Return (%)'] / 100  # 소수로

# 3. 티커명 통일
kr_mu.index = [f"{code}" for code in kr_mu.index]

#  금 (commodity) 기대수익률 추정 전용 기간 설정
comm_end_date = datetime.today()
comm_start_date = comm_end_date - timedelta(days=365 * 15)  # 최대한 길게

#  데이터 수집
gld_price = yf.download('GLD', start=comm_start_date, end=comm_end_date, interval='1mo')['Close']
fx_price = yf.download('KRW=X', start=comm_start_date, end=comm_end_date, interval='1mo')['Close']
fx_price[fx_price['KRW=X'] <= 1] = fx_price[fx_price['KRW=X'] <= 1]*10000


#  병합 및 정리
comm_df = pd.concat([gld_price, fx_price], axis=1).dropna()
comm_df.columns = ['GLD', 'FX']

#  환노출 금 가격 계산
comm_df['KRW_GOLD'] = comm_df['GLD'] * comm_df['FX']
comm_df['log_return'] = np.log(comm_df['KRW_GOLD'] / comm_df['KRW_GOLD'].shift(1))
comm_df = comm_df.dropna()

#  ARIMA 예측 기반 기대수익률 산출
model = ARIMA(comm_df['log_return'], order=(1, 0, 1))
result = model.fit()
forecast = result.get_forecast(steps=1).predicted_mean  

expected_annual_return = (forecast.mean() * 12) - rf  # 월수익률 → 연환산
print(f" 환노출 금 기대수익률 (ARIMA 기반): {expected_annual_return:.4%}")

# 3. 원자재 ETF 기대수익률 (역사적 평균)
comm_mu = pd.Series({'411060.KS': expected_annual_return})

# 4. 통합
mu = pd.concat([kr_mu, foreign_mu, comm_mu])

# 리스트 곱하기 (순서 일치해야 함)
adjusted_mu = mu * 1



# 켈리 최적 포트폴리오 비중 계산
kelly_weights = optimize_weights(adjusted_mu, cov_matrix, objective='kelly', ridge= 0.1, sum_to_one= False) *0.5






etf_name_map = {
    '133690.KS': 'TIGER 미국나스닥100',
    '381180.KS': 'TIGER 미국필라델피아반도체',
    '182480.KS': 'TIGER 미국MSCI리츠(합성 H)',
    '192090.KS': 'KODEX 차이나CSI300',
    '462010.KS': 'TIGER 2차전지소재Fn',
    '396500.KS': 'KODEX 반도체 탑10',
    '091180.KS': 'KODEX 자동차',
    '227560.KS': 'TIGER 200 생활소비재',
    '411060.KS': 'ACE KRX금현물',
}



mu_named = mu.rename(index=etf_name_map)
kelly_named = kelly_weights.rename(index=etf_name_map)


print(" 기대수익률 (mu):")
print(mu_named.sort_values(ascending=False).apply(lambda x: f"{x:.2%}"))

print("\n 켈리 최적 비중:")
print(kelly_named.sort_values(ascending=False).apply(lambda x: f"{x:.2%}"))


print(round((1 -kelly_weights.sum())*100,2))
