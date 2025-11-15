# energy_forecasting_pipeline.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ---------- Utilities ----------
def load_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)

def handle_missing_values(df):
    numcols = df.select_dtypes(include=[np.number]).columns
    for c in numcols:
        if df[c].isna().sum():
            df[c] = df[c].interpolate(method='linear', limit_direction='both')
    return df

# ---------- Data Loading ----------
def load_all(data_dir="/content"):
    gas_prices = load_csv(f"{data_dir}/weekly_gasoline_prices.csv")
    stocks = load_csv(f"{data_dir}/all_stocks_and_etfs.csv")
    stock_descriptions = load_csv(f"{data_dir}/stock_descriptions.csv")
    commodities = load_csv(f"{data_dir}/all_commodities.csv")
    supply = load_csv(f"{data_dir}/weekly_supply_estimates.csv")
    gasoline_makeup = load_csv(f"{data_dir}/monthly_gasoline_makeup_percentages.csv")
    transportation_stats = load_csv(f"{data_dir}/monthly_transportation_statistics.csv", delimiter='\t')
    return dict(
        gas_prices=gas_prices, stocks=stocks, stock_descriptions=stock_descriptions,
        commodities=commodities, supply=supply, gasoline_makeup=gasoline_makeup,
        transportation_stats=transportation_stats
    )

# ---------- Preprocessing ----------
def preprocess(d):
    gas_prices = d['gas_prices'].copy()
    stocks = d['stocks'].copy()
    stock_descriptions = d['stock_descriptions'].copy()
    commodities = d['commodities'].copy()
    transportation_stats = d['transportation_stats'].copy()

    gas_prices['Date'] = pd.to_datetime(gas_prices['Date'])
    stocks['Date'] = pd.to_datetime(stocks['Date-Time'])
    commodities['Date'] = pd.to_datetime(commodities['Date_Time'])
    transportation_stats_cols = transportation_stats.columns.tolist()
    date_col = transportation_stats_cols[0]
    transportation_stats['Date'] = pd.to_datetime(transportation_stats[date_col].str.split(',').str[0], errors='coerce')
    transportation_stats['Month'] = transportation_stats['Date'].dt.month
    transportation_stats['Year'] = transportation_stats['Date'].dt.year

    # Regular gasoline US monthly mean
    regular_gas = gas_prices[gas_prices['Type_Clean'].str.contains('Regular', na=False)]
    regular_gas_us = regular_gas[regular_gas['Geography'] == 'US'].sort_values('Date')
    regular_gas_us = handle_missing_values(regular_gas_us)
    regular_gas_us['Year_Month'] = regular_gas_us['Date'].dt.to_period('M')
    monthly_gas = regular_gas_us.groupby('Year_Month')['Price'].mean().reset_index()
    monthly_gas['Year_Month_str'] = monthly_gas['Year_Month'].astype(str)
    monthly_gas['Month'] = monthly_gas['Year_Month'].dt.month
    monthly_gas['Year'] = monthly_gas['Year_Month'].dt.year
    monthly_gas['Price_Change_Pct'] = monthly_gas['Price'].pct_change() * 100

    # Energy stocks filtering
    energy_symbols = stock_descriptions[stock_descriptions['Sector'].str.contains('ENERGY', na=False)]['Symbol'].unique()
    energy_stocks_data = stocks[stocks['Ticker_Symbol'].isin(energy_symbols)].copy()
    energy_stocks_data = handle_missing_values(energy_stocks_data)

    # Market ETFs (SPY etc.)
    market_etfs = stocks[stocks['Ticker_Symbol'].isin(['SPY', 'QQQ', 'DIA'])].copy()
    market_etfs = handle_missing_values(market_etfs)

    crude_oil = commodities[commodities['Commodity_Simple'] == 'CRUDE_OIL'].copy()
    crude_oil = handle_missing_values(crude_oil)

    # Transportation selection (best-effort)
    miles_col = [c for c in transportation_stats.columns if 'miles' in c.lower() or 'traveled' in c.lower()]
    gas_price_col = [c for c in transportation_stats.columns if 'price' in c.lower() and 'gas' in c.lower()]
    relevant_cols = ['Date', 'Month', 'Year'] + ([miles_col[0]] if miles_col else []) + ([gas_price_col[0]] if gas_price_col else [])
    trans_selected = transportation_stats[[c for c in relevant_cols if c in transportation_stats.columns]].copy()

    return dict(
        regular_gas_us=regular_gas_us, monthly_gas=monthly_gas,
        energy_stocks_data=energy_stocks_data, market_etfs=market_etfs,
        crude_oil=crude_oil, trans_selected=trans_selected
    )

# ---------- Feature creation & aggregation ----------
def compute_monthly_returns(df):
    df = df.sort_values(['Ticker_Symbol', 'Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    df_monthly = df.groupby(['Ticker_Symbol', 'Month'])['Close'].last().reset_index()
    df_monthly['Return'] = df_monthly.groupby('Ticker_Symbol')['Close'].pct_change() * 100
    return df_monthly

def prepare_gas_and_returns(monthly_gas, energy_monthly_returns, market_monthly_returns):
    energy_sector_avg = energy_monthly_returns.groupby('Month')['Return'].mean().reset_index()
    energy_sector_avg['Month'] = energy_sector_avg['Month'].astype('period[M]')
    energy_sector_avg['Month_int'] = energy_sector_avg['Month'].dt.month

    spy_returns = market_monthly_returns[market_monthly_returns['Ticker_Symbol'] == 'SPY'].copy()
    spy_returns['Month'] = spy_returns['Month'].astype('period[M]')
    spy_returns['Month_int'] = spy_returns['Month'].dt.month

    mg = monthly_gas.copy()
    mg['Month_int'] = mg['Month']

    gas_and_returns = pd.merge(mg[['Month_int','Price','Year','Price_Change_Pct']],
                               energy_sector_avg[['Month_int','Return']],
                               on='Month_int', how='inner')
    gas_and_returns = pd.merge(gas_and_returns,
                               spy_returns[['Month_int','Return']].rename(columns={'Return':'Market_Return'}),
                               on='Month_int', how='inner')
    gas_and_returns = gas_and_returns.sort_values('Month_int')
    gas_and_returns['Prev_Gas_Price'] = gas_and_returns['Price'].shift(1)
    gas_and_returns['Gas_Price_Change'] = gas_and_returns['Price'].pct_change() * 100
    gas_and_returns = gas_and_returns.dropna().reset_index(drop=True)
    return gas_and_returns

# ---------- EDA plots (call selectively) ----------
def plot_gas_trends(regular_gas_us, monthly_gas):
    plt.figure(figsize=(10,4))
    plt.plot(regular_gas_us['Date'], regular_gas_us['Price'], linewidth=1)
    plt.title('US Regular Gasoline Prices')
    plt.tight_layout(); plt.show()

def plot_return_distributions(energy_monthly_returns, market_monthly_returns):
    energy_avg = energy_monthly_returns.groupby('Month')['Return'].mean().reset_index()
    spy = market_monthly_returns[market_monthly_returns['Ticker_Symbol']=='SPY']
    plt.figure(figsize=(8,4))
    plt.hist(energy_avg['Return'].dropna(), bins=30, alpha=0.7, label='Energy')
    plt.hist(spy['Return'].dropna(), bins=30, alpha=0.5, label='SPY')
    plt.legend(); plt.tight_layout(); plt.show()

# ---------- Regression & Correlation ----------
def linear_regression_predict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression(); lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return lr, X_test, y_test, y_pred

# ---------- Granger causality ----------
def run_granger(ts_df, maxlag=6):
    results = grangercausalitytests(ts_df[['Price','Return']], maxlag=maxlag, verbose=False)
    pvals = {lag: results[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag+1)}
    return pd.DataFrame({'Lag':list(pvals.keys()), 'p-value':list(pvals.values())})

# ---------- VAR modeling ----------
def prepare_ts_features(gas_and_returns):
    ts = gas_and_returns[['Price','Return']].copy()
    ts = ts.dropna()
    for col in ['Price','Return']:
        m = ts[col].mean(); s = ts[col].std()
        ts = ts[(ts[col] >= m - 3*s) & (ts[col] <= m + 3*s)]
    ts['Price_MA5'] = ts['Price'].rolling(5).mean()
    ts['Price_MA20'] = ts['Price'].rolling(20).mean()
    ts['Price_Volatility'] = ts['Price'].rolling(10).std()
    ts['Return_MA5'] = ts['Return'].rolling(5).mean()
    ts['Return_Volatility'] = ts['Return'].rolling(10).std()
    for lag in range(1,6):
        ts[f'Price_Lag{lag}'] = ts['Price'].shift(lag)
        ts[f'Return_Lag{lag}'] = ts['Return'].shift(lag)
    ts['Price_Momentum'] = ts['Price'].pct_change(5)
    ts['Return_Momentum'] = ts['Return'].pct_change(5)
    ts = ts.dropna()
    return ts

def create_train_test_split(ts_data, test_size=0.2):
    train_size = int(len(ts_data) * (1 - test_size))
    train = ts_data.iloc[:train_size].copy()
    test = ts_data.iloc[train_size:].copy()
    return train, test

def build_var_model(train, test):
    train_num = train.select_dtypes(include=[np.number]).copy()
    test_num = test.select_dtypes(include=[np.number]).copy()
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_num), index=train_num.index, columns=train_num.columns)
    model = VAR(train_scaled)
    best_lag, best_aic = 1, float('inf')
    for lag in range(1, min(12, len(train_scaled)//10)+1):
        try:
            m = model.fit(lag)
            if m.aic < best_aic:
                best_aic = m.aic; best_lag = lag
        except Exception:
            continue
    var_model = model.fit(best_lag)
    # forecasting step-by-step
    last_window = train_scaled.values[-best_lag:]
    preds = []
    for i in range(len(test_num)):
        f = var_model.forecast(last_window, steps=1)
        preds.append(f[0])
        if i < len(test_num)-1:
            last_window = np.vstack([last_window[1:], test_num.iloc[[i]].values])
    preds = np.array(preds)
    preds_df = pd.DataFrame(scaler.inverse_transform(preds), index=test_num.index, columns=test_num.columns)
    return test_num, preds_df['Price'], preds_df['Return'], var_model

# ---------- ML models ----------
def build_ml_models(train, test):
    X_cols = [c for c in train.columns if c not in ['Price','Return']]
    X_train_price, y_train_price = train[X_cols], train['Price']
    X_test_price, y_test_price = test[X_cols], test['Price']
    X_train_ret, y_train_ret = train[X_cols], train['Return']
    X_test_ret, y_test_ret = test[X_cols], test['Return']

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    best_price = {'name':None,'r2':-np.inf,'pred':None}
    best_return = {'name':None,'r2':-np.inf,'pred':None}
    for name, model in models.items():
        model.fit(X_train_price, y_train_price)
        price_pred = model.predict(X_test_price)
        price_r2 = r2_score(y_test_price, price_pred)

        model_r = model.__class__(**model.get_params())
        model_r.fit(X_train_ret, y_train_ret)
        ret_pred = model_r.predict(X_test_ret)
        ret_r2 = r2_score(y_test_ret, ret_pred)

        if price_r2 > best_price['r2']:
            best_price.update({'name':name,'r2':price_r2,'pred':price_pred,'y':y_test_price})
        if ret_r2 > best_return['r2']:
            best_return.update({'name':name,'r2':ret_r2,'pred':ret_pred,'y':y_test_ret})
    return best_price, best_return, X_cols

# ---------- Plot helpers ----------
def plot_predictions(test_index, actual_price, pred_price, actual_ret, pred_ret, title_suffix="Model"):
    fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    ax[0].plot(test_index, actual_price, label='Actual'); ax[0].plot(test_index, pred_price, label='Predicted')
    ax[0].set_title(f'Gas Price - {title_suffix}'); ax[0].legend()
    ax[1].plot(test_index, actual_ret, label='Actual'); ax[1].plot(test_index, pred_ret, label='Predicted')
    ax[1].set_title(f'Energy Returns - {title_suffix}'); ax[1].legend()
    plt.tight_layout(); plt.show()

# ---------- Full pipeline ----------
def run_pipeline(data_dir="/content", do_plots=False):
    raw = load_all(data_dir)
    pre = preprocess(raw)

    energy_monthly_returns = compute_monthly_returns(pre['energy_stocks_data'])
    market_monthly_returns = compute_monthly_returns(pre['market_etfs'])
    gas_and_returns = prepare_gas_and_returns(pre['monthly_gas'], energy_monthly_returns, market_monthly_returns)

    # quick EDA plots (optional)
    if do_plots:
        plot_gas_trends(pre['regular_gas_us'], pre['monthly_gas'])
        plot_return_distributions(energy_monthly_returns, market_monthly_returns)

    # correlations
    corr_prev_gas_energy = gas_and_returns['Prev_Gas_Price'].corr(gas_and_returns['Return'])
    corr_gas_change_energy = gas_and_returns['Gas_Price_Change'].corr(gas_and_returns['Return'])
    corr_prev_gas_market = gas_and_returns['Prev_Gas_Price'].corr(gas_and_returns['Market_Return'])
    print("Correlations:", corr_prev_gas_energy, corr_gas_change_energy, corr_prev_gas_market)

    # Regression
    X = gas_and_returns[['Prev_Gas_Price']]; y = gas_and_returns['Return']
    lr_model, X_test, y_test, y_pred = linear_regression_predict(X, y)
    print("Linear Regression R2:", r2_score(y_test, y_pred))

    # Granger causality
    granger_df = run_granger(gas_and_returns, maxlag=6)
    print("Granger p-values:\n", granger_df)

    # Predictive modeling
    ts_data = prepare_ts_features(gas_and_returns)
    train, test = create_train_test_split(ts_data, test_size=0.2)
    test_num, var_price_pred, var_return_pred, var_model = build_var_model(train, test)
    print("VAR done.")

    best_price, best_return, feature_cols = build_ml_models(train, test)
    print("Best ML price model:", best_price['name'], "R2:", best_price['r2'])
    print("Best ML return model:", best_return['name'], "R2:", best_return['r2'])

    # Plot results
    plot_predictions(test.index, test_num['Price'], var_price_pred, test_num['Return'], var_return_pred, title_suffix="VAR")
    plot_predictions(best_price['y'].index, best_price['y'], best_price['pred'], best_return['y'].index, best_return['pred'], title_suffix="Best ML")

    # Feature importance if RF used
    if best_price['name'] == 'Random Forest':
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(train[feature_cols], train['Price'])
        fi = pd.DataFrame({'Feature': feature_cols, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
        print("Top features:\n", fi.head(10))

    return {
        'corrs': (corr_prev_gas_energy, corr_gas_change_energy, corr_prev_gas_market),
        'lr_r2': r2_score(y_test, y_pred),
        'granger': granger_df,
        'var_model': var_model
    }

# ---------- Entry point ----------
if __name__ == "__main__":
    results = run_pipeline(data_dir="/content", do_plots=False)
    print("Pipeline finished.")
