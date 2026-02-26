import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 BEI Single Stock Strategy Research Engine")

# ======================
# INPUT
# ======================

ticker = st.text_input("Masukkan ticker saham (contoh: BBRI.JK)", "BBRI.JK")
years = st.slider("Years of historical data", 2, 10, 10)
hold_days = st.slider("Holding Period (days)", 1, 10, 5)
stop_loss = st.slider("Stop Loss (%)", 1, 10, 3) / 100

run = st.button("🚀 Analyze All Strategies")

# ======================
# 1. FETCH DATA
# ======================

@st.cache_data
def fetch_data(ticker, years=10):
    if not ticker.upper().endswith(".JK"):
        ticker = ticker.upper() + ".JK"

    df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = df.columns.str.lower()
    df = df.dropna().copy()
    df = df.sort_index()
    return df

# ======================
# 2. INDICATOR BUILDER
# ======================

def indicator_builder(df):
    df = df.copy()

    df["sma5"]   = df["close"].rolling(5).mean()
    df["sma20"]  = df["close"].rolling(20).mean()
    df["sma50"]  = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()

    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_std"]   = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_bw"]    = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    df["value"] = df["close"] * df["volume"]
    df["llv5"]  = df["low"].shift(1).rolling(5).min()

    df["prev_close"]   = df["close"].shift(1)
    df["prev_volume"]  = df["volume"].shift(1)
    df["prev_high"]    = df["high"].shift(1)
    df["prev_low"]     = df["low"].shift(1)
    df["prev_2_close"] = df["close"].shift(2)
    df["prev_3_close"] = df["close"].shift(3)
    df["prev_2_high"]  = df["high"].shift(2)

    df = df.dropna()
    return df

# ======================
# 3. STRATEGY FUNCTIONS
# ======================

def strategy_functions(df):
    signals = {}

    signals["MA50 Bounce"] = (
        (df["llv5"] > df["sma50"]) &
        (df["close"] >= df["sma50"] * 0.99) &
        (df["close"] <= df["sma50"] * 1.02) &
        (df["value"] > 1_000_000_000)
    )

    signals["MA200 Bounce"] = (
        (df["llv5"] > df["sma200"]) &
        (df["close"] >= df["sma200"] * 0.99) &
        (df["close"] <= df["sma200"] * 1.02) &
        (df["value"] > 1_000_000_000)
    )

    signals["Bollinger Mid"] = (
        (df["close"] >= df["bb_mid"] * 0.98) &
        (df["close"] <= df["bb_mid"] * 1.02) &
        (df["close"] > df["bb_mid"]) &
        (df["ema10"] > df["ema20"]) &
        (df["ema20"] > df["ema50"]) &
        (df["bb_bw"] >= 0.1) &
        (df["value"] > 1_000_000_000)
    )

    signals["BB Bottom Reversal"] = (
        (df["prev_low"] < df["bb_lower"].shift(1)) &
        (df["prev_close"] < df["bb_lower"].shift(1)) &
        (df["close"] > df["bb_lower"]) &
        (df["volume"] > df["prev_volume"]) &
        (df["high"] > df["prev_high"]) &
        (df["close"] > df["prev_close"]) &
        (df["value"] > 1_000_000_000)
    )

    signals["V1.1 / V1.3"] = (
        (df["volume"] > df["prev_volume"]) &
        (df["close"] > df["prev_close"]) &
        (df["close"] > df["sma5"]) &
        (df["value"] > 5_000_000_000)
    )

    signals["V1.2"] = (
        (df["close"] > df["sma5"]) &
        (df["prev_close"] > df["sma5"]) &
        (df["prev_2_close"] > df["sma5"]) &
        (df["prev_2_high"] / df["prev_3_close"] >= 1.10) &
        (df["prev_close"] < df["prev_2_close"]) &
        (df["close"] < df["prev_close"]) &
        (df["value"] > 1_000_000_000)
    )

    signals["V2.1 / V2.2"] = (
        (df["volume"] > df["prev_volume"]) &
        (df["close"] > df["prev_close"]) &
        (df["close"] > df["sma5"]) &
        (df["high"] / df["prev_close"] >= 1.10) &
        (df["value"] > 5_000_000_000)
    )

    return signals

# ======================
# 4. BACKTEST ENGINE
# ======================

def backtest_engine(df, signal_series, hold_days=5, stop_loss=0.03):
    trades = []
    signal_dates = df.index[signal_series.reindex(df.index, fill_value=False)]

    for entry_date in signal_dates:
        try:
            entry_idx = df.index.get_loc(entry_date)
        except KeyError:
            continue

        entry_price = df["close"].iloc[entry_idx]
        exit_price = None
        exit_date = None
        exit_reason = "Hold"

        for i in range(1, hold_days + 1):
            if entry_idx + i >= len(df):
                break
            day_close = df["close"].iloc[entry_idx + i]
            ret = (day_close - entry_price) / entry_price

            if ret <= -stop_loss:
                exit_price = day_close
                exit_date = df.index[entry_idx + i]
                exit_reason = "Stop Loss"
                break

            if i == hold_days:
                exit_price = day_close
                exit_date = df.index[entry_idx + i]
                exit_reason = "Hold Exit"

        if exit_price is None:
            continue

        ret_pct = (exit_price - entry_price) / entry_price
        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "return_pct": round(ret_pct * 100, 3),
            "status": "Win" if ret_pct > 0 else "Loss",
            "exit_reason": exit_reason
        })

    return pd.DataFrame(trades)

# ======================
# 5. CALCULATE METRICS
# ======================

def calculate_metrics(trades_df):
    if trades_df.empty or len(trades_df) == 0:
        return {
            "total_trades": 0, "winrate": 0, "avg_win": 0,
            "avg_loss": 0, "expectancy": 0, "sharpe": 0, "max_drawdown": 0
        }

    rets = trades_df["return_pct"] / 100
    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    total_trades = len(trades_df)
    winrate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    expectancy = (winrate * avg_win) + ((1 - winrate) * avg_loss)

    sharpe = 0
    if len(rets) > 1 and rets.std() != 0:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252)

    equity = (1 + rets).cumprod()
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_drawdown = drawdown.min()

    return {
        "total_trades": total_trades,
        "winrate": round(winrate * 100, 2),
        "avg_win": round(avg_win * 100, 3),
        "avg_loss": round(avg_loss * 100, 3),
        "expectancy": round(expectancy * 100, 3),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_drawdown * 100, 2)
    }

# ======================
# 6. RANKING ENGINE
# ======================

def ranking_engine(all_metrics):
    rows = []
    for strategy_name, m in all_metrics.items():
        if m["total_trades"] < 10:
            score = -999
        else:
            score = (
                (m["sharpe"] * 0.5) +
                (m["expectancy"] * 0.3) -
                (abs(m["max_drawdown"]) * 0.2)
            )
        rows.append({
            "Strategy": strategy_name,
            "Total Trades": m["total_trades"],
            "Winrate (%)": m["winrate"],
            "Avg Win (%)": m["avg_win"],
            "Avg Loss (%)": m["avg_loss"],
            "Expectancy (%)": m["expectancy"],
            "Sharpe": m["sharpe"],
            "Max DD (%)": m["max_drawdown"],
            "Score": round(score, 4)
        })

    df_rank = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    df_rank.index += 1
    return df_rank

# ======================
# SESSION STATE INIT
# ======================

if "results" not in st.session_state:
    st.session_state.results = None

# ======================
# RUN ANALYSIS
# ======================

if run:
    with st.spinner("Mengambil data dan menjalankan backtest semua strategi..."):
        df_raw = fetch_data(ticker, years=years)

        if df_raw is None:
            st.error("❌ Data tidak ditemukan. Pastikan ticker benar.")
            st.stop()

        df = indicator_builder(df_raw)
        signals = strategy_functions(df)

        all_trades = {}
        all_metrics = {}

        for name, signal_series in signals.items():
            trades_df = backtest_engine(df, signal_series, hold_days=hold_days, stop_loss=stop_loss)
            all_trades[name] = trades_df
            all_metrics[name] = calculate_metrics(trades_df)

        ranking_df = ranking_engine(all_metrics)
        best_strategy = ranking_df.iloc[0]["Strategy"]
        best_metrics = all_metrics[best_strategy]

        # Simpan ke session state
        st.session_state.results = {
            "df": df,
            "df_raw": df_raw,
            "signals": signals,
            "all_trades": all_trades,
            "all_metrics": all_metrics,
            "ranking_df": ranking_df,
            "best_strategy": best_strategy,
            "best_metrics": best_metrics,
        }

# ======================
# TAMPILKAN HASIL
# ======================

if st.session_state.results is not None:
    df            = st.session_state.results["df"]
    df_raw        = st.session_state.results["df_raw"]
    signals       = st.session_state.results["signals"]
    all_trades    = st.session_state.results["all_trades"]
    all_metrics   = st.session_state.results["all_metrics"]
    ranking_df    = st.session_state.results["ranking_df"]
    best_strategy = st.session_state.results["best_strategy"]
    best_metrics  = st.session_state.results["best_metrics"]

    # ─────────────────────────────────────────
    # SECTION 1 – STRATEGY DECISION PANEL
    # ─────────────────────────────────────────
    st.header("🏆 Strategy Decision Panel")

    sharpe_best = best_metrics["sharpe"]
    if sharpe_best > 1.5:
        verdict = "🔥 Strong Risk-Adjusted Edge"
        color = "success"
    elif sharpe_best > 1.0:
        verdict = "✅ Good Edge – Worth Trading"
        color = "success"
    elif sharpe_best > 0.5:
        verdict = "⚠️ Weak Positive Edge – Trade with Caution"
        color = "warning"
    elif sharpe_best > 0:
        verdict = "😐 Marginal Edge – Needs Improvement"
        color = "warning"
    else:
        verdict = "❌ No Statistical Edge"
        color = "error"

    if color == "success":
        st.success(f"Best Strategy: **{best_strategy}** → {verdict}")
    elif color == "warning":
        st.warning(f"Best Strategy: **{best_strategy}** → {verdict}")
    else:
        st.error(f"Best Strategy: **{best_strategy}** → {verdict}")

    if best_metrics["total_trades"] >= 50:
        confidence = "High Statistical Confidence"
    elif best_metrics["total_trades"] >= 20:
        confidence = "Moderate Confidence"
    else:
        confidence = "Low Confidence"

    st.info(f"Confidence Level: {confidence} ({best_metrics['total_trades']} trades)")

    if best_metrics["expectancy"] > 0 and best_metrics["sharpe"] > 1:
        insight = "Strategy shows stable positive expectancy and strong risk-adjusted return."
    elif best_metrics["expectancy"] > 0:
        insight = "Strategy profitable but risk profile needs monitoring."
    else:
        insight = "No consistent profitability detected."

    st.caption(f"💡 Insight: {insight}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Strategy", best_strategy)
    col2.metric("Sharpe Ratio", best_metrics["sharpe"])
    col3.metric("Winrate (%)", best_metrics["winrate"])
    col4.metric("Expectancy (%)", best_metrics["expectancy"])
    col5.metric("Max Drawdown (%)", best_metrics["max_drawdown"])

    # ─────────────────────────────────────────
    # SECTION 2 – RANKING TABLE
    # ─────────────────────────────────────────
    st.header("📋 Strategy Ranking Table")
    st.dataframe(ranking_df, use_container_width=True)

    csv = ranking_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Ranking CSV", csv, "ranking.csv", "text/csv")

    # ─────────────────────────────────────────
    # SECTION 3 – DETAIL PER STRATEGY
    # ─────────────────────────────────────────
    st.header("🔍 Detail Per Strategy")

    for _, row in ranking_df.iterrows():
        name = row["Strategy"]
        m = all_metrics[name]
        trades = all_trades[name]

        with st.expander(f"📌 {name} — Sharpe: {m['sharpe']} | Winrate: {m['winrate']}% | Trades: {m['total_trades']}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Trades", m["total_trades"])
            c2.metric("Winrate (%)", m["winrate"])
            c3.metric("Expectancy (%)", m["expectancy"])
            c4.metric("Max DD (%)", m["max_drawdown"])

            if not trades.empty:
                st.subheader("Trade Log")
                st.dataframe(
                    trades[["entry_date", "exit_date", "entry_price", "exit_price", "return_pct", "status", "exit_reason"]],
                    use_container_width=True
                )
            else:
                st.info("Tidak ada trade untuk strategi ini pada periode data.")

    # ─────────────────────────────────────────
    # SECTION 4 – EQUITY CURVE
    # ─────────────────────────────────────────
    st.header("📈 Equity Curve – Best Strategy vs Buy & Hold")

    best_trades = all_trades[best_strategy]

    if not best_trades.empty:
        rets = best_trades["return_pct"] / 100
        equity_strategy = (1 + rets).cumprod().reset_index(drop=True)

        bh_returns = df_raw["close"].pct_change().dropna()
        bh_equity = (1 + bh_returns).cumprod().reset_index(drop=True)

        min_len = min(len(equity_strategy), len(bh_equity))
        chart_df = pd.DataFrame({
            "Strategy Equity": equity_strategy.values[:min_len],
            "Buy & Hold": bh_equity.values[:min_len]
        })

        st.line_chart(chart_df)
    else:
        st.info("Tidak ada data equity untuk ditampilkan.")

    # ─────────────────────────────────────────
    # SECTION 5 – SIGNAL VISUALIZATION ON PRICE CHART
    # ─────────────────────────────────────────
    st.header("🕯️ Signal Visualization on Price Chart")

    strategy_selected = st.selectbox(
        "Pilih strategi untuk ditampilkan sinyalnya:",
        options=list(signals.keys())
    )

    sig = signals[strategy_selected].reindex(df.index, fill_value=False)
    signal_dates = df.index[sig]
    signal_prices = df.loc[signal_dates, "close"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#4fa3e0", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=signal_dates,
        y=signal_prices,
        mode="markers",
        name=f"Signal: {strategy_selected}",
        marker=dict(
            symbol="triangle-up",
            size=10,
            color="lime",
            line=dict(color="green", width=1)
        )
    ))

    fig.update_layout(
        title=f"{ticker.upper()} – {strategy_selected} Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.05)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Total sinyal terdeteksi: {len(signal_dates)}")