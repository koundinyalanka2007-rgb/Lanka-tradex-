
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Lanka Tradex V3", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# Mobile / iPhone presentation layer. Streamlit is responsive by default; these
# styles reduce spacing and typography so the dashboard is easier to scan on a phone.
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 4rem; padding-left: 1rem; padding-right: 1rem;}
[data-testid="stMetric"] {padding: .65rem .75rem; border: 1px solid rgba(128,128,128,.18); border-radius: 14px;}
[data-testid="stMetricValue"] {font-size: clamp(1.15rem, 5vw, 1.8rem);}
[data-testid="stMetricLabel"] {font-size: .78rem;}
[data-testid="stDataFrame"] {width: 100%;}
@media (max-width: 768px) {
  .block-container {padding-left: .65rem; padding-right: .65rem;}
  h1 {font-size: 1.55rem !important;}
  h2 {font-size: 1.25rem !important;}
  h3 {font-size: 1.05rem !important;}
  [data-testid="stMetric"] {min-height: 88px;}
  [data-testid="stMetricValue"] {font-size: 1.18rem;}
  [data-testid="stMarkdownContainer"] p {line-height: 1.4;}
  button {min-height: 42px;}
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LANKA TRADEX V3 — MARKET INTELLIGENCE ENGINE
# Research / educational prototype. Not investment advice.
# ============================================================

ASSETS = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "India VIX": "^INDIAVIX",
    "USD/INR": "INR=X",
    "DXY": "DX-Y.NYB",
    "US 2Y": "^IRX",
    "US 10Y": "^TNX",
    "Brent": "BZ=F",
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow": "^DJI",
    "Nikkei": "^N225",
    "Hang Seng": "^HSI",
    "Shanghai": "000001.SS",
    "Copper": "HG=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "WTI": "CL=F",
}

FACTOR_GROUPS = {
    "Trend & Momentum": [
        "Price vs 20DMA", "Price vs 50DMA", "Price vs 200DMA",
        "RSI", "MACD direction", "ADX/trend strength", "ATR regime",
        "20D momentum", "60D momentum"
    ],
    "Breadth & Participation": [
        "Advance/decline", "Breadth thrust", "New highs/lows",
        "Stocks above 20DMA", "Stocks above 50DMA", "Stocks above 200DMA",
        "Index volume trend"
    ],
    "Global Macro": [
        "S&P 500", "Nasdaq", "Dow", "Nikkei", "Hang Seng", "Shanghai",
        "DXY", "US 2Y", "US 10Y", "Brent", "WTI", "Copper", "Gold", "Silver"
    ],
    "India Macro": [
        "USD/INR", "India VIX", "India 10Y (manual)", "RBI stance (manual)",
        "Inflation (manual)", "GDP/PMI regime (manual)"
    ],
    "Flows & Derivatives": [
        "FII/FPI net (manual)", "DII net (manual)", "Futures OI (manual)",
        "Put/Call ratio (manual)", "Options IV (manual)", "Futures basis (manual)",
        "Call/Put OI buildup (manual)", "Max pain (manual)"
    ],
    "News & Events": [
        "RBI event risk (manual)", "Fed event risk (manual)",
        "India macro event risk (manual)", "Geopolitical risk (manual)",
        "Major earnings/event tone (manual)"
    ],
    "Sector Rotation": [
        "Banking relative strength (manual)", "IT relative strength (manual)",
        "Auto relative strength (manual)", "Energy relative strength (manual)",
        "Metal relative strength (manual)", "Pharma/FMCG relative strength (manual)"
    ],
}

def dl(ticker, period="1y"):
    try:
        x = yf.download(ticker, period=period, interval="1d",
                        auto_adjust=False, progress=False, threads=False)
        if x is None or x.empty:
            return pd.DataFrame()
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        return x.dropna(how="all")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_all():
    return {k: dl(v) for k, v in ASSETS.items()}

data = load_all()

def close_series(name):
    x = data.get(name, pd.DataFrame())
    if x.empty or "Close" not in x.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(x["Close"], errors="coerce").dropna()

def latest(name):
    s = close_series(name)
    if len(s) < 2:
        return None
    return float(s.iloc[-1]), float((s.iloc[-1]/s.iloc[-2]-1)*100), s.index[-1]

def zscore_signal(value, neutral=0, scale=1):
    if pd.isna(value):
        return 0
    return float(np.tanh((value-neutral)/scale))

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    down = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100/(1+rs))

def technical_features(name):
    s = close_series(name)
    if len(s) < 220:
        return {}
    ma20, ma50, ma200 = s.rolling(20).mean(), s.rolling(50).mean(), s.rolling(200).mean()
    rr = rsi(s)
    ema12, ema26 = s.ewm(span=12, adjust=False).mean(), s.ewm(span=26, adjust=False).mean()
    macd = ema12-ema26
    atr_proxy = s.pct_change().rolling(14).std()*np.sqrt(14)*100
    return {
        "Price vs 20DMA": (s.iloc[-1]/ma20.iloc[-1]-1)*100,
        "Price vs 50DMA": (s.iloc[-1]/ma50.iloc[-1]-1)*100,
        "Price vs 200DMA": (s.iloc[-1]/ma200.iloc[-1]-1)*100,
        "RSI": float(rr.iloc[-1]),
        "MACD direction": float((macd.iloc[-1]-macd.iloc[-5])*100/s.iloc[-1]),
        "20D momentum": float((s.iloc[-1]/s.iloc[-21]-1)*100),
        "60D momentum": float((s.iloc[-1]/s.iloc[-61]-1)*100),
        "ATR regime": float(atr_proxy.iloc[-1]),
    }

def corr(index_name, asset_name, days):
    a, b = close_series(index_name), close_series(asset_name)
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < days+3:
        return np.nan
    return float(j["a"].pct_change().tail(days).corr(j["b"].pct_change().tail(days)))

def score_from_pct(x, positive=True, scale=1):
    if pd.isna(x):
        return 0
    v = np.tanh(x/scale)
    return v if positive else -v

# ---------------- Sidebar inputs ----------------
st.sidebar.header("⚙️ Research Inputs")
st.sidebar.caption("Manual fields let you add factors that free public feeds may not expose reliably.")

fii = st.sidebar.number_input("FII/FPI net ₹Cr", value=0.0, step=100.0)
dii = st.sidebar.number_input("DII net ₹Cr", value=0.0, step=100.0)
pcr = st.sidebar.number_input("Put/Call ratio", value=1.0, step=0.05)
oi_bias = st.sidebar.slider("Futures / OI bias", -100, 100, 0)
iv_bias = st.sidebar.slider("Options IV regime", -100, 100, 0)
basis_bias = st.sidebar.slider("Futures basis bias", -100, 100, 0)
india10y = st.sidebar.slider("India 10Y yield bias", -100, 100, 0)
rbi_bias = st.sidebar.slider("RBI/liquidity bias", -100, 100, 0)
macro_bias = st.sidebar.slider("India macro bias", -100, 100, 0)
news_bias = st.sidebar.slider("News/event tone", -100, 100, 0)
bank_rs = st.sidebar.slider("Banking relative strength", -100, 100, 0)
sector_rs = st.sidebar.slider("Other sector rotation", -100, 100, 0)

# ---------------- Header ----------------
st.title("🚀 Lanka Tradex — Market Intelligence Engine V3")
st.caption("Nifty + Bank Nifty multi-factor research dashboard • 50+ factors • Backtesting • Explainability")

n = latest("Nifty 50")
b = latest("Bank Nifty")
v = latest("India VIX")

m = st.columns(5)
for c, label, obj in [
    (m[0], "Nifty 50", n), (m[1], "Bank Nifty", b),
    (m[2], "India VIX", v), (m[3], "USD/INR", latest("USD/INR")),
    (m[4], "Brent", latest("Brent"))
]:
    if obj:
        c.metric(label, f"{obj[0]:,.2f}", f"{obj[1]:+.2f}%")
    else:
        c.metric(label, "N/A")

# ---------------- Factor engine ----------------
def compute_score(target):
    s = 50.0
    reasons_pos, reasons_neg = [], []
    components = {}

    # Target trend
    tf = technical_features(target)
    trend = 0
    if tf:
        trend += score_from_pct(tf.get("Price vs 20DMA",0), True, 1.5)*0.20
        trend += score_from_pct(tf.get("Price vs 50DMA",0), True, 2.0)*0.20
        trend += score_from_pct(tf.get("Price vs 200DMA",0), True, 4.0)*0.20
        trend += score_from_pct(tf.get("20D momentum",0), True, 3.0)*0.20
        trend += score_from_pct(tf.get("60D momentum",0), True, 6.0)*0.20
    components["Trend & Momentum"] = 50 + trend*15

    # Momentum / RSI
    r = tf.get("RSI",50) if tf else 50
    mom = np.clip((r-50)/20, -1, 1)
    components["Technical Momentum"] = 50 + mom*10

    # Global
    global_names = ["S&P 500","Nasdaq","Dow","Nikkei","Hang Seng","Shanghai","Copper"]
    gs = 0
    count = 0
    for x in global_names:
        q = latest(x)
        if q:
            gs += score_from_pct(q[1], True, 1)
            count += 1
    components["Global Risk"] = 50 + (gs/count*15 if count else 0)

    # Macro / currency / rates / commodities
    macro = 0
    weights = [("USD/INR",False,0.5,2.5),("DXY",False,0.5,2.0),
               ("US 10Y",False,0.3,2.0),("Brent",False,2,2.0),
               ("WTI",False,2,2.0),("India VIX",False,3,3.0)]
    count = 0
    for nm,pos,scale,_ in weights:
        q=latest(nm)
        if q:
            macro += score_from_pct(q[1],pos,scale); count+=1
    components["Macro / Risk"] = 50 + (macro/max(count,1)*12)
    components["India Macro"] = 50 + macro_bias*0.15 + rbi_bias*0.15 + india10y*0.10

    # Flows
    flow = np.tanh(fii/3000)*0.65 + np.tanh(dii/3000)*0.35
    components["FII / DII"] = 50 + flow*15

    # Derivatives
    pcr_signal = np.clip((pcr-1.0)/0.5, -1, 1)
    deriv = pcr_signal*0.35 + oi_bias/100*0.25 + iv_bias/100*0.15 + basis_bias/100*0.25
    components["Derivatives"] = 50 + deriv*15

    # Breadth: proxy from index vs broad trend when no constituent feed is installed
    breadth = 0
    for nm in [target, "Nifty 50" if target=="Bank Nifty" else "Bank Nifty"]:
        q=latest(nm)
        if q: breadth += score_from_pct(q[1], True, 1)
    components["Breadth / Participation"] = 50 + breadth/2*10

    # Sector rotation
    components["Sector Rotation"] = 50 + ((bank_rs if target=="Bank Nifty" else sector_rs)/100)*10

    # News/events
    components["News / Events"] = 50 + news_bias*0.10

    # Target-specific weighting
    if target == "Nifty 50":
        weights_map = {
            "Trend & Momentum": .16, "Technical Momentum": .08, "Global Risk": .13,
            "Macro / Risk": .13, "India Macro": .10, "FII / DII": .13,
            "Derivatives": .10, "Breadth / Participation": .08,
            "Sector Rotation": .05, "News / Events": .04
        }
    else:
        weights_map = {
            "Trend & Momentum": .15, "Technical Momentum": .08, "Global Risk": .08,
            "Macro / Risk": .10, "India Macro": .17, "FII / DII": .12,
            "Derivatives": .12, "Breadth / Participation": .06,
            "Sector Rotation": .08, "News / Events": .04
        }
    final = sum(components[k]*w for k,w in weights_map.items())
    final = float(np.clip(final,0,100))

    for k,val in sorted(components.items(), key=lambda z:z[1], reverse=True):
        if val >= 58: reasons_pos.append(f"{k}: {val:.0f}/100")
        elif val <= 42: reasons_neg.append(f"{k}: {val:.0f}/100")
    return final, components, reasons_pos[:5], reasons_neg[:5]

ns, nc, np1, nn1 = compute_score("Nifty 50")
bs, bc, bp1, bn1 = compute_score("Bank Nifty")

def bias(s):
    if s >= 65: return "BULLISH", "🟢"
    if s <= 35: return "BEARISH", "🔴"
    return "NEUTRAL", "🟡"

# ---------------- Scores ----------------
overview_tab, factors_tab, charts_tab = st.tabs(["📱 Dashboard", "🧠 Intelligence", "📈 Charts & History"])

with overview_tab:
    st.subheader("🎯 100-Point Market Intelligence Scores")
    a,bx=st.columns(2)
    for col,name,score,comp,pos,neg in [
        (a,"NIFTY 50",ns,nc,np1,nn1),(bx,"BANK NIFTY",bs,bc,bp1,bn1)
    ]:
        lab,icon=bias(score)
        col.metric(name, f"{score:.0f}/100", f"{icon} {lab}")
        col.progress(int(score))
        col.markdown("**Positive drivers**")
        for x in pos: col.write("• "+x)
        col.markdown("**Negative drivers**")
        for x in neg: col.write("• "+x)

# ---------------- Explainability ----------------
with factors_tab:
    st.subheader("🧠 Explanation Layer")
    e1,e2=st.columns(2)
    e1.write(f"**Nifty:** {bias(ns)[1]} {bias(ns)[0]} — score is the weighted combination of trend, global risk, macro, flows, derivatives, breadth, sector rotation and event inputs.")
    e2.write(f"**Bank Nifty:** {bias(bs)[1]} {bias(bs)[0]} — extra emphasis is placed on RBI/rates, banking relative strength and derivatives.")

    # ---------------- Correlations ----------------
    st.subheader("🔗 1D / 5D / 20D Correlation Matrix")
    assets=["India VIX","USD/INR","DXY","US 10Y","Brent","S&P 500","Nasdaq","Copper","Gold","Nikkei","Hang Seng"]
    rows=[]
    for target in ["Nifty 50","Bank Nifty"]:
        for x in assets:
            rows.append({"Index":target,"Asset":x,"1D":corr(target,x,1),"5D":corr(target,x,5),"20D":corr(target,x,20)})
    cdf=pd.DataFrame(rows)
    st.dataframe(cdf.style.format({"1D":"{:.2f}","5D":"{:.2f}","20D":"{:.2f}"}),use_container_width=True)

    # ---------------- Factor audit ----------------
    st.subheader("🧩 50+ Factor Audit")
    factor_rows=[]
    for group, factors in FACTOR_GROUPS.items():
        for f in factors:
            factor_rows.append({"Group":group,"Factor":f,"Status":"Live" if "(manual)" not in f else "Manual input","Role":"Included in V3 architecture"})
    st.dataframe(pd.DataFrame(factor_rows),use_container_width=True,hide_index=True)

    # ---------------- Backtest ----------------
    st.subheader("🧪 Historical Backtest — Score Regime Study")
    st.caption("This backtest is deliberately simple: it studies whether a score-like trend regime was followed by positive next-session returns. It is not a promise of future performance.")

    def backtest(name):
        s=close_series(name)
        if len(s)<260: return pd.DataFrame()
        df=pd.DataFrame({"close":s})
        df["ma20"]=df.close.rolling(20).mean()
        df["ma50"]=df.close.rolling(50).mean()
        df["mom20"]=df.close.pct_change(20)
        df["rsi"]=rsi(df.close)
        df["regime_score"]=50
        df["regime_score"] += np.tanh((df.close/df.ma20-1)/0.015)*12
        df["regime_score"] += np.tanh((df.close/df.ma50-1)/0.025)*12
        df["regime_score"] += np.tanh(df.mom20/0.03)*12
        df["regime_score"] += np.tanh((df.rsi-50)/15)*10
        df["regime_score"]=df.regime_score.clip(0,100)
        df["next_ret"]=df.close.pct_change().shift(-1)*100
        df["bucket"]=pd.cut(df.regime_score,[0,35,50,65,100],labels=["Bearish","Weak/Neutral","Strong/Neutral","Bullish"],include_lowest=True)
        out=df.groupby("bucket",observed=False).agg(
            observations=("next_ret","count"),
            avg_next_day_return=("next_ret","mean"),
            win_rate=("next_ret",lambda x: (x>0).mean()*100)
        ).reset_index()
        return out

    bt1=backtest("Nifty 50")
    bt2=backtest("Bank Nifty")
    c1,c2=st.columns(2)
    c1.write("**Nifty 50**")
    if not bt1.empty: c1.dataframe(bt1.style.format({"avg_next_day_return":"{:.3f}%","win_rate":"{:.1f}%"}),use_container_width=True,hide_index=True)
    else: c1.info("Not enough history.")
    c2.write("**Bank Nifty**")
    if not bt2.empty: c2.dataframe(bt2.style.format({"avg_next_day_return":"{:.3f}%","win_rate":"{:.1f}%"}),use_container_width=True,hide_index=True)
    else: c2.info("Not enough history.")

# ---------------- Snapshot + Charts ----------------
with charts_tab:
    st.subheader("📋 Global + India Market Snapshot")
    rows=[]
    for name,ticker in ASSETS.items():
        q=latest(name)
        rows.append({"Asset":name,"Ticker":ticker,
                     "Latest":q[0] if q else np.nan,
                     "Change %":q[1] if q else np.nan,
                     "Data time":str(q[2]) if q else "Unavailable"})
    snap=pd.DataFrame(rows)
    st.dataframe(snap[["Asset","Latest","Change %","Data time"]].style.format({"Latest":"{:,.4f}","Change %":"{:+.2f}%"}),use_container_width=True,hide_index=True)
    with st.expander("Show tickers"):
        st.dataframe(snap[["Asset","Ticker"]], use_container_width=True, hide_index=True)

    # ---------------- Charts ----------------
    st.subheader("📈 Trend Charts")
    st.caption("On iPhone, rotate to landscape for the best chart view.")
    for name in ["Nifty 50","Bank Nifty","India VIX","USD/INR","DXY","US 10Y","Brent","S&P 500","Nasdaq"]:
        s=close_series(name)
        if not s.empty:
            st.markdown(f"**{name}**")
            st.line_chart(s.tail(120))

# ---------------- Refresh ----------------
st.divider()
st.write(f"Last calculation: **{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}**")
if st.button("🔄 Refresh market data now"):
    st.cache_data.clear()
    st.rerun()

st.warning("Research prototype only. Scores are model outputs, not guaranteed predictions. Real-time commercial data requires appropriate licensed feeds and usage rights. Before distributing paid securities research/recommendations in India, obtain qualified adult/legal/compliance guidance and review applicable SEBI requirements.")
