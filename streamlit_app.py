import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime

st.set_page_config(
    page_title="Return Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f0f13;
        color: white;
    }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0f0f13; }
    ::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #4361ee; }

    .gradient-text {
        background: linear-gradient(90deg, #4361ee 0%, #06d6a0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }

    .subtitle {
        color: #888899;
        text-align: center;
        font-weight: 300;
        margin-top: 0px;
        margin-bottom: 20px;
    }

    hr { border-color: #2a2a3a; margin-bottom: 30px; }

    .stTextInput > div > div > input {
        background-color: #1a1a24 !important;
        color: white !important;
        border: 1px solid #2a2a3a !important;
        border-radius: 8px !important;
    }

    .stDateInput > div > div > input {
        background-color: #1a1a24 !important;
        color: white !important;
        border: 1px solid #2a2a3a !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        background-color: #4361ee !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        width: 100%;
        height: 42px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #5b76ff !important;
        box-shadow: 0 0 15px rgba(67, 97, 238, 0.4);
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #888899;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    .metric-value { font-size: 1.25rem; font-weight: 800; }
    .metric-delta { font-size: 0.72rem; margin-top: 5px; color: #888899; }

    .custom-table-wrapper {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2a2a3a;
        margin-top: 20px;
        margin-bottom: 30px;
    }
    table.custom-table {
        width: 100%;
        border-collapse: collapse;
        color: white;
        text-align: right;
    }
    table.custom-table th {
        background-color: #4361ee;
        color: white;
        padding: 12px 15px;
        font-weight: 600;
    }
    table.custom-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #2a2a3a;
    }
    table.custom-table tr:nth-child(even) { background-color: #1a1a24; }
    table.custom-table tr:nth-child(odd)  { background-color: #13131c; }
    table.custom-table th:first-child,
    table.custom-table td:first-child { text-align: left; }

    /* Fix Streamlit's default white bg boxes */
    .block-container { background-color: #0f0f13; }
    section[data-testid="stSidebar"] { background-color: #0f0f13; }
    .stExpander { background-color: #1a1a24; border: 1px solid #2a2a3a; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="gradient-text">Overnight vs Intraday Return Analyzer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Replicating Knuteson (2023) — <em>Nothing to See Here</em></div>',
            unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ── Input Row ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    ticker_input = st.text_input("Ticker", value="AAPL",
                                 placeholder="e.g. AAPL, RELIANCE.NS, ^NSEI",
                                 label_visibility="collapsed")
with col2:
    dates = st.date_input(
        "Date Range",
        value=(datetime.date(1990, 1, 1), datetime.date.today()),
        max_value=datetime.date.today(),
        label_visibility="collapsed"
    )
with col3:
    analyze_btn = st.button("Analyze")

# ── Data Functions ────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_and_process(ticker, start_date, end_date):
    raw = yf.download(ticker, start=start_date, end=end_date,
                      progress=False, auto_adjust=False)
    if raw is None or len(raw) == 0:
        return None, None

    # Flatten MultiIndex if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    needed = ['Open', 'Close', 'Adj Close']
    if not all(c in raw.columns for c in needed):
        return None, None

    df = raw[needed].copy()
    df = df.dropna()
    if len(df) < 2:
        return None, None

    # ── Knuteson Raw ──────────────────────────────────────────────────────────
    df['overnight_r'] = (df['Open'] - df['Close'].shift(1)
                         ) / df['Close'].shift(1)
    df['intraday_r'] = (df['Close'] - df['Open']) / df['Open']
    df['overnight_r'] = df['overnight_r'].fillna(0)
    df['intraday_r'] = df['intraday_r'].fillna(0)

    df['cumul_overnight'] = (1 + df['overnight_r']).cumprod() - 1
    df['cumul_intraday'] = (1 + df['intraday_r']).cumprod() - 1

    # ── Adjusted ──────────────────────────────────────────────────────────────
    df['adj_factor'] = df['Adj Close'] / df['Close']
    df['total_r'] = df['Adj Close'].pct_change().fillna(0)
    df['adj_factor_overnight'] = (
        df['Adj Close'] / df['Close']) / (df['Adj Close'].shift(1) / df['Close'].shift(1))
    df['adj_overnight'] = (1 + df['overnight_r']) * \
        df['adj_factor_overnight'] - 1
    df['adj_intraday'] = df['total_r'] - df['adj_overnight']

    df['cumul_adj_overnight'] = (1 + df['adj_overnight']).cumprod() - 1
    df['cumul_adj_intraday'] = (1 + df['adj_intraday']).cumprod() - 1

    # ── Yearly ────────────────────────────────────────────────────────────────
    df['Year'] = df.index.year
    yearly = (df.groupby('Year')
                .apply(lambda x: pd.Series({
                    'Overnight': (1 + x['overnight_r']).prod() - 1,
                    'Intraday':  (1 + x['intraday_r']).prod() - 1,
                }), include_groups=False)
              .reset_index())

    return df, yearly


def calc_metrics(df):
    n = len(df)
    yrs = n / 252

    def safe_ann(c): return (1+c)**(1/yrs)-1 if yrs > 0 and (1+c) > 0 else 0
    def sharpe(r): return (r.mean()/r.std())*np.sqrt(252) if r.std() > 0 else 0

    def maxdd(c):
        wealth = 1 + c
        peak = wealth.cummax()
        return (wealth/peak - 1).min()

    cum_o = df['cumul_overnight'].iloc[-1]
    cum_i = df['cumul_intraday'].iloc[-1]
    cum_ao = df['cumul_adj_overnight'].iloc[-1]
    cum_ai = df['cumul_adj_intraday'].iloc[-1]

    div = abs(cum_o)/(abs(cum_o)+abs(cum_i)
                      ) if (abs(cum_o)+abs(cum_i)) > 0 else 0

    return dict(
        n=n, yrs=yrs,
        start_yr=df.index[0].year, end_yr=df.index[-1].year,
        cum_o=cum_o,   cum_i=cum_i,
        ann_o=safe_ann(cum_o),  ann_i=safe_ann(cum_i),
        sh_o=sharpe(df['overnight_r']), sh_i=sharpe(df['intraday_r']),
        win_o=(df['overnight_r'] > 0).mean(), win_i=(df['intraday_r'] > 0).mean(),
        dd_o=maxdd(df['cumul_overnight']), dd_i=maxdd(df['cumul_intraday']),
        div=div,
        cum_ao=cum_ao, cum_ai=cum_ai,
        ann_ao=safe_ann(cum_ao), ann_ai=safe_ann(cum_ai),
        sh_ao=sharpe(df['adj_overnight']), sh_ai=sharpe(df['adj_intraday']),
    )


# ── Main ─────────────────────────────────────────────────────────────────────
run = analyze_btn or ('last_ticker' not in st.session_state)

if run:
    ticker = ticker_input.upper().strip()
    st.session_state['last_ticker'] = ticker

    if not isinstance(dates, (list, tuple)) or len(dates) != 2:
        st.warning("Please select both a start and end date.")
        st.stop()

    start_d, end_d = dates

    with st.spinner(f"Fetching {ticker} from Yahoo Finance…"):
        try:
            df, yearly = fetch_and_process(ticker, start_d, end_d)
        except Exception as e:
            st.error(f"Network error: {e}")
            st.stop()

    if df is None:
        st.markdown(f"""
        <div style="background:#8a0011;padding:18px;border-radius:10px;margin-bottom:20px;">
            <strong>❌ Ticker '{ticker}' not found or returned no data.</strong><br>
            Try: AAPL, TSLA, MSFT, RELIANCE.NS, TCS.NS, ^NSEI
        </div>""", unsafe_allow_html=True)
        st.stop()

    if len(df) < 252:
        st.markdown("""
        <div style="background:#b58900;color:#000;padding:15px;border-radius:10px;margin-bottom:20px;">
            <strong>⚠ Warning:</strong> Less than 1 year of data — results may not be statistically meaningful.
        </div>""", unsafe_allow_html=True)

    m = calc_metrics(df)

    # Company info
    try:
        info = yf.Ticker(ticker).info
        company = info.get('longName', ticker)
        exchange = info.get('exchange', '')
    except:
        company, exchange = ticker, ''

    # ── Verdict Banner ────────────────────────────────────────────────────────
    d = m['div']
    if d > 0.85:
        bg, msg = "#8a0011", "⚠️ Strong overnight bias detected — historically suspicious pattern"
    elif d > 0.70:
        bg, msg = "#7a5c00", "🟡 Moderate overnight bias — worth investigating"
    elif d > 0.55:
        bg, msg = "#1e3a8a", "🔵 Slight overnight tilt — within plausible range"
    else:
        bg, msg = "#064e3b", "✅ Returns appear balanced"

    st.markdown(f"""
    <div style="background:{bg};padding:20px;border-radius:12px;margin-bottom:25px;
                border:1px solid rgba(255,255,255,0.1);">
        <h3 style="margin:0;color:white;">{msg}</h3>
        <p style="margin:5px 0 0;color:rgba(255,255,255,0.75);font-size:.9rem;">
            {ticker} &nbsp;|&nbsp; {exchange} &nbsp;|&nbsp; {company}
            &nbsp;&nbsp;·&nbsp;&nbsp; Divergence Ratio: <strong>{d:.3f}</strong>
        </p>
    </div>""", unsafe_allow_html=True)

    # ── Metric Cards ──────────────────────────────────────────────────────────
    def card(label, val, delta="", color="#ffffff"):
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-delta">{delta}</div>
        </div>"""

    def pct(v, big=False):
        fmt = f"{v:+,.1f}%" if big else f"{v:+.2%}"
        return fmt

    cols = st.columns(8)
    cards_data = [
        ("Cumul. Overnight",  pct(m['cum_o']*100, True),
         f"ann {m['ann_o']:+.2%}", "#4361ee"),
        ("Cumul. Intraday",   pct(m['cum_i']*100, True),  f"ann {m['ann_i']:+.2%}",
            "#06d6a0" if m['cum_i'] >= 0 else "#ef233c"),
        ("Overnight Ann.",    f"{m['ann_o']:+.2%}", "", "#ffffff"),
        ("Intraday Ann.",     f"{m['ann_i']:+.2%}", "",
            "#06d6a0" if m['ann_i'] >= 0 else "#ef233c"),
        ("Overnight Sharpe",  f"{m['sh_o']:.2f}", "", "#ffffff"),
        ("Intraday Sharpe",   f"{m['sh_i']:.2f}", "",
            "#06d6a0" if m['sh_i'] >= 0 else "#ef233c"),
        ("Divergence Ratio",  f"{m['div']:.3f}", "1.0 = all overnight",
            "#ef233c" if m['div'] > 0.85 else ("#ffd166" if m['div'] > 0.7 else "#06d6a0")),
        ("Data Range",
         f"{m['start_yr']}–{m['end_yr']}", f"{m['n']:,} days", "#888899"),
    ]
    for col, (lbl, val, dlt, clr) in zip(cols, cards_data):
        with col:
            st.markdown(card(lbl, val, dlt, clr), unsafe_allow_html=True)

    # ── Chart helper ─────────────────────────────────────────────────────────
    DARK = "#0f0f13"
    GRID = "#1e1e2e"

    def cum_chart(df, title, sub, col_o, col_i, c_o, c_i, height=540):
        yo = (1 + df[col_o]).clip(lower=1e-6)
        yi = (1 + df[col_i]).clip(lower=1e-6)

        def rgba(hex6, a):
            r, g, b = int(hex6[1:3], 16), int(
                hex6[3:5], 16), int(hex6[5:7], 16)
            return f"rgba({r},{g},{b},{a})"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=yo, mode='lines',
            name='Overnight (Close→Open)',
            line=dict(color=c_o, width=2),
            fill='tozeroy', fillcolor=rgba(c_o, 0.08),
            hovertemplate='%{x|%Y-%m-%d}<br>Overnight: %{customdata:.2%}<extra></extra>',
            customdata=df[col_o].values,
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=yi, mode='lines',
            name='Intraday (Open→Close)',
            line=dict(color=c_i, width=2),
            fill='tozeroy', fillcolor=rgba(c_i, 0.08),
            hovertemplate='%{x|%Y-%m-%d}<br>Intraday: %{customdata:.2%}<extra></extra>',
            customdata=df[col_i].values,
        ))

        # End-of-series annotations
        last = df.index[-1]
        vo, vi = df[col_o].iloc[-1], df[col_i].iloc[-1]
        fig.add_annotation(x=last, y=float(yo.iloc[-1]),
                           text=f"{vo:+,.1%}", showarrow=True, arrowhead=2, ax=-50, ay=-30,
                           font=dict(color=c_o, size=11), arrowcolor=c_o, yref="y")
        fig.add_annotation(x=last, y=float(yi.iloc[-1]),
                           text=f"{vi:+,.1%}", showarrow=True, arrowhead=2, ax=-50, ay=30,
                           font=dict(color=c_i, size=11), arrowcolor=c_i, yref="y")

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><span style='font-size:12px;color:#888899'>{sub}</span>",
                x=0.02, y=0.97, font=dict(size=15)
            ),
            paper_bgcolor=DARK, plot_bgcolor=DARK,
            font=dict(color="white", family="Inter"),
            yaxis=dict(type="log", title="Value of $1 Invested (log)",
                       gridcolor=GRID, zerolinecolor=GRID, tickformat=".2f"),
            xaxis=dict(title="Date", gridcolor=GRID, zerolinecolor=GRID),
            legend=dict(x=0.02, y=0.12, bgcolor="rgba(15,15,19,0.85)",
                        bordercolor="#2a2a3a", borderwidth=1),
            hovermode="x unified",
            height=height,
            margin=dict(l=50, r=30, t=70, b=40),
        )
        return fig

    # ── Charts 1 & 2 ─────────────────────────────────────────────────────────
    fig1 = cum_chart(df,
        f"{ticker} — Knuteson Method (Raw Prices)",
        "Overnight: close→open  |  Intraday: open→close",
        'cumul_overnight', 'cumul_intraday', '#4361ee', '#06d6a0')
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = cum_chart(df,
        f"{ticker} — Dividend & Split Adjusted",
        "Accounts for dividends and stock splits",
        'cumul_adj_overnight', 'cumul_adj_intraday', '#7b88ff', '#52e8c9')
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Annual bars ──────────────────────────────────────────────────
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=yearly['Year'], y=yearly['Overnight'],
        name='Overnight', marker_color='#4361ee',
        hovertemplate='%{x}: %{y:.2%}<extra>Overnight</extra>'
    ))
    bar_colors = ['#06d6a0' if v >=
                  0 else '#ef233c' for v in yearly['Intraday']]
    fig3.add_trace(go.Bar(
        x=yearly['Year'], y=yearly['Intraday'],
        name='Intraday', marker_color=bar_colors,
        hovertemplate='%{x}: %{y:.2%}<extra>Intraday</extra>'
    ))
    fig3.update_layout(
        title=dict(text=f"<b>Annual Overnight vs Intraday Returns</b>",
                   x=0.02, font=dict(size=14)),
        barmode='group',
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        font=dict(color="white", family="Inter"),
        yaxis=dict(title="Annual Return", tickformat=".0%", gridcolor=GRID),
        xaxis=dict(gridcolor=GRID, dtick=2),
        hovermode="x unified",
        height=320,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 4: Distributions ────────────────────────────────────────────────
    d1, d2 = st.columns(2)
    for col_widget, col_data, color, title_str in [
        (d1, 'overnight_r', '#4361ee', 'Daily Overnight Return Distribution'),
        (d2, 'intraday_r',  '#06d6a0', 'Daily Intraday Return Distribution'),
    ]:
        with col_widget:
            fig_d = px.histogram(df, x=col_data, nbins=120,
                                 color_discrete_sequence=[color],
                                 title=f"<b>{title_str}</b>")
            mean_v = df[col_data].mean()
            fig_d.add_vline(x=mean_v, line_dash="dash", line_color="#ffd166",
                            annotation_text=f"μ={mean_v:.3%}",
                            annotation_font_color="#ffd166",
                            annotation_position="top right")
            fig_d.update_layout(
                paper_bgcolor=DARK, plot_bgcolor=DARK,
                font=dict(color="white", family="Inter"),
                xaxis=dict(title="Daily Return",
                           tickformat=".1%", gridcolor=GRID),
                yaxis=dict(title="Count", gridcolor=GRID),
                height=310, margin=dict(l=40, r=20, t=50, b=40),
                showlegend=False
            )
            st.plotly_chart(fig_d, use_container_width=True)

    # ── Summary Table ─────────────────────────────────────────────────────────
    def cv(v, pct=True, rev=False):
        """Colour-coded value HTML."""
        s = f"{v:+.2%}" if pct else f"{v:+.2f}"
        pos = "#ef233c" if rev else "#06d6a0"
        neg = "#06d6a0" if rev else "#ef233c"
        c = pos if v >= 0 else neg
        return f'<span style="color:{c}">{s}</span>'

    def na(): return '<span style="color:#444">—</span>'

    rows = [
        ("Cumulative Overnight Return",  cv(m['cum_o']),    cv(
            m['cum_ao']),   cv(m['cum_ao']-m['cum_o'])),
        ("Cumulative Intraday Return",   cv(m['cum_i']),    cv(
            m['cum_ai']),   cv(m['cum_ai']-m['cum_i'])),
        ("Annualized Overnight Return",  cv(m['ann_o']),    cv(
            m['ann_ao']),   cv(m['ann_ao']-m['ann_o'])),
        ("Annualized Intraday Return",   cv(m['ann_i']),    cv(
            m['ann_ai']),   cv(m['ann_ai']-m['ann_i'])),
        ("Overnight Sharpe Ratio",       cv(m['sh_o'],  False), cv(
            m['sh_ao'], False), cv(m['sh_ao']-m['sh_o'], False)),
        ("Intraday Sharpe Ratio",        cv(m['sh_i'],  False), cv(
            m['sh_ai'], False), cv(m['sh_ai']-m['sh_i'], False)),
        ("Overnight Win Rate",           cv(m['win_o']),    na(), na()),
        ("Intraday Win Rate",            cv(m['win_i']),    na(), na()),
        ("Max Overnight Drawdown",       cv(m['dd_o'], rev=True), na(), na()),
        ("Max Intraday Drawdown",        cv(m['dd_i'], rev=True), na(), na()),
        ("Divergence Ratio",
            f'<span style="color:{"#ef233c" if m["div"]>0.85 else "#ffd166" if m["div"]>0.7 else "#06d6a0"}">{m["div"]:.4f}</span>',
            na(), na()),
        ("Data Points (Trading Days)",   f'{m["n"]:,}', f'{m["n"]:,}', na()),
    ]

    tbody = "".join(
        f"<tr>{''.join(f'<td>{c}</td>' for c in row)}</tr>" for row in rows)

    st.markdown(f"""
    <div class="custom-table-wrapper">
      <table class="custom-table">
        <thead>
          <tr><th>Metric</th><th>Knuteson Raw</th><th>Adjusted</th><th>Difference</th></tr>
        </thead>
        <tbody>{tbody}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

    # ── Raw Data Expander ─────────────────────────────────────────────────────
    with st.expander("📊 View Raw Computed Data", expanded=False):
        show = df[['Open', 'Close', 'Adj Close',
                   'overnight_r', 'intraday_r',
                   'cumul_overnight', 'cumul_intraday']].copy()
        show.columns = ['Open', 'Close', 'Adj Close',
                        'Overnight %', 'Intraday %',
                        'Cumul. Overnight', 'Cumul. Intraday']
        fmt = {
            'Open': '{:.2f}', 'Close': '{:.2f}', 'Adj Close': '{:.2f}',
            'Overnight %': '{:.4%}', 'Intraday %': '{:.4%}',
            'Cumul. Overnight': '{:.4%}', 'Cumul. Intraday': '{:.4%}',
        }
        def hi_o(v): return "color: #4361ee" if v >= 0 else "color: #7b88ff"
        def hi_i(v): return "color: #06d6a0" if v >= 0 else "color: #ef233c"

        try:
            # pandas >= 2.1
            styled = (show.style
                          .format(fmt)
                          .map(hi_o, subset=['Overnight %'])
                          .map(hi_i, subset=['Intraday %']))
        except AttributeError:
            # pandas < 2.1
            styled = (show.style
                          .format(fmt)
                          .applymap(hi_o, subset=['Overnight %'])
                          .applymap(hi_i, subset=['Intraday %']))
        st.dataframe(styled, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#555566;margin-top:50px;font-size:.82rem;padding-bottom:20px;">
    Data from Yahoo Finance via yfinance &nbsp;|&nbsp;
    Methodology: Knuteson (2023) — <em>Nothing to See Here</em> &nbsp;|&nbsp;
    Not financial advice
</div>
""", unsafe_allow_html=True)
