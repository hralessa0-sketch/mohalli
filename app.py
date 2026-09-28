"""
محللي - أداة تعليمية لاختيار الأسهم الأمريكية مع فلتر شرعي مبدئي.
ليست توصية مالية. القرار النهائي دائماً لك.
"""

import datetime as dt
import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="محللي", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stTabs, .stAlert, .stExpander, .stButton, .stMetric,
[data-testid="stSidebar"] { font-family: 'IBM Plex Sans Arabic', Tahoma, sans-serif; direction: rtl; text-align: right; }
[data-testid="stDataFrame"], .stCode { direction: ltr; }
h1 { color: #1F5E4F; }
.rule { border-right: 4px solid #1F5E4F; padding: 8px 14px; margin: 8px 0; background: rgba(31,94,79,0.06); }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- الإعدادات
PAGES = {
    "كبيرة": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "متوسطة": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "صغيرة": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
BANNED_SECTORS = ["Financials"]
BANNED_INDUSTRIES = ["Casinos & Gaming", "Brewers", "Distillers & Vintners", "Tobacco"]
JOURNAL_FILE = "journal.csv"
JOURNAL_COLS = ["Date", "Symbol", "Window", "Entry", "Stop", "Target", "Shares", "Status"]

FATHER_RULES = [
    "لا تدخل صفقة بدون ما تعرف وين بتطلع لو خسرت.",
    "لا تخاطر في صفقة وحدة بأكثر من ١-٢٪ من فلوسك.",
    "وقف الخسارة وعد بينك وبين نفسك، لا تخلفه.",
    "عدم الدخول قرار. الأيام اللي ما فيها فرص، اصبر.",
    "لا تطارد سهم طار، السوق ما يخلص.",
    "لا تستلف عشان تتداول، ولا تحط فلوس تحتاجها.",
    "الغنى يجي من الصبر والإضافة الشهرية، مو من صفقة وحدة.",
    "سجل كل صفقة، الربحانة والخسرانة. السجل هو معلمك الحقيقي.",
]


# ---------------------------------------------------------------- جلب البيانات
@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def load_universe():
    tables = []
    for size, url in PAGES.items():
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        t = pd.read_html(io.StringIO(html))[0]
        t = t[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
        t["Size"] = size
        tables.append(t)
    u = pd.concat(tables, ignore_index=True)
    u["Symbol"] = u["Symbol"].str.replace(".", "-", regex=False)
    return u.drop_duplicates("Symbol").rename(columns={"Security": "Name", "GICS Sector": "Sector"})


def fetch_one(sym):
    try:
        info = yf.Ticker(sym).info
    except Exception:
        return None
    return {
        "Symbol": sym,
        "Cap": info.get("marketCap"),
        "Debt": info.get("totalDebt"),
        "Cash": info.get("totalCash"),
        "Price": info.get("currentPrice"),
        "RevGrowth": info.get("revenueGrowth"),
        "EarnGrowth": info.get("earningsGrowth"),
        "Margin": info.get("profitMargins"),
        "ROE": info.get("returnOnEquity"),
        "PE": info.get("forwardPE"),
        "MA200": info.get("twoHundredDayAverage"),
        "Target": info.get("targetMeanPrice"),
    }


def sharia_reason(r):
    if r["Sector"] in BANNED_SECTORS or r["GICS Sub-Industry"] in BANNED_INDUSTRIES:
        return "النشاط غير مباح"
    if r["Cap"] <= 0:
        return "بيانات ناقصة"
    if r["Debt"] / r["Cap"] >= 0.30:
        return "ديون مرتفعة"
    if r["Cash"] / r["Cap"] >= 0.30:
        return "نقد مرتفع"
    return ""


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_fundamentals(symbols):
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows = [r for r in ex.map(fetch_one, symbols) if r]
    return pd.DataFrame(rows)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_prices(symbols):
    data = yf.download(list(symbols), period="1y", auto_adjust=True, progress=False, threads=True)
    return data["Close"], data["High"], data["Low"], data["Volume"]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def earnings_date(sym):
    try:
        return yf.Ticker(sym).calendar.get("Earnings Date", [None])[0]
    except Exception:
        return None


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def get_news(sym):
    try:
        key = st.secrets["FINNHUB_KEY"]
    except Exception:
        return []
    today = dt.date.today()
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": sym, "from": str(today - dt.timedelta(days=7)), "to": str(today), "token": key},
            timeout=15,
        )
        return r.json()[:5]
    except Exception:
        return []


# ---------------------------------------------------------------- الحسابات
def compute_atr(close, high, low, n=14):
    prev = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    return tr.rolling(n).mean().iloc[-1]


def compute_rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / down)


def position_size(capital, risk_pct, price, stop):
    risk_per_share = price - stop
    if risk_per_share <= 0 or price <= 0:
        return 0.0, 0.0, 0.0
    shares = (capital * risk_pct / 100) / risk_per_share
    shares = min(shares, capital * 0.20 / price)  # ما نحط أكثر من ٢٠٪ من الفلوس في سهم واحد
    shares = round(shares, 2)
    return shares, round(shares * price, 2), round(shares * risk_per_share, 2)


def days_until(d):
    if d is None or pd.isna(d):
        return None
    return (d - dt.date.today()).days


def decide(rr, days, max_rr=2):
    if rr < max_rr:
        return "❌ مستبعد", "العائد ما يستاهل المخاطرة."
    if days is not None and 0 <= days < 14:
        return "⏳ انتظر", f"باقي {days} يوم على إعلان الأرباح. انتظر النتائج وبعدين قرر."
    return "✅ مرشح", "الشروط مكتملة."


def investment_picks(meta, atr, only_halal):
    d = meta.dropna(subset=["Price", "RevGrowth", "EarnGrowth", "Margin", "ROE", "PE", "MA200", "Target"]).copy()
    d = d[(d.PE > 0) & (d.Price > 0)]
    d["Upside"] = d.Target / d.Price - 1
    d["Trend"] = d.Price / d.MA200 - 1
    good_high = ["RevGrowth", "EarnGrowth", "Margin", "ROE", "Upside", "Trend"]
    score = sum(d[c].rank(pct=True) for c in good_high) + d["PE"].rank(pct=True, ascending=False)
    d["Score"] = (score / 7 * 100).round()
    d = d[(d.Trend > 0) & (d.Upside > 0.05)]
    if only_halal:
        d = d[d.Sharia]
    d["Company"] = d.Name.str.replace(r"\s*\(.*\)", "", regex=True)
    top = d.sort_values("Score", ascending=False).drop_duplicates("Company").groupby("Sector").head(2).head(10).copy()
    top["ATR"] = top.Symbol.map(atr)
    top = top.dropna(subset=["ATR"])
    top["Stop"] = (top.Price - 2.5 * top.ATR).round(2)
    top["TargetPrice"] = top.Target.round(2)
    top["RR"] = ((top.TargetPrice - top.Price) / (top.Price - top.Stop)).round(1)
    return top


def investment_reason(r):
    parts = []
    if r.RevGrowth > 0.10:
        parts.append(f"إيراداتها تنمو {r.RevGrowth * 100:.0f}٪")
    if r.EarnGrowth > 0.10:
        parts.append(f"أرباحها تنمو {r.EarnGrowth * 100:.0f}٪")
    if r.Margin > 0.15:
        parts.append(f"هامش ربحها {r.Margin * 100:.0f}٪")
    parts.append(f"سعرها فوق متوسط ٢٠٠ يوم بـ {r.Trend * 100:.0f}٪ (اتجاه صاعد)")
    parts.append(f"المحللون يتوقعون صعود {r.Upside * 100:.0f}٪")
    return "، ".join(parts) + "."


def swing_picks(meta, close, high, low, vol, only_halal):
    last = close.iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    high20 = close.rolling(20).max().iloc[-2]
    vol_ratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-2]
    rsi = compute_rsi(close).iloc[-1]
    atr = compute_atr(close, high, low)
    d = pd.DataFrame({"Price": last, "MA20": ma20, "MA50": ma50, "High20": high20,
                      "VolRatio": vol_ratio, "RSI": rsi, "ATR": atr}).dropna()
    d = d.join(meta.set_index("Symbol")[["Name", "Sector", "Size", "Sharia", "Reason"]], how="inner")
    cond = ((d.Price > d.MA20) & (d.MA20 > d.MA50) & (d.Price >= d.High20)
            & (d.VolRatio > 1.5) & d.RSI.between(50, 72) & (d.Price > 5))
    d = d[cond]
    if only_halal:
        d = d[d.Sharia]
    d["Stop"] = (d.Price - 1.5 * d.ATR).round(2)
    d["TargetPrice"] = (d.Price + 3 * d.ATR).round(2)
    d["RR"] = 2.0
    d = d.sort_values("VolRatio", ascending=False).head(10)
    return d.reset_index().rename(columns={"index": "Symbol"})


# ---------------------------------------------------------------- السجل
def load_journal():
    if "journal" not in st.session_state:
        if os.path.exists(JOURNAL_FILE):
            st.session_state.journal = pd.read_csv(JOURNAL_FILE)
        else:
            st.session_state.journal = pd.DataFrame(columns=JOURNAL_COLS)
    return st.session_state.journal


def save_journal(df):
    st.session_state.journal = df
    df.to_csv(JOURNAL_FILE, index=False)


def add_to_journal(sym, window, entry, stop, target, shares):
    j = load_journal()
    row = pd.DataFrame([{"Date": str(dt.date.today()), "Symbol": sym, "Window": window, "Entry": entry,
                         "Stop": stop, "Target": target, "Shares": shares, "Status": "مفتوحة"}])
    save_journal(pd.concat([j, row], ignore_index=True))


# ---------------------------------------------------------------- عرض سهم
def show_stock(r, window, capital, risk_pct, key_prefix):
    days = days_until(earnings_date(r.Symbol)) if window == "استثمار" else None
    label, note = decide(r.RR, days)
    title = f"{label}  {r.Symbol} · {r.Name}"
    if window == "استثمار":
        title += f"  (درجة {int(r.Score)})"
    with st.expander(title):
        if window == "استثمار":
            st.write("**ليش اخترته:** " + investment_reason(r))
        else:
            st.write(f"**ليش اخترته:** كسر أعلى سعر له في آخر ٢٠ يوم، وحجم التداول اليوم "
                     f"{r.VolRatio:.1f} ضعف المعتاد، واتجاهه القصير صاعد.")
        st.write(f"**القرار:** {label}، {note}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("السعر الحالي", f"${r.Price:,.2f}")
        c2.metric("الهدف", f"${r.TargetPrice:,.2f}")
        c3.metric("وقف الخسارة", f"${r.Stop:,.2f}")
        c4.metric("العائد مقابل المخاطرة", f"{r.RR}")

        shares, cost, loss = position_size(capital, risk_pct, r.Price, r.Stop)
        if shares > 0:
            st.markdown(
                f"<div class='rule'>لو قررت تدخل: اشتر تقريباً <b>{shares}</b> سهم بتكلفة <b>${cost:,.2f}</b>. "
                f"ولو ضرب الوقف، أقصى خسارتك <b>${loss:,.2f}</b> بس.</div>",
                unsafe_allow_html=True,
            )
        if window == "مضاربة":
            st.caption("لو ما وصل الهدف ولا الوقف خلال ٧ أيام، اطلع. المضاربة وقتها محدود.")
        elif days is not None:
            st.caption(f"إعلان الأرباح القادم بعد {days} يوم.")

        news = get_news(r.Symbol)
        if news:
            st.write("**آخر الأخبار:**")
            for n in news:
                st.markdown(f"- [{n.get('headline', '')}]({n.get('url', '')}) ({n.get('source', '')})")

        st.caption("قبل الشراء: تأكد من التوافق الشرعي في Musaffa أو Zoya، وطهّر نسبة الدخل غير المباح من أرباحك.")
        if st.button("سجّلها في سجلي", key=f"{key_prefix}_{r.Symbol}"):
            add_to_journal(r.Symbol, window, r.Price, r.Stop, r.TargetPrice, shares)
            st.success("انضافت لسجلك.")


# ---------------------------------------------------------------- الواجهة
st.title("📈 محللي")
st.markdown(f"<div class='rule'>💡 {FATHER_RULES[dt.date.today().toordinal() % len(FATHER_RULES)]}</div>",
            unsafe_allow_html=True)

with st.sidebar:
    st.header("إعداداتك")
    capital = st.number_input("كم فلوسك المخصصة للتداول؟ ($)", min_value=100.0, value=1000.0, step=100.0)
    risk_pct = st.slider("أقصى خسارة في الصفقة الوحدة (٪ من فلوسك)", 0.5, 2.0, 1.0, 0.25)
    only_halal = st.toggle("الأسهم المتوافقة شرعياً فقط", value=True)
    if st.button("🔄 تحديث البيانات"):
        st.cache_data.clear()
        st.rerun()
    st.caption("أداة تعليمية، مو توصية مالية. القرار النهائي لك.")

with st.spinner("أجهز البيانات... أول مرة تاخذ عدة دقائق لأني أفحص ١٥٠٠ شركة، بعدها تصير سريعة."):
    universe = load_universe()
    fund = load_fundamentals(tuple(universe.Symbol))
    meta = universe.merge(fund, on="Symbol")
    meta[["Cap", "Debt", "Cash"]] = meta[["Cap", "Debt", "Cash"]].fillna(0)
    meta["Reason"] = meta.apply(sharia_reason, axis=1)
    meta["Sharia"] = meta["Reason"] == ""
    close, high, low, vol = load_prices(tuple(meta.Symbol))
    atr_all = compute_atr(close, high, low)

tab_inv, tab_swing, tab_log, tab_learn = st.tabs(["🏦 استثمار", "⚡ مضاربة", "📒 سجلي", "🎓 تعلّم"])

with tab_inv:
    st.subheader("أسهم للاستثمار (٣ شهور وفوق)")
    st.caption("شركات أرقامها قوية واتجاهها صاعد، بحد أقصى سهمين من كل قطاع عشان تتنوع.")
    picks = investment_picks(meta, atr_all, only_halal)
    if picks.empty:
        st.info("ما لقيت أسهم تنطبق عليها الشروط اليوم. وعدم الدخول قرار صحيح.")
    for _, r in picks.iterrows():
        show_stock(r, "استثمار", capital, risk_pct, "inv")

with tab_swing:
    st.subheader("فرص مضاربة (أيام إلى أسبوع)")
    st.caption("أسهم كسرت قمتها الأخيرة بحجم تداول قوي. أخطر من الاستثمار، فخل مبلغها أصغر.")
    swings = swing_picks(meta, close, high, low, vol, only_halal)
    if swings.empty:
        st.info("ما فيه فرص مضاربة واضحة اليوم. المضارب الشاطر يعرف متى ما يدخل.")
    for _, r in swings.iterrows():
        show_stock(r, "مضاربة", capital, risk_pct, "swing")

with tab_log:
    st.subheader("سجل صفقاتي")
    j = load_journal()
    if j.empty:
        st.info("سجلك فاضي. لما تدخل صفقة (حقيقية أو وهمية) اضغط \"سجّلها في سجلي\" تحت السهم.")
    else:
        j = j.copy()
        for c in ["Entry", "Stop", "Target", "Shares"]:
            j[c] = pd.to_numeric(j[c], errors="coerce")
        j["Now"] = j.Symbol.map(close.iloc[-1] if not close.empty else {}).round(2)
        j["PnL$"] = ((j.Now - j.Entry) * j.Shares).round(2)
        j["PnL%"] = ((j.Now / j.Entry - 1) * 100).round(1)

        def advice(r):
            if r.Status != "مفتوحة" or pd.isna(r.Now):
                return ""
            if r.Now <= r.Stop:
                return "🔴 ضرب الوقف، اطلع"
            if r.Now >= r.Target:
                return "🟢 وصل الهدف، فكر تبيع"
            held = (dt.date.today() - dt.date.fromisoformat(str(r.Date))).days
            if r.Window == "مضاربة" and held > 7:
                return "⏰ انتهت مدة المضاربة، اطلع"
            return "⚪ خلك ماسك"

        j["Advice"] = j.apply(advice, axis=1)
        st.dataframe(j, use_container_width=True)

        open_idx = j.index[j.Status == "مفتوحة"].tolist()
        if open_idx:
            pick = st.selectbox("قفّل صفقة:", open_idx, format_func=lambda i: f"{j.Symbol[i]} ({j.Date[i]})")
            if st.button("قفّلها"):
                base = load_journal().copy()
                base.loc[pick, "Status"] = f"مقفلة {dt.date.today()}"
                save_journal(base)
                st.rerun()

        closed = j[j.Status != "مفتوحة"]
        if not closed.empty:
            wins = (closed["PnL$"] > 0).mean() * 100
            st.metric("نسبة الصفقات الرابحة", f"{wins:.0f}٪")

    st.download_button("⬇️ نزّل نسخة من سجلك", load_journal().to_csv(index=False).encode("utf-8-sig"),
                       "journal.csv", "text/csv")
    up = st.file_uploader("⬆️ استرجع سجلك من نسخة سابقة", type="csv")
    if up is not None and st.button("استرجع"):
        save_journal(pd.read_csv(up))
        st.rerun()
    st.caption("نزّل نسخة من سجلك كل أسبوع، لأن الموقع ممكن يمسح الملفات لما يعيد تشغيل نفسه.")

with tab_learn:
    st.subheader("وصايا قبل كل صفقة")
    for rule in FATHER_RULES:
        st.markdown(f"<div class='rule'>{rule}</div>", unsafe_allow_html=True)

    st.subheader("كيف يصير الواحد غني فعلاً؟")
    st.write("مو بصفقة وحدة تضاعف فلوسك، بل بمبلغ تضيفه كل شهر، وعائد معقول، وسنين من الصبر. جرّب بنفسك:")
    c1, c2, c3 = st.columns(3)
    start = c1.number_input("تبدأ بـ ($)", 0, 1_000_000, 1000, 500)
    monthly = c2.number_input("تضيف كل شهر ($)", 0, 100_000, 300, 50)
    yearly = c3.slider("العائد السنوي المتوقع (٪)", 4, 15, 9)
    years = st.slider("كم سنة؟", 1, 30, 15)
    balance, paid, rows = float(start), float(start), []
    for y in range(1, years + 1):
        for _ in range(12):
            balance = balance * (1 + yearly / 100 / 12) + monthly
            paid += monthly
        rows.append({"السنة": y, "اللي دفعته": round(paid), "اللي صار عندك": round(balance)})
    growth = pd.DataFrame(rows).set_index("السنة")
    st.line_chart(growth)
    st.markdown(
        f"<div class='rule'>بعد {years} سنة: دفعت <b>${paid:,.0f}</b>، وصار عندك <b>${balance:,.0f}</b>. "
        f"الفرق <b>${balance - paid:,.0f}</b> هو شغل الوقت والصبر.</div>",
        unsafe_allow_html=True,
    )
    st.caption("السوق الأمريكي تاريخياً أعطى متوسط ٧-١٠٪ سنوياً على المدى الطويل، مع سنين فيها خسارة. الماضي ما يضمن المستقبل.")
