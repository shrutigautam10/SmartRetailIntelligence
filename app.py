import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

st.set_page_config(
    page_title="Smart Retail Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# GLOBAL THEME & CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg-base:    #07090f;
    --bg-card:    #0d1117;
    --bg-raised:  #141924;
    --border:     #1e2a3a;
    --accent-1:   #00e5ff;   /* electric cyan   */
    --accent-2:   #ff3cac;   /* hot magenta     */
    --accent-3:   #ffcc00;   /* neon yellow     */
    --accent-4:   #7b2fff;   /* deep violet     */
    --accent-5:   #00ff9f;   /* mint green      */
    --accent-6:   #ff6b35;   /* fire orange     */
    --text-hi:    #eef2ff;
    --text-lo:    #6b7a99;
    --radius:     14px;
    --shadow:     0 8px 32px rgba(0,0,0,.55);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-base) !important;
    color: var(--text-hi) !important;
}
.stApp { background: var(--bg-base) !important; }

/* ── Block container ── */
.block-container {
    padding: 1.8rem 2.2rem 3rem !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-hi) !important; }

/* ── Radio buttons ── */
div[role="radiogroup"] label {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 14px;
    margin: 4px 0;
    cursor: pointer;
    transition: all .2s;
    display: block;
    font-family: 'DM Sans', sans-serif;
}
div[role="radiogroup"] label:hover {
    border-color: var(--accent-1);
    background: #101e2f;
}

/* ── Headings ── */
h1, h2, h3, h4, h5 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text-hi) !important;
    letter-spacing: -0.02em;
}

/* ── Page title banner ── */
.page-banner {
    background: linear-gradient(120deg, #0d1117 0%, #0f1e2e 50%, #0d1117 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent-1);
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.8rem;
}
.page-banner h2 {
    margin: 0 0 4px;
    font-size: 1.5rem;
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.page-banner p { color: var(--text-lo) !important; margin: 0; font-size: .9rem; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 18px 22px !important;
    box-shadow: var(--shadow) !important;
    transition: transform .2s;
}
div[data-testid="stMetric"]:hover { transform: translateY(-3px); }
div[data-testid="stMetricLabel"] { color: var(--text-lo) !important; font-size: .8rem !important; }
div[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 1.8rem !important; color: var(--accent-1) !important; }

/* ── Chart wrapper ── */
.chart-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    background: var(--bg-raised);
    border-radius: 10px 10px 0 0;
    padding: 9px 20px;
    color: var(--text-lo) !important;
    font-family: 'DM Sans', sans-serif;
    border: 1px solid var(--border);
    border-bottom: none;
    transition: all .2s;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-hi) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0a1f3c, #0d2b4e) !important;
    color: var(--accent-1) !important;
    border-color: var(--accent-1) !important;
    font-weight: 600;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; }

/* ── Selectbox / inputs ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > input {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-hi) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-4)) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 26px !important;
    font-size: .95rem !important;
    transition: all .25s !important;
    box-shadow: 0 0 18px rgba(0,229,255,.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 28px rgba(0,229,255,.5) !important;
}

/* ── Success / warning ── */
div[data-testid="stSuccess"] {
    background: rgba(0,255,159,.12) !important;
    border: 1px solid var(--accent-5) !important;
    border-radius: var(--radius) !important;
    color: var(--accent-5) !important;
}
div[data-testid="stWarning"] {
    background: rgba(255,204,0,.1) !important;
    border: 1px solid var(--accent-3) !important;
    border-radius: var(--radius) !important;
}

/* ── KPI badge strip ── */
.kpi-strip {
    display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.4rem;
}
.kpi-badge {
    flex: 1 1 160px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
}
.kpi-badge::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
}
.kpi-c1::before { background: var(--accent-1); }
.kpi-c2::before { background: var(--accent-2); }
.kpi-c3::before { background: var(--accent-3); }
.kpi-c4::before { background: var(--accent-5); }
.kpi-label { font-size: .72rem; color: var(--text-lo); text-transform: uppercase; letter-spacing: .1em; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 1.55rem; font-weight: 800; margin-top: 4px; }
.kpi-c1 .kpi-value { color: var(--accent-1); }
.kpi-c2 .kpi-value { color: var(--accent-2); }
.kpi-c3 .kpi-value { color: var(--accent-3); }
.kpi-c4 .kpi-value { color: var(--accent-5); }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MATPLOTLIB GLOBAL STYLE
# ─────────────────────────────────────────
HIST_COLORS  = ["#00e5ff","#ff3cac","#ffcc00","#7b2fff","#00ff9f","#ff6b35","#06b6d4","#f472b6"]
PIE_COLORS_A = ["#00e5ff","#ff3cac","#ffcc00","#7b2fff","#00ff9f","#ff6b35"]
PIE_COLORS_B = ["#f97316","#06b6d4","#84cc16","#e11d48","#a78bfa","#fbbf24"]
BG_DARK  = "#0d1117"
BG_AXES  = "#141924"
TEXT_CLR = "#eef2ff"
GRID_CLR = "#1e2a3a"

def apply_dark_style(fig, ax):
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_AXES)
    ax.title.set_color(TEXT_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.tick_params(colors=TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(linestyle="--", alpha=0.25, color=GRID_CLR)

def styled_histogram(data, title, xlabel, color, bins=20):
    fig, ax = plt.subplots(figsize=(7, 4))
    apply_dark_style(fig, ax)
    n, bins_e, patches = ax.hist(data.dropna(), bins=bins, edgecolor="none", color=color, alpha=.85)
    # gradient overlay via individual patch colours
    cmap = plt.cm.cool
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.plasma(i / len(patches)))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    plt.tight_layout()
    return fig

def styled_pie(values, labels, title, colors):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(BG_DARK)
    wedges, texts, autotexts = ax.pie(
        values, labels=None, autopct="%1.1f%%", startangle=130,
        colors=colors,
        wedgeprops=dict(linewidth=2.5, edgecolor=BG_DARK),
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_color(BG_DARK); at.set_fontsize(9); at.set_fontweight("bold")
    # donut hole
    centre = plt.Circle((0, 0), 0.52, fc=BG_DARK)
    ax.add_patch(centre)
    ax.set_title(title, color=TEXT_CLR, fontsize=13, fontweight="bold", pad=14)
    # custom legend
    legend_patches = [mpatches.Patch(color=colors[i % len(colors)], label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, framealpha=0, labelcolor=TEXT_CLR, fontsize=9)
    plt.tight_layout()
    return fig

def styled_bar(x, y, title, xlabel, ylabel, color):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    apply_dark_style(fig, ax)
    bars = ax.bar(x, y, color=color, width=0.55, edgecolor="none", zorder=3)
    # gradient on bars
    cmap = plt.cm.plasma
    for i, bar in enumerate(bars):
        bar.set_color(cmap(0.2 + 0.6 * i / max(len(bars)-1, 1)))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load("sales_model.pkl")
    except:
        return None

@st.cache_data(show_spinner=False)
def load_data():
    try:
        return pd.read_csv("SuperMarket Analysis.csv")
    except:
        return None

model        = load_model()
df_dashboard = load_data()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:18px 0 10px;">
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                    background:linear-gradient(90deg,#00e5ff,#ff3cac);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            ⚡ Smart Retail
        </div>
        <div style="font-size:.75rem;color:#6b7a99;margin-top:2px;">Intelligence System v2</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠  Overview", "📊  Dashboard", "📦  Inventory",
         "🏷️  Discount Recommender", "🤖  Prediction", "🌐  Open Environment"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-size:.78rem;color:#6b7a99;line-height:1.6;">
        AI-Powered Retail Analytics<br>
        Built with Streamlit · ML · Matplotlib
    </div>
    """, unsafe_allow_html=True)

page_key = page.split("  ", 1)[-1] if "  " in page else page


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(120deg,#0d1117,#0a1a2e,#0d1117);
            border:1px solid #1e2a3a; border-left:4px solid #00e5ff;
            border-radius:14px; padding:1.5rem 2rem; margin-bottom:2rem;">
    <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
                background:linear-gradient(90deg,#00e5ff 0%,#ff3cac 50%,#7b2fff 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Smart Retail Intelligence System
    </div>
    <div style="color:#6b7a99;font-size:.9rem;margin-top:6px;">
        AI-powered analytics · Sales prediction · Business intelligence dashboard
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HELPER — KPI strip
# ─────────────────────────────────────────
def show_kpis(df):
    total_sales    = df["Sales"].sum()        if "Sales"       in df.columns else 0
    total_profit   = df["gross income"].sum() if "gross income" in df.columns else 0
    total_quantity = df["Quantity"].sum()     if "Quantity"    in df.columns else 0
    avg_rating     = df["Rating"].mean()      if "Rating"      in df.columns else 0

    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi-badge kpi-c1">
            <div class="kpi-label">Total Sales</div>
            <div class="kpi-value">${total_sales:,.0f}</div>
        </div>
        <div class="kpi-badge kpi-c2">
            <div class="kpi-label">Total Profit</div>
            <div class="kpi-value">${total_profit:,.0f}</div>
        </div>
        <div class="kpi-badge kpi-c3">
            <div class="kpi-label">Units Sold</div>
            <div class="kpi-value">{int(total_quantity):,}</div>
        </div>
        <div class="kpi-badge kpi-c4">
            <div class="kpi-label">Avg Rating</div>
            <div class="kpi-value">{avg_rating:.2f} ★</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════
if page_key == "Overview":
    st.markdown("""
    <div class="page-banner">
        <h2>Project Overview</h2>
        <p>Your AI retail cockpit — everything in one place.</p>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(3)
    features = [
        ("📊", "Business Dashboard", "KPI monitoring, branch & city analysis, profit trends"),
        ("📦", "Inventory Insights", "Stock demand analysis and low-demand identification"),
        ("🏷️", "Discount Engine", "ML-driven discount recommendations per product line"),
        ("🗺️", "Region Analysis", "Branch-wise and region-wise sales breakdown"),
        ("🤖", "Sales Prediction", "ML model predictions by region, category & ship mode"),
        ("🌐", "Open Environment", "Upload any retail CSV for instant analytics"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="chart-wrap" style="min-height:120px;">
                <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;
                            color:#eef2ff;margin-bottom:6px;">{title}</div>
                <div style="font-size:.82rem;color:#6b7a99;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    if df_dashboard is not None:
        st.markdown("---")
        show_kpis(df_dashboard)

        st.markdown("### Distribution Overview")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.pyplot(styled_histogram(df_dashboard["Sales"], "Sales Distribution", "Sales ($)", "#00e5ff"))
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.pyplot(styled_histogram(df_dashboard["gross income"], "Profit Distribution", "Profit ($)", "#ff3cac"))
            st.markdown('</div>', unsafe_allow_html=True)

        if "Rating" in df_dashboard.columns:
            c3, c4 = st.columns(2)
            with c3:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.pyplot(styled_histogram(df_dashboard["Rating"], "Customer Rating Distribution", "Rating", "#ffcc00"))
                st.markdown('</div>', unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.pyplot(styled_histogram(df_dashboard["Quantity"], "Quantity Distribution", "Quantity", "#7b2fff"))
                st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════
elif page_key == "Dashboard":
    st.markdown("""
    <div class="page-banner">
        <h2>Business Dashboard</h2>
        <p>Visual analytics across product lines, branches, and cities.</p>
    </div>""", unsafe_allow_html=True)

    if df_dashboard is not None:
        show_kpis(df_dashboard)

        # Pie charts row
        p1, p2 = st.columns(2)
        with p1:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            if "Product line" in df_dashboard.columns and "Sales" in df_dashboard.columns:
                sprod = df_dashboard.groupby("Product line")["Sales"].sum()
                st.pyplot(styled_pie(sprod.values, sprod.index.tolist(),
                                     "Sales by Product Line", PIE_COLORS_A))
            st.markdown('</div>', unsafe_allow_html=True)
        with p2:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            if "Branch" in df_dashboard.columns and "Sales" in df_dashboard.columns:
                sbranch = df_dashboard.groupby("Branch")["Sales"].sum()
                st.pyplot(styled_pie(sbranch.values, sbranch.index.tolist(),
                                     "Sales by Branch", PIE_COLORS_B))
            st.markdown('</div>', unsafe_allow_html=True)

        # Second pie row
        p3, p4 = st.columns(2)
        with p3:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            if "Gender" in df_dashboard.columns and "Sales" in df_dashboard.columns:
                sgender = df_dashboard.groupby("Gender")["Sales"].sum()
                st.pyplot(styled_pie(sgender.values, sgender.index.tolist(),
                                     "Sales by Gender", ["#00e5ff","#ff3cac"]))
            st.markdown('</div>', unsafe_allow_html=True)
        with p4:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            if "Payment" in df_dashboard.columns:
                spay = df_dashboard["Payment"].value_counts()
                st.pyplot(styled_pie(spay.values, spay.index.tolist(),
                                     "Payment Method Split", PIE_COLORS_A[:3]))
            st.markdown('</div>', unsafe_allow_html=True)

        # Tabs — bar charts + histograms
        t1, t2, t3, t4, t5 = st.tabs([
            "Sales by City", "Profit by Product", "Quantity", "Branch Sales", "Histograms"
        ])

        with t1:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            if "City" in df_dashboard.columns and "Sales" in df_dashboard.columns:
                cs = df_dashboard.groupby("City")["Sales"].sum().sort_values(ascending=False)
                st.pyplot(styled_bar(cs.index, cs.values, "Total Sales by City", "City", "Sales ($)", "#00e5ff"))
            st.markdown('</div>', unsafe_allow_html=True)

        with t2:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            if "Product line" in df_dashboard.columns and "gross income" in df_dashboard.columns:
                pp = df_dashboard.groupby("Product line")["gross income"].sum().sort_values(ascending=False)
                st.pyplot(styled_bar(pp.index, pp.values, "Profit by Product Line", "Product", "Profit ($)", "#ff3cac"))
            st.markdown('</div>', unsafe_allow_html=True)

        with t3:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            if "Product line" in df_dashboard.columns and "Quantity" in df_dashboard.columns:
                qp = df_dashboard.groupby("Product line")["Quantity"].sum().sort_values(ascending=False)
                st.pyplot(styled_bar(qp.index, qp.values, "Quantity by Product Line", "Product", "Quantity", "#ffcc00"))
            st.markdown('</div>', unsafe_allow_html=True)

        with t4:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            if "Branch" in df_dashboard.columns and "Sales" in df_dashboard.columns:
                bs = df_dashboard.groupby("Branch")["Sales"].sum().sort_values(ascending=False)
                st.pyplot(styled_bar(bs.index, bs.values, "Branch-wise Sales", "Branch", "Sales ($)", "#7b2fff"))
            st.markdown('</div>', unsafe_allow_html=True)

        with t5:
            h1, h2 = st.columns(2)
            with h1:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.pyplot(styled_histogram(df_dashboard["Sales"], "Sales Histogram", "Sales ($)", "#00e5ff"))
                st.markdown('</div>', unsafe_allow_html=True)
            with h2:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.pyplot(styled_histogram(df_dashboard["gross income"], "Profit Histogram", "Profit ($)", "#ff3cac"))
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Dashboard dataset 'SuperMarket Analysis.csv' not found.")


# ═══════════════════════════════════════════
# PAGE: INVENTORY
# ═══════════════════════════════════════════
elif page_key == "Inventory":
    st.markdown("""
    <div class="page-banner">
        <h2>Inventory Stock Details</h2>
        <p>Monitor quantities, identify low-demand lines, and act before stockouts.</p>
    </div>""", unsafe_allow_html=True)

    if df_dashboard is not None and "Product line" in df_dashboard.columns and "Quantity" in df_dashboard.columns:
        inv = df_dashboard.groupby("Product line")["Quantity"].sum().reset_index()
        inv.columns = ["Product Line", "Total Quantity Sold"]

        avg_qty = inv["Total Quantity Sold"].mean()
        inv["Status"] = inv["Total Quantity Sold"].apply(
            lambda x: "⚠️ Low Demand" if x < avg_qty else "✅ Healthy"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Inventory Summary")
            st.dataframe(inv, use_container_width=True, hide_index=True)
        with c2:
            # pie of inventory
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.pyplot(styled_pie(inv["Total Quantity Sold"].values,
                                 inv["Product Line"].tolist(),
                                 "Quantity Share by Product Line", PIE_COLORS_A))
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### Low Demand Lines")
        low = inv[inv["Total Quantity Sold"] < avg_qty]
        st.dataframe(low, use_container_width=True, hide_index=True)

        st.markdown("#### Quantity Histogram")
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.pyplot(styled_histogram(df_dashboard["Quantity"], "Quantity Distribution", "Quantity", "#ffcc00"))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Inventory data not available.")


# ═══════════════════════════════════════════
# PAGE: DISCOUNT RECOMMENDER
# ═══════════════════════════════════════════
elif page_key == "Discount Recommender":
    st.markdown("""
    <div class="page-banner">
        <h2>Discount Recommender</h2>
        <p>Rule-based AI engine — identifies where discounts will drive maximum uplift.</p>
    </div>""", unsafe_allow_html=True)

    if df_dashboard is not None and {"Product line","Sales","Quantity","Rating"}.issubset(df_dashboard.columns):
        disc = df_dashboard.groupby("Product line").agg(
            {"Sales":"mean","Quantity":"mean","Rating":"mean"}
        ).reset_index()

        sm = disc["Sales"].mean()
        rm = disc["Rating"].mean()

        def recommend(row):
            if row["Sales"] < sm and row["Rating"] < rm:
                return "🔴  15% Discount"
            elif row["Sales"] < sm:
                return "🟡  10% Discount"
            else:
                return "🟢  No Discount"

        disc["Recommendation"] = disc.apply(recommend, axis=1)
        disc.columns = ["Product Line","Avg Sales","Avg Quantity","Avg Rating","Recommendation"]

        st.dataframe(disc.style.format({"Avg Sales":"{:.2f}","Avg Quantity":"{:.1f}","Avg Rating":"{:.2f}"}),
                     use_container_width=True, hide_index=True)

        h1, h2 = st.columns(2)
        with h1:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.pyplot(styled_histogram(df_dashboard["Rating"], "Customer Rating Histogram", "Rating", "#ff3cac"))
            st.markdown('</div>', unsafe_allow_html=True)
        with h2:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            disc_counts = disc["Recommendation"].value_counts()
            st.pyplot(styled_pie(disc_counts.values, disc_counts.index.tolist(),
                                 "Discount Recommendation Split", ["#ef4444","#f59e0b","#22c55e"]))
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Required columns not found.")


# ═══════════════════════════════════════════
# PAGE: PREDICTION
# ═══════════════════════════════════════════
elif page_key == "Prediction":
    st.markdown("""
    <div class="page-banner">
        <h2>Sales Prediction</h2>
        <p>ML model predicts sales from category, region, and shipping inputs.</p>
    </div>""", unsafe_allow_html=True)

    if model is None:
        st.warning("sales_model.pkl not found. Please train and save the model first.")
    else:
        tab1, tab2 = st.tabs(["Manual Prediction", "Region-wise Prediction"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                category = st.selectbox("Category", ["Furniture","Office Supplies","Technology"])
                sub_category = st.selectbox("Sub-Category",
                    ["Bookcases","Chairs","Labels","Tables","Storage"])
            with c2:
                region    = st.selectbox("Region", ["South","West","Central","East"])
                ship_mode = st.selectbox("Ship Mode",
                    ["Second Class","Standard Class","First Class","Same Day"])

            if st.button("⚡ Predict Sales"):
                input_data = pd.DataFrame({
                    "Category_" + category: [1],
                    "Sub-Category_" + sub_category: [1],
                    "Region_" + region: [1],
                    "Ship Mode_" + ship_mode: [1]
                })
                for col in model.feature_names_in_:
                    if col not in input_data.columns:
                        input_data[col] = 0
                input_data = input_data[model.feature_names_in_]
                prediction = model.predict(input_data)
                st.success(f"✅ Predicted Sales: **${prediction[0]:,.2f}**")

        with tab2:
            regions = ["South","West","Central","East"]
            pvals   = []
            for reg in regions:
                ti = pd.DataFrame({
                    "Category_Furniture":[1],
                    "Sub-Category_Chairs":[1],
                    "Region_"+reg:[1],
                    "Ship Mode_Standard Class":[1]
                })
                for col in model.feature_names_in_:
                    if col not in ti.columns:
                        ti[col] = 0
                ti = ti[model.feature_names_in_]
                pvals.append(model.predict(ti)[0])

            rpdf = pd.DataFrame({"Region":regions,"Predicted Sales":pvals})
            st.dataframe(rpdf.style.format({"Predicted Sales":"${:,.2f}"}),
                         use_container_width=True, hide_index=True)

            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.pyplot(styled_bar(rpdf["Region"], rpdf["Predicted Sales"],
                                 "Region-wise Predicted Sales", "Region", "Predicted Sales ($)", "#00e5ff"))
            st.markdown('</div>', unsafe_allow_html=True)

            # Prediction pie
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.pyplot(styled_pie(rpdf["Predicted Sales"].values, rpdf["Region"].tolist(),
                                 "Prediction Share by Region", PIE_COLORS_B))
            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE: OPEN ENVIRONMENT
# ═══════════════════════════════════════════
elif page_key == "Open Environment":
    st.markdown("""
    <div class="page-banner">
        <h2>Open Environment</h2>
        <p>Upload any retail CSV — instant analytics, histograms, and downloadable insights.</p>
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop your CSV here", type=["csv"])

    if uploaded_file is not None:
        dfu = pd.read_csv(uploaded_file)

        st.markdown("#### Data Preview")
        st.dataframe(dfu.head(10), use_container_width=True)

        numeric_cols = dfu.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            st.markdown("#### Numeric Summary")
            st.dataframe(dfu[numeric_cols].describe().style.format("{:.2f}"),
                         use_container_width=True)

        # Auto-detect common columns and draw histograms
        hist_targets = [c for c in ["Sales","gross income","Quantity","Rating","Unit price","Tax 5%"] if c in dfu.columns]
        colors_cycle = HIST_COLORS

        if hist_targets:
            st.markdown("#### Distribution Histograms")
            pairs = [hist_targets[i:i+2] for i in range(0, len(hist_targets), 2)]
            for pair in pairs:
                cols_row = st.columns(len(pair))
                for ci, col_name in enumerate(pair):
                    with cols_row[ci]:
                        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                        st.pyplot(styled_histogram(dfu[col_name],
                                                   f"{col_name} Distribution",
                                                   col_name,
                                                   colors_cycle[hist_targets.index(col_name) % len(colors_cycle)]))
                        st.markdown('</div>', unsafe_allow_html=True)

        # Pie for any category column found
        for cat_col in ["Product line","Branch","City","Gender","Payment","Category"]:
            if cat_col in dfu.columns:
                st.markdown(f"#### {cat_col} Breakdown")
                vc = dfu[cat_col].value_counts().head(8)
                st.markdown('<div class="chart-wrap" style="max-width:520px">', unsafe_allow_html=True)
                st.pyplot(styled_pie(vc.values, vc.index.tolist(),
                                     f"Sales share — {cat_col}", PIE_COLORS_A))
                st.markdown('</div>', unsafe_allow_html=True)
                break

        st.download_button(
            "⬇️  Download Processed Data",
            dfu.to_csv(index=False),
            file_name="processed_data.csv",
            mime="text/csv"
        )