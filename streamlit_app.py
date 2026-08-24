import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="MetroPT APU Predictive Maintenance",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global CSS
st.markdown("""
<style>
.stApp {
    background-color: #09090b !important;
    color: #fafafa;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
}
header[data-testid="stHeader"] { display: none !important; }
div[data-testid="stToolbar"]   { display: none !important; }
div[data-testid="stDecoration"]{ display: none !important; }

section[data-testid="stMain"] > div { padding-top: 0 !important; }
.block-container { padding-top: 2rem !important; max-width: 1600px !important; }

.v-card {
    background-color: rgba(9,9,11,0.5);
    border: 1px solid rgba(39,39,42,0.5);
    border-radius: 0.75rem;
    padding: 20px;
}

.v-subsys {
    border: 2px solid #27272a;
    border-radius: 4px;
    padding: 12px 16px;
    text-align: center;
    font-weight: 700;
    font-size: 14px;
    color: #fafafa;
    cursor: pointer;
    transition: all 0.3s;
    background: rgba(9,9,11,0.5);
}
.v-subsys.active {
    border-color: #fafafa;
    background: rgba(250,250,250,0.1);
    box-shadow: 0 0 0 2px rgba(250,250,250,0.15);
}

.v-pred-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(39,39,42,0.5);
}
.v-pred-row:last-child { border-bottom: none; }
.v-pred-label {
    font-size: 18px;
    font-weight: 700;
    color: #fafafa;
}
.v-pred-value {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 18px;
    font-weight: 700;
    color: #fafafa;
}
.v-dot-green {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background-color: #22c55e;
    box-shadow: 0 0 8px rgba(16,185,129,0.5);
    display: inline-block;
}

.v-shap-bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    font-size: 12px;
    color: #a1a1aa;
}
.v-shap-bar-label { width: 110px; text-align: right; font-weight: 500; }
.v-shap-bar-track { flex: 1; height: 20px; background: transparent; border-radius: 0 4px 4px 0; }
.v-shap-bar-fill  { height: 100%; border-radius: 0 4px 4px 0; }

.v-footer {
    text-align: center;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid rgba(39,39,42,0.5);
}
.v-footer-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #a1a1aa;
    background: rgba(39,39,42,0.3);
    padding: 6px 12px;
    border-radius: 9999px;
    border: 1px solid rgba(39,39,42,0.5);
}
</style>
""", unsafe_allow_html=True)

# Live Data Simulation
np.random.seed(int(datetime.now().second))
pressure_val = round(float(np.random.uniform(7.8, 8.4)), 1)
temp_val = round(float(np.random.uniform(59.5, 62.0)), 1)
current_val = round(float(np.random.uniform(7.6, 8.4)), 1)
tp3_val = round(float(np.random.uniform(9.4, 9.9)), 1)
h1_val = round(float(np.random.uniform(0.04, 0.06)), 3)
latency_ms = int(np.random.uniform(18, 25))
now_str = datetime.now().strftime("%I:%M:%S %p")

# Lucide Icons (Data URIs)
shield_icon = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 24 24' fill='none' stroke='%23fafafa' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/%3E%3Cline x1='12' y1='8' x2='12' y2='12'/%3E%3Cline x1='12' y1='16' x2='12.01' y2='16'/%3E%3C/svg%3E"
zap_icon = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23f59e0b' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M13 2L3 14h9l-1 8 10-12h-9l1-8z'/%3E%3C/svg%3E"
server_icon = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%233b82f6' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2' y='2' width='20' height='8' rx='2' ry='2'/%3E%3Crect x='2' y='14' width='20' height='8' rx='2' ry='2'/%3E%3Cline x1='6' y1='6' x2='6.01' y2='6'/%3E%3Cline x1='6' y1='18' x2='6.01' y2='18'/%3E%3C/svg%3E"
check_icon = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2322c55e' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 11.08V12a10 10 0 1 1-5.93-9.14'/%3E%3Cpolyline points='22 4 12 14.01 9 11.01'/%3E%3C/svg%3E"
activity_icon = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%236366f1' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='22 12 18 12 15 21 9 3 6 12 2 12'/%3E%3C/svg%3E"
cpu_icon = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23a855f7' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='4' y='4' width='16' height='16' rx='2' ry='2'/%3E%3Crect x='9' y='9' width='6' height='6'/%3E%3Cline x1='9' y1='1' x2='9' y2='4'/%3E%3Cline x1='15' y1='1' x2='15' y2='4'/%3E%3Cline x1='9' y1='20' x2='9' y2='23'/%3E%3Cline x1='15' y1='20' x2='15' y2='23'/%3E%3Cline x1='20' y1='9' x2='23' y2='9'/%3E%3Cline x1='20' y1='15' x2='23' y1='15'/%3E%3Cline x1='1' y1='9' x2='4' y2='9'/%3E%3Cline x1='1' y1='15' x2='4' y2='15'/%3E%3C/svg%3E"
check_circle_large = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 24 24' fill='none' stroke='%2322c55e' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 11.08V12a10 10 0 1 1-5.93-9.14'/%3E%3Cpolyline points='22 4 12 14.01 9 11.01'/%3E%3C/svg%3E"

# 1. HEADER (Exact Vercel Spec)
st.html(f"""
<div style="display:flex; justify-content:space-between; align-items:center; padding-bottom:24px; border-bottom:1px solid rgba(39,39,42,0.4); margin-bottom:28px;">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="padding:12px; background:#09090b; border:1px solid rgba(39,39,42,0.5); border-radius:0.75rem; display:flex; align-items:center; justify-content:center; width:56px; height:56px; box-sizing:border-box; flex-shrink:0;">
            <img src="{shield_icon}" width="32" height="32" style="display:block;" />
        </div>
        <div>
            <h1 style="margin:0; font-size:30px; font-weight:800; letter-spacing:-0.025em; color:#fafafa; line-height:1.2;">MetroPT APU Predictive Maintenance</h1>
            <p style="margin:4px 0 0 0; font-size:14px; color:#a1a1aa; font-weight:400;">Real-time monitoring and AI diagnostics for railway Air Production Units.</p>
        </div>
    </div>
    <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center;">
        <div style="display:inline-flex; align-items:center; gap:6px; padding:6px 10px; background:#09090b; border:1px solid rgba(39,39,42,0.5); border-radius:6px; font-size:12px; font-weight:500; color:#a1a1aa;">
            <img src="{zap_icon}" width="14" height="14" style="display:block;" />
            <span>Inference {latency_ms}ms</span>
        </div>
        <div style="display:inline-flex; align-items:center; gap:6px; padding:6px 10px; background:#09090b; border:1px solid rgba(39,39,42,0.5); border-radius:6px; font-size:12px; font-weight:500; color:#a1a1aa;">
            <img src="{server_icon}" width="14" height="14" style="display:block;" />
            <span>Redis</span>
            <img src="{check_icon}" width="14" height="14" style="display:block;" />
        </div>
        <div style="display:inline-flex; align-items:center; gap:6px; padding:6px 10px; background:#09090b; border:1px solid rgba(39,39,42,0.5); border-radius:6px; font-size:12px; font-weight:500; color:#a1a1aa;">
            <img src="{activity_icon}" width="14" height="14" style="display:block;" />
            <span>WebSocket</span>
            <span style="color:#22c55e; font-weight:700; margin-left:1px;">(Live)</span>
        </div>
        <div style="display:inline-flex; align-items:center; gap:6px; padding:6px 10px; background:#09090b; border:1px solid rgba(39,39,42,0.5); border-radius:6px; font-size:12px; font-weight:500; color:#a1a1aa;">
            <img src="{cpu_icon}" width="14" height="14" style="display:block;" />
            <span>Model v2.1</span>
        </div>
    </div>
</div>
""")

# 2. ALERT BANNER: "Continue Operation" (Exact Vercel Spec)
st.html(f"""
<div style="background:rgba(34,197,94,0.1); border:1px solid rgba(34,197,94,0.3); border-radius:1rem; padding:24px; display:flex; align-items:center; gap:20px; box-shadow:0 0 30px rgba(16,185,129,0.05); margin-bottom:28px;">
    <div style="padding:16px; background:rgba(34,197,94,0.2); border-radius:9999px; flex-shrink:0; display:flex; align-items:center; justify-content:center;">
        <img src="{check_circle_large}" width="32" height="32" style="display:block;" />
    </div>
    <div>
        <h2 style="margin:0; font-size:20px; font-weight:800; color:#22c55e; letter-spacing:0.05em; text-transform:uppercase; line-height:1.2;">Continue Operation</h2>
        <p style="margin:4px 0 0 0; font-size:14px; color:rgba(250,250,250,0.8); font-weight:500;">All telemetry nominal. No immediate maintenance required.</p>
    </div>
</div>
""")

# 3. SENSOR GAUGES (3 columns)
def make_gauge(value, vmin, vmax, color, unit):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            'suffix': f"<br><span style='font-size:12px;color:#a1a1aa;font-weight:700'>{unit}</span>",
            'font': {'size': 30, 'color': '#fafafa', 'family': 'Inter, -apple-system, sans-serif'},
        },
        gauge={
            'axis': {
                'range': [vmin, vmax],
                'tickwidth': 0,
                'tickcolor': 'rgba(0,0,0,0)',
                'tickvals': [],
                'showticklabels': False,
            },
            'bar': {'color': color, 'thickness': 0.35},
            'bgcolor': '#27272a',
            'borderwidth': 0,
            'bordercolor': 'rgba(0,0,0,0)',
            'steps': [],
            'threshold': {'line': {'width': 0}, 'value': value},
        },
    ))
    fig.update_layout(
        height=150,
        margin=dict(l=30, r=30, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa'),
    )
    return fig

g1, g2, g3 = st.columns(3)

with g1:
    st.html("""<div class="v-card" style="padding-bottom:0;"><div style="display:flex;align-items:center;gap:8px;font-size:14px;font-weight:600;color:#a1a1aa;text-transform:uppercase;letter-spacing:0.05em;">Pressure</div></div>""")
    st.plotly_chart(make_gauge(pressure_val, 0, 12, '#3b82f6', 'BAR'), use_container_width=True, key='g1')

with g2:
    st.html("""<div class="v-card" style="padding-bottom:0;"><div style="display:flex;align-items:center;gap:8px;font-size:14px;font-weight:600;color:#a1a1aa;text-transform:uppercase;letter-spacing:0.05em;">Temperature</div></div>""")
    st.plotly_chart(make_gauge(temp_val, 20, 100, '#ef4444', 'C'), use_container_width=True, key='g2')

with g3:
    st.html("""<div class="v-card" style="padding-bottom:0;"><div style="display:flex;align-items:center;gap:8px;font-size:14px;font-weight:600;color:#a1a1aa;text-transform:uppercase;letter-spacing:0.05em;">Motor Current</div></div>""")
    st.plotly_chart(make_gauge(current_val, 0, 15, '#f59e0b', 'AMPS'), use_container_width=True, key='g3')

# MAIN ANALYTICAL GRID
left_col, right_col = st.columns([2, 1])

with left_col:
    # Interactive APU Schematic
    st.html(f"""
    <div class="v-card" style="margin-bottom:24px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <div style="font-size:18px; font-weight:700; color:#fafafa; display:flex; align-items:center; gap:8px;">
                Interactive APU Schematic
            </div>
            <div style="font-size:12px; display:flex; align-items:center; gap:6px; color:#a1a1aa; background:rgba(39,39,42,0.5); padding:4px 8px; border-radius:4px; border:1px solid rgba(39,39,42,0.5);">
                <span style="width:8px;height:8px;border-radius:50%;background:#22c55e;display:inline-block;"></span>
                Live Data Connected
            </div>
        </div>
        <p style="font-size:14px; color:#a1a1aa; margin-bottom:24px;">Click physical subsystems to inspect real-time sensor metrics and localized AI failure attribution.</p>

        <div style="display:flex; min-height:250px;">
            <div style="flex:1; background:rgba(39,39,42,0.2); border-radius:8px; padding:24px; display:flex; align-items:center; justify-content:center; border-right:1px solid rgba(39,39,42,0.5); position:relative;">
                <div style="width:100%; max-width:360px; position:relative; aspect-ratio:4/3;">
                    <div class="v-subsys" style="position:absolute; top:8%; left:8%; width:84%; height:22%; border-radius:9999px; display:flex; align-items:center; justify-content:center;">
                        Reservoir
                    </div>
                    <div style="position:absolute; top:30%; left:50%; width:2px; height:10%; background:#27272a;"></div>
                    <div class="v-subsys" style="position:absolute; top:42%; left:4%; width:22%; height:22%; display:flex; align-items:center; justify-content:center;">
                        Motor
                    </div>
                    <div style="position:absolute; top:53%; left:26%; width:6%; height:2px; background:#27272a;"></div>
                    <div class="v-subsys active" style="position:absolute; top:40%; left:28%; width:44%; height:30%; display:flex; align-items:center; justify-content:center;">
                        Compressor
                    </div>
                    <div style="position:absolute; top:70%; left:50%; width:2px; height:6%; background:#27272a;"></div>
                    <div class="v-subsys" style="position:absolute; top:76%; left:44%; width:12%; height:18%; display:flex; align-items:center; justify-content:center; writing-mode:vertical-rl; text-orientation:mixed; font-size:12px;">
                        Valves
                    </div>
                </div>
            </div>

            <div style="width:260px; padding:24px; display:flex; flex-direction:column; gap:12px; background:rgba(9,9,11,0.5);">
                <div style="border-bottom:1px solid rgba(39,39,42,0.5); padding-bottom:12px;">
                    <h3 style="margin:0; font-size:20px; font-weight:700; color:#fafafa; letter-spacing:-0.025em;">Compressor</h3>
                </div>
                <div style="display:flex; flex-direction:column; gap:12px; font-size:14px;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#a1a1aa;">Health Status</span>
                        <span style="color:#22c55e; font-weight:700;">100.0%</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#a1a1aa;">Temperature</span>
                        <span style="color:#fafafa; font-weight:600;">{temp_val} C</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#a1a1aa;">Pressure/Current</span>
                        <span style="color:#fafafa; font-weight:600;">{pressure_val} bar</span>
                    </div>
                    <div style="margin-top:8px; padding-top:12px; border-top:1px solid rgba(39,39,42,0.5); display:flex; flex-direction:column; gap:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#a1a1aa;">SHAP Impact</span>
                            <span style="font-weight:700; color:#ef4444; background:rgba(239,68,68,0.1); padding:2px 8px; border-radius:4px; font-size:13px;">+0.0%</span>
                        </div>
                        <div style="display:flex; flex-direction:column; gap:4px; margin-top:4px;">
                            <span style="font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#a1a1aa;">Primary Reason</span>
                            <span style="font-size:14px; font-weight:500; color:#fafafa;">Normal operation</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """)

    # Live Sensor Telemetry
    st.html("""
    <div class="v-card" style="padding-bottom:0;">
        <div style="margin-bottom:4px;">
            <div style="font-size:18px; font-weight:700; color:#fafafa;">Live Sensor Telemetry</div>
            <p style="font-size:14px; color:#a1a1aa; margin:4px 0 0 0;">Raw time-series data from the edge device.</p>
        </div>
    </div>
    """)

    t_idx = pd.date_range(datetime.now().strftime("%Y-%m-%d"), periods=60, freq="2s")
    np.random.seed(42)
    df = pd.DataFrame({
        "TP2": np.random.normal(pressure_val, 0.3, 60),
        "TP3": np.random.normal(tp3_val, 0.2, 60),
        "H1":  np.random.normal(0.05, 0.005, 60) * 100,
    }, index=t_idx)

    fig_tele = go.Figure()
    fig_tele.add_trace(go.Scatter(x=df.index, y=df["TP2"], mode='lines', name='Pressure TP2',
                                  line=dict(color='#10b981', width=2)))
    fig_tele.add_trace(go.Scatter(x=df.index, y=df["TP3"], mode='lines', name='Pressure TP3',
                                  line=dict(color='#3b82f6', width=2)))
    fig_tele.add_trace(go.Scatter(x=df.index, y=df["H1"],  mode='lines', name='Pressure H1',
                                  line=dict(color='#8b5cf6', width=2)))
    fig_tele.update_layout(
        height=350,
        margin=dict(l=0, r=5, t=5, b=30),
        paper_bgcolor='rgba(9,9,11,0.5)',
        plot_bgcolor='rgba(9,9,11,0.5)',
        xaxis=dict(showgrid=False, color='#a1a1aa', tickformat='%H:%M:%S'),
        yaxis=dict(showgrid=True, gridcolor='rgba(39,39,42,0.5)', gridwidth=1,
                   griddash='dot', color='#a1a1aa'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=12, color='#a1a1aa'), bgcolor='rgba(0,0,0,0)'),
    )
    st.plotly_chart(fig_tele, use_container_width=True, key='telemetry')

with right_col:
    # AI Predictive Analysis
    st.html("""
    <div class="v-card" style="margin-bottom:24px;">
        <div style="font-size:18px; font-weight:700; color:#fafafa; display:flex; align-items:center; gap:8px; margin-bottom:16px;">
            AI Predictive Analysis
        </div>
        <div style="display:flex; flex-direction:column; gap:0;">
            <div class="v-pred-row">
                <span class="v-pred-label">2 Hours</span>
                <span class="v-pred-value">0% <span class="v-dot-green"></span></span>
            </div>
            <div class="v-pred-row">
                <span class="v-pred-label">4 Hours</span>
                <span class="v-pred-value">0% <span class="v-dot-green"></span></span>
            </div>
            <div class="v-pred-row">
                <span class="v-pred-label">8 Hours</span>
                <span class="v-pred-value">0% <span class="v-dot-green"></span></span>
            </div>
        </div>
        <div style="height:1px; background:rgba(39,39,42,0.5); margin:16px 0;"></div>
        <div style="display:flex; flex-direction:column; gap:8px; font-size:14px;">
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#a1a1aa;">Overall Risk</span>
                <span style="color:#22c55e; font-weight:700;">NORMAL</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#a1a1aa;">Confidence</span>
                <span style="color:#fafafa; font-weight:700;">99.0%</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:8px; padding-top:8px; border-top:1px solid rgba(39,39,42,0.5);">
                <span style="color:#a1a1aa;">Recommendation</span>
                <span style="color:#fafafa; font-weight:600; text-align:right;">Continue operation</span>
            </div>
        </div>
    </div>
    """)

    # Top AI Contributors
    shap_data = [
        ("TP2", 12.0, "#ef4444"),
        ("TP3", 8.0, "#f59e0b"),
        ("H1", 5.0, "#3b82f6"),
        ("DV pressure", 3.0, "#3b82f6"),
        ("Reservoirs", 2.0, "#3b82f6"),
    ]
    max_shap = max(s[1] for s in shap_data)

    shap_bars_html = ""
    for name, val, color in shap_data:
        w_pct = (val / max_shap) * 100
        shap_bars_html += f"""
        <div class="v-shap-bar-row">
            <div class="v-shap-bar-label">{name}</div>
            <div class="v-shap-bar-track">
                <div class="v-shap-bar-fill" style="width:{w_pct}%; background:{color};"></div>
            </div>
        </div>"""

    st.html(f"""
    <div class="v-card" style="height:300px; margin-bottom:24px;">
        <div style="font-size:18px; font-weight:700; color:#fafafa; display:flex; align-items:center; gap:8px; margin-bottom:2px;">
            Top AI Contributors
        </div>
        <p style="font-size:12px; color:#a1a1aa; margin-bottom:16px;">The physical sensors driving the AI's current risk assessment.</p>
        <div style="display:flex; flex-direction:column; gap:2px;">
            {shap_bars_html}
        </div>
    </div>
    """)

    # 4H Risk Trend
    st.html("""
    <div class="v-card" style="margin-bottom:24px; padding:12px 20px;">
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:14px; font-weight:600; color:#a1a1aa; margin-bottom:8px;">
            <div style="display:flex; align-items:center; gap:6px;">
                4H Risk Trend
            </div>
            <span style="color:#22c55e; font-weight:700;">0.0%</span>
        </div>
        <div style="height:60px; position:relative;">
            <svg width="100%" height="60" viewBox="0 0 200 60" preserveAspectRatio="none">
                <polyline points="0,55 20,54 40,56 60,53 80,55 100,54 120,56 140,55 160,54 180,55 200,54"
                          fill="none" stroke="#22c55e" stroke-width="2"/>
            </svg>
        </div>
    </div>
    """)

    # Event Timeline
    st.html(f"""
    <div class="v-card" style="height:250px;">
        <div style="font-size:14px; font-weight:600; color:#a1a1aa; display:flex; align-items:center; gap:8px; margin-bottom:16px;">
            Event Timeline
        </div>
        <div style="position:relative; padding-left:20px;">
            <div style="position:absolute; left:5px; top:0; bottom:0; width:1px; background:rgba(39,39,42,0.5);"></div>
            <div style="display:flex; gap:12px; padding-bottom:16px; position:relative;">
                <div style="position:absolute; left:-18px; top:4px; width:10px; height:10px; border-radius:50%; background:#22c55e; border:2px solid #09090b; z-index:1;"></div>
                <div>
                    <div style="font-size:12px; font-family:monospace; color:#a1a1aa;">{now_str}</div>
                    <div style="font-size:14px; font-weight:500; color:#fafafa;">System initialized</div>
                </div>
            </div>
        </div>
    </div>
    """)

# Footer
st.html(f"""
<div class="v-footer">
    <span class="v-footer-pill">
        MetroPT Dataset &middot; Stacked Ensemble (LightGBM + XGBoost + CNN) &middot; Inference {latency_ms}ms &middot; Last Update {now_str}
    </span>
</div>
""")
