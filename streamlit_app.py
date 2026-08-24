import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Streamlit Page Config
st.set_page_config(
    page_title="MetroPT APU Predictive Maintenance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching exact Vercel Next.js dashboard pixel sizes and colors
st.markdown("""
<style>
    /* Dark Background matching Vercel UI */
    .stApp {
        background-color: #070a12;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    }

    /* Hide Streamlit Header Elements */
    header[data-testid="stHeader"] { visibility: hidden; height: 0px; }
    div[data-testid="stToolbar"] { visibility: hidden; }

    /* Custom Glassmorphism Card Box */
    .card-box {
        background-color: #0d1322;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Status Banner - Normal */
    .banner-normal {
        background-color: #064e3b;
        border: 1px solid #059669;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .banner-title {
        color: #34d399;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .banner-subtitle {
        color: #a7f3d0;
        font-size: 13px;
        margin-top: 2px;
    }

    /* Gauge Card Headers */
    .gauge-header {
        font-size: 12px;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }

    /* System Badges */
    .badge-pill {
        background-color: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 11px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .dot-online {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10b981;
        display: inline-block;
    }

    /* Metric Row */
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid #1e293b;
    }
    .metric-row:last-child {
        border-bottom: none;
    }

    /* Schematic Box */
    .schematic-box {
        border: 2px solid #334155;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        background-color: #0f172a;
        color: #f8fafc;
        font-weight: 600;
        font-size: 13px;
    }
    .schematic-box.selected {
        border-color: #f8fafc;
        box-shadow: 0 0 12px rgba(248, 250, 252, 0.2);
    }

    /* Timeline Event Item */
    .timeline-item {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Top Title Bar & System Badges
st.html("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-top: 10px;">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="background: #0f172a; padding: 12px; border-radius: 12px; border: 1px solid #1e293b; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 22px;">🛡️</span>
            </div>
            <div>
                <h1 style="margin:0; font-size: 26px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">MetroPT APU Predictive Maintenance</h1>
                <p style="margin:2px 0 0 0; font-size: 13px; color: #64748b;">Real-time monitoring and AI diagnostics for railway Air Production Units.</p>
            </div>
        </div>
        <div style="display: flex; gap: 8px;">
            <span class="badge-pill">⚡ Inference 21ms</span>
            <span class="badge-pill"><span class="dot-online"></span> Redis</span>
            <span class="badge-pill"><span class="dot-online"></span> WebSocket (Live)</span>
            <span class="badge-pill">⚙️ Model v2.1</span>
        </div>
    </div>
""")

# Green Status Banner
st.html("""
    <div class="banner-normal">
        <div style="background-color: #059669; color: white; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 16px;">
            ✓
        </div>
        <div>
            <div class="banner-title">CONTINUE OPERATION</div>
            <div class="banner-subtitle">All telemetry nominal. No immediate maintenance required.</div>
        </div>
    </div>
""")

# Live Telemetry Data Values matching screenshot
np.random.seed(42)
pressure_val = round(float(8.1 + (np.random.rand() * 0.2 - 0.1)), 1)
temp_val = round(float(60.7 + (np.random.rand() * 0.4 - 0.2)), 1)
current_val = round(float(8.0 + (np.random.rand() * 0.4 - 0.2)), 1)

# Gauge Component Function
def create_gauge(value: float, max_v: float, unit: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': f" <br><span style='font-size:12px;color:#94a3b8;font-weight:600;'>{unit}</span>", 'font': {'size': 28, 'color': '#ffffff', 'family': 'Inter, sans-serif'}},
        gauge={
            'axis': {'range': [0, max_v], 'tickwidth': 0, 'tickcolor': "rgba(0,0,0,0)"},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': "#1e293b",
            'bordercolor': "rgba(0,0,0,0)",
        }
    ))
    fig.update_layout(
        height=140,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# Row 1: Three Gauge Cards
g_col1, g_col2, g_col3 = st.columns(3)

with g_col1:
    st.html('<div class="card-box"><div class="gauge-header"><span>⏱️</span> PRESSURE</div>')
    st.plotly_chart(create_gauge(pressure_val, 15, "BAR", "#3b82f6"), use_container_width=True, key="gauge_p")
    st.html('</div>')

with g_col2:
    st.html('<div class="card-box"><div class="gauge-header"><span>🌡️</span> TEMPERATURE</div>')
    st.plotly_chart(create_gauge(temp_val, 100, "°C", "#ef4444"), use_container_width=True, key="gauge_t")
    st.html('</div>')

with g_col3:
    st.html('<div class="card-box"><div class="gauge-header"><span>⚡</span> MOTOR CURRENT</div>')
    st.plotly_chart(create_gauge(current_val, 20, "AMPS", "#f59e0b"), use_container_width=True, key="gauge_c")
    st.html('</div>')

# Row 2: Interactive APU Schematic (Left) + AI Predictive Analysis (Right)
mid_col1, mid_col2 = st.columns([3, 2])

with mid_col1:
    st.html("""
        <div class="card-box">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-weight: 700; font-size: 15px; color: #ffffff;">⚙️ Interactive APU Schematic</div>
                <span class="badge-pill" style="color: #34d399; border-color: #065f46;"><span class="dot-online"></span> Live Data Connected</span>
            </div>
            <p style="font-size: 12px; color: #64748b; margin-bottom: 24px;">Click physical subsystems to inspect real-time sensor metrics and localized AI failure attribution.</p>
            
            <div style="display: flex; gap: 30px; align-items: center;">
                <div style="flex: 2;">
                    <!-- Subsystem Boxes -->
                    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                        <div class="schematic-box" style="width: 140px;">Reservoir</div>
                    </div>
                    <div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
                        <div class="schematic-box" style="width: 100px;">Motor</div>
                        <div class="schematic-box selected" style="width: 120px;">Compressor</div>
                    </div>
                    <div style="display: flex; justify-content: center; margin-top: 20px;">
                        <div class="schematic-box" style="width: 90px; border-color: #94a3b8;">Valves</div>
                    </div>
                </div>
                
                <div style="flex: 1; border-left: 1px solid #1e293b; padding-left: 20px;">
                    <div style="font-weight: 700; font-size: 15px; margin-bottom: 12px; color: #ffffff;">Valves</div>
                    <div style="font-size: 13px; color: #94a3b8; margin-bottom: 6px;">Health Status <span style="float: right; color: #34d399; font-weight: 700;">100.0%</span></div>
                    <div style="font-size: 13px; color: #94a3b8; margin-bottom: 6px;">Temperature <span style="float: right; color: #f1f5f9; font-weight: 600;">9.0°C</span></div>
                    <div style="font-size: 13px; color: #94a3b8; margin-bottom: 16px;">Pressure/Current <span style="float: right; color: #f1f5f9; font-weight: 600;">0.21 bar</span></div>
                    
                    <div style="font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 0.5px;">SHAP IMPACT <span style="float: right; color: #ef4444;">+0.0%</span></div>
                    <div style="font-size: 11px; font-weight: 700; color: #64748b; margin-top: 10px;">PRIMARY REASON</div>
                    <div style="font-size: 13px; color: #f1f5f9; font-weight: 600;">Normal operation</div>
                </div>
            </div>
        </div>
    """)

with mid_col2:
    st.html("""
        <div class="card-box" style="height: 100%;">
            <div style="font-weight: 700; font-size: 15px; color: #ffffff; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                <span>🧠</span> AI Predictive Analysis
            </div>
            
            <div class="metric-row">
                <span style="font-size: 13px; font-weight: 600; color: #cbd5e1;">2 Hours</span>
                <span style="font-size: 13px; font-weight: 700; color: #ffffff;">0% <span class="dot-online" style="margin-left: 6px;"></span></span>
            </div>
            
            <div class="metric-row">
                <span style="font-size: 13px; font-weight: 600; color: #cbd5e1;">4 Hours</span>
                <span style="font-size: 13px; font-weight: 700; color: #ffffff;">0% <span class="dot-online" style="margin-left: 6px;"></span></span>
            </div>
            
            <div class="metric-row">
                <span style="font-size: 13px; font-weight: 600; color: #cbd5e1;">8 Hours</span>
                <span style="font-size: 13px; font-weight: 700; color: #ffffff;">0% <span class="dot-online" style="margin-left: 6px;"></span></span>
            </div>
            
            <div style="margin-top: 18px; padding-top: 12px; border-top: 1px solid #1e293b;">
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8; margin-bottom: 8px;">
                    <span>Overall Risk</span>
                    <span style="color: #34d399; font-weight: 700;">NORMAL</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8; margin-bottom: 14px;">
                    <span>Confidence</span>
                    <span style="color: #ffffff; font-weight: 600;">0.0%</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8;">
                    <span>Recommendation</span>
                    <span style="color: #ffffff; font-weight: 600;">Continue operation</span>
                </div>
            </div>
        </div>
    """)

# Row 3: Live Sensor Telemetry (Left) + Top AI Contributors & 4H Risk Trend (Right)
bot_col1, bot_col2 = st.columns([3, 2])

with bot_col1:
    st.html("""
        <div class="card-box" style="margin-bottom: 0px; border-bottom: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <div style="font-weight: 700; font-size: 15px; color: #ffffff;">Live Sensor Telemetry</div>
            <div style="font-size: 12px; color: #64748b;">Raw time-series data from the edge device.</div>
        </div>
    """)
    
    # Generate Multi-Colored Line Chart
    t_steps = pd.date_range("00:01:35", periods=60, freq="1s")
    df_chart = pd.DataFrame({
        "TP2": np.random.normal(8.0, 0.4, 60),
        "TP3": np.random.normal(9.5, 0.3, 60),
        "H1": np.random.normal(6.5, 0.5, 60),
        "Motor_current": np.random.normal(7.8, 0.6, 60)
    }, index=t_steps)

    fig_lines = go.Figure()
    fig_lines.add_trace(go.Scatter(x=df_chart.index, y=df_chart["TP2"], mode='lines', name='TP2', line=dict(color='#3b82f6', width=1.5)))
    fig_lines.add_trace(go.Scatter(x=df_chart.index, y=df_chart["TP3"], mode='lines', name='TP3', line=dict(color='#10b981', width=1.5)))
    fig_lines.add_trace(go.Scatter(x=df_chart.index, y=df_chart["H1"], mode='lines', name='H1', line=dict(color='#06b6d4', width=1.5)))
    fig_lines.add_trace(go.Scatter(x=df_chart.index, y=df_chart["Motor_current"], mode='lines', name='Motor current', line=dict(color='#a855f7', width=1.5)))

    fig_lines.update_layout(
        height=260,
        margin=dict(l=30, r=20, t=10, b=30),
        paper_bgcolor='#0d1322',
        plot_bgcolor='#0d1322',
        xaxis=dict(showgrid=True, gridcolor='#1e293b', color='#64748b'),
        yaxis=dict(showgrid=True, gridcolor='#1e293b', color='#64748b'),
        showlegend=False
    )
    st.plotly_chart(fig_lines, use_container_width=True, key="live_lines")

with bot_col2:
    st.html("""
        <div class="card-box" style="margin-bottom: 12px;">
            <div style="font-weight: 700; font-size: 14px; color: #ffffff; margin-bottom: 4px;">📊 Top AI Contributors</div>
            <div style="font-size: 11px; color: #64748b; margin-bottom: 14px;">The physical sensors driving the AI's current risk assessment.</div>
            
            <div style="font-size: 12px; color: #94a3b8; display: flex; flex-direction: column; gap: 8px;">
                <div style="display: flex; justify-content: space-between;"><span>TP2</span><span>0.12</span></div>
                <div style="display: flex; justify-content: space-between;"><span>TP3</span><span>0.08</span></div>
                <div style="display: flex; justify-content: space-between;"><span>H1</span><span>0.05</span></div>
                <div style="display: flex; justify-content: space-between;"><span>DV pressure</span><span>0.03</span></div>
                <div style="display: flex; justify-content: space-between;"><span>Reservoirs</span><span>0.02</span></div>
            </div>
        </div>
        
        <div class="card-box" style="margin-bottom: 12px; padding: 14px 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 12px; font-weight: 600; color: #94a3b8;">📈 4H Risk Trend</span>
                <span style="font-size: 13px; font-weight: 700; color: #34d399;">0.0%</span>
            </div>
            <div style="height: 4px; background: #10b981; border-radius: 2px; margin-top: 8px;"></div>
        </div>
        
        <div class="card-box" style="padding: 14px 20px;">
            <div style="font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 8px;">🕐 Event Timeline</div>
            <div class="timeline-item">
                <span class="dot-online" style="margin-top: 4px;"></span>
                <div>
                    <div style="font-size: 10px; color: #64748b;">""" + datetime.now().strftime("%I:%M:%S %p") + """</div>
                    <div style="font-size: 12px; color: #f1f5f9; font-weight: 600;">System initialized</div>
                </div>
            </div>
        </div>
    """)

# Footer
st.html("""
    <div style="text-align: center; font-size: 11px; color: #475569; margin-top: 24px; padding: 12px 0;">
        <span style="background-color: #0f172a; border: 1px solid #1e293b; padding: 6px 18px; border-radius: 20px;">
            MetroPT Dataset • Stacked Ensemble (LightGBM + XGBoost + CNN) • Inference 21ms • Last Update """ + datetime.now().strftime("%I:%M:%S %p") + """
        </span>
    </div>
""")
