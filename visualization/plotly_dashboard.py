# ==============================================================================
# PLOTLY DASHBOARD - visualization/plotly_dashboard.py (FINAL VISUAL FIXES)
# ==============================================================================
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
import os
import joblib
import numpy as np
from typing import Optional, Dict
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

# Initialize Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CSS Styles
CUSTOM_CSS = """
:root {
    --primary-color: #6366f1;
    --secondary-color: #8b5cf6;
    --accent-color: #ec4899;
    --dark-bg: #0f172a;
    --card-bg: #1e293b;
    --ml-card-bg: #1e1b4b;
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--dark-bg);
    color: var(--text-primary);
    overflow-x: hidden;
    font-size: 18px;
    line-height: 1.6;
}

.animated-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    background: var(--dark-bg);
    background-image: 
        radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 40% 20%, rgba(236, 72, 153, 0.1) 0%, transparent 50%);
    animation: pulse 15s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.navbar {
    background: rgba(30, 41, 59, 0.9);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    padding: 1.5rem 0;
    position: sticky;
    top: 0;
    z-index: 1000;
    animation: fadeInUp 0.6s ease-out;
}

.nav-brand {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}

.hero-section {
    padding: 6rem 0 4rem 0;
    text-align: center;
    animation: fadeInUp 0.8s ease-out 0.2s backwards;
}

.hero-title {
    font-size: 4rem;
    font-weight: 800;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 1.5rem;
    color: var(--text-secondary);
    max-width: 900px;
    margin: 0 auto 2rem;
    line-height: 1.8;
    font-weight: 400;
}

/* --- GRID SYSTEMS --- */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-bottom: 2rem;
}

.feature-card {
    background: var(--card-bg);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 2.5rem;
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease-out backwards;
}

.feature-card:hover {
    transform: translateY(-5px);
    border-color: var(--primary-color);
    box-shadow: 0 10px 40px rgba(99, 102, 241, 0.2);
}

.ml-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 2rem;
    margin-bottom: 4rem;
}

.ml-card {
    background: linear-gradient(145deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 2.5rem;
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease-out backwards;
    position: relative;
    overflow: hidden;
}

.ml-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(to bottom, #6366f1, #ec4899);
}

.ml-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 50px rgba(139, 92, 246, 0.15);
    border-color: var(--accent-color);
}

.feature-icon, .ml-icon {
    font-size: 2.5rem;
    margin-bottom: 1.5rem;
}
.feature-icon { color: var(--primary-color); }
.ml-icon { color: var(--accent-color); }

.feature-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.feature-desc {
    color: var(--text-secondary);
    line-height: 1.7;
    font-size: 1.1rem;
}

/* --- DASHBOARD ELEMENTS --- */
.dashboard-card {
    background: var(--card-bg);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 2.5rem;
    transition: all 0.3s ease;
    animation: fadeInUp 0.8s ease-out backwards;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    margin-bottom: 2rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2);
}

.card-icon {
    width: 56px;
    height: 56px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.75rem;
    color: white;
}
.card-icon.primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.card-icon.secondary { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
.card-icon.accent { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }

.card-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
}

.section-divider {
    height: 4px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 2px;
    margin: 4rem 0;
    opacity: 0.5;
}

.footer {
    margin-top: 5rem;
    padding: 4rem 0;
    border-top: 1px solid rgba(99, 102, 241, 0.2);
    text-align: center;
    color: var(--text-secondary);
}

.graph-container {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.05);
}

/* --- ABOUT SECTION STYLES --- */
.info-box {
    background: rgba(15, 23, 42, 0.6); /* Increased opacity for better contrast */
    border-radius: 12px;
    padding: 2rem;
    height: 100%;
    border: 1px solid rgba(99, 102, 241, 0.2);
}

.tech-badge {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 1rem;
    font-weight: 600;
    display: inline-block;
    margin: 6px;
    transition: all 0.2s ease;
}

.tech-badge:hover {
    background: rgba(99, 102, 241, 0.3);
    transform: translateY(-2px);
    color: white;
}
"""

class DashboardGenerator:
    """Generates interactive Plotly dashboard"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        logger.info("Dashboard Generator initialized")
    
    def generate_dashboard(self, results_dir: str, graphs_dir: str):
        """Generate interactive dashboard"""
        try:
            logger.info("Starting Dashboard Server...")
            
            app = Dash(__name__, external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                dbc.icons.BOOTSTRAP, 
                'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap'
            ])
            
            app.index_string = f'''
            <!DOCTYPE html>
            <html>
                <head>
                    {{%metas%}}
                    <title>{{%title%}}</title>
                    {{%favicon%}}
                    {{%css%}}
                    <style>{CUSTOM_CSS}</style>
                </head>
                <body>
                    {{%app_entry%}}
                    <footer>
                        {{%config%}}
                        {{%scripts%}}
                        {{%renderer%}}
                    </footer>
                </body>
            </html>
            '''
            
            # --- Load Data ---
            sup_fig, unsup_fig = self._create_split_performance_charts(results_dir)
            feat_fig = self._create_feature_importance_chart("data/models", "data/features/final_user_features.csv")
            temp_path = "data/features/test_data_cleaned_temporal.csv"
            if not os.path.exists(temp_path):
                temp_path = "data/features/final_user_features.csv"
            temp_fig = self._create_temporal_chart(temp_path)
            
            # --- Layout ---
            app.layout = html.Div([
                html.Div(className="animated-bg"),
                
                # Navbar
                html.Nav([
                    dbc.Container([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className="bi bi-shield-check me-3"),
                                    "CIB Detection"
                                ], className="nav-brand")
                            ], width=12)
                        ], align="center")
                    ])
                ], className="navbar"),
                
                # Main Container 
                dbc.Container([
                    # Hero
                    html.Div([
                        html.H1("Detection of Coordinated Inauthentic Behavior", className="hero-title"),
                        html.P([
                            "CIB detection framework integrating Social Network Analysis, Stylometric NLP, and machine learning to uncover coordinated behavioral anomalies"
                        ], className="hero-subtitle"),
                    ], className="hero-section"),
                    
                    # --- ROW 1: STANDARD FEATURE CARDS ---
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-diagram-3 feature-icon"),
                            html.H3("Social Network Analysis", className="feature-title"),
                            html.P("Graph-based detection using Centrality and Modularity to find coordinated user clusters.", className="feature-desc")
                        ], className="feature-card", style={'animationDelay': '0.2s'}),
                        
                        html.Div([
                            html.I(className="bi bi-vector-pen feature-icon"),
                            html.H3("Stylometric NLP", className="feature-title"),
                            html.P("Analysis of writing style, vocabulary richness, and sentence structure to detect automation.", className="feature-desc")
                        ], className="feature-card", style={'animationDelay': '0.4s'}),
                        
                        html.Div([
                            html.I(className="bi bi-clock-history feature-icon"),
                            html.H3("Temporal Analysis", className="feature-title"),
                            html.P("Detection of algorithmic posting schedules, burstiness, and inter-post time intervals.", className="feature-desc")
                        ], className="feature-card", style={'animationDelay': '0.6s'}),
                    ], className="feature-grid"),

                    # --- ROW 2: ML CARDS (DISTINCT STYLING) ---
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-robot ml-icon"),
                            html.H3("Supervised Learning", className="feature-title"),
                            html.P("Classification using XGBoost, Random Forest, Logistic Regression.", className="feature-desc")
                        ], className="ml-card", style={'animationDelay': '0.8s'}),
                        
                        html.Div([
                            html.I(className="bi bi-cpu ml-icon"),
                            html.H3("Unsupervised ML", className="feature-title"),
                            html.P("Pattern discovery with K-Means, DBSCAN, and Isolation Forest.", className="feature-desc")
                        ], className="ml-card", style={'animationDelay': '1.0s'}),
                    ], className="ml-grid"),
                    
                    html.Div(className="section-divider"),
                    
                    # Performance Section
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Div(html.I(className="bi bi-trophy"), className="card-icon primary"),
                                html.H2("Model Performance Leaderboard", className="card-title")
                            ], className="card-header"),
                            
                            dbc.Tabs([
                                dbc.Tab(
                                    html.Div([
                                        dcc.Graph(id='supervised-performance', figure=sup_fig, config={'displayModeBar': False}, className="graph-container")
                                    ], style={'padding': '1rem'}),
                                    label="Supervised Models",
                                ),
                                dbc.Tab(
                                    html.Div([
                                        dcc.Graph(id='unsupervised-performance', figure=unsup_fig, config={'displayModeBar': False}, className="graph-container")
                                    ], style={'padding': '1rem'}),
                                    label="Unsupervised Models",
                                ),
                            ], active_tab="tab-0"),
                        ], className="dashboard-card", style={'animationDelay': '0.3s'}),
                    ]),
                    
                    # Analytics Row
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Div(html.I(className="bi bi-bar-chart-steps"), className="card-icon secondary"),
                                    html.H2("Top Predictive Features", className="card-title")
                                ], className="card-header"),
                                dcc.Graph(id='feature-importance', figure=feat_fig, config={'displayModeBar': False}, className="graph-container")
                            ], className="dashboard-card", style={'animationDelay': '0.5s'})
                        ], md=6),
                        
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Div(html.I(className="bi bi-activity"), className="card-icon accent"),
                                    html.H2("User Activity Distribution", className="card-title")
                                ], className="card-header"),
                                dcc.Graph(id='temporal-activity', figure=temp_fig, config={'displayModeBar': False}, className="graph-container")
                            ], className="dashboard-card", style={'animationDelay': '0.7s'})
                        ], md=6),
                    ], className="mb-4"),
                    
                    html.Div(className="section-divider"),

                    # --- UPDATED ABOUT SECTION (HIGH VISIBILITY) ---
                    html.Div([
                        # Header & Description
                        html.Div([
                            html.H2("About", className="text-center mb-4", style={'fontWeight': '700', 'letterSpacing': '-1px', 'color': 'white'}),
                            html.P("This platform implements a production-ready pipeline for detecting Coordinated Inauthentic Behavior (CIB). By fusing social network analysis with stylometric NLP and unsupervised anomaly detection, it moves beyond simple volume-based filtering to identify complex behavioral patterns.",
                                   className="text-center mb-5", 
                                   # Explicitly set color to light grey for readability
                                   style={'maxWidth': '800px', 'margin': '0 auto', 'fontSize': '1.15rem', 'color': '#cbd5e1'}), 
                        ]),

                        dbc.Row([
                            # Column 1: ML Models
                            dbc.Col([
                                html.Div([
                                    html.H4([html.I(className="bi bi-cpu-fill me-2"), "ML Engine"], className="mb-4", style={'color': '#818cf8', 'fontWeight': '600'}),
                                    
                                    html.H6("Supervised Classifiers", className="text-white mb-2 small text-uppercase", style={'letterSpacing': '1px', 'opacity': '0.9'}),
                                    html.Div([
                                        html.Span("XGBoost", className="tech-badge"),
                                        html.Span("Random Forest", className="tech-badge"),
                                        html.Span("Logistic Regression", className="tech-badge"),
                                        
                                    ], className="mb-4"),
                                    
                                    html.H6("Unsupervised Detection", className="text-white mb-2 small text-uppercase", style={'letterSpacing': '1px', 'opacity': '0.9'}),
                                    html.Div([
                                        html.Span("DBSCAN", className="tech-badge"),
                                        html.Span("K-Means", className="tech-badge"),
                                        html.Span("Isolation Forest", className="tech-badge"),
                                    ], className="mb-3"),
                                    
                                    # REMOVED ENSEMBLE VOTING SECTION
                                    
                                ], className="info-box")
                            ], md=6, className="mb-4 mb-md-0"),

                            # Column 2: System Specs
                            dbc.Col([
                                html.Div([
                                    html.H4([html.I(className="bi bi-hdd-network-fill me-2"), "System Pipeline"], className="mb-4", style={'color': '#818cf8', 'fontWeight': '600'}),
                                    
                                    html.Div([
                                        html.Div([
                                            html.I(className="bi bi-layers-half me-3", style={'fontSize': '1.5rem', 'color': '#6366f1'}),
                                            html.Div([
                                                html.H6("120-Dim Feature Extraction", className="mb-0 text-white"),
                                                html.Small("Stylometry + Temporal + BERT Embeddings", style={'color': '#94a3b8'}) # Explicit light color
                                            ])
                                        ], className="d-flex align-items-center mb-4"),

                                        html.Div([
                                            html.I(className="bi bi-share-fill me-3", style={'fontSize': '1.5rem', 'color': '#8b5cf6'}),
                                            html.Div([
                                                html.H6("Graph Network Analysis", className="mb-0 text-white"),
                                                html.Small("Centrality, Modularity & Clique Detection", style={'color': '#94a3b8'}) # Explicit light color
                                            ])
                                        ], className="d-flex align-items-center mb-4"),

                                         html.Div([
                                            html.I(className="bi bi-shield-lock-fill me-3", style={'fontSize': '1.5rem', 'color': '#ec4899'}),
                                            html.Div([
                                                html.H6("Hard Mode Training", className="mb-0 text-white"),
                                                html.Small("Excludes volume metrics to prevent overfitting", style={'color': '#94a3b8'}) # Explicit light color
                                            ])
                                        ], className="d-flex align-items-center"),
                                    ])
                                ], className="info-box")
                            ], md=6),
                        ])
                    ], className="dashboard-card", style={'animationDelay': '0.9s'}),
                    
                    # Footer
                    html.Div([
                        html.P("© 2026 CIB Detection Platform ", style={'fontSize': '1.1rem'}),
                        html.P("Press Ctrl+C in terminal to stop server.", style={'fontSize': '0.9rem', 'marginTop': '1rem', 'color': '#64748b'})
                    ], className="footer")
                    
                ], fluid=False, style={'maxWidth': '1600px', 'padding': '2rem'})
            ], style={'minHeight': '100vh', 'position': 'relative'})
            
            logger.info("Dashboard generated successfully")
            port = self.config.get('dashboard', {}).get('port', 8050)
            logger.info(f"OPEN BROWSER AT: http://127.0.0.1:{port}/")
            app.run_server(debug=False, port=port)
            
        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
            import traceback
            traceback.print_exc()

    # ==================== HELPER METHODS (UNCHANGED) ====================
    def _create_split_performance_charts(self, results_dir: str):
        empty_fig = self._empty_fig("No Data")
        try:
            results_path = os.path.join(results_dir, 'evaluation_metrics.csv')
            if not os.path.exists(results_path): return empty_fig, empty_fig
            df = pd.read_csv(results_path).fillna(0)
            unsupervised_models = ['kmeans', 'dbscan', 'isolation_forest']
            sup_df = df[(df['f1_score'] > 0) & (~df['model'].isin(unsupervised_models))].copy()
            if not sup_df.empty:
                sup_plot = sup_df.melt(id_vars=['model'], value_vars=['accuracy', 'f1_score'], var_name='Metric', value_name='Score')
                sup_fig = px.bar(sup_plot, x='model', y='Score', color='Metric', barmode='group', color_discrete_map={'accuracy': '#667eea', 'f1_score': '#f5576c'})
                sup_fig.update_layout(yaxis_range=[0, 1.1], showlegend=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f1f5f9', family='Inter', size=14), xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)'), margin=dict(l=40, r=40, t=40, b=40))
            else: sup_fig = self._empty_fig("No Supervised Models")
            
            unsup_df = df[df['model'].isin(unsupervised_models)].copy()
            if not unsup_df.empty:
                unsup_fig = px.bar(unsup_df, x='model', y='silhouette', title="Clustering Quality", color_discrete_sequence=['#4facfe'])
                unsup_fig.update_layout(yaxis_range=[-1, 1], plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f1f5f9', family='Inter', size=14), xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)'), margin=dict(l=40, r=40, t=60, b=40), title=dict(font=dict(size=16)))
            else: unsup_fig = self._empty_fig("No Unsupervised Models")
            return sup_fig, unsup_fig
        except Exception: return empty_fig, empty_fig

    def _create_feature_importance_chart(self, models_dir: str, features_path: str):
        try:
            model = None
            if not os.path.exists(models_dir): return self._empty_fig("Models Directory Missing")
            for name in ['xgboost.pkl', 'random_forest.pkl']:
                p = os.path.join(models_dir, name)
                if os.path.exists(p):
                    model = joblib.load(p)
                    break
            if model is None: return self._empty_fig("No Trained Models Found")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if hasattr(model, 'feature_names_in_'): final_names = list(model.feature_names_in_)
                else:
                    if not os.path.exists(features_path): return self._empty_fig("Features File Missing")
                    df_feat = pd.read_csv(features_path)
                    exclude = ['user_id', 'community_id', 'screen_name', 'created_at', 'label', 'is_bot', 'total_posts', 'posts_per_day', 'statuses_count', 'Unnamed: 0']
                    final_names = [c for c in df_feat.columns if c not in exclude and np.issubdtype(df_feat[c].dtype, np.number)]
                if len(importances) != len(final_names): return self._empty_fig("Dimension Mismatch")
                fi_df = pd.DataFrame({'Feature': final_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)
                fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale=[[0, '#667eea'], [1, '#f5576c']])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f1f5f9', family='Inter', size=14), xaxis=dict(showgrid=True, gridcolor='rgba(99, 102, 241, 0.1)'), yaxis=dict(categoryorder='total ascending', showgrid=False), margin=dict(l=40, r=40, t=40, b=40))
                return fig
            else: return self._empty_fig("No Feature Info")
        except Exception: return self._empty_fig("Error Loading Features")

    def _create_temporal_chart(self, features_path: str):
        try:
            if os.path.exists(features_path):
                df = pd.read_csv(features_path)
                x_col = next((c for c in ['total_posts', 'statuses_count', 'followers_count'] if c in df.columns), None)
                y_col = next((c for c in ['posts_per_day', 'favourites_count', 'listed_count'] if c in df.columns and c != x_col), None)
                if x_col and y_col:
                    fig = px.scatter(df, x=x_col, y=y_col, opacity=0.6, color='label' if 'label' in df.columns else None, color_discrete_sequence=['#667eea', '#f5576c'])
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f1f5f9', family='Inter', size=14), xaxis=dict(showgrid=True, gridcolor='rgba(99, 102, 241, 0.1)'), yaxis=dict(showgrid=True, gridcolor='rgba(99, 102, 241, 0.1)'), margin=dict(l=40, r=40, t=40, b=40))
                    return fig
        except Exception: pass
        return self._empty_fig("No Activity Data")

    def _empty_fig(self, text):
        return go.Figure().update_layout(title=text, xaxis={"visible": False}, yaxis={"visible": False}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8', family='Inter', size=16), annotations=[{"text": text, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 20, "color": "#64748b"}}])

if __name__ == "__main__":
    print("Starting Professional CIB Dashboard...")
    dashboard = DashboardGenerator()
    dashboard.generate_dashboard(results_dir="data/results", graphs_dir="data/graphs")