import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import minimize

# Simulate sector returns (10 sectors, 100 observations)
np.random.seed(42)
returns = np.random.randn(100, 10)

# Create sectors and covariance matrix
sectors = [f"Sector {i+1}" for i in range(10)]
cov_matrix = np.cov(returns.T)

# Benchmark Weights: Random Lognormal Distribution
np.random.seed(42)
raw_weights = np.random.lognormal(mean=0, sigma=1, size=10)
benchmark_weights = raw_weights / np.sum(raw_weights)

# Scenario Calculations

# Scenario 1: 25% Absolute Weight for Selected Sectors
scenario_1_weights = benchmark_weights.copy()
scenario_1_weights[:4] = 0.25
scenario_1_weights[4:] = (1 - np.sum(scenario_1_weights[:4])) / 6

# Scenario 2: 50% Active Weight for Selected Sectors
scenario_2_active_weights = np.zeros(10)
scenario_2_active_weights[:4] = 0.5 / 4
scenario_2_active_weights[4:] = -0.5 / 6
scenario_2_weights = benchmark_weights + scenario_2_active_weights

# Scenario 3: Optimized Weights
def optimize_weights():
    # Overweights optimization
    overweights_cov = cov_matrix[:4, :4]
    initial_weights_over = np.full(4, 0.5 / 4)

    def objective_over(w):
        w = w * (0.5 / np.sum(w))
        risk_contrib = w * np.dot(overweights_cov, w) / np.dot(w, np.dot(overweights_cov, w))
        return np.var(risk_contrib)

    res_over = minimize(objective_over, initial_weights_over, bounds=[(0, None)] * 4, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 0.5})
    optimized_weights_over = res_over.x * (0.5 / np.sum(res_over.x))

    # Underweights optimization
    underweights_cov = cov_matrix[4:, 4:]
    initial_weights_under = np.full(6, -0.5 / 6)

    def objective_under(w):
        w = w * (-0.5 / np.sum(w))
        risk_contrib = w * np.dot(underweights_cov, w) / np.dot(w, np.dot(underweights_cov, w))
        return np.var(risk_contrib)

    res_under = minimize(objective_under, initial_weights_under, bounds=[(None, 0)] * 6, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) + 0.5})
    optimized_weights_under = res_under.x * (-0.5 / np.sum(res_under.x))

    # Combine optimized weights
    optimized_active_weights = np.zeros(10)
    optimized_active_weights[:4] = optimized_weights_over
    optimized_active_weights[4:] = optimized_weights_under
    return benchmark_weights + optimized_active_weights

scenario_3_weights = optimize_weights()

# App Initialization
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),  # Track URL changes

    html.H1("Sector Allocation and Risk Optimization", style={"color": "#003399"}),

    # Navigation Links
    html.Div([
        html.A("Scenario 1: 25% Absolute Weight", href="/scenario-1", style={"margin-right": "20px", "color": "#0072CE"}),
        html.A("Scenario 2: 50% Active Weight", href="/scenario-2", style={"margin-right": "20px", "color": "#0072CE"}),
        html.A("Scenario 3: Optimized Weights", href="/scenario-3", style={"color": "#0072CE"}),
    ], style={"margin-bottom": "20px"}),

    # Weights Chart
    html.Div(id="weights-chart-container"),

    # Risk Contribution Chart
    html.Div(id="risk-contribution-chart-container")
])


# Callbacks
@app.callback(
    [Output("weights-chart-container", "children"),
     Output("risk-contribution-chart-container", "children")],
    [Input("url", "pathname")]
)
def update_dashboard(pathname):
    if pathname == "/scenario-1":
        absolute_weights = scenario_1_weights
    elif pathname == "/scenario-2":
        absolute_weights = scenario_2_weights
    elif pathname == "/scenario-3":
        absolute_weights = scenario_3_weights
    else:
        absolute_weights = scenario_1_weights

    active_weights = absolute_weights - benchmark_weights

    # Weights Chart
    weights_fig = go.Figure()
    weights_fig.add_trace(go.Bar(x=sectors, y=active_weights, name="Active Weights", marker_color="#0072CE"))
    weights_fig.add_trace(go.Bar(x=sectors, y=absolute_weights, name="Absolute Weights", marker_color="#82B1FF"))
    weights_fig.add_trace(go.Scatter(x=sectors, y=benchmark_weights, mode="lines", name="Benchmark Weights",
                                      line=dict(dash="dash", color="black")))
    weights_fig.update_layout(title="Active and Absolute Weights", yaxis_title="Weight", barmode="group",
                               template="plotly_white", legend_title="Weight Type")

    # Active Risk Contribution
    active_risk = np.dot(active_weights, np.dot(cov_matrix, active_weights))
    marginal_contrib = active_weights * np.dot(cov_matrix, active_weights)
    risk_contrib = marginal_contrib / active_risk
    sector_contributions_percent = 100 * risk_contrib / np.sum(risk_contrib)

    risk_contribution_fig = go.Figure()
    risk_contribution_fig.add_trace(go.Bar(
        x=sectors,
        y=sector_contributions_percent,
        name="Risk Contribution",
        marker_color="#0072CE",
        text=[f"{val:.2f}%" for val in sector_contributions_percent],
        textposition="auto"
    ))
    risk_contribution_fig.update_layout(title="Active Risk Contribution (%)", yaxis_title="Contribution (%)",
                                        template="plotly_white", legend_title="Contribution Type")

    return dcc.Graph(figure=weights_fig), dcc.Graph(figure=risk_contribution_fig)


if __name__ == "__main__":
    app.run_server(debug=False)
