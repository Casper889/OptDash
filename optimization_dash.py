import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Simulate sector returns (10 sectors, 100 observations)
np.random.seed(42)
returns = np.random.randn(100, 10)

# Create a DataFrame for sector returns
sectors = [f"Sector {i+1}" for i in range(10)]
sector_returns = pd.DataFrame(returns, columns=sectors)

# Calculate the covariance matrix
cov_matrix = sector_returns.cov().values

# Define Sectors
num_sectors = 10
num_overweights = 4
num_underweights = num_sectors - num_overweights
overweight_indices = list(range(num_overweights))
underweight_indices = list(range(num_overweights, num_sectors))

# Benchmark Weights: Random Lognormal Distribution
np.random.seed(42)
raw_weights = np.random.lognormal(mean=0, sigma=1, size=num_sectors)
benchmark_weights = raw_weights / np.sum(raw_weights)

# Scenario Calculations

# Scenario 1: 25% Absolute Weight for Selected Sectors
scenario_1_weights = benchmark_weights.copy()
scenario_1_weights[overweight_indices] = 0.25
scenario_1_weights[underweight_indices] = (1 - np.sum(scenario_1_weights[overweight_indices])) / num_underweights

# Scenario 2: 50% Active Weight for Selected Sectors
scenario_2_active_weights = np.zeros(num_sectors)
scenario_2_active_weights[overweight_indices] = 0.5 / num_overweights
scenario_2_active_weights[underweight_indices] = -0.5 / num_underweights
scenario_2_weights = benchmark_weights + scenario_2_active_weights

# Scenario 3: Optimized Weights
def optimize_weights():
    # Optimize overweights
    overweights_cov = cov_matrix[np.ix_(overweight_indices, overweight_indices)]
    initial_weights_over = np.full(num_overweights, 0.5 / num_overweights)  # Initial guess

    def objective_over(w):
        w = w * (0.5 / np.sum(w))  # Normalize weights
        marginal_contrib = w * np.dot(overweights_cov, w)
        total_risk = np.dot(w, np.dot(overweights_cov, w))
        risk_contrib = marginal_contrib / total_risk
        return np.var(risk_contrib)  # Minimize variance of risk contributions

    constraints_over = {'type': 'eq', 'fun': lambda w: np.sum(w) - 0.5}
    bounds_over = [(0, None) for _ in range(num_overweights)]
    res_over = minimize(objective_over, initial_weights_over, method='SLSQP', bounds=bounds_over, constraints=constraints_over)
    optimized_weights_over = res_over.x * (0.5 / np.sum(res_over.x))

    # Optimize underweights
    underweights_cov = cov_matrix[np.ix_(underweight_indices, underweight_indices)]
    initial_weights_under = np.full(num_underweights, -0.5 / num_underweights)  # Initial guess

    def objective_under(w):
        w = w * (-0.5 / np.sum(w))  # Normalize weights
        marginal_contrib = w * np.dot(underweights_cov, w)
        total_risk = np.dot(w, np.dot(underweights_cov, w))
        risk_contrib = marginal_contrib / total_risk
        return np.var(risk_contrib)  # Minimize variance of risk contributions

    constraints_under = {'type': 'eq', 'fun': lambda w: np.sum(w) + 0.5}
    bounds_under = [(None, 0) for _ in range(num_underweights)]
    res_under = minimize(objective_under, initial_weights_under, method='SLSQP', bounds=bounds_under, constraints=constraints_under)
    optimized_weights_under = res_under.x * (-0.5 / np.sum(res_under.x))

    # Combine weights
    optimized_active_weights = np.zeros(num_sectors)
    optimized_active_weights[overweight_indices] = optimized_weights_over
    optimized_active_weights[underweight_indices] = optimized_weights_under
    return benchmark_weights + optimized_active_weights

scenario_3_weights = optimize_weights()

# App Initialization
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Layout
app.layout = html.Div([
    html.H1("Sector Allocation and Risk Optimization", style={"color": "#003399"}),

    # Tabs for Main Dashboard and Covariance Matrix
    dcc.Tabs(id="main-tabs", value="dashboard", children=[
        dcc.Tab(label="Dashboard", value="dashboard"),
        dcc.Tab(label="Covariance Matrix", value="covariance"),
    ]),
    html.Div(id="main-content")
], style={"font-family": "Arial, sans-serif", "margin": "20px"})


# Callbacks for Main Tabs
@app.callback(
    Output('main-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab(tab):
    if tab == "dashboard":
        return html.Div([
            # Dropdown for Scenario Selection
            html.Div([
                html.H2("Select Scenario", style={"color": "#003399"}),
                dcc.Dropdown(
                    id='scenario-dropdown',
                    options=[
                        {'label': 'Scenario 1: 25% Absolute Weight', 'value': 'scenario_1'},
                        {'label': 'Scenario 2: 50% Active Weight', 'value': 'scenario_2'},
                        {'label': 'Scenario 3: Optimized Weights', 'value': 'scenario_3'}
                    ],
                    value='scenario_1',
                    style={"width": "50%"}
                )
            ]),

            # Weights Chart
            html.Div([
                html.H2("Weights Chart", style={"color": "#003399"}),
                dcc.Graph(id='weights-chart')
            ]),

            # Active Risk Contribution Chart
            html.Div([
                html.H2("Active Risk Contribution", style={"color": "#003399"}),
                dcc.Graph(id='risk-contribution-chart')
            ])
        ])
    elif tab == "covariance":
        # Covariance Matrix Tab
        cov_matrix_fig = px.imshow(cov_matrix, x=sectors, y=sectors, title="Covariance Matrix", 
                                   color_continuous_scale="Viridis")
        return html.Div([
            html.H2("Covariance Matrix", style={"color": "#003399"}),
            dcc.Graph(figure=cov_matrix_fig)
        ])


# Callbacks for Dashboard Charts
@app.callback(
    [Output('weights-chart', 'figure'),
     Output('risk-contribution-chart', 'figure')],
    [Input('scenario-dropdown', 'value')]
)
def update_charts(scenario):
    # Determine weights based on the selected scenario
    if scenario == 'scenario_1':
        absolute_weights = scenario_1_weights
    elif scenario == 'scenario_2':
        absolute_weights = scenario_2_weights
    elif scenario == 'scenario_3':
        absolute_weights = scenario_3_weights
    else:
        absolute_weights = benchmark_weights

    # Calculate active weights
    active_weights = absolute_weights - benchmark_weights

    # Weights Chart
    weights_fig = go.Figure()
    weights_fig.add_trace(go.Bar(x=sectors, y=active_weights, name="Active Weights", marker_color="#0072CE"))
    weights_fig.add_trace(go.Bar(x=sectors, y=absolute_weights, name="Absolute Weights", marker_color="#82B1FF"))
    weights_fig.add_trace(go.Scatter(x=sectors, y=benchmark_weights, mode='lines', name="Benchmark Weights",
                                      line=dict(dash="dash", color="black")))
    weights_fig.update_layout(title="Active and Absolute Weights", yaxis_title="Weight", barmode="group",
                               template="plotly_white", legend_title="Weight Type")

    # Active Risk Contribution
    # Overweights
    active_risk_over = np.dot(active_weights[overweight_indices], 
                              np.dot(cov_matrix[np.ix_(overweight_indices, overweight_indices)], 
                                     active_weights[overweight_indices]))
    marginal_contrib_over = active_weights[overweight_indices] * np.dot(cov_matrix[np.ix_(overweight_indices, overweight_indices)], active_weights[overweight_indices])
    risk_contrib_over = marginal_contrib_over / active_risk_over
    risk_contrib_over *= 0.5  # Scale by 0.5

    # Underweights
    active_risk_under = np.dot(active_weights[underweight_indices], 
                               np.dot(cov_matrix[np.ix_(underweight_indices, underweight_indices)], 
                                      active_weights[underweight_indices]))
    marginal_contrib_under = active_weights[underweight_indices] * np.dot(cov_matrix[np.ix_(underweight_indices, underweight_indices)], active_weights[underweight_indices])
    risk_contrib_under = marginal_contrib_under / active_risk_under
    risk_contrib_under *= 0.5  # Scale by 0.5

    # Combine Contributions
    sector_contributions = np.zeros(num_sectors)
    sector_contributions[overweight_indices] = risk_contrib_over
    sector_contributions[underweight_indices] = risk_contrib_under
    sector_contributions_percent = 100 * sector_contributions / np.sum(sector_contributions)

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

    return weights_fig, risk_contribution_fig


if __name__ == '__main__':
    app.run_server(debug=False)
