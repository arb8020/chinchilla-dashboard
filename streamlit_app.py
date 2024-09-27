import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
st.set_page_config(layout="wide")

# Core computation functions
def f(N, C):
    D = C / (6 * N)
    return 406.4 / (D**0.32) + 410.7 / (N**0.28) + 1.69

def find_optimal_N(C, N_range):
    losses = [f(N, C) for N in N_range]
    min_index = np.argmin(losses)
    return N_range[min_index], losses[min_index]

def find_optimal_D(C, D_range):
    N_range = [C / (6 * D) for D in D_range]
    losses = [f(N, C) for N in N_range]
    min_index = np.argmin(losses)
    return D_range[min_index], losses[min_index]

def calculate_optimal_points(C_values, range_values, is_N_mode):
    if is_N_mode:
        return [find_optimal_N(C, range_values) for C in C_values]
    else:
        return [find_optimal_D(C, range_values) for C in C_values]

def find_closest_points(fixed_values, optimal_values, C_values, is_N_mode):
    closest_points = []
    for fixed_value in fixed_values:
        closest_index = min(range(len(optimal_values)), key=lambda i: abs(np.log(optimal_values[i]) - np.log(fixed_value)))
        closest_C = C_values[closest_index]
        if is_N_mode:
            closest_loss = f(fixed_value, closest_C)
            closest_D = closest_C / (6 * fixed_value)
            closest_points.append((fixed_value, closest_C, closest_D, closest_loss))
        else:
            closest_N = closest_C / (6 * fixed_value)
            closest_loss = f(closest_N, closest_C)
            closest_points.append((closest_N, closest_C, fixed_value, closest_loss))
    return closest_points

# UI Functions
def sidebar_inputs():
    st.sidebar.header('Mode Selection')
    
    st.sidebar.header('Values to Analyze')
    is_N_mode = st.sidebar.radio("Select input mode:", ("Model Size (N)", "Dataset Size (D)")) == "Model Size (N)"
    if is_N_mode:
        fixed_values_input = st.sidebar.text_input('Enter model sizes (comma-separated, in parameters)', '1.17e8, 3.45e8, 7.74e8, 1.558e9')
    else:
        fixed_values_input = st.sidebar.text_input('Enter dataset sizes (comma-separated, in tokens)', '1.66e7, 4.35e7, 8.76e7, 1.64e8')
    fixed_values = [float(val.strip()) for val in fixed_values_input.split(',')]
    
    st.sidebar.header('Computation Parameters')
    C_min = st.sidebar.number_input('Minimum Compute (FLOPs)', value=1e16, format='%e')
    C_max = st.sidebar.number_input('Maximum Compute (FLOPs)', value=1e22, format='%e')
    num_C_calc = st.sidebar.number_input('Number of Compute Values for Calculation', value=1000, min_value=100, max_value=10000)
    
    if is_N_mode:
        N_min = st.sidebar.number_input('Minimum Model Size (parameters)', value=1e7, format='%e')
        N_max = st.sidebar.number_input('Maximum Model Size (parameters)', value=1e10, format='%e')
        num_range = st.sidebar.number_input('Number of Model Size Points', value=1000, min_value=100, max_value=10000)
    else:
        D_min = st.sidebar.number_input('Minimum Dataset Size (tokens)', value=1e6, format='%e')
        D_max = st.sidebar.number_input('Maximum Dataset Size (tokens)', value=1e9, format='%e')
        num_range = st.sidebar.number_input('Number of Dataset Size Points', value=1000, min_value=100, max_value=10000)
    
    st.sidebar.header('Visualization Parameters')
    num_C_display = st.sidebar.slider('Number of Compute Curves to Display', min_value=1, max_value=50, value=10)
    
    
    
    
    return is_N_mode, C_min, C_max, num_C_calc, N_min if is_N_mode else D_min, N_max if is_N_mode else D_max, num_range, num_C_display, fixed_values

def create_main_plot(range_values, C_values_display, optimal_values, optimal_losses, closest_points, is_N_mode):
    fig = go.Figure()
    
    # Create a color scale for the orange gradient
    orange_scale = pc.sequential.Oranges

    # Display subset of compute curves with orange gradient
    for i, C in enumerate(C_values_display):
        color = pc.sample_colorscale(orange_scale, i/(len(C_values_display)-1))[0]
        if is_N_mode:
            losses = [f(N, C) for N in range_values]
        else:
            losses = [f(C/(6*D), C) for D in range_values]
        fig.add_trace(go.Scatter(x=range_values, y=losses, name=f'C = {C:.2e}', mode='lines', line=dict(color=color)))
    
    # Plot optimal points in blue
    fig.add_trace(go.Scatter(x=optimal_values, y=optimal_losses, mode='markers', marker=dict(color='white', size=1),
                             name='Optimal Points'))
    
    # Plot closest points for fixed values in green
    for point in closest_points:
        fig.add_trace(go.Scatter(x=[point[0] if is_N_mode else point[2]], y=[point[3]], mode='markers', 
                                 marker=dict(color='white', size=15, symbol='star'),
                                 name=f'{"Model" if is_N_mode else "Dataset"} Size: {point[0] if is_N_mode else point[2]:.2e}'))
    
    fig.update_layout(
        title=f'Chinchilla Optimal Training: Loss vs {"Model" if is_N_mode else "Dataset"} Size for Different Compute Budgets',
        xaxis_title=f'{"Model" if is_N_mode else "Dataset"} Size ({("parameters" if is_N_mode else "tokens")})',
        yaxis_title='Loss',
        xaxis_type="log",
        yaxis_type="log",
        legend_title="Compute Budgets",
        hovermode="closest"
    )
    
    return fig

def display_analysis_results(closest_points, is_N_mode):
    st.header('Analysis Results')
    st.subheader(f'Optimal values for specified {"model" if is_N_mode else "dataset"} sizes')
    
    # Create a DataFrame from the closest_points data
    data = []
    for N, C, D, loss in closest_points:
        data.append({
            "Model Size (parameters)": f"{N:.2e}",
            "Compute Budget (FLOPs)": f"{C:.2e}",
            "Dataset Size (tokens)": f"{D:.2e}",
            "Loss": f"{loss:.4f}"
        })
    
    df = pd.DataFrame(data)
    
    # Display the DataFrame as a table
    st.table(df)

# Main application
def main():
    st.title('Chinchilla Optimal Training Analysis')
    st.latex(r"""
    L(N, D) = \frac{406.4}{D^{0.32}} + \frac{410.7}{N^{0.28}} + 1.69
    """)
    
    
    # Get inputs from sidebar
    is_N_mode, C_min, C_max, num_C_calc, range_min, range_max, num_range, num_C_display, fixed_values = sidebar_inputs()
    
    # Calculate ranges
    C_values_calc = np.logspace(np.log10(C_min), np.log10(C_max), num_C_calc)
    range_values = np.logspace(np.log10(range_min), np.log10(range_max), num_range)
    C_values_display = np.logspace(np.log10(C_min), np.log10(C_max), num_C_display)
    
    # Perform calculations
    optimal_values, optimal_losses = zip(*calculate_optimal_points(C_values_calc, range_values, is_N_mode))
    closest_points = find_closest_points(fixed_values, optimal_values, C_values_calc, is_N_mode)
    
    # Create and display plots
    main_plot = create_main_plot(range_values, C_values_display, optimal_values, optimal_losses, closest_points, is_N_mode)
    st.plotly_chart(main_plot, use_container_width=True, height=800)
    
    display_analysis_results(closest_points, is_N_mode)

if __name__ == "__main__":
    main()