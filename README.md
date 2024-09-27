# Chinchilla Optimal Training Analysis

This dashboard helps figure out Chinchilla-optimal parameters for your LLM training run, given either a desired model size, or a desired dataset size. 

We compute a bunch of curves for different compute budgets, and find the optimal loss points for each one using the equation from the paper. 

Finally, the user inputs either their model size in parameters, or dataset size in tokens. This input is then matched to the closest possible optimal point on some fixed-compute curve, and those parameters are displayed in a table at the bottom

### Requirements
- plotly
- pandas
- numpy
- streamlit