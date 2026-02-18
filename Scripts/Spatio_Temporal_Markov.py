import pandas as pd
import numpy as np
import os
from itertools import chain, combinations

def Generate_Markov_Matrices(
    case_study_id,
    plot_parameters_df,
    plot_dependency_df,
    regional_dependency_df,
    base_output_dir,
    removal_chance
):
    """

    :param case_study_id:
    :param plot_parameters_df:
    :param plot_dependency_df:
    :param regional_dependency_df:
    :param base_output_dir:
    :param removal_chance:
    :return:
    """

    # Define base directory for output
    base_dir = os.path.join(base_output_dir, f"{case_study_id}")
    os.makedirs(base_dir, exist_ok=True)  # Ensure the directory exists

    # --- Confirm that the Parameter Dataframes will run ---

    ## Check if DataFrames are empty
    if plot_parameters_df.empty or plot_dependency_df.empty:
        raise ValueError("Error: One or both input DataFrames are empty.")

    ## Extract plots
    plots_from_parameters = set(plot_parameters_df["Plot"])
    plots_from_dependency = set(plot_dependency_df.index)  # Assumes index represents plots

    ## Check for consistent number of plots
    if len(plots_from_parameters) != len(plots_from_dependency):
        raise ValueError("Error: Number of plots in plot_parameters_df and plot_dependency_df are different.\n"
                         "Ensure plot_parameters_df and plot_dependency_df have the same number of plots.")

    ## Check for unique plot names
    if len(plots_from_parameters) != len(plot_parameters_df["Plot"]):
        raise ValueError("Error: Duplicate plot names have been found in plot_parameters_df.\n"
                         "Ensure all plots have different names")

    ## Check if dependency matrix is square
    if plot_dependency_df.shape[0] != plot_dependency_df.shape[1]:
        raise ValueError("Error: plot_dependency_df must be a square matrix (rows and columns must match).")

    ## Check if dependency matrix has the same plot names as columns and index
    if set(plot_dependency_df.columns) != plots_from_dependency:
        raise ValueError("Error: plot_dependency_df must have the same plot names in both rows and columns.")

    # --- MATRIX GENERATION ---

    # Canonical ordering so rows/cols align deterministically (numeric sort: Plot 2 < Plot 10)
    import re
    def _plotnum(p: str) -> int:
        m = re.search(r'(\d+)', str(p))
        return int(m.group(1)) if m else 10 ** 9

    ordered_plots = sorted(plot_parameters_df["Plot"].astype(str).tolist(), key=_plotnum)
    plot_parameters_df = plot_parameters_df.set_index("Plot").loc[ordered_plots].reset_index()
    plot_dependency_df = plot_dependency_df.loc[ordered_plots, ordered_plots]

    # ---- Inter-region dependency (γ_{r,s}) prep ----
    import re

    def _canon_region_label(x: object) -> str:
        s = str(x).strip()
        # already "Region k" → standardise spacing/case
        if s.lower().startswith("region"):
            m = re.search(r'(\d+)', s)
            return f"Region {m.group(1)}" if m else s.title()
        # plain number → "Region k"
        if s.isdigit():
            return f"Region {int(s)}"
        return s

    # Canonicalise regions in the plot parameters
    plot_parameters_df["Region"] = plot_parameters_df["Region"].apply(_canon_region_label)
    regions = sorted(plot_parameters_df["Region"].unique().tolist())

    # Load & align the regional dependency matrix to the same canonical labels
    regional_dependency_df = regional_dependency_df.copy()
    regional_dependency_df.index = regional_dependency_df.index.map(_canon_region_label)
    regional_dependency_df.columns = regional_dependency_df.columns.map(_canon_region_label)

    missing_rows = set(regions) - set(regional_dependency_df.index)
    missing_cols = set(regions) - set(regional_dependency_df.columns)
    if missing_rows or missing_cols:
        raise ValueError(
            "regional_dependency_df must contain all regions.\n"
            f"Missing rows: {sorted(missing_rows)}, missing cols: {sorted(missing_cols)}"
        )

    regional_dependency_df = regional_dependency_df.loc[regions, regions]

    # Must be symmetric; set diagonal to 0 safely
    a = regional_dependency_df.to_numpy(dtype=float, copy=True)  # writable
    if not np.allclose(a, a.T, atol=1e-12):
        raise ValueError("regional_dependency_df must be symmetric.")
    np.fill_diagonal(a, 0.0)
    regional_dependency_df.iloc[:, :] = a

    # Extract plots and required mappings per the write-up
    plots = ordered_plots
    region_map = dict(zip(plot_parameters_df["Plot"], plot_parameters_df["Region"]))
    lambda_r = plot_parameters_df.groupby("Region")["Climate Return Period"].first().to_dict()
    if "Impact Probability" in plot_parameters_df.columns:
        _phi_col = "Impact Probability"
    elif "Percentage chance plot hit" in plot_parameters_df.columns:
        _phi_col = "Percentage chance plot hit"
    else:
        raise ValueError("Expected either 'Impact Probability' or 'Percentage chance plot hit' in plot_parameters_df.")

    _phi_raw = plot_parameters_df[_phi_col].astype(float).to_numpy(copy=True)
    # If any value looks like a percentage (>1), scale the whole column to [0,1]
    if np.nanmax(_phi_raw) > 1.0:
        _phi_raw = _phi_raw / 100.0

    phi = dict(zip(plot_parameters_df["Plot"], _phi_raw))

    # Reinvestment threshold fraction η (from caller)
    eta = float(removal_chance)

    # Per-plot climate event probability from its region
    p_event = {p: 1.0 / float(lambda_r[region_map[p]]) for p in plots}

    def per_plot_probs(p):
        pe = p_event[p]
        p_leave = pe * (phi[p] * eta)
        p_stay = (1.0 - pe) + pe * (1.0 - phi[p] * eta)
        return p_stay, p_leave

    # ----- State space (powerset of plots) -----
    def powerset(iterable):
        s = list(iterable)
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

    all_states = powerset(plots)
    state_names = [", ".join(state) if state else "None" for state in all_states]
    num_states = len(all_states)

    # Function to compute transition matrices
    def compute_transition_matrix(investment_choices=None):
        """Computes the transition matrix for given investment choices."""
        transition_matrix = np.zeros((num_states, num_states), dtype=float)
        investment_choices = set(investment_choices) if investment_choices else set()

        for i, current_state in enumerate(all_states):
            row_probs = np.zeros(num_states, dtype=float)
            expanded_state = set(current_state).union(investment_choices)

            if not expanded_state:
                row_probs[i] = 1.0  # Ensuring P(None → None) = 1
                transition_matrix[i, :] = row_probs
                continue

            # Compute individual stay/leave probabilities for each plot
            stay_probs = {}
            leave_probs = {}

            for p in expanded_state:
                stay_probs[p], leave_probs[p] = per_plot_probs(p)

            # Identify valid next states (no spontaneous gains)
            valid_next_states = [j for j, next_state in enumerate(all_states)
                                 if set(next_state).issubset(expanded_state)]

            # Assign Transition Probabilities
            for j in valid_next_states:
                S = set(all_states[j])  # survivors
                L = expanded_state - S  # lost

                # Independent baseline: product of per-plot stay/leave
                base = (
                        np.prod([stay_probs[p] for p in S]) *
                        np.prod([leave_probs[p] for p in L])
                ) if (S or L) else 1.0

                # Intra-region joint loss multiplier: unique pairs within the SAME region only (i<j)
                if len(L) >= 2:
                    L_sorted = sorted(L)
                    pairs = [(a, b) for idx, a in enumerate(L_sorted) for b in L_sorted[idx + 1:]
                             if region_map[a] == region_map[b]]
                    joint = np.prod([1.0 + float(plot_dependency_df.loc[a, b]) for a, b in pairs]) if pairs else 1.0
                else:
                    joint = 1.0

                # Inter-region multiplier: if losses span multiple regions, apply product over region pairs
                if len(L) >= 2:
                    loss_regions = sorted({region_map[p] for p in L})
                    if len(loss_regions) >= 2:
                        inter = 1.0
                        for idx, r1 in enumerate(loss_regions):
                            for r2 in loss_regions[idx + 1:]:
                                inter *= (1.0 + float(regional_dependency_df.loc[r1, r2]))
                    else:
                        inter = 1.0
                else:
                    inter = 1.0

                row_probs[j] = max(0.0, float(base * joint * inter))

            # Normalize row safely
            row_sum = row_probs.sum()
            if row_sum <= 0.0 or not np.isfinite(row_sum):
                # Fallback to independent baseline if dependencies zero/NaN the row
                row_probs[:] = 0.0
                for j in valid_next_states:
                    S = set(all_states[j]);
                    L = expanded_state - S
                    row_probs[j] = (
                            np.prod([stay_probs[p] for p in S]) *
                            np.prod([leave_probs[p] for p in L])
                    ) if (S or L) else 1.0
                row_sum = row_probs.sum()

            row_probs /= row_sum
            transition_matrix[i, :] = row_probs

        return transition_matrix

    # Generate all investment options (powerset of plots)
    investment_options = [set(combo) for combo in powerset(plots)]

    # Compute and save transition matrices for each investment scenario
    for investment_choice in investment_options:
        transition_matrix = compute_transition_matrix(investment_choice)

        def _plotnum(p):
            s = str(p).replace("Plot", "").strip()
            return int(s) if s.isdigit() else 10 ** 9

        investment_name = "_".join(sorted(investment_choice, key=_plotnum)) if investment_choice else "Standard"

        file_name = os.path.join(base_dir, f"Markov_Matrix_Invest_{investment_name}.csv")
        pd.DataFrame(transition_matrix, index=state_names, columns=state_names).to_csv(file_name)
        print(f"Saved transition matrix for investment in {investment_name}.")

