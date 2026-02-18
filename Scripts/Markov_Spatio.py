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
    Generate Markov transition matrices for a given case study.

    Parameters:
    - case_study_id (str): Identifier for the case study (e.g., "_case_study_1")
    - plot_parameters_df (DataFrame): Plot characteristics
    - plot_dependency_df (DataFrame): Dependencies between plots
    - regional_dependency_df (DataFrame): Dependencies between regions
    - base_output_dir (str): Directory where output matrices will be saved
    - removal_chance (float): Chance a plot is removed when hit
    """

    # Create output directory for the case study
    output_dir = os.path.join(base_output_dir, f"{case_study_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Ensure regional dependency matrix is writable and diagonal is zero
    regional_dependency_df = regional_dependency_df.astype(float).copy()
    arr = regional_dependency_df.to_numpy(copy=True)  # writable NumPy array
    np.fill_diagonal(arr, 0.0)
    regional_dependency_df.iloc[:, :] = arr

    # Extract data
    plots = plot_parameters_df["Plot"].tolist()
    region_dict = dict(zip(plot_parameters_df["Plot"], plot_parameters_df["Region"]))
    hit_chance_dict = dict(zip(plot_parameters_df["Plot"], plot_parameters_df["Percentage chance plot hit"]))
    climate_risk_dict = plot_parameters_df.groupby("Region")["Climate Return Period"].first().to_dict()

    def powerset(iterable):
        s = list(iterable)
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

    all_states = powerset(plots)
    state_names = [", ".join(state) if state else "None" for state in all_states]
    num_states = len(all_states)

    def compute_inter_region_dependency(regions):
        if len(regions) <= 1:
            return 1.0
        factor = 1.0
        region_list = sorted(regions)
        for i in range(len(region_list)):
            for j in range(i + 1, len(region_list)):
                r1 = f"Region {region_list[i]}"
                r2 = f"Region {region_list[j]}"
                factor *= (1 + regional_dependency_df.loc[r1, r2])
        return factor

    def compute_transition_matrix(investment_choices=None):
        transition_matrix = np.zeros((num_states, num_states))
        investment_set = set(investment_choices) if investment_choices else set()

        for i, current_state_tuple in enumerate(all_states):
            row_probs = np.zeros(num_states)
            current_state = set(current_state_tuple)
            expanded_state = current_state.union(investment_set)

            if not expanded_state:
                row_probs[i] = 1.0
                transition_matrix[i, :] = row_probs
                continue

            valid_next_states = [
                j for j, next_state_tuple in enumerate(all_states)
                if set(next_state_tuple).issubset(expanded_state)
            ]

            for j in valid_next_states:
                next_state = set(all_states[j])
                region_map = {}
                for plot in expanded_state:
                    r = region_dict[plot]
                    region_map.setdefault(r, []).append(plot)

                prob_state = 1.0

                for region_id, plots_in_region in region_map.items():
                    p_cyclone = 1.0 / climate_risk_dict[region_id]
                    p_no_cyclone = 1 - p_cyclone

                    surviving = set(plots_in_region).intersection(next_state)
                    lost = set(plots_in_region) - surviving

                    if not lost:
                        p_survive = np.prod([
                            1 - (hit_chance_dict[plot] * removal_chance)
                            for plot in surviving
                        ])
                        p_region = p_no_cyclone + p_cyclone * p_survive
                    else:
                        p_survive = np.prod([
                            1 - (hit_chance_dict[plot] * removal_chance)
                            for plot in surviving
                        ]) if surviving else 1.0
                        p_lost = np.prod([
                            hit_chance_dict[plot] * removal_chance
                            for plot in lost
                        ])
                        joint_loss_factor = np.prod([
                            1 + plot_dependency_df.loc[i, j]
                            for i in lost for j in lost if i != j
                        ]) if len(lost) > 1 else 1.0
                        p_region = p_cyclone * p_survive * p_lost * joint_loss_factor

                    prob_state *= p_region

                regions = set(region_map.keys())
                inter_region_factor = compute_inter_region_dependency(regions)
                prob_state *= inter_region_factor

                row_probs[j] = prob_state

            row_sum = row_probs.sum()
            if row_sum > 0:
                row_probs /= row_sum
            transition_matrix[i, :] = row_probs

        return transition_matrix

    # Generate transition matrices for all investment combinations
    investment_options = powerset(plots)  # keep order
    for investment_choice in investment_options:
        transition_matrix = compute_transition_matrix(investment_choice)
        investment_name = "_".join(investment_choice) if investment_choice else "Standard"
        file_name = os.path.join(output_dir, f"Markov_Matrix_Invest_{investment_name}.csv")

        df_matrix = pd.DataFrame(transition_matrix, index=state_names, columns=state_names)
        df_matrix.to_csv(file_name)
        print(f"Saved transition matrix for investment in {investment_name}.")
