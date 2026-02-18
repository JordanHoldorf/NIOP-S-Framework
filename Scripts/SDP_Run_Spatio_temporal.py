import os
import numpy as np
import pandas as pd
from itertools import chain, combinations
import mdptoolbox

def run_sdp_analysis(case_study_id, project_root, carbon_credit_prices, discount_rates,
                     delta_intra, Nmax=25):
    """
    Runs SDP analysis for a given case study across multiple discount rates and carbon credit prices.

    Parameters:
        case_study_id (str): Identifier for the case study (e.g., '_case_study_1').
        project_root (str): Root path of the project.
        carbon_credit_prices (list of float): List of carbon credit prices.
        discount_rates (list of float): List of discount rates in percent.
        Nmax (int): Planning horizon in years.
    """
    plots_dir = os.path.join(project_root, "Data", "Plots_Data")
    markov_dir = os.path.join(project_root, "Data", "Markov_Matrix", f"{case_study_id}")
    output_dir = os.path.join(project_root, "Data", "SDP_Outputs", f"{case_study_id}")

    plot_parameters_df = pd.read_excel(os.path.join(plots_dir, case_study_id, "Plot_Parameters.xlsx"))

    def powerset(iterable):
        s = list(iterable)
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

    all_states = powerset(plot_parameters_df["Plot"])
    decision_names = ["No Investment" if not state else f"Invest {', '.join(state)}" for state in all_states]
    state_names = [", ".join(state) if state else "None" for state in all_states]
    num_states = len(all_states)
    num_decisions = num_states

    for discount_rate in discount_rates:
        discount_factor = 1 / (1 + (discount_rate * 0.01))

        for carbon_credit_price in carbon_credit_prices:
            sdp_path = os.path.join(output_dir, f"Discount_rate_{discount_rate}", f"Carbon_credit_price_{carbon_credit_price}")
            os.makedirs(sdp_path, exist_ok=True)

            print(f"\nRunning SDP for Case Study {case_study_id} | Discount: {discount_rate}% | Price: ${carbon_credit_price}")

            reward_matrix_sdp = np.zeros((num_decisions, num_states, num_states))
            for a in range(num_decisions):
                invested_plots = set(all_states[a])

                for b in range(num_states):
                    current_set = set(all_states[b])
                    for c in range(num_states):
                        next_set = set(all_states[c])

                        # Feasibility: cannot gain plots that weren't already present or invested this period
                        if not next_set.issubset(current_set.union(invested_plots)):
                            reward_matrix_sdp[a, b, c] = -1e6
                            continue

                        # --- Net carbon in tonnes first ---
                        sequestration_tonnes = np.sum(
                            plot_parameters_df.loc[plot_parameters_df["Plot"].isin(next_set), "Area"].values *
                            plot_parameters_df.loc[
                                plot_parameters_df["Plot"].isin(next_set), "Sequestration Rate"].values
                        )

                        lost_carbon_tonnes = np.sum(
                            plot_parameters_df.loc[
                                plot_parameters_df["Plot"].isin(invested_plots), "Lost Carbon"].values *
                            plot_parameters_df.loc[plot_parameters_df["Plot"].isin(invested_plots), "Area"].values
                        )

                        net_carbon_tonnes = sequestration_tonnes - lost_carbon_tonnes

                        # Apply floor: if net carbon is negative, credits are zero (no buyback)
                        net_carbon_value = max(0.0, net_carbon_tonnes) * carbon_credit_price

                        # --- per-region cluster rebate on investment cost (this period's invested plots only) ---
                        df_inv = plot_parameters_df.loc[
                            plot_parameters_df["Plot"].isin(invested_plots)
                        ].copy().reset_index(drop=True)
                        df_inv["PlotCost"] = df_inv["Cost per Hectare"] * df_inv["Area"]

                        C_total = df_inv["PlotCost"].sum()
                        region_counts = df_inv["Region"].value_counts()
                        eligible_regs = region_counts[region_counts >= 2].index.tolist()
                        cost_elig = df_inv.loc[df_inv["Region"].isin(eligible_regs), "PlotCost"].sum()
                        cost_inel = C_total - cost_elig
                        f_disc = delta_intra if eligible_regs else 0.0
                        investment_cost = cost_elig * (1 - f_disc) + cost_inel
                        # ---------------------------------------------------------------------

                        reward = net_carbon_value - investment_cost
                        reward_matrix_sdp[a, b, c] = reward

            np.save(os.path.join(sdp_path, "Reward_Matrix.npy"), reward_matrix_sdp)
            pd.DataFrame(reward_matrix_sdp.reshape(num_decisions, -1)).to_csv(os.path.join(sdp_path, "Reward_Matrix.csv"), index=False)

            markov_files = [
                "Markov_Matrix_Invest_Standard.csv" if not state else f"Markov_Matrix_Invest_{'_'.join(state)}.csv"
                for state in all_states
            ]

            standard_file = os.path.join(markov_dir, "Markov_Matrix_Invest_Standard.csv")
            if os.path.exists(standard_file):
                standard_matrix = pd.read_csv(standard_file).drop(columns=["Unnamed: 0"], errors="ignore")
                no_investment_3d = np.expand_dims(standard_matrix.to_numpy(), axis=0)
            else:
                print(f"Warning: Standard matrix not found: {standard_file}")
                no_investment_3d = None

            invest_arrays = []
            for file_name in markov_files[1:]:
                path = os.path.join(markov_dir, file_name)
                if os.path.exists(path):
                    df = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")
                    invest_arrays.append(np.expand_dims(df.to_numpy(), axis=0))
                else:
                    print(f"Warning: Missing Markov file: {path}")

            invest_3d = np.concatenate(invest_arrays, axis=0) if invest_arrays else None

            if no_investment_3d is not None and invest_3d is not None:
                transitions = np.concatenate((no_investment_3d, invest_3d), axis=0)
            elif no_investment_3d is not None:
                transitions = no_investment_3d
            elif invest_3d is not None:
                transitions = invest_3d
            else:
                raise ValueError("No transition matrices found.")

            np.save(os.path.join(sdp_path, "Transition_Matrix.npy"), transitions)
            pd.DataFrame(transitions.reshape(num_decisions, -1)).to_csv(os.path.join(sdp_path, "Transition_Matrix.csv"), index=False)

            mdptoolbox.util.check(P=transitions, R=reward_matrix_sdp)
            sdp = mdptoolbox.mdp.FiniteHorizon(transitions, reward_matrix_sdp, discount_factor, Nmax)
            sdp.run()

            value_df = pd.DataFrame(sdp.V)
            value_df.insert(0, "State", state_names)
            value_df = pd.melt(value_df, id_vars='State', var_name="Years", value_name="Value")

            policy_df = pd.DataFrame(sdp.policy)
            policy_df.insert(0, "State", state_names)
            policy_df = pd.melt(policy_df, id_vars='State', var_name="Years", value_name="Policy")
            policy_df = pd.merge(value_df, policy_df)
            policy_df['Policy'] = policy_df['Policy'].astype(int).map(lambda x: decision_names[x])
            policy_df['Years'] = policy_df['Years'].astype(int) + 1
            policy_df.to_csv(os.path.join(sdp_path, "Optimal_Policy_Results.csv"), index=False)