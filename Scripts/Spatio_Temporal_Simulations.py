
def Generate_Cyclone_Seed_Data(plot_parameters_path,
                               regional_dependency_path,
                               output_dir,
                               num_simulations,
                               num_years,
                               seed_value=23):
    """
    Generate cyclone (climate) seed data for simulations with inter-region spillover.

    Behavior per year:
      1) Each region r independently has a cyclone with probability 1/λ_r.
      2) Any region that has a cyclone can trigger cyclones in other regions
         via inter-region dependency γ_{r,s} (multi-hop cascades allowed).
      3) For every region that is active (base or spillover), each plot in that region
         is hit independently using its per-plot hit probability.
      4) If a plot is hit, Percent_Area_Lost ~ Uniform(0,1). Otherwise 0.

    Saves results to <output_dir>/Cyclone_Seed_Data.csv with columns:
      Simulation, Year, Plot, Percent_Area_Lost
    """
    import os
    import numpy as np
    import pandas as pd

    print("\n--- Generating Cyclone Seed Data (with inter-region spillover) ---")

    # RNG seed
    np.random.seed(seed_value)

    # --- Load plot parameters ---
    plot_parameters_df = pd.read_excel(plot_parameters_path)
    plot_parameters_df['Region'] = plot_parameters_df['Region'].astype(str)
    plot_parameters_df['Plot'] = plot_parameters_df['Plot'].astype(str)

    # Core lists/dicts
    plots = plot_parameters_df['Plot'].tolist()
    regions = sorted(plot_parameters_df['Region'].unique().tolist())
    region_return_periods = (
        plot_parameters_df.groupby('Region')['Climate Return Period']
        .first().astype(float).to_dict()
    )

    # Per-plot hit probability: prefer 'Impact Probability' if present, else fallback
    if 'Impact Probability' in plot_parameters_df.columns:
        plot_hit_probs = (plot_parameters_df
                          .set_index('Plot')['Impact Probability']
                          .astype(float).to_dict())
    else:
        plot_hit_probs = (plot_parameters_df
                          .set_index('Plot')['Percentage chance plot hit']
                          .astype(float).to_dict())

    # Plots by region
    plots_by_region = {
        r: (plot_parameters_df.loc[plot_parameters_df['Region'] == r, 'Plot']
             .astype(str).tolist())
        for r in regions
    }

    # --- Load and align inter-region dependency (γ_{r,s}) ---
    regional_dep = pd.read_excel(regional_dependency_path, index_col=0)

    # Canonicalize headers so "Region 3", "region 3", "R3" -> "3"
    def _canon_region(x: object) -> str:
        s = str(x).strip()
        sl = s.lower()
        if sl.startswith("region"):
            s = s[6:].strip()  # remove "region"
        elif sl.startswith("r") and s[1:].strip().isdigit():
            s = s[1:].strip()  # remove "r"
        return s  # now should be like "3"

    regional_dep.index = regional_dep.index.map(_canon_region)
    regional_dep.columns = regional_dep.columns.map(_canon_region)

    # Validate we have exactly the regions from plot parameters
    missing_rows = set(regions) - set(regional_dep.index.astype(str))
    missing_cols = set(regions) - set(regional_dep.columns.astype(str))
    if missing_rows or missing_cols:
        raise ValueError(
            "regional_dependency_df must contain all regions present in the data.\n"
            f"Missing rows: {sorted(missing_rows)}; missing cols: {sorted(missing_cols)}"
        )

    # Align and clean
    regional_dep = regional_dep.loc[regions, regions].astype(float)
    np.fill_diagonal(regional_dep.values, 0.0)

    # Base event probability per region
    p_region = {r: 1.0 / float(region_return_periods[r]) for r in regions}

    # Map γ → spillover probability in [0,1)
    def _spill_prob(gamma, k=1.0):
        # negatives give no extra chance; positive γ increases smoothly toward 1
        g = max(0.0, float(gamma))
        return 1.0 - np.exp(-k * g)

    # --- Initialize output array ---
    simulation_results = np.zeros((num_simulations, num_years, len(plots)), dtype=float)

    # --- Simulate cyclone events with multi-hop spillover ---
    for sim in range(num_simulations):
        for year in range(num_years):
            plots_hit_this_year = set()

            # 1) Independent base activations: Bernoulli(1/λ_r)
            active_regions = {r: (np.random.rand() < p_region[r]) for r in regions}

            # 2) Multi-hop spillover until no new regions activate
            newly_active = {r for r in regions if active_regions[r]}
            while newly_active:
                activated_this_round = set()

                # Try to activate each inactive region s from all newly-active sources
                for s in regions:
                    if active_regions[s]:
                        continue

                    # Combine spillover from all newly active sources r into s:
                    # P(spill to s) = 1 - ∏_r (1 - p_spill(r→s))
                    p_no_spill = 1.0
                    for r in newly_active:
                        gamma_rs = regional_dep.loc[r, s]
                        p_no_spill *= (1.0 - _spill_prob(gamma_rs))

                    p_spill = 1.0 - p_no_spill
                    if np.random.rand() < p_spill:
                        activated_this_round.add(s)

                if not activated_this_round:
                    break

                for s in activated_this_round:
                    active_regions[s] = True
                newly_active = activated_this_round

            # 3) Within each active region, hit plots by per-plot probabilities (independent)
            for r in regions:
                if not active_regions[r]:
                    continue
                for plot in plots_by_region[r]:
                    if np.random.rand() < plot_hit_probs[plot]:
                        plots_hit_this_year.add(plot)

            # 4) Severity: U(0,1) if hit, else 0
            for i, plot in enumerate(plots):
                simulation_results[sim, year, i] = np.random.rand() if plot in plots_hit_this_year else 0.0

    # --- Save results ---
    os.makedirs(output_dir, exist_ok=True)
    cyclone_seed_path = os.path.join(output_dir, "Cyclone_Seed_Data.csv")

    sim_list, year_list, plot_list, loss_list = [], [], [], []
    for sim in range(num_simulations):
        for year in range(num_years):
            for i, plot in enumerate(plots):
                sim_list.append(sim + 1)
                year_list.append(year + 1)
                plot_list.append(plot)
                loss_list.append(simulation_results[sim, year, i])

    df_cyclone_seed = pd.DataFrame({
        "Simulation": sim_list,
        "Year": year_list,
        "Plot": plot_list,
        "Percent_Area_Lost": loss_list
    })

    df_cyclone_seed.to_csv(cyclone_seed_path, index=False)
    print(f"Cyclone Seed Data saved to: {cyclone_seed_path}")


def Run_Simulation_Using_Cyclone_Seed(plot_parameters_path, cyclone_seed_path, sdp_outputs_dir,
                                      output_dir, num_years, num_simulations,
                                      reinvestment_threshold,
                                      f_disc, carbon_credit_value, discount_rate,
                                      case_study_numbers):
    """
    Run investment simulations using previously generated cyclone seed data.
    Saves results to output_dir/Results_Case_Study_X.csv
    """
    import os
    import pandas as pd
    import numpy as np
    import math
    import time

    # --- Helpers ---
    def _canon_state(state_str):
        """Canonicalize a state string into a consistent sorted format (numeric by plot number)."""
        if not state_str or str(state_str).strip().lower() == "none":
            return "None"
        parts = [s.strip() for s in str(state_str).split(",") if s.strip()]

        def _pnum(p):
            # accepts "Plot 3" or "3"
            s = p.replace("Plot", "").strip()
            return int(s) if s.isdigit() else 10 ** 9  # non-numeric pushed to the end

        parts = sorted(parts, key=_pnum)
        # standardize the label to "Plot X"
        parts = [f"Plot {_pnum(p)}" for p in parts]
        return ", ".join(parts)

    print("\n--- Running Investment Simulations ---")

    # Load base data
    plot_parameters_df = pd.read_excel(plot_parameters_path)
    cyclone_seed_df = pd.read_csv(cyclone_seed_path)

    os.makedirs(output_dir, exist_ok=True)

    for cs_num in case_study_numbers:
        case_study_folder = f"Case_Study_{cs_num}"
        print(f"\n--- Simulating for {case_study_folder} ---")

        case_start_time = time.time()  # ⏱️ Start case-level timer

        # Convert discount rate from decimal to int (e.g. 0.03 → 3)
        # normalize labels so we don’t get “200.0” vs “200”
        discount_rate_label = int(round(discount_rate * 100))  # 0.03 -> 3
        price_label = int(round(carbon_credit_value))  # 200.0 -> 200

        # Policy input path
        optimal_policy_path = os.path.join(
            sdp_outputs_dir,
            case_study_folder,
            f"Discount_rate_{discount_rate_label}",
            f"Carbon_credit_price_{price_label}",
            "Optimal_Policy_Results.csv"
        )

        if not os.path.exists(optimal_policy_path):
            raise FileNotFoundError(optimal_policy_path)

        # --- CSV LOAD + FIXES ---
        optimal_policy_df = pd.read_csv(optimal_policy_path)

        rename_map = {
            'Years': 'Year', 'years': 'Year', 'year': 'Year',
            'Policy': 'Policy', 'policy': 'Policy',
            'Initial_State': 'State', 'initial_state': 'State', 'state': 'State'
        }
        optimal_policy_df.rename(columns=rename_map, inplace=True)

        optimal_policy_df['Year'] = optimal_policy_df['Year'].astype(int)

        # ✅ Ensure Policy column exists
        if 'Policy' not in optimal_policy_df.columns:
            raise ValueError(
                f"Policy column missing in {optimal_policy_path}, "
                f"found {optimal_policy_df.columns.tolist()}"
            )

        # ✅ Ensure State column exists
        if 'State' not in optimal_policy_df.columns:
            raise ValueError(
                f"State column missing in {optimal_policy_path}. "
                f"Columns present: {optimal_policy_df.columns.tolist()}"
            )

        # ✅ Normalize investment actions
        optimal_policy_df['Plots_Invested_In'] = optimal_policy_df['Policy'].apply(
            lambda x: 'None' if x == 'No Investment' else x.replace('Invest ', '')
        )

        # Canonicalize the policy state once (avoid re-applying per-row in the loop)
        optimal_policy_df['State'] = optimal_policy_df['State'].fillna('None').astype(str)
        optimal_policy_df['State_canon'] = optimal_policy_df['State'].apply(_canon_state)

        # --- Load the case-specific Plot_Parameters.xlsx ---
        plots_root = os.path.dirname(os.path.dirname(os.path.dirname(plot_parameters_path)))
        case_plot_params_path = os.path.join(plots_root, "Data", "Plots_Data", f"Case_Study_{cs_num}", "Plot_Parameters.xlsx")

        if not os.path.exists(case_plot_params_path):
            raise FileNotFoundError(f"Case file not found: {case_plot_params_path}")

        case_plot_params_df = pd.read_excel(case_plot_params_path)

        # Active plots for this case (sorted numerically if labeled 'Plot 1', 'Plot 2', etc.)
        def _plotnum(p):
            s = str(p).replace("Plot", "").strip()
            return int(s) if s.isdigit() else 10 ** 9

        active_plots = sorted(case_plot_params_df['Plot'].tolist(), key=_plotnum)

        # Per-plot dictionaries from the case file
        plot_areas = case_plot_params_df.set_index('Plot')['Area'].to_dict()
        plot_sequestration_rates = case_plot_params_df.set_index('Plot')['Sequestration Rate'].to_dict()
        plot_investment_costs = case_plot_params_df.set_index('Plot')['Cost per Hectare'].to_dict()
        plot_region_map = case_plot_params_df.set_index('Plot')['Region'].to_dict()

        # Optional sanity check: policy shouldn’t reference plots outside this case
        _policy_plots = set()
        for s in optimal_policy_df['Plots_Invested_In'].unique():
            if s != 'None':
                _policy_plots.update([p.strip() for p in s.split(',')])
        _extra = _policy_plots - set(active_plots)
        if _extra:
            raise ValueError(f"Policy references plots not in {case_study_folder}: {_extra}")

        results = []

        for sim in range(1, num_simulations + 1):
            plot_states = {plot: 0 for plot in active_plots}
            cumulative_area_restored_per_plot = {plot: 0 for plot in active_plots}
            current_state = 'None'

            cumulative_carbon_sequestered = 0
            cumulative_npv = 0
            cumulative_discounted_carbon_value = 0
            cumulative_investment_cost = 0
            cumulative_profit = 0
            cumulative_area_restored = 0
            cumulative_area_over_years = 0
            total_num_investments = 0
            cumulative_discounted_investment_cost = 0.0
            restored_plots = set()

            for year in range(1, num_years + 1):

                prev_plot_states = plot_states.copy()

                start_state = current_state

                policy_action = optimal_policy_df[
                    (optimal_policy_df['State_canon'] == _canon_state(current_state)) &
                    (optimal_policy_df['Year'] == year)
                    ]

                investment_decision = 'None'
                plots_to_invest = []

                # Require an exact match; fail loud if not found (prevents silent drift)
                if len(policy_action) != 1:
                    raise RuntimeError(
                        f"No unique policy row for State='{_canon_state(current_state)}', Year={year}. "
                        f"(matches={len(policy_action)}) — check canonicalization and duplicates."
                    )

                invested_plots_str = policy_action.iloc[0]['Plots_Invested_In']
                investment_decision = invested_plots_str
                if invested_plots_str != 'None':
                    plots_to_invest = [p.strip() for p in invested_plots_str.split(',')]

                area_restored = 0
                num_investments = 0
                investment_cost = 0

                # gather per-plot costs
                new_costs = {}
                for plot in plots_to_invest:
                    if plot in plot_states:
                        # restore area
                        restored_area = plot_areas[plot] - plot_states[plot]
                        plot_states[plot] = plot_areas[plot]
                        area_restored += restored_area
                        cumulative_area_restored_per_plot[plot] += restored_area

                        # raw cost
                        cost = math.ceil(restored_area) * plot_investment_costs.get(plot, 0)
                        new_costs[plot] = cost

                        num_investments += 1
                        total_num_investments += 1
                        restored_plots.add(plot)

                # base cost before discount
                investment_cost_base = sum(new_costs.values())

                # count how many new plots per region
                region_counts = {}
                for p in new_costs:
                    r = plot_region_map[p]
                    region_counts[r] = region_counts.get(r, 0) + 1

                # identify any region with ≥2 new plots
                eligible_regs = [r for r, c in region_counts.items() if c >= 2]

                if eligible_regs:
                    # apply flat 10% discount to those clusters only
                    cost_elig = sum(c for p, c in new_costs.items()
                                    if plot_region_map[p] in eligible_regs)
                    cost_inel = investment_cost_base - cost_elig
                    investment_cost = cost_elig * (1 - f_disc) + cost_inel
                else:
                    # no same-region cluster → full price
                    investment_cost = investment_cost_base

                # Apply cyclone damage
                sim_seed = cyclone_seed_df[
                    (cyclone_seed_df['Simulation'] == sim) &
                    (cyclone_seed_df['Year'] == year)
                ]
                for _, row in sim_seed.iterrows():
                    plot = row['Plot']
                    percent_area_lost = row['Percent_Area_Lost']
                    if plot in plot_states:
                        damage = plot_states[plot] * percent_area_lost
                        plot_states[plot] = max(plot_states[plot] - damage, 0)

                # Calculate metrics
                # --- Calculate metrics (aligns with manuscript & SDP utility) ---

                # per-plot investment carbon loss (J_i) dictionary (move this once above the year loop if you like)
                plot_investment_carbon_loss = case_plot_params_df.set_index('Plot')['Lost Carbon'].to_dict()

                # Gross sequestration this year from plots currently held (area after damage)
                gross_carbon_this_year = sum(
                    plot_states[p] * plot_sequestration_rates.get(p, 0.0)
                    for p in active_plots if p in restored_plots
                )

                # Investment carbon loss this year (for newly restored area only)
                investment_carbon_loss_this_year = 0.0
                for p in plots_to_invest:
                    restored_area_p = plot_areas[p] - prev_plot_states[p]
                    if restored_area_p > 0:
                        investment_carbon_loss_this_year += restored_area_p * float(
                            plot_investment_carbon_loss.get(p, 0.0)
                        )

                # Net, credit-eligible carbon (no buyback if negative)
                carbon_sequestered_this_year = max(0.0, gross_carbon_this_year - investment_carbon_loss_this_year)

                # Store cumulative as NET (title unchanged)
                cumulative_carbon_sequestered += carbon_sequestered_this_year

                # Monetary flows use NET carbon
                carbon_revenue_this_year = carbon_sequestered_this_year * carbon_credit_value
                profit_this_year = carbon_revenue_this_year - investment_cost
                cumulative_profit += profit_this_year

                # Discounting
                disc_factor = (1.0 + discount_rate) ** (year - 1)
                cumulative_npv += profit_this_year / disc_factor

                discounted_carbon_value_this_year = carbon_revenue_this_year / disc_factor
                cumulative_discounted_carbon_value += discounted_carbon_value_this_year

                discounted_investment_this_year = investment_cost / disc_factor
                cumulative_discounted_investment_cost += discounted_investment_this_year

                # Profitability Index uses discounted sums
                profitability_index = (
                    cumulative_discounted_carbon_value / cumulative_discounted_investment_cost
                    if cumulative_discounted_investment_cost > 0 else np.nan
                )

                current_area = sum(plot_states.values())
                cumulative_area_over_years += current_area
                cumulative_area_restored = sum(cumulative_area_restored_per_plot.values())

                owned_plots = [plot for plot in active_plots
                               if plot_states[plot] / plot_areas[plot] >= reinvestment_threshold]

                for p in active_plots:
                    if plot_states[p] > prev_plot_states[p] and p not in plots_to_invest:
                        raise AssertionError(f"Area increased for {p} without investment.")

                end_state = _canon_state(", ".join(owned_plots)) if owned_plots else "None"

                result_record = {
                    'Simulation': sim,
                    'Year': year,
                    'Start_State': start_state,
                    'End_State': end_state,
                    'Investment_Decision': investment_decision,
                    'Current_Area': current_area,
                    'Cumulative_Area': cumulative_area_over_years,
                    'Num_Investments': num_investments,
                    'Area_Restored': area_restored,
                    'Cumulative_Area_Restored': cumulative_area_restored,
                    'Carbon_Sequestered': carbon_sequestered_this_year,
                    'Cumulative_Carbon_Sequestered': cumulative_carbon_sequestered,
                    'Profit_This_Year': profit_this_year,
                    'Cumulative_Profit': cumulative_profit,
                    'Cumulative_NPV': cumulative_npv,
                    'Profitability_Index': profitability_index,
                    'Total_Num_Investments': total_num_investments
                }

                for plot in active_plots:
                    result_record[f"Area_{plot}"] = plot_states.get(plot, 0)
                    result_record[f"Cumulative_Area_Restored_{plot}"] = cumulative_area_restored_per_plot.get(plot, 0)

                results.append(result_record)
                current_state = end_state

        # Save path
        results_path = os.path.join(output_dir, f"Results_{case_study_folder}.csv")

        # Save results (always overwrites)
        results_df = pd.DataFrame(results)

        # Organize columns
        columns_order = ['Simulation', 'Year', 'Start_State', 'End_State', 'Investment_Decision',
                         'Current_Area', 'Cumulative_Area']
        columns_order += [f"Area_{plot}" for plot in active_plots]
        columns_order += ['Num_Investments', 'Area_Restored', 'Cumulative_Area_Restored']
        columns_order += [f"Cumulative_Area_Restored_{plot}" for plot in active_plots]
        columns_order += ['Carbon_Sequestered', 'Cumulative_Carbon_Sequestered',
                          'Profit_This_Year', 'Cumulative_Profit', 'Cumulative_NPV',
                          'Profitability_Index', 'Total_Num_Investments']

        results_df = results_df[columns_order]
        results_df.to_csv(results_path, index=False)

        case_end_time = time.time()
        case_duration = case_end_time - case_start_time
        print(f"✅ {case_study_folder} completed in {case_duration:.2f} seconds.")

        print(f"✅ Results saved: {results_path}")
