import pandas as pd
import numpy as np
import os


def create_master_plot_data(output_dir, num_regions=None, num_plots=None, user_inputs=None, overwrite_data=False):
    """
    Creates master data files: regional dependency matrix, plot parameters, and plot dependency matrix.
    The regional dependency diagonal is set to 1.0 to indicate perfect self-dependency.

    Parameters:
    - output_dir (str): Directory to write the Excel files.
    - num_regions (int, optional): Number of regions. Prompted if None.
    - num_plots (int, optional): Number of plots. Prompted if None.
    - user_inputs (dict, optional): Pre-specified inputs for automated runs.
    - overwrite_data (bool): If False and files exist, skip generation.
    """
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    regional_file = os.path.join(output_dir, "Regional_Dependency_Matrix.xlsx")
    plot_params_file = os.path.join(output_dir, "Plot_Parameters.xlsx")
    plot_dependency_file = os.path.join(output_dir, "Plot_Dependency_Matrix.xlsx")

    # Skip if not overwriting and all files exist
    if not overwrite_data and all(os.path.exists(f) for f in [regional_file, plot_params_file, plot_dependency_file]):
        print(f"All required files already exist in {output_dir}. Skipping data generation.")
        return

    if user_inputs is None:
        user_inputs = {}

    # --- Regional dependency matrix ---
    # Determine number of regions
    if num_regions is None:
        num_regions = int(input("Enter the number of regions: "))

    # Collect climate return period for each region
    region_climate_rp = {}
    for region in range(1, num_regions + 1):
        key = f"climate_rp_{region}"
        prompt = f"Enter climate return period for Region {region} (e.g., 10 or 20): "
        region_climate_rp[region] = float(user_inputs.get(key) or input(prompt))

    # Initialize regional dependency DataFrame with diagonal = 1.0
    region_names = [f"Region {i}" for i in range(1, num_regions + 1)]
    regional_df = pd.DataFrame(
        np.zeros((num_regions, num_regions)),
        index=region_names,
        columns=region_names
    )

    print("\nDefine dependencies between regions (0.0 to 1.0). Diagonal set to 1.0.")
    for i, ri in enumerate(region_names):
        for j, rj in enumerate(region_names):
            if i == j:
                regional_df.iat[i, j] = 1.0
            else:
                key = f"region_dep_{i}_{j}"
                prompt = f"Dependency between {ri} and {rj} (0.0-1.0): "
                val = float(user_inputs.get(key) or input(prompt))
                regional_df.iat[i, j] = val

    regional_df.to_excel(regional_file, index=True)
    print(f"Saved regional dependency matrix to {regional_file}")

    # --- Plot parameters ---
    if num_plots is None:
        num_plots = int(input("\nEnter the number of plots: "))

    plot_names = []
    regions = []
    percent_hit = []
    areas = []
    lost_carbon = []
    cost_per_ha = []
    sequestration = []

    print("\nEnter plot details:")
    for i in range(num_plots):
        # Name
        key = f"plot_name_{i}"
        name = user_inputs.get(key) or input(f"Enter name for plot {i+1}: ")
        # Region index
        key = f"plot_region_{i}"
        region_idx = int(user_inputs.get(key) or input(f"Enter region (1-{num_regions}) for {name}: "))
        # Hit chance
        key = f"plot_hit_{i}"
        hit = float(user_inputs.get(key) or input(f"Enter percentage chance plot hit for {name} (0.0-1.0): "))
        # Area
        key = f"plot_area_{i}"
        area = float(user_inputs.get(key) or input(f"Enter area (ha) for {name}: "))
        # Lost carbon
        key = f"plot_lost_c_{i}"
        lc = float(user_inputs.get(key) or input(f"Enter lost carbon for {name}: "))
        # Cost per hectare
        key = f"plot_cost_{i}"
        cost = float(user_inputs.get(key) or input(f"Enter cost per hectare for {name}: "))
        # Sequestration rate
        key = f"plot_seq_{i}"
        seq = float(user_inputs.get(key) or input(f"Enter sequestration rate for {name}: "))

        plot_names.append(name)
        regions.append(region_idx)
        percent_hit.append(hit)
        areas.append(area)
        lost_carbon.append(lc)
        cost_per_ha.append(cost)
        sequestration.append(seq)

    plot_parameters_df = pd.DataFrame({
        "Plot": plot_names,
        "Region": regions,
        "Climate Return Period": [region_climate_rp[r] for r in regions],
        "Percentage chance plot hit": percent_hit,
        "Area": areas,
        "Lost Carbon": lost_carbon,
        "Cost per Hectare": cost_per_ha,
        "Sequestration Rate": sequestration
    })
    plot_parameters_df.to_excel(plot_params_file, index=False)
    print(f"Saved plot parameters to {plot_params_file}")

    # --- Plot dependency matrix ---
    print("\nDefine dependencies between plots within the same region (0.0 to 1.0).")
    dep_df = pd.DataFrame(
        np.zeros((num_plots, num_plots)),
        index=plot_names,
        columns=plot_names
    )

    for region_idx in range(1, num_regions + 1):
        # filter plots in this region
        idxs = [i for i, r in enumerate(regions) if r == region_idx]
        for i in idxs:
            for j in idxs:
                if i == j:
                    dep_df.iat[i, j] = 1.0
                else:
                    key = f"plot_dep_{i}_{j}"
                    prompt = f"Dependency between {plot_names[i]} and {plot_names[j]} (0.0-1.0): "
                    val = float(user_inputs.get(key) or input(prompt))
                    dep_df.iat[i, j] = val

    dep_df.to_excel(plot_dependency_file, index=True)
    print(f"Saved plot dependency matrix to {plot_dependency_file}")

    print("\nMaster data files have been successfully created!")


def create_case_study_files(output_dir):
    print("\nGenerating case studies from existing master files...")

    plot_parameters_df = pd.read_excel(os.path.join(output_dir, "Plot_Parameters.xlsx"))
    dependency_df = pd.read_excel(os.path.join(output_dir, "Plot_Dependency_Matrix.xlsx"), index_col=0)
    regional_df = pd.read_excel(os.path.join(output_dir, "Regional_Dependency_Matrix.xlsx"), index_col=0)

    plot_names = plot_parameters_df["Plot"].tolist()
    num_case_studies = int(input("\nHow many case studies would you like to generate? "))

    for cs in range(1, num_case_studies + 1):
        print(f"\n--- Case Study {cs} ---")
        case_study_plots = []
        for plot in plot_names:
            response = input(f"Include {plot} in case study {cs}? (y/n): ").strip().lower()
            if response == 'y':
                case_study_plots.append(plot)

        cs_plot_df = plot_parameters_df[plot_parameters_df["Plot"].isin(case_study_plots)]
        cs_dependency_df = dependency_df.loc[case_study_plots, case_study_plots]
        cs_regions = cs_plot_df["Region"].unique()
        cs_region_names = [f"Region {r}" for r in cs_regions]
        cs_regional_df = regional_df.loc[cs_region_names, cs_region_names]

        # CREATE THE FOLDER FIRST
        cs_folder = os.path.join(output_dir, f"Case_Study_{cs}")
        os.makedirs(cs_folder, exist_ok=True)

        # THEN SAVE THE FILES
        cs_plot_file = os.path.join(cs_folder, "Plot_Parameters.xlsx")
        cs_dependency_file = os.path.join(cs_folder, "Plot_Dependency_Matrix.xlsx")
        cs_regional_file = os.path.join(cs_folder, "Regional_Dependency_Matrix.xlsx")

        cs_plot_df.to_excel(cs_plot_file, index=False)
        cs_dependency_df.to_excel(cs_dependency_file)
        cs_regional_df.to_excel(cs_regional_file)

        print(f"Saved case study {cs} plot parameters to {cs_plot_file}")
        print(f"Saved case study {cs} plot dependencies to {cs_dependency_file}")
        print(f"Saved case study {cs} regional dependencies to {cs_regional_file}")
