# Updated: 05/2024
# Jordan Holdorf
# Master script to generate Markov matrices and run SDP analysis for multiple case studies

import os
import sys
import pandas as pd

# Import required modules
from SDP_Run_Spatio_temporal import run_sdp_analysis
from Spatio_Temporal_Markov import Generate_Markov_Matrices
from Define_Parameters import create_master_plot_data, create_case_study_files
from Spatio_Temporal_Simulations import Generate_Cyclone_Seed_Data, Run_Simulation_Using_Cyclone_Seed


# === Directory Setup ===
current_dir = os.path.dirname(os.path.abspath(__file__))  # Chapter_5
scripts_dir = os.path.dirname(current_dir)                # Scripts
code_root = scripts_dir                  # Code
sys.path.append(code_root)  # Add Code to system path

# === Parameters ===
case_study_numbers = [1, 2, 3, 4]  # List of case studies to process
create_master_data = False   # Whether to create master data files
create_case_study_data = False  # Whether to create case study files
run_SDP_Case_Study_Analysis = True # To re-run / create SDP outputs
run_simulations = True  # Set to False to skip running the investment simulations


# Economic parameters
discount_rates = [3, 4, 5]   # Discount rates in percent
carbon_credit_prices = [66, 200, 1000]  # Carbon prices in USD

# Restoration loss threshold (η)
reinvestment_threshold = 0.4

# === File and Folder Paths ===
Plots_Data_dir = os.path.join(code_root, "Data", "Plots_Data")
os.makedirs(Plots_Data_dir, exist_ok=True)

# Master file paths
master_files = [
    os.path.join(Plots_Data_dir, "Plot_Parameters.xlsx"),
    os.path.join(Plots_Data_dir, "Plot_Dependency_Matrix.xlsx"),
    os.path.join(Plots_Data_dir, "Regional_Dependency_Matrix.xlsx")
]
regional_dependency_path = os.path.join(Plots_Data_dir, "Regional_Dependency_Matrix.xlsx")
# Create master files if requested or missing
if create_master_data or not all(os.path.exists(f) for f in master_files):
    create_master_plot_data(output_dir=Plots_Data_dir, overwrite_data=create_master_data)
else:
    print("\nMaster files already exist and creation not requested. Skipping master file generation.")

# Create case study files if requested
if create_case_study_data:
    create_case_study_files(output_dir=Plots_Data_dir)

# === Validate Case Study Files ===
if not create_case_study_data:
    for cs_num in case_study_numbers:
        suffix = f"Case_Study_{cs_num}"
        files_to_check = [
            os.path.join(Plots_Data_dir, suffix, f"Plot_Parameters.xlsx"),
            os.path.join(Plots_Data_dir, suffix, f"Plot_Dependency_Matrix.xlsx"),
            os.path.join(Plots_Data_dir, suffix, f"Regional_Dependency_Matrix.xlsx")
        ]
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing case study file: {file_path}. Set create_case_study_data=True or check files.")


if run_SDP_Case_Study_Analysis:
    # === Process Each Case Study ===
    for cs_num in case_study_numbers:
        suffix = f"Case_Study_{cs_num}"
        label = f"Case Study {cs_num}"
        print(f"\n--- Processing {label} ---")

        # Define input paths
        plot_params_path = os.path.join(Plots_Data_dir, suffix, f"Plot_Parameters.xlsx")
        plot_dep_path = os.path.join(Plots_Data_dir, suffix, f"Plot_Dependency_Matrix.xlsx")
        regional_dep_path = os.path.join(Plots_Data_dir, suffix, f"Regional_Dependency_Matrix.xlsx")

        # Load data
        print(f"\nLoading data for {label}...")
        plot_parameters_df = pd.read_excel(plot_params_path)
        plot_dependency_df = pd.read_excel(plot_dep_path, index_col=0)
        regional_dependency_df = pd.read_excel(regional_dep_path, index_col=0)

        # Output directory for Markov matrices
        markov_output_dir = os.path.join(code_root, "Data", "Markov_Matrix")
        os.makedirs(markov_output_dir, exist_ok=True)

        print(f"\nGenerating Markov matrices for {label}...")
        Generate_Markov_Matrices(
            case_study_id=suffix,
            plot_parameters_df=plot_parameters_df,
            plot_dependency_df=plot_dependency_df,
            regional_dependency_df=regional_dependency_df,
            base_output_dir=markov_output_dir,
            removal_chance=reinvestment_threshold
        )

        print("Markov Matrices Generated.")

        # Run SDP analysis
        print(f"\nRunning SDP analysis for {label}...")
        run_sdp_analysis(
            case_study_id=suffix,
            project_root=code_root,
            carbon_credit_prices=carbon_credit_prices,
            discount_rates=discount_rates,
            delta_intra=0.10,
            Nmax=25
        )
        print(f"SDP Analysis Completed for {label}.")

    print("\nAll Case Studies Processed Successfully!")

# Simulation parameters
num_years = 25
num_simulations = 10

cyclone_seed_random_seed = 23  # Random seed for reproducibility

# === 6. Spatio-Temporal Investment Simulation ===
if run_simulations:
    print("\n--- Starting Spatio-Temporal Simulation ---")

    # Define paths
    Plots_Data_dir = os.path.join(code_root, "Data", "Plots_Data")
    Simulation_Data_dir = os.path.join(code_root, "Data", "Simulation_Data")
    SDP_Outputs_dir = os.path.join(code_root, "Data", "SDP_Outputs")

    plot_parameters_path = os.path.join(Plots_Data_dir, "Plot_Parameters.xlsx")
    cyclone_seed_output_dir = Simulation_Data_dir
    cyclone_seed_output_path = os.path.join(Simulation_Data_dir, "Cyclone_Seed_Data.csv")
    simulation_results_output_dir = Simulation_Data_dir

    # Step 1: Generate Cyclone Seed Data
    Generate_Cyclone_Seed_Data(
        plot_parameters_path=plot_parameters_path,
        output_dir=cyclone_seed_output_dir,
        regional_dependency_path=regional_dependency_path,
        num_simulations=num_simulations,
        num_years=num_years,
        seed_value=cyclone_seed_random_seed
    )

    for discount_rate in discount_rates:
        for carbon_credit_value in carbon_credit_prices:
            print(
                f"\n--- Running simulations for Discount Rate {discount_rate}%, Carbon Credit ${carbon_credit_value} ---")

            scenario_output_dir = os.path.join(
                simulation_results_output_dir,
                f"Discount_Rate_{discount_rate}",
                f"Carbon_Credit_Price_{carbon_credit_value}"
            )
            os.makedirs(scenario_output_dir, exist_ok=True)

            Run_Simulation_Using_Cyclone_Seed(
                plot_parameters_path=plot_parameters_path,
                cyclone_seed_path=cyclone_seed_output_path,
                sdp_outputs_dir=SDP_Outputs_dir,
                output_dir=scenario_output_dir,
                num_years=num_years,
                num_simulations=num_simulations,
                f_disc = 0.10,
                reinvestment_threshold=reinvestment_threshold,
                carbon_credit_value=carbon_credit_value,
                discount_rate=discount_rate / 100.0,
                case_study_numbers=case_study_numbers
            )

    print("\nSpatio-Temporal Simulation Completed Successfully!")
