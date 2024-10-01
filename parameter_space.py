import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
import logging
# import seaborn as sns
from tqdm import tqdm
from r_a_script import initialize_parameters, solve_ode, get_states, calculate_durations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='script_log.log', filemode='w')

# def initialize_parameters():
#     # Define fixed parameters and initial conditions
#     params = {
#         'x_0': 5, 'r_0': 0.5, 'k': 15.0, 'theta': 0.05, 'sigma': 0.25, 
#         'dt': 0.1, 'fin': 60000, 'r0': 0.0, 'a0': 0.0
#     }
#     params['num_steps'] = int(params['fin'] / params['dt'])
#     params['t'] = np.linspace(0, params['fin'], params['num_steps'])
#     print(f"Parameters initialized: {params}")
#     return params

def define_param_sets():
    # Define ranges for I and w
    I_values = np.linspace(1.4, 4, num=60)  # Adjust 'num' for the number of steps in the range
    w_values = np.linspace(3.5, 7.5, num=60)  # Adjust 'num' for the number of steps in the range

    # Create a grid of parameter sets with varying I and w
    param_sets = [{'w': w, 'b': 1.0, 'I': I, 'name': f'ParamSet_w{w}_I{I}'} for w in w_values for I in I_values]
    print(f"Param sets: {param_sets}")
    return param_sets


def get_prob_up_down():
    logging.info("Initializing parameters.")
    params = initialize_parameters()
    # xi = np.zeros(params['num_steps'])
    # noise = update_noise(xi, params['theta'], params['dt'], params['sigma'], params['num_steps'])
    param_sets = define_param_sets()

    logging.info("Starting the calculation of probabilities.")
    # Initialize an empty list to store the results
    results = []

    # Iterate over each parameter set
    for idx, param_set in tqdm(enumerate(param_sets), total=len(param_sets), desc="Processing"):
        logging.info(f"Processing parameter set: {param_set}")
        r_values, a_values = solve_ode(params['r0'], params['a0'], params, param_set)
        states = get_states(r_values, 0.5)

        up_durations, down_durations = calculate_durations(states, params['dt'])

        total_up_duration = sum(up_durations)
        total_down_duration = sum(down_durations)
        up_down_ratio = float(total_up_duration) / total_down_duration if total_down_duration != 0 else np.nan

        # Store the results
        results.append({
            'I': param_set['I'],
            'w': param_set['w'],
            'b': param_set['b'],
            'P(UP):P(DOWN)': round(up_down_ratio, 2)
        })

    logging.info("Calculation completed. Converting results to DataFrame.")
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    # Construct a unique file name based on parameter ranges, dt, and fin
    file_name = f'results_dt{params["dt"]}_fin{params["fin"]}_tau_20.csv'

    # Save the results to a CSV file
    results_df.to_csv(file_name, index=False)

    return results_df

def main():
    # Call the function and get the results DataFrame
    logging.info("Starting the simulation.")
    results_df = get_prob_up_down()
    logging.info("Simulation completed. Results obtained.")
    # print(results_df.describe())

if __name__ == "__main__":
    main()