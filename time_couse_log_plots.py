import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
from r_a_script import initialize_parameters, define_param_sets, solve_ode, get_states, calculate_durations, plot_rate_and_adaptation, update_noise

def get_user_input(param_sets):
    # List available parameter sets
    print("Available dynamical regimes: ")
    for p in param_sets:
        print(p['name'])
    # Get user input
    selected_name = input("Enter the name of the dynamical regime you want to simulate: ")
    return selected_name

def find_param_set(selected_name, param_sets):
    for param_set in param_sets:
        if param_set['name'] == selected_name:
            return param_set
    print("Dynamical regime not found.")
    return None

def plot_rate(ax_rate, t, r_values, a_values, states, param_set):
    # Plot population rate and adaptation
    ax_rate.plot(t, r_values, color='blue', linewidth=0.9)

    # Shade DOWN and UP states
    for i in range(len(t) - 1):
        color = 'lightblue' if states[i] == 'DOWN' else 'pink'
        ax_rate.axvspan(t[i], t[i+1], color=color, alpha=0.3)

    # Set titles and labels
    ax_rate.set_title(param_set['name'], pad=0)
    ax_rate.set_ylabel('Pop. Rate')

def plot_histogram(ax_hist, up_durations, down_durations):
    # Plot histograms for durations on a log scale
    sns.histplot(up_durations, color='red', bins=50, ax=ax_hist, kde=True, alpha=0.35, log_scale=(True, False))
    sns.histplot(down_durations, color='blue', bins=50, ax=ax_hist, kde=True, alpha=0.35, log_scale=(True, False))
    # Set labels
    ax_hist.set_xlabel('Duration (log scale)')
    ax_hist.set_ylabel('Frequency')

def plot_distribution(ax_dist, up_durations, down_durations):
    # Plot the first distribution
    sns.kdeplot(up_durations, kde=True, color='red', ax=ax_dist, bins=100, label='Up Durations',log_scale=(True, False))
    sns.kdeplot(down_durations, kde=True, color='blue', ax=ax_dist, bins=100, label='Down Durations',log_scale=(True, False))
    # Set labels
    ax_dist.set_xlabel('Duration (log scale)')
    ax_dist.set_ylabel('Frequency')

def main():
    params = initialize_parameters()
    param_sets = define_param_sets()
    selected_name = get_user_input(param_sets)
    selected_param_set = find_param_set(selected_name, param_sets)
    xi = np.zeros(params['num_steps'])
    noise = update_noise(xi, params['theta'], params['dt'], params['sigma'], params['num_steps'])
    print(noise)

    if selected_param_set:
        fig, (ax_rate, ax_hist, ax_dist) = plt.subplots(3, 1, figsize=(8, 12), gridspec_kw={'hspace': 0.8})

        r_values, a_values = solve_ode(params['r0'], params['a0'], params, selected_param_set, noise)
        states = get_states(r_values, 0.5)
        up_durations, down_durations = calculate_durations(states, params['dt'])

        plot_rate(ax_rate, params['t'], r_values, a_values, states, selected_param_set)
        plot_histogram(ax_hist, up_durations, down_durations)
        plot_distribution(ax_dist, up_durations, down_durations)

        # Adjust axis limits and layout
        ax_rate.set_xlim(0, 2000)
        ax_rate.set_ylim(0, 1.05)  # For R(t)

        # ax_hist.set_xlim(0, 1000)

        # Layout adjustments
        # plt.tight_layout()
        file_name = f"plots/
        {selected_param_set['name']}_dt{params['dt']}_fin{params['fin']}_pop_rate{params['r0']}_adap{params['a0']}_1.png"
        plt.savefig(file_name)
        # plt.show()

if __name__ == "__main__":
    main()
