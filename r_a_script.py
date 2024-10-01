import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm

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

def initialize_parameters():
    # Define fixed parameters and initial conditions
    params = {
        'x_0': 5, 'r_0': 0.5, 'k': 15.0, 'theta': 0.05, 'sigma': 0.25, 
        'dt': 0.1, 'fin': 10000, 'r0': 0.0, 'a0': 0.0, 'tau_a':15
    }
    params['num_steps'] = int(params['fin'] / params['dt'])
    params['t'] = np.linspace(0, params['fin'], params['num_steps'])
    print(f"Parameters initialized: {params}")
    return params

# def define_param_sets():
#     # Define parameter sets for different dynamical regimes
#     return [
#         {'w': 6.0, 'b': 1.0, 'I': 2.5, 'name': 'Oscillatory'},
#         {'w': 6.3, 'b': 1.0, 'I': 2.35, 'name': 'Bistable'},
#         {'w': 6.0, 'b': 1.0, 'I': 2.4, 'name': 'Excitable_DOWN'},
#         {'w': 6.0, 'b': 1.0, 'I': 2.6, 'name': 'Excitable_UP'}
#     ]

def define_param_sets():
    # Define parameter sets for different dynamical regimes
    return [
        {'w': 6.0, 'b': 1.0, 'I': 2.5, 'name': 'Oscillatory'},
        {'w': 6.3, 'b': 1.0, 'I': 2.35, 'name': 'Bistable'},
        {'w': 5.0, 'b': 1.0, 'I': 2.6, 'name': 'Excitable_DOWN'},
        {'w': 5.0, 'b': 1.0, 'I': 3.6, 'name': 'Excitable_UP'}
    ]

# def update_noise(xi, theta, dt, sigma, num_steps):
#     # dW = np.random.normal()
#     # print(theta, sigma, dt, num_steps)
#     for it in range(num_steps-1):
#         xi[it + 1] = (1 - theta* dt) * xi[it] + sigma * np.sqrt(2 * theta * dt) * np.random.randn()
#     # return xi * (1 - theta * dt) + sigma * np.sqrt(2 * theta * dt) * dW
#     print("Variance in OU noise:", np.var(xi))
#     return xi

def update_noise(xi, theta, sigma):
    dW = np.random.normal(0, np.sqrt(0.05))  # Updated time step for the noise
    return xi - theta * xi * 0.05 + sigma * np.sqrt(2 * theta) * dW

# def fhn_ode(r, a, w, b, I, xi, x_0, r_0, k):
#     R_inf = lambda x: 1 / (1 + np.exp(-(x - x_0)))
#     A_inf = lambda r: 1 / (1 + np.exp(-k * (r - r_0)))
#     drdt = -r + R_inf(w*r - b*a + I + xi)
#     dadt = -a + A_inf(r)
#     return drdt, dadt

def fhn_ode(r, a, w, b, I, xi, x_0, r_0, k, tau_a):
    R_inf = lambda x: 1 / (1 + np.exp(-(x - x_0)))
    A_inf = lambda r: 1 / (1 + np.exp(-k * (r - r_0)))
    
    drdt = -r + R_inf(w*r - b*a + I + xi)
    dadt = (-a + A_inf(r))/tau_a
    
    return drdt, dadt

def get_states(values, threshold):
    return ['UP' if v >= threshold else 'DOWN' for v in values]

def calculate_durations(states, dt):
    up_durations, down_durations = [], []
    current_duration = 0
    current_state = states[0]

    for state in states:
        if state == current_state:
            current_duration += dt
        else:
            if current_state == 'UP':
                up_durations.append(current_duration)
            elif current_state == 'DOWN':
                down_durations.append(current_duration)
            current_duration = dt
            current_state = state

    # Add the last duration
    if current_state == 'UP':
        up_durations.append(current_duration)
    elif current_state == 'DOWN':
        down_durations.append(current_duration)

    return up_durations, down_durations

def solve_ode(r0, a0, params, param_set):
    print(f"Solving ODE for parameter set: {param_set}")
    r_values, a_values = [], []
    r, a = r0, a0
    xi = 0.0
    for i in tqdm(range(params['num_steps']), desc=f"Simulating {param_set['name']}"):
        # drdt, dadt = fhn_ode(r, a, param_set['w'], param_set['b'], param_set['I'], noise[i], params['x_0'], params['r_0'], params['k'], params['tau_a'])
        drdt, dadt = fhn_ode(r, a, param_set['w'], param_set['b'], param_set['I'], xi, params['x_0'], params['r_0'], params['k'], params['tau_a'])
        r += drdt * params['dt']
        a += dadt * params['dt']
        xi = update_noise(xi, params['theta'], params['sigma'])
        r_values.append(r)
        a_values.append(a)
    return r_values, a_values



def plot_rate(ax_rate, t, r_values,  states, param_set):
    # Plot population rate and adaptation
    ax_rate.plot(t, r_values, color='black', linewidth=1)

    # # Shade DOWN and UP states
    # for i in range(len(t) - 1):
    #     if states[i] == 'UP':
    #         color = 'pink'
    #         ax_rate.axvspan(t[i], t[i+1], color=color, alpha=0.3)

    # Set titles and labels
    # ax_rate.set_title(param_set['name'], pad=0)
    # ax_rate.set_ylabel('Pop. Rate', fontsize=16)
    ax_rate.set_yticklabels([])

def plot_adaptation(ax_adapt, t, a_values, states, param_set):
    ax_adapt.plot(t, a_values, color='black', linewidth=0.9)
    # Shade DOWN and UP states
    for i in range(len(t) - 1):
        if states[i] == 'UP':
            color = 'pink'
            ax_adapt.axvspan(t[i], t[i+1], color=color, alpha=0.3)
    # ax_adapt.set_title(param_set['name'], pad=0)
    # ax_adapt.set_ylabel('Adaptation', fontsize=16)
    ax_adapt.set_yticklabels([])

def plot_histogram(ax_hist, up_durations, down_durations):
    # Plot histograms for durations
    # sns.histplot(up_durations, color='red', ax=ax_hist, label='UP', stat="density", element="step", edgecolor='darkred', fill=True, alpha=0.35)
    # sns.histplot(down_durations, color='blue', ax=ax_hist, label='DOWN', stat="density", element="step", edgecolor='darkblue', fill=True, alpha=0.35)
    ax_hist.hist(up_durations, bins=40, label='UP', color='red', histtype='bar', edgecolor='darkred', linewidth=1.5, alpha=0.35)
    ax_hist.hist(down_durations, bins=40, label='DOWN', color='blue', histtype='bar', edgecolor='darkblue', linewidth=1.5, alpha=0.35)
    # Set labels
    # ax_hist.set_xlabel('Time (AU)', fontsize=24)
    # ax_hist.set_ylabel('Frequency', fontsize=16)
    ax_hist.set_xticklabels([])

    # Remove y-axis tick labels
    ax_hist.set_yticklabels([])

def main():
    params = initialize_parameters()
    param_sets = define_param_sets()
    selected_name = get_user_input(param_sets)
    selected_param_set = find_param_set(selected_name, param_sets)
    # xi = np.zeros(params['num_steps'])
    # noise = update_noise(xi, params['theta'], params['dt'], params['sigma'], params['num_steps'])
    param_sets = define_param_sets()

    if selected_param_set:
        # Solve ODE and get other data
        r_values, a_values = solve_ode(params['r0'], params['a0'], params, selected_param_set)
        states = get_states(r_values, 0.5)
        up_durations, down_durations = calculate_durations(states, params['dt'])

        # Output some calculations
        print(f"#up_durations:{len(up_durations)},#down_durations:{len(down_durations)}")
        total_up_duration = sum(up_durations)
        total_down_duration = sum(down_durations)
        print(f"Total UP time:{total_up_duration},Total DOWN time:{total_down_duration}\n, P(UP):P(DOWN) = {round(float(total_up_duration/total_down_duration),2)}")

        # Create and plot each figure separately
        # Figure for rate and adaptation plot
    # #   Create and plot rate figure
        fig_rate = plt.figure(figsize=(3, 1))
        ax_rate = fig_rate.add_subplot(111)

        # Create and plot adaptation figure
        # fig_adapt = plt.figure(figsize=(6, 2))
        # ax_adapt = fig_adapt.add_subplot(111)

        # Plot data on separate axes
        plot_rate(ax_rate, params['t'], r_values, states, selected_param_set)
        # plot_adaptation(ax_adapt, params['t'], a_values, states, selected_param_set)

        # Set limits for each500)
        # ax_adapt.set_xlim(0, 1000)
        ax_rate.set_xlim(0,1000)
        ax_rate.set_ylim(0, 1)
        # ax_adapt.set_ylim(0, 1)

        ax_rate.xaxis.set_visible(False)  # Remove x-axis
        ax_rate.spines['top'].set_visible(False)
        ax_rate.spines['right'].set_visible(False)
        ax_rate.spines['bottom'].set_visible(False)

        # ax_adapt.xaxis.set_visible(False)  # Remove x-axis
        # ax_adapt.spines['top'].set_visible(False)
        # ax_adapt.spines['right'].set_visible(False)
        # ax_adapt.spines['bottom'].set_visible(False)

        plt.tight_layout()
        fig_rate.savefig(f"./plots/final/pop_rate_{selected_param_set['name']}.svg")
        print("rate plot saved")
        # fig_adapt.savefig(f"./plots/final/adapt_{selected_param_set['name']}.svg")
        # print("adaptation plot saved")

        # # Figure for histogram plot
        # fig_hist = plt.figure(figsize=(6, 2))
        # ax_hist = fig_hist.add_subplot(111)
        # plot_histogram(ax_hist, up_durations, down_durations)
        # ax_hist.set_xlim(0, 1000)     

        # ax_hist.spines['top'].set_visible(False)
        # ax_hist.spines['right'].set_visible(False)
        
        # fig_hist.savefig(f"./plots/final/hist_{selected_param_set['name']}.svg")
        print("all plots saved")
        # file_name = f"dt{params['dt']}_fin{params['fin']}_pop_rate{params['r0']}_adap{params['a0']}_1.png"
        # plt.savefig(file_name)
        # plt.show()
        # plt.close(fig_adapt)
        plt.close(fig_rate)
        # plt.close(fig_hist)
        # fig, (ax_rate, ax_adapt, ax_hist) = plt.subplots(3, 1, figsize=(6, 6), gridspec_kw={'hspace': 0, 'wspace': 0})

        # # Plot data on separate axes
        # plot_rate(ax_rate, params['t'], r_values, states, selected_param_set)
        # plot_adaptation(ax_adapt, params['t'], a_values, states, selected_param_set)
        # plot_histogram(ax_hist, up_durations, down_durations)

        # # Set limits and customize each axis
        # for ax in [ax_rate, ax_adapt, ax_hist]:
        #     ax.set_xlim(0, 700)
        #     if ax != ax_hist:
        #         ax.set_ylim(0, 1)
        #     ax.xaxis.set_visible(False)  # Remove x-axis
        #     ax.spines['top'].set_visible(False)
        #     ax.spines['right'].set_visible(False)
        #     ax.spines['bottom'].set_visible(False)
        #     ax.set_yticklabels([])  # Optionally remove y-tick labels

        # # Optionally, you can set x-axis visibility to True for the last subplot
        # ax_hist.xaxis.set_visible(True)
        # ax_hist.spines['bottom'].set_visible(True)

        # # Save the figure
        # file_name = f"./plots/final/combined_{selected_param_set['name']}.png"
        # plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        # print("Combined plot saved")

        # # Show the plot
        # # plt.show()
        # plt.close(fig)


if __name__ == "__main__":
    main()
