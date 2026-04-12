import pickle
import numpy as np
import matplotlib.pyplot as plt

def get_mu(y, N_trial):
    return 1/N_trial * np.sum(np.log10(y), axis=1)

def get_sig(y, mu, N_trial):
    if N_trial > 1:
        return np.sqrt(1/(N_trial - 1) * np.sum((np.log10(y) - np.repeat(mu.reshape(mu.shape[0],1), y.shape[1], axis=1))**2, axis=1))
    return np.zeros((mu.shape[0]))

def plot_with_fill(ax, x, data, label, color, linestyle='-', marker=None):
    avg = data[:, 0]
    std = data[:, 1]
    ax.plot(x, 10**avg, color=color, label=label, linewidth=2, linestyle=linestyle, marker=marker)
    ax.fill_between(x, 10**(avg - std), 10**(avg + std), color=color, alpha=0.2)

# Configuration
cmap = plt.get_cmap('tab10')
datasets = ['time', 'twitter', 'facebook']
dataset_names = ['SBM (time-dependent)', 'Twitter', 'Facebook']

# Create a 3x2 grid
fig, axes = plt.subplots(3, 2, figsize=(12, 15))

for idx, dataset in enumerate(datasets):
    ax_time = axes[idx, 0] # Left column: Timing
    ax_kmin = axes[idx, 1] # Right column: k_min - m
    
    color_index = 0
    if dataset == 'time':
        methods_list = ['ls', 'qcbp', 'weighted_qcbp']
        basis_list = [('total-order', 8), ('hyperbolic-cross', 20)]
        nb_samples = [i for i in range(25, 325, 25)]
    else:
        methods_list = ['ls', 'qcbp']
        if dataset == 'facebook':
            nb_samples = [100, 300, 500, 700]
            basis_list = [('total-order', 11)]
        else: # twitter
            nb_samples = [5 * i for i in range(1, 13)]
            basis_list = [('total-order', 4)]

    for basis in basis_list:
        basis_mapping = {'total-order': 'TD', 'hyperbolic-cross': 'HC'}
        basis_name = basis_mapping.get(basis[0], basis[0])
        
        for method in methods_list:
            method_mapping = {'ls': 'LS', 'qcbp': 'QCBP', 'weighted_qcbp': 'wQCBP'}
            method_name = method_mapping.get(method, method)
            color = cmap(color_index)
            
            try:
                file_path = f'{dataset}_timings_{method}_{basis[1]}_{basis[0]}.pkl'
                with open(file_path, 'rb') as f:
                    loaded_dict = pickle.load(f)
            except FileNotFoundError:
                continue

            # Data processing for Timing (Log Scale Math)
            avg_std_log = {'offline': [], 'evaluation': [], 'full': []}
            # Data processing for k_min (Linear Mean Math)
            stats_linear = {'offline': [], 'evaluation': [], 'full': []}

            for subdict in loaded_dict:
                # Calculate offline component
                subdict['offline'] = subdict['coefficients']
                # For Plot 1 (Log space)
                for key in avg_std_log.keys():
                    field = np.array(subdict[key])
                    mu = get_mu(field[None,:], field.shape[0])[0]
                    sigma = get_sig(field[None,:], np.array(mu)[None], field.shape[0])[0]
                    avg_std_log[key].append(np.array([mu, sigma]))

                # For Plot 2 (Linear space for k_min calculation)
                for key in stats_linear.keys():
                    stats_linear[key].append([np.mean(subdict[key]), np.std(subdict[key])])

            # --- Plot 1: Computational Time ---
            for key in avg_std_log:
                avg_std_log[key] = np.array(avg_std_log[key])
            
            plot_with_fill(ax_time, nb_samples, avg_std_log['offline'], 
                           f'Surrogate (coeffs) - {method_name}, {basis_name}', color)
            plot_with_fill(ax_time, nb_samples, avg_std_log['evaluation'], 
                           f'Surrogate (online) - {method_name}, {basis_name}', color, linestyle='--')
            
            # --- Plot 2: k_min - m ---
            for key in stats_linear:
                stats_linear[key] = np.array(stats_linear[key])
            
            k_list = []
            for i, m in enumerate(nb_samples):
                # Using linear means for the k_min formula
                t_full = stats_linear['full'][i, 0]
                t_off = stats_linear['offline'][i, 0]
                t_eval = stats_linear['evaluation'][i, 0]
                val = (m * t_full + t_off) / (t_full - t_eval) - m
                k_list.append(val)
                
            ax_kmin.plot(nb_samples, k_list, color=color, label=f'{method_name}-{basis_name}', linewidth=2)
            
            color_index += 1

    # Final touch for Time Plot (per row)
    plot_with_fill(ax_time, nb_samples, avg_std_log['full'], 'Full order model', color='black', marker='*')
    ax_time.set_yscale('log')
    ax_time.set_ylabel('Comp. time (log)')
    ax_time.set_title(f'{dataset_names[idx]} - Timings')
    if dataset == 'facebook':
        ax_time.legend(fontsize=7, bbox_to_anchor=(1, 0.9))
    else:
        ax_time.legend(fontsize=7, loc= 'best')# loc='upper left')
    ax_time.grid(True, linestyle='--', alpha=0.6)

    # Final touch for k_min Plot (per row)
    ax_kmin.set_ylabel(r'$k_{min}-m$')
    ax_kmin.set_title(f'{dataset_names[idx]} - Break-even')
 
    ax_kmin.legend(fontsize=7, loc ='best')# loc='upper left')
    ax_kmin.grid(True, linestyle='--', alpha=0.6)
    
    # Common X label for bottom row
    if idx == 2:
        ax_time.set_xlabel(r'$\#$ of sample points')
        ax_kmin.set_xlabel(r'$\#$ of sample points')

plt.tight_layout()
plt.savefig('combined_metrics_3x2.pdf')
plt.show()