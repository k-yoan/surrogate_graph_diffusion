import pickle
import numpy as np
import matplotlib.pyplot as plt


def get_mu(y, N_trial):
  return 1/N_trial * np.sum(np.log10(y), axis=1)

def get_sig(y, mu, N_trial):
  return np.sqrt(1/(N_trial - 1) * np.sum((np.log10(y) - np.repeat(mu.reshape(mu.shape[0],1), y.shape[1], axis=1))**2, axis=1)) if N_trial > 1 else np.zeros((mu.shape[0]))

color_index = 0
cmap = plt.get_cmap('tab10')
for dataset in ['time', 'twitter', 'facebook']:
    plt.figure(figsize=(15, 6))
    if dataset == 'time':
        methods_list = ['ls', 'qcbp', 'weighted_qcbp']
        basis_list = [('total-order',8), ('hyperbolic-cross',20)]
        nb_samples = [i for i in range(25, 325, 25)]
    else:
        methods_list = ['ls', 'qcbp']
        
        if dataset == 'facebook':
            nb_samples = [100,300,500,700]
            basis_list = [('total-order', 11)]
            
        else: #twitter
            nb_samples = [5*i for i in range(1,13)]
            basis_list = [('total-order', 4)]

    for method in methods_list:
        for basis in basis_list:
            color = cmap(color_index)
            print(f'Processing {dataset} - {method} - {basis[0]}')
            with open(f'{dataset}_timings_{method}_{basis[1]}_{basis[0]}.pkl', 'rb') as f:
                loaded_dict = pickle.load(f)
        
            avg_std_dict = {}
            for key in loaded_dict[0].keys():
                for subdict in loaded_dict:
                    if key not in avg_std_dict:
                        avg_std_dict[key] = []
                    avg_std_dict[key].append([np.mean(subdict[key]), np.std(subdict[key])])
                    
            
            avg_std_dict_compressed = {}
            avg_std_dict_compressed['offline']= []
            avg_std_dict_compressed['evaluation'] = []
            avg_std_dict_compressed['full'] = []
            for subdict in loaded_dict:
                subdict['offline'] = np.zeros_like(subdict['full'])
                for key in loaded_dict[0].keys():
                    if key not in ['evaluation', 'full', 'training_samples']:
                        subdict['offline'] += subdict[key]
                for key in avg_std_dict_compressed.keys():
                    field = np.array(subdict[key])
                    mu = get_mu(field[None,:], field.shape[0])[0]
                    sigma = get_sig(field[None,:], np.array(mu)[None], field.shape[0])[0]
                    
                    avg_std_dict_compressed[key].append(np.array([mu,sigma]))
            for key in avg_std_dict_compressed.keys():
                avg_std_dict_compressed[key] = np.array(avg_std_dict_compressed[key])
                
            
            x = nb_samples

            avg = avg_std_dict_compressed['offline'][:,0]
            std = avg_std_dict_compressed['offline'][:,1]
            plt.plot(x, 10**avg, color=color, label=f'Offline {dataset} - {method} - {basis[0]}', linewidth=2)
            plt.fill_between(x, 10**(avg - std), 10**(avg + std), color=color, alpha=0.2)

            avg_full = avg_std_dict_compressed['full'][:,0]
            std_full = avg_std_dict_compressed['full'][:,1]

            plt.plot(x, 10**avg_full, color=color, marker='*', label=f'Online - Full order model - {dataset} - {method} - {basis[0]}', linewidth=2)
            plt.fill_between(x, 10**(avg_full - std_full), 10**(avg_full + std_full), color=color, alpha=0.2)

            avg_evaluation = avg_std_dict_compressed['evaluation'][:,0]
            std_evaluation = avg_std_dict_compressed['evaluation'][:,1]

            plt.plot(x, 10**avg_evaluation, color=color, linestyle='--', label=f'Online - Surrogate model - {dataset} - {method} - {basis[0]}', linewidth=2)
            plt.fill_between(x, 10**(avg_evaluation - std_evaluation), 10**(avg_evaluation + std_evaluation), color=color, alpha=0.2)
            color_index += 1

    plt.xlabel(r'$\#$ of samples')
    plt.ylabel('Computational time')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

                # Show or save the plot

    plt.savefig(f'timings_{dataset}.pdf')
                
                