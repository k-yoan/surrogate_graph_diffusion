import pickle
import numpy as np
import matplotlib.pyplot as plt


def get_mu(y, N_trial):
  return 1/N_trial * np.sum(np.log10(y), axis=1)

def get_sig(y, mu, N_trial):
  return np.sqrt(1/(N_trial - 1) * np.sum((np.log10(y) - np.repeat(mu.reshape(mu.shape[0],1), y.shape[1], axis=1))**2, axis=1)) if N_trial > 1 else np.zeros((mu.shape[0]))

cmap = plt.get_cmap('tab10')
for dataset in ['time', 'twitter', 'facebook']:
    color_index = 0 
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
                            
            avg_std_dict_compressed2 = {}
            avg_std_dict_compressed2['coefficients'] = []
            avg_std_dict_compressed2['evaluation'] = []
            avg_std_dict_compressed2['full'] = []
            for subdict in loaded_dict:
                subdict['offline'] = np.zeros_like(subdict['full'])
                for key in loaded_dict[0].keys():
                    if key not in ['evaluation', 'full', 'training_samples']:
                        subdict['offline'] += subdict[key]
                for key in avg_std_dict_compressed2.keys():
                    avg_std_dict_compressed2[key].append([np.mean(subdict[key]), np.std(subdict[key])])
            for key in avg_std_dict_compressed2.keys():
                avg_std_dict_compressed2[key] = np.array(avg_std_dict_compressed2[key])
                
            k_list = []
            for i, m in enumerate(nb_samples):
               k_list.append((m* avg_std_dict_compressed2['full'][i,0] + avg_std_dict_compressed2['coefficients'][i,0])/(avg_std_dict_compressed2['full'][i,0] - avg_std_dict_compressed2['evaluation'][i,0]))
                
            
            plt.plot(nb_samples, k_list, color=color, label=f'{dataset} - {method} - {basis[0]}', linewidth=2)
            color_index += 1
    plt.xlabel(r'$\#$ of samples points')
    plt.ylabel(r'Minimum threshold $k_min$')
    plt.xlim([5,6])
    plt.ylim([5,6])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(f'k_timings_{dataset}.pdf')
