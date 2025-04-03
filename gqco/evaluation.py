
import numpy as np
import statistics
from math import exp, log10
import torch

from gqco.utils import print0, fix_seed, arange_token, get_answer
from gqco.solve import solve_from_token
from gqco.data import generate_data





def make_log(args, model, task, log_dict={}, seed=0, current_size=None):
    min_value, max_value, mean_value, std_value, min_pr, max_pr = performance_evaluation(
        args,
        task, 
        model, 
        seed_eval=seed, 
        nsample=10, 
        ntry=10, 
        current_size=current_size,
        num_shot=-1
    ) 

    log_dict_add = {
        'Average of min energy': min_value,
        'Average of max energy': max_value,
        'Average of mean energy': mean_value,
        'Average of std of energy': std_value,
        'Average of log10 probability for min': min_pr,
        'Average of log10 probability for max': max_pr
    }

    log_dict.update(log_dict_add)

    return log_dict




def check_accuracy(args, model, task, log_dict={}, seed=0, num_problems=1000):

    model.eval()

    if args.tune_size > 0:
        size_list = [args.tune_size]
    else:
        if log_dict['current_size'] == 3:
            size_list = [3]
        else:
            size_list = [n for n in range(3, log_dict['current_size']+1)]
    

    tmp_acc = 0
    acc_rate_list = []
    for size in size_list:

        acc_list = []
        err_list = []
        for itr in range(num_problems):

            itr_seed = int(int((str(int(373+itr)) + str(int(seed))))  % (2**30))
            fix_seed(itr_seed)
            ## Generate data
            adj, size, record = generate_data(args, num_clone=10, seed=itr_seed, device=args.device, size=size)

            ## Generate circuits
            with torch.no_grad():
                out_tokens, probs_all, _, logits_all = model.forward(record, temperature=1.0, same_token_penalty=0.0, masked_tokens=task.bad_tokens[size]) 
            tokens_list = [arange_token(t, args) for t in out_tokens.detach().cpu().tolist()]


            # energies = [task.compute_energy(t, adj, args.num_shot) for t in tokens_list]
            energies = []
            energy_cache = {}
            for t in tokens_list:
                t_tuple = tuple(t)
                if t_tuple not in energy_cache:
                    energy = task.compute_energy(t, adj, args.num_shot)
                    energy_cache[t_tuple] = energy
                else:
                    energy = energy_cache[t_tuple]
                energies.append(energy)


            ## Get the best
            idx_min = energies.index(min(energies))
            token_min = tokens_list[idx_min]
            energy_min = energies[idx_min]

            dict_pred, dict_true, qc = solve_from_token(task, token_min, adj, is_print=False)

           
            min_keys_pred, min_val_pred = get_answer(dict_pred, metric='max')
            min_keys_true, min_val_true = get_answer(dict_true, metric='min')

            is_correct = 0
            if len(set(min_keys_pred) & set(min_keys_true)):
                is_correct = 1

            abs_error = abs(energy_min - min_val_true)

            acc_list.append(is_correct)
            err_list.append(abs_error)


        this_acc = sum(acc_list) / num_problems
        log_dict_add = {
            f'Accuracy (size: {size})': this_acc,
            f'MAE (size: {size})': sum(err_list) / num_problems
        }
        tmp_acc += sum(acc_list)
        acc_rate_list.append(sum(acc_list) / num_problems)

        log_dict.update(log_dict_add)

    log_dict.update({'Accuracy': tmp_acc / (len(size_list) * num_problems)})

    accs = {
        'max' : max(acc_rate_list),
        'min' : min(acc_rate_list),
        'ave' : sum(acc_rate_list) / len(size_list)
    }
    return log_dict, accs





   

def performance_evaluation(args, task, model, seed_eval=0, nsample=10, ntry=10, num_shot=-1, current_size=None):

    min_values = []
    max_values = []
    mean_values = []
    std_values = []
    min_pr = []
    max_pr = []
    min_tokens = []
    sizes = []
    for itr in range(nsample):

        seed_itr = int(int((str(int(373+itr)) + str(int(seed_eval))))  % (2**30))

        adj, size, record = generate_data(args, num_clone=ntry, seed=seed_itr, device=args.device, current_size=current_size)

        fix_seed(int(seed_itr))
        with torch.no_grad():  
            out_tokens, _, probs_token, _ = model.forward(record, temperature=1, same_token_penalty=0.0, masked_tokens=task.bad_tokens[size]) 
        tokens_list = [arange_token(t, args) for t in out_tokens.detach().cpu().tolist()]

        # es = [task.compute_energy(t, adj, num_shot) for t in out_tokens.detach().cpu().tolist()]
        es = []
        energy_cache = {}
        for t in tokens_list:
            t_tuple = tuple(t)
            if t_tuple not in energy_cache:
                energy = task.compute_energy(t, adj, args.num_shot)
                energy_cache[t_tuple] = energy
            else:
                energy = energy_cache[t_tuple]
            es.append(energy)

        pr = torch.sum(torch.log(probs_token), dim=1).detach().cpu().float().numpy().reshape(-1).tolist()

        min_values.append(min(es))
        max_values.append(max(es))
        mean_values.append(sum(es) / len(es))
        std_values.append(statistics.stdev(es))
        min_pr.append(pr[es.index(min(es))])
        max_pr.append(pr[es.index(max(es))])
        min_tokens.append(tokens_list[es.index(min(es))])
        sizes.append(size)


    
    mean_min_value = np.mean(np.array(min_values))
    mean_max_value = np.mean(np.array(max_values))
    mean_mean_value = np.mean(np.array(mean_values))
    mean_std_value = np.mean(np.array(std_values))
    mean_min_pr = np.mean(np.array(min_pr))
    mean_max_pr = np.mean(np.array(max_pr))


    for t in min_tokens:
        print0(t, args)

    return mean_min_value, mean_max_value, mean_mean_value, mean_std_value, log10(exp(1))*mean_min_pr, log10(exp(1))*mean_max_pr

