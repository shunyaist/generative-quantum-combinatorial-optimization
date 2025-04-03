import torch
from torch.nn import functional as F

from gqco.utils import arange_token







def compute_loss(task, adj, tokens, probs, args):
    if args.loss_type in ['ppo']:
        return compute_loss_ppo(task, adj, tokens, probs, args)
    elif args.loss_type in ['ppo-dpo']:
        return compute_loss_ppo_dpo(task, adj, tokens, probs, args)
    else:
        return compute_loss_dpo(task, adj, tokens, probs, args)






def compute_loss_dpo(task, adj, tokens, probs, args):

    tokens_list = [arange_token(t, args) for t in tokens.detach().tolist()]
    len_record = len(tokens_list)


    # neg_energies = [-task.compute_energy(t, adj, args.num_shot) for t in tokens_list]
    neg_energies = []
    energy_cache = {}
    for t in tokens_list:
        t_tuple = tuple(t)
        if t_tuple not in energy_cache:
            energy = -task.compute_energy(t, adj, args.num_shot)
            energy_cache[t_tuple] = energy
        else:
            energy = energy_cache[t_tuple]
        neg_energies.append(energy)


    ## Max tokens
    index_of_max = neg_energies.index(max(neg_energies))
    index_of_min = neg_energies.index(min(neg_energies))



    ## Define iterator
    if args.loss_mode == 'one-to-all':
        ## w: max_index, ell: 1,2,3,...
        iterator1 = [index_of_max]
        iterator2 = lambda y: filter(lambda x: x != index_of_max, range(len_record))
    elif args.loss_mode == 'all-to-all':
        ## w: 1,2,3,... ell: w+1,w+2,...
        iterator1 = range(len_record-1)
        iterator2 = lambda x: range(x+1, len_record)
    elif args.loss_mode == 'maxmin-to-all':
        ## w: 1,2,3,... ell: w+1,w+2,...
        if args.epoch < 1000:
            iterator1 = [index_of_max]
            iterator2 = lambda x: range(len_record)
        else:
            iterator1 = [index_of_max, index_of_min]
            iterator2 = lambda y: filter(lambda x: x != index_of_max and x != index_of_min, range(len_record))
    elif args.loss_mode == 'max-to-min':
        ## w: max_index. ell: min_index
            iterator1 = [index_of_max]
            iterator2 = lambda x: [index_of_min]




    ## Compute loss
    loss = torch.tensor(0.0).to(args.device)
    count = 0
    for w in iterator1:
        token_w = tokens_list[w]
        probs_w = probs[w, :len(token_w)].clone()
        onehot_w = F.one_hot(torch.tensor(token_w), num_classes=args.vocab_size).to(args.device)
        log_psum_w = torch.sum(torch.log(probs_w**onehot_w) / len(token_w))

        for ell in iterator2(w):
            token_ell = tokens_list[ell]

            probs_ell = probs[ell, :len(token_ell)].clone()

            onehot_ell = F.one_hot(torch.tensor(token_ell), num_classes=args.vocab_size).to(args.device)

            log_psum_ell = torch.sum(torch.log(probs_ell**onehot_ell) / len(token_ell))

            neg_energy_w = neg_energies[w]
            neg_energy_ell = neg_energies[ell]

            if neg_energy_w > neg_energy_ell:
                s = +1
                count += 1
            elif neg_energy_w < neg_energy_ell:
                s = -1
                count += 1
            elif neg_energy_w == neg_energy_ell:
                s = 0

            _l = s * (log_psum_w - neg_energy_w - log_psum_ell + neg_energy_ell)

            if args.loss_type in ['dpo', 'ppo-dpo']:
                loss += (s**2) * torch.log(1 + torch.exp( - torch.clamp(args.dpo_beta * _l, min=-10)))

            if args.loss_type == 'wdpo':
                loss += abs(neg_energy_w - neg_energy_ell) * torch.log(1 + torch.exp( - torch.clamp(args.dpo_beta * _l, min=-10)))

            if args.loss_type == 'ipo':
                loss += (s**2) * (_l - 1 / (2 * args.ipo_beta))**2

    if count != 0:
        loss /= count
    

    ## Negative log likelihood loss
    loss += - 1 * log_psum_w

    min_energy = -max(neg_energies)
    max_energy = -min(neg_energies)
    mean_energy = - sum(neg_energies) / len(neg_energies)

    

    return loss, log_psum_w, min_energy, max_energy, mean_energy






def compute_loss_ppo(task, adj, tokens, probs, args):

    tokens_list = [arange_token(t, args) for t in tokens.detach().tolist()]

    neg_energies = [-task.compute_energy(t, adj, args.num_shot) for t in tokens_list]

    max_e = max(neg_energies)
    min_e = min(neg_energies)
    if min_e == max_e:
        neg_energies_scal = [1 for _ in neg_energies]
    else:
        neg_energies_scal = [(e - min_e)/(max_e - min_e) for e in neg_energies]

    ## Compare
    loss = torch.tensor(0.0).to(args.device)
    count = 0
    for i, (t, p, e) in enumerate(zip(tokens_list, probs, neg_energies_scal)):

        p_cut = p[:len(t)].clone()
        onehot = F.one_hot(torch.tensor(t), num_classes=args.vocab_size).to(args.device)
        log_psum = torch.sum(torch.log(p_cut**onehot) / len(t))

        loss += - log_psum * e
    loss /= len(tokens_list)

    return loss





def compute_loss_ppo_dpo(task, adj, tokens, probs, args):
    loss1 = compute_loss_ppo(task, adj, tokens, probs, args)
    loss2 = compute_loss_dpo(task, adj, tokens, probs, args)
    return loss1 +  loss2
