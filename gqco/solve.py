import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Math




def adj_to_text(adj):
    txt = ''
    
    for i in range(len(adj)):
        w = adj[i, i]
        w_str = f'{w:.4f}'
        if w > 0:
            txt += rf'+{w_str}Z_{{{i}}}'
        if w < 0:
            txt += rf'{w_str}Z_{{{i}}}'
    
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            w = adj[i, j]
            w_str = f'{w:.4f}'
            ij_str = f'{i}{j}'
            if w > 0:
                txt += rf'+{w_str} Z_{{{i}}} Z_{{{j}}}'
            if w < 0:
                txt += rf'{w_str} Z_{{{i}}} Z_{{{j}}}'

    
    display(Math(txt))
    return txt




def solve_from_token(taskobj, token, adj, is_print=True):

    while token and token[-1] == 0:
        token.pop()

    qc = taskobj.get_circuit(token, size=len(adj))

    if is_print:
        print('-- Token --')
        print(token)
        print('\n')

        print('-- Circuit --')
        qc.draw()
        print('\n')

        print('-- Transpiled circuit --')
        qc.draw(is_transpile=True)


    vector = qc.get_state()
    probs = np.abs(vector)**2

    pred = probs_to_result(probs)
    true = brute_solver(adj)

    return pred, true, qc


def probs_to_result(probs):
    result = {}

    nqubit = int(np.log2(len(probs)))
        
    for i, b in enumerate(range(2**nqubit)):
        bit = f'{b:0>{nqubit}b}'[::-1]
        result[bit] = probs[i]

    return result   


def brute_solver(data):

    result = {}

    nqubit = len(data)
        
    for b in range(2**nqubit):

        bit = f'{b:0>{nqubit}b}'
        z = [-int(v)*2+1 for v in bit]
        
        val = 0
        for i in range(nqubit):
            for j in range(i, nqubit):
                w = data[i][j]
                if i == j:
                    val += w*z[i]
                if i != j:
                    val += w*z[i]*z[j]
   
        result[bit] = val.tolist()

    return result


def plot_from_dict(dict_pred, dict_true, savefile, is_legend=False, figsize=(5, 4), is_true=True):

    dict_pred = {key: dict_pred[key] for key in dict_true if key in dict_pred}

    # Extract keys and values
    keys = list(dict_pred.keys())
    values_pred = list(dict_pred.values())
    values_true = list(dict_true.values())

    # Create the figure and the first axis
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Create a bar plot for dict_pred
    ax1.grid(axis='x', which='both', color='gray', alpha=0.5, linestyle='--', linewidth=0.5, zorder=0)
    ax1.bar(keys, values_pred, color=sns.color_palette('Blues', 24)[12], label="Machine's answer", alpha=1, zorder=2)
    ax1.set_xlabel('Keys', size=11)
    ax1.set_ylabel("Probability amplitude", size=11)
    ax1.tick_params(axis="y")


    if is_true:
        # Create the second axis
        ax2 = ax1.twinx()
        
        # Create a scatter plot for dict_true
        min_value = min(values_true)
        min_indices = [i for i, value in enumerate(values_true) if value == min_value]
        
        # Scatter plot with different color for the minimum value
        colors = [sns.color_palette('Blues', 24)[-1]] * len(values_true)
        for i in min_indices:
            colors[i] = 'tomato'  # Change color of the minimum value
        
        ax2.scatter(keys, values_true, color=colors, label='Exact value', zorder=3)
        ax2.set_ylabel('True cost', size=11)
        ax2.tick_params(axis='y')

    ax1.set_xlabel(None)

    
    # Show the plot
    if is_legend:
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.96), ncol=2)
    ax1.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(f'{savefile}-probs.svg', bbox_inches='tight')
    plt.show()


def adj_to_text(adj):
    txt = ''
    
    for i in range(len(adj)):
        w = adj[i, i]
        w_str = f'{w:.4f}'
        if w > 0:
            txt += rf'+{w_str}Z_{{{i}}}'
        if w < 0:
            txt += rf'{w_str}Z_{{{i}}}'
    
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            w = adj[i, j]
            w_str = f'{w:.4f}'
            ij_str = f'{i}{j}'
            if w > 0:
                txt += rf'+{w_str} Z_{{{i}}} Z_{{{j}}}'
            if w < 0:
                txt += rf'{w_str} Z_{{{i}}} Z_{{{j}}}'

    
    display(Math(txt))
    return txt
