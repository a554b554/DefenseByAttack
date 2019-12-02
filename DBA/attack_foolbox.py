import foolbox
import torch
import torchvision.models as models
import numpy as np


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def epoch_foolbox(loader, fmodel, attack, use_tqdm=True, eps=0.031, n_test=1000, d='linf', **kwargs):
    n_total = 0
    n_err = 0
    dists = []

    ne_record = 0
    if use_tqdm:
        pbar = tqdm(total=n_test)

    for X, y in loader:
        X, y = X.numpy(), y.numpy()
        # X, y = X[0,:].numpy(), y.numpy()
        # print(X.shape, y)
        adv = attack(X, y, **kwargs)
        # print(adv.shape)
        if adv is not None:
            if np.argmax(fmodel.forward(adv))!=y:
                #attack success
                ne_record+=1
                # print('success', ne_record)
                if d == 'linf':
                    dists.append(np.abs(X-adv).max())
                elif d == 'l2':
                    dists.append(np.linalg.norm(X-adv, ord=2))
                else:
                    raise NotImplementedError
        
        n_total += 1
        if use_tqdm:
            pbar.update(1)

        if n_total>=n_test:
            break
    n_err = (np.array(dists)<=eps+0.001).sum()
    return n_err/n_total, dists