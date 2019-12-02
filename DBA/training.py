import time
import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from DBA import stn 
from DBA import attack


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


def epoch(loader, model, opt=None, device=None, use_tqdm=False):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    if opt is None:
        model.eval()
    else:
        model.train()

    if use_tqdm:
        pbar = tqdm(total=len(loader))

    for X,y in loader:
        X,y = X.to(device), y.to(device)
  

        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

        if use_tqdm:
            pbar.update(1)

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def epoch_stn(loader, model, opt=None, device=None, use_tqdm=False, max_theta=0, max_d=0):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    if opt is None:
        model.eval()
    else:
        model.train()

    if use_tqdm:
        pbar = tqdm(total=len(loader))

    for X,y in loader:
        X,y = X.to(device), y.to(device)
        aff = stn.gen_rand_aff(max_theta, max_d, X.shape[0]).to(device)
        X = stn.stn(aff, X)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

        if use_tqdm:
            pbar.update(1)

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def epoch_transfer_train(loader, model_source, model_target, opt, attack, device, use_tqdm=True, **kwargs):
    model_source.eval()
    model_target.train()
    if use_tqdm:
        pbar = tqdm(total=len(loader))
    
    model_source.to(device)
    model_target.to(device)

    total_loss, total_err = 0.,0.

    for X,y in loader:
        X, y = X.to(device), y.to(device)
        delta = attack(model_source, X, y, **kwargs)
        yp_target = model_target(X+delta)
        loss = nn.CrossEntropyLoss()(yp_target, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_err += (yp_target.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

        if use_tqdm:
            pbar.update(1)
    
    return total_err/len(loader.dataset), total_loss/len(loader.dataset)



def epoch_transfer_attack(loader, model_source, model_target, attack, device, success_only=False, use_tqdm=True, n_test=None, **kwargs):
    source_err = 0.
    target_err = 0.
    target_err2 = 0.

    success_total_n = 0
    

    model_source.eval()
    model_target.eval()

    total_n = 0

    if use_tqdm:
        pbar = tqdm(total=n_test)

    model_source.to(device)
    model_target.to(device)
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model_source, X, y, **kwargs)

        if success_only:
            raise NotImplementedError
        else:
            yp_target = model_target(X+delta).detach()
            yp_source = model_source(X+delta).detach()
            # yp_origin = model_target(X).detach()
        source_err += (yp_source.max(dim=1)[1] != y).sum().item()
        target_err += (yp_target.max(dim=1)[1] != y).sum().item()
        # target_err2 += (yp_origin.max(dim=1)[1] != y).sum().item()
        # success_total_n += (yp_origin.max(dim=1)[1] == y)
        if use_tqdm:
            pbar.update(X.shape[0])

        total_n += X.shape[0]
        if n_test is not None:
            if total_n >= n_test:
                break

    return source_err / total_n, target_err / total_n, 0

def epoch_BPDA(loader, model, device, stepsize, epsilon, samples=1, iters=20, randomize=True, use_tqdm=True, epoch_test=None):
    total_loss, total_err = 0., 0.
    total_n = 0

    if epoch_test is None:
        pbar = tqdm(total=len(loader))
    else:
        pbar = tqdm(total=epoch_test)

    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        delta = model.BPDA(X, stepsize, epsilon, samples, iters, randomize)
        model.eval()
        yp = model(X+delta)
        loss_nn = nn.CrossEntropyLoss()(yp, y)

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss_nn.item() * X.shape[0]
        total_n += X.shape[0]
        if use_tqdm:
            pbar.update(1)
        if epoch_test is not None:
            if i >= epoch_test:
                break
    
    return total_err / total_n, total_loss/ total_n
        

def epoch_free_adversarial(loader, model, m, epsilon, opt, device, use_tqdm=False):
    """free adversarial training"""
    total_loss, total_err = 0.,0.
    total_n = 0

    pbar = tqdm(total=len(loader))


    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = torch.zeros_like(X, requires_grad=True)
        for i in range(m):
            model.train()
            yp = model(X+delta)
            loss_nn = nn.CrossEntropyLoss()(yp, y)

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss_nn.item() * X.shape[0]
            total_n += X.shape[0]

            #update network
            opt.zero_grad()
            loss_nn.backward()
            opt.step()

            #update perturbation
            delta.data = delta + epsilon*delta.grad.detach().sign()
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.grad.zero_()
        
        if use_tqdm:
            pbar.update(1)
    
    return total_err / total_n, total_loss / total_n


def epoch_ALP(loader, model, attack, alp_weight=0.5,
                opt=None, device=None, use_tqdm=False, n_test=None, **kwargs):
    """Adversarial Logit Pairing epoch over the dataset"""
    total_loss, total_err = 0.,0.

    # assert(opt is not None)
    model.train()

    if use_tqdm:
        if n_test is None:
            pbar = tqdm(total=len(loader.dataset))
        else:
            pbar = tqdm(total=n_test)
    total_n = 0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        model.eval()
        with torch.no_grad():
            clean_logit = model(X)
        delta = attack(model, X, y, **kwargs)
        
        model.train()
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y) + alp_weight*nn.MSELoss()(yp, clean_logit)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        if use_tqdm:
            pbar.update(X.shape[0])

        total_n += X.shape[0]

        if n_test is not None:
            if total_n >= n_test:
                break
        
    return total_err / total_n, total_loss / total_n

def epoch_adversarial(loader, model, attack, 
                opt=None, device=None, use_tqdm=False, n_test=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.

    if opt is None:
        model.eval()
    else:
        model.train()

    if use_tqdm:
        if n_test is None:
            pbar = tqdm(total=len(loader.dataset))
        else:
            pbar = tqdm(total=n_test)
    total_n = 0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        model.eval()
        delta = attack(model, X, y, **kwargs)
        if opt:
            model.train()
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        if use_tqdm:
            pbar.update(X.shape[0])

        total_n += X.shape[0]

        if n_test is not None:
            if total_n >= n_test:
                break
        
    return total_err / total_n, total_loss / total_n

def get_activation(model, activation, name):
    def hook(model, input, output):
        activation[name] = output.cpu().detach()
    return hook

def register_layer(model, layer, activation, name):
    layer.register_forward_hook(get_activation(model, activation, name))




def squared_l2_norm(x):
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            for idx_batch in range(batch_size):
                grad_idx = grad[idx_batch]
                grad_idx_norm = l2_norm(grad_idx)
                grad_idx /= (grad_idx_norm + 1e-8)
                x_adv[idx_batch] = x_adv[idx_batch].detach() + step_size * grad_idx
                eta_x_adv = x_adv[idx_batch] - x_natural[idx_batch]
                norm_eta = l2_norm(eta_x_adv)
                if norm_eta > epsilon:
                    eta_x_adv = eta_x_adv * epsilon / l2_norm(eta_x_adv)
                x_adv[idx_batch] = x_natural[idx_batch] + eta_x_adv
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def epoch_trade(loader, model, 
                opt, device=None, **kwargs):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        opt.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=opt,
                           **kwargs)
                        #    step_size=args.step_size,
                        #    epsilon=args.epsilon,
                        #    perturb_steps=args.num_steps,
                        #    beta=args.beta)
        loss.backward()
        opt.step()

    return 0, 0


def epoch_recon(loader, model, optnet, opt, criterion, device, use_tqdm=True):
    total_loss = 0
    total_n = 0
    model.to(device)
    if opt is None:
        model.eval()
    else:
        model.train()
    
    if use_tqdm:
        pbar = tqdm(total=len(loader.dataset))

    
    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        out = model(imgs)
        if opt is not None:
            opt.zero_grad()
        adv_img = optnet.adversary(imgs) + imgs
        loss = criterion(out, adv_img)
        loss.backward()
        if opt is not None:
            opt.step()
        total_loss += loss.item() * imgs.shape[0]
        total_n += imgs.shape[0]
        if use_tqdm:
            pbar.update(imgs.shape[0])
    return total_loss/total_n

def epoch_surrogate(loader, basemodel, ae, optnet, attack, device, use_tqdm=True, n_test=None, **kwargs):
    total_loss, total_err = 0.,0.

    basemodel.eval()
    ae.eval()
    optnet.eval()

    surrogate_model = nn.Sequential(ae, basemodel)
    surrogate_model.eval()

    if use_tqdm:
        if n_test is None:
            pbar = tqdm(total=len(loader.dataset))
        else:
            pbar = tqdm(total=n_test)
    total_n = 0

    for X,y in loader:
        X,y = X.to(device), y.to(device)
        
        delta = attack(surrogate_model, X, y, **kwargs)
        
        yp = optnet(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
  
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        if use_tqdm:
            pbar.update(X.shape[0])

        total_n += X.shape[0]

        if n_test is not None:
            if total_n >= n_test:
                break
        
    return total_err / total_n, total_loss / total_n

# optnet.eval()
# test_err, test_loss = training.epoch(test_loader, optnet, device=device, use_tqdm=True)
# print('test:', 1-test_err)

# err, loss = training.epoch_BPDA(test_loader, optnet, device=device, 
#                                 stepsize=0.003, epsilon=0.031, 
#                                 samples=1, iters=20, randomize=True, use_tqdm=True, epoch_test=20)
# print('BPDA-I:', 1-err)

# bpdanet = copy.deepcopy(basenet)

# bpdanet.load_state_dict(torch.load(base_path))
# source_err, err1, err2 = training.epoch_transfer_attack(test_loader,
#                                        bpdanet, optnet, 
#                                         attack=attack.pgd_linf_untargeted, device=device, n_test=5000, use_tqdm=True,
#                                         epsilon=0.031,alpha=0.003,num_iter=20,randomize=True)
# print('DA:', 1-source_err, 1-err1)

# attacknet1 = resnet.ResNet18()
# attacknet1.load_state_dict(torch.load('./models/models/advmodel/resnet18_cifar.pth', map_location='cpu'))
# source_err, err1, err2 = training.epoch_transfer_attack(test_loader,
#                                        attacknet1, optnet, 
#                                         attack=attack.pgd_linf_untargeted, device=device, n_test=5000, use_tqdm=True,
#                                         epsilon=0.031,alpha=0.003,num_iter=20,randomize=True)
# print('transfer from clean', 1-source_err, 1-err1)

# attacknet1 = vgg.vgg19_bn()
# attacknet1.load_state_dict(torch.load('./models/models/advmodel/vgg19_bn2.pth', map_location='cpu'))
# source_err, err1, err2 = training.epoch_transfer_attack(test_loader,
#                                        attacknet1, optnet, 
#                                         attack=attack.pgd_linf_untargeted, device=device, n_test=5000, use_tqdm=True,
#                                         epsilon=0.031,alpha=0.003,num_iter=20,randomize=True)
# print('transfer from vgg', 1-source_err, 1-err1)

# attacknet2 = resnet.ResNet18()
# attacknet2.load_state_dict(torch.load('./models/models/advmodel/resnet18_cifar_adv.pth', map_location='cpu'))
# source_err, err1, err2 = training.epoch_transfer_attack(test_loader,
#                                        attacknet2, optnet, 
#                                         attack=attack.pgd_linf_untargeted, device=device, n_test=5000, use_tqdm=True,
#                                         epsilon=0.031,alpha=0.003,num_iter=20,randomize=True)
# print('transfer from adv', 1-source_err, 1-err1)

def eval_robustness(optnet, test_loader, device, use_tqdm=True, evalmethods=[]):
    optnet.eval()
    for evalmethod in evalmethods:
        if evalmethod['name'] == 'std':
            test_err, test_loss = epoch(test_loader, optnet, device=device, use_tqdm=True)
            print('standard acc:', 1-test_err)
        elif evalmethod['name'] == 'BPDA':
            err, loss = epoch_BPDA(test_loader, optnet, device=device, 
                                stepsize=0.003, epsilon=0.031, 
                                samples=evalmethod['samples'], iters=evalmethod['iters'], randomize=True, use_tqdm=use_tqdm, epoch_test=evalmethod['epoch_test'])
            print('BPDA:', 1-err)
        elif evalmethod['name'] == 'DA':
            danet = copy.deepcopy(optnet.basemodel)
            source_err, err1, err2 = epoch_transfer_attack(test_loader,
                                       danet, optnet, 
                                        attack=attack.pgd_linf_untargeted, device=device, n_test=evalmethod['n_test'], use_tqdm=True,
                                        epsilon=0.031, alpha=0.003, num_iter=20, randomize=True)
            print('DA:', 'source acc:', 1-source_err, 'targ acc:', 1-err1)
        elif evalmethod['name'] == 'transfer':
            net = evalmethod['net']
            name = evalmethod['model_name']
            source_err, err1, err2 = epoch_transfer_attack(test_loader,
                                       net, optnet, 
                                        attack=attack.pgd_linf_untargeted, device=device, n_test=evalmethod['n_test'], use_tqdm=True,
                                        epsilon=0.031, alpha=0.003, num_iter=20, randomize=True)
            print('transfer from', name, 'source acc:', 1-source_err, 'targ acc:', 1-err1)

# print('BPDA-I:', 1-err)


