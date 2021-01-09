import simple_projections as sp
import torch_logreg
import dual_robust_opt 
import numpy as np

def projected_loss(loss_values, rho, alpha, tmax):
    """
    given v = loss_values, finds
    \min_{p \in D_1,rho} E_{i\sim p}[v_i]
    ie the chi-square ball element maximizing the expectation.
    """
    wv = np.ones(len(loss_values))/float(len(loss_values))
    for i in range(tmax):
        wv += alpha*loss_values/np.sqrt(i+1.0)
        wv = sp.project_onto_chi_square_ball(wv, rho, 1e-5)
    return wv


def eta_minval_factory(x,y,k):
    return lambda eta: torch_logreg.train(x, y, 1000, eta, k)[1][-1]**(1.0/k)

def get_lm_via_dual(x,y,ktype,eps):
    robust_pack = dual_robust_opt.robust_dual_opt(ktype, eps)
    if ktype == 'cvar':
        k = 1
    else:
        k = 2
    eta_function = eta_minval_factory(x, y, k)
    opt_eta = robust_pack.bisect_losses(eta_function, 10.0)
    model, cost_list = torch_logreg.train(x, y, 1000, opt_eta, k)
    coef, icept = torch_logreg.dump_model(model)
    return model, coef, icept, opt_eta

def eval_model(model, xheld):
    return torch_logreg.predict(model[0], xheld)


