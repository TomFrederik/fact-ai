import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')


np.random.seed(0)

mus = [[0,1],[np.sqrt(1.0/3.0), 0],[-1*np.sqrt(1.0/3.0), 0]]
x_seq = []
for x1 in np.linspace(-0.6, 0.6, num=20):
    for x2 in np.linspace(-0.1, 1.1, num=20):
        x_seq.append((x1, x2))

eps_seq = [0.2, 0.4, 0.8]#np.linspace(0.05, 0.2, num=3)


def threshplus(x):
    y = x.copy()
    y[y<0]=0
    return y

def loss_map_chi_factory(loss_values, eps):
    return lambda x: np.sqrt(2)*np.sqrt(0.5*(1.0/eps-1.0)**2.0+1.0)*np.sqrt(np.mean(threshplus(loss_values-x)**2.0)) + x

def get_losses(lossfun, target_values, x_seq, eps_seq):
    all_losses_sq = []
    ERM_loss = []
    for x in x_seq:
        loss_values = lossfun(x, target_values)
        chi_losses = []
        ERM_loss.append(np.mean(loss_values))
        for i in range(len(eps_seq)):
            chi_loss = loss_map_chi_factory(loss_values, eps_seq[i])
            cutpt = optimize.fminbound(chi_loss, np.min(loss_values)-1000.0,np.max(loss_values), xtol=0.01)
            chi_losses.append(chi_loss(cutpt))
        all_losses_sq.append(chi_losses)
    losses_by_qtl_sq = zip(*all_losses_sq)
    return losses_by_qtl_sq, ERM_loss


def draw_samples(N, frac_vec):
    alphas = frac_vec
    deviation_values = np.random.randn(N, len(mus[0]))/5.0
    z_index = np.random.choice(len(alphas), N, p=alphas)
    target_values = np.array([mus[z_index[i]] + deviation_values[i] for i in range(N)])
    return target_values, z_index

lossfun_abs = lambda x,y: np.sqrt(np.sum((x-y)**2.0,axis=1))

alpha_vec = []
for p1 in np.arange(0.1,0.9,0.05):
    for p2 in np.arange(0.1,0.9-p1,0.05):
        alpha_vec.append([p1, p2, 1.0-(p1+p2)])


from tqdm import tqdm
loss_list = []
for av_in in tqdm(alpha_vec):
    target_values, z_index = draw_samples(1000, av_in)
    loss_pack = get_losses(lossfun_abs, target_values, x_seq, eps_seq)
    min_erm = np.argmin(loss_pack[1])
    min_idxs = [np.argmin(lvec) for lvec in loss_pack[0]]
    target_values, z_index = draw_samples(1000, np.array([1.0,1.0,1.0])/3.0)
    eval_pt = lambda x: [np.mean(lossfun_abs(target_values[z_index==i],x)) for i in [0,1,2]]
    lerm = eval_pt(x_seq[min_erm])
    ldro = [eval_pt(x_seq[mi]) for mi in min_idxs]
    loss_list.append([lerm, ldro])


import egtsimplex


def f_interp_meta(x,t, alpha_vec, delta_pack):
    al_x, al_y, al_z = zip(*alpha_vec)
    dv = (al_x - x[0])**2.0 + (al_y - x[1])**2.0 + (al_z - x[2])**2.0
    dind = np.argmin(dv)
    return [delta_pack[i][dind] for i in range(3)]

def f_interp_fact(alpha_vec, delta_pack):
    return lambda x, t: f_interp_meta(x, t, alpha_vec, delta_pack)

def loss_mapper(pr, x, alpha=1.0, delta=0.05):
    z=pr*np.exp(-alpha*np.array(x)) + delta
    z=z/np.sum(z)
    return z

delta_list = []
for i in range(len(alpha_vec)):
    av_in = alpha_vec[i]
    lerm = loss_list[i][0]
    ldro = loss_list[i][1]
    erm_delta = loss_mapper(av_in,lerm)-av_in
    dro_delta = [loss_mapper(av_in,lsub)-av_in for lsub in ldro]
    delta_list.append([erm_delta, dro_delta])


lvec = np.arange(0.0, 0.12, 0.005)
#cm = 'viridis_r'
cm = 'Blues'
dynamics=egtsimplex.simplex_dynamics(f_interp_fact(alpha_vec, zip(*[delt[0] for delt in delta_list ])))
fig,ax=plt.subplots()
dynamics.plot_simplex(ax, levels=lvec, cmap=cm)
plt.tight_layout()
plt.savefig('test_vec_erm.png')

dynamics=egtsimplex.simplex_dynamics(f_interp_fact(alpha_vec, zip(*[delt[1][1] for delt in delta_list ])))
fig,ax=plt.subplots()
dynamics.plot_simplex(ax, levels=lvec, cmap=cm)
plt.tight_layout()
plt.savefig('test_vec_dro1.png')

dynamics=egtsimplex.simplex_dynamics(f_interp_fact(alpha_vec, zip(*[delt[1][2] for delt in delta_list ])))
fig,ax=plt.subplots()
dynamics.plot_simplex(ax, levels=lvec, cmap=cm)
plt.tight_layout()
plt.savefig('test_vec_dro2.png')


