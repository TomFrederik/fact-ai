import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')


np.random.seed(0)

mus = [-1,1]
x_seq = np.linspace(-2, 2, num=200)
eps_seq = np.linspace(0.05, 0.7, num=20)


def threshplus(x):
    y = x.copy()
    y[y<0]=0
    return y

def loss_map_cvar_factory(loss_values, eps):
    return lambda x: np.mean(threshplus(loss_values-x))/eps + x

def loss_map_chi_factory(loss_values, eps):
    return lambda x: np.sqrt(2)*(1.0/eps-1.0)*np.sqrt(np.mean(threshplus(loss_values-x)**2.0)) + x

def get_losses(lossfun, target_values, x_seq, eps_seq):
    all_losses = []
    all_losses_sq = []
    ERM_loss = []
    for x in x_seq:
        loss_values = [lossfun(x, y) for y in target_values]
        cvar_losses = []
        chi_losses = []
        ERM_loss.append(np.mean(loss_values))
        for i in range(len(eps_seq)):
            cvar_loss = loss_map_cvar_factory(loss_values, eps_seq[i])
            cutpt = optimize.fminbound(cvar_loss, np.min(loss_values),np.max(loss_values))
            cvar_losses.append(cvar_loss(cutpt))
            chi_loss = loss_map_chi_factory(loss_values, eps_seq[i])
            cutpt = optimize.fminbound(chi_loss, np.min(loss_values)-1000.0,np.max(loss_values))
            chi_losses.append(chi_loss(cutpt))
        all_losses.append(cvar_losses)
        all_losses_sq.append(chi_losses)
    losses_by_qtl = zip(*all_losses)
    losses_by_qtl_sq = zip(*all_losses_sq)
    return losses_by_qtl, losses_by_qtl_sq, ERM_loss


def draw_samples(N, frac):
    alphas = [1.0-frac, frac]
    deviation_values = np.random.randn(N)/5.0
    z_index = np.random.choice(len(alphas), N, p=alphas)
    target_values = np.array(mus)[z_index]+deviation_values
    return target_values

def plot_losses(loss_pack, target_values, epsin, iseq = range(0, len(eps_seq), 3), title='', fileout='figs/test'):
    cmap = plt.get_cmap('cool')
    f, axarr = plt.subplots(2, sharex=True, figsize=(8,4))
    axarr[1].hist(target_values)
    ERM_loss = loss_pack[2]
    losses_by_qtl = loss_pack[0]
    losses_by_qtl_sq = loss_pack[1]
    for i in iseq:
        axarr[0].plot(x_seq, losses_by_qtl[i]-np.min(losses_by_qtl[i]),
                      label='{:.2f}'.format(epsin[i]), color=cmap(i/float(max(iseq))), linewidth=2.0)
    axarr[0].plot(x_seq, ERM_loss-np.min(ERM_loss), label='ERM', color=cmap(1.0), linewidth=2.0)
    axarr[0].legend(title='Minority fraction bound')
    axarr[0].set_ylim(0,1)
    f.suptitle(title+'DRO with CVAR ball')
    f.tight_layout()
    f.savefig(fileout+'_cvar.png')
    plt.close()
    #chisq
    #f, axarr = plt.subplots(2, sharex=True, figsize=(8, 4))
    f = plt.figure(figsize=(8,4))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    for i in iseq:
        ax1.plot(x_seq, losses_by_qtl_sq[i]-np.min(losses_by_qtl_sq[i]),
                      label='{:.2f}'.format(epsin[i]), color=cmap(i/float(max(iseq))), linewidth=2.0)
    ax1.plot(x_seq, ERM_loss-np.min(ERM_loss), label='ERM', color=cmap(1.0), linewidth=2.0)
    ax1.legend(title='Minority fraction bound')
    ax1.set_ylim(0,1)
    ax1.set_xlim(-2,2)
    ax1.set_ylabel('loss')
    ax2 = plt.subplot2grid((3,1), (2, 0))
    ax2.hist(target_values)
    ax2.set_xlim(-2,2)
    ax2.set_ylabel('frequency')
    f.suptitle('Fair parameter estimation using DRO')
    f.tight_layout()
    f.savefig(fileout+'_chisq.png')
    plt.close()


lossfun_sq = lambda x,y: (x-y)**2.0
lossfun_abs = lambda x,y: np.abs(x-y)

for alpha in [0.5, 0.3, 0.2, 0.1, 0.05]:
    print alpha
    target_values = draw_samples(1000, alpha)
    loss_pack = get_losses(lossfun_sq, target_values, x_seq, eps_seq)
    #plot_losses(loss_pack, target_values, eps_seq, title='mean estimation, alpha='+str(alpha)+' ',fileout='figs/meanest'+str(alpha))

for alpha in [0.5, 0.3, 0.2, 0.1, 0.05]:
    print alpha
    target_values = draw_samples(1000, alpha)
    loss_pack = get_losses(lossfun_abs, target_values, x_seq, eps_seq)
    #plot_losses(loss_pack, target_values, eps_seq, title='median estimation, alpha='+str(alpha)+' ',fileout='figs/medest'+str(alpha))


alpha=0.1
target_values = draw_samples(1000, 0.1)
eps_seq_2 = [0.05, 0.1, 0.15, 0.2, 0.3]
loss_pack = get_losses(lossfun_abs, target_values, x_seq, eps_seq_2)

#plot_losses(loss_pack, target_values, eps_seq_2, range(len(eps_seq_2)), title='median estimation, alpha='+str(alpha)+' ',fileout='figs/medest'+str(alpha))


epsin = 0.3
all_losses_sq = []
ERM_loss = []
cut_seq = []
all_losses = []
class_loss = []
for x in x_seq:
    loss_values = [lossfun_abs(x, y) for y in target_values]
    all_losses.append(loss_values)
    chi_losses = []
    ERM_loss.append(np.mean(loss_values))
    lvarr=np.array(loss_values)
    class_loss.append(max(np.mean(lvarr[target_values>0]),
                          np.mean(lvarr[target_values<0])))
    chi_loss = loss_map_chi_factory(loss_values, epsin)
    cutpt = optimize.fminbound(chi_loss, np.min(loss_values)-1000.0,np.max(loss_values))
    cut_seq.append(cutpt)
    chi_losses.append(chi_loss(cutpt))
    all_losses_sq.append(chi_losses)




losses_by_qtl_sq = zip(*all_losses_sq)

chisq_risk = losses_by_qtl_sq[0]

cut_val = cut_seq[np.argmin(chisq_risk)]
wts = all_losses[np.argmin(chisq_risk)]
rlu = all_losses[np.argmin(chisq_risk)] - cut_val
rlu[rlu < 0] = 0

plt.figure(figsize=(6,3))
plt.plot(x_seq, chisq_risk, linewidth=3.0, label='DRO')
plt.plot(x_seq, ERM_loss, linewidth=3.0, label='ERM')
plt.plot(x_seq, class_loss, linewidth=3.0, label='Minority Risk')
plt.ylabel('Risk')
plt.legend(loc='lower right')
plt.savefig('test_risk.png')
plt.close()


plt.figure(figsize=(6,3))
plt.hist(target_values, 50, normed=True, histtype='step', weights=rlu,label='DRO distribution',linewidth=3.0)
plt.hist(target_values, 50, normed=True, histtype='step',label='Empirical distribution',linewidth=3.0)
plt.legend()
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('test_mu.png')

plt.close()
