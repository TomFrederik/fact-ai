import numpy as np
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import robust_class
from tqdm import tqdm
import robust_class
import torch_logreg

R=3.0
alpha_init = [0.5,0.5]
N = 1000
mean_vecs = [[-R/2.0,0],[R/2.0,0]]
tilt = 2.0
population_decision = [[-tilt/R, math.sqrt(R**2.0-1.0)/R],[tilt/R, math.sqrt(R**2.0-1.0)/R]]


def draw_xyz(alpha, N=1000):
    x_value_sub = []
    y_value_sub= []
    z_index = np.random.choice(len(alpha_init), N, p=alpha)
    deviation_values = np.random.randn(N, 2)/2.0
    xval = deviation_values+np.array(mean_vecs)[z_index,:]
    x_value_sub = xval
    y_value_sub = np.sum(np.array(population_decision)[z_index,:]*deviation_values,axis=1)
    return x_value_sub, y_value_sub, z_index




b_add = 50
rep_list = []
coef_reps = []
for replicate in range(10):
    lamb = [1000, 1000]
    alpha = alpha_init
    np.random.seed(replicate)
    x_values = []
    y_values = []
    acc_seq = []
    alpha_seq = []
    coef_seq = []
    icept_seq = []
    for m in tqdm(range(500)):
        Ndraw = np.random.poisson(lamb)
        x_value_sub, y_value_sub, z_index = draw_xyz(alpha, sum(Ndraw))
        x_held, y_held, z_held = draw_xyz([0.5,0.5], 5000)
        model_out = torch_logreg.train(np.array(x_value_sub), np.array(y_value_sub)>0, 2000, 0.0, 0.0, lr=0.01)
        held_eval = robust_class.eval_model(model_out,np.array(x_held)).squeeze()*y_held > 0
        class_acc = [np.mean(held_eval[z_held==zi]) for zi in [0,1]]
        acc_seq.append(class_acc)
        acc_in = np.array(class_acc)/0.85
        acc_in[acc_in>0.99]=0.99
        lamb = lamb*acc_in + b_add
        mod_val = torch_logreg.dump_model(model_out[0])
        coef_seq.append(mod_val[0])
        icept_seq.append(mod_val[1])
        alpha = lamb / np.sum(lamb)
        alpha_seq.append(alpha.copy())
    rep_list.append((acc_seq, alpha_seq))
    coef_reps.append((coef_seq, icept_seq))

plt.figure(figsize=(6,3))
x_held, y_held, z_held = draw_xyz([0.5,0.5])
z_vec = (np.array(y_held) > 0).astype(int)

plt.scatter(np.array(x_held)[z_vec==0,0],
            np.array(x_held)[z_vec==0,1],
            c = 'red', marker='_')
plt.scatter(np.array(x_held)[z_vec==1,0],
            np.array(x_held)[z_vec==1,1],
            c = 'black',marker='+')
for idx in [99,249,499]:
    x_seq = np.linspace(-R,R,100)
    y_seq = -1*(coef_reps[3][0][idx][0][0]*x_seq+coef_reps[3][1][idx])/coef_reps[3][0][idx][0][1]
    plt.plot(x_seq, y_seq, linewidth=3.0)

plt.savefig('exscatter_new.pdf')
plt.close()
    
###
np.random.seed(0)
x_held, y_held, z_held = draw_xyz([0.5,0.5])
minpop_frac = [0.01, 0.05, 0.07, 0.1, 0.15, 0.2, 0.5]
held_values = []
for minfrac_seq in minpop_frac:
    print 'new iter'
    x_value_sub, y_value_sub, z_index = draw_xyz([minfrac_seq, 1.0-minfrac_seq], 10000)
    #
    model_out = torch_logreg.train(np.array(x_value_sub), np.array(y_value_sub)>0, 5000, 0.95, 2.0)
    held_eval = robust_class.eval_model(model_out,np.array(x_held)).squeeze()*y_held > 0
    class_acc = [np.mean(held_eval[z_held==zi]) for zi in [0,1]]
    #
    model_out_2 = torch_logreg.train(np.array(x_value_sub), np.array(y_value_sub)>0, 5000, 0.95, 0.0)
    held_eval_2 = robust_class.eval_model(model_out_2,np.array(x_held)).squeeze()*y_held > 0
    class_acc_2 = [np.mean(held_eval_2[z_held==zi]) for zi in [0,1]]
    t2 = class_acc_2
    print np.min(class_acc)
    print np.min(t2)
    held_values.append([class_acc, t2])



held_min_vals = zip(*[[np.min(hsub) for hsub in held]for held in held_values])
held_max_vals = zip(*[[np.max(hsub) for hsub in held]for held in held_values])
type_names = ['DRO', 'ERM']

cyc=['b','g']

plt.figure(figsize=(4,2))
for i in range(len(type_names)):
    plt.plot(minpop_frac, held_min_vals[i],label=type_names[i], color=cyc[i])

for i in range(len(type_names)):
    plt.plot(minpop_frac, held_max_vals[i], color=cyc[i], linestyle=':')

plt.legend()
plt.xlabel('Minority size')
plt.ylabel('Accuracy')
plt.xlim(0.0,0.2)
plt.ylim(0.5,1.0)
plt.tight_layout()
plt.savefig('fracacc_new.pdf')
plt.close()


######
######
######
b_add = 50
rep_list_2 = []
for replicate in range(10):
    lamb = [1000, 1000]
    alpha = alpha_init
    np.random.seed(replicate)
    x_values = []
    y_values = []
    acc_seq_2 = []
    alpha_seq_2 = []
    coef_seq_2 = []
    icept_seq_2 = []
    for m in tqdm(range(500)):
        Ndraw = np.random.poisson(lamb)
        x_value_sub, y_value_sub, z_index = draw_xyz(alpha, sum(Ndraw))
        x_held, y_held, z_held = draw_xyz([0.5,0.5], 5000)
        model_out = torch_logreg.train(np.array(x_value_sub), np.array(y_value_sub)>0, 2000, 0.95, 2.0, lr=0.1)
        held_eval = robust_class.eval_model(model_out,np.array(x_held)).squeeze()*y_held > 0
        class_acc = [np.mean(held_eval[z_held==zi]) for zi in [0,1]]
        acc_seq_2.append(class_acc)
        acc_in = np.array(class_acc)/0.85
        acc_in[acc_in>0.99]=0.99
        lamb = lamb*acc_in + b_add
        mod_val = torch_logreg.dump_model(model_out[0])
        coef_seq_2.append(mod_val[0])
        icept_seq_2.append(mod_val[1])
        alpha = lamb / np.sum(lamb)
        alpha_seq_2.append(alpha.copy())
    rep_list_2.append((acc_seq_2, alpha_seq_2))

getstats = lambda x: (np.median(np.array(x),axis=1), np.percentile(np.array(x), [25, 75], axis=1))

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(np.array(x), 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

getstats = lambda x: (np.median(np.array(x),axis=0), np.percentile(np.array(x), [10, 90], axis=0))
def get_summaries(replist):
    min_acc = []
    max_acc = []
    min_frac = []
    max_frac = []
    for rep in replist:
        accs = zip(*rep[0])
        frcs = zip(*rep[1])
        min_idx = np.argmin([np.mean(frcs[0]),np.mean(frcs[1])])
        max_idx = np.argmax([np.mean(frcs[0]),np.mean(frcs[1])])
        min_acc.append(running_mean(accs[min_idx], 50))
        max_acc.append(running_mean(accs[max_idx], 50))
        min_frac.append(running_mean(frcs[min_idx], 50))
        max_frac.append(running_mean(frcs[max_idx], 50))
    return getstats(min_acc), getstats(max_acc), getstats(min_frac), getstats(max_frac)

summ_test = get_summaries(rep_list)
summ_test_2 = get_summaries(rep_list_2)

subseq = np.linspace(0,450,50).astype(int)

def make_ivt(lin,subseq):
    return [np.array(lin[0])[subseq] - np.array(lin[1][0])[subseq], np.array(lin[1][1])[subseq]-np.array(lin[0])[subseq]]

plt.figure(figsize=(4,2))
plt.errorbar(subseq, np.array(summ_test[0][0])[subseq], yerr=make_ivt(summ_test[0],subseq), label='ERM', color='green')
plt.errorbar(subseq, np.array(summ_test_2[0][0])[subseq], yerr=make_ivt(summ_test_2[0],subseq), label='DRO',color='blue')
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('minority group accuracy')
plt.tight_layout()
plt.savefig('toy_class_time_acc_new.pdf')

plt.figure(figsize=(4,2))
plt.errorbar(subseq, np.array(summ_test[2][0])[subseq], yerr=make_ivt(summ_test[2],subseq), label='ERM',color='green')
plt.errorbar(subseq, np.array(summ_test_2[2][0])[subseq], yerr=make_ivt(summ_test_2[2],subseq), label='DRO',color='blue')
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('minority group fraction')
plt.tight_layout()
plt.savefig('toy_class_time_frac_new.pdf')


