import numpy as np, json
from exp import run, steps_to, STEPS
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
M = ["reinforce","baseline","grpo"]; N = {"reinforce":"REINFORCE (no baseline)","baseline":"REINFORCE + value baseline","grpo":"GRPO (group-relative)"}
L = [0.1,0.3,1,3,10]; S = range(100,130)
R = {}
for m in M:
    for lr in L:
        cs = np.array([run(m,lr,s) for s in S]); st = np.array([steps_to(c) for c in cs])
        R[(m,lr)] = dict(mean=st.mean(), sd=st.std(), med=np.median(st), fail=int((st>=STEPS).sum()), curve=cs.mean(0))
        print(m,lr,round(st.mean(),1),round(st.std(),1),int((st>=STEPS).sum()))
json.dump({f"{m}|{lr}":{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in d.items() if k!='curve'} for (m,lr),d in R.items()}, open("sweep.json","w"))
col = {"reinforce":"#C0504D","baseline":"#8C8C8C","grpo":"#1F3A5F"}
fig,ax = plt.subplots(1,2,figsize=(9,3.4),dpi=200)
for m in M: ax[0].plot(R[(m,0.3)]["curve"],label=N[m],color=col[m],lw=1.8)
ax[0].set(xlabel="Training step",ylabel="Mean success rate",title="Learning curves (learning rate 0.3)",ylim=(0,1.02))
ax[0].legend(fontsize=7,frameon=False,loc="lower right")
for m in M: ax[1].errorbar(L,[R[(m,l)]["mean"] for l in L],yerr=[R[(m,l)]["sd"]/np.sqrt(30)*1.96 for l in L],marker="o",ms=4,capsize=2,label=N[m],color=col[m],lw=1.5)
ax[1].set(xscale="log",xlabel="Learning rate",ylabel="Steps to 90% success",title="Sensitivity to learning rate"); 
for a in ax: a.spines[["top","right"]].set_visible(False); a.tick_params(labelsize=8)
plt.tight_layout(); plt.savefig("fig.png")
