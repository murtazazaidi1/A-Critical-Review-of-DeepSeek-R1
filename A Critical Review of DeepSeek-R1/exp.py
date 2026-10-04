import numpy as np, json
K, V, P, G, STEPS = 3, 6, 8, 16, 600   # tokens/answer, vocab, prompts, group size, updates

def run(method, lr, seed):
    rng = np.random.default_rng(seed)
    target = rng.integers(0, V, size=(P, K))          # hidden correct answer per prompt
    th = np.zeros((P, K, V)); base = np.zeros(P)
    curve = []
    for t in range(STEPS):
        z = th - th.max(-1, keepdims=True); pr = np.exp(z); pr /= pr.sum(-1, keepdims=True)
        cum = pr.cumsum(-1)
        u = rng.random((P, G, K, 1))
        a = (u > cum[:, None]).sum(-1).clip(0, V - 1)           # sampled tokens [P,G,K]
        r = (a == target[:, None]).all(-1).astype(float)        # rule-based accuracy reward
        curve.append(r.mean())
        if method == "reinforce":   adv = r
        elif method == "baseline":  adv = r - base[:, None]; base += 0.1 * (r.mean(1) - base)
        else:                       adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-6)  # GRPO
        oh = np.eye(V)[a]                                        # [P,G,K,V]
        g = ((oh - pr[:, None]) * adv[:, :, None, None]).mean(1)
        th += lr * g
    return np.array(curve)

def steps_to(c, thr=0.9, w=10):
    s = np.convolve(c, np.ones(w) / w, "valid")
    i = np.where(s >= thr)[0]
    return i[0] + w if len(i) else STEPS

if __name__ == "__main__":
    methods = ["reinforce", "baseline", "grpo"]; grid = [0.1, 0.3, 1, 3, 10, 30]
    best = {}
    for m in methods:                                            # tune lr on seeds 0-9
        sc = {lr: np.mean([steps_to(run(m, lr, s)) for s in range(10)]) for lr in grid}
        best[m] = min(sc, key=sc.get); print(m, sc)
    res = {}
    for m in methods:                                            # evaluate on fresh seeds 100-129
        cs = np.array([run(m, best[m], s) for s in range(100, 130)])
        st = np.array([steps_to(c) for c in cs])
        res[m] = dict(lr=best[m], mean_curve=cs.mean(0).tolist(), sd_curve=cs.std(0).tolist(),
                      steps_mean=float(st.mean()), steps_sd=float(st.std()),
                      steps_median=float(np.median(st)), final=float(cs[:, -50:].mean()),
                      failed=int((st >= STEPS).sum()), early=float(cs[:, :50].mean()))
        print(m, {k: v for k, v in res[m].items() if "curve" not in k})
    json.dump(res, open("res.json", "w"))
