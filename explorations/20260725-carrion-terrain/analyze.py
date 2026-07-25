# 问题：腐食 v2 的 carn_frac 正效应（ON>OFF）是否跨地形稳健，还是绑当前河系？
# 读入：outputs/20260725-carrion-terrain/results.jsonl（48 行，4 地形×ON/OFF×seed0-5）
# 每对臂仅 carrion_enabled 差异（overrides 已核）。地形=复制单元，founder seed=伪复制。
# 输出：逐地形配对差(carn_frac末态+late_carn+population)、Wilcoxon p、效应量、同向数；
#       跨地形符号一致性；wn2_b0.40 反向诊断（death 分因）；合并检验。
import json, numpy as np
from scipy.stats import wilcoxon

rows=[json.loads(l) for l in open('outputs/20260725-carrion-terrain/results.jsonl')]
def key(r): return (r['overrides']['ridge_wavenumber'], r['overrides']['ridge_base_frac'])
terrains=sorted({key(r) for r in rows})
def get(terr,on,seed,field):
    for r in rows:
        if key(r)==terr and r['overrides']['carrion_enabled']==on and r['seed']==seed:
            return r[field]
    return None

def paired(terr, field):
    on=np.array([get(terr,True,s,field) for s in range(6)])
    off=np.array([get(terr,False,s,field) for s in range(6)])
    return on,off

def boot_ci(d, n=10000, seed=1):
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(d),size=(n,len(d)))
    means=d[idx].mean(1)
    return np.percentile(means,[2.5,97.5])

def rank_r(on,off):
    d=on-off
    d=d[d!=0]
    if len(d)==0: return float('nan')
    try:
        w=wilcoxon(on,off)
        # rank-biserial for wilcoxon
        from scipy.stats import rankdata
        r=rankdata(np.abs(d))
        rp=r[d>0].sum(); rn=r[d<0].sum()
        return (rp-rn)/r.sum()
    except Exception:
        return float('nan')

def wilc_p(on,off):
    d=on-off
    if np.all(d==0): return float('nan')
    try: return wilcoxon(on,off).pvalue
    except Exception: return float('nan')

print("="*70)
for field in ['carnivore_frac','late_carn','population']:
    print(f"\n### FIELD: {field}")
    print(f"{'terrain':<14}{'ON_mean':>9}{'OFF_mean':>9}{'diff':>9}{'CI_lo':>8}{'CI_hi':>8}{'sameON>OFF':>11}{'wilc_p':>9}{'rbr':>7}")
    all_diffs={}
    for terr in terrains:
        on,off=paired(terr,field)
        d=on-off
        all_diffs[terr]=d
        ci=boot_ci(d)
        same=int((d>0).sum()) if field!='population' else int((d<0).sum())  # for pop, cost = ON<OFF
        p=wilc_p(on,off); r=rank_r(on,off)
        tag=f"wn{terr[0]}_b{terr[1]:.2f}"
        print(f"{tag:<14}{on.mean():>9.4f}{off.mean():>9.4f}{d.mean():>+9.4f}{ci[0]:>+8.4f}{ci[1]:>+8.4f}{same:>9}/6{p:>9.4f}{r:>+7.2f}")
    # cross-terrain sign test on per-terrain mean diff
    md=np.array([all_diffs[t].mean() for t in terrains])
    if field=='population':
        print(f"  跨地形: {int((md<0).sum())}/4 地形 ON<OFF(代价方向); 均值diff所在: {md.round(4)}")
    else:
        print(f"  跨地形: {int((md>0).sum())}/4 地形 ON>OFF(正效应方向); 地形均值diff: {md.round(4)}")

print("\n" + "="*70)
print("### wn2_b0.40 反向诊断")
terr=(2,0.4)
for field in ['carnivore_frac','late_carn','population','min_pop','death_predation_frac','death_starvation_frac','death_thirst_frac','carrion_total','hunt_success','mean_energy']:
    on,off=paired(terr,field)
    print(f"  {field:<22} ON={on.mean():>9.4f}  OFF={off.mean():>9.4f}  diff={on.mean()-off.mean():>+9.4f}")

print("\n### 各地形 baseline(OFF) carn_frac 排序 + carrion_total(ON)")
for terr in terrains:
    off=paired(terr,'carnivore_frac')[1]
    conON=paired(terr,'carrion_total')[0]
    tag=f"wn{terr[0]}_b{terr[1]:.2f}"
    print(f"  {tag:<14} OFF carn_frac={off.mean():.4f}   ON carrion_total={conON.mean():.1f}")

print("\n" + "="*70)
print("### 合并检验(地形为复制单元, seed 伪复制)")
# A) 跨 4 地形的 per-terrain 均值 diff 符号检验(n=4, 太小, 只报方向)
for field in ['carnivore_frac','late_carn','population']:
    md=np.array([paired(t,field)[0].mean()-paired(t,field)[1].mean() for t in terrains])
    print(f"  {field}: 4地形均值diff={md.round(4)}  正向{int((md>0).sum())}/4")
# B) pool 全 24 配对(标注伪复制)
print("\n  pooled 24 配对(founder seed 跨4地形混合, 伪复制警告):")
for field in ['carnivore_frac','late_carn','population']:
    on=[];off=[]
    for terr in terrains:
        o,f=paired(terr,field); on+=list(o); off+=list(f)
    on=np.array(on);off=np.array(off);d=on-off
    p=wilc_p(on,off);r=rank_r(on,off);ci=boot_ci(d)
    print(f"    {field:<16} diff={d.mean():>+8.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] 正向{int((d>0).sum())}/24 wilc_p={p:.4f} rbr={r:+.2f}")

print("\n### 跨5套河系符号(含默认地形18配对: late_carn 16/18 p=0.0034 ON>OFF, pop -5.6% p=0.024)")
print("  默认地形 late_carn/carn_frac 方向: ON>OFF (已判正)")
for terr in terrains:
    d=paired(terr,'late_carn')[0]-paired(terr,'late_carn')[1]
    tag=f"wn{terr[0]}_b{terr[1]:.2f}"
    print(f"  {tag:<14} late_carn 地形均值diff={d.mean():+.4f}  {'ON>OFF' if d.mean()>0 else 'ON<OFF'}")
