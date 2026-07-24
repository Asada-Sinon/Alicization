# Question: In the 6 formal OFF (neutral-drift) runs, does carn defence > herb defence
# hold (as seen in the throwaway seed99 OFF run: carn_armor>herb_armor, carn_spike>herb_spike)?
# This tests whether P3's premise (carn genes are just drifting) is being lifted by
# founder/linkage rather than selection. Reads the same 12 logs.
import json, numpy as np
BASE="/home/michael/workspace/pi05/temp/alicization/outputs/20260724-defence-evo"
def load(arm,s):
    l=[x for x in open(f"{BASE}/{arm}_seed{s}.log") if x.startswith("JSON ")][-1]
    return json.loads(l[5:])
off={s:load("off",s) for s in range(6)}
on ={s:load("on", s) for s in range(6)}
print("OFF runs: carn vs herb defence (drift arm) ---")
for g in ["armor","spike"]:
    ch=np.array([off[s][f"carn_{g}"] for s in range(6)])
    hh=np.array([off[s][f"herb_{g}"] for s in range(6)])
    ndir=int((ch>hh).sum())
    print(f"  {g}: OFF carn mean={ch.mean():.4f}  herb mean={hh.mean():.4f}  n(carn>herb)={ndir}/6")
    for s in range(6):
        print(f"     seed{s}: carn={ch[s]:.4f} herb={hh[s]:.4f} {'carn>herb' if ch[s]>hh[s] else 'carn<=herb'}")
print("\nON runs: carn vs herb armor (selection arm) ---")
ch=np.array([on[s]["carn_armor"] for s in range(6)]); hh=np.array([on[s]["herb_armor"] for s in range(6)])
print(f"  armor: ON carn mean={ch.mean():.4f}  herb mean={hh.mean():.4f}  n(herb>carn)={int((hh>ch).sum())}/6")
