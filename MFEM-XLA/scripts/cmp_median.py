#!/usr/bin/env python3
# cmp_median.py A_dir B_dir [n]: per-case median wall (over repetitions) B/A, geomean, top/bottom n.
import json,glob,os,sys,statistics as st,math
def load(d):
    r={}
    for f in glob.glob(d+'/bench_*.json'):
        b=os.path.basename(f)[6:-5]
        try: j=json.load(open(f))
        except Exception: continue
        per={}
        for x in j['benchmarks']:
            if x.get('run_type')=='aggregate': continue
            per.setdefault(x['name'],[]).append(x['real_time']*{'ns':1e-9,'us':1e-6,'ms':1e-3,'s':1}[x['time_unit']])
        for k,v in per.items(): r[(b,k)]=st.median(v)
    return r
A=load(sys.argv[1]); B=load(sys.argv[2]); n=int(sys.argv[3]) if len(sys.argv)>3 else 12
rs=sorted((B[k]/A[k],k) for k in A if k in B)
print(len(rs),'cases, median geomean %.4f, faster %d, slower %d'%(math.exp(sum(math.log(r) for r,_ in rs)/len(rs)),sum(r<1 for r,_ in rs),sum(r>1 for r,_ in rs)))
for r,k in rs[:n]+[(None,None)]+rs[-n:]:
    if r is None: print('...'); continue
    print(f"{r:6.3f} {k[0]:22} {k[1]:40} {A[k]*1e6:11.2f} {B[k]*1e6:11.2f}")
