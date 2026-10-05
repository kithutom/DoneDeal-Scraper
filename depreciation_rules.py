import os, pandas as pd, numpy as np
d=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','carData.csv'),encoding='utf-8-sig')
d=d[d.price_available & d.price_eur.notna() & (d.year<=2026)].copy()
d['age']=2026-d.year
# expected price for age: median by year (smoothed via rolling over neighbouring years)
med=d.groupby('year').price_eur.median().sort_index()
med=np.exp(np.log(med).rolling(3,center=True,min_periods=1).mean())
d['expected']=d.year.map(med)
d['ratio']=d.price_eur/d.expected
d['dep']=pd.cut(d.ratio,[0,0.85,1.15,99],labels=['MoreDep','Normal','LessDep'])
d['value']=pd.qcut(d.price_eur,3,labels=['LowPrice','MidPrice','HighPrice'])
T=d.trim.fillna('').str.upper()
perf=T.str.contains(r'^(?:GTI|R$|GTD|GTE|EDITION-R|R-LINE|GTI )') | T.isin(['R'])
items=pd.DataFrame(index=d.index)
items['Age']=pd.cut(d.age,[-1,3,7,11,99],labels=['0-3y','4-7y','8-11y','12y+']).astype(str)
items['Km']=pd.cut(d.mileage_km,[-1,50e3,100e3,150e3,200e3,1e7],labels=['<50k','50-100k','100-150k','150-200k','200k+']).astype(str)
items['Fuel']=d.fuel_type.replace({'Plug-in Hybrid':'Hybrid'})
items['Trans']=d.transmission.fillna('?')
items['Body']=np.where(d.body_type=='Estate','Estate',np.where(d.body_type=='Hatchback','Hatch','Other'))
items['Owners']=pd.cut(d.previous_owners,[-1,1,3,99],labels=['0-1','2-3','4+']).astype(str)
items['Import']=d.is_import.map({True:'Yes',False:'No'}).fillna('?')
items['Seller']=d.seller_type.fillna('?')
items['Warranty']=d.warranty.map({True:'Yes',False:'No'})
items['NCT']=pd.cut(d.nct_months_remaining,[-99,0,6,99],labels=['expired','<=6m','>6m']).astype(str)
items['Trim']=np.where(perf,'Performance',np.where(T=='','?','Standard'))
items['Dublin']=np.where(d.county=='Dublin','Yes','No')
tx=[{f'{c}={v}' for c,v in r.items() if v not in ('?','nan')} for _,r in items.iterrows()]
def mine(target_series,target,minsup=0.03,maxlen=3):
    N=len(tx); tgt=(target_series==target).values; base=tgt.mean()
    allitems=sorted(set().union(*tx)); out=[]
    sets=[frozenset([i]) for i in allitems]
    sup={}
    for k in range(1,maxlen+1):
        new=[]
        for s in sets:
            m=np.array([s<=t for t in tx]); n=m.sum()
            if n/N<minsup: continue
            new.append(s); nt=(m&tgt).sum()
            if nt/N>=minsup/2:
                conf=nt/n; out.append((' & '.join(sorted(s)),n,round(nt/N,3),round(conf,2),round(conf/base,2)))
        sets=list({a|b for a in new for b in new if len(a|b)==k+1 and len({x.split('=')[0] for x in a|b})==k+1})
    r=pd.DataFrame(out,columns=['antecedent','n','support','conf','lift'])
    return base, r
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',90)
for tgt_s,t in [(d.value,'LowPrice'),(d.dep,'MoreDep'),(d.dep,'LessDep')]:
    base,r=mine(tgt_s,t)
    print(f'\n=== => {t}  (base rate {base:.2f}) ===')
    print(r[(r.lift>=1.4)&(r.n>=25)].sort_values(['lift','support'],ascending=False).head(25).to_string(index=False))
    print('single items:'); print(r[~r.antecedent.str.contains('&')].sort_values('lift',ascending=False).to_string(index=False))
