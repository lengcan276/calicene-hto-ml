"""Reproduce Section S8 from the supplied conformer-level tables.
Requires Python 3, numpy and pandas. Run: python reproduce_S8.py
Does not require the original project directory, RDKit, or quantum-chemistry software.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT/'S2_all159_inclusion.csv', float_precision='round_trip').sort_values('analysis_order')
assert len(d) == 159 and not d.duplicated(['parent_identifier','conformer_identifier']).any()
assert np.allclose(d.S1_eV-d.T2_eV, d.gap_S1_T2_eV, atol=1e-12, rtol=0)
raw = pd.read_csv(ROOT/'S2_donor_group_measurements.csv',float_precision='round_trip')
eligible = raw[(raw.kind!='terminal') & (raw.nitro_like==0)]
agg=eligible.groupby(['Molecule','conformer']).twist_deg.mean()
expected=d.set_index(['parent_identifier','conformer_identifier']).donor_twist_deg
for key,value in agg.items():assert np.isclose(value,expected.loc[key],atol=1e-12,rtol=0)
m=d[d.measurable==1].copy()
assert len(m)==137
assert np.array_equal(d.measurable,d.donor_twist_deg.notna().astype(int))
assert np.array_equal(d.included_within_parent,((d.measurable==1)&(d.n_measured_conformers_in_parent>=2)).astype(int))
a=m.groupby('parent_identifier').filter(lambda g:len(g)>=2).copy()
assert len(a)==134 and a.parent_identifier.nunique()==32
for file,frame in [('S2_measured137.csv',m),('S2_within134.csv',a)]:
 saved=pd.read_csv(ROOT/file,float_precision='round_trip')
 assert np.array_equal(saved.analysis_order,frame.analysis_order)
 assert np.allclose(saved.donor_twist_deg,frame.donor_twist_deg,atol=1e-12,rtol=0)
x=(a.donor_twist_deg-a.groupby('parent_identifier').donor_twist_deg.transform('mean')).to_numpy()
y=(a.gap_S1_T2_eV-a.groupby('parent_identifier').gap_S1_T2_eV.transform('mean')).to_numpy()
assert np.allclose(x,a.centered_twist_deg,atol=1e-12,rtol=0)
assert np.allclose(y,a.centered_gap_eV,atol=1e-12,rtol=0)
r=float(np.corrcoef(x,y)[0,1])
indices=[np.flatnonzero(a.parent_identifier.to_numpy()==g) for g in np.unique(a.parent_identifier)]
rng=np.random.default_rng(20260910);exceed=0
for _ in range(5000):
 yp=y.copy()
 for ix in indices:yp[ix]=y[rng.permutation(ix)]
 exceed+=abs(np.corrcoef(x,yp)[0,1])>=abs(r)-1e-12
directions=[]
for parent,g in a.groupby('parent_identifier'):
 if len(g)<3 or g.donor_twist_deg.std()<=0 or g.gap_S1_T2_eV.std()<=0:continue
 rho=float(g.donor_twist_deg.rank().corr(g.gap_S1_T2_eV.rank()))
 directions.append({'parent_identifier':parent,'n':len(g),'spearman_rho':rho,'twist_min_deg':float(g.donor_twist_deg.min()),'twist_max_deg':float(g.donor_twist_deg.max()),'same_direction':bool(np.sign(rho)==np.sign(r))})
assert len(directions)==24 and sum(t['n'] for t in directions)==118
assert sum(t['same_direction'] for t in directions)==19
assert abs(r-0.35594994805708263)<1e-12 and exceed==0
result={'intended':159,'measured':137,'within_parent_conformers':134,'within_parent_parents':32,'pearson_r':r,'permutations':5000,'random_seed':20260910,'exceedances':int(exceed),'two_sided_p':(exceed+1)/5001,'parent_direction_count':24,'positive_direction':19,'opposite_direction':5,'singletons':m[m.n_measured_conformers_in_parent==1][['parent_identifier','conformer_identifier']].to_dict('records'),'numpy_version':np.__version__,'pandas_version':pd.__version__}
(ROOT/'reproduced_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
pd.DataFrame(directions).to_csv(ROOT/'reproduced_Table_S8C.csv',index=False,float_format='%.17g')
print(json.dumps(result,indent=2))
