import importlib.util, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load(name,rel):
    spec=importlib.util.spec_from_file_location(name,ROOT/rel)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

a=load("ana","code/analyze_current_llm.py")

def test_geom6():
    assert abs(a.geom6([1,1,1,1,1,1])-1)<1e-12
    assert a.geom6([1,1,1,1,1,0])==0

def test_majority_and_dist():
    assert a.majority([0,0,1])==0
    d=a.empirical_dist([0,1,1])
    assert np.allclose(d,[1/3,2/3,0])

def test_association_table_schema():
    df=pd.DataFrame({"BBQ_ambig_bias":[1,2,3,4],"E_EA_proxy":[.2,.3,.4,.5]})
    out=a.association_tests(df)
    assert set(out["test"])=={"Spearman","Pearson"}
    assert (out["n"]==4).all()
