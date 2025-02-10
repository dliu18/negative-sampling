Commands to generate all figures:

# Summary tables 

```
cd gen_figs/
python metric-summary.py
```

# Performance by Graph Property

```
python performance-vs-graph-feature.py
```

# Performance by Node Property

```
./scripts/post-process.sh line
./scripts/post-process.sh n2v
```

# Efficiency of Repulsion-less approach 
```
cd gen_figs/
python time-delta.py ../../outputs/kdd25/metric_summary.csv ../../figs/kdd25/time-delta.pdf
```

# SBM Series 

```
 python sbm-clustering.py --base_model=line
 python sbm-clustering.py --base_model=n2v
```