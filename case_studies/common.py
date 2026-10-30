"""Shared plotting style and output paths for the Brightwell's case studies."""
import numpy as np, json, matplotlib, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'figures').mkdir(exist_ok=True); (ROOT / 'results').mkdir(exist_ok=True)
matplotlib.use('Agg'); import matplotlib.pyplot as plt
# Book palette: forest, coral, gold, charcoal (see the design reference)
G='#0b5d3f';GD='#073d2b';A='#d85a35';B='#b58a32';T='#222b26';M='#4a544e';L='#cccccc';GR='#9a9a9a'
plt.rcParams.update({'font.family':['Source Sans 3','Inter','DejaVu Sans'],'font.size':7.5,'axes.facecolor':'white','figure.facecolor':'white','axes.edgecolor':L,'axes.labelcolor':M,
 'xtick.color':M,'ytick.color':M,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.titlesize':8,'axes.titleweight':'bold','axes.titlecolor':T,'axes.titlelocation':'left','legend.frameon':False,'legend.fontsize':6.5})
def fig(w=5.6,h=2.7): return plt.subplots(figsize=(w,h))
def save(f,name): f.tight_layout(pad=.4); f.savefig(ROOT / 'figures' / f'{name}.svg'); plt.close(f)
def dump(name,d): json.dump(d,open(ROOT / 'results' / f'{name}.json','w'),indent=1,default=float)
