"""Task 2 (rank prediction) figures from results/task2_scores.csv; same style as build_figures.py."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parents[1];F=R/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#17343e','text.color':'#17343e','axes.titleweight':'bold','savefig.facecolor':'white'})
blue='#1c6070';teal='#378b80';orange='#b3693b';gray='#73878d'
def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(F/(name+'.svg'),bbox_inches='tight');plt.close(fig)
rows=list(csv.DictReader((R/'results/task2_scores.csv').open()))
prog=[r for r in rows if r['group']=='progress'];abl=[r for r in rows if r['group']=='ablation']
# 1 progress
fig,ax=plt.subplots(figsize=(8.8,3.1));x=range(len(prog));v=[float(r['website_score']) for r in prog]
ax.plot(x,v,color=blue,lw=2,marker='o',ms=7);
for i,(r,s) in enumerate(zip(prog,v)):ax.annotate(f'{s:.4f}',(i,s),textcoords='offset points',xytext=(18 if i==0 else 0,-16 if i==0 else 9),ha='center',fontsize=9.5)
ax.set_xticks(list(x));import textwrap;ax.set_xticklabels([textwrap.fill(r['label'],14) for r in prog],fontsize=8.6);ax.set_ylim(.35,.84);ax.set_ylabel('Website score');ax.grid(axis='y',color='#dde5e6',lw=.8);ax.set_axisbelow(True)
save(fig,'task2_progress')
# 2 ablation
fig,ax=plt.subplots(figsize=(8.6,3.0));ab=sorted(abl,key=lambda r:-float(r['website_score']));names=[r['label'] for r in ab];vals=[float(r['website_score']) for r in ab]
cols=[teal if i==0 else (orange if r['tag'] in ('e9','e10') else blue) for i,r in enumerate(ab)]
b=ax.barh(range(len(ab)),vals,color=cols,height=.62);ax.set_yticks(range(len(ab)));ax.set_yticklabels(names,fontsize=9.5);ax.invert_yaxis();ax.set_xlim(.74,.772);ax.bar_label(b,labels=[f'{s:.4f}' for s in vals],padding=3,fontsize=9.5);ax.set_xlabel('Website score (axis truncated)')
save(fig,'task2_ablation')
# 3 pipeline
fig,ax=plt.subplots(figsize=(10,3.0));ax.axis('off');ax.set(xlim=(-.02,1.02),ylim=(-.14,1.02))
def box(x,y,w,h,t,c=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.008',facecolor='white',edgecolor=c,linewidth=1.5));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
def arr(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.5,color=gray))
box(.00,.38,.13,.28,'Bag of 5-20\ngames of one\nplayer');box(.21,.68,.27,.27,'Rank-conditioned move LM\n+ bag-level SORD head\n(6 models)',blue);box(.21,.02,.27,.33,'Frozen HumanSL, 8 profiles\nper-move features\nbag-pooled Transformer\n(14 models)',teal)
box(.56,.38,.25,.28,'1 x mean log-prob (bag)\n+ 2 x mean log-prob (seq)',orange);box(.87,.38,.13,.28,'Expected-\nscore decode\nD ... 6D')
arr((.13,.55),(.21,.78));arr((.13,.49),(.21,.22));arr((.48,.78),(.56,.60));arr((.48,.22),(.56,.44));arr((.80,.52),(.87,.52))
ax.text(0,-.12,'Task 2 system; log-probabilities over ten ordered rank classes are averaged within each family, then blended.',fontsize=9,color=gray)
save(fig,'task2_pipeline')
print('task2 figures built')
