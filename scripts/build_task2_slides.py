"""Insert the Task 2 (rank prediction) section into slides/STAT5011_Final_Presentation.pptx (python-pptx).
Run once on the Task-1-only deck (22 slides); the result has 30 slides. Source numbers: results/task2_scores.csv."""
from pathlib import Path
import csv,sys,copy
from pptx import Presentation
from pptx.util import Emu,Pt
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE,XL_LABEL_POSITION
R=Path(__file__).resolve().parents[1];PP=R/'slides/STAT5011_Final_Presentation.pptx'
P=Presentation(PP);assert len(P.slides)==22,'already built'
px=lambda v:Emu(int(v*9525))
C=dict(bg='F7F5F0',ink='102B35',blue='1C6070',teal='378B80',orange='B3693B',muted='536F78')
rgb=lambda h:RGBColor.from_string(h)
rows=list(csv.DictReader((R/'results/task2_scores.csv').open()))
def tb(s,t,x,y,w,h,size,color='ink',bold=False):
    a=s.shapes.add_textbox(px(x),px(y),px(w),px(h));tf=a.text_frame;tf.word_wrap=True
    for i,line in enumerate(t.split('\n')):
        para=tf.paragraphs[0] if i==0 else tf.add_paragraph();r=para.add_run();r.text=line;f=r.font;f.name='Arial';f.size=Pt(size);f.bold=bold;f.color.rgb=rgb(C.get(color,color))
    return a
def slide(title,sub,note):
    s=P.slides.add_slide(P.slide_layouts[[l.name for l in P.slide_layouts].index('Blank')]);s.background.fill.solid();s.background.fill.fore_color.rgb=rgb(C['bg'])
    tb(s,title,64,42,1150,85,32.25,'ink',True);tb(s,sub,68,125,1130,64,18.75,'muted');tb(s,'00',1160,671,55,25,12,'muted');s.notes_slide.notes_text_frame.text=note;return s
def foot(s,t):tb(s,t,68,642,1070,40,12.75,'muted')
def two(s,h1,b1,h2,b2):
    tb(s,h1,75,215,510,55,24,'blue',True);tb(s,b1,75,292,520,300,21.75);tb(s,h2,680,215,510,55,24,'blue',True);tb(s,b2,680,292,520,300,21.75)
def tab(s,head,body,x,y,w,h,size=17,widths=None):
    t=s.shapes.add_table(len(body)+1,len(head),px(x),px(y),px(w),px(h)).table
    for j,hh in enumerate(head):
        c=t.cell(0,j);c.text=hh;c.fill.solid();c.fill.fore_color.rgb=rgb(C['ink'])
    for i,r in enumerate(body,1):
        for j,v in enumerate(r):
            c=t.cell(i,j);c.text=v;c.fill.solid();c.fill.fore_color.rgb=rgb('FFFFFF' if i%2 else 'EBEFED')
    for i in range(len(body)+1):
        for j in range(len(head)):
            for para in t.cell(i,j).text_frame.paragraphs:
                for r in para.runs:r.font.name='Arial';r.font.size=Pt(size);r.font.bold=(i==0);r.font.color.rgb=rgb('FFFFFF' if i==0 else C['ink'])
    if widths:
        for j,wd in enumerate(widths):t.columns[j].width=px(wd)
def chart(s,kind,cats,vals,x,y,w,h,lo,hi,labels='0.0000'):
    cd=CategoryChartData();cd.categories=cats;cd.add_series('Website score',vals)
    gf=s.shapes.add_chart(kind,px(x),px(y),px(w),px(h),cd);ch=gf.chart;ch.has_legend=False;ch.has_title=False
    pl=ch.plots[0];pl.has_data_labels=True;dl=pl.data_labels;dl.number_format=labels;dl.number_format_is_linked=False;dl.font.size=Pt(14);dl.font.name='Arial'
    ser=pl.series[0];
    if kind==XL_CHART_TYPE.LINE_MARKERS:
        ser.format.line.color.rgb=rgb(C['blue']);ser.format.line.width=Pt(3);ser.smooth=False;ser.marker.format.fill.solid();ser.marker.format.fill.fore_color.rgb=rgb(C['blue']);dl.position=XL_LABEL_POSITION.ABOVE
    else:
        ser.format.fill.solid();ser.format.fill.fore_color.rgb=rgb(C['blue']);dl.position=XL_LABEL_POSITION.OUTSIDE_END;pl.gap_width=60
    va=ch.value_axis;va.minimum_scale=lo;va.maximum_scale=hi;va.tick_labels.font.size=Pt(13);va.major_gridlines.format.line.color.rgb=rgb('DDE5E6');va.format.line.fill.background()
    ch.category_axis.tick_labels.font.size=Pt(13);return ch
src='Source: results/task2_scores.csv; task2/REPRODUCE_TASK2_BEST.md. Website scores are public interim results.'
new=[]
s=slide('Task 2: predicting player rank','Same game records, a different target: an ordinal class from a bag of games','Task 2 of the competition. Each question is a bag of 5, 10, 15 or 20 games by one player; we output one of ten ordered rank classes. Exact class scores 1, adjacent class e^-1. Because the classes are ordered, both the loss and the decision rule should use that order. Same 1,000,000 training games; 800 test questions.');new.append(s)
two(s,'Input and output','A bag of 5, 10, 15 or 20 games\nof one player\nTen ordered classes:\nD, C, B, A, 1D, 2D ... 6D\n800 test questions','Scoring [9]','Exact class: 1\nAdjacent class: e^-1 = 0.3679\nOrder matters, so use an\nordinal loss and decision rule')
foot(s,'Official Task 2 topic page [9]; training rank labels come from the same 1,000,000-game collection.')
s=slide('Task 2 system','Two model families produce log-probabilities over the ten classes','Six bag-level fine-tuned language models and fourteen HumanSL sequence Transformers. Log-probabilities are averaged inside each family, blended 1 to 2, and decoded with the expected-score rule.');new.append(s)
s.shapes.add_picture(str(R/'figures/task2_pipeline.png'),px(100),px(210),width=px(1080));foot(s,'Blend weights 1 (bag LMs) and 2 (HumanSL sequences) were fixed on 800 held-out players; no tuning on the private split.')
s=slide('Ordinal training and decoding','Soft labels over neighbouring classes; decide by expected score','SORD gives each class a soft target that decays with distance from the true class, so the model is penalised less for near misses. The decision then maximises expected competition score, which favours a central class when the posterior is spread over neighbours.');new.append(s)
two(s,'SORD soft-label loss [10]','q_k ∝ exp(−|k − y| / τ)\ncross-entropy between q and\nthe predicted class distribution\nBag: mean of per-game pooled\nstates, one head per bag','Expected-score decision','rank = argmax_k Σ_j e^(−|j−k|) P(j)\nWith spread-out P(j) this picks\na class between the plausible ones\nlog P = 1·mean(bag) + 2·mean(seq)')
foot(s,'Diaz and Marathe, Soft Labels for Ordinal Regression, CVPR 2019.')
s=slide('HumanSL sequence features','The frozen human-move network, read as a per-move strength signal','For every move of a game we ask the frozen KataGo HumanSL network how likely that move was under eight rank profiles, plus entropy, top-1 agreement, win rate and score lead. A small Transformer reads the 250-ply feature sequence, is pooled over the bag and trained with SORD. Fourteen seeds and data subsets are averaged.');new.append(s)
two(s,'Per move (27 features)','Played-move log-probability\nunder 8 profiles, 20k ... 8d\nPolicy entropy, top-1 agreement\nWin rate and score lead','Sequence model','3-layer Transformer, width 192\nUp to 250 plies per game\nBag mean, SORD head\n14 models, 80 epochs each')
foot(s,'KataGo b18c384nbt-humanv0 [5] is an external, frozen network; run in PyTorch with KataGo sources.')
prog=[r for r in rows if r['group']=='progress']
s=slide('Task 2 score by development stage','HumanSL sequence features gave the largest single gain','Website score by stage: language model alone 0.3976, SORD 0.6236, HumanSL gradient boosting 0.6915 and 0.7006, bag models plus HumanSL sequences on 100 plies 0.7248, 250 plies 0.7676, three extra 350-ply models 0.7683. The jump from 0.7248 to 0.7676 is the main finding.');new.append(s)
chart(s,XL_CHART_TYPE.LINE_MARKERS,['LM only','SORD LM','+ GBM','+ GBM v2','+ bag LMs, seq (100 plies)','250 plies','+ 350-ply models'],[float(r['website_score']) for r in prog],68,190,1140,440,.3,.85)
foot(s,src)
s=slide('More plies of HumanSL features','Late-game moves carry rank information; beyond 250 plies the gain is within noise','Re-extracting HumanSL features for plies 101-250 improved a matched held-out gradient-boosted model from 0.6859 to 0.7083 and the website blend from 0.7248 to 0.7676. Plies 251-350 showed no clear gain in a matched test (0.6989 vs 0.7039 validation) and a 0.0007 website gain, so we treat them as ensemble diversity, not a finding.');new.append(s)
tab(s,['HumanSL plies','Matched held-out check','Website blend'],[['1 - 100','GBM 0.6859','0.7248*'],['1 - 250','GBM 0.7083','0.7676'],['1 - 350','Seq. val 0.6989 vs 0.7039 (250)','0.7683']],68,215,1140,230,20,[260,620,260])
tb(s,'* rank_hsl7 used 100-ply features with fewer models; website blends differ in other components, so the gain is not attributed to plies alone.',68,480,1140,70,17,'muted')
foot(s,src)
ab=[r for r in rows if r['group']=='ablation']
s=slide('Task 2 blend ablations','Bag term helps, extra fine-tuned LM hurts, seeds saturate','Eight variants uploaded on November 21-22. Removing the bag models costs 0.0157, adding a fine-tuned game-level LM costs 0.0212, and seven, eleven and fourteen sequence models differ by under 0.002, so seeds are saturating.');new.append(s)
chart(s,XL_CHART_TYPE.BAR_CLUSTERED,[r['label'].split(': ',1)[1] if ': ' in r['label'] else r['label'] for r in ab][::-1],[float(r['website_score']) for r in ab][::-1],68,190,1140,440,.74,.775)
foot(s,src+' Axis truncated.')
s=slide('Task 2 lessons and limits','Interim result: 0.7683, rank 15 of 165 on 22 November','Best file e13 scored 0.7683, rank 15 of 165 on 22 November. Local held-out differences below 0.005 did not predict the website order, so we report only public scores. Reproduction: artifacts for CPU byte-identical regeneration, scripts and seeds are in the repository; GPU retraining gives a close but not identical file. The external HumanSL network and official data are required.');new.append(s)
two(s,'What we measured','Best public score 0.7683\n(e13, 14 sequence models)\nBag term: −0.0157 if removed\nFine-tuned game LM: −0.0212 if added','Limits','Local gains < 0.005 did not transfer\nPublic score only; private pending\nExternal frozen HumanSL network\nCPU regeneration is byte-identical')
foot(s,'Submitted file e13 SHA-256 09c1a5a5dce8eef3...bb59d0d (results/task2_best_prediction.csv); recipe in task2/REPRODUCE_TASK2_BEST.md.')
# reorder: new slides after slide 18
lst=P.slides._sldIdLst;ids=list(lst);newids=ids[22:];old=ids[:22]
for e in ids:lst.remove(e)
for e in old[:18]+newids+old[18:]:lst.append(e)
# edits
sl=list(P.slides)
def settext(shape,t):
    r=shape.text_frame.paragraphs[0].runs;r[0].text=t
    for x in r[1:]:x.text=''
for i,s in enumerate(sl,1):
    for sh in s.shapes:
        if sh.has_text_frame and sh.left>px(1100) and sh.top>px(650) and sh.text_frame.text.strip().isdigit():settext(sh,f'{i:02d}')
t=sl[0]
for sh in t.shapes:
    if sh.has_text_frame:
        tx=sh.text_frame.text
        if tx.startswith('Identifying Go Players'):
            ps=sh.text_frame.paragraphs;ps[0].runs[0].text='Identifying Go Players and'
            if len(ps)>1:ps[1].runs[0].text='Predicting Their Rank'
            else:
                for x in ps[0].runs[1:]:x.text=''
        if tx.startswith('AI CUP 2026 Task 1'):settext(sh,'AI CUP 2026 Tasks 1 and 2  |  Evidence through 22 November 2026')
c=sl[-3];assert c.shapes[0].text_frame.text=='Conclusions',c.shapes[0].text_frame.text
for sh in c.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith('Achieved'):settext(sh,'Task 1: 0.8686865   Task 2: 0.7683 (public)')
    if sh.has_text_frame and sh.text_frame.text.startswith('Coordinate Transformers'):
        ps=sh.text_frame.paragraphs;newp=copy.deepcopy(ps[-1]._p);ps[-1]._p.addnext(newp)
        ps=sh.text_frame.paragraphs;ps[-1].runs[0].text='HumanSL sequences over 250 plies drove rank prediction (0.7248 to 0.7676).'
        for x in ps[-1].runs[1:]:x.text=''
rf=sl[-2];assert rf.shapes[0].text_frame.text=='References'
last=[sh for sh in rf.shapes if sh.has_text_frame and sh.text_frame.text.startswith('[8]')][0]
for k,(txt,top) in enumerate([('[9] AIdea / NDHU — Task 2 official topic page (rank prediction)',589),('[10] Diaz and Marathe — Soft Labels for Ordinal Regression (SORD), CVPR 2019',637)]):
    el=copy.deepcopy(last._element);rf.shapes._spTree.append(el);sh=rf.shapes[-1];sh.top=px(top);settext(sh,txt)
P.save(PP);print('slides',len(P.slides))
