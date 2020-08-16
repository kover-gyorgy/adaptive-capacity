# -*- coding: utf-8 -*-

import uuid
import os
import flask
import pathlib as pl
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_table.Format import Format, Scheme, Sign, Symbol
import dash_table
import plotly.graph_objs as go
import plotly.colors as col
import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
import math
import matplotlib
import matplotlib.cm
import scipy
from scipy.integrate import solve_ivp
from dash.dependencies import Input, Output, State
import zipfile
from os.path import basename


app = dash.Dash(__name__, external_scripts=[
  'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML',
]
)

table_header_style = {
    "backgroundColor": "rgb(2,21,70)",
    "color": "white",
    "textAlign": "center",
}

APP_PATH = str(pl.Path(__file__).parent.resolve())


downloadable_BCSDataSet_csv    = "downloadable/BCSDataSet.txt"
downloadable_BCSDataSet_zip    = "downloadable/BCSDataSet.zip"

downloadable_BCSData_csv    = "downloadable/BCSData.txt"
downloadable_parameters_csv = "downloadable/parameters.txt"
downloadable_BCSSim_csv     = "downloadable/BCSSim.txt"
downloadable_BCSDataSim_zip = "downloadable/BCSDataSim.zip"
downloadable_BCSDataSim_Nozip = "downloadable/BCSDataSim"
downloadable_BCSDataSim_zipRand = downloadable_BCSDataSim_zip

downloadable_groupCorrMatrix_csv     = "downloadable/corrMatrix.txt"       
downloadable_groupParameters_csv     = "downloadable/corrParameters.txt"
downloadable_BiVar_zip = "downloadable/BiVar.zip"
downloadable_BiVar_Nozip = "downloadable/BiVar"
downloadable_BiVar_zipRand = downloadable_BiVar_zip

downloadable_descStat_csv     = "downloadable/descrStat.txt"   
downloadable_descParameters_csv     = "downloadable/descrParameters.txt"
downloadable_Descr_zip = "downloadable/DescrStat.zip"
downloadable_Descr_Nozip = "downloadable/DescrStat"
downloadable_Descr_zipRand = downloadable_Descr_zip

pourcorr3 = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "pourcorr3.csv")))
fichierfinal = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "fichierfinal.csv")))
pkdata = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "Mydata.csv")))
paramCor = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "param.cor.csv")))
char2    = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "char2.csv")))
Mydata = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "Mydata.csv")))
paramCor['DT1'] = paramCor['te1']-paramCor['tb1']
paramCor['DT2'] = paramCor['te2']-paramCor['tb2']
paramCor['DT3'] = paramCor['te3']-paramCor['tb3']

paramCor_Orig = paramCor.copy()
paramCorBiVar = paramCor.copy()
paramCorHist  = paramCor.copy()
 
pkdataNew = pkdata[['Ewe','BCS.time','Year','Parity','Age_mating','LS','BCS1']].copy()
pkdataNew.columns = ['ID','BCS_stage','Year','Parity','Age_at_first_mating','Litter_size','BCS'] 
pkdataNew.to_csv(downloadable_BCSDataSet_csv,index=False)
with zipfile.ZipFile(downloadable_BCSDataSet_zip, 'w') as zipMe:        
    zipMe.write(downloadable_BCSDataSet_csv,basename(downloadable_BCSDataSet_csv), compress_type=zipfile.ZIP_DEFLATED)


#days of measures for p1, p2 and p3 
d_P1 = [0,40,80,160,190,230,250,310,330]
d_P2 = [x+(2-1)*360 for x in d_P1]
d_P3 = [x+(3-1)*360 for x in d_P1]

tb2N=d_P1[7]+150
tb3N=d_P2[7]+150
tb1 =0
tb2 =0
tb3 =0
te1 =0
te2 =0
te3 =0 
BCSm =0
PMax =0

def TimePosBis(Time, t1,t2): 
  if (Time-t1>=0) and (Time-t2<=0):
      return(1)
  else:
      return(0)


def TimePosTer(Time, t1):
  if (Time-t1>=0):
      return(1)
  else:
      return(0)
 
def BCSModel2(Time, State, kb1, kb2, kb3, kp1, kp2, kp3): 
    global tb1, tb2, tb3, te1, te2, te3, BCSm, Pmax
    BCS1,BCS2,BCS3,P1,P2,P3,BCS = State
    if math.isnan(tb1):  #absence of 1st pertrubation
        tb1=0
        te1=0
    if math.isnan(tb2):  #absence of 2nd pertrubation
        tb2=tb2N
        te2=0
    if math.isnan(tb3):  #absence of 3d pertrubation
        kp3=0
        tb3=tb3N
        te3=0
        kb3=0  
    dBCS1 =  kb1*(BCSm -BCS1)*TimePosBis(Time, tb1,tb2)- kp1*(Pmax - P1)*TimePosBis(Time, tb1,te1)
    dBCS2 =  kb2*(BCSm -BCS2)*TimePosBis(Time, tb2,tb3)- kp2*(Pmax - P2)*TimePosBis(Time, tb2,te2)
    dBCS3 =  kb3*(BCSm -BCS3)*TimePosTer(Time, tb3)- kp3*(Pmax - P3)*TimePosBis(Time, tb3,te3)    
    dP1   =  kp1*(Pmax - P1)*TimePosBis(Time, tb1,te1)
    dP2   =  kp2*(Pmax - P2)*TimePosBis(Time, tb2,te2)
    dP3   =  kp3*(Pmax - P3)*TimePosBis(Time, tb3,te3)    
    dBCS  = (kb1*(BCSm -BCS1)*TimePosBis(Time, tb1,tb2)- kp1*(Pmax - P1)*TimePosBis(Time, tb1,te1) + 
             kb2*(BCSm -BCS2)*TimePosBis(Time, tb2,tb3) - kp2*(Pmax - P2)*TimePosBis(Time, tb2,te2)+ 
             kb3*(BCSm -BCS3)*TimePosTer(Time, tb3)- kp3*(Pmax - P3)*TimePosBis(Time, tb3,te3))
    
    return([ dBCS1,dBCS2,dBCS3,dP1,dP2,dP3,dBCS])
    
def descriptives():    
    data = [['Mean',paramCorHist.kb1.mean()*1000,paramCorHist.kb2.mean()*1000,paramCorHist.kb3.mean()*1000,paramCorHist.kp1.mean()*1000,paramCorHist.kp2.mean()*1000,paramCorHist.kp3.mean()*1000,paramCorHist.DT1.mean(),paramCorHist.DT2.mean(),paramCorHist.DT3.mean(),paramCorHist.res.mean()],
            ['Sd',paramCorHist.kb1.std()*1000,paramCorHist.kb2.std()*1000,paramCorHist.kb3.std()*1000,paramCorHist.kp1.std()*1000,paramCorHist.kp2.std()*1000,paramCorHist.kp3.std()*1000,paramCorHist.DT1.std(),paramCorHist.DT2.std(),paramCorHist.DT3.std(),paramCorHist.res.std()],
            ['Min',paramCorHist.kb1.min()*1000,paramCorHist.kb2.min()*1000,paramCorHist.kb3.min()*1000,paramCorHist.kp1.min()*1000,paramCorHist.kp2.min()*1000,paramCorHist.kp3.min()*1000,paramCorHist.DT1.min(),paramCorHist.DT2.min(),paramCorHist.DT3.min(),paramCorHist.res.min()],
            ['Max',paramCorHist.kb1.max()*1000,paramCorHist.kb2.max()*1000,paramCorHist.kb3.max()*1000,paramCorHist.kp1.max()*1000,paramCorHist.kp2.max()*1000,paramCorHist.kp3.max()*1000,paramCorHist.DT1.max(),paramCorHist.DT2.max(),paramCorHist.DT3.max(),paramCorHist.res.max()],
            ['$1^{st}$ Quartile',paramCorHist.kb1.quantile(0.25)*1000,paramCorHist.kb2.quantile(0.25)*1000,paramCorHist.kb3.quantile(0.25)*1000,paramCorHist.kp1.quantile(0.25)*1000,paramCorHist.kp2.quantile(0.25)*1000,paramCorHist.kp3.quantile(0.25)*1000,paramCorHist.DT1.quantile(0.25),paramCorHist.DT2.quantile(0.25),paramCorHist.DT3.quantile(0.25),paramCorHist.res.quantile(0.25)],
            ['Median',paramCorHist.kb1.median()*1000,paramCorHist.kb2.median(),paramCorHist.kb3.median()*1000,paramCorHist.kp1.median()*1000,paramCorHist.kp2.median()*1000,paramCorHist.kp3.median()*1000,paramCorHist.DT1.median(),paramCorHist.DT2.median(),paramCorHist.DT3.median(),paramCorHist.res.median()],
            ['$3^{rd}$ Quartile',paramCorHist.kb1.quantile(0.75)*1000,paramCorHist.kb2.quantile(0.75)*1000,paramCorHist.kb3.quantile(0.75)*1000,paramCorHist.kp1.quantile(0.75)*1000,paramCorHist.kp2.quantile(0.75)*1000,paramCorHist.kp3.quantile(0.75)*1000,paramCorHist.DT1.quantile(0.75),paramCorHist.DT2.quantile(0.75),paramCorHist.DT3.quantile(0.75),paramCorHist.res.quantile(0.75)]

          ]
    lParanalKbkpTime = pd.DataFrame(data,columns=['stat','kb1','kb2','kb3','kp1','kp2','kp3','DT1','DT2','DT3','RSS'])   
 
    return lParanalKbkpTime

ParanalKbkpTime = descriptives()
                                                 
xTv = ['kb1', 'kb2', 'kb3', 'kp1', 'kp2', 'kp3', 'DT1', 'DT2', 'DT3']
xT = ['$ k_{b}^{1} $', '$ k_{b}^{2} $', '$ k_{b}^{3} $', '$ k_{p}^{1} $', '$ k_{p}^{2} $', '$ k_{p}^{3} $', ' $\Delta T^1$ ', ' $\Delta T^2$ ', ' $\Delta T^3$ ']

def heat_plot(ppGroup): 
    global downloadable_BiVar_zipRand 
    paramCorDrop = paramCorBiVar[['kb1', 'kb2', 'kb3', 'kp1', 'kp2', 'kp3', 'DT1', 'DT2', 'DT3']]
    corrMatrix  = paramCorDrop.corr()
    paramCorBiVar1000 = paramCorBiVar.copy()
    paramCorBiVar1000.kb1=paramCorBiVar1000.kb1*1000
    paramCorBiVar1000.kb2=paramCorBiVar1000.kb2*1000
    paramCorBiVar1000.kb3=paramCorBiVar1000.kb3*1000
    paramCorBiVar1000.kp1=paramCorBiVar1000.kp1*1000
    paramCorBiVar1000.kp2=paramCorBiVar1000.kp2*1000
    paramCorBiVar1000.kp3=paramCorBiVar1000.kp3*1000
    header = ['ID','kb1', 'kb2', 'kb3', 'kp1', 'kp2', 'kp3', 'DT1', 'DT2', 'DT3']
    paramCorBiVar1000.to_csv(downloadable_groupParameters_csv,columns = header,index=False)
    corrMatrix.to_csv(downloadable_groupCorrMatrix_csv)
    
    lista_files = [downloadable_groupParameters_csv,downloadable_groupCorrMatrix_csv]
    downloadable_BiVar_zipRand = downloadable_BiVar_Nozip+ppGroup+".zip"
    with zipfile.ZipFile(downloadable_BiVar_zipRand, 'w') as zipMe:        
        for file in lista_files:
            zipMe.write(file,basename(file), compress_type=zipfile.ZIP_DEFLATED)
            
    cM = corrMatrix.copy()
    npcM = cM.to_numpy()
    ZZ = np.zeros((9,9))+0.0
    
    npcMS = np.array(["%.2f" % w for w in npcM.reshape(npcM.size)])
    npcMS = npcMS.reshape(npcM.shape)
    
    npcM_O =npcM.copy()
    npcM *= 1 - np.tri(*npcM.shape, k=0)
    
    y = ['kb1', 'kb2', 'kb3', 'kp1', 'kp2', 'kp3', 'DT1', 'DT2', 'DT3']
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    xa = []
    ya = []
    aT = []
    
    for n, row in enumerate(npcM):
        for m in range(n):
            if ~np.isnan(npcM_O[n][m]):
                xa.append(x[m]-1)
                ya.append(y[n]-1)
                aT.append(npcMS[n][m])
            
    for m in range(len(xT)):
            xa.append(x[m]-1)
            ya.append(y[m]-1)
            aT.append('<b>'+xT[m]+'</b>')
    
    xaC = []
    yaC = [] 
    raC = [] 
    CoC = [] 
    
    cmap = matplotlib.cm.get_cmap('RdBu') 
    for n, row in enumerate(npcM):
        for m in range(n):
            yaC.append(x[m]-1)
            xaC.append(y[n]-1)     
            if np.isnan(npcM_O[n][m]):
                raC.append(0)
            else:    
                raC.append(math.sqrt(abs(npcM_O[n][m])/3.14)*0.8)  
            if np.isnan(npcM_O[n][m]):
                rgba = cmap((0+1)/2)
            else:    
                rgba = cmap((npcM_O[n][m]+1)/2)
            rgba = tuple(int((255*x)) for x in rgba[0:3])
            rgba = 'rgb'+str(rgba)
            CoC.append(rgba)
      
           
    figHeat = go.Figure(data=go.Heatmap(
                        x=xT,
                        y=xT,
                        z=ZZ, #npcM, 
                        text=npcMS,
                        showlegend=False,
                        hovertemplate =
                     #   '<br>X: %{x}<br>'+
                     #   '<br>Y: %{y}<br>'+
                        '<br>r: %{text}<br><extra></extra>',
                        #text = ['0.5'],
                        xgap=1,
                        ygap=1,
                        colorscale=col.diverging.RdBu,
                        zmax=1,
                        zmin=-1,
                        zmid=0
                        ))
    figHeat.update_layout(
    #    title={
    #        'text': 'Correlations',
    #        'y':0.9,
    #        'x':0.5,
    #        'xanchor': 'center',
    #        'yanchor': 'top'},
        #annotations=annotations,
        annotations=[
            dict(
                x=xpos,
                y=ypos,
                xref='x1',
                yref='y1',
                text=texts,
                font=dict(size=16),
                showarrow=False
        ) for xpos, ypos, texts in zip(xa, ya, aT)
        ],
        shapes=[
            dict(
                type="circle",
                xref="x1",
                yref="y1",
                x0=xpos-raCirc,
                y0=ypos-raCirc,
                x1=xpos+raCirc,
                y1=ypos+raCirc,
                line_color="black",
                fillcolor=rgbaC, #"PaleTurquoise",
                line_width=1
        ) for xpos, ypos, raCirc, rgbaC in zip(xaC, yaC, raC, CoC)
        ],
        yaxis_autorange='reversed', 
        xaxis_visible=False,
        yaxis_visible=False,
        autosize=False,
        width=600,
        height=500,
        #height=500,
        margin=dict(t=50, b=25  )#r=0, b=0, t=0, pad=0 )
    )
    
    xa1 = []
    ya1 = []  
    xa2 = []
    ya2 = []    
    for m in range(len(xT)+1):
        xa1=m
        ya1=0
        xa2=m
        ya2=8
        figHeat.add_shape(
            dict(
                type="line",
                x0=xa1-0.5,
                y0=ya1-0.5,
                x1=xa2-0.5,
                y1=ya2+0.5,
                line_color="black",
                line_width=1) 
            )
        figHeat.add_shape(
            dict(
                type="line",
                x0=ya1-0.5,
                y0=xa1-0.5,
                x1=ya2+0.5,
                y1=xa2-0.5,
                line_color="black",
                line_width=1) 
            )
    return figHeat


#HeatHeat = heat_plot()
   

sc_X = 'kb1'
sc_Y = 'kp1'
hist_Var = 'RSS'

pourff123 = pourcorr3[(pd.notnull(pourcorr3.kb1))  & (pd.notnull(pourcorr3.kb2)) & (pd.notnull(pourcorr3.kb3))]
pourff12  = pourcorr3[(pd.notnull(pourcorr3.kb1))  & (pd.notnull(pourcorr3.kb2)) & (pd.isnull(pourcorr3.kb3))]
pourff1   = pourcorr3[(pd.notnull(pourcorr3.kb1))  & (pd.isnull(pourcorr3.kb2)) & (pd.isnull(pourcorr3.kb3))]
subjects = paramCor.ID.unique()

    

def scatter_plot():
    scX = paramCorBiVar[sc_X] * 1000 
    scY = paramCorBiVar[sc_Y] * 1000
    if sc_X in ['DT1', 'DT2', 'DT3']:
        scX = paramCorBiVar[sc_X] 
    if sc_Y in ['DT1', 'DT2', 'DT3']:
        scY = paramCorBiVar[sc_Y]
        
    figure = go.Figure(data=go.Scatter(x=scX, y=scY, mode='markers',showlegend=False, name='BCS',
        marker=dict(
            color='rgba(0, 0, 0, 0.0)',
            size=6,
            line=dict(
                color='Black',
                width=1
            )
        )))
    figure.update_layout(xaxis_title=xT[xTv.index(sc_X)], yaxis_title=xT[xTv.index(sc_Y)])
    #figure.update_layout(title='Scatter Plot of \\( \Delta k_b\\) and \\(k_p\\) 67')
    #figure.update_layout(title='Scatter Plot of $$k_b$$ and $$k_p$$')
    #figure.update_layout(title='Scatter Plot')
    figure.update_layout(
        title={
        #'text': 'Scatter Plot of \(k_b\) and \(k_p\) 67',
        #'text': 'Scatter Plot of $'+xT[xTv.index(sc_Y)]+'$ and $'+xT[xTv.index(sc_X)]+'$', 
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        width=600,
        height=550,
        #height=510,
        margin=dict(t=50,  )#r=0, b=0, t=0, pad=0 )
        )
    figure.update_xaxes(showline=True, linewidth=2, linecolor='black')
    figure.update_yaxes(showline=True, linewidth=2, linecolor='black')
    figure.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    figure.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        
    return figure


def hist_graph(name,ppGroup):
    global downloadable_Descr_zipRand 
    if name== 'kb1': 
        xx = paramCorHist.kb1.copy()*1000
        nameTex = '$ k_{b}^{1} $'
    if name== 'kb2': 
        xx = paramCorHist.kb2.copy()*1000
        nameTex = '$ k_{b}^{2} $'
    if name== 'kb3': 
        xx = paramCorHist.kb3.copy()*1000
        nameTex = '$ k_{b}^{3} $'
    if name== 'kp1': 
        xx = paramCorHist.kp1.copy()*1000
        nameTex = '$ k_{p}^{1} $'
    if name== 'kp2': 
        xx = paramCorHist.kp2.copy()*1000
        nameTex = '$ k_{p}^{2} $'
    if name== 'kp3': 
        xx = paramCorHist.kp3.copy()*1000
        nameTex = '$ k_{p}^{3} $'
    if name== 'RSS': 
        xx = paramCorHist.res.copy()
        nameTex = '$ RSE $'
    if name== 'tb1': 
        xx = paramCorHist.tb1.copy()
        nameTex = '$ t_{b}^{1} $'
    if name== 'tb2': 
        xx = paramCorHist.tb2.copy()
        nameTex = '$ t_{b}^{2} $'
    if name== 'tb3': 
        xx = paramCorHist.tb3.copy()
        nameTex = '$ t_{b}^{3} $'
    if name== 'DeltaT1': 
        xx = paramCorHist.DT1.copy()    
        nameTex = '$ \Delta T_1 $'
    if name== 'DeltaT2': 
        xx = paramCorHist.DT2.copy()
        nameTex = '$ \Delta T_2 $'
    if name== 'DeltaT3': 
        xx = paramCorHist.DT3.copy()
        nameTex = '$ \Delta T_3 $'
    nanCnt=np.count_nonzero(~np.isnan(xx))
    if nanCnt==0: 
        figure = go.Figure(data=[go.Histogram( nbinsx=300,     
            marker=dict(
                color='LightSkyBlue',
                line=dict(
                    color='black',
                    width=1
                )))])
    else :        
        figure = go.Figure(data=[go.Histogram(x=xx, nbinsx=300,     
            marker=dict(
                color='LightSkyBlue',
                line=dict(
                    color='black',
                    width=1
                )))])
    figure.update_layout(
#        title={
#        'text': nameTex,
#        'y':0.9,
#        'x':0.5,
#        'xanchor': 'center',
#        'yanchor': 'top'},
       yaxis_title_text='Frequency', # yaxis label
       autosize=False,
       width=700,
       height=450,
       margin=dict(t=40, b=40, l=0, r=0  )#r=0, b=0, t=0, pad=0 )
    )
    figure.update_xaxes(showline=True, linewidth=2, linecolor='black')
    figure.update_yaxes(showline=True, linewidth=2, linecolor='black')
    figure.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    figure.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    
    paramCorHist1000=paramCorHist.copy()
    paramCorHist1000.kb1=paramCorHist1000.kb1*1000
    paramCorHist1000.kb2=paramCorHist1000.kb2*1000
    paramCorHist1000.kb3=paramCorHist1000.kb3*1000
    paramCorHist1000.kp1=paramCorHist1000.kp1*1000
    paramCorHist1000.kp2=paramCorHist1000.kp2*1000
    paramCorHist1000.kp3=paramCorHist1000.kp3*1000
    ParanalKbkpTimeQ = ParanalKbkpTime.copy()
    ParanalKbkpTimeQ.stat[4]= 'Q1'
    ParanalKbkpTimeQ.stat[6]= 'Q3'
    header = ['ID','kb1', 'kb2', 'kb3', 'kp1', 'kp2', 'kp3', 'DT1', 'DT2', 'DT3']
    paramCorHist1000.to_csv(downloadable_descParameters_csv,columns = header,index=False)         
    ParanalKbkpTimeQ.to_csv(downloadable_descStat_csv,index=False)
    lista_files = [downloadable_descStat_csv, downloadable_descParameters_csv]
    downloadable_Descr_zipRand = downloadable_Descr_Nozip+ppGroup+".zip"
    with zipfile.ZipFile(downloadable_Descr_zipRand, 'w') as zipMe:        
        for file in lista_files:
            zipMe.write(file,basename(file), compress_type=zipfile.ZIP_DEFLATED)
     
    return figure

tabs_styles = {
    'height': '40px'
}
tab_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'color': 'rgb(2, 21, 70)',
    'backgroundColor': 'white',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': 'rgb(2, 21, 70)',
    'color': 'white',
    'padding': '6px'
}

app.title = 'adaptive capacity'
server = app.server

app.layout = html.Div([   
   
    html.Div(
        className="pkcalc-banner",
        children=[
            html.H2("PhenoBR: a model to phenotype body condition’ dynamics in ruminants"),
        ], style={'margin-bottom': '20px'} 
    ),
          
        html.Div(id='speck-control-tabs', className='control-tabs', children=[ 
            dcc.Tabs(id='speck-tabs', value='what-is', children=[
                dcc.Tab(
                    label='Introduction',
                    value='what-is',
                    style=tab_style, 
                    selected_style=tab_selected_style,
                    children=html.Div(className='control-tabBB', children=[
                      html.Div(
                           [
                            
                           
                         html.Div(
                           [ 
                            html.P(children=[
                                    html.Strong('What is PhenoBR:')]),
                            html.P(children=[
                                   'PhenoBR is a software to support phenotyping of ruminant robustness when facing frequent negative '
                                   ' energy balances throughout their reproductive cycles. This information is of main concern in the'
                                   ' context of genetic selection for robustness and resilience. Body reserves (BR) are the main sources of energy in'
                                   ' ruminants facing negative energy balance challenges e.g. during highly demanding reproductive'
                                   ' cycles or feed scarcity periods. PhenoBR is based on a dynamic model describing the variations'
                                   ' of Body condition score as the indicator of the body reserves in ruminants.'])
                           ], style={'background-color': 'lightblue','padding': '4px','border-width': 'thin','border-style':'solid', 'margin-bottom': '20px'} 
                            ),
                            html.P(children=[html.Strong('Authors:'),
                                             ' Masoomeh Taghipoor, György Kövér, Dominique Hazard, Eliel Gonzales, Tiphaine Macé'
                                            ], style={'background-color': 'lightblue','padding': '2px','border-width': 'thin','border-style':'solid'}),
                            ],     style={'width': '50%','height': '50px','vertical-align': 'middle',
                                        'display': 'table-cell',
                                        'margin': '0px',
                                        'padding': '0px',
                                     #   'background-color': 'lightblue'
                                        }),
                       html.Div([
                                  html.Img(src=app.get_asset_url('img4.png'), style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                                  html.P(children=[
                                             'The scheme represent the regulations of the model for one productive cycle. '
                                             'Flux to BCS is regulated by the difference between \(BCS_i\)  and the \(BCS_m\). '
                                             'The flux to  \(p_i\)  is activated in the interval \([t_b^i,t_e^i ]\)  and will stop '
                                             'when it reaches \(p_m\).  From the beginning of the perturbation, the decrease of \(BCS_i\) '
                                             'is counterbalanced by all internal physiological mechanisms of the ewes looking to '
                                             'keep the \(BCS_i\) close to \(BCS_m\).'
                                            ], style={'width': '500px','background-color': 'lightblue','padding': '2px','border-width': 'thin','border-style':'solid','margin-left': 'auto', 'margin-right': 'auto'}), 
                                       html.Div(
                                                id="download-areaDataset",
                                                className="section",
                                                children=[
                                                  html.Form(
                                                        action=downloadable_BCSDataSet_zip,
                                                        method="get", 
                                                        children=[
                                                            html.Button(
                                                                className="button",
                                                                type="submit",
                                                                title ="Download the raw dataset as a zip file",
                                                                children=[
                                                                    "download"
                                                                ],style={'display': 'inline-block','border': '2px solid rgb(2, 21, 70)'}
                                                            )
                                                        ]
                                                    )
                                                  
                                                ],style={'display': 'block','margin-top': '20px','margin-left': '200px','margin-right': 'auto'}
                                            ),
                                ],
                                 style={#'width': '65%',
                                        'display': 'table-cell','margin-left': 'auto','margin-right': 'auto','text-align':'center'
                                     #   'background-color': 'lightblue'
                                        }) 

                        ])
                ), 
                dcc.Tab(
                    label='The PhenoBR model',
                    value='what-is2',
                    style=tab_style, selected_style=tab_selected_style,
                    children=html.Div(className='control-tab', children=[
                      html.Div(
                           [
                            html.P('Author #1 György Kövér', style={'background-color': 'lightblue','padding': '2px','border-width': 'thin','border-style':'solid'}),
                            html.P('Author #2 György Kövér', style={'background-color': 'lightblue','padding': '2px','border-width': 'thin','border-style':'solid'}),
                            html.H4(className='what-isSS', children='What is PhenoBR?', style={'background-color': 'lightblue','padding': '2px','border-width': 'thin','border-style':'solid'}),
                            html.P('PhenoBR is a generic dynamic model, based on a system of ordinary differential equations. ' 
                               'Its aim is to describe the BCS variations of ruminants over several productive cycles. '
                               'The animal model used in this work is the reproductive ewe, during its whole lifespan. '
                               'The parameters of this model represent animal characteristics in terms of its capacities '
                               'to mobilize and to restore BR in function of the physiological point in a given production cycle. '
                               'Two main types of parameters are expected. First ones are time related parameters '
                               '(i.e. time related to the beginning of BR mobilization and the interval of BR mobilization '
                               'period for each productive cycle). The second category of parameters are related to the intensity '
                               'of each BR mobilization period and then the capacity for recovering initial BR status.', style={'background-color': 'lightblue','padding': '2px','border-width': 'thin','border-style':'solid'}) 
                           ],
                                 style={'width': '50%','height': '50px','vertical-align': 'middle',
                                        'display': 'table-cell',
                                        'margin': '0px',
                                        'padding': '0px',
                                     #   'background-color': 'lightblue'
                                        }),
                        ])
                ), 
                dcc.Tab(
                    label='Individual response',
                    value='what-is3',style=tab_style, 
                    selected_style=tab_selected_style,
                    children=html.Div(className='control-tab', children=[
                      html.P(children='Characterize the response of one animal to NEB challenge during one or several productive cycle', 
                                               style={'background-color': 'lightblue','margin-top':'7px','margin-bottom':'20px','vertical-align': 'middle',
                                                      'border-width': 'thin','border-style':'solid','text-align':'center','font-size': '14pt','fontWeight': 'bold'}),
                      html.Div(
                           [
#############                             
                                html.Div(
                                    id="dropdown-controls",
                                    children=[
                                        # dropdown 1
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id="parity",
                                                    options=[{ 'label': 'Ewes with parity: 1',  'value': 'G1'  },
                                                             { 'label': 'Ewes with parity: 2',  'value': 'G2'  },
                                                             { 'label': 'Ewes with parity: 3',  'value': 'G3'  },
                                                             { 'label': 'Ewes with parity: 12', 'value': 'G12' },
                                                             { 'label': 'Ewes with parity: 13', 'value': 'G13' },
                                                             { 'label': 'Ewes with parity: 23', 'value': 'G23' },
                                                             { 'label': 'Ewes with parity: 123', 'value': 'G123' },
                                                             { 'label': 'The complete datasett', 'value': 'ALL' }
                                                            ],
                                                    value='G123',
                                                    clearable=False,
                                                    optionHeight = 24,
                                                    style={'margin': '0px','margin-right': '7px',
                                                           'padding': '0px'},),
                                            ],
                                            style={'width': '36%',
                                                   'display': 'inline-block','vertical-align': 'middle'}),
                                                
                                                        # dropdown 2
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id="PKSubject",
                                                    options=[{
                                                        'label': i,
                                                        'value': i
                                                    } for i in subjects],
                                                    value='51180',
                                                    clearable=False,
                                                    optionHeight = 24),
                                            ],
                                            style={'width': '14%',
                                                   'display': 'inline-block','vertical-align': 'middle'}),
                                                             
                                                    ],
                                                #style={'width':'85%', 'margin': '0 auto'}
                                                ),
#############          
                                html.Div( 
                                    children=[     
                                                dcc.Graph(id='BCS-graph',
                                                            config={
                                                               'displayModeBar': False},
                                                            style={'margin': '0px','margin-top': '7px','margin-right': '7px',
                                                                   'padding': '0px'}
                                                            ),             
                                                    ],
                                                style={'margin': 'auto'}),          
#############                    
                           ],
                                 style={'width': '70%',
                                        'display': 'inline-block',
                                        'padding': '0',
                                        'vertical-align': 'top'
                                        }),
                      html.Div(
                           [
#############                           
                                        html.P(children='In this part you select an animal based on the number of parities, to see how its response in term of the variation of is summerized with 4 parameters per parity ( $k_b$, $k_p$, $t_b$, $\Delta T$ ). ', 
                                               style={'background-color': 'lightblue','display': 'inline-block','margin-top':'45px','margin-left': 'auto', 'margin-right': 'auto','padding': '2px','border-width': 'thin','border-style':'solid'}),
                                          
                                        html.P(children='Intensities of reserves’ mobilization and recovery', 
                                               style={'background-color': 'lightblue','display': 'inline-block','margin-top':'15px','margin-left': 'auto', 'margin-right': 'auto','padding': '2px','border-width': 'thin','border-style':'solid'}),
                                        html.Div(
                                                [
                                                    dash_table.DataTable(
                                                        id='tableFich',
                                                        columns=[{"name":"Parity", "id": "parity"},
                                                                 {"name":"$ k_b $", "id": "kb", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                                                                 {"name":"$ k_p $", "id": "kp", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                                                                ],
                                                        style_cell={'textAlign': 'center', 'whiteSpace' : 'normal', 'minWidth': '40px', 'width': '40px', 'maxWidth': '40px', 'padding': '2px' },
                                                        style_table={'textAlign': 'center', 'maxHeight': 400},
                                                        data=fichierfinal.to_dict('records'),
                                                        style_header=table_header_style,
                                                    )
                                                ],
                                                ),
                                        html.P(children='Time related parameters', 
                                               style={'background-color': 'lightblue','display': 'inline-block','margin-top':'15px','vertical-align': 'middle','padding': '2px','border-width': 'thin','border-style':'solid'}),
                                        html.Div(
                                                [
                                                    dash_table.DataTable(
                                                        id='tableFich2',
                                                        columns=[{"name":"Parity", "id": "parity"},
                                                                 {"name":"$ t_b $", "id": "tb"},
                                                                 {"name":"$\Delta T$", "id": "DT"}
                                                                ],
                                                        style_cell={'textAlign': 'center', 'whiteSpace' : 'normal', 'minWidth': '40px', 'width': '40px', 'maxWidth': '40px', 'padding': '2px' },
                                                        style_table={'textAlign': 'center', 'maxHeight': 400},
                                                        data=fichierfinal.to_dict('records'),
                                                        style_header=table_header_style,
                                                    )
                                                ],
                                                ),
                                        html.Div(
                                                id="download-areaBCS",
                                                className="section",
                                                children=[
                                                 
                                                ]
                                            )
                           ],   
                                 style={'width': '30%',
                                        'display': 'inline-block',
                                        'padding': '0',
                                        'vertical-align': 'middle','margin': 'auto','text-align': 'center'
                                        }),       
                        ])
                ), 
                dcc.Tab(
                    label='Group analysis',
                    value='what-is4',style=tab_style, 
                    selected_style=tab_selected_style,
                    children=html.Div(className='control-tab', children=[ html.P(children='Characteristics of a selected group of animals  -  Descriptive analysis', 
                                               style={'background-color': 'lightblue','margin-top':'7px','margin-bottom':'20px','vertical-align': 'middle',
                                                      'border-width': 'thin','border-style':'solid','text-align':'center','font-size': '14pt','fontWeight': 'bold'}),
                      html.Div(
                           [
#############                             
                                html.Div(
                                    id="dropdown-controls4",
                                    children=[
                                        # dropdown 1
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id="parityHist",
                                                    options=[{ 'label': 'Ewes with parity: 1',  'value': 'G1'  },
                                                             { 'label': 'Ewes with parity: 2',  'value': 'G2'  },
                                                             { 'label': 'Ewes with parity: 3',  'value': 'G3'  },
                                                             { 'label': 'Ewes with parity: 12', 'value': 'G12' },
                                                             { 'label': 'Ewes with parity: 13', 'value': 'G13' },
                                                             { 'label': 'Ewes with parity: 23', 'value': 'G23' },
                                                             { 'label': 'Ewes with parity: 123', 'value': 'G123' },
                                                             { 'label': 'The complete dataset', 'value': 'ALL' }
                                                            ],
                                                    value='G123',
                                                    clearable=False,
                                                    optionHeight = 24,
                                                    style={'margin': '0px','margin-right': '35px',
                                                           'padding': '0px'},),
                                            ],
                                            style={'width': '100%',
                                                   'vertical-align': 'middle'}),
              
   
                                                             
                                                    ],
                                                ), 
#############          
#############                    
                           ],
                                 style={'width': '25%',
                                        'display': 'inline-block', 'margin-left': 'auto', 'margin-right': 'auto',
                                        'padding': '0',
                                        'vertical-align': 'top'
                                        }),
                      html.Div(
                           [
                                html.Div(
                                    [
                                        dash_table.DataTable(
                                            id='tableKbkpTime',
                                            columns=[{"name":"", "id": "stat"},
                                                     {"name":"\( k_{b}^{1} \)", "id": "kb1", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                                                     {"name":"\( k_{b}^{2} \)", "id": "kb2", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                                                     {"name":"\( k_{b}^{3} \)", "id": "kb3", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                                                     {"name":"\( k_{p}^{1} \)", "id": "kp1", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                                                     {"name":"\( k_{p}^{2} \)", "id": "kp2", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                                                     {"name":"\( k_{p}^{3} \)", "id": "kp3", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                                                     {"name":"\(\Delta T^1\)", "id": "DT1", 'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed)},
                                                     {"name":"\(\Delta T^2\)", "id": "DT2", 'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed)},
                                                     {"name":"\(\Delta T^3\)", "id": "DT3", 'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed)},
                                                     {"name":"\(RSE\)", "id": "RSS", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)}
                                                    ],
                                            style_cell={'textAlign': 'center', 'whiteSpace' : 'normal', 'maxWidth': '120px','padding': '2py' },
                                            style_table={'textAlign': 'center', 'maxHeight': 400},
                                            data=ParanalKbkpTime.to_dict('records'),
                                            style_header=table_header_style,
                                        )
                                    ],
                                    ),                   
                                        
                                        
                                        
                                        
                                        
                                        
                               html.Div(
                                    [         
                                            dcc.Graph(
                                                id = 'graphHist',    
                                                figure=hist_graph(hist_Var,'G123'),
                                                            config={
                                                               'displayModeBar': False},
                                                            style={'margin': '0px','margin-top': '7px','margin-right': '7px',
                                                                   'padding': '0px'}
                                                            )
                                    ],
                                    style={ 'display': 'inline-block', 'margin': '0px','margin-right': '7px','padding': '0px'}
                                    ),
                               html.Div(
                                    [
                                            dcc.RadioItems(id = 'input-radio-button',
                                                                  options = [dict(label = '$ k_{b}^{1} $', value = 'kb1'),
                                                                             dict(label = '$ k_{b}^{2} $', value = 'kb2'),
                                                                             dict(label = '$ k_{b}^{3} $', value = 'kb3'),
                                                                             dict(label = '$ k_{p}^{1} $', value = 'kp1'),
                                                                             dict(label = '$ k_{p}^{2} $', value = 'kp2'),
                                                                             dict(label = '$ k_{p}^{3} $', value = 'kp3'),
                                                                             dict(label = '$ RSE $', value = 'RSS'),
                                                                             dict(label = '$ t_{b}^{1} $', value = 'tb1'),
                                                                             dict(label = '$ t_{b}^{2} $', value = 'tb2'),
                                                                             dict(label = '$ t_{b}^{3} $', value = 'tb3'),
                                                                             dict(label = '$ \Delta T^1 $', value = 'DeltaT1'),
                                                                             dict(label = '$ \Delta T^2 $', value = 'DeltaT2'),
                                                                             dict(label = '$ \Delta T^3 $', value = 'DeltaT3')
                                                                            ],
                                                                  value = 'RSS',
                                          labelStyle={"padding-left": "10px",'display': 'block','margin-top': '4px',},
                                          style={"padding-left": "50px",  "vertical-align": "top",'margin-top': '40px'},
                                          ),
                                    ],
                                    style={ 'display': 'inline-block',"vertical-align": "top"}
                                    ),
                                       html.Div(
                                                id="download-areaDescr",
                                                className="section",
                                                children=[
                                                
                                                ],style={'display': 'inline-block','vertical-align': 'middle','margin-top': '10px','margin-left': '200px'}    #,'float': 'right'
                                            ),
                                    
                                    
                           ],   
                                 style={'width': '75%',
                                        'display': 'inline-block',
                                        'padding': '0',
                                        'vertical-align': 'middle','margin': 'auto','text-align': 'center'
                                        }),  
                      
                        ])
                ),
                dcc.Tab(
                    label='Correlation analysis',
                    value='what-is5',style=tab_style, 
                    selected_style=tab_selected_style,
                    children=html.Div(className='control-tab', children=[
                      html.P(children='Characteristics of a selected group of animals  -  Correlation analysis', 
                                               style={'background-color': 'lightblue','margin-top':'7px','margin-bottom':'20px','vertical-align': 'middle',
                                                      'border-width': 'thin','border-style':'solid','text-align':'center','font-size': '14pt','fontWeight': 'bold'}),
                      html.Div(
                           [
#############                             
                                html.Div(
                                    id="dropdown-controls2",
                                    children=[
                                        # dropdown 1
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id="parityBiVar",
                                                    options=[{ 'label': 'Ewes with parity: 1',  'value': 'G1'  },
                                                             { 'label': 'Ewes with parity: 2',  'value': 'G2'  },
                                                             { 'label': 'Ewes with parity: 3',  'value': 'G3'  },
                                                             { 'label': 'Ewes with parity: 12', 'value': 'G12' },
                                                             { 'label': 'Ewes with parity: 13', 'value': 'G13' },
                                                             { 'label': 'Ewes with parity: 23', 'value': 'G23' },
                                                             { 'label': 'Ewes with parity: 123', 'value': 'G123' },
                                                             { 'label': 'The complete dataset', 'value': 'ALL' }
                                                            ],
                                                    value='G123',
                                                    clearable=False,
                                                    optionHeight = 24,
                                                    style={'margin-left': 'auto','margin-right': 'auto',
                                                           'padding': '0px'},),
                                            ],
                                            style={'width': '280px',
                                                   'display': 'inline-block','vertical-align': 'middle','margin-left': 'auto','margin-right': 'auto'}),
              
   
                                                             
                                                    ],style={'margin-left': 'auto','margin-right': 'auto'},
                                                ), 
#############          
                                    
                                html.Div( 
                                    children=[  
                                                dcc.Graph(id='graphHeat',
                                                            figure=heat_plot('G123'),
                                                            config={
                                                               'displayModeBar': False},
                                                            style={'margin': '0px','margin-top': '7px','margin-right': '7px',
                                                                   'padding': '0px'}
                                                            ),             
                                                    ],
                                                style={ 'display': 'inline-block', 'margin': '0px','margin-right': '7px','padding': '0px'}),          
#############                    
                           ],
                                 style={'width': '50%',
                                        'display': 'inline-block', 'margin-left': 'auto', 'margin-right': 'auto',
                                        'padding': '0',
                                        'vertical-align': 'top'
                                        }),
                      html.Div(
                           [
#############                              
                                html.Div(
                                    id="dropdown-controls3",
                                    children=[
                                        # dropdown 1
                                            html.Div(
                                                [   html.P("""Horizontal axis""")
                                                ],
                                                style={'display': 'inline-block','margin': '0px','margin-right': '7px','padding': '0px'}
                                                ),
                                            html.Div(
                                                [
                                                    dcc.Dropdown(
                                                        id="XVar",
                                                        options=[{
                                                            'label': i[1] ,
                                                            'value': i[0] 
                                                        }  for i in zip(xTv,xT) ],
                                                        value='kb1',
                                                        clearable=False,
                                                        optionHeight = 24,
                                                        style={'margin': '0px','margin-right': '7px', 
                                                           'padding': '0px'},),
                                                ],
                                                style={'width': '90px',
                                                       'display': 'inline-block','vertical-align': 'middle'}),
                                                
                                                
                                            html.Div(
                                                [   html.P("""Vertical axis""")
                                                ],
                                                style={'display': 'inline-block','margin': '0px','margin-right': '7px','padding': '0px'}),
                                            html.Div(
                                                [
                                                    dcc.Dropdown(
                                                        id="YVar",
                                                        options=[{
                                                            'label': i[1] ,
                                                            'value': i[0] 
                                                        }  for i in zip(xTv,xT) ],
                                                        value='kp1',
                                                        clearable=False,
                                                        optionHeight = 24,
                                                        style={'margin': '0px','margin-right': '7px', 
                                                           'padding': '0px'},),
                                                ],
                                                style={'width': '90px',
                                                           'display': 'inline-block','vertical-align': 'middle'}),
        
   
                                                             
                                                    ],
                                                ),
#############          
                                html.Div( 
                                    children=[  
                                                dcc.Graph(id='Scatter-graph',
                                                            figure=scatter_plot(),
                                                            config={
                                                               'displayModeBar': False},
                                                            style={'margin': '0px','margin-top': '7px','margin-right': '7px',
                                                                   'padding': '0px'}
                                                            ),   
                                       html.Div(
                                                id="download-area",
                                                className="section",
                                                children=[
                                                  
                                                ],style={'display': 'block','margin-top': '10px','margin-left': 'auto','margin-right': 'auto'}
                                            ),          
                                                    ],
                                                style={ 'display': 'inline-block', 'margin': '0px','margin-right': '7px','padding': '0px'}), 
                           ],   
                                 style={'width': '50%',
                                        'display': 'inline-block',
                                        'padding': '0',
                                        'vertical-align': 'middle','margin': 'auto','text-align': 'center'
                                        }),       
                        ])
                ),
                
             ], style=tabs_styles,
            colors={
            "border": "black",
            "primary": "red",
            "background": "orange"
            }),
                
        ]),
                                   
     
                              
   
])
          
def build_download_buttonBCS(uri):
    """Generates a download button for the resource"""
    button = html.Form(
                                                        action=uri,
                                                        method="get",
                                                        children=[
                                                            html.Button(
                                                                className="button",
                                                                type="submit",
                                                                title ="Download the parameters, the measured and simulated BCS data as a zip file",
                                                                children=[
                                                                    "download"
                                                                ],style={'border': '2px solid rgb(2, 21, 70)'}
                                                            )
                                                        ],style={'margin-top': '15px'}
                                                    )        
    return button       

def build_download_buttonDescr(uri):
    """Generates a download button for the resource"""
    button = html.Form(
                                                        action=uri,
                                                        method="get", 
                                                        children=[
                                                            html.Button(
                                                                className="button",
                                                                type="submit",
                                                                title ="Download the parameters of the selected parity group and the descriptive statistics as a zip file",
                                                                children=[
                                                                    "download"
                                                                ],style={'display': 'inline-block','border': '2px solid rgb(2, 21, 70)'}
                                                            )
                                                        ]
                                                    )
    return button

# uri = downloadable_BiVar_zip
def build_download_button(uri):
    """Generates a download button for the resource"""
    button = html.Form(
                                                        action=uri,
                                                        method="get", 
                                                        children=[
                                                            html.Button(
                                                                className="button",
                                                                type="submit",
                                                                title ="Download the parameters of the selected parity group and the correlation coefficients as a zip file",
                                                                children=[
                                                                    "download"
                                                                ],style={'display': 'inline-block','border': '2px solid rgb(2, 21, 70)'}
                                                            )
                                                        ]
                                                    )
    return button

@app.callback(
    [    
    dash.dependencies.Output("PKSubject", "options"),
    ],
    [dash.dependencies.Input("parity", "value"),
     ],
)
def update_date_dropdown(pGroup):
    global paramCor
    if pGroup == 'ALL': 
        paramCor = paramCor_Orig.copy()
    else: 
        paramCor = paramCor_Orig[paramCor_Orig.Group == pGroup]
    subjectsG = paramCor.ID.unique()
    return [[{'label': i, 'value': i} for i in subjectsG]]

@app.callback(
    [    
    dash.dependencies.Output("tableKbkpTime", "data"),
    dash.dependencies.Output('graphHist', 'figure'),
    dash.dependencies.Output("download-areaDescr", "children"),
    ],
    [dash.dependencies.Input("parityHist", "value"),dash.dependencies.Input("input-radio-button", "value"),]
)
def update_date_dropdownHist(pGroup,name):
    global paramCorHist
    global hist_Var 
    global ParanalKbkpTime
    global downloadable_Descr_zipRand
    hist_Var = name
    if pGroup == 'ALL': 
        paramCorHist = paramCor_Orig.copy()
    else: 
        paramCorHist = paramCor_Orig[paramCor_Orig.Group == pGroup]
    ParanalKbkpTime = descriptives()
    histfigure = hist_graph(hist_Var,pGroup)
    return ParanalKbkpTime.to_dict('records'),histfigure,build_download_buttonDescr(downloadable_Descr_zipRand)


@app.callback(
    [    
    dash.dependencies.Output('Scatter-graph', 'figure'),
    dash.dependencies.Output('graphHeat', 'figure'),
    dash.dependencies.Output("download-area", "children"),
    ],
    [dash.dependencies.Input("parityBiVar", "value"),
     dash.dependencies.Input('XVar', 'value'),dash.dependencies.Input('YVar', 'value')],
)
def update_date_dropdownBivar(pGroup,pXVar,pYVar):
    global paramCorBiVar
    global sc_X
    global sc_Y
    global downloadable_BiVar_zipRand
    sc_X = pXVar
    sc_Y = pYVar
    if pGroup == 'ALL': 
        paramCorBiVar = paramCor_Orig.copy()
    else: 
        paramCorBiVar = paramCor_Orig[paramCor_Orig.Group == pGroup]
    heatfigure = heat_plot(pGroup)
    scatterfigure=scatter_plot()
    return scatterfigure,heatfigure,build_download_button(downloadable_BiVar_zipRand)

@app.callback(
    [
    dash.dependencies.Output('BCS-graph', 'figure'),
    dash.dependencies.Output('tableFich', 'data'),
    dash.dependencies.Output('tableFich2', 'data'),
    dash.dependencies.Output("download-areaBCS", "children"),
    ],
    [dash.dependencies.Input('PKSubject', 'value')])
def display_table(Ewe): 
    global tb1, tb2, tb3, te1, te2, te3, BCSm, Pmax
    global downloadable_BCSDataSim_zipRand
    dffTFich = fichierfinal[fichierfinal.ID==int(Ewe)].copy()
    dffTFich['DT'] = dffTFich['te']-dffTFich['tb']
    dffTFich['kb'] = dffTFich['kb']*1000
    dffTFich['kp'] = dffTFich['kp']*1000
    header = ['ID','parity','kb','kp','tb','DT']
    dffTFich.to_csv(downloadable_parameters_csv,columns = header,index=False)
    dffT = pkdata[pkdata.Ewe==int(Ewe)]
    dff = dffT[pd.notnull(dffT.BCS1)].copy()
    dffM = dffT[pd.notnull(dffT.BCS1)].copy()
    dffM['BCS'] = dffM['BCS1']
    dffM['ID'] = int(Ewe)
    header = ['ID','Parity','day','BCS']
    dffM.to_csv(downloadable_BCSData_csv,columns = header,index=False)
    figure = go.Figure(data=go.Scatter(x=dff.day, y=dff.BCS1, mode='markers',showlegend=False, name='BCS',
        marker=dict(
            color='rgba(0, 0, 0, 0.0)',
            size=6,
            line=dict(
                color='Black',
                width=1
            )
        )))
    figure.update_layout(xaxis_title="days of age", yaxis_title="BCS")
#    figure.update_layout(
#        title={
#        'text': 'Ewe {}'.format(Ewe),
#        'y':0.9,
#        'x':0.5,
#        'xanchor': 'center',
#        'yanchor': 'top'})
    figure.update_xaxes(range=[1, 1100])
    figure.update_yaxes(range=[2, 4])

    figure.update_layout(
        height=500,
        margin=dict(t=40, b=40 )#r=0, b=0, t=0, pad=0 )
    )


    paramCor_ID = paramCor[paramCor["ID"]==int(Ewe)]
    
    Pmax = 2
    BCSm = paramCor_ID.iloc[0].BCSm
    BCSopt=BCSm-0.5
    tb1 = paramCor_ID.iloc[0].tb1
    tb2 = paramCor_ID.iloc[0].tb2
    tb3 = paramCor_ID.iloc[0].tb3
    te1 = paramCor_ID.iloc[0].te1
    te2 = paramCor_ID.iloc[0].te2
    te3 = paramCor_ID.iloc[0].te3
     
    
    paramCor_ID = paramCor_ID.fillna(0)
    kb1 = paramCor_ID.iloc[0].kb1
    kb2 = paramCor_ID.iloc[0].kb2
    kb3 = paramCor_ID.iloc[0].kb3
    kp1 = paramCor_ID.iloc[0].kp1
    kp2 = paramCor_ID.iloc[0].kp2
    kp3 = paramCor_ID.iloc[0].kp3
    
    argsB = (kb1, kb2, kb3, kp1, kp2, kp3)
    
    char_ID = char2[char2["Ewe.ID"]==int(Ewe)]
    char_ID_np = char_ID.to_numpy()
    
    if char_ID_np[0][5]>0:
        yinit = [char_ID_np[0][5],char_ID_np[1][5],char_ID_np[2][5],0,0,0,char_ID_np[0][5]]
    elif char_ID_np[1][5]>0 and char_ID_np[0][5]==0:
        yinit = [char_ID_np[0][5],char_ID_np[1][5],char_ID_np[2][5],0,0,0,char_ID_np[1][5]]
    elif char_ID_np[2][5]>0 and char_ID_np[0][5]==0 and char_ID_np[1][5]==0:
        yinit = [char_ID_np[0][5],char_ID_np[1][5],char_ID_np[2][5],0,0,0,char_ID_np[2][5]]
      
    Mydata_ID = Mydata[Mydata["Ewe"]==int(Ewe)]
    timesMax  = Mydata_ID.day.max()    
    times     = np.linspace(0, timesMax, timesMax+1, endpoint=True)

    sol = solve_ivp(BCSModel2, [0, timesMax], yinit,t_eval=times, atol = 1e-10, rtol = 1e-10, args=argsB, dense_output=False)

    figure.add_trace(go.Scatter(x=times, y=sol.y[6,:],mode='lines',line=go.scatter.Line(color="blue"),showlegend=False, name='BSpline'))
    figure.update_xaxes(showline=True, linewidth=2, linecolor='black')
    figure.update_yaxes(showline=True, linewidth=2, linecolor='black')
    figure.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    figure.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
                          
    dataset_sim = pd.DataFrame({'day': times, 'BCS_sim': sol.y[6,:]})
    dataset_sim['ID'] = int(Ewe)
    header = ['ID','day','BCS_sim']
    dataset_sim.to_csv(downloadable_BCSSim_csv,columns = header,index=False)
    
    lista_files = [downloadable_parameters_csv,downloadable_BCSData_csv,downloadable_BCSSim_csv]
    downloadable_BCSDataSim_zipRand = downloadable_BCSDataSim_Nozip+str(Ewe)+".zip"
    with zipfile.ZipFile(downloadable_BCSDataSim_zipRand, 'w') as zipMe:        
        for file in lista_files:
            zipMe.write(file,basename(file), compress_type=zipfile.ZIP_DEFLATED)
    return figure,dffTFich.to_dict('records'),dffTFich.to_dict('records'),build_download_buttonBCS(downloadable_BCSDataSim_zipRand)




@app.server.route('/downloadable/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(
        os.path.join(root_dir, 'downloadable'), path
    )


if __name__ == '__main__':
    app.run_server(debug=True)