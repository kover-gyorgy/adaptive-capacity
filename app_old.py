# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:25:45 2020

@author: Kövér György
"""

import os
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

# external JavaScript files
external_scripts = [
    { "src": "https://use.fontawesome.com/ca367b0f57.js" },
    {
        "src": "https://code.jquery.com/jquery-3.5.1.min.js",
        "integrity": "sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0=",
        "crossorigin": "anonymous"
    },
]

# external CSS stylesheets
external_stylesheets = [
    "assets/font-awesome.min.css"
]



table_header_style = {
    "backgroundColor": "rgb(2,21,70)",
    "color": "white",
    "textAlign": "center",
}

APP_PATH = str(pl.Path(__file__).parent.resolve())

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
    data = [['Mean',paramCor.kb1.mean()*1000,paramCor.kb2.mean()*1000,paramCor.kb3.mean()*1000,paramCor.kp1.mean()*1000,paramCor.kp2.mean()*1000,paramCor.kp3.mean()*1000,paramCor.res.mean()],
            ['Sd',paramCor.kb1.std()*1000,paramCor.kb2.std()*1000,paramCor.kb3.std()*1000,paramCor.kp1.std()*1000,paramCor.kp2.std()*1000,paramCor.kp3.std()*1000,paramCor.res.std()],
            ['Min',paramCor.kb1.min()*1000,paramCor.kb2.min()*1000,paramCor.kb3.min()*1000,paramCor.kp1.min()*1000,paramCor.kp2.min()*1000,paramCor.kp3.min()*1000,paramCor.res.min()],
            ['Max',paramCor.kb1.max()*1000,paramCor.kb2.max()*1000,paramCor.kb3.max()*1000,paramCor.kp1.max()*1000,paramCor.kp2.max()*1000,paramCor.kp3.max()*1000,paramCor.res.max()],
            ['Q1',paramCor.kb1.quantile(0.25)*1000,paramCor.kb2.quantile(0.25)*1000,paramCor.kb3.quantile(0.25)*1000,paramCor.kp1.quantile(0.25)*1000,paramCor.kp2.quantile(0.25)*1000,paramCor.kp3.quantile(0.25),paramCor.res.quantile(0.25)],
            ['Median',paramCor.kb1.median()*1000,paramCor.kb2.median(),paramCor.kb3.median()*1000,paramCor.kp1.median()*1000,paramCor.kp2.median()*1000,paramCor.kp3.median()*1000,paramCor.res.median()],
            ['Q3',paramCor.kb1.quantile(0.75)*1000,paramCor.kb2.quantile(0.75)*1000,paramCor.kb3.quantile(0.75)*1000,paramCor.kp1.quantile(0.75)*1000,paramCor.kp2.quantile(0.75)*1000,paramCor.kp3.quantile(0.75)*1000,paramCor.res.quantile(0.75)]
          ]
    lParanalKbkp = pd.DataFrame(data,columns=['stat','kb1','kb2','kb3','kp1','kp2','kp3','RSS'])   
 
    data = [['Mean',paramCor.tb1.mean(),paramCor.tb2.mean(),paramCor.tb3.mean(),paramCor.DT1.mean(),paramCor.DT2.mean(),paramCor.DT3.mean()],
            ['Sd' ,paramCor.tb1.std(),paramCor.tb2.std(),paramCor.tb3.std(),paramCor.DT1.std(),paramCor.DT2.std(),paramCor.DT3.std()],
            ['Min',paramCor.tb1.min(),paramCor.tb2.min(),paramCor.tb3.min(),paramCor.DT1.min(),paramCor.DT2.min(),paramCor.DT3.min()],
            ['Max',paramCor.tb1.max(),paramCor.tb2.max(),paramCor.tb3.max(),paramCor.DT1.max(),paramCor.DT2.max(),paramCor.DT3.max()],
            ['Q1',paramCor.tb1.quantile(0.25),paramCor.tb2.quantile(0.25),paramCor.tb3.quantile(0.25),paramCor.DT1.quantile(0.25),paramCor.DT2.quantile(0.25),paramCor.DT3.quantile(0.25)],
            ['Median',paramCor.tb1.median(),paramCor.tb2.median(),paramCor.tb3.median(),paramCor.DT1.median(),paramCor.DT2.median(),paramCor.DT3.median()],
            ['Q3',paramCor.tb1.quantile(0.75),paramCor.tb2.quantile(0.75),paramCor.tb3.quantile(0.75),paramCor.DT1.quantile(0.75),paramCor.DT2.quantile(0.75),paramCor.DT3.quantile(0.75)]
          ]
    lParanalTime = pd.DataFrame(data,columns=['stat','tb1','tb2','tb3','DT1','DT2','DT3'])  
    return lParanalKbkp, lParanalTime

ParanalKbkp, ParanalTime = descriptives()

xT = ['kb1', 'kb2', 'kb3', 'kp1', 'kp2', 'kp3', 'DT1', 'DT2', 'DT3']

def heat_plot(): 
    paramCorDrop = paramCor[['kb1', 'kb2', 'kb3', 'kp1', 'kp2', 'kp3', 'DT1', 'DT2', 'DT3']]
    corrMatrix  = paramCorDrop.corr()
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
                        hovertemplate =
                        '<br>X: %{x}<br>'+
                        '<br>Y: %{y}<br>'+
                        '<br>r: %{text}<br>',
                        #text = ['0.5'],
                        xgap=1,
                        ygap=1,
                        colorscale=col.diverging.RdBu,
                        zmax=1,
                        zmin=-1,
                        zmid=0
                        ))
    figHeat.update_layout(
        title={
            'text': 'Correlations',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
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
        width=700,
        height=700
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
    scX = paramCor[sc_X] * 1000 
    scY = paramCor[sc_Y] * 1000
    if sc_X in ['DT1', 'DT2', 'DT3']:
        scX = paramCor[sc_X] 
    if sc_Y in ['DT1', 'DT2', 'DT3']:
        scY = paramCor[sc_Y]
        
    figure = go.Figure(data=go.Scatter(x=scX, y=scY, mode='markers',showlegend=False, name='BCS',
        marker=dict(
            color='rgba(0, 0, 0, 0.0)',
            size=6,
            line=dict(
                color='Black',
                width=1
            )
        )))
    figure.update_layout(xaxis_title=sc_X, yaxis_title=sc_Y)
    figure.update_layout(
        title={
        'text': 'Scatter Plot',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        width=700,
        height=700)
        
    return figure


def hist_graph(name):
    
    if name== 'kb1': 
        xx = paramCor.kb1*1000
    if name== 'kb2': 
        xx = paramCor.kb2*1000
    if name== 'kb3': 
        xx = paramCor.kb3*1000
    if name== 'kp1': 
        xx = paramCor.kp1*1000
    if name== 'kp2': 
        xx = paramCor.kp2*1000
    if name== 'kp3': 
        xx = paramCor.kp3*1000
    if name== 'RSS': 
        xx = paramCor.res
    if name== 'tb1': 
        xx = paramCor.tb1
    if name== 'tb2': 
        xx = paramCor.tb2
    if name== 'tb3': 
        xx = paramCor.tb3
    if name== 'DeltaT1': 
        xx = paramCor.DeltaT1
    if name== 'DeltaT2': 
        xx = paramCor.DeltaT2
    if name== 'DeltaT3': 
        xx = paramCor.DeltaT3

    figure = go.Figure(data=[go.Histogram(x=xx, nbinsx=300,
        marker=dict(
            color='LightSkyBlue',
            line=dict(
                color='black',
                width=1
            )))])
    figure.update_layout(
        title={
        'text': name,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
       yaxis_title_text='Frequency', # yaxis label
       autosize=False,
       width=1000,
       height=700
    )
    return figure



app = dash.Dash(__name__,
                external_scripts=external_scripts,
                external_stylesheets=external_stylesheets)

app.title = 'adaptive capacity'
server = app.server

app.layout = html.Div([
    html.Div(
        id="home",
        style={ 'display': 'none' }
    ),
   
   
    html.Div(
        className="pkcalc-banner",
        children=[
            html.H2("Quantification of animal adaptive capacity"),
        ],
    ),
                                html.P(
                                    id="instructions",
                                    children="Authors: X. Y."
                                ),
                                html.P(
                                    id="instructions2",
                                    children="The ewes are groupped by the number of their parities."
                                    " Select a group then select a ewe by the rightmost dropdown list."
                                ),
                                        
    # wrapper required for scrolling
    html.Div(
        id="dropdown-controls-wrapper",
        children=[        
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
                         { 'label': 'All the Ewes in the dataset', 'value': 'ALL' }
                        ],
                value='G123',
                clearable=False,
                optionHeight = 25),
        ],
        style={'width': '35%',
               'display': 'inline-block'}),
            
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
        style={'width': '25%',
               'display': 'inline-block'}),
      
                    # scroll to top button  
                    dcc.Link(
                        className="scroll-to-top-container",
                        href='#home',
                        children=[
                            html.A(
                                children=[
                                    html.I(className='fa fa-arrow-up', **{'aria-hidden': 'true'})
                                ],
                                href='#home',
                                title="Scroll to the top"
                            ),
                        ]
                    )
                   
                ],
            ),
        ],
    ),      
            
    dcc.Graph(id='BCS-graph',
                config={
                   'displayModeBar': False
                }),
    html.Div(
        [
            dash_table.DataTable(
                id='table',
                columns=[{"name":"Physiological stages for BW", "id": "BW.time"},
                         {"name":"Physiological stages for BCS", "id": "BCS.time"},
                         {"name":"Year", "id": "Year"},
                         {"name":"Parity", "id": "Parity"},
                         {"name":"Age at first mating", "id": "Age_mating"},
                         {"name":"Litter size", "id": "LS"},
                         {"name":"Numerical stages", "id": "state.num"},
                         {"name":"Days", "id": "day"},
                         {"name":"BW", "id": "BW"},
                         {"name":"BCS", "id": "BCS"},
                         {"name":"BCS maximum", "id": "BCSm"},
                         {"name":"BCS before FDA", "id": "BCS1"}
                        ],
                style_cell={'textAlign': 'center', 'whiteSpace' : 'normal', 'maxWidth': '120px','padding': '2py' },
                style_table={'textAlign': 'center', 'overflowX': 'scroll','maxHeight': 400},
                data=pkdata.to_dict('records'),
                style_header=table_header_style,
                fixed_rows={'headers': True, 'data': 0}
            )
        ],
        ),
                                html.P(
                                    id="instructionsA",
                                    children="Estimation of the model parameters"
                                ),
                                html.P(
                                    id="instructionsA2",
                                    children="The table below contains the identification of the ewe,"
                                    " the number of parities she had, the values of tb te kp kb and the value of the residual (opti)."
                                ),
    html.Div(
        [
            dash_table.DataTable(
                id='tableFich',
                columns=[{"name":"Ewe ID", "id": "ID"},
                         {"name":"Parity", "id": "parity"},
                         {"name":"kb", "id": "kb", 'type': 'numeric', 'format': Format(precision=8, scheme=Scheme.fixed)},
                         {"name":"kp", "id": "kp", 'type': 'numeric', 'format': Format(precision=8, scheme=Scheme.fixed)},
                         {"name":"tb", "id": "tb"},
                         {"name":"te", "id": "te"},
                         {"name":"BCSm", "id": "BCSm"},
                         {"name":"Pmax", "id": "Pmax"},
                         {"name":"opti", "id": "opt", 'type': 'numeric', 'format': Format(precision=8, scheme=Scheme.fixed)}
                        ],
                style_cell={'textAlign': 'center', 'whiteSpace' : 'normal', 'maxWidth': '120px','padding': '2px' },
                style_table={'textAlign': 'center', 'overflowX': 'scroll','maxHeight': 400},
                data=fichierfinal.to_dict('records'),
                style_header=table_header_style,
            )
        ],
        ),
                                html.P(
                                    id="instructionsB",
                                    children="Simulation based on the estimated parameters"
                                ),
                                html.P(
                                    id="instructionsB2",
                                    children="The last step was made to select only a portion of ewes, "
                                    "for example ewes that do not match correctly with the model "
                                    "in order to observe their parameters associated with a graph (below)."
                                ),
    html.Div(
        [
            dash_table.DataTable(
                id='tableKbkp',
                columns=[{"name":"", "id": "stat"},
                         {"name":"kb1", "id": "kb1", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                         {"name":"kb2", "id": "kb2", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                         {"name":"kb3", "id": "kb3", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                         {"name":"kp1", "id": "kp1", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                         {"name":"kp2", "id": "kp2", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                         {"name":"kp3", "id": "kp3", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)},
                         {"name":"RSS", "id": "RSS", 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)}
                        ],
                style_cell={'textAlign': 'center', 'whiteSpace' : 'normal', 'maxWidth': '120px','padding': '2py' },
                style_table={'textAlign': 'center', 'overflowX': 'scroll','maxHeight': 400},
                data=ParanalKbkp.to_dict('records'),
                style_header=table_header_style,
            )
        ],
        ),
    html.Div(
        [
            dash_table.DataTable(
                id='tableTime',
                columns=[{"name":"",        "id": "stat"},
                         {"name":"tb1",     "id": "tb1", 'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed)},
                         {"name":"tb2",     "id": "tb2", 'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed)},
                         {"name":"tb3",     "id": "tb3", 'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed)},
                         {"name":"DeltaT1", "id": "DT1", 'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed)},
                         {"name":"DeltaT2", "id": "DT2", 'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed)},
                         {"name":"DeltaT3", "id": "DT3", 'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed)}
                        ],
                style_cell={'textAlign': 'center', 'whiteSpace' : 'normal', 'maxWidth': '120px','padding': '2py' },
                style_table={'textAlign': 'center', 'overflowX': 'scroll','maxHeight': 400},
                data=ParanalTime.to_dict('records'),
                style_header=table_header_style,
            )
        ],
        ),
    html.Div(
        [         
                dcc.Graph(
                    id = 'graphHist',    
                    figure=hist_graph(hist_Var),
                    config={'displayModeBar': False}
                    )
        ],
        style={ 'display': 'inline-block'}
        ),
    html.Div(
        [
                 html.P(
                                    children="Select a trait to view the histogram"
                                ),
                dcc.RadioItems(id = 'input-radio-button',
                                      options = [dict(label = 'kb1', value = 'kb1'),
                                                 dict(label = 'kb2', value = 'kb2'),
                                                 dict(label = 'kb3', value = 'kb3'),
                                                 dict(label = 'kp1', value = 'kp1'),
                                                 dict(label = 'kp2', value = 'kp2'),
                                                 dict(label = 'kp3', value = 'kp3'),
                                                 dict(label = 'RSS', value = 'RSS'),
                                                 dict(label = 'tb1', value = 'tb1'),
                                                 dict(label = 'tb2', value = 'tb2'),
                                                 dict(label = 'tb3', value = 'tb3'),
                                                 dict(label = 'DeltaT1', value = 'DeltaT1'),
                                                 dict(label = 'DeltaT2', value = 'DeltaT2'),
                                                 dict(label = 'DeltaT3', value = 'DeltaT3')
                                                ],
                                      value = 'RSS',
              labelStyle={"padding-left": "10px",'display': 'inline-block'},
              style={"padding-left": "50px", "max-width": "1000px", "margin": "center"},
              ),
        ],
        style={ 'display': 'inline-block'}
        ),
    html.Div(
        [         
                dcc.Graph(
                    id = 'graphHeat',    
                    #figure=figHeat, 
                    figure=heat_plot(), 
                    config={'displayModeBar': False}
                    )
        ],
#        style={ 'display': 'inline-block'}
        ),    
        
    html.Div(
        [   html.P("""Horizontal axis""",
                            style={'margin-right': '1em'})
        ],
        style={'display': 'inline-block'}
        ),
    html.Div(
        [
            dcc.Dropdown(
                id="XVar",
                options=[{
                    'label': i,
                    'value': i
                } for i in xT],
                value='kb1',
                clearable=False,
                optionHeight = 24,style=dict(display='inline-block',width='60%',verticalAlign="middle"))
        ],
        style={'width': '15%',
               'display': 'inline-block'}),
        
        
    html.Div(
        [   html.P("""Vertical axis""",
                            style={'margin-right': '1em'})
        ],
        style={'display': 'inline-block'}),
    html.Div(
        [
            dcc.Dropdown(
                id="YVar",
                options=[{
                    'label': i,
                    'value': i
                } for i in xT],
                value='kp1',
                clearable=False,
                optionHeight = 24,style=dict(display='inline-block',width='60%',verticalAlign="middle"))
        ],
        style={'width': '15%',
               'display': 'inline-block'}),
        
        
    dcc.Graph(id='Scatter-graph',  
                figure=scatter_plot(),
                config={
                   'displayModeBar': False
                }),
])

#
#@app.callback(
#    dash.dependencies.Output('graphHist', 'figure'),
#    [dash.dependencies.Input("input-radio-button", "value")],
#)
#def update_date_radioItem(name):
#    
#    global hist_Var 
#    hist_Var = name
#    figure = hist_graph(hist_Var)
#    return figure

#
#@app.callback(
#    dash.dependencies.Output('Scatter-graph', 'figure'),
#    [dash.dependencies.Input('XVar', 'value'),dash.dependencies.Input('YVar', 'value')])
#def scatter_XYVar(pXVar,pYVar):
#    global sc_X
#    global sc_Y
#    sc_X = pXVar
#    sc_Y = pYVar
#    fig = scatter_plot()
#    return fig


@app.callback(
    [    
    dash.dependencies.Output("PKSubject", "options"),
    dash.dependencies.Output("tableKbkp", "data"),
    dash.dependencies.Output("tableTime", "data"),
    dash.dependencies.Output('graphHist', 'figure'),
    dash.dependencies.Output('graphHeat', 'figure'),
    dash.dependencies.Output('Scatter-graph', 'figure')
    ],
    [dash.dependencies.Input("parity", "value"),dash.dependencies.Input("input-radio-button", "value"),
     dash.dependencies.Input('XVar', 'value'),dash.dependencies.Input('YVar', 'value')],
)
def update_date_dropdown(pGroup,name,pXVar,pYVar):
    global paramCor
    global hist_Var 
    global sc_X
    global sc_Y
    hist_Var = name
    sc_X = pXVar
    sc_Y = pYVar
    if pGroup == 'ALL': 
        paramCor = paramCor_Orig.copy()
    else: 
        paramCor = paramCor_Orig[paramCor_Orig.Group == pGroup]
    subjectsG = paramCor.ID.unique()
    ParanalKbkp, ParanalTime = descriptives()
    histfigure = hist_graph(hist_Var)
    heatfigure = heat_plot()
    scatterfigure=scatter_plot()
    return [{'label': i, 'value': i} for i in subjectsG],ParanalKbkp.to_dict('records'),ParanalTime.to_dict('records'),histfigure,heatfigure,scatterfigure


@app.callback(
    [
    dash.dependencies.Output('BCS-graph', 'figure'),
    dash.dependencies.Output('table', 'data'),
    dash.dependencies.Output('tableFich', 'data'),
    ],
    [dash.dependencies.Input('PKSubject', 'value')])
def display_table(Ewe): 
    global tb1, tb2, tb3, te1, te2, te3, BCSm, Pmax
    dffTFich = fichierfinal[fichierfinal.ID==int(Ewe)]
    dffT = pkdata[pkdata.Ewe==int(Ewe)]
    #print(len(dff))
    dff = dffT[pd.notnull(dffT.BCS1)]
    #print(len(dff))
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
    figure.update_layout(
        title={
        'text': 'Ewe {}'.format(Ewe),
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    figure.update_xaxes(range=[1, 1100])
    figure.update_yaxes(range=[2, 4])

    figure.update_layout(
        height=500
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
                          
            

    
    return figure,dffT.to_dict('records'),dffTFich.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True)