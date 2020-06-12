# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:21:50 2020

@author: Kövér György
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:02:24 2020

@author: Kövér György
"""

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
import pandas as pd
import numpy as np
import scipy.interpolate as interpolate

table_header_style = {
    "backgroundColor": "rgb(2,21,70)",
    "color": "white",
    "textAlign": "center",
}

APP_PATH = str(pl.Path(__file__).parent.resolve())

pourcorr3 = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "pourcorr3.csv")))
fichierfinal = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "fichierfinal.csv")))
pkdata = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "Mydata.csv")))


pourff123 = pourcorr3[(pd.notnull(pourcorr3.kb1))  & (pd.notnull(pourcorr3.kb2)) & (pd.notnull(pourcorr3.kb3))]
pourff12  = pourcorr3[(pd.notnull(pourcorr3.kb1))  & (pd.notnull(pourcorr3.kb2)) & (pd.isnull(pourcorr3.kb3))]
pourff1   = pourcorr3[(pd.notnull(pourcorr3.kb1))  & (pd.isnull(pourcorr3.kb2)) & (pd.isnull(pourcorr3.kb3))]
subjects = pourff123.ID.unique()


app = dash.Dash()
app.title = 'adaptive capacity'

app.layout = html.Div([
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
    
    html.Div(
        [
            dcc.Dropdown(
                id="parity",
                options=[{ 'label': 'Ewes with only the 1st parity', 'value': 'P1' },
                         { 'label': 'Ewes with the 1st and 2nd parity', 'value': 'P12' },
                         { 'label': 'Ewes with three parity', 'value': 'P123' }
                        ],
                value='P123',
                clearable=False,
                optionHeight = 25),
        ],
        style={'width': '35%',
               'display': 'inline-block'}
    ),
    
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
               'display': 'inline-block'}
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
])


@app.callback(
    dash.dependencies.Output("PKSubject", "options"),
    [dash.dependencies.Input("parity", "value")],
)
def update_date_dropdown(name):
    
    subjectsG = pourff123.ID.unique()
    if name== 'P1': 
        subjectsG = pourff1.ID.unique()
    if name== 'P12': 
        subjectsG = pourff12.ID.unique()
    return [{'label': i, 'value': i} for i in subjectsG]


@app.callback(
    [
    dash.dependencies.Output('BCS-graph', 'figure'),
    dash.dependencies.Output('table', 'data'),
    dash.dependencies.Output('tableFich', 'data'),
    ],
    [dash.dependencies.Input('PKSubject', 'value')])
def display_table(Ewe):
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
    figure.update_layout(xaxis_title="day", yaxis_title="BCS")
    figure.update_layout(
        title={
        'text': 'Ewe {}'.format(Ewe),
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    figure.update_yaxes(range=[2, 4])

    x = np.array(dff.day)
    y = np.array(dff.BCS1)
    t2 = x[0::2].copy()
    t3 = t2[:-1].copy()
    t4 = np.delete(t3,[0])
    t, c, k = interpolate.splrep(x, y,t=t4, s=0.1, k=3)
    xmin, xmax = x.min(), x.max()
    xx = np.linspace(xmin, xmax, xmax-xmin+1 )
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    figure.add_trace(go.Scatter(x=xx, y=spline(xx),mode='lines',line=go.scatter.Line(color="blue"),showlegend=False, name='BSpline'))
    
    fichff = fichierfinal[(fichierfinal.ID==int(Ewe)) & (fichierfinal.parity==1) & (pd.notnull(fichierfinal.tb)) ]
    fichff1_empty = len(fichff)==0
    tb1 = np.array(fichff.tb)
    tb1y = spline(tb1)
    te1 = np.array(fichff.te)
    te1y = spline(te1)
    figure.add_trace(go.Scatter(x=tb1, y=tb1y,mode='markers+text',showlegend=False, name='t<sub>b1</sub>', text = 't<sub>b1</sub>',textposition='top center',
    textfont=dict(
        size=16,
        color="red"
    ),
        marker=dict(
            color='red',
            size=10,
            line=dict(
                color='red',
                width=1
            )
        )))
    figure.add_trace(go.Scatter(x=te1, y=te1y,mode='markers+text',showlegend=False, name='te1', text = 't<sub>e1</sub>',textposition='top center',
    textfont=dict(
        size=16,
        color="green"
    ),
        marker=dict(
            color='green',
            size=10,
            line=dict(
                color='green',
                width=1
            )
        )))                       
            

    fichff = fichierfinal[(fichierfinal.ID==int(Ewe)) & (fichierfinal.parity==2) & (pd.notnull(fichierfinal.tb)) ]
    fichff2_empty = len(fichff)==0
    tb2 = np.array(fichff.tb)
    te2 = np.array(fichff.te)
    tb2y = spline(tb2)
    te2y = spline(te2)
    figure.add_trace(go.Scatter(x=tb2, y=tb2y,mode='markers+text',showlegend=False, name='tb2', text = 't<sub>b2</sub>',textposition='top center',
    textfont=dict(
        size=16,
        color="red"
    ),
        marker=dict(
            color='red',
            size=10,
            line=dict(
                color='red',
                width=1
            )
        )))
    figure.add_trace(go.Scatter(x=te2, y=te2y,mode='markers+text',showlegend=False, name='te2', text = 't<sub>e2</sub>',textposition='top center',
    textfont=dict(
        size=16,
        color="green"
    ),
        marker=dict(
            color='green',
            size=10,
            line=dict(
                color='green',
                width=1
            )
        )))
    fichff = fichierfinal[(fichierfinal.ID==int(Ewe)) & (fichierfinal.parity==3) & (pd.notnull(fichierfinal.tb)) ]
    fichff3_empty = len(fichff)==0
    tb3 = np.array(fichff.tb)
    te3 = np.array(fichff.te)
    tb3y = spline(tb3)
    te3y = spline(te3)
    figure.add_trace(go.Scatter(x=tb3, y=tb3y,mode='markers+text',showlegend=False, name='tb3', text = 't<sub>b3</sub>',textposition='top center',
    textfont=dict(
        size=16,
        color="red"
    ),
        marker=dict(
            color='red',
            size=10,
            line=dict(
                color='red',
                width=1
            )
        )))
    figure.add_trace(go.Scatter(x=te3, y=te3y,mode='markers+text',showlegend=False, name='te3', text = 't<sub>e3</sub>',textposition='top center',
    textfont=dict(
        size=16,
        color="green"
    ),
        marker=dict(
            color='green',
            size=10,
            line=dict(
                color='green',
                width=1
            )
        )))
    if not fichff1_empty :       
        figure.add_annotation(dict(
            x=te1[0]-5,
            y=te1y[0]-0.1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=5,
            arrowsize=1.5,
            ax=tb1[0],
            ay=tb1y[0]-0.1,
            text=""))  
        figure.add_annotation(dict(
            x=tb1[0]+(te1[0]-5 - tb1[0] )/2, 
            y=tb1y[0]-0.1+(te1y[0]-tb1y[0])/2-0.12, 
            xref="x",
            yref="y",
            showarrow=False,
            text="K<sub>p1</sub><br>K<sub>b1</sub>",
            font=dict(size=16,color="black")
            ))    
    
    if not fichff2_empty :               
        figure.add_annotation(dict(
            x=te2[0]-5,
            y=te2y[0]-0.1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=5,
            arrowsize=1.5,
            ax=tb2[0],
            ay=tb2y[0]-0.1,
            text=""))   
        figure.add_annotation(dict(
            x=tb2[0]+(te2[0]-5 - tb2[0] )/2, 
            y=tb2y[0]-0.1+(te2y[0]-tb2y[0])/2-0.12, 
            xref="x",
            yref="y",
            showarrow=False,
            text="K<sub>p2</sub><br>K<sub>b2</sub>",
            font=dict(size=16,color="black")
            ))       
    
    if not fichff3_empty :
        figure.add_annotation(dict(
            x=te3[0]-5,
            y=te3y[0]-0.1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=5,
            arrowsize=1.5,
            ax=tb3[0],
            ay=tb3y[0]-0.1,
            text=""))  
        figure.add_annotation(dict(
            x=tb3[0]+(te3[0]-5 - tb3[0] )/2, 
            y=tb3y[0]-0.1+(te3y[0]-tb3y[0])/2-0.12, 
            xref="x",
            yref="y",
            showarrow=False,
            text="K<sub>p3</sub><br>K<sub>b3</sub>",
            font=dict(size=16,color="black")
            ))       
    
    if (not fichff1_empty) & (not fichff2_empty) :
        figure.add_annotation(dict(
            x=tb2[0]-5,
            y=tb2y[0]-0.1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=5,
            arrowsize=1.5,
            ax=te1[0],
            ay=te1y[0]-0.1,
            text=""))    
        figure.add_annotation(dict(
            x=te1[0]+(tb2[0]-5 - te1[0] )/2, 
            y=te1y[0]-0.1+(tb2y[0]-te1y[0])/2-0.12, 
            xref="x",
            yref="y",
            showarrow=False,
            text="k<sub>b1</sub>",
            font=dict(size=16,color="black")
            ))    
    
    if (not fichff2_empty) & (not fichff3_empty) :
        figure.add_annotation(dict(
            x=tb3[0]-5,
            y=tb3y[0]-0.1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=5,
            arrowsize=1.5,
            ax=te2[0],
            ay=te2y[0]-0.1,
            text=""))     
        figure.add_annotation(dict(
            x=te2[0]+(tb3[0]-5 - te2[0] )/2, 
            y=te2y[0]-0.1+(tb3y[0]-te2y[0])/2-0.12, 
            xref="x",
            yref="y",
            showarrow=False,
            text="k<sub>b2</sub>",
            font=dict(size=16,color="black")
            ))    
    
    if (not fichff1_empty) & (fichff2_empty) & (not fichff3_empty) :
        figure.add_annotation(dict(
            x=tb3[0]-5,
            y=tb3y[0]-0.1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=5,
            arrowsize=1.5,
            ax=te1[0],
            ay=te1y[0]-0.1,
            text=""))   
        figure.add_annotation(dict(
            x=te1[0]+(tb3[0]-5 - te1[0] )/2, 
            y=te1y[0]-0.1+(tb3y[0]-te1y[0])/2-0.12, 
            xref="x",
            yref="y",
            showarrow=False,
            text="k<sub>b1-3</sub>",
            font=dict(size=16,color="black")
            ))    
            
    
    
    return figure,dffT.to_dict('records'),dffTFich.to_dict('records')
    


if __name__ == '__main__':
    app.run_server(debug=True)