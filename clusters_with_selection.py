import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from sklearn.cluster import KMeans
import hdbscan
import pickle as pkl
import sys
from plotly.subplots import make_subplots


app = dash.Dash(__name__)

# read in the data files for plotting
plot_df = pd.read_csv(sys.argv[1],index_col=False) # file name containing the cluster characteristics. 'plot_df.csv' from analyze_clusters.ipynb

cluster_label = sys.argv[2]

if len(sys.argv) > 3:
    port = sys.argv[3] # the port number used on your browser. Default value can be 8888. If you want to run multiple apps at the same time, \
else:
    port = 8888

# list of clustering methods available
available_indicators = [
                'ICA (2 comp)',
                'ICA (3 comp)',
                'TSNE',
                'UMAP',
                ]

# title of the plot based on clustering method - Infoshield, TrafficLight, etc
if 'LSH' in cluster_label:
    title = 'Infoshield Clusters'
elif 'label' in cluster_label:
    title = 'TrafficLight'
else:
    title = cluster_label


# CHANGE DIMENSIONS ACCORDING TO COLUMNS IN YOUR DATASET
dimension_cols = ["Cluster Size", "Phone Count", "Loc Count", "Phone Entropy", "Loc Radius",\
            "Person Name Count",\
            "Valid URLs", "Invalid URLs", "Ads/week"]

app.layout = html.Div([
    html.Div(children=[
            html.H1(children='Micro-cluster feature embeddings'),
            dcc.Dropdown( # dropdown menu for choosing clustering method/ UMAP/ TSNE
                id='analysis-type',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='ICA (2 comp)'
            ),
            dcc.Graph( # scatter plot showing the micro-clusters using method chosen above
            id='micro-cluster-scatter',
            responsive=True)
            ], style={'width': '49%', 'display': 'inline-block'}),

    html.Div(children=[ # pair-wise scatter plot 
            html.H1(children='Cluster Characterization'),
            html.Div(children=title),
            dcc.Dropdown( # dropdown menu for choosing clustering method/ UMAP/ TSNE
                id='feature-type',
                options=[{'label': i, 'value': i} for i in dimension_cols],
                value='all',
                multi=True
            ),
            dcc.Graph(
                id='main-plot',
            )
            ],style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={"height":"100vh"})

# values to be displayed when hovering over a micro-cluster point in a plot
# REPLACE WITH COLUMNS FROM YOUR DATASET
hover_cols = ['Cluster Size Val','Phone Count Val', 'Loc Count Val', 'Loc Radius Val', \
             'Loc Radius Val','Person Name Count Val','Valid URLs Val', 'Invalid URLs Val',\
             'Ads/week Val', 'Num URLs Val', 'cluster_id']


'''
=========================CLUSTER CHARACTERIZATION PLOT==================================== 
'''

# function for highlighting in red the chosen cluster over all pair-plots
@app.callback( 
Output('main-plot','figure'), # output: 'main-plot' figure. Corresponding to app.layout
Input('main-plot','clickData'), # input: 'main-plot' with clicked data point. 
Input('micro-cluster-scatter','selectedData'), # input: 'micro-cluster-scatter' with points selected using lasso-tool
Input('feature-type', 'value')) 
def highlight_same_clusters(clickData, selected_clusters, selected_feats):
    if selected_feats == 'all':
        selected_feats = dimension_cols
    if selected_clusters: # if some selection of points has been made
        selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
        for item in selected_clusters['points']:
            selected_points.append(item['pointIndex'])
        to_plot = plot_df.iloc[selected_points] # plotting only the selected points
    else: # if no selection has been made
        to_plot = plot_df.copy()

    # dim_list = []
    # for col in dimension_cols:
    #     dim_list.append(dict(label=col, values=to_plot[col]))
    
    if clickData: # if a certain point has been clicked on (we want to track that point across pair-plots)
        cluster_id = clickData['points'][0]['customdata'][-1] # extract info from JSON format
        
        # pair-plots
        # fig = go.Figure(data=go.Splom(
        #     dimensions=dim_list, showlowerhalf=False,
        #     marker=dict(showscale=False, line_color='white', line_width=0.5)
        #     ))


        fig = px.scatter_matrix(to_plot, dimensions=selected_feats,opacity=0.6,\
                       hover_data=hover_cols)
        fig.update_layout(height=1600, width=1600)

        # highlight the current clicked point across all pair-plots in red
        # dim_list_update = []
        # for col in dimension_cols:
        #     dim_list_update.append(dict(label=col, values=to_plot[to_plot.cluster_id==cluster_id][col]))
        # fig.add_traces(go.Splom(
        #     dimensions=dim_list_update, showlowerhalf=False,
        #     marker=dict(showscale=False, line_color='white', line_width=0.5)
        #     ))
        fig.add_traces(
        px.scatter_matrix(to_plot[to_plot.cluster_id==cluster_id], \
                      dimensions=selected_feats).update_traces(marker_color="red").data
        )
    else: # if no point has been clicked on, show default with all points blue
        # fig = go.Figure(data=go.Splom(
        #     dimensions=dim_list, showlowerhalf=False,
        #     marker=dict(showscale=False, line_color='white', line_width=0.5)
        #     ))
        print(to_plot[selected_feats])
        fig = px.scatter_matrix(to_plot, dimensions=selected_feats,opacity=0.6,\
                       hover_data=hover_cols)
        fig.update_layout(height=1600, width=1600)
    return fig

'''
=============================MICRO-CLUSTER FEATURE EMBEDDING======================================
'''
# function for showing the clicked point from pair-plots on the scatter plot
@app.callback( 
Output('micro-cluster-scatter', 'figure'),
Input('analysis-type', 'value'),
Input('main-plot','clickData'),
Input('micro-cluster-scatter','clickData'),
Input('main-plot','selectedData'))
def update_graph(selected_clustering, clickData, clickData_from_mcs, selectedData):
    data_cols = ['x','y']

    # the ICA (2&3), TSNE and UMAP are precomputed and saved to disk for plotting
    if selected_clustering == available_indicators[0]: # ICA 2 comp
        df = pd.read_csv("data/is_ica.zip",index_col=False) #CHANGE THE DATA FILES ACCORDINGLY
        color_labels = None
    elif selected_clustering == available_indicators[1]: # ICA 3 comp
        df = pd.read_csv("data/is_ica_3.zip",index_col=False)
        color_labels = None
    elif selected_clustering == available_indicators[2]: # TSNE
        tsne_res = pkl.load(open("data/all_tsne_res.pkl",'rb'))
        # df = pd.read_csv("data/is_tsne.zip",index_col=False)
        color_labels =  None
    elif selected_clustering == available_indicators[3]: # UMAP
        umap_res = pkl.load(open("data/umap_res.pkl",'rb'))
        # df = pd.read_csv("data/is_umap.zip",index_col=False)
        color_labels = None

    if selected_clustering == available_indicators[1]:
        fig = px.scatter_matrix(df, dimensions=['x','y','z'], hover_data=hover_cols, height=1000)
    elif selected_clustering == available_indicators[2]: #TSNE with diff perplexity values. Displays a grid of plots
        perp_vals = list(tsne_res.keys())
        titles = []
        for p in perp_vals:
            titles.append(str(p))
        fig = make_subplots(
                    rows=1, cols=len(perp_vals),\
                    subplot_titles=tuple(titles), \
                    horizontal_spacing=0.01, vertical_spacing=0.01, \
                    shared_xaxes=True, shared_yaxes=True, \
                    x_title='Perplexity Values'
            )

        template_str = ""
        for i, col in enumerate(hover_cols):
            if i != len(hover_cols)-1:
                template_str += (col+":%{customdata["+str(i)+"]}<br>")
            else:
                template_str += (col+":%{customdata["+str(i)+"]}")

        for i, p in enumerate(perp_vals):
            fig.add_scatter(x=list(tsne_res[p].x), y=list(tsne_res[p].y), \
                                customdata=tsne_res[p][hover_cols], \
                                hovertemplate=template_str,\
                                mode='markers', marker={'opacity':0.3, 'color': 'blue'}, \
                                row=1, col=i+1)
                
            if i == 0:
                fig.layout.annotations[i].update(text=str(p))

        fig.update_layout(height=1600, width=1000, showlegend=False)

    elif selected_clustering == available_indicators[3]: # UMAP
        nbr_sizes = [10, 50, 100, 200, 500, 1000]
        mini_dists = [0, 0.01, 0.05, 0.1, 0.5, 1]

        titles = []
        for d in mini_dists:
            titles.append(str(d))

        template_str = ""
        for i, col in enumerate(hover_cols):
            if i != len(hover_cols)-1:
                template_str += (col+":%{customdata["+str(i)+"]}<br>")
            else:
                template_str += (col+":%{customdata["+str(i)+"]}")

        
        fig = make_subplots(rows=len(nbr_sizes), cols=len(mini_dists), \
                subplot_titles=tuple(titles[:6]), \
                horizontal_spacing=0.2, vertical_spacing=0.2, \
                shared_xaxes=True, shared_yaxes=True, \
                x_title='Min. distance', y_title='Nbrhood Size')
        for i in range(len(nbr_sizes)):
            for j in range(len(mini_dists)):
                fig.add_scatter(x=list(umap_res[i][j].x), y=list(umap_res[i][j].y), \
                                customdata=umap_res[i][j][hover_cols], \
                                hovertemplate=template_str,\
                                mode='markers', marker={'opacity':0.3, 'color': '#17becf'}, \
                                row=i+1, col=j+1)
                
                if j == len(mini_dists)-1:
                    fig.update_yaxes(side='right', title_text=str(nbr_sizes[i]), row=i+1, col=j+1)
                if i == 0:
                    fig.layout.annotations[j].update(text=str(mini_dists[j]))

        fig.update_layout(height=1500, width=1000, showlegend=False, title_text='Min. Distance')

    else:
        fig = px.scatter(df, x='x',y='y', hover_data=hover_cols, height=1600)

    if clickData:
        if (not selected_clustering in [available_indicators[2], available_indicators[3]]): # if a certain point has been clicked on (only if not TSNE and UMAP)
            cluster_id = clickData['points'][0]['customdata'][-1] # retrieve info of clicked point
            # if it's ICA with 3 comp, then pair-plot instead of regular scatter plot
            if selected_clustering == available_indicators[1]:
                fig.add_traces(
                    px.scatter_matrix(df[df.cluster_id==cluster_id], dimensions=['x','y','z'],
                                  hover_data=hover_cols).update_traces(marker_color="red",marker_size=20,marker_symbol='star').data
                    )
            else: # regular scatter plot
                fig.add_traces(
                    px.scatter(df[df.cluster_id==cluster_id], \
                                  x='x',y='y', hover_data=hover_cols).update_traces(marker_color="red",marker_size=20,marker_symbol='star').data
                    )
    
        else:
            cluster_id = clickData['points'][0]['customdata'][-1]
            if selected_clustering == available_indicators[2]: # TSNE
                for i, p in enumerate(perp_vals):
                    dd = tsne_res[p]
                    hover_trace = dict(type='scatter', \
                        x=list(dd[dd.cluster_id==cluster_id].x), y=list(dd[dd.cluster_id==cluster_id].y), \
                        customdata=dd[hover_cols], \
                        hovertemplate=template_str,\
                        mode='markers', marker={'symbol':'star', 'color': 'red','size':10})
                    fig.append_trace(hover_trace, 1, i+1)

            if selected_clustering == available_indicators[3]:
                nbr_sizes = [10, 50, 100, 200, 500, 1000]
                mini_dists = [0, 0.01, 0.05, 0.1, 0.5, 1]

                template_str = ""
                for i, col in enumerate(hover_cols):
                    if i != len(hover_cols)-1:
                        template_str += (col+":%{customdata["+str(i)+"]}<br>")
                    else:
                        template_str += (col+":%{customdata["+str(i)+"]}")

                for i in range(len(nbr_sizes)):
                    for j in range(len(mini_dists)):
                        dd = umap_res[i][j]
                        hover_trace = dict(type='scatter',\
                            x=list(dd[dd.cluster_id==cluster_id].x), y=list(dd[dd.cluster_id==cluster_id].y), \
                            customdata=dd[hover_cols], \
                            hovertemplate=template_str,\
                            mode='markers', marker={'symbol':'star', 'color': 'red','size':10}, \
                            )
                        fig.append_trace(hover_trace, i+1, j+1)
                        if j == len(mini_dists)-1:
                            fig.update_yaxes(side='right', title_text=str(nbr_sizes[i]), row=i+1, col=j+1)
                        if i == 0:
                            fig.layout.annotations[j].update(text=str(mini_dists[j]))

                fig.update_layout(height=1500, width=1000, showlegend=False, title_text='Min. Distance')



    if clickData_from_mcs: # only for UMAP and TSNE
        cluster_id = clickData_from_mcs['points'][0]['customdata'][-1] # retrieve info of hovered point
        if selected_clustering == available_indicators[2]: # TSNE
            for i, p in enumerate(perp_vals):
                dd = tsne_res[p]
                hover_trace = dict(type='scatter', \
                    x=list(dd[dd.cluster_id==cluster_id].x), y=list(dd[dd.cluster_id==cluster_id].y), \
                    customdata=dd[hover_cols], \
                    hovertemplate=template_str,\
                    mode='markers', marker={'opacity':1, 'color': 'black','size':10})
                fig.append_trace(hover_trace, 1, i+1)

        if selected_clustering == available_indicators[3]:
            nbr_sizes = [10, 50, 100, 200, 500, 1000]
            mini_dists = [0, 0.01, 0.05, 0.1, 0.5, 1]

            template_str = ""
            for i, col in enumerate(hover_cols):
                if i != len(hover_cols)-1:
                    template_str += (col+":%{customdata["+str(i)+"]}<br>")
                else:
                    template_str += (col+":%{customdata["+str(i)+"]}")

            for i in range(len(nbr_sizes)):
                for j in range(len(mini_dists)):
                    dd = umap_res[i][j]
                    hover_trace = dict(type='scatter',\
                        x=list(dd[dd.cluster_id==cluster_id].x), y=list(dd[dd.cluster_id==cluster_id].y), \
                        customdata=dd[hover_cols], \
                        hovertemplate=template_str,\
                        mode='markers', marker={'opacity':1, 'color': 'black','size':10}, \
                        )
                    fig.append_trace(hover_trace, i+1, j+1)
                    if j == len(mini_dists)-1:
                        fig.update_yaxes(side='right', title_text=str(nbr_sizes[i]), row=i+1, col=j+1)
                    if i == 0:
                        fig.layout.annotations[j].update(text=str(mini_dists[j]))

            fig.update_layout(height=1500, width=1000, showlegend=False, title_text='Min. Distance')

    

    if selectedData:
        selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
        for item in selectedData['points']:
            selected_points.append(item['pointIndex'])
        if (not selected_clustering in [available_indicators[2], available_indicators[3]]): # if a certain point has been clicked on (only if not TSNE and UMAP)
            # if it's ICA with 3 comp, then pair-plot instead of regular scatter plot
            if selected_clustering == available_indicators[1]:
                fig.add_traces(
                    px.scatter_matrix(df.iloc[selected_points], dimensions=['x','y','z'],
                                  hover_data=hover_cols).update_traces(marker_color="red",marker_size=20,marker_symbol='star').data
                    )
            else: # regular scatter plot
                fig.add_traces(
                    px.scatter(df.iloc[selected_points], \
                                  x='x',y='y', hover_data=hover_cols).update_traces(marker_color="red",marker_size=20,marker_symbol='star').data
                    )
    
        else:
            if selected_clustering == available_indicators[2]: # TSNE
                for i, p in enumerate(perp_vals):
                    dd = tsne_res[p]
                    hover_trace = dict(type='scatter', \
                        x=list(dd.iloc[selected_points].x), y=list(dd.iloc[selected_points].y), \
                        customdata=dd.iloc[selected_points][hover_cols], \
                        hovertemplate=template_str,\
                        mode='markers', marker={'color': 'red','size':10})
                    fig.append_trace(hover_trace, 1, i+1)

            if selected_clustering == available_indicators[3]:
                nbr_sizes = [10, 50, 100, 200, 500, 1000]
                mini_dists = [0, 0.01, 0.05, 0.1, 0.5, 1]

                template_str = ""
                for i, col in enumerate(hover_cols):
                    if i != len(hover_cols)-1:
                        template_str += (col+":%{customdata["+str(i)+"]}<br>")
                    else:
                        template_str += (col+":%{customdata["+str(i)+"]}")

                for i in range(len(nbr_sizes)):
                    for j in range(len(mini_dists)):
                        dd = umap_res[i][j]
                        hover_trace = dict(type='scatter',\
                            x=list(dd.iloc[selected_points].x), y=list(dd.iloc[selected_points].y), \
                            customdata=dd.iloc[selected_points][hover_cols], \
                            hovertemplate=template_str,\
                            mode='markers', marker={'symbol':'star', 'color': 'red','size':10}, \
                            )
                        fig.append_trace(hover_trace, i+1, j+1)
                        if j == len(mini_dists)-1:
                            fig.update_yaxes(side='right', title_text=str(nbr_sizes[i]), row=i+1, col=j+1)
                        if i == 0:
                            fig.layout.annotations[j].update(text=str(mini_dists[j]))

                fig.update_layout(height=1500, width=1000, showlegend=False, title_text='Min. Distance')


    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=port)