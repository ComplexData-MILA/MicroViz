'''
Author: Pratheeksha Nair
'''
import dash
import re
from dash import dcc
from dash import html, callback_context
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from sklearn.cluster import KMeans
import hdbscan
import pickle as pkl
import sys
import visdcc
import dash_daq as daq
import numpy as np
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import dash_mantine_components as dmc
import json, os, ast

from utils import get_summary, plotlyfromjson, get_location_time_info, get_feature_weights
# from read_docx import get_full_text



# list of clustering methods available
available_indicators = [
				'ICA (2 components)',
				'TSNE',
				'UMAP',
				]

# dimension_cols = ["Cluster Size", "Phone Count", "Loc Count", "Phone Entropy", "Loc Radius",\
# 			"Ads/week", "Num Social", 'Person Name Count']
dimension_cols = ["Cluster Size", "Phone Count", "Loc Count", "Phone Entropy", "Loc Radius",\
			"Person Name Count", "Valid URLs", "Invalid URLs", "Ads/week"]

short_dim_cols = dimension_cols.copy()

hover_cols = ['Cluster Size Val','Phone Count Val', 'Loc Count Val', 'Loc Radius Val', \
			 'Loc Radius Val','Person Name Count Val','Valid URLs Val', 'Invalid URLs Val',\
			 'Ads/week Val', 'Num Social Val', 'cluster_id']

# read in the data files for plotting
path_name = 'marinus_canada_full/'
global current_sum 
current_sum = 0

if 'locanto' in path_name:
	full_df = pd.read_csv(path_name+"locanto_7k_with_entities_LSH_labels.csv",index_col=False) # file name containing the cluster characteristics. 'plot_df.csv' from analyze_clusters.ipynb
	plot_df = pd.read_csv(path_name+"plot_df.csv",index_col=False)
	# plot_df = plot_df[plot_df['Cluster Size Val'] > 10]
	# print(full_df.columns)
	full_df.rename(columns={'Name':'name', 'location':'cleaned_loc'}, inplace=True)
	cities_df = full_df[['cleaned_loc', 'geolocation']].set_index('cleaned_loc')
	weak_label_map = pkl.load(open(path_name+"final_labels.pkl",'rb'))
	marker_size = 10
	full_df['Weak Label'] = full_df[cluster_label].apply(lambda x: weak_label_map[x])
	micro_to_meta = full_df[[cluster_label, 'Meta label']].set_index(cluster_label).to_dict()['Meta label']
	plot_df['Meta Cluster ID'] = plot_df['cluster_id'].apply(lambda x: micro_to_meta[x])
	has_labels = False
	cluster_label = 'LSH label'
elif 'trimmed' in path_name:
	cluster_label = 'LSH label'
	full_df = pd.read_csv(path_name+"CanadianAds_names_trimmed_LSH_labels.csv",index_col=False, nrows=500)
	# top_clusters = full_df.groupby(cluster_label).size().sort_values()[-1000:].index.values
	# full_df = full_df[full_df[cluster_label].isin(top_clusters)]
	plot_df = pd.read_csv(path_name+"plot_df.csv",index_col=False)
	plot_df = plot_df[plot_df.cluster_id.isin(full_df['LSH label'])]

	cities = pd.read_csv(path_name+"cities.csv", index_col='id')
	full_df['location'] = full_df['city_id'].apply(lambda x: cities.loc[x]['name'])
	full_df['geolocation'] = full_df['city_id'].apply(lambda x: str(cities.loc[x].xcoord) + " " + str(cities.loc[x].ycoord))
	full_df['img_urls'] = None
	# micro_to_true = full_df[[cluster_label, 'label']].set_index(cluster_label).to_dict()['label']
	# cities = pd.read_csv(path_name+"cities.csv")
	# print(full_df['location'])
	# full_df['city'] = full_df['location'].apply(lambda x: cities[cities.id==x].name)
	full_df.rename(columns={'Name':'name', 'location':'cleaned_loc', \
		'phone':'phone_num', 'date_posted':'post_date', 'body':'description'}, inplace=True)

	cities_df = full_df[['cleaned_loc', 'geolocation']].set_index('cleaned_loc')
	marker_size = 5
	has_labels = False
	micro_to_meta = full_df[[cluster_label, 'Meta label']].set_index(cluster_label).to_dict()['Meta label']
	plot_df['Meta Cluster ID'] = plot_df['cluster_id'].apply(lambda x: micro_to_meta[x])

elif 'full' in path_name:
	cluster_label = 'LSH label'
	full_df = pd.read_csv(path_name+"ht2018_no_dupl_trimmed_LSH_labels.csv",index_col=False)

	# top_clusters = full_df.groupby(cluster_label).size().sort_values()[-1000:].index.values
	# full_df = full_df[full_df[cluster_label].isin(top_clusters)]
	plot_df = pd.read_csv(path_name+"plot_df.csv",index_col=False)
	# plot_df = plot_df[plot_df.cluster_id.isin(full_df['LSH label'])]

	cities = pd.read_csv(path_name+"cities.csv", index_col='id')
	full_df['location'] = full_df['city_id'].apply(lambda x: cities.loc[x]['name'])
	full_df['geolocation'] = full_df['city_id'].apply(lambda x: str(cities.loc[x].xcoord) + " " + str(cities.loc[x].ycoord))
	full_df['img_urls'] = None
	# full_df['name'] = None
	# micro_to_true = full_df[[cluster_label, 'label']].set_index(cluster_label).to_dict()['label']
	# cities = pd.read_csv(path_name+"cities.csv")
	# print(full_df['location'])
	# full_df['city'] = full_df['location'].apply(lambda x: cities[cities.id==x].name)
	full_df.rename(columns={'names_in_body':'name', 'location':'cleaned_loc', \
		'phone':'phone_num', 'body':'description'}, inplace=True)
	print(full_df.columns)
	cities_df = full_df[['cleaned_loc', 'geolocation']].set_index('cleaned_loc')
	marker_size = 5
	has_labels = False
	micro_to_meta = full_df[[cluster_label, 'Meta label']].set_index(cluster_label).to_dict()['Meta label']
	plot_df['Meta Cluster ID'] = plot_df['cluster_id'].apply(lambda x: micro_to_meta[x])
	
elif 'marinus_canada2' in path_name:
	full_df = pd.read_csv(path_name+"CanadianAds_names-800k_LSH_labels2.csv",index_col=False)
	plot_df = pd.read_csv(path_name+"plot_df.csv",index_col=False)

	cities = pd.read_csv(path_name+"cities.csv", index_col='id')
	full_df['location'] = full_df['city_id'].apply(lambda x: cities.loc[x]['name'])
	full_df['geolocation'] = full_df['city_id'].apply(lambda x: str(cities.loc[x].xcoord) + " " + str(cities.loc[x].ycoord))
	full_df['img_urls'] = None
	# micro_to_true = full_df[[cluster_label, 'label']].set_index(cluster_label).to_dict()['label']
	# cities = pd.read_csv(path_name+"cities.csv")
	# print(full_df['location'])
	# full_df['city'] = full_df['location'].apply(lambda x: cities[cities.id==x].name)
	full_df.rename(columns={'Name':'name', 'location':'cleaned_loc', \
		'phone':'phone_num', 'date_posted':'post_date', 'body':'description'}, inplace=True)

	cities_df = full_df[['cleaned_loc', 'geolocation']].set_index('cleaned_loc')
	full_df['Meta label'] = full_df['meta_label2'].copy()
	marker_size = 5
	has_labels = False
	cluster_label = 'meta_label2'
	plot_df['Meta Cluster ID'] = plot_df['cluster_id'].copy()
	micro_to_meta = full_df[['Meta label','meta_label2']].set_index('meta_label2').to_dict()['Meta label']
elif 'canada' in path_name:
	full_df = pd.read_csv(path_name+"CanadianAds-1_names_LSH_labels.csv",index_col=False)
	plot_df = pd.read_csv(path_name+"plot_df.csv",index_col=False)

	cities = pd.read_csv(path_name+"cities.csv", index_col='id')
	full_df['location'] = full_df['city_id'].apply(lambda x: cities.loc[x].name)
	full_df['geolocation'] = full_df['city_id'].apply(lambda x: str(cities.loc[x].xcoord) + " " + str(cities.loc[x].ycoord))
	full_df['img_urls'] = None
	# micro_to_true = full_df[[cluster_label, 'label']].set_index(cluster_label).to_dict()['label']
	# cities = pd.read_csv(path_name+"cities.csv")
	# print(full_df['location'])
	# full_df['city'] = full_df['location'].apply(lambda x: cities[cities.id==x].name)
	full_df.rename(columns={'Name':'name', 'location':'cleaned_loc', \
		'phone':'phone_num', 'date_posted':'post_date', 'body':'description'}, inplace=True)

	cities_df = full_df[['cleaned_loc', 'geolocation']].set_index('cleaned_loc')
	marker_size = 5
	has_labels = False
	cluster_label = 'LSH label'
	micro_to_meta = full_df[[cluster_label, 'Meta label']].set_index(cluster_label).to_dict()['Meta label']
	plot_df['Meta Cluster ID'] = plot_df['cluster_id'].apply(lambda x: micro_to_meta[x])
elif 'marinus' in path_name:
	full_df = pd.read_csv(path_name+"merged_data_3_class_no_dupl_LSH_labels.csv",index_col=False)
	plot_df = pd.read_csv(path_name+"plot_df.csv",index_col=False)
	micro_to_true = full_df[[cluster_label, 'label']].set_index(cluster_label).to_dict()['label']
	# cities = pd.read_csv(path_name+"cities.csv")
	# print(full_df['location'])
	# full_df['city'] = full_df['location'].apply(lambda x: cities[cities.id==x].name)
	full_df.rename(columns={'Name':'name', 'location':'cleaned_loc'}, inplace=True)

	cities_df = full_df[['cleaned_loc', 'geolocation']].set_index('cleaned_loc')
	marker_size = 5
	has_labels = True
	cluster_label = 'LSH label'
	micro_to_meta = full_df[[cluster_label, 'Meta label']].set_index(cluster_label).to_dict()['Meta label']
	plot_df['Meta Cluster ID'] = plot_df['cluster_id'].apply(lambda x: micro_to_meta[x])
else:
	full_df = pd.read_csv(path_name+"annoncexxx_filtered_infoshield_LSH_labels.csv",index_col=False) # file name containing the cluster characteristics. 'plot_df.csv' from analyze_clusters.ipynb
	plot_df = pd.read_csv(path_name+"plot_df.csv",index_col=False)
	cities_df = pd.read_csv(path_name+"geoloc_info.csv",index_col='city_ascii')
	marker_size = 4
	has_labels = False
	cluster_label = 'LSH label'
	micro_to_meta = full_df[[cluster_label, 'Meta label']].set_index(cluster_label).to_dict()['Meta label']
	plot_df['Meta Cluster ID'] = plot_df['cluster_id'].apply(lambda x: micro_to_meta[x])

plot_df.set_index('cluster_id', drop=False, inplace=True)

full_text = " ".join(st for st in full_df['description'].values)
full_text_words = full_text.split()
vocab = Counter(full_text_words)
most_freq_words = vocab.most_common(int(0.2*len(vocab.keys())))
add_to_stop = [w for w,f in most_freq_words]


# micro_to_meta = full_df[[cluster_label, 'Meta label']].set_index(cluster_label).to_dict()['Meta label']


# plot_df['Meta Cluster ID'] = plot_df['cluster_id'].apply(lambda x: micro_to_meta[x])
try:
	template_txt = pd.read_csv(path_name+"template_texts.csv", index_col=False)
except:
	template_txt = full_df[[cluster_label,'description']].set_index(cluster_label)
	template_txt.rename(columns={'description':'Template'},inplace=True)
	template_txt['Extra'] = template_txt.Template.copy()

# DO THE SAME AS ABOVE FOR WEAK LABELS AND TRUE LABELS

# full_df = full_df[full_df[cluster_label] != -1]
# top_clusters = full_df.groupby(cluster_label).size().sort_values()[-10:].index.values
# largest_clusters = full_df[full_df[cluster_label].isin(top_clusters)]
top_clusters = full_df.groupby(cluster_label).size().sort_values()[-10:].index.values
largest_clusters = full_df[full_df[cluster_label].isin(top_clusters)]
# largest_clusters = full_df[full_df[cluster_label].isin(range(50))]

current_selected_points = 0
current_selected_points2 = 0
current_selected_points3 = 0

list_of_buttons = []

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


data_contents = html.Div([
	# html.Br(),
	html.Div([
		html.H3(children='Ad count over time'),
		dcc.Graph(
			id='ads_over_time',
		)
	], style={'width': '32%', 'float':'left','display': 'inline-block', 'padding': '0 20'}),
	html.Div([
		dbc.DropdownMenu(
		    label="Meta data-type",
		    size='lg',
		    children=[
		        # dbc.DropdownMenuItem("Cluster Size", id='Cluster Size'),
		        dbc.DropdownMenuItem("Phone Number", id='Phone Number'),
		        dbc.DropdownMenuItem("Img URL", id='Img URL'),
		        dbc.DropdownMenuItem("Name", id='Name'),
		    ], style={'float':'right','margin-top':'5px'}
		), 
		html.H3(children='Meta-data over time'), html.Br(),
		dcc.Graph(id='metadata_time'),
	], style={'display': 'inline-block', 'width': '35%','float':'center'}),
	html.Div([
		html.H3(children='Geographical Spread of ads'),
		dcc.Graph(id='geo_plot')], style={'width': '30%', 'float':'right'}),
	html.Div([
		dbc.Row(children=[
			dbc.Col(html.H3(children='Ad templates')),
			dbc.Col(html.Div([html.Div(id='toggle-text',children=['Show full text']),
			    daq.ToggleSwitch(
			        id='my-toggle-text',
			        value=False
			    ), dbc.Tooltip("Show full text",placement='bottom',autohide=False, \
			    target='my-toggle-text')], style={'float':'left', 'margin-left':'-400px', 'margin-bottom':'10px'}))
			    # style={'float':'left', 'margin-left':'-900px', 'margin-bottom':'10px'}))
			]),
		dcc.Textarea(id='text_box', readOnly=True,
			style={'height':350,'width':'100%'})],style={'width':'55%', 'float':'left'}),

		html.Div([
		html.H3(children='Word Cloud'), 
		# html.Img(id='word-cloud',src='data:image/png;base64,{}'.format(wc_img)),
		html.Div([dcc.Graph(id='word-cloud')],style={'width':'100%'}),
		],style={'float':'right', 'width':'40%','margin-right':'40px'})
])

mini_dists = [0, 0.01, 0.05, 0.1, 0.5, 1]
char_contents = html.Div([
	html.Div(children=[
			html.H3(children='Feature embeddings'),

			dcc.Dropdown( # dropdown menu for choosing clustering method/ UMAP/ TSNE
				id='analysis-type',
				options=[{'label': ind, 'value': i} for i, ind in enumerate(available_indicators)],
				value=0
			),
			html.Br(),
			html.Div(
			    [
			    dbc.Row(children=[
			        dbc.Col(children=[dbc.Button("Meta-Cluster Labels", id='b1', color="primary", disabled=False, \
			        	className="me-1"),
			        dbc.Button("Weak Labels", id='b2', color="secondary", disabled=True, \
			        	className="me-1"),
			        dbc.Button("True Labels", id='b3', color="dark", disabled=True, \
			        	className="me-1")], style={'width':'70%'}),
			        dbc.Col(dcc.Dropdown( # dropdown menu for choosing clustering method/ UMAP/ TSNE
						id='attribute-type',
						options=[{'label': ind, 'value': i} for i, ind in enumerate(dimension_cols)],
						value=None, 
					),style={'width':'30%'}),
			        dbc.Tooltip("Clusters with shared meta data",placement='bottom', target='b1', style={'fontSize':20}),
			        dbc.Tooltip("Weak labels inferred from the data",placement='bottom', target='b2', style={'fontSize':20}),
			        dbc.Tooltip("True M.O labels of clusters",placement='bottom', target='b3', style={'fontSize':20}),
			    	dbc.Tooltip("Choose an attribute to color the points by",placement='right', target='attribute-type', style={'fontSize':20})
			    ])
			    ], style={'display':'inline-block','float':'right', 'width':'85%'}),
			
			html.Br(), 
			html.Br(),

			dcc.Graph( # scatter plot showing the micro-clusters using method chosen above
			id='micro-cluster-scatter',
			responsive=True,style={'height':'40vh'}
			),

			html.Div(id='slider-comp', children=[ # Create slides to hide/show for UMAP
        	dcc.Slider(
		    id='slider',
            min=0,  max=5,
            value=0, step=1,
            # marks={i: '{}'.format(10 ** i) for i in mini_dists},
            marks={str(i): {'style':{'fontSize':17},'label':str(mini_dists[i])} for i in range(len(mini_dists))},
            updatemode='drag'
            # tooltip={"placement": "bottom", "always_visible": True}
            ),
        	html.Div(id='slider-name',children=['Minimum distance'],style={'textAlign':'center', 'fontSize':20})
            ], style= {'display': 'none'} # <-- This is the line that will be changed by the dropdown callback
		    ),

			dcc.Graph( # only for UMAP and TSNE
			id='enlarged-graph',
			responsive=True, style={'display':'none'}
			)
			], style={'width': '44%', 'display': 'inline-block', 'margin-top': '25px','margin-left':'-10px'}),

	html.Div(children=[ # pair-wise scatter plot 
			html.H3(children='Cluster Characterization'),
			# html.Div(children="InfoShield Clusters"),
			dcc.Dropdown( # dropdown menu for choosing clustering method/ UMAP/ TSNE
				id='feature-type', 
				options=[{'label': i, 'value': i} for i in dimension_cols],
				value=None,
				multi=True
			),
			dbc.Row(children=[
				dbc.Col(dcc.RadioItems(
				id='scale-type',
				options=[{'label': i, 'value': i} for i in ['Log', 'Linear']],
				value='Log',
				labelStyle={'display': 'inline-block', 'marginTop': '6px'},
				inputStyle={'margin-right':'5px', 'margin-left':'5px', 'margin-top':'10px'}
				)),

				dbc.Col([
				dbc.Button(children="Show Individual", id='hist', size='lg', color="primary", active=True, disabled=False, \
					className="me-1", style={'display':'inline-block',\
				'margin-top':'15px', 'margin-left':'270px'}),\
				# dbc.Tooltip("Show/hide relevant features as decided by an ML classifier",placement='bottom',autohide=False, \
			    # target='feats'),
			    dbc.Tooltip("Show/hide individual feature distribution",placement='bottom',autohide=False, \
			    target='hist', style={'fontSize':20}),
				html.Div([html.Div(id='toggle-switch',children=['Show all points']),
			    daq.ToggleSwitch(
			        id='my-toggle-switch',
			        value=False
			    ), dbc.Tooltip("Show/hide unselected points",placement='bottom',autohide=False, \
			    target='my-toggle-switch')], \
			    style={'float':'right','display':'inline-block'})])
				]
			),
			dcc.Graph(
				id='hist-plot', responsive=True, style={'display':'inline-block'}),
			dcc.Graph(
				id='main-plot', responsive=False
			, style={'display':'inline-block'}),
			],style={'width': '55%', 'float': 'right', 'margin-top': '25px'}),

	], style={"height":"100vh"})

current_files = []
current_snapshots = []
if os.path.isdir(path_name+"snapshots/"):
	for file in os.listdir(path_name+"snapshots/"):
		filename = os.fsdecode(file)
		contents = pkl.load(open(path_name+"snapshots/"+filename, 'rb'))
		current_files.append(contents)
		

save_contents =  dmc.Accordion(id='saved_items', multiple=True, 
	children=[
            dmc.AccordionItem(
                [
                    html.P(str(contents['comment'] + " - " + str(contents['label']))),
                    dbc.Button(id={"type":"dynamic-button","index":'load'+str(contents['title'])},children="Load snapshot"),
                	# dbc.Spinner(html.Div(id="loading-output")),
                ],
                label=str(contents['title']),
            ) for contents in current_files
        ],
        # dbc.Button("Load", id="loading-button", n_clicks=0),        
)

# layout of the app
app.layout = dbc.Container(
	[
		# dcc.Store(id="store"),
		html.Div([
		html.H2("VisPaD: Tool for Visualization and Pattern Discovery"),

		dbc.Row(children=
				[
				dbc.Col(html.Div(id="meta_clusters_n")),
				dbc.Col(html.Div(id="micro_clusters_n")),
                dbc.Col(html.Div(id="ads_n")),
                dbc.Col(html.Div(id="phone_n")),
                dbc.Col(html.Div(id="img_url_n")),
                dbc.Col(html.Div(id="location_n")),
                dbc.Col(html.Div(id="name_n"))
				], align='center', style={'margin-top':'10px', 'fontSize':30}
			),
		dbc.Row(children=
				[
				dbc.Col(html.Div(id="meta_clusters")),
				dbc.Col(html.Div(id="micro_clusters")),
                dbc.Col(html.Div(id="ads")),
                dbc.Col(html.Div(id="phone")),
                dbc.Col(html.Div(id="img_url")),
                dbc.Col(html.Div(id="location")),
                dbc.Col(html.Div(id="name")),
				], align='center'
				, style={'margin-top':'-2px', 'fontSize':25}
			)], style={'textAlign': 'center','backgroundColor': 'blue', \
			'color':'white','height':'129px'}
		),		
		# html.Div([ # top row header
		# html.Br(),
		# ]),
		html.Div(
		    [
		        # dbc.Button("Open modal", id="open", n_clicks=0),
		        dbc.Button(children="Save current selection", id='save_option', color="primary", n_clicks=0, \
		        	active=True, disabled=False, className="me-1", style={'display':'inline-block','float':'right', 'margin-top':'10px','margin-right':'20px'}),
		        dbc.Modal(
		            [
		                dbc.ModalHeader(dbc.ModalTitle("Save your selection")),
		                dbc.ModalBody(
			                [
	                        dbc.Label("Selection title:"),
	                        dbc.Input(id="selection_title", type="text", placeholder='S1', debounce=True),
	                        dbc.Label("Notes/comments:"),
	                        dbc.Input(id="comment", type="text", debounce=True),
	                        dbc.Form(html.Div(
									    [
									        dbc.Label("Choose one"),
									        dbc.RadioItems(
									            options=[
									                {"label": "Spam", "value": 'Spam'},
									                {"label": "Suspicious", "value": 'Suspicious'},
									                {"label": "Not interesting", "value": 'Not interesting'},
									            ],
									            value=1,
									            id="label-input",
									        ),
									    ]
									),)
	                    	]
	                    	),
		                dbc.ModalFooter([
		                    dbc.Button("Save", id="close", className="ms-auto", n_clicks=0),
		                    dbc.Button("Cancel", id="cancel", n_clicks=0),]
		                ),
		            ],
		            id="modal",
		            is_open=False,
		        ),
		    ]
		),
		dbc.Tabs(
			[
				dbc.Tab(data_contents, label="Inspect Clusters", label_style={'fontSize':20,}, tab_id="data", style={'fontSize':20}),
				dbc.Tab(char_contents, label="Analysis", label_style={'fontSize':20}, tab_id="scatter", style={'fontSize':20}),
				dbc.Tab(save_contents, label="Saved snapshots", label_style={'fontSize':20}, tab_id="saved", style={'fontSize':20})
			],
			id="tabs",
			active_tab="data"
		),
		visdcc.Run_js(id = 'javascriptLog', run = ""),
		# html.Div(id="tab-content", className="p-4"),
	], fluid=True, style={'width':'100%'})


# @app.callback(
#     Output("loading-output", "children"), [Input("loading-button", "n_clicks")]
# )
# def load_output(n):
#     if n:
#         time.sleep(1)
#         return f"Output loaded {n} times"
#     return "Output not reloaded yet"

'''
=============================SUMMARY OF CHOSEN CLUSTERS======================================
'''
@app.callback(
Output('meta_clusters_n', 'children'),
Output('micro_clusters_n', 'children'),
Output('ads_n', 'children'),
Output('phone_n', 'children'),
Output('img_url_n', 'children'),
Output('location_n', 'children'),
Output('name_n', 'children'),
Input('tabs','active_tab')) # input: currently active tab
def update_summary_heads(active_tab):
	if has_labels:
		return "Meta-clusters", "Micro-clusters", "Ads", "Phone Numbers", "Email IDs", "Locations", "Names"
	else:
		return "Meta-clusters", "Micro-clusters", "Ads", "Phone Numbers", "Img URLs", "Locations", "Names"


@app.callback(
Output('meta_clusters', 'children'),
Output('micro_clusters', 'children'),
Output('ads', 'children'),
Output('phone', 'children'),
Output('img_url', 'children'),
Output('location', 'children'),
Output('name', 'children'),
Input({'type': 'dynamic-button', 'index': ALL}, "n_clicks"),
Input('enlarged-graph', 'selectedData'), # input: selected points from the enlarged scatter-plot (TSNE/UMAP)
Input('main-plot', 'selectedData'), # input: selected points from the pair-plots
Input('micro-cluster-scatter', 'selectedData')) # input: selected points from the feature embedding plot
def update_summary(load_id, selectedData, selected_from_pair_plots, selected_from_ica):
	if len(np.where(np.array(load_id)==1)[0]) != 0:
		selected = json.loads(dash.callback_context.triggered[0]['prop_id'][:-9])['index'][4:]
		selected_points = pkl.load(open(path_name+"snapshots/"+selected+".pkl",'rb'))['clusters']
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
	elif selectedData:
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selectedData['points']:
			selected_points.append(item['customdata'][-1])

		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
	
	elif selected_from_pair_plots:  
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selected_from_pair_plots['points']:
			selected_points.append(item['customdata'][-1])

		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
	
	elif selected_from_ica:
		selected_points = []
		for item in selected_from_ica['points']:
			selected_points.append(item['customdata'][-1])
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
	else:
		selected_df = largest_clusters

	num_meta = selected_df['Meta label'].nunique()
	num_clusters = selected_df[cluster_label].nunique()
	num_ads = len(selected_df)
	num_names = selected_df.name.count()
	num_unique_names = selected_df.name.nunique()
	num_imgs = selected_df.img_urls.count()
	num_unique_imgs = selected_df.img_urls.nunique()
	num_phones = selected_df.phone_num.count()
	# print("Unique count:")
	# print(selected_df.phone_num.unique())
	num_unique_phones = selected_df.phone_num.nunique()
	num_locs = selected_df.cleaned_loc.count()
	num_unique_locs = selected_df.cleaned_loc.nunique()

	if has_labels:
		num_emails = selected_df.email.count()
		num_unique_emails = selected_df.email.nunique()
		return str(num_meta), str(num_clusters), str(num_ads), str(num_phones)+" ("+str(num_unique_phones)+")", \
		str(num_emails)+" ("+str(num_unique_emails)+")", str(num_locs)+" ("+str(num_unique_locs)+")", \
		str(num_names)+" ("+str(num_unique_names)+")"

	else:
		return str(num_meta), str(num_clusters), str(num_ads), str(num_phones)+" ("+str(num_unique_phones)+")", \
		str(num_imgs)+" ("+str(num_unique_imgs)+")", str(num_locs)+" ("+str(num_unique_locs)+")", \
		str(num_names)+" ("+str(num_unique_names)+")"



'''
=============================SAVE CURRENT SELECTION======================================
'''
@app.callback(
Output("modal", "is_open"),
Output('save_option', 'n_clicks'),
Output('close', 'n_clicks'),
Output('cancel', 'n_clicks'),
Output("saved_items", "children"),
Input("saved_items", "children"),
Input('selection_title', 'value'),
Input('comment', 'value'),
Input('label-input', 'value'),
Input('save_option', 'n_clicks'),
Input('close', 'n_clicks'),
Input('cancel', 'n_clicks'),
State("modal", "is_open"), 
Input('enlarged-graph', 'selectedData'), # input: selected points from the enlarged scatter-plot (TSNE/UMAP)
Input('main-plot', 'selectedData'), # input: selected points from the pair-plots
Input('micro-cluster-scatter', 'selectedData')) # input: selected points from the feature embedding plot
def show_state(state, save_title, save_comment, save_label, \
	save_clicks, close_clicks, cancel_clicks, is_open, \
	selectedData, selected_from_pair_plots, selected_from_ica):
	
	if cancel_clicks:
		return not is_open, 0, 0, 0, state
	if save_clicks:
		return not is_open, 0, 0, 0, state

	if close_clicks:
		if selectedData:
			selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
			for item in selectedData['points']:
				selected_points.append(item['customdata'][-1])

			selected_df = full_df[full_df[cluster_label].isin(selected_points)]
		
		elif selected_from_pair_plots:  
			selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
			for item in selected_from_pair_plots['points']:
				selected_points.append(item['customdata'][-1])

			selected_df = full_df[full_df[cluster_label].isin(selected_points)]
		
		elif selected_from_ica:
			selected_points = []
			for item in selected_from_ica['points']:
				selected_points.append(item['customdata'][-1])
			selected_df = full_df[full_df[cluster_label].isin(selected_points)]
		else:
			selected_df = largest_clusters

		new_state = {"props":{
		"children": [{"props": {"children": str(save_comment) + " - " + str(save_label)}, "type": 'P', \
		"namespace": 'dash_html_components'},
					 {"props": {"children" : "Load snapshot", \
					 "id":{"type":"dynamic-button","index":'load'+str(save_title)}}, \
					 "type":"Button", "namespace": 'dash_bootstrap_components'}],
		"label" : str(save_title)
		}, "type": 'AccordionItem', "namespace":'dash_mantine_components'}
		new_state = re.sub( "(?<={)\'|\'(?=})|(?<=\[)\'|\'(?=\])|\'(?=:)|(?<=: )\'|\'(?=,)|(?<=, )\'", "\"", str(new_state))

		state.append(json.loads(new_state))

		res = {'title': str(save_title), 'comment': str(save_comment), 'clusters': selected_df[cluster_label].unique(), 'label':str(save_label)}
		if not os.path.isdir(path_name+"snapshots/"):
			os.mkdir(path_name+"snapshots/")

		pkl.dump(res, open(path_name+"snapshots/"+save_title+".pkl",'wb'))
		list_of_buttons.append('load'+str(save_title))
		current_snapshots.append(res)
		return not is_open, 0, 0, 0, state
	else:
		return is_open, 0, 0, 0, state



'''
=============================ADS POSTED OVER TIME======================================
'''
@app.callback(
Output('ads_over_time', 'figure'), # output: ads poster over time
Input({'type': 'dynamic-button', 'index': ALL}, "n_clicks"),
# Input('tabs','active_tab'), # input: currently active tab
Input('enlarged-graph', 'selectedData'), # input: selected points from the enlarged scatter-plot (TSNE/UMAP)
Input('main-plot', 'selectedData'), # input: selected points from the pair-plots
Input('micro-cluster-scatter', 'selectedData'))
def update_ads_over_time(load_id, selectedData, selected_from_pair_plots, selected_from_ica):
	global current_sum
	print(current_sum)
	if sum(np.where(np.array(load_id)==1)[0]) != current_sum:
	# if len(np.where(np.array(load_id)==1)[0]) != 0:
		current_sum = sum(np.where(np.array(load_id)==1)[0])
		print(dash.callback_context.triggered[0]['prop_id'])
		selected = json.loads(dash.callback_context.triggered[0]['prop_id'][:-9])['index'][4:]
		selected_points = pkl.load(open(path_name+"snapshots/"+selected+".pkl",'rb'))['clusters']
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
	elif selectedData:
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selectedData['points']:
			selected_points.append(item['customdata'][-1])

		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
		# selected_df = full_df.loc[selected_points]
	
	elif selected_from_pair_plots:  
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selected_from_pair_plots['points']:
			selected_points.append(item['customdata'][-1])

		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
		# selected_df = full_df.loc[selected_points]
	
	elif selected_from_ica:
		selected_points = []
		for item in selected_from_ica['points']:
			selected_points.append(item['customdata'][-1])
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
		# selected_df = full_df.loc[selected_points]

	else:
		selected_df = largest_clusters

	top_clusters = selected_df.groupby(cluster_label).size().sort_values(ascending=False)[:10].index.values

	selected_df = selected_df[selected_df[cluster_label].isin(top_clusters)]
	selected_df['post_date'] = selected_df['post_date'].apply(lambda x: x.split()[0])
	selected_df['post_date'] = pd.to_datetime(selected_df['post_date'], infer_datetime_format=True)

	dates = []
	counts = []
	meta_data_types = []
	micro_cluster_ids = []
	meta_cluster_ids = []

	for meta_id, meta_df in selected_df.groupby('Meta label'):
		for micro_id, micro_df in meta_df.groupby(cluster_label):
			for grp in micro_df.groupby('post_date'):

				if pd.isnull(grp[0]) or micro_id == -1:
					continue
				# for cluster sizes
				# print(grp[0])
				dates.append(grp[0])
				counts.append(len(grp))
				micro_cluster_ids.append(str(micro_id)+"("+str(len(micro_df))+")")
				meta_cluster_ids.append(str(meta_id))
				# meta_data_types.append("Cluster Size")

	plot_data = pd.DataFrame({'Post Date':dates, 'Count':counts, \
	 'Micro Cluster ID': micro_cluster_ids, 'Meta Cluster ID': meta_cluster_ids})

	plot_data = plot_data.sort_values(by='Count', ascending=False)
	plot_data = plot_data.sort_values(by='Post Date')

	plot_data['Micro Cluster ID'] = 'C'+plot_data['Micro Cluster ID'].astype(str)
	plot_data['Meta Cluster ID'] = 'M'+plot_data['Meta Cluster ID'].astype(str)


	try:
		fig = px.scatter(plot_data, x='Post Date', y='Micro Cluster ID', \
			color='Meta Cluster ID', size='Count', marginal_y='histogram', color_discrete_sequence=px.colors.qualitative.Vivid)
	except Exception:
		fig = px.scatter(plot_data, x='Post Date', y='Micro Cluster ID', \
			color='Meta Cluster ID', size='Count', marginal_y='histogram', color_discrete_sequence=px.colors.qualitative.Vivid)

	fig.update_layout(
			font_size=20,
			xaxis=dict(
				tickformat="%d-%b\n%Y"
			),
			yaxis = dict(
				tickmode='array',
				ticktext=plot_data['Micro Cluster ID'],
				title='Cluster ID'
			), showlegend=False)

	current_selected_points = selected_df['Meta label'].unique()
	# print('Size:')
	# print(current_selected_points)
	return fig


'''
=============================META DATA OVER TIME======================================
'''
@app.callback(
Output('metadata_time', 'figure'), # output: ads poster over time
# Input('tabs','active_tab'), # input: currently active tab
Input({'type': 'dynamic-button', 'index': ALL}, "n_clicks"),
Input('enlarged-graph', 'selectedData'), # input: selected points from the enlarged scatter-plot (TSNE/UMAP)
Input('main-plot', 'selectedData'), # input: selected points from the pair-plots
Input('micro-cluster-scatter', 'selectedData'),
# Input('Cluster Size', 'n_clicks'),
Input('Phone Number', 'n_clicks'),
Input('Img URL', 'n_clicks'),
Input('Name', 'n_clicks')) 
def update_meta_data(load_id, selectedData, selected_from_pair_plots, selected_from_ica, \
	phone_num, img_url, name):
	if len(np.where(np.array(load_id)==1)[0]) != 0:
		selected = json.loads(dash.callback_context.triggered[0]['prop_id'][:-9])['index'][4:]
		selected_points = pkl.load(open(path_name+"snapshots/"+selected+".pkl",'rb'))['clusters']
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
	elif selectedData:
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selectedData['points']:
			selected_points.append(item['customdata'][-1])

		# selected_df = full_df.loc[selected_points]
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
	
	elif selected_from_pair_plots:  
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selected_from_pair_plots['points']:
			selected_points.append(item['customdata'][-1])

		# selected_df = full_df.loc[selected_points]
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]			
	
	elif selected_from_ica:
		selected_points = []
		for item in selected_from_ica['points']:
			selected_points.append(item['customdata'][-1])
		# selected_df = full_df.loc[selected_points]
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]

	else:
		selected_df = largest_clusters


	top_clusters = selected_df.groupby(cluster_label).size().sort_values(ascending=False)[:10].index.values

	selected_df = selected_df[selected_df[cluster_label].isin(top_clusters)]
	# print(selected_df[['Meta label','phone_num']])

	selected_df['post_date'] = pd.to_datetime(selected_df['post_date'], infer_datetime_format=True, errors='coerce')
	selected_df = selected_df.sort_values(by='post_date')


	dates = []
	counts = []
	meta_data_types = []
	micro_cluster_ids = []
	meta_cluster_ids = []

	for meta_id, meta_df in selected_df.groupby('Meta label'):
		for micro_id, micro_df in meta_df.groupby(cluster_label):
			for grp in micro_df.groupby('post_date'):

				if pd.isnull(grp[0]) or micro_id == -1:
					continue
				# for phone numbers
				try:
					dates.append(grp[0].strftime('%Y-%m-%d'))
				except:
					continue
				counts.append(grp[1].phone_num.nunique())
				micro_cluster_ids.append(str(micro_id))
				meta_cluster_ids.append(str(meta_id))
				meta_data_types.append("Phone Number")

				# for img urls
				dates.append(grp[0].strftime('%Y-%m-%d'))
				counts.append(grp[1].img_urls.nunique())
				micro_cluster_ids.append(str(micro_id))
				meta_cluster_ids.append(str(meta_id))
				meta_data_types.append("Img URL")

				# for names
				dates.append(grp[0].strftime('%Y-%m-%d'))
				counts.append(grp[1].name.nunique())
				micro_cluster_ids.append(str(micro_id))
				meta_cluster_ids.append(str(meta_id))
				meta_data_types.append("Name")

				# for cluster sizes
				dates.append(grp[0].strftime('%Y-%m-%d'))
				counts.append(len(micro_df))
				micro_cluster_ids.append(str(micro_id))
				meta_cluster_ids.append(str(meta_id))
				meta_data_types.append("Cluster Size")

	meta_data = pd.DataFrame({'Post Date':dates, 'Count':counts, 'Type':meta_data_types, \
	 'Micro Cluster ID': micro_cluster_ids, 'Meta Cluster ID': meta_cluster_ids})

	# based on button selection
	ctx = dash.callback_context

	if not ctx.triggered:
		chosen_type = "Phone Number"

	elif len(ctx.triggered) != 1:
		chosen_type = 'Phone Number'
	else:
		chosen_type = ctx.triggered[0]["prop_id"].split(".")[0]
		
	if chosen_type not in ["Phone Number", 'Img URL', 'Name']:
		chosen_type = 'Phone Number'
	plot_data = meta_data[meta_data.Type==chosen_type]

	plot_data.drop_duplicates(inplace=True)
	plot_data = plot_data.sort_values(by='Count', ascending=False)
	plot_data = plot_data.sort_values(by='Post Date')
	# print(plot_data)
	plot_data['Micro Cluster ID'] = 'C'+plot_data['Micro Cluster ID'].astype(str)
	plot_data['Meta Cluster ID'] = 'M'+plot_data['Meta Cluster ID'].astype(str)


	try:
		fig = px.scatter(plot_data, x='Post Date', y='Micro Cluster ID', \
			color='Meta Cluster ID', size='Count', color_discrete_sequence=px.colors.qualitative.Vivid)
	except Exception:
		fig = px.scatter(plot_data, x='Post Date', y='Micro Cluster ID', \
			color='Meta Cluster ID', size='Count', color_discrete_sequence=px.colors.qualitative.Vivid)
	
	fig.update_layout(
		font_size=20,
		xaxis=dict(
			tickformat="%d-%b\n%Y"
		),
		yaxis = dict(
			tickmode='array',
			ticktext=plot_data['Micro Cluster ID'],
			title='Cluster ID'
	), title_text=chosen_type)

	current_selected_points2 = selected_df['Meta label'].unique()
	return fig



'''
=============================GEO DATA======================================
'''
@app.callback(
Output('geo_plot', 'figure'), # output: ads poster over time
Input({'type': 'dynamic-button', 'index': ALL}, "n_clicks"),
# Input('tabs','active_tab'), # input: currently active tab
Input('enlarged-graph', 'selectedData'), # input: selected points from the enlarged scatter-plot (TSNE/UMAP)
Input('main-plot', 'selectedData'), # input: selected points from the pair-plots
Input('micro-cluster-scatter', 'selectedData'))
def update_meta_data(load_id, selectedData, selected_from_pair_plots, selected_from_ica):
	if len(np.where(np.array(load_id)==1)[0]) != 0:
		selected = json.loads(dash.callback_context.triggered[0]['prop_id'][:-9])['index'][4:]
		selected_points = pkl.load(open(path_name+"snapshots/"+selected+".pkl",'rb'))['clusters']
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
	elif selectedData:
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selectedData['points']:
			selected_points.append(item['customdata'][-1])

		# selected_df = full_df.loc[selected_points]
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
	
	elif selected_from_pair_plots:  
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selected_from_pair_plots['points']:
			selected_points.append(item['customdata'][-1])

		# selected_df = full_df.loc[selected_points]
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
	
	elif selected_from_ica:
		selected_points = []
		for item in selected_from_ica['points']:
			selected_points.append(item['customdata'][-1])

		# selected_df = full_df.loc[selected_points]
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]

	else:
		selected_df = largest_clusters

	top_clusters = selected_df.groupby(cluster_label).size().sort_values()[-10:].index.values
	selected_df = selected_df[selected_df[cluster_label].isin(top_clusters)]
	
	if 'locanto' not in path_name and 'marinus' not in path_name:
		selected_df = pd.merge(selected_df, cities_df, left_on='cleaned_loc', right_on='city_ascii', left_index=True, \
		how='left', sort=False)
	selected_df['post_date'] = pd.to_datetime(selected_df.post_date, infer_datetime_format=True)
	# selected_df['post_date'] = pd.to_datetime(selected_df.post_date, format='%Y-%m-%d', errors='coerce')
	# selected_df['year-month'] = selected_df['post_date'].apply(lambda x: str(x.year) + ' ' + str(x.month) \
		# if not pd.isnull(x) else x)		

	ads_per_location_time_df = get_location_time_info(selected_df, cities_df)
	ads_per_location_time_df['Meta Cluster ID'] = 'M'+ads_per_location_time_df['Meta Cluster ID'].astype('str')
	ads_per_location_time_df.sort_values(by='date',inplace=True)

	try:
		fig = px.scatter_geo(ads_per_location_time_df, lat='lat',lon='lon', 
                 hover_name="plot_text", size='plot_counts', hover_data={'date':False,'plot_counts':False, \
                 'count':True},
                 animation_frame="date", scope='north america', \
                 color_discrete_sequence=px.colors.qualitative.Vivid, color='Meta Cluster ID')
	except Exception:
		fig = px.scatter_geo(ads_per_location_time_df, lat='lat',lon='lon', 
                 hover_name="plot_text", size='plot_counts', hover_data={'date':False,'plot_counts':False, \
                 'count':True},
                 animation_frame="date", scope='north america', color='Meta Cluster ID', \
                 color_discrete_sequence=px.colors.qualitative.Vivid)

	fig.update_layout(font_size=20,
		xaxis=dict(
			tickformat="%d-%b\n%Y"
		),
		)
	current_selected_points3 = selected_df['Meta label'].unique()
	# print('Geo:')
	# print(current_selected_points3, len(current_selected_points3))
	return fig

# print(current_selected_points == current_selected_points2 and current_selected_points == current_selected_points3)

'''
=============================MICRO-CLUSTER AD DESCRIPTIONS======================================
'''
@app.callback(
Output('text_box', 'value'), # output: text box for ad descriptions
Output('javascriptLog','run'), # output: scroll bar for text box
Input({'type': 'dynamic-button', 'index': ALL}, "n_clicks"),
# Input('tabs','active_tab'), # input: currently active tab
Input('enlarged-graph','selectedData'), # input: selected points from the enlarged scatter-plot (TSNE/UMAP)
Input('main-plot', 'selectedData'),
Input('my-toggle-text', 'value')) # input: selected points from the pair-plots
def update_ad_text(load_id, selectedData, selected_from_pair_plots, toggle_value):
	# print the ad text in the data tab
	ad_text = "var textarea = document.getElementById('text_box'); textarea.scrollTop = textarea.scrollHeight;"
	if len(np.where(np.array(load_id)==1)[0]) != 0:
		# print(ast.literal_eval(dash.callback_context.triggered[0]['prop_id'][:-9]))
		selected = json.loads(dash.callback_context.triggered[0]['prop_id'][:-9])['index'][4:]
		selected_points = pkl.load(open(path_name+"snapshots/"+selected+".pkl",'rb'))['clusters']
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
	elif selectedData:
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selectedData['points']:
			selected_points.append(item['customdata'][-1])

	elif selected_from_pair_plots:
		selected_points = []
		for item in selected_from_pair_plots['points']:
			selected_points.append(item['customdata'][-1])

	else:
		selected_points = largest_clusters[cluster_label].unique()

	sel_data = full_df[full_df[cluster_label].isin(selected_points)]
	ordered_clusters = sel_data.groupby(cluster_label).size().sort_values(ascending=False)

	txts = ""
	for row in ordered_clusters.index:
		if row == -1:
			continue
		txts += ("C"+str(row)+":\n")

		if toggle_value:
			if not pd.isna(template_txt.loc[row]['Extra'].values[0]):
				grp = sel_data[sel_data[cluster_label]==row]
				txts += ("\n\n".join(d for d in grp.description.values))
			else:
				txts += template_txt.loc[row]['Extra']

		else:
			# print(template_txt.loc[row]['Template'].values[0])
			if type(template_txt.loc[row]['Template']) == str:
				tt = template_txt.loc[row]['Template']
			else:
				tt = template_txt.loc[row]['Template'].values[0]
			if not pd.isna(tt):
				txts += tt 

		txts += "\n----------------------\n"

	# txts = ""
	# for row in ordered_clusters.index:
	# 	if row == -1:
	# 		continue
	# 	txts += ("C"+str(row)+":\n")

	# 	if toggle_value:
	# 		if pd.isna(template_txt.loc[row]['Extra']):
	# 			grp = sel_data[sel_data[cluster_label]==row]
	# 			txts += grp.description.values[0]
	# 			# txts += ("\n----\n".join(d for d in grp.description.values))
	# 		else:
	# 			txts += template_txt.loc[row]['Extra']
	# 	# for grp in sel_data.groupby(cluster_label):
		
	# 		# txts += "\n\nCluster : C" + str(row) + "\n"
	# 		# grp = sel_data[sel_data[cluster_label]==row]
	# 		# txts += ("\n".join(d for d in grp.description.values))

	# 	else:
	# 		txts += template_txt.loc[row]['Template']

		# txts += "\n----------------------\n"

		# print('Txt:')
		# print(sel_data[cluster_label].unique(), sel_data[cluster_label].nunique(), len(selected_points))
	
	return txts, ad_text

'''
=============================WORD-CLOUD IMAGES======================================
'''
@app.callback(
Output('word-cloud', 'figure'),
# Output({'type': 'dynamic-button', 'index': ALL}, "n_clicks"),
Input({'type': 'dynamic-button', 'index': ALL}, "n_clicks"),
Input('enlarged-graph','selectedData'), # input: selected points from the enlarged scatter-plot (TSNE/UMAP)
Input('main-plot', 'selectedData'),
Input('micro-cluster-scatter', 'selectedData'))
def update_word_cloudes(load_id, selectedData, selected_from_pair_plots, selected_from_ica):
	if len(np.where(np.array(load_id)==1)[0]) != 0:
		selected = json.loads(dash.callback_context.triggered[0]['prop_id'][:-9])['index'][4:]
		selected_points = pkl.load(open(path_name+"snapshots/"+selected+".pkl",'rb'))['clusters']
		selected_df = full_df[full_df[cluster_label].isin(selected_points)]
	elif selectedData:
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selectedData['points']:
			selected_points.append(item['customdata'][-1])

	elif selected_from_pair_plots:
		selected_points = []
		for item in selected_from_pair_plots['points']:
			selected_points.append(item['customdata'][-1])
	elif selected_from_ica:
		selected_points = []
		for item in selected_from_ica['points']:
			selected_points.append(item['customdata'][-1])
	else:
		selected_points = largest_clusters[cluster_label].unique()

	# wc_df = full_df[full_df[cluster_label].isin(selected_points)].text.values
	wc_df = template_txt.loc[selected_points].Template.values
	text = "\n".join(t for t in wc_df if not pd.isna(t))
	stopwords = STOPWORDS.copy()
	stopwords != set(add_to_stop)
	stopwords |= set(["the", 'RT', 'trump', 'joebiden', 'Template', "COVID",'coronavirus', 'Ethnicity', \
		'Age','Ethnicity','Extension', 'CELL', 'Name', 'Phone', 'Call', 'Post'])
	wordcloud = WordCloud(stopwords=stopwords,
					  width=1800, 
                      height=800,
                      min_font_size=7,
                      collocations=False,
                      # prefer_horizontal=0.5,
                      background_color='white',
                      # background_color="rgba(255, 255, 255, 0)", 
                      max_words=60,
                      repeat=False,
                      mode="RGBA").generate(text)
	fig = px.imshow(wordcloud)

	fig.update_xaxes(visible=False)
	fig.update_yaxes(visible=False)
	return fig

'''
=============================MICRO-CLUSTER FEATURE EMBEDDING======================================
'''
# function for showing the clicked point from pair-plots on the scatter plot
@app.callback( 
Output('micro-cluster-scatter', 'figure'),
Input('analysis-type', 'value'),
Input('main-plot','clickData'),
Input('main-plot','selectedData'),
Input('slider','value'),
Input('b1', 'n_clicks'), # input: if meta cluster label is clicked
Input('b2', 'n_clicks'), # input: if weak label is clicked
Input('b3', 'n_clicks'), # input: if true label is clicked
Input('attribute-type', 'value'))
def update_graph(selected_clustering, clickData, selectedData, mini_dist_ind, b1_click, b2_click, b3_click, attribute_value):
	perp_vals = [5, 10, 20, 30, 40, 50]

	show_color_bar = True
	if b1_click and b1_click%2==1: # if button has been clicked on
		# show meta-cluster labels
		color_label = 'Meta Cluster ID'
		color_dict = micro_to_meta.copy()
	elif b2_click and b2_click%2==1:
		# show weak labels
		color_label = 'Weak Label'
		color_dict = {0:'Other', 1:'Spam', 2:'HT'}
	elif b3_click and b3_click%2==1:
		# show true labels
		color_label = 'MO Label'
		color_dict = {0:'Spam', 1:'HT', 2:'Massage Parlor'}
	elif not pd.isna(attribute_value):
		color_label = 'attribute'
		color_dict = {0:'Q1', 1:'IQR', 2:'Q3'}
	else:
		color_label = 'Color'
		show_color_bar = False
		color_dict = dict(zip(list(micro_to_meta.keys()), ['blue']*len(micro_to_meta)))


	# the ICA, TSNE and UMAP are precomputed and saved to disk for plotting
	if selected_clustering == 0: # ICA 2 comp
		df = pd.read_csv(path_name+"is_ica.zip",index_col=False) #CHANGE THE DATA FILES ACCORDINGLY
		df = df[df.cluster_id.isin(full_df['LSH label'])]
		df.set_index('cluster_id',drop=False,inplace=True)
		if color_label == 'Meta Cluster ID':
			df[color_label] = 'M'+df['cluster_id'].apply(lambda x: color_dict[x]).astype('str')
			labels={color_label:color_label}
		elif color_label == 'Weak Label':
			df[color_label] = df['cluster_id'].apply(lambda x: color_dict[weak_label_map[x]])
			labels={color_label:color_label}
		elif color_label == 'MO Label':
			df[color_label] = df['cluster_id'].apply(lambda x: color_dict[micro_to_true[x]])
			labels={color_label:color_label}
		elif color_label == 'attribute':
			col = dimension_cols[attribute_value]
			col_vals = df[col].values
			if 0 in col_vals:
				col_vals = list(col_vals[col_vals!=0])
				col_vals.append(0)
			else:
				col_vals = list(col_vals[col_vals!=0])
			vals = sorted(col_vals)
			q75, q25 = np.percentile(vals, [75 ,25])
			df[color_label] = df[col].apply(lambda x: 0 if x < q25 else 1 if x >= q25 and x < q75 else 2)
			df[color_label] = df[color_label].apply(lambda x: color_dict[x])
			labels={color_label:col}
		else:
			df[color_label] = ['blue']*len(df)
			labels={color_label:color_label}

		if color_label != 'Color':
			fig = px.scatter(df, x='x',y='y', hover_data=hover_cols, height=1600, labels=labels,\
			color=color_label, color_discrete_sequence=px.colors.qualitative.Vivid)
		else:
			try:
				fig = px.scatter(df, x='x',y='y', hover_data=hover_cols, height=1600, \
				color=color_label)
			except:
				fig = px.scatter(df, x='x',y='y', hover_data=hover_cols, height=1600, \
				color=color_label)
		fig.update_traces(marker=dict(size=marker_size), showlegend=show_color_bar)
		fig.update_layout(dragmode='lasso', font_size=20)

	elif selected_clustering == 1: # TSNE
		# fig = plotlyfromjson("data/tsne_plot.json")
		# fig.update_traces(marker=dict(size=3))
		tsne_res = pkl.load(open(path_name+"all_tsne_res.pkl",'rb')) # Since we are looking at multiple parameter values
		# # df = pd.read_csv("data/is_tsne.zip",index_col=False)
		perp_vals = list(tsne_res.keys())
		titles = []
		for p in perp_vals:
			titles.append(str(p))
		fig = make_subplots(
					rows=1, cols=len(perp_vals),\
					subplot_titles=tuple(titles), \
					horizontal_spacing=0.01, vertical_spacing=0.01, \
					shared_xaxes=True, shared_yaxes=True, \
					# title_text='Perplexity Values'
			)

		template_str = ""
		for i, col in enumerate(hover_cols):
			if i != len(hover_cols)-1:
				template_str += (col+":%{customdata["+str(i)+"]}<br>")
			else:
				template_str += (col+":%{customdata["+str(i)+"]}")

		for i, p in enumerate(perp_vals):
			dd = tsne_res[p]
			dd = dd[dd.cluster_id.isin(full_df['LSH label'])]
			fig.add_scatter(x=list(dd.x), y=list(dd.y), \
								customdata=dd[hover_cols], \
								hovertemplate=template_str,\
								# marker_color=dd[color_label], \
								mode='markers', marker={'opacity':0.3, 'color':'blue', 'size':marker_size}, \
								row=1, col=i+1)
				
			if i == 0:
				fig.layout.annotations[i].update(text=str(p))

		fig.update_layout(height=350, width=650, dragmode="lasso", showlegend=False,\
		font_size=20, title_text='Perplexity Values')
		# fig.update_layout(height=1600, width=1000, showlegend=False)

	elif selected_clustering == 2: # UMAP
		umap_res = pkl.load(open(path_name+"umap_res.pkl",'rb')) # with multiple parameter values 
		# df = pd.read_csv("data/is_umap.zip",index_col=False)	
		nbr_sizes = [10, 50, 100, 200, 500, 1000]
		mini_dists = [0, 0.01, 0.05, 0.1, 0.5, 1]

		titles = []
		for d in nbr_sizes:
			titles.append(str(d))

		template_str = ""
		for i, col in enumerate(hover_cols):
			if i != len(hover_cols)-1:
				template_str += (col+":%{customdata["+str(i)+"]}<br>")
			else:
				template_str += (col+":%{customdata["+str(i)+"]}")

		
		fig = make_subplots(rows=1, cols=len(nbr_sizes), \
				subplot_titles=tuple(titles[:6]), \
				# horizontal_spacing=0.02, vertical_spacing=0.05, \
				shared_xaxes=True, shared_yaxes=True, \
				)

		# mini_dist_ind = 2
		for i in range(len(nbr_sizes)):
			dd = umap_res[i][mini_dist_ind]
			dd = dd[dd.cluster_id.isin(full_df['LSH label'])]
			fig.add_scatter(x=list(dd.x), y=list(dd.y), \
								customdata=dd[hover_cols], \
								hovertemplate=template_str,\
								# marker_color=dd[color_label],\
								mode='markers', marker={'opacity':0.3, 'color':'blue', 'size':marker_size}, \
								row=1, col=i+1)

		fig.update_layout(title_text="Nbrhood Size",height=350, width=700, showlegend=False, font_size=20)
		

	if clickData:
		if selected_clustering == 0: # if a certain point has been clicked on (only if not TSNE and UMAP)
			cluster_id = clickData['points'][0]['customdata'][-1] # retrieve info of clicked point
			fig.add_traces(
				px.scatter(df[df.cluster_id==cluster_id], \
							  x='x',y='y', hover_data=hover_cols).update_traces(marker_color="red",marker_size=4,marker_symbol='star').data
				)
	
		else:
			cluster_id = clickData['points'][0]['customdata'][-1]
			if selected_clustering == 1: # TSNE
				tsne_res = pkl.load(open(path_name+"all_tsne_res.pkl",'rb'))
				for i, p in enumerate(perp_vals):
					dd = tsne_res[p]
					dd = dd[dd.cluster_id.isin(full_df['LSH label'])]
					dd.set_index('cluster_id', drop=False, inplace=True)
					hover_trace = dict(type='scatter', \
						x=list(dd[dd.cluster_id==cluster_id].x), y=list(dd[dd.cluster_id==cluster_id].y), \
						customdata=dd[hover_cols], \
						hovertemplate=template_str,\
						mode='markers', marker={'symbol':'star', 'color': 'red','size':marker_size})
					fig.append_trace(hover_trace, 1, i+1)

			if selected_clustering == 2:
				umap_res = pkl.load(open(path_name+"umap_res.pkl",'rb'))
				nbr_sizes = [10, 50, 100, 200, 500, 1000]
				# mini_dists = [0, 0.01, 0.05, 0.1, 0.5, 1]

				template_str = ""
				for i, col in enumerate(hover_cols):
					if i != len(hover_cols)-1:
						template_str += (col+":%{customdata["+str(i)+"]}<br>")
					else:
						template_str += (col+":%{customdata["+str(i)+"]}")

				for i in range(len(nbr_sizes)):
					dd = umap_res[i][mini_dist_ind]
					dd = dd[dd.cluster_id.isin(full_df['LSH label'])]
					dd.set_index('cluster_id', drop=False, inplace=True)
					fig.add_scatter(x=list(dd[dd.cluster_id==cluster_id].x), y=list(dd[dd.cluster_id==cluster_id].y),\
										customdata=dd[hover_cols], \
										hovertemplate=template_str,\
										mode='markers', marker={'symbol':'star', 'color': 'red','size':marker_size}, \
										row=1, col=i+1)

				fig.update_layout(height=350, width=700, showlegend=False, font_size=20)
	

	if selectedData:
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selectedData['points']:
			selected_points.append(item['customdata'][-1])
		if selected_clustering == 0: # if a certain point has been clicked on (only if not TSNE and UMAP)
			fig.add_traces(
				px.scatter(df.loc[selected_points], \
								  x='x',y='y', hover_data=hover_cols).update_traces(marker_color="black",marker_size=marker_size,marker_symbol='star').data
			)
	
		else:
			if selected_clustering == 1: # TSNE
				tsne_res = pkl.load(open(path_name+"all_tsne_res.pkl",'rb'))
				for i, p in enumerate(perp_vals):
					dd = tsne_res[p]
					dd = dd[dd.cluster_id.isin(full_df['LSH label'])]
					dd.set_index('cluster_id', drop=False, inplace=True)
					hover_trace = dict(type='scatter', \
						x=list(dd.loc[selected_points].x), y=list(dd.loc[selected_points].y), \
						customdata=dd.loc[selected_points][hover_cols], \
						hovertemplate=template_str,\
						mode='markers', marker={'color': 'black','size':marker_size})
					fig.append_trace(hover_trace, 1, i+1)

			if selected_clustering == 2:
				nbr_sizes = [10, 50, 100, 200, 500, 1000]
				umap_res = pkl.load(open(path_name+"umap_res.pkl",'rb'))

				template_str = ""
				for i, col in enumerate(hover_cols):
					if i != len(hover_cols)-1:
						template_str += (col+":%{customdata["+str(i)+"]}<br>")
					else:
						template_str += (col+":%{customdata["+str(i)+"]}")

				for i in range(len(nbr_sizes)):
					dd = umap_res[i][mini_dist_ind]
					dd = dd[dd.cluster_id.isin(full_df['LSH label'])]
					dd.set_index('cluster_id', drop=False, inplace=True)
					fig.add_scatter(x=list(dd.loc[selected_points].x), y=list(dd.loc[selected_points].y), \
										customdata=dd[hover_cols], \
										hovertemplate=template_str,\
										mode='markers', marker={'color': 'black','size':marker_size}, \
										row=1, col=i+1)

				fig.update_layout(height=350, width=700, showlegend=False, font_size=20)

	return fig
	


'''
=============================DISPLAYING SLIDER======================================
'''
@app.callback(
Output('slider-comp','style'), # output: visibility of slider
Input('analysis-type','value')) # input: type of vis. Only make visible if UMAP
def display_slider(selected_clustering):
	if selected_clustering == 2:
		return {'display':'block'}
	else:
		return {'display': 'None'}


'''
=============================ENLARGING THE CLICKED SUB PLOT======================================
'''
@app.callback(
Output('enlarged-graph','style'), # output: the display style of the enlarged graph. Basically to unhide it
Output('enlarged-graph','figure'), # output: the enlarged graph
Input('analysis-type','value'), # input: the type of vis - TSNE/UMAP
Input('micro-cluster-scatter','clickData'), # input: selected point from pair-plots for highlighting in red
Input('main-plot','selectedData'), # input: selected data points for highlighting in pair-plots
Input('slider','value'), # input: minimum distance value for UMAP
Input('b1', 'n_clicks'), # input: if meta cluster label is clicked
Input('b2', 'n_clicks'), # input: if weak label is clicked
Input('b3', 'n_clicks'), # input: if true label is clicked
Input('attribute-type', 'value'))
def enlarge_subplot(selected_clustering, clickData, selectedData, mini_dist_ind, b1_click, b2_click, b3_click, attribute_value):

	tsne_res = pkl.load(open(path_name+"all_tsne_res.pkl",'rb'))
	umap_res = pkl.load(open(path_name+"umap_res.pkl",'rb'))

	show_color_bar = True
	if b1_click and b1_click%2==1: # if button has been clicked on
		# show meta-cluster labels
		color_label = 'Meta Cluster ID'
		color_dict = micro_to_meta.copy()
	elif b2_click and b2_click%2==1:
		# show weak labels
		color_label = 'Weak Label'
		color_dict = {0: 'Other', 1: 'Spam', 2: 'HT'}
	elif b3_click and b3_click%2==1:
		# show true labels
		color_label = 'MO Label'
		color_dict = {0:'Spam', 1:'HT', 2:'Massage Parlor'}
	elif not pd.isna(attribute_value):
		color_label = 'attribute'
		color_dict = {0:'Q1', 1:'IQR', 2:'Q3'}
	else:
		color_label = 'Color'
		show_color_bar = False
		color_dict = dict(zip(list(micro_to_meta.keys()), ['blue']*len(micro_to_meta)))

	if selected_clustering == 0 or not clickData: # not TSNE/UMAP
		dd = tsne_res[10]
		dd = dd[dd.cluster_id.isin(full_df['LSH label'])]
		if color_label == 'Meta Cluster ID':
			dd[color_label] = 'M'+dd['cluster_id'].apply(lambda x: color_dict[x]).astype(str)
			labels={color_label:color_label}
		elif color_label == 'Weak Label':
			dd[color_label] = dd['cluster_id'].apply(lambda x: color_dict[weak_label_map[x]])
			labels={color_label:color_label}
		elif color_label == 'MO Label':
			dd[color_label] = dd['cluster_id'].apply(lambda x: color_dict[micro_to_true[x]])
			labels={color_label:color_label}
		elif color_label == 'attribute':
			col = dimension_cols[attribute_value]
			col_vals = dd[col].values
			# if 0 in col_vals:
			# 	col_vals = list(col_vals[col_vals!=0])
			# 	col_vals.append(0)
			vals = sorted(col_vals)
			# print(vals)
			q75, q25 = np.percentile(vals, [75 ,25])
			dd[color_label] = dd[col].apply(lambda x: 0 if x < q25 else 1 if x >= q25 and x < q75 else 2)
			dd[color_label] = dd[color_label].apply(lambda x: color_dict[x])
			labels={color_label:col}
		else:
			dd[color_label] = ['blue'] * len(dd)
			labels={color_label:color_label}
		try:
			fig = px.scatter(dd, x='x',y='y', hover_data=hover_cols, color=color_label)
		except Exception:
			if color_label == 'Color':
				fig = px.scatter(dd, x='x',y='y', hover_data=hover_cols, color=color_label)
			else:
				fig = px.scatter(dd, x='x',y='y', hover_data=hover_cols, labels=labels, color=color_label, color_discrete_sequence=px.colors.qualitative.Vivid)
		fig.update_traces(showlegend=show_color_bar)
		fig.update_layout(font_size=20)
		return {'display':'none'}, fig

	if clickData: # clicking has happened
		if selected_clustering == 1:
			perp_vals = [5, 10, 20, 30, 40, 50]
			perp_index = clickData['points'][0]['curveNumber']
			if perp_index >= len(perp_vals):
				perp_index = perp_index % len(perp_vals) 
			perplexity = perp_vals[perp_index]
			dd = tsne_res[perplexity]
			dd = dd[dd.cluster_id.isin(full_df['LSH label'])]

			if color_label == 'Meta Cluster ID':
				dd[color_label] = 'M'+dd['cluster_id'].apply(lambda x: color_dict[x]).astype(str)
				labels={color_label:color_label}
			elif color_label == 'Weak Label':
				dd[color_label] = dd['cluster_id'].apply(lambda x: color_dict[weak_label_map[x]])
				labels={color_label:color_label}
			elif color_label == 'MO Label':
				dd[color_label] = dd['cluster_id'].apply(lambda x: color_dict[micro_to_true[x]])
				labels={color_label:color_label}
			elif color_label == 'attribute':
				col = dimension_cols[attribute_value]
				col_vals = dd[col].values
				if 0 in col_vals:
					col_vals = list(col_vals[col_vals!=0])
					col_vals.append(0)
				else:
					col_vals = list(col_vals[col_vals!=0])
				vals = sorted(col_vals)
				q75, q25 = np.percentile(vals, [75 ,25])
				dd[color_label] = dd[col].apply(lambda x: 0 if x < q25 else 1 if x >= q25 and x < q75 else 2)
				dd[color_label] = dd[color_label].apply(lambda x: color_dict[x])
				labels={color_label:col}
			else:
				dd[color_label] = ['blue'] * len(dd)
				labels={color_label:color_label}
			title = 'Perplexity:'+str(perplexity)
			
		else:
			nbr_sizes = [10, 50, 100, 200, 500, 1000]
			nbr_size_ind = clickData['points'][0]['curveNumber']
			
			if nbr_size_ind >= len(nbr_sizes):
				nbr_size_ind = nbr_size_ind % 6

			nbr_size = nbr_sizes[nbr_size_ind]
			dd = umap_res[nbr_size_ind][mini_dist_ind]
			dd = dd[dd.cluster_id.isin(full_df['LSH label'])]
			if color_label == 'Meta Cluster ID':
				dd[color_label] = 'M'+dd['cluster_id'].apply(lambda x: color_dict[x]).astype(str)
				labels={color_label:color_label}
			elif color_label == 'Weak Label':
				dd[color_label] = dd['cluster_id'].apply(lambda x: color_dict[weak_label_map[x]])
				labels={color_label:color_label}
			elif color_label == 'MO Label':
				dd[color_label] = dd['cluster_id'].apply(lambda x: color_dict[micro_to_true[x]])
				labels={color_label:color_label}
			elif color_label == 'attribute':
				col = dimension_cols[attribute_value]
				col_vals = dd[col].values
				if 0 in col_vals:
					col_vals = list(col_vals[col_vals!=0])
					col_vals.append(0)
				else:
					col_vals = list(col_vals[col_vals!=0])
				vals = sorted(col_vals)
				q75, q25 = np.percentile(vals, [75 ,25])
				dd[color_label] = dd[col].apply(lambda x: 0 if x < q25 else 1 if x >= q25 and x < q75 else 2)
				dd[color_label] = dd[color_label].apply(lambda x: color_dict[x])
				labels={color_label:col}
			else:
				dd[color_label] = ['blue'] * len(dd)
				labels={color_label:color_label}
			title = 'Nbrhood Size:'+str(nbr_size)
			
		if color_label != 'Color':
			fig = px.scatter(dd, x='x',y='y', hover_data=hover_cols, color=color_label, labels=labels,\
			color_discrete_sequence=px.colors.qualitative.Vivid)
		else:
			fig = px.scatter(dd, x='x',y='y', hover_data=hover_cols, color=color_label)
		fig.update_layout(width=1000, title_text=title, dragmode='lasso', font_size=20)
		fig.update_traces(marker=dict(size=marker_size), showlegend=show_color_bar)

		if selectedData: # some points in main plot have been selected
			selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
			for item in selectedData['points']:
				print(item)
				selected_points.append(item['customdata'][-1])
			dd.set_index('cluster_id', drop=False, inplace=True)
			fig.add_traces(
				px.scatter(dd.loc[selected_points], \
								  x='x',y='y', hover_data=hover_cols).update_traces(marker_color="cyan",marker_size=marker_size,marker_symbol='star').data
			)
			fig.update_layout(font_size=20)
		return {'display': 'inline-block','margin-left':'40px'}, fig

	

'''
=========================CLUSTER CHARACTERIZATION PLOT==================================== 
'''

# function for highlighting in red the chosen cluster over all pair-plots
@app.callback( 
Output('main-plot','figure'), # output: 'main-plot' figure. Corresponding to app.layout
Input('analysis-type','value'), # input: the type of vis - TSNE/UMAP
Input('main-plot','clickData'), # input: 'main-plot' with clicked data point. 
Input('micro-cluster-scatter','selectedData'), # input: 'micro-cluster-scatter' with points selected using lasso-tool
Input('feature-type', 'value'), # input: 'feature-type' selects which feature columns to show in the pair-plot
Input('scale-type', 'value'), # input: log or linear scale
Input('enlarged-graph','selectedData'),
Input('my-toggle-switch','value')) # input: selected data points from enlarged graph
def highlight_same_clusters(selected_clustering, clickData, selected_clusters, selected_feats, scale, enlarged_selected, toggle_value):

	# if no features are currently selected, then display all
	if not selected_feats:
		# log or linear scale as decided by the radio button
		if scale == 'Log':
			cols = dimension_cols
			labels = dimension_cols
		else:
			cols = list(set(hover_cols)-{'cluster_id'})
			labels={col:col[:-4] for col in cols}
		selected_feats = cols
	else:
		if scale == 'Linear':
			selected_feats = [feat+" Val" for feat in selected_feats]
			labels = selected_feats
		else:
			labels = selected_feats
	print(selected_clustering)
	if selected_clusters and selected_clustering==0: # if some selection of points has been made
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selected_clusters['points']:
			selected_points.append(item['customdata'][-1])

		if toggle_value: # we want to show all the points but grey out the non-selected ones
			to_plot = plot_df.copy()
		else:
			to_plot = plot_df.loc[selected_points]
			classifier_df = plot_df.copy()
			classifier_df['Y'] = np.zeros(len(plot_df))
			classifier_df['Y'].loc[selected_points] = 1
			X = classifier_df[dimension_cols].to_numpy()
			Y = classifier_df['Y'].values
			important_features = np.array(dimension_cols)[get_feature_weights(X, Y)]
			print("Imp feats", important_features)

			if len(selected_feats) == len(dimension_cols):
				selected_feats = important_features
			if scale == 'Linear':
				selected_feats = [feat+" Val" for feat in selected_feats]
			labels = selected_feats


	elif enlarged_selected: # if some selections of points has been made from the enlarged plot
		selected_points = []
		for item in enlarged_selected['points']:
			selected_points.append(item['customdata'][-1])

		if toggle_value: # we want to show all the points but grey out the non-selected ones
			to_plot = plot_df.copy()
		else:
			to_plot = plot_df.loc[selected_points]
			# print(full_df.columns)
			classifier_df = plot_df.copy()
			classifier_df['Y'] = np.zeros(len(plot_df))
			classifier_df['Y'].loc[selected_points] = 1
			X = classifier_df[dimension_cols].to_numpy()
			Y = classifier_df['Y'].values
			important_features = np.array(dimension_cols)[get_feature_weights(X, Y)]
			print("imp features = ", important_features)

			if len(selected_feats) == len(dimension_cols):
				selected_feats = important_features
			if scale == 'Linear':
				selected_feats = [feat+" Val" for feat in selected_feats]
			labels = selected_feats


	else: # if no selection has been made
		to_plot = plot_df.copy()


	if clickData: # if a certain point has been clicked on (we want to track that point across pair-plots)
		cluster_id = clickData['points'][0]['customdata'][-1] # extract info from JSON format

		# pair-plots
		fig = px.scatter_matrix(to_plot, dimensions=selected_feats,opacity=0.6,\
					   hover_data=hover_cols, labels=labels)
		fig.update_traces(marker=dict(size=3))
		if len(selected_feats) > 4:
			fig.layout.font.size = 10
			fig.update_layout(height=1200,width=1200, dragmode='lasso')
		else:
			fig.update_layout(height=700,width=1200, dragmode='lasso', font_size=20)
		# highlight the current clicked point across all pair-plots in red
		fig.add_traces(
		px.scatter_matrix(to_plot[to_plot.cluster_id==cluster_id], \
					  dimensions=selected_feats, labels=labels).update_traces(marker_color="cyan").data
		)
		fig.update_traces(marker=dict(size=marker_size))
		if len(selected_feats) > 4:
			# fig.layout.font.size = 10
			fig.update_layout(height=1200,width=1200, dragmode='lasso', hoverlabel={'font':{'size':50}})
			# fig.update_layout({"yaxis"+str(i+1): {'title':""} for i in range(len(selected_feats))})
		else:
			fig.update_layout(height=700,width=1200, dragmode='lasso', font_size=20)
			# fig.update_layout({"yaxis"+str(i+1): {'title':""} for i in range(len(selected_feats))})
	else: 
		try:
			# fig = ff.create_scatterplotmatrix(to_plot[selected_feats], diag='histogram')
			fig = px.scatter_matrix(to_plot, dimensions=selected_feats,opacity=0.6,\
					   hover_data=hover_cols, labels=labels)
		except:
			# fig = ff.create_scatterplotmatrix(to_plot[selected_feats], diag='histogram')
			fig = px.scatter_matrix(to_plot, dimensions=selected_feats,opacity=0.6,\
					   hover_data=hover_cols, labels=labels)
		fig.update_traces(marker=dict(size=marker_size, color='blue'))
		if len(selected_feats) > 4:
			# fig.update_layout({ax:{"tickmode":"array","tickvals":[]} for ax in fig.to_dict()["layout"] if "axis" in ax})			
			fig.update_layout(height=1200,width=1200, dragmode='lasso')
			# fig.update_layout({"yaxis"+str(i+1): {'title':""} for i in range(len(selected_feats))})
		else:
			fig.update_layout(height=500,width=1200, dragmode='lasso', font_size=13)
			# fig.update_layout({"yaxis"+str(i+1): {'title':""} for i in range(len(selected_feats))})
		
		if toggle_value and (enlarged_selected or selected_clusters):
			fig.add_traces(
				px.scatter_matrix(to_plot[~to_plot.index.isin(selected_points)], \
			dimensions=selected_feats, labels=labels, opacity=0.1).update_traces(marker=dict(size=marker_size,color='grey')).data
			)
		
	return fig


'''
=============================DISPLAYING HISTOGRAM======================================
'''
@app.callback(
Output('main-plot','style'), # output: visibility of pair-plot
Output('hist-plot', 'style'), # output: visibility of histogram
Output('hist-plot','figure'), # output: histogram plot figure
Output('hist','children'), # output: title on the histogram button
Input('feature-type', 'value'), # input: 'feature-type' selects which feature columns to show in the pair-plot
Input('main-plot', 'selectedData'), # input: selected points from the pair-plots
Input('main-plot', 'figure'), # input: the pair-plot figure so that we can extract the current columsn displayed
Input('enlarged-graph', 'selectedData'), # input: selected points from the feature embedding plot
Input('micro-cluster-scatter', 'selectedData'), # input: selected points from ICA
Input('scale-type', 'value'), # input: log or linear scale
Input('hist','n_clicks')) # input: type of vis. Only make visible if UMAP
def display_histogram(feature_values, selected_from_pair_plots, pair_plot, selectedData, selected_from_ica, scale, histogram_button):

	if not histogram_button:
		try: # this is a hacky way of avoiding the Invalid Value error that randomly happens in Dash. Found from plotly community page
			return {'display':'inline-block'}, {'display': 'None'}, px.scatter([]), "Show Individual"
		except Exception:
			return {'display':'inline-block'}, {'display': 'None'}, px.scatter([]), "Show Individual"
		
	if histogram_button%2==1:
		if not feature_values:
			feature_values = [item['label'] for item in pair_plot['data'][0]['dimensions']]
			if len(feature_values) == len(short_dim_cols):
				feature_values = dimension_cols
				if scale == 'Linear':
					feature_values = [feat+" Val" for feat in feature_values]
		else:
			if scale == 'Linear':
				feature_values = [feat+" Val" for feat in feature_values]

		to_plot = plot_df[feature_values]
		if selectedData:
			selected_points = []
			for item in selectedData['points']:
				selected_points.append(item['customdata'][-1])
		elif selected_from_pair_plots:
			selected_points = []
			for item in selected_from_pair_plots['points']:
				selected_points.append(item['customdata'][-1])
		elif selected_from_ica:
			selected_points = []
			for item in selected_from_ica['points']:
				selected_points.append(item['customdata'][-1])

		else:
			selected_points = to_plot.index.values

		to_plot = to_plot.loc[selected_points]

		fig = px.box(to_plot)
		fig.update_layout(font_size=20)
		
		return {'display':'None'}, {'display':'block'}, fig, "Show Pair-Plots"
	else:
		return {'display':'inline-block'}, {'display': 'None'}, px.scatter([]), "Show Individual"
		


if __name__ == '__main__':
	app.run_server(debug=True, port=8801)