import pandas as pd
import numpy as np
import sys
import ast
import os
from tqdm import tqdm

from collections import Counter
# import networkx as nx
# from networkx.drawing.nx_agraph import graphviz_layout

from sklearn.metrics import adjusted_rand_score as ari

import pickle as pkl

from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE

import umap

from features import find_loc_radii, find_entropy, num_names, url_count, get_num_ads_per_week, get_num_alert_words


data_file = sys.argv[1]
path_name = ["".join(s) for s in data_file.split('/')[:-1]][0]+"/"

data = pd.read_csv(data_file)

if 'locanto' in data_file.lower():
	has_reviews = True
	has_labels = False
	add_social = False
	add_age = False
else:
	has_reviews = False
	if 'marinus' in data_file.lower():
		if 'canada' in data_file.lower():
			has_labels = False
		else:
			has_labels = True
		cities = pd.read_csv(path_name+"cities.csv", index_col='id')
		data['location'] = data['city_id'].apply(lambda x: cities.loc[x]['name'])
		data['geolocation'] = data['city_id'].apply(lambda x: str(cities.loc[x].xcoord) + " " + str(cities.loc[x].ycoord))
		data['img_urls'] = None
		if 'post_date' not in data.columns:
			data = data.rename(columns={'phone':'phone_num', 'date_posted':'post_date', 'body':'description'})
		else:
			data = data.rename(columns={'phone':'phone_num', 'body':'description'})
		add_social = True
		add_age = True
	elif 'synthetic' in data_file.lower():
		has_labels = True
		cities = pd.read_csv(path_name+"cities.csv", index_col='id')
		data['location'] = data['location'].apply(lambda x: cities.loc[x]['name'])
		data['img_urls'] = None
		data = data.rename(columns={'phone':'phone_num', 'date_posted':'post_date', 'body':'description'})
		add_social = False
		add_age = False
	else:
		has_labels = False
		add_social = False
		add_age = False
		geolocation = pd.read_csv("data/geoloc_info.csv",index_col=False)

		data['geolocation'] = data['cleaned_loc'].apply(lambda x: str(geolocation[geolocation.city_ascii==x].lat.values[0])\
							 + " " + str(geolocation[geolocation.city_ascii==x].lng.values[0]))


if 'dict+rule' in data.columns:
	data['Name'] = data['dict+rule'].copy()
elif 'name' in data.columns:
	data['Name'] = data['name'].copy()
elif 'Name' in data.columns:
	pass
elif 'names_in_body' in data.columns:
	data['Name'] = data['names_in_body'].copy()
else:
	data['Name'] = None
# convert the format of extracted names from all string to list of strings
# data['Name'] = data['Name'].apply(lambda x: ast.literal_eval(x))
# print(data['Name'])


possible_links_phone_loc = {}
possible_links_img_loc = {}

# geolocations = pd.read_csv("data/geoloc_info.csv",index_col='city_ascii')

if not os.path.isfile(path_name+'/cluster_sizes.pkl'):
# if True:
# if the feature files have not already been saved, compute them
	# computing features
	cluster_sizes_loc = {}
	phone_count_loc = {}
	loc_count_loc = {}
	loc_radii_loc = {}
	phone_entropy_loc = {}
	valid_phone_ratio_loc = {}
	# num_addresses_loc = {} # we're ignoring this feature for now
	num_names_loc = {}
	num_valid_urls_loc = {}
	num_invalid_urls_loc = {}
	num_ads_per_week_loc = {}
	num_urls_loc = {}
	loc_radii = {}
	age_entropy = {}
	mad_values_loc = {}
	num_social = {}
	num_emails_loc = {}

	num_spam_alert_words_loc = {}
	num_ht_alert_words_loc = {}

	final_labels = {}

	for ads in tqdm(data.groupby('LSH label')):
		possible_links_phone_loc[ads[0]] = ads[1].phone_num.unique()
		possible_links_img_loc[ads[0]] = ads[1].img_urls.unique()
		cluster_sizes_loc[ads[0]] = len(ads[1])
		phone_count_loc[ads[0]] = ads[1].phone_num.nunique()
		loc_count_loc[ads[0]] = ads[1].location.nunique()
		num_spam_alert_words_loc[ads[0]], num_ht_alert_words_loc[ads[0]] = get_num_alert_words(ads[1].description.values)
		loc_radii_loc[ads[0]] = find_loc_radii(ads[1].geolocation.values)
		phone_entropy_loc[ads[0]] = find_entropy(ads[1].phone_num.values)
		num_names_loc[ads[0]] = num_names(ads[1]['Name'].values)
		# num_addresses_loc[ads[0]] = get_addresses(ads[1].description.values)
		num_valid_urls_loc[ads[0]], num_invalid_urls_loc[ads[0]], num_urls_loc[ads[0]] = url_count(ads[1].description.values)
		num_ads_per_week_loc[ads[0]] = get_num_ads_per_week(ads[1], col_name='post_date')

		if add_social:
			num_social[ads[0]] = ads[1].social.nunique()

		if has_labels:
			num_emails_loc[ads[0]] = ads[1].email.nunique()
			# print(ads[1].label)
			final_labels[ads[0]] = ads[1].label.values[0]

		elif num_spam_alert_words_loc[ads[0]] > 0:
			final_labels[ads[0]] = 1
		elif num_ht_alert_words_loc[ads[0]] > 0:
			final_labels[ads[0]] = 2
		else:
			final_labels[ads[0]] = 0
		

	# saving the cluster features
	pkl.dump(cluster_sizes_loc, open(path_name+"cluster_sizes.pkl",'wb'))
	pkl.dump(phone_count_loc, open(path_name+"phone_count.pkl",'wb'))
	pkl.dump(loc_count_loc, open(path_name+"loc_count.pkl",'wb'))
	pkl.dump(phone_entropy_loc, open(path_name+"phone_entropy.pkl",'wb'))
	# pkl.dump(valid_phone_ratio_loc, open(path_name+"valid_phone_ratio.pkl",'wb'))
	# pkl.dump(num_addresses_loc, open(path_name+"num_addresses.pkl",'wb'))
	pkl.dump(loc_radii_loc, open(path_name+"loc_radii.pkl",'wb'))
	pkl.dump(num_names_loc, open(path_name+"num_names.pkl",'wb'))
	pkl.dump(num_valid_urls_loc, open(path_name+"num_valid_urls.pkl",'wb'))
	pkl.dump(num_invalid_urls_loc, open(path_name+"num_invalid_urls.pkl",'wb'))
	pkl.dump(num_ads_per_week_loc, open(path_name+"num_ads_per_week.pkl",'wb'))
	pkl.dump(num_urls_loc, open(path_name+"num_urls.pkl",'wb'))
	# pkl.dump(list(cluster_sizes_loc.keys()), open(path_name+"cluster_ids.pkl",'wb'))
	pkl.dump(final_labels, open(path_name+"final_labels.pkl",'wb'))
	if add_social:
		pkl.dump(num_social, open(path_name+"num_social.pkl",'wb'))
	if has_labels:
		pkl.dump(num_emails_loc, open(path_name+"num_emails.pkl",'wb'))
else:
	# Read in cluster features
	cluster_sizes_loc = pkl.load(open(path_name+"cluster_sizes.pkl",'rb'))
	phone_count_loc = pkl.load(open(path_name+"phone_count.pkl",'rb'))
	loc_count_loc = pkl.load(open(path_name+"loc_count.pkl",'rb'))
	phone_entropy_loc = pkl.load(open(path_name+"phone_entropy.pkl",'rb'))
	# valid_phone_ratio_loc = pkl.load(open(path_name+"valid_phone_ratio.pkl",'rb'))
	# num_addresses_loc = pkl.load(open(path_name+"num_addresses.pkl",'rb'))
	# loc_radii_loc = pkl.load(open(path_name+"loc_radii.pkl",'rb'))
	# num_names_loc = pkl.load(open(path_name+"num_names.pkl",'rb'))
	num_valid_urls_loc = pkl.load(open(path_name+"num_valid_urls.pkl",'rb'))
	num_invalid_urls_loc = pkl.load(open(path_name+"num_invalid_urls.pkl",'rb'))
	num_ads_per_week_loc = pkl.load(open(path_name+"num_ads_per_week.pkl",'rb'))
	num_urls_loc = pkl.load(open(path_name+"num_urls.pkl",'rb'))
	# clus_ids = pkl.load(open(path_name+"cluster_ids.pkl",'rb'))
	final_labels = pkl.load(open(path_name+"final_labels.pkl",'rb'))
	if add_social:
		num_social = pkl.load(open(path_name+"num_social.pkl",'rb'))
	if has_labels:
		num_emails_loc = pkl.load(open(path_name+"num_emails_loc.pkl",'rb'))
	# loc_radii_loc = {}
	num_names_loc = {}
	for ads in tqdm(data.groupby('LSH label')):
	# 	loc_radii_loc[ads[0]] = find_loc_radii(ads[1].geolocation.values)
		num_names_loc[ads[0]] = num_names(ads[1]['Name'].values)
	# pkl.dump(loc_radii_loc, open(path_name+"loc_radii.pkl",'wb'))
	pkl.dump(num_names_loc, open(path_name+"num_names.pkl",'wb'))
	loc_radii_loc = pkl.load(open(path_name+"loc_radii.pkl",'rb'))
	# num_names_loc = pkl.load(open(path_name+"num_names.pkl",'rb'))



def remove(dd, to_remove):
	# function that removes the singleton and noise clusters
	new_dd = {}
	for id, v in dd.items():
		if id not in to_remove:
			new_dd[id] = v
	return new_dd

# remove singleton and noise clusters
to_remove = []
for id, cl in cluster_sizes_loc.items():
	if id == -1 or cl <= 10:
		to_remove.append(id)

# making pair plots of the features
if os.path.isfile(path_name+'/plot_df.csv'): # if the plot file exists, load it
# if False:
	plot_df = pd.read_csv(path_name+"/plot_df.csv", index_col=False)
	# plot_df['Loc Radius'] = [np.log(clsize) if clsize !=0 else 0 for id,clsize in remove(loc_radii_loc, to_remove).items()]
	plot_df['Person Name Count'] = [np.log(clsize) if clsize !=0 else 0 for id,clsize in remove(num_names_loc, to_remove).items()]
	plot_df.to_csv(path_name+"/plot_df.csv",index=False)
else:
	# data needs to be in dataframe format
	plot_df = pd.DataFrame(columns = ["Cluster Size", "Phone Count", "Loc Count", "Phone Entropy", \
						"Loc Radius","Person Name Count",\
						"Valid URLs", "Invalid URLs", "Ads/week", 'Num URLs'])

	plot_df['Cluster Size'] = [np.log(clsize) for id, clsize in remove(cluster_sizes_loc, to_remove).items()]
	plot_df['Phone Count'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in remove(phone_count_loc, to_remove).items()]
	plot_df['Loc Count'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in remove(loc_count_loc, to_remove).items()]
	plot_df['Phone Entropy'] = [clsize if clsize != 0 else 0 for id,clsize in remove(phone_entropy_loc, to_remove).items()]
	# plot_df['Valid/Invalid Phone'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in remove(valid_phone_ratio_loc, to_remove).items()]
	# plot_df['Num Addresses'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in remove(num_addresses_loc, to_remove).items()]
	plot_df['Loc Radius'] = [np.log(clsize) if clsize !=0 else 0 for id,clsize in remove(loc_radii_loc, to_remove).items()]
	plot_df['Person Name Count'] = [np.log(clsize) if clsize !=0 else 0 for id,clsize in remove(num_names_loc, to_remove).items()]
	plot_df['Valid URLs'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in remove(num_valid_urls_loc, to_remove).items()]
	plot_df['Invalid URLs'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in remove(num_invalid_urls_loc, to_remove).items()]
	plot_df['Ads/week'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in remove(num_ads_per_week_loc, to_remove).items()]
	plot_df['Num URLs'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in remove(num_urls_loc, to_remove).items()]
	if has_labels:
		plot_df['Num Emails'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in remove(num_emails_loc, to_remove).items()]
		plot_df['Num Emails Val'] = [clsize for id,clsize in remove(num_emails_loc, to_remove).items()]

	if add_social:
		plot_df['Num Social'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in remove(num_social, to_remove).items()]
		plot_df['Num Social Val'] = [clsize for id,clsize in remove(num_social, to_remove).items()]

	plot_df['Cluster Size Val'] = [clsize for id, clsize in remove(cluster_sizes_loc, to_remove).items()]
	plot_df['Phone Count Val'] = [clsize for id,clsize in remove(phone_count_loc, to_remove).items()]
	plot_df['Loc Count Val'] = [clsize for id,clsize in remove(loc_count_loc, to_remove).items()]
	plot_df['Phone Entropy Val'] = [clsize for id,clsize in remove(phone_entropy_loc, to_remove).items()]
	# plot_df['Valid/Invalid Phone Val'] = [clsize for id,clsize in remove(valid_phone_ratio_loc, to_remove).items()]
	# plot_df['Num Addresses Val'] = [clsize for id,clsize in remove(num_addresses_loc, to_remove).items()]
	plot_df['Loc Radius Val'] = [clsize for id,clsize in remove(loc_radii_loc, to_remove).items()]
	plot_df['Person Name Count Val'] = [clsize for id,clsize in remove(num_names_loc, to_remove).items()]
	plot_df['Valid URLs Val'] = [clsize for id,clsize in remove(num_valid_urls_loc, to_remove).items()]
	plot_df['Invalid URLs Val'] = [clsize for id,clsize in remove(num_invalid_urls_loc, to_remove).items()]
	plot_df['Ads/week Val'] = [clsize for id,clsize in remove(num_ads_per_week_loc, to_remove).items()]
	plot_df['Num URLs Val'] = [clsize for id,clsize in remove(num_urls_loc, to_remove).items()]
	plot_df['cluster_id'] = [k for k in remove(cluster_sizes_loc, to_remove).keys()]

	
	plot_df.to_csv(path_name+"/plot_df.csv",index=False)


# if True:
if 'Meta label' not in data.columns:
	## meta-cluster labelling
	meta_cluster_labels = {}
	meta_label = 0
	for cluster_1 in tqdm(data['LSH label'].unique()):
		if cluster_1 in meta_cluster_labels.keys():
			continue
		else:
			meta_cluster_labels[cluster_1] = meta_label
			meta_label += 1
		for cluster_2 in data['LSH label'].unique():
			if cluster_1 != cluster_2:
				phone1 = data[data['LSH label']==cluster_1].phone_num.unique()
				phone2 = data[data['LSH label']==cluster_2].phone_num.unique()
				common_phones = set(phone1) & set(phone2)
				
				img1 = data[data['LSH label']==cluster_1].img_urls.unique()
				img2 = data[data['LSH label']==cluster_2].img_urls.unique()
				common_img = set(img1) & set(img2)
				
				name1 = data[data['LSH label']==cluster_1].Name.unique()
				name2 = data[data['LSH label']==cluster_2].Name.unique()
				common_name = set(name1) & set(name2)

				email1 = data[data['LSH label']==cluster_1].email.unique()
				email2 = data[data['LSH label']==cluster_2].email.unique()
				common_email = set(email1) & set(email2)
				
				if len(common_phones) == 1 and pd.isna(list(common_phones)[0]):
					continue    
				
				if len(common_img) == 1 and pd.isna(list(common_img)[0]):
					continue
				
				if len(common_name) == 1 and pd.isna(list(common_name)[0]):
					continue

				if len(common_email) == 1 and pd.isna(list(common_email)[0]):
					continue
				
				if len(common_phones) != 0 or len(common_img) != 0 or len(common_name) != 0 or len(common_email) != 0:
					meta_flag = True
				else:
					meta_flag = False     
			
				if meta_flag:
					meta_cluster_labels[cluster_2] = meta_cluster_labels[cluster_1]   


	data['Meta label'] = data['LSH label'].apply(lambda x: meta_cluster_labels[x])
	data.to_csv(data_file, index=False)


# print(plot_df.columns)
# plot_df.drop(columns=['Num Addresses', 'Valid/Invalid Phone'], inplace=True)

hover_cols = ['Cluster Size Val','Phone Count Val', 'Loc Count Val', 'Phone Entropy Val', \
			 'Loc Radius Val','Person Name Count Val','Valid URLs Val', 'Invalid URLs Val',\
			 'Ads/week Val', 'Num URLs Val', 'cluster_id']

filtered_df = plot_df[plot_df['Cluster Size Val'] > 10]


## ICA	
print(".....ICA....")
# print(filtered_df)
transformer2 = FastICA(n_components=2, random_state=0)
is_ica2 = transformer2.fit_transform(filtered_df)

is_ica_df2 = pd.DataFrame(is_ica2, columns=['x','y'], index=filtered_df.index)
is_ica_df21 = is_ica_df2.join(filtered_df, how='inner')
is_ica_df21.to_csv(path_name+"is_ica.zip",index=False)


## TSNE
print("....TSNE.....")
perp_vals = [5, 10, 20, 30, 40, 50]
tsne_res = np.empty(len(perp_vals),dtype=object)
for i, perp in tqdm(enumerate(perp_vals)):
	tsne_emb = TSNE(perplexity=perp, n_iter=5000).fit_transform(filtered_df)

	tsne_emb = pd.DataFrame(tsne_emb, columns=['x','y'], index=filtered_df.index)
	tsne_emb2 = tsne_emb.join(filtered_df, how='inner')
	tsne_res[i] = tsne_emb2 

tsne_dict = {}
for perp, tsne_embs in zip(perp_vals, tsne_res):
	tsne_dict[perp] = tsne_embs
	
pkl.dump(tsne_dict, open(path_name+"all_tsne_res.pkl",'wb'))


## UMAP
print("...UMAP....")

nbr_sizes = [10, 50, 100, 200, 500, 1000]
mini_dists = [0, 0.01, 0.05, 0.1, 0.5, 1]

is_np = filtered_df[set(filtered_df.columns)-set(hover_cols)].to_numpy()

umap_res = np.empty(shape=[len(nbr_sizes),len(mini_dists)],dtype=object)
for i, nbr in tqdm(enumerate(nbr_sizes)):
	for j, dist in enumerate(mini_dists):
		reducer = umap.UMAP(n_neighbors=nbr, min_dist=dist)
		embedding = reducer.fit_transform(is_np)
		is_umap = pd.DataFrame(embedding, columns=['x','y'], index=filtered_df.index)
		is_umap = is_umap.join(filtered_df, how='inner')
		umap_res[i][j] = is_umap


pkl.dump(umap_res, open(path_name+"umap_res.pkl",'wb'))