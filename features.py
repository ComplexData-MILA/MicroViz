from urlextract import URLExtract
import validators
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import haversine as hs
from scipy.stats import entropy
from collections import Counter
import ast
from tqdm import tqdm

# FUNCTION THAT COUTNS THE NUMBER OF VALID, INVALID AND TOTAL URLS IN THE AD TEXT
def url_count(texts):
    num_invalid_urls = 0
    num_valid_urls = 0
    num_urls = 0
    extractor = URLExtract()
    for text in texts:
        urls = extractor.find_urls(text)
        num_urls += len(urls)
        for url in urls:
            valid=validators.url(url)
            if valid:
                num_valid_urls += 1
            else:
                num_invalid_urls += 1
    return num_valid_urls, num_invalid_urls, num_urls


# FUNCTION THAT CALCULATES THE RADIUS OF LOCATIONS COVERED IN A CLUSTER
def find_loc_radii(list_locs):
    all_radii = []
    for loc1 in tqdm(list_locs):
        for loc2 in list_locs:
            if loc1 == loc2:
                continue
            if type(loc1) == str:
                    x1 = float(loc1.split()[0])
                    y1 = float(loc1.split()[1])

                    x2 = float(loc2.split()[0])
                    y2 = float(loc2.split()[1])
                    all_radii.append(hs.haversine((x1, y1),(x2, y2)))
            else:
                try:
                    if type(loc1[0]) == np.ndarray:
                        loc1 = loc1[0]
                except:
                    continue
                x1 = loc1[0]
                y1 = loc1[1]
                
                if type(loc2[0]) == np.ndarray:
                    loc2 = loc2[0]

                x2 = loc2[0]
                y2 = loc2[1]
                all_radii.append(hs.haversine((x1, y1),(x2, y2)))
    if len(all_radii) != 0:
        return max(all_radii)
    else:
        return 0.


# FUNCTION RETURNS THE ENTROPY OF PHONE NUMBERS
def find_entropy(numbers, base=None):
    number_freq = list(Counter(numbers).values())
    ent = entropy(number_freq, base=base)
    return ent

# FUNCTION RETURNS THE NUMBER OF ALERT WORDS IN THE AD REVIEWS/AD DESCRIPTION

spam_alert_words = ['scam', 'spam', 'fake' ,'steal', 'scammer', 'fraud', 'one night', 'weekend',  \
               'robbed', 'boyfriend', 'cops', 'bitcoin', 'money',  'waste',  'fake', 'exotic'
                    ]
ht_alert_words = ['exotic', 'gfe', 'GF', 'midget',  'cuddle',  'asian', 'native', \
                  'eskimo']

def get_num_alert_words(texts, reviews=None):
    num_susp_words_spam = 0
    num_susp_words_ht = 0
    for text in texts:
        if reviews != None and all(reviews) != None:
            for word in spam_alert_words:
                if word in reviews:
                    num_susp_words_spam += 1
        for word in ht_alert_words:
            if word in text:
                # print(text, word)
                num_susp_words_ht += 1
    return num_susp_words_spam, num_susp_words_ht    

# FUNCTION RETURNS THE NUMBER OF ADS IN A WEEK
def get_num_ads_per_week(cluster, col_name='date_posted'):
    num_ads = 0
    ads_per_date = {}
    ads_per_week = []
    # print(cluster.columns, col_name)
    if col_name in cluster.columns:
        # print(cluster[col_name])
        dates = list(cluster[col_name].unique())
        for date in dates:
            ads_per_date[date] = len(cluster[cluster[col_name] == date])
        dates.sort()
    else:
        dates = list(pd.to_datetime(cluster[col_name].unique(), infer_datetime_format=True))
        for date in dates:
            mod_date = date.strftime("%m/%d/%Y %H:%M:%S")
            ads_per_date[date] = len(cluster[cluster[col_name] == mod_date])
        dates.sort()

    for i in range(len(dates)-1):
        try:
            d1 = pd.to_datetime(dates[i], infer_datetime_format=True)
        except:
            continue
        try:
            d2 = pd.to_datetime(dates[i+1], infer_datetime_format=True)
        except:
            continue

        day1 = (d1 - timedelta(days=d1.weekday()))
        day2 = (d2 - timedelta(days=d2.weekday()))
        num_weeks = (day2 - day1).days / 7
        if num_weeks == 0:
            ads_per_week.append(ads_per_date[dates[i]] + ads_per_date[dates[i+1]])
        else:
            ads_per_week.append(1)
    if len(ads_per_week) == 0:
        return 0.
    else:
        return max(ads_per_week)



# FUNCTION RETURNS THE NUMBER OF NAMES IN AN AD CLUSTER
def num_names(list_names):
    n_names_per_ad = []
    num_names = 0
    for names in list_names:
        if pd.isna(names):
            continue
        nn = names.split()
        new_l = []
        for n in nn:
            if n[0] == '[':
                n = n[1:]
            if n[-1] in [']', ',']:
                n = n[:-1]
            new_l.append(n[1:-1])
        n_names_per_ad.append(len(np.unique(new_l)))
        num_names += len(new_l)
    # return np.max(n_names_per_ad)
    return num_names



