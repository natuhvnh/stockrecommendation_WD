import pandas as pd 
import numpy as np
import heapq 
from keras.models import load_model
import copy
from utils import *

model = None
topK = None
account_dict = None
share_dict = None
customer_data = None
stock_data = None
categorical_columns = None
wide_columns = None
all_columns = None
val_to_idx = None

def evaluate_model(training_model, top_match, customer, stock, cate_cols, wide_cols, all_cols):
  from statistics import mean 
  global model
  global account_dict
  global share_dict
  global topK
  global customer_data
  global stock_data
  global categorical_columns
  global wide_columns
  global all_columns
  global val_to_idx

  model = training_model
  topK = top_match
  categorical_columns = cate_cols
  wide_columns = wide_cols
  all_columns = all_cols
  customer_data = customer
  stock_data = stock
  # load data
  test = pd.read_pickle('data/test.pickle')
  account_dict = load_dict_from_pickle('data/account_dict.pickle')
  share_dict = load_dict_from_pickle('data/share_dict.pickle')
  val_to_idx = load_dict_from_pickle('val_to_idx.pickle')
  # 
  test = test[['main_account', 'ShareCode', 'test_sample']]
  map_item_score = {}
  result = []
  for index, row in test.iterrows():
    data = {'main_account': row['main_account'], 'ShareCode': row['test_sample']}
    data = pd.DataFrame(data) 
    data['main_account'] = data['main_account'].map(account_dict)
    data['ShareCode'] = data['ShareCode'].map(share_dict)
    data = pd.merge(data, customer_data, left_on="main_account", right_on="main_account", how='left')
    data = pd.merge(data, stock_data, left_on="ShareCode", right_on="ShareCode", how='left')
    data['So_huu_nha_nuoc'] = data['So_huu_nha_nuoc'].fillna(0)
    data['So_huu_nuoc_ngoai'] = data['So_huu_nha_nuoc'].fillna(0)
    data = data.fillna("No info")
    data = data.drop(columns=['main_account', 'ShareCode'])
    data_wide = pd.get_dummies(data, columns=[x for x in categorical_columns])
    data_wide = data_wide.reindex(columns = wide_columns[1:], fill_value=0)
    data_wide = data_wide.values
    for col, encode in val_to_idx.items():
      data[col] = data[col].map(encode) # faster than replace
    data_deep = [data[c] for c in all_columns]
    data_wd = [data_wide] + data_deep
    predictions = model.predict(data_wd, verbose=0)
    map_item_score = dict(zip(row['test_sample'], predictions.reshape(101,)))
    ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
    positive_item = row['ShareCode']
    hit_ratio = get_hit_ratio(ranklist, positive_item)
    result.append(hit_ratio)
  hit_ratio_topK = mean(result)
  # hit_ratio_topK = test['result'].sum(axis=0)/test.shape[0]
  return hit_ratio_topK

def get_result(row):
  map_item_score = {}
  #
  users = copy.copy(row['main_account'])
  users = account_dict[users]
  items = copy.copy(row['test_sample'])
  positive_item = copy.copy(row['ShareCode'])
  positive_item = share_dict[positive_item]
  #
  for i in range(len(items)):
    items[i] = share_dict[items[i]]
    data = {'main_account':[users], 'ShareCode':[items[i]]} 
    data = pd.DataFrame(data)
    data = pd.merge(data, customer_data, left_on="main_account", right_on="main_account", how='left')
    data = pd.merge(data, stock_data, left_on="ShareCode", right_on="ShareCode", how='left')
    data['So_huu_nha_nuoc'] = data['So_huu_nha_nuoc'].fillna(0)
    data['So_huu_nuoc_ngoai'] = data['So_huu_nha_nuoc'].fillna(0)
    data = data.fillna("No info")
    data = data.drop(columns=['main_account', 'ShareCode'])
    data_wide = pd.get_dummies(data, columns=[x for x in categorical_columns])
    data_wide = data_wide.reindex(columns = wide_columns[1:], fill_value=0)
    data_wide = data_wide.values
    #
    for col, encode in val_to_idx.items():
      data[col] = data[col].map(encode) # faster than replace
    data_deep = [data[c] for c in all_columns]
    #
    data_wd = [data_wide] + data_deep
    predictions = model.predict(data_wd, verbose=0)
    map_item_score[items[i]] = predictions
  #
  ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
  hit_ratio = get_hit_ratio(ranklist, positive_item)
  return hit_ratio

def get_hit_ratio(ranklist, positive_item):
  for item in ranklist:
    if item == positive_item:
      return 1
  return 0


