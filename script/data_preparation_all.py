import pandas as pd
import numpy as np

df = pd.read_csv('./train_tracking.csv.gz', compression='gzip').merge(pd.read_csv('./train_session.csv.gz', compression='gzip'), on='sid', how='inner')

# df = df.head(10000)

df.duration = pd.to_timedelta(df.duration).dt.total_seconds()

col_actions = ['ADD_TO_BASKET', 'CAROUSEL',
       'LIST_PRODUCT', 'PA', 'PRODUCT', 'PURCHASE_PRODUCT', 'SEARCH',
       'SHOW_CASE']

##### Target, duration totale / sid

prepared_data = \
pd.DataFrame(df.groupby('sid', as_index=False).agg({'target':['last'], 'duration': ['last']}).values, 
             columns=['sid', 'target', 'duration'])

##### Count de chaque type d'action / sid

prepared_data = \
prepared_data.merge(df.pivot_table(columns='type_simplified', index='sid', values='target', aggfunc='count', fill_value=0).reset_index(),
                    on='sid')

prepared_data.columns = [c+'_count' if c in col_actions else c for c in prepared_data.columns]

##### Nb produits différents / sid

prepared_data = \
prepared_data.merge(pd.DataFrame(df.groupby("sid").dproducts.nunique()).reset_index(level=0).rename(columns={"dproducts":"different_products_in_session"}),
                    on='sid')

##### Tps de chaque type action

df['time_action'] = df.groupby('sid').duration.diff()

prepared_data = \
prepared_data.merge(df.groupby(['sid', 'type_simplified'], as_index=False).time_action.sum().pivot_table(columns='type_simplified', index='sid', values='time_action', aggfunc='sum', fill_value=0).reset_index(),
                    on='sid')

prepared_data.columns = [c+'_time' if c in col_actions else c for c in prepared_data.columns]

##### Probabilities to reach target / action

df['type_avt'] = df.type_simplified.shift(1)
df['same_sid'] = (df.sid==df.sid.shift(1))
df.loc[~df.same_sid, 'type_avt'] = None
df['passage'] = df.type_avt + ' -> ' + df.type_simplified
df['count_r'] = 1*df.same_sid

table_proba_target = df.groupby('passage' , as_index=False).agg({'target' : 'mean','count_r' : lambda x : x.sum()/len(df.passage)})

df.loc[:,'proba_t'] = df.passage.map(dict(zip(table_proba_target.passage, table_proba_target.target)))
df.loc[:,'proba_pass'] = df.passage.map(dict(zip(table_proba_target.passage, table_proba_target.count_r)))
df.loc[:,'proba_t_mean_cum'] = df.groupby('sid').proba_t.cumsum()/(1+df.groupby('sid').cumcount())

best_feature_mc = df.groupby('sid', as_index=False).agg({'proba_t' : ['mean','last'],'proba_t_mean_cum' : 'last', 'proba_pass' : ['mean','last', 'max']})
best_feature_mc.columns = ['sid', 'proba_evnt_t_mean', 'proba_evnt_t_last',  'proba_pass_t_cum_last','proba_pass_mean','proba_pass_last','proba_pass_max'] 

prepared_data = \
prepared_data.merge(best_feature_mc, on='sid')

##### % des actions sur la somme des actions / sid

prepared_data['sum_actions'] = prepared_data[[c+'_count' for c in col_actions]].apply(sum, axis=1)

for col in col_actions:
    prepared_data[col+'_count_perc'] = prepared_data[col+'_count']/prepared_data['sum_actions']

##### % du tps passé sur l'action sur la somme du tps passé / sid

prepared_data.loc[7112, 'sid']

df.loc[df.sid==prepared_data.loc[126576, 'sid'], 'duration']

prepared_data.loc[prepared_data.duration==0.0]

for col in col_actions:
    prepared_data[col+'_time_perc'] = prepared_data.apply(lambda row: row[col+'_time']/row['duration'] if row['duration']>0 else np.nan, axis=1)

##### Save data

name = 'prepared_data_with_propotions'
prepared_data.to_csv('./'+name+'.csv', index=None)
