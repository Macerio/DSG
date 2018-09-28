import pandas as pd
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from tqdm import tqdm

class DataPrepMC(TransformerMixin):

	def __init__(self):
		pass

	def fit(X, y=None, **kwargs):
        df = X
        df['type_avt'] = df.type_simplified.shift(1)
        df['same_sid'] = (df.sid==df.sid.shift(1))
        df.loc[~df.same_sid, 'type_avt'] = None
        df['passage'] = df.type_avt + ' -> ' + df.type_simplified
        df['count_r'] = 1*df.same_sid
        self.table_proba_target = df.groupby('passage' , as_index=False).agg({'target' : 'mean','count_r' : lambda x : x.sum()/len(df.passage)})
        self.proba_type  = df.groupby('type_simplified' , as_index=False).agg({'target' : 'mean'})        
		return self

	def transform(X, **kwargs):
        df = X
        df.duration = pd.to_timedelta(df.duration).dt.total_seconds()
        
        # Target, duration totale / sid
        if is_train:
            prepared_data = \
            pd.DataFrame(df.groupby('sid', as_index=False).agg({'target':['last'], 'duration': ['last']}).values, 
                         columns=['sid', 'target', 'duration'])
        else:
            prepared_data = \
            pd.DataFrame(df.groupby('sid', as_index=False).agg({'duration': ['last']}).values, 
                         columns=['sid', 'duration'])
        
        # Probabilities to reach target / action
        table_proba_target = self.table_proba_target
        proba_type = self.proba_type
        df['type_avt'] = df.type_simplified.shift(1)
        df['same_sid'] = (df.sid==df.sid.shift(1))
        df.loc[~df.same_sid, 'type_avt'] = None
        df['passage'] = df.type_avt + ' -> ' + df.type_simplified
        df.loc[:,'proba_type'] = df.type_simplified.map(dict(zip(proba_type.type_simplified, proba_type.target)))
        df.loc[:,'proba_t'] = df.passage.map(dict(zip(table_proba_target.passage, table_proba_target.target)))
        df.loc[:,'proba_pass'] = df.passage.map(dict(zip(table_proba_target.passage, table_proba_target.count_r)))
        df.loc[:,'proba_t_mean_cum'] = df.groupby('sid').proba_t.cumsum()/(1+df.groupby('sid').cumcount())
        df.loc[:,'proba_A_sh_B'] = df.proba_type*df.proba_t
        
        
        # Get the mean of prod price
        def get_prod(col) : 
            try : 
                regex = r"(\d{1,}\.\d{2,})"#r"'price': (\d*.\d*)"
                matches = re.finditer(regex, col, re.MULTILINE)
                return [float(match.group()) for match in matches]
            except : 
                return 

        df['parse_price'] = None
        df.loc[~df.products.isnull(),'parse_price'] = df.loc[~df.products.isnull(),'products'].progress_apply(lambda u: get_prod(u))
        df['mean_price'] =  df.parse_price.progress_apply(lambda x : np.mean(x) if x else None )
        df['ecart_mean_price'] =  df.parse_price.progress_apply(lambda x :np.var(x)/np.mean(x) if x else None )

        # Get the mean popularity of the product
        def get_vote(col) : 
            try : 
                regex = r"'rvoter': (\d{1,}\.\d{1,})"#r"'price': (\d*.\d*)"
                matches = re.finditer(regex, col, re.MULTILINE)
                return [match.group() for match in matches]
            except : 
                return

        df['parse_pop'] = None
        df.loc[~df.carproducts.isnull(),'parse_pop'] = df.loc[~df.carproducts.isnull(),'carproducts'].progress_apply(lambda u: get_vote(u))
        df.loc[:,'parse_pop2'] = df.loc[:,'parse_pop'].progress_apply(lambda u: np.mean([float(re.sub(r"'rvoter': ",'',e)) for e in u ] ) if u else None)
        
        best_feature_mc = df.groupby('sid', as_index=False).agg({'proba_t' : ['mean','last'],'proba_t_mean_cum' : 'last', 'proba_pass' : ['mean','last', 'max'], \
                                                                'proba_type' : 'mean', 'proba_A_sh_B' : 'mean', 'parse_pop2' : ['mean', 'max'], \
                                                                'mean_price' : 'mean', 'ecart_mean_price' : ['mean', 'max']})
        best_feature_mc.columns = ['sid', 'proba_evnt_t_mean', 'proba_evnt_t_last',  'proba_pass_t_cum_last','proba_pass_mean','proba_pass_last','proba_pass_max', \
                                  'proba_type', 'proba_A_sh_B', 'parse_pop2_mean', 'parse_pop2_max', 'mean_price', 'ecart_mean_price_mean', 'ecart_mean_price_max'] 
        prepared_data = \
        prepared_data.merge(best_feature_mc, on='sid')
        
		return prepared_data