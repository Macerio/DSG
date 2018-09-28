import pandas as pd
from sklearn.base import TransformerMixin

class DataPrepAP(TransformerMixin):

	def __init__(self):
		pass

	def fit(self, X, y=None, **kwargs):
		return self

	def transform(self, X, **kwargs):
		df = X
		df_agg_sid = df.groupby("sid").agg({"siteid":"first"})
		pv = pd.DataFrame(df.pivot_table(index="sid", 
										 columns="category_product_id_level1", 
										 values="type", 
										 aggfunc="count").reset_index())
		pv = pv.add_prefix('cat_level1_')
		df_agg_sid = pd.merge(df_agg_sid, pv, left_on='sid', right_on='cat_level1_sid', how='left')
		return df_agg_sid