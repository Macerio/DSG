import pandas as pd
from sklearn.base import TransformerMixin

class DataPrepAP(TransformerMixin):

	def __init__(self):
		pass

	def fit(self, X, y=None, **kwargs):
		return self

	def transform(self, X, **kwargs):
		df = X
		df_agg_sid = df.groupby("sid").agg({"siteid":"first"}).reset_index()
		replace_dict = {"cF8tnO1rK7fIBxVIs+AW4w==":0, "5QmFu8A6HVU6cIT8YqnAZg==":2, "Sa7a/unMwmgA2MyaUZidxg==":1}
		df_agg_sid.siteid = df_agg_sid.siteid.map(replace_dict)
		pv = pd.DataFrame(df.pivot_table(index="sid", 
		                                 columns="category_product_id_level1", 
		                                 values="type", 
		                                 aggfunc="count").reset_index())
		pv = pv.add_prefix('cat_level1_')
		df_agg_sid = pd.merge(df_agg_sid, pv, left_on='sid', right_on='cat_level1_sid', how='left')
		return df_agg_sid