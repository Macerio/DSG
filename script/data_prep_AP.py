import pandas as pd
from sklearn.base import TransformerMixin

class DataPrepAP(TransformerMixin):

	def __init__(self):
		pass

	def fit(X, y=None, **kwargs):
		return self

	def transform(X, **kwargs):
		pv = pd.DataFrame(df.pivot_table(index="sid", columns="category_product_id_level1", values="type", aggfunc="count").reset_index())
		pv = pv.add_prefix('cat_level1_')
		df = pd.merge(df, pv, left_on='sid', right_on='cat_level1_sid', how='left')