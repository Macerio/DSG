import pandas as pd
from sklearn.base import TransformerMixin

class DataPrepFR(TransformerMixin):

	def __init__(self):
		pass

	def fit(self, X, y=None, **kwargs):
		return self

	def transform(self, X, **kwargs):
		return X