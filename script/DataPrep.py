from data_prep_MC import data_prep_MC
from data_prep_AP import data_prep_AP
from data_prep_FR import data_prep_FR

import pandas as pd

class DataPrep():
	def __init__(self):
		self.data_prep_MC = data_prep_MC()
		self.data_prep_AP = data_prep_AP()
		self.data_prep_FR = data_prep_FR()
	
	def fit(X,y):
		self.data_prep_MC.fit(X,y)
		self.data_prep_AP.fit(X,y)
		self.data_prep_FR.fit(X,y)

	def transform(X):
		MC = data_prep_MC.transform(X)
		AP = data_prep_AP.transform(X)
		FR = data_prep_FR.transform(X)
		final = pd.merge(pd.merge(MC, AP, on="sid"), FR, on="sid")
		return final