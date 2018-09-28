from data_prep_MC import DataPrepMC
from data_prep_AP import DataPrepAP
from data_prep_FR import DataPrepFR

import pandas as pd

class DataPrep():
	
	def __init__(self):
		self.DataPrepMC = DataPrepMC()
		self.DataPrepAP = DataPrepAP()
		self.DataPrepFR = DataPrepFR()
	
	def fit(X,y):
		self.DataPrepMC.fit(X,y)
		self.DataPrepAP.fit(X,y)
		self.DataPrepFR.fit(X,y)

	def transform(X):
		MC = DataPrepMC.transform(X)
		AP = DataPrepAP.transform(X)
		FR = DataPrepFR.transform(X)
		final = pd.merge(pd.merge(MC, AP, on="sid"), FR, on="sid")
		return final