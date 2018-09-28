from data_prep_MC import DataPrepMC
from data_prep_AP import DataPrepAP
from data_prep_FR import DataPrepFR

import pandas as pd

class DataPrep():
	
	def __init__(self):
		self.DataPrepMC = DataPrepMC()
		self.DataPrepAP = DataPrepAP()
		self.DataPrepFR = DataPrepFR()
	
	def fit(self,X,y):
		self.DataPrepMC.fit(X=X,y=y)
		self.DataPrepAP.fit(X=X,y=y)
		self.DataPrepFR.fit(X=X,y=y)

	def transform(self,X):
		MC = self.DataPrepMC.transform(X=X)
		AP = self.DataPrepAP.transform(X=X)
		FR = self.DataPrepFR.transform(X=X)
		return MC.merge(AP, on="sid").merge(FR, on="sid")





