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
		print("Fit MC")
		self.DataPrepMC.fit(X=X,y=y)
		print("Fit AP")
		self.DataPrepAP.fit(X=X,y=y)
		print("Fit FR")
		self.DataPrepFR.fit(X=X,y=y)

	def transform(self,X):
		print("Transform MC")
		MC = self.DataPrepMC.transform(X=X)
		print("Transform AP")
		AP = self.DataPrepAP.transform(X=X)
		print("Transform FR")
		FR = self.DataPrepFR.transform(X=X)
		return MC.merge(AP, on="sid").merge(FR, on="sid")





