from DataPrep import DataPrep
import gc
import pandas as pd

# Reading data
nrows = None
nrows = 10000
if nrows==5:
	print('DEBUG MODE')
else:
	print('PRODUCTION MODE')
	
print('reading train...')
df = pd.read_csv('../../data/train_tracking.csv.gz', compression='gzip', nrows=nrows)
print('reading target...')
target = pd.read_csv('../../data/train_session.csv.gz', compression='gzip', nrows=nrows)
print('reading products...')
products = pd.read_csv('../../data/dsg18_cdiscount_productid_category.csv.gz', compression='gzip', nrows=nrows).astype(str)
products = products.loc[products.category_product_id_level1.str.len() <= 2,:]
df.sku = df.sku.astype(str)
df = df.merge(products.rename(columns={'product_id':'sku'}), on='sku', how='left')
df = df.merge(target, on='sid')
del products
gc.collect()

# Preprocessing data
DP = DataPrep()
print('DataPrep fit...')
DP.fit(X=df[[c for c in df.columns if c!='target']], y=df['target'])

print('DataPrep transform...')
train = DP.transform(X=df[[c for c in df.columns if c!='target']])
train = train.merge(target, on='sid')

del df
del target
gc.collect()
