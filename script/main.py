from DataPrep import DataPrep

import pandas as pd

df = pd.read_csv('./train_tracking.csv.gz', compression='gzip') 
target = pd.read_csv('./train_session.csv.gz', compression='gzip')
products = pd.read_csv('./dsg18_cdiscount_productid_category.csv.gz', compression='gzip')

products = products.astype(str)
products = products.loc[products.category_product_id_level1.str.len() <= 2,:]
df = pd.merge(df, products, left_on="sku", right_on="product_id", how="left")

######## FELIX A TOI DE JOUER #########
#######################################
#######################################
###############POUR####################
#######################################
################LE#####################
#######################################
##############MODEL####################
#######################################
#######################################
