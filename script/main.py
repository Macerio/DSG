from DataPrep import DataPrep
from data_prep_AP import DataPrepAP

import pandas as pd


def main() : 
    print(">>> Loading Data")
    df = pd.read_csv('../../data/train_tracking.csv.gz', compression='gzip') 
    target = pd.read_csv('../../data/train_session.csv.gz', compression='gzip')
    products = pd.read_csv('../../data/dsg18_cdiscount_productid_category.csv.gz', compression='gzip')
    
    print(">>> Starting Data Transformation")
    products = products.astype(str)
    products = products.loc[products.category_product_id_level1.str.len() <= 2,:]
    df = pd.merge(df, products, left_on="sku", right_on="product_id", how="left")
    
    data_prep_AP = DataPrepAP()
    
    
if __name__ == '__main__':
    main()

######## FELIX A TOI DE JOUER #########
#######################################
#######################################
###############POUR####################
#######################################
################LE#####################
#######################################
###############MODEL###################
#######################################
#######################################
