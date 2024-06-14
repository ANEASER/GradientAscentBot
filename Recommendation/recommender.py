import numpy as np
import mysql.connector as mysql
import pandas as pd



df = pd.read_csv("./CSV/item_info.csv")
df["index"] = df.index 

cosine_sim_loaded = np.load('output.txt')

def get_name_from_index(index):
    return df[df.index	 == index]["item_name"].values[0]

def get_index_from_name(itemname):
    return df[df.item_name == itemname]["index"].values[0]



recommended_products = []


def recommend(product_user_likes):
    similar_products = []

    for product in product_user_likes:
        product_index = get_index_from_name(product)
        similar_products_loaded = list(enumerate(cosine_sim_loaded[product_index]))

        sorted_similar_product = sorted(similar_products_loaded, key=lambda x: x[1], reverse=True)

        sorted_similar_product = [x for x in sorted_similar_product if x[1] > 0]
        
        unique_similar_product = []
        seen_indexes = set()

        for index, cosine_similarity in sorted_similar_product:
            if index not in seen_indexes:
                unique_similar_product.append((index, cosine_similarity))
                seen_indexes.add(index)


        for index, _ in unique_similar_product[:3]:
            if get_name_from_index(index) not in similar_products:
                if get_name_from_index(index) not in product_user_likes:
                    similar_products.append(get_name_from_index(index))
        
    
    
    return {"recommendation": similar_products[:5]}