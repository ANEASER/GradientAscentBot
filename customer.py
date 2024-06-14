import numpy as np
import pandas as pd


customer = pd.read_csv('DB/customer_segmentation.csv')
trx_items = pd.read_csv('DB/trx_item.csv')

def identify_customer(customer_id):
    found_customer = customer[customer['customer_code'] == customer_id]
    if found_customer.empty:
        return "Customer not found"
    else:
        
        return found_customer['cluster'].iloc[0]


def customer_transactions(customer_id):
    personalized = trx_items[trx_items['customer_code'] == customer_id]
    if personalized.empty:
        return "No Prior found"
    else:
        grouped = personalized.groupby('item code').agg({
            'item_name': 'first',  # Assuming item_name is the same for each item code
            'sales_quantity': 'sum'
        }).reset_index()
        grouped = grouped.sort_values('sales_quantity', ascending=False)
        item_names_array  = grouped['item_name'].values
        item_names_array = item_names_array[:5].tolist()
        return item_names_array
