# author: Lori Fang
# date: 2020-05-19

"""
This script extracts every customer's first order from the shopify_orders 
and then match it to every customer by their unique client number. Then it will remove useless info
and extract important info from selected features. Then it will export the clean data for the classification model.

Usage: data_cleaning.py --dbname=<dbname> --user=<user> --password=<password> --host=<host> --out_dir=<out_dir>

Options:
--dbname=<dbname>          Database name (riversol_TEST_DB).
--user=<user>              User name.
--password=<password>      Password for the user name.
--host=<host>              host(IP address).
--out_dir=<out_dir>        Path to directory where cleaned data will be exported.

"""

import psycopg2
import pandas as pd
import numpy as np
import re
from docopt import docopt
import gender_guesser.detector as gender

def get_website(text, target): #target = "First Visit" or "Order Url"
    if text == None:
        return None
    else:
        regex = target + "+[^,;]+,"
        match = re.search(regex, text)
        if match == None:
            return None
        else:
            link = match.group()
            website = re.search('(?<=utm_source=).([^&^%]+)', link)
            if website == None:
                return None
            else:
                return website.group()

def get_product_type(name):
    if name == None:
        return None
    elif re.search('Anti-Aging', name, re.IGNORECASE) or re.search('Aging', name, re.IGNORECASE) or re.search('Age', name, re.IGNORECASE):
        return 'Anti-Aging'
    elif re.search('Redness', name, re.IGNORECASE) or re.search('Red', name, re.IGNORECASE):
        return 'Redness'
    else:
        return 'Other'

def get_skin_type(name):
    if name == None:
        return None
    elif re.search('normal to dry', name, re.IGNORECASE) or re.search('normal / dry', name, re.IGNORECASE):
        return 'Normal to Dry'
    elif re.search('normal to oily', name, re.IGNORECASE) or re.search('normal / oily', name, re.IGNORECASE):
        return 'Normal to Oily'
    elif re.search('very dry', name, re.IGNORECASE):
        return 'Very Dry'
    elif re.search('dry', name, re.IGNORECASE):
        return 'Dry'
    elif re.search('combination', name, re.IGNORECASE):
        return 'Combination'
    elif re.search('very oily', name, re.IGNORECASE):
        return 'Very Oily'
    elif re.search('oily', name, re.IGNORECASE):
        return 'Oily'
    else:
        return 'Unknown'

def standardize_name(name):
    if name == None or name == "":
        return None
    elif name[-1]==" ":
        name = name[0:-1]
    return name.upper()

def generalize_campaign(campaign):
    campaigns = ["redditad", "pinterest", "Messenger_Stories", "Instagram_Stories", 
             "Instagram_Feed", "Instagram_Explore", "influencer", "googleshopping", "Facebook_Mobile_Feed", 
            "facebook_messenger", "Facebook_Marketplace", "Facebook_Instant_Articles", "facebook_IG_plus",
            "Facebook_Desktop_Feed", "cbcarticle", "Bingros", 'bing', 
             "6168286054243", "6166380916443", "6121570192043", "6104146934443", "6104145954643"]
    count = len(campaigns)
    while count >0:
        if campaigns[count-1] in str(campaign):
            index = count-1
            count = -1
            return campaigns[index]
        else:
            count-=1
    if count == 0:
        return "other"
    
def buy_or_not(cid, purchaser):
    if cid in purchaser:
        return True
    else:
        return False

opt = docopt(__doc__)

def main(dbname, user, password, host, out_dir):
    # get row data from tables
    sql = \
    """
    CREATE TEMP VIEW first_order(customer_id, order_id, ordered_at, customer_total_spent, total_price, note_attributes, cancelled_at, order_tag) AS
    SELECT customer_id, order_id, created_at, customer_total_spent, total_price, note_attributes, cancelled_at, tags
        FROM   (SELECT customer_id, order_id, created_at, customer_total_spent, total_price, note_attributes, cancelled_at, tags,
                   RANK() OVER (PARTITION BY customer_id ORDER BY created_at ASC) AS rk
                   FROM   shopify_orders) t
        WHERE  rk = 1;
    SELECT c.*, f.order_id, f.ordered_at, f.customer_total_spent, f.total_price, f.note_attributes, f.cancelled_at, f.order_tag, i.name, i.product_id, i.variant_id, i.vendor
    FROM shopify_customers c
    LEFT JOIN first_order f
    ON f.customer_id = c.customer_id
    LEFT JOIN shopify_line_items i
    ON i.order_id = f.order_id;
    """
    
    sql_dup_email = "SELECT * FROM duplicate_emails;"
    
    sql_purchaser = \
    """
    CREATE TEMP VIEW my_order(customer_id, order_id, email, processed_at, customer_total_spent, total_line_items_price, total_discounts, total_price, tags) AS
    SELECT customer_id, order_id, email, processed_at, customer_total_spent, total_line_items_price, total_discounts, total_price, tags
        FROM shopify_orders
        WHERE total_line_items_price > 0 AND total_price > 0;

    SELECT o.*, i.name
    FROM my_order o
    LEFT JOIN shopify_line_items i
    ON i.order_id = o.order_id;
    """
    
    # will change the bdname, user, password, and host into input variables later.
    conn = psycopg2.connect(dbname=str(dbname), user=str(user), password=str(password), host=str(host))
    cur = conn.cursor()
    cur.execute(sql)
    data = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(data, columns=colnames)
    
    # only keep features that matter
    df = df[['customer_id', 'first_name', 'accepts_marketing', 'email', 'tags', 
             'default_address_province', 'default_address_country', 
             'ordered_at',  'total_price', 'orders_count', 'order_tag',
             'note_attributes', 'cancelled_at', 'name', 
             'default_address_address1', 'default_address_address2', 'default_address_company', 'default_address_zip']] 
    
    # get duplicate emails table
    cur = conn.cursor()
    cur.execute(sql_dup_email)
    table = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    dup_emails = pd.DataFrame(table, columns=colnames) 
    
    # get purchaser transaction log
    cur = conn.cursor()
    cur.execute(sql_purchaser)
    translog = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    transactions = pd.DataFrame(translog, columns=colnames) 
          
    # remove customers with cancelled orders
    df = df[~df['cancelled_at'].notna()]
           
    # remove outliers who paid a lot for the sample
    df = df[df['total_price']<20]
    
    # remove customers that were not sample takers
    df = df[df['name'].str.contains('Sample', regex = True, na=False)]
    df = df.reset_index(drop=True)
    
    # drop all duplicated emails and custsomers with no email
    df = df.drop_duplicates(subset = 'email', keep = False)
    duplicate_list = dup_emails["duplicate_email"].tolist()
    duplicate_list = list(pd.Series(duplicate_list).dropna())
    unique_duplicate_emails = list(set(duplicate_list))
    df = df[~df['email'].isin(unique_duplicate_emails)]
    
    # create y-variable of whether or not customers made at least 1 purchase after taking sample
    df["maybe_buy"] = df["orders_count"]>1
    
    # get first interaction website and first order webesite
    df['fv_site'] = df[['note_attributes']].applymap(lambda text: get_website(text, "First Visit"))

    # get gender from first name
    d = gender.Detector()
    df['gender'] = df[['first_name']].applymap(lambda name: d.get_gender(name))

    # standardize province and country names
    df["default_address_province"] = df[['default_address_province']].applymap(lambda name: standardize_name(name))
    df["default_address_country"] = df[['default_address_country']].applymap(lambda name: standardize_name(name))
    df["location"] = df["default_address_province"] +", "+ df["default_address_country"]
    df.loc[df['location'] == "NEWFOUNDLAND AND LABRADOR, CANADA", ['location']] = "NEWFOUNDLAND, CANADA"
    
    # add month and year of the first order
    df["ordered_at"] = pd.to_datetime(df["ordered_at"], utc=True).dt.date
    df["ordered_month"] = df[["ordered_at"]].applymap(lambda time: time.month)
    df["ordered_year"] = df[["ordered_at"]].applymap(lambda time: time.year)
    newest = max(df['ordered_at'])
    df["days_from_sample"] = df[["ordered_at"]].applymap(lambda date: (newest-date).days)

    # apply product type categorization to all rows
    df['product_type'] = df[['name']].applymap(lambda name: get_product_type(name))

    # categorize skin types based on product name information
    df['skin_type'] = df[['name']].applymap(lambda name: get_skin_type(name))
    
    # if the first order was at not charge (free shipping)
    df["free_shipping"] = df["total_price"]==0
            
    # replace NaN with "unknown"
    for i in df.columns:
        df[i] = df[i].replace({np.nan:"unknown"})
        
    # extract campaign website
    df["fv_site"] = df[['fv_site']].applymap(lambda campaign: generalize_campaign(campaign))
        
    # remove tags with fraud, test, retailer, and scammer
    df = df[~df['tags'].str.contains('FRAUD|test|Retailer|Scammer', regex = True)]
    df = df[~df['order_tag'].str.contains('(?i)ws_order|wholesale', regex = True, na = False)]
    df = df[(df.order_tag=='') | (df.order_tag=='UK SAMPLE')]
    
    df = df[['customer_id', 'accepts_marketing', 'ordered_month', 'ordered_year', 'days_from_sample',
             'location', 'gender', 'free_shipping', 'product_type', 
            'skin_type', 'fv_site', 'maybe_buy']]   
        
    purchaser = transactions[transactions["customer_total_spent"]>20]["customer_id"].unique()
    df["buy"] = df[["customer_id"]].applymap(lambda cid: buy_or_not(cid, purchaser))
    # move double sample takers
    df = df[~((df["maybe_buy"]==True)&(df["buy"]==False))]
    df = df.drop(columns = ['maybe_buy'])

    # export cleaned data
    try:
        df.to_csv(out_dir + '/cleaned_df.csv', index=False)
    except Exception as e:
        print(f"Directory does not exist. Exception: {e}")

if __name__ == "__main__":
    main(opt["--dbname"], opt["--user"], opt["--password"], opt["--host"], opt["--out_dir"])
