import functions
import sqlite3
from pickle import dump
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV
import fields


def get_total_sql():
    return f"""
    select froms.offer_id, froms.item_id, tos.item_id, trade.to_status, trade.to_reason, item_to.*, item_from.* 
    from
        (select offer_id, item_id  from offer_item where side ='to') tos,
        (select offer_id, item_id  from offer_item where side ='from') froms, 
        (select id, {','.join(fields.get_fields())} from items) as item_from,
        (select id, {','.join(fields.get_fields())} from items) as item_to,
        trade
    where 
        froms.offer_id = tos.offer_id 
        and trade.offer_id = froms.offer_id
        and froms.offer_id in (select offer_id from offer_item where side = 'from' group by offer_id having count(*)=1) 
        and froms.offer_id in (select offer_id from offer_item where side = 'to' group by offer_id having count(*)=1) 
        and froms.item_id = item_from.id 
        and tos.item_id = item_to.id
    """


def convert_to_offer_row(sql_row):
    offer = dict()
    fields_used = fields.get_fields()
    offer["result"] = offer_result_from_sql(sql_row)
    if offer["result"] == -1:
        return None
    for i in range(0, len(fields_used)):
        shift = 5
        val_from = float(sql_row[shift + i + 1])
        val_to = float(sql_row[shift + len(fields_used) + i + 2])
        if val_to == -1 or val_from == -1:
            return None
        if fields_used[i] in ["bundles_packages", "bundles_all", "giveaway_count"]:
            val_from = functions.get_sigmoid(val_from)
            val_to = functions.get_sigmoid(val_to)
        offer[fields_used[i]] = val_to - val_from
    return offer


def offer_result_from_sql(sql_row):
    if sql_row[3] in ['declined'] and sql_row[4] in ['do not want', 'not worth it to this user', '', 'countered']:
        return 0
    if sql_row[3] in ['completed', 'accepted']:
        return 1
    return -1


def reverse_row(offer_row):
    rev_data = dict()
    for key in offer_row.keys():
        rev_data[key] = offer_row[key] * -1
    rev_data['result'] = 1
    return rev_data


if __name__ == '__main__':
    conn = sqlite3.connect("barter.sqlite")
    cur = conn.execute(get_total_sql())
    offers = list()

    for row in cur:
        try:
            data_row = convert_to_offer_row(cur.fetchone())
            if data_row is not None:
                offers.append(data_row)
                rev = reverse_row(data_row)
                if rev is not None:
                    offers.append(rev)
        except:
            pass
    print(len(offers))
    offer = pd.DataFrame(offers)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    x_raw = offer.drop(["result"], 1)
    y = offer["result"]
    scaler = preprocessing.MaxAbsScaler()
    x = scaler.fit_transform(x_raw)
    clf = LogisticRegressionCV(cv=5,
                               random_state=0,
                               max_iter=1000,
                               solver='liblinear',
                               penalty='l2',
                               refit=True,
                               scoring='roc_auc').fit(x, y)
    print(clf.score(x, y))
    dump(scaler, open('scaler.pkl', 'wb'))
    weights = np.asarray(clf.coef_)
    for i in range(0, len(weights[0])):
        print(str(weights[0][i]) + "\t" + fields.get_fields()[i])
    dump(weights, open('weights.pkl', 'wb'))
