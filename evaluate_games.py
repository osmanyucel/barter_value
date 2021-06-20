import sqlite3

import fields
import functions
from pickle import load
import numpy as np
from pandas import DataFrame


def numericize(s, field_list):
    res = list()
    for ind in range(0, len(s)):
        value = float(s[ind])
        if field_list[ind] in ["bundles_packages", "bundles_all", "giveaway_count"]:
            value = functions.get_sigmoid(value)
        res.append(value)
    return res


if __name__ == '__main__':
    conn = sqlite3.connect("barter.sqlite")
    scaler = load(open('scaler.pkl', 'rb'))
    sql = "select id," + ','.join(fields.get_fields()) + " from items"
    print(sql)
    cur = conn.execute(sql)
    weights = load(open('weights.pkl', 'rb'))
    for i in range(len(weights[0])):
        print("%s | %.6f" % (fields.get_fields()[i], weights[0][i]))
    for row in cur:
        try:
            game_id = row[0]
            vals = np.asarray(numericize(row[1:], fields.get_fields()))
            sv = scaler.transform(DataFrame(vals).transpose())
            val = np.dot(sv, weights.transpose())[0][0]
            sql = "update items set barter_value = " + str(val) + " where id = " + str(game_id)
            conn.execute(sql)
        except:  # We skip the games with missing values
            pass
    conn.commit()
