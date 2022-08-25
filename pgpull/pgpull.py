#!/usr/bin/python
# -*- coding: utf-8 -*-"""
"""
This script applies a predictive model to pull prospect data from a postgres database.
It has been generalized from its original use to be  utility/example of:
    passing arguments to a py program
    connecting to a postgres database (deserves a rewrite to use classes)
    processing a table to large for client memory, done in chunks
    dynamically generating SQL from 
    using joblib to serialize models
    deserves to have logging implemented instead of print statements

In addition to arguments, the script expects
    the list of column/variable names, split into qualitatives and quantitatives
    the name of the source table, see REPLACE_TABLE_NAME_HERE occurrences
    a file 'predict.pkl' that was saved by the joblib tool

Then:

:param str   servername:  host for database connection
:param str   portnumber:  port number on host for database connection
:param str   dbname:      name of the database running on host
:param str   username:    database users name
:param str   userpass:    database users password
:param str   chunksize:   number of rows to process in each chunk
:param float threshold:   proportion over which to accept a prediction probability
:param str   mode:        'test' will limit the run to 5 chunks

:return: file named 'predictions.csv'

"""
from datetime import datetime
print()
print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),"Importing libraries")
print()
# Load the basic python modules (should not need to installation)
import os
import sys
import importlib
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
import psutil
process = psutil.Process(os.getpid())
import collections
import numpy as np

# Load modules that may need installation first
try:
    from winreg import QueryInfoKey
except:
    os.command("pip install winreg")
    from winreg import QueryInfoKey
try:
    import joblib
except:
    os.command("pip install joblib")
    import joblib
try:
    import pandas as pd
except:
    os.command("pip install pandas")
    import pandas as pd
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
except:
    os.command("pip install -U scikit-learn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
try:
    import psycopg2
except:
    os.command("pip install psycopg2")
    import psycopg2

# Process the command line arguments principally for the postgres database connection.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v","--verbosity", help="increase output verbosity")
parser.add_argument("-s", "--servername", 
                    type=str,
                    required=True,
                    help="The hostname or IP address of the database server",
                    action="store")
parser.add_argument("-n", "--portnumber", 
                    type=str,
                    required=True,
                    help="The port number on the database server",
                    action="store")
parser.add_argument("-d", "--dbname", 
                    type=str,
                    required=True,
                    help="The name of the database on the database server",
                    action="store")
parser.add_argument("-u", "--username", 
                    type=str,
                    required=True,
                    help="The user account that has privileges to access the DB",
                    action="store")
parser.add_argument("-p", "--userpass", 
                    type=str,
                    required=True,
                    help="The privileged user's password",
                    action="store")
parser.add_argument("-c", "--chunksize", 
                    type=int,
                    required=True,
                    help="The number of rows in a chunk",
                    action="store")
parser.add_argument("-t", "--threshold", 
                    type=float,
                    required=True,
                    help="The threshold proportion for accepting a prediction",
                    action="store")
parser.add_argument("-m", "--mode", 
                    type=str,
                    help="Whether this run generates a data file or a test count distribution.",
                    action="store")
args = parser.parse_args()

# confirm the parameters
print("")
print("Running predictions with:")
print("        username = '",args.username,"'",sep="")
print("        password = '",args.userpass,"'",sep="")
print("        database = '",args.dbname,"'",sep="")
print("      servername = '",args.servername,"'",sep="")
print("      portnumber = '",args.portnumber,"'",sep="")
print("       threshold = ",args.threshold,"",sep="")
print("       chunksize = ",args.chunksize,"",sep="")
print("            mode = '",args.mode,"'",sep="")

# test the database connection
try:
    con = psycopg2.connect(host=args.servername, port=args.portnumber, database=args.dbname, user=args.username, password=args.userpass)
    cur = con.cursor()
    cur.execute('SELECT version()')
    result = cur.fetchone()[0]
except OperationalError as e:
    print("Database connection failed:", e)
    sys.exit(1)

def dbfetch(query):
    """
    Connects to the requested database, executes a SQL query and returns the result.

    :param:  str query:  query string of SQL SELECT statement
    :return: dataframe:  pandas Dataframe with rows selected by query
    """
    con = None
    try:
        con = psycopg2.connect(host=args.servername, port=args.portnumber, database=args.dbname, user=args.username, password=args.userpass)
        result = pd.read_sql_query(query,con)
    except Exception as e:  #psycopg2.DatabaseError as e
        print(f'Error {e}')
        sys.exit(1)
    finally:
        if con:
            con.close()
    return result

def chunkpredict(prospects):
    """
    Applies the model to generate a prediction on each row of the requested chunk.

    :param:  dataframe chunk:  query string of SQL SELECT statement
    :return: list:  predictions for each row in the passed dataframe
    """
    """
    import random
        y_pred = []
    for i in range(chunk.shape[0]):
        y_pred.append(random.uniform(0, 1))
    mute =
    """
    # save the quant columns

    master = prospects[quantfields]
    for c in master.columns:
        master[c] = master[c].replace('U', 0)
    dums = pd.get_dummies(prospects[qualfields])
    # for some weird reason, get_dummies can generate duplicate colnames that have diff meanings
    cols=pd.Series(dums.columns)
    for dup in dums.columns[dums.columns.duplicated(keep=False)]: 
        cols[dums.columns.get_loc(dup)] = ([dup + '.' + str(d_idx) if d_idx != 0 else dup 
                                        for d_idx in range(dums.columns.get_loc(dup).sum())])
    dums.columns=cols
    # join the quant fields with dummy one-hot encoded fields
    master = pd.concat([master, dums], axis=1)
    master = master.fillna(0)
    print("checkpoint 1",master.isna().sum().sum())
    for c in master.columns:
        master[c] = master[c].astype(float, errors = 'raise')
        master[c] = (master[c] - master[c].mean())/master[c].std(ddof=0)
    print("checkpoint 2",master.isna().sum().sum())
    for c in modelvars:
        if c not in master.columns:
            print("adding a missing, new empty column "+c)
            master[c] = 0
    master = master[modelvars]
    master = master.fillna(0)
    print("Ready to predict")
    print()
    master.isna()
    y_pred = model.predict(master)
    return y_pred

if __name__ == "__main__":
    print()
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),"Starting main process")
    print()
    model = joblib.load('predictor.pkl')
    # initialize vars, expected columns in the target table
    #
    # REPLACE the a,b,c,d,e,f with your own variable names
    #
    modelvars = ['a','b','c','d','e','f']
    quantfields = ['d','e','f']
    querycols = qualfields + quantfields
    query ="SELECT "+", ".join(querycols)+" FROM {TABLE NAME HERE}"
    chunk = 0
    predrows = 0    
    rowsintable = dbfetch("SELECT COUNT(*) FROM {TABLE NAME HERE}")["count"][0]
    print()
    print(
        "Date & Time".ljust(20),
        "Chunk/Chunks".ljust(20),
        "Chunk Dur".ljust(12),
        "Pred End".ljust(12),
        "Pred Rate".ljust(12),
        "Pred Filesize".ljust(14),
        "Mem%".ljust(6))
    print("-"*20,"-"*20,"-"*12,"-"*12,"-"*12,"-"*14,"-"*6)
    jobstart = datetime.now()

    # loop over chunks
    while chunk*args.chunksize < rowsintable:
        chunk += 1
        if args.mode == 'test' and chunk == 5:
            break
        chunkstart = datetime.now()
        s = query
        s += " LIMIT " + str(args.chunksize)
        s += " OFFSET " + str((chunk-1) * args.chunksize)
        # get the next chunk
        trying = 0
        while trying < 3:
            trying +=1
            try:
                chunkfetch = dbfetch(s)
                break
            except:
                print("Database fetch for chunk",chunk,"failed. Retrying")
        if trying == 3:
            print("Chunk",chunk,"database fetch failed 3 times, skipping")
            continue
        # process it: standardize and predict
        chunkfetch["prediction"] = chunkpredict(chunkfetch)
        # write predicted best rows to file
        if chunk == 1 :
            chunkfetch.truncate().to_csv("prospects.csv",index=False, header=True, mode='w+')
        chunkfetch[chunkfetch.prediction > args.threshold].to_csv('prospects.csv', mode='a', index=False, header=False)
        # print the summary rows for this chunk
        chunkdur= str(datetime.now() - chunkstart)
        pctdone = min((chunk*args.chunksize/rowsintable),1.0)
        predtime = str(chunkstart + (datetime.now() - chunkstart)/pctdone)
        print(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S").ljust(20),
            (str(chunk)+"/"+str(rowsintable)).ljust(20),
            chunkdur[chunkdur.find(":")+1:chunkdur.find(".")].ljust(12),
            predtime[predtime.find(" ")+1:predtime.find(".")].ljust(12),
            "{:.00%}".format(chunkfetch[chunkfetch.prediction > args.threshold].shape[0]/chunkfetch.shape[0]).ljust(12),
            str(sum(chunkfetch.prediction > args.threshold)).ljust(12),
            "{:.{}f}".format(os.path.getsize('prospects.csv'),2), #/(1024**3)
            "{:.00%}".format(psutil.virtual_memory().available/psutil.virtual_memory().total).ljust(6)  
        )
        # capture the distribution of this chunk predictions
        chunkfetch["cat"] = \
            np.where(chunkfetch['prediction']>.95, '95', np.where(chunkfetch['prediction']>.90, '90',
            np.where(chunkfetch['prediction']>.85, '85', np.where(chunkfetch['prediction']>.80, '80',
            np.where(chunkfetch['prediction']>.75, '75', np.where(chunkfetch['prediction']>.70, '70',
            np.where(chunkfetch['prediction']>.65, '65', np.where(chunkfetch['prediction']>.60, '60',
            np.where(chunkfetch['prediction']>.55, '55', np.where(chunkfetch['prediction']>.50, '50',
            np.where(chunkfetch['prediction']>.45, '45', np.where(chunkfetch['prediction']>.40, '40',
            np.where(chunkfetch['prediction']>.35, '35', np.where(chunkfetch['prediction']>.30, '30',
            np.where(chunkfetch['prediction']>.25, '25', np.where(chunkfetch['prediction']>.20, '20',
            np.where(chunkfetch['prediction']>.15, '15', np.where(chunkfetch['prediction']>.10, '10',
            np.where(chunkfetch['prediction']>.05, '05', '00')))))))))))))))))))
        if chunk == 1:
            distribution = pd.DataFrame(chunkfetch.cat.value_counts()).sort_index(axis=0)
            distribution = distribution.rename(columns={"cat": "chunk 1"})
        else:
            chunkdist =    pd.DataFrame(chunkfetch.cat.value_counts()).sort_index(axis=0)
            chunkdist =    chunkdist.rename(columns={"cat": "chunk"+str(chunk)})
            distribution = pd.concat([distribution, chunkdist], axis=1, join='outer', ignore_index=False)
    print()
    print("Prediction complete")
    print()
    print("Distribution")
    distribution['Total'] = distribution.sum(axis=1)
    print(distribution.to_string())
