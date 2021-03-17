import sys, os, time, bz2, zlib, pickle, math, json, csv
from word_definite import *

def open_pickle(filename):
    
    loaded_p = pickle.load(open(filename, 'rb'))


    nodelist = loaded_p["nodelist"]
    conflicts_Dict = loaded_p["conflicts_Dict"]
    featVMat = loaded_p["featVMat"]
    
    return (nodelist, conflicts_Dict, featVMat)

#print(open_pickle("input/0.p"))