from multiprocessing import Process
import multiprocessing as mp
import os, sys
from sentences import *
import numpy as np
import pickle
from ebm_train_clique import *
import word_definite as WD
from word_definite import *

class word_definite:
    def __init__(self,derived, lemma, cng, pos, chunk_id):
        self.lemma = lemma
        self.derived = derived
        self.cng = str(cng)
        self.tup = "{}_{}".format(self.lemma, self.cng)
        # self.form = form
        self.pos = pos
        self.chunk_id = chunk_id        
        # Fields Required for heap
        self.dist = np.inf
        self.src = -1
        self.id = -1
        self.isConflicted = False
    def __str__(self):
        return 'WD_Node[C: %d, P: %d, %s @(%s) => %s]' %(self.chunk_id, self.pos, self.lemma, self.cng, self.derived)
    def __repr__(self):
        return str(self)

def pooled_Test(modelFile, vpid, queue, testfolder, filePerProcess = 100, _dump = False, _outFile = None):
    #print("ebm_TestPool_Unit_clique")
    n_chkpt = 100
    print('Child process with vpid:{}, pid:{} started.'.format(vpid, os.getpid()))
    trainer = Trainer()
    trainer.Load(modelFile)

    TestFiles = []
    for f in os.listdir(testfolder):
        if '.p' in f:
            TestFiles.append(f)
    #print()
    print('TestFIles : ',TestFiles)
    print('vpid:{}: Range is {} -> {} / {}'.format(vpid, vpid*filePerProcess, vpid*filePerProcess + filePerProcess, len(TestFiles)))
    if _dump:
        _outFile = '{}_proc{}.csv'.format(_outFile, vpid)
        with open(_outFile, 'w') as fh:
            print('File refreshed', _outFile)
            
    #loaded_SKT = pickle.load(open('Simultaneous_CompatSKT_ho.p', 'rb'))
    #loaded_DCS = pickle.load(open('Simultaneous_DCS_ho.p', 'rb'))
    #print("SKT",loaded_SKT)
    #print("DCS",loaded_DCS)
    #exit(1)
    
    #loader = pickle.load(open('../bz2Dataset_10K.p', 'rb'))
    #TestFiles = loader['TestFiles']
    #TrainFiles = loader['TrainFiles']

    for i in range(vpid*filePerProcess, vpid*filePerProcess + filePerProcess):
        #if i % n_chkpt == 0:
            #print('Checkpoint {}, vpid: {}'.format(i, vpid))
            #sys.stdout.flush()
        try:
            fn = TestFiles[i]
        except:
            break
        #fn = fn.replace('.ds.bz2', '.p2')

        p_name = testfolder + TestFiles[i]
        print(p_name)
        loaded_p = pickle.load(open(p_name, 'rb'))
        
        #nodelist, conflicts_Dict, featVMat
        nodelist = loaded_p["nodelist"]
        conflicts_Dict = loaded_p["conflicts_Dict"]
        featVMat = loaded_p["featVMat"]
        #EBM(self, nodelist, conflicts_Dict, featVMat, _outFile = None)
        try:
            if _dump:
                results = trainer.EBM(nodelist, conflicts_Dict, featVMat, _outFile = _outFile)
            else:
                results = trainer.EBM(nodelist, conflicts_Dict, featVMat)
        except EOFError as e:
            print('BADFILE', p_name)

        if results is not None:
                queue.put(results)

        print(" results : ",results)

    print('Child process with vpid:{}, pid:{} closed.'.format(vpid, os.getpid()))

