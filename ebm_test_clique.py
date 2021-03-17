import multiprocessing as mp
import ebm_TestPool_Unit_clique
import csv
from shutil import copyfile
import numpy as np
import time
import sys
from optparse import OptionParser

from collections import defaultdict
rwfinal=0
pwfinal=0
rlfinal=0
plfinal=0
fnum=0

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

def Evaluate(result_arr):
    print('Files Processed: ', len(result_arr))
    recalls = []
    recalls_of_word = []
    precisions = []
    precisions_of_words = []
    fully_Correct_l = 0
    fully_Correct_w = 0
    for entry in result_arr:
        (word_match, lemma_match, n_dcsWords, n_output_nodes) = entry
        recalls.append(lemma_match/n_dcsWords)
        recalls_of_word.append(word_match/n_dcsWords)
        precisions.append(lemma_match/n_output_nodes)
        precisions_of_words.append(word_match/n_output_nodes)
        if lemma_match == n_dcsWords:
            fully_Correct_l += 1
        if word_match == n_dcsWords:
            fully_Correct_w += 1
    print('Avg. Micro Recall of Words: {}'.format(np.mean(np.array(recalls))))
    print('Avg. Micro Recall of Word++s: {}'.format(np.mean(np.array(recalls_of_word))))
    print('Avg. Micro Precision of Words: {}'.format(np.mean(np.array(precisions))))
    print('Avg. Micro Precision of Word++s: {}'.format(np.mean(np.array(precisions_of_words))))
    
    rl = np.mean(np.array(recalls))
    pl = np.mean(np.array(precisions))
    print('F-Score of Wordss: ', (2*pl*rl)/(pl+rl))
    print('Fully Correct Wordwise: {}'.format(fully_Correct_l/len(recalls_of_word)))
    print('Fully Correct Word++wise: {}'.format(fully_Correct_w/len(recalls_of_word)))
    print('[{:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}]'.format(100*np.mean(np.array(recalls)), 100*np.mean(np.array(recalls_of_word)), 100*np.mean(np.array(precisions)), \
           100*np.mean(np.array(precisions_of_words)), 100*(2*pl*rl)/(pl+rl), 100*fully_Correct_l/len(recalls_of_word),\
           100*fully_Correct_w/len(recalls_of_word)))
    sys.stdout.flush()
    
tag = None
proc_count = 4

def main():
    #print("ebm_test_clique")
    global proc_count, tag
    ho_folders = {
        'PR2': 'skt_dcs_DS.bz2_4K_pmi_rfe_heldout',
        'BR2': 'skt_dcs_DS.bz2_4K_bigram_rfe_heldout',
        'PM2': 'skt_dcs_DS.bz2_4K_pmi_mir_hel`dout',
        'BM2': 'skt_dcs_DS.bz2_4K_bigram_mir_heldout',
        'PR3': 'skt_dcs_DS.bz2_1L_pmi_rfe_heldout',
        'BR3': 'skt_dcs_DS.bz2_1L_bigram_rfe_heldout',
        'PM3': 'skt_dcs_DS.bz2_1L_pmi_mir_heldout2',
        'BM3': 'skt_dcs_DS.bz2_1L_bigram_heldout'
    }
    modelList = {
        'PR2': 'outputs/train_{}/nnet.p'.format('t8006684774222'),
        'BR2': 'outputs/train_{}/nnet.p'.format('t7978761528557'),
        'PM2': 'outputs/train_{}/nnet.p'.format('t7323235797178'),
        'BM2': 'outputs/train_{}/nnet.p'.format('t7978754709018'),
        'PR3': 'outputs/train_{}/nnet.p'.format('t8006711065860'),
        'BR3': 'outputs/train_{}/nnet.p'.format('t8103694133496'),
        'PM3': 'outputs/train_{}/nnet.p'.format('t8006607913382'),
        'BM3': 'outputs/train_{}/nnet.p'.format('t7274036680592')
    }
    modelFile = modelList[tag]
    print('Tag: {}, ModelFile: {}'.format(tag, modelFile))
    print('ProcCount: {}'.format(proc_count))
    _dump = True
    # if _dump:
    #     _outFile = 'outputs/{}_NLoss'.format(tag)
    # else:
    #     _outFile = None
    _outFile = 'outputs/new_NLoss'
    print('OutFile: ', _outFile)

    # Backup the model file
    copyfile(modelFile, modelFile + '.bk')

    # Create Queue, Result array
    queue = mp.Queue()
    result_arr = []

    #print('Source: ', 'input/')
    # Start 6 workers - 8 slows down the pc
    # proc_count = 4
    procs = [None]*proc_count
    for i in range(0,1):
        vpid = i
        #print("ebm_test_clique.py to ebm_TestPool_Unit_clique")
        procs[i] = mp.Process(target = ebm_TestPool_Unit_clique.pooled_Test, args = \
                              (modelFile, vpid, queue, 'input/', 100, _dump, _outFile))
    # Start Processes
    for i in range(0,1):
        procs[i].start()
        
    # Fetch partial results
    stillRunning = True
    printer_timer = 100
    while stillRunning:
        stillRunning = False
        for i in range(0,1):
            p = procs[i]
            # print('Process with\t vpid: {}\t ->\t pid: {}\t ->\t running status: {}'.format(i, p.pid, p.is_alive()))
            if p.is_alive():
                stillRunning = True
        
        
        if printer_timer == 0:
            printer_timer = 100
            while not queue.empty():
                result_arr.append(queue.get())
            # Evaluate results till now
            # if len(result_arr) > 0:
            #     Evaluate(result_arr)

        printer_timer -= 1
        
        time.sleep(1)
    while not queue.empty():
        result_arr.append(queue.get())
    #Evaluate(result_arr)
    #for i in range(proc_count):
    #    procs[i].join()

    
def setArgs(_tag, _pc):
    global proc_count, tag
    tag = _tag
    proc_count = _pc
    print('Tag, ProcCount: {}, {}'.format(tag, proc_count))
    
if __name__ == '__main__':
    

    parser = OptionParser()
    parser.add_option("-t", "--tag", dest="tag",
                      help="Tag for feature set to use", metavar="TAG")
    parser.add_option("-p", "--procs", dest="proc_count", default = 1,
                      help="Number of child process", metavar="PROCS")

    (options, args) = parser.parse_args()

    options = vars(options)
    _tag = options['tag']
    if _tag is None:
        raise Exception('None is tag')
    pc = int(options['proc_count'])
    setArgs(_tag, pc)
    
    main()