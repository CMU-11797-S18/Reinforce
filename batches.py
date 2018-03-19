#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:43:31 2018

@author: eti
"""

from collections import defaultdict
from itertools import repeat
import json
import numpy as np

def create_batches( data , shared , batch_size ) :
    
    ddata = defaultdict(list)
    dshared = defaultdict(list)
    questions = data['q']
    rx = data['*x']
    y = np.array(data['y'])[:,0,:,0]#data['y']
    ans = data['answerss']
    ids = data['ids'] 
    idxs= data['idxs']
    para = shared['x']
    #@TODO
    ###change y 
     
    
    num_batches = len(questions) // batch_size
    ##sort questions and rx and y 
    questions , rx  ,y , ans , ids , idxs =  \
    zip(*sorted(zip(questions , rx , y , ans , ids , idxs ) , key = lambda x : len(x[0]))) 
    
    ##divide questions into batches and pad
    for i in range(num_batches) :
          #####        
          q1 = questions[i*batch_size : (i +1)*batch_size]
          rx1 = rx[i*batch_size : (i +1)*batch_size]
          y1 = y[i*batch_size : (i +1)*batch_size]
          a1 = ans[i*batch_size : (i +1)*batch_size]
          id1 = ids[i*batch_size : (i +1)*batch_size]
          idx = idxs[i*batch_size : (i +1)*batch_size]
          ###pad q1
          max_len = len(q1[-1])
          q1 = [q+list(repeat(0, max_len -len(q))) for q in q1 ]
          
          #adding data
          ddata['q'].append(q1)
          ddata['rx'].append(rx1)
          ddata['y'].append(y1)
          ddata['answerss'].append(a1)
          ddata['ids'].append(id1)
          ddata['idxs'].append(idx)               

          #make a list of all the required paras
          p1  = [para[i][j] for i,j in rx1]
          #padd paras
          max_para = len(max(p1, key=len)) #len(p1[-1])
          p1 = [p+list(repeat(0, max_para -len(p))) for p in p1 ]
          
          dshared['p'].append(p1)
     

    dshared['word2vec'] = shared['word2vec']
    dshared['lower_word2vec'] = shared['lower_word2vec']
    ##save as json
    json.dump(ddata, open('data_batches', 'w'))
    json.dump(dshared, open('shared_batches', 'w'))
    
    
    



if __name__ == "__main__":
    
    data = json.load(open('/home/eti/Desktop/Reinforce/bi-att-flow/data/squad/data_train.json' , 'r'))
    shared = json.load(open('/home/eti/Desktop/Reinforce/bi-att-flow/data/squad/shared_train.json' , 'r'))
    create_batches( data , shared , 60)
      