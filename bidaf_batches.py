#BIDAF Implementation
from __future__ import division
import numpy as np
import dynet as dy
from time import time
import json
    
class BiDAF():
    def __init__(self, pc, word_emb_dim, hidden_dim, load_model=False , num_batch):
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.num_batch = num_batch
        #self.lookup_table = pc.add_lookup_parameters((100000, word_emb_dim)) # or use pretrained glove embeddings
        if not load_model:
            self.W_ss_ = pc.add_parameters((1, 3*hidden_dim))
            self.b_ss_ = pc.add_parameters((1))
            self.W_p1_ = pc.add_parameters((1, 5*hidden_dim))
            self.b_p1_ = pc.add_parameters((1))
            self.W_p2_ = pc.add_parameters((1, 5*hidden_dim))
            self.b_p2_ = pc.add_parameters((1))

            self.contextLSTM = dy.VanillaLSTMBuilder(1, word_emb_dim, hidden_dim, pc)
            self.queryLSTM = dy.VanillaLSTMBuilder(1, word_emb_dim, hidden_dim, pc)
            self.modellingLSTM = dy.VanillaLSTMBuilder(1, 4 * hidden_dim, hidden_dim, pc) #TODO
            self.outputLSTM = dy.VanillaLSTMBuilder(1,hidden_dim,hidden_dim, pc)

        else:
            self.W_ss_, self.b_ss_, self.W_p1_, self.b_p1_, self.W_p2_, self.b_p2_, self.contextLSTM, self.queryLSTM, self.modellingLSTM, self.outputLSTM = dy.load('model9',pc)


    def similarity_score(self,h,u):
        concat = dy.concatenate([h,u,dy.cmult(h,u)],d=0)
        score = self.W_ss * concat + self.b_ss

        return score.scalar_value()

    def c2q_attention(self, sim_matrix, query_states):
        attention_vector = dy.softmax(sim_matrix) # K * 
        #@TODO
        c2q = [dy.esum([b * dy.select_cols(attention_vector,[i])[j]  for j,b in enumerate(query_states)]) for i in range(self.T)]

        return c2q

    def q2c_attention(self, sim_matrix, context_states):
        attention_vector = dy.softmax(dy.max_dim(sim_matrix,d=1))
        weighted_vectors = [b * a for a,b in zip(attention_vector, context_states)]
        #@TODO
        return [dy.esum(weighted_vectors) for _ in range(self.T)]

    def similarity_matrix(self, context_states, query_states):
        sim_matrix = np.zeros((self.num_batch , self.J,self.T))
        for k in self.num_batch :
            for i in range(len(query_states)):
                for j in range(len(context_states)):
                    sim_matrix[k][i][j] = self.similarity_score(context_states[j][k],query_states[i][k])

        return dy.inputTensor(sim_matrix)

    def span_scores(self,combined_input1, combined_input2):
        
        s1 = [self.W_p1*combined_input1[i] + self.b_p1 for i in range(self.T)]
        s2 = [self.W_p2*combined_input2[i] + self.b_p2 for i in range(self.T)]

        p1 = np.zeros(self.T)
        p2 = np.zeros(self.T)
        
        for i in range(self.T):
            p1[i] = s1[i].scalar_value()
            p2[i] = s2[i].scalar_value()
        return p1, p2


    def complete_forward_pass(self, inputs):
        dy.renew_cg()
        #context_embs = inputs[0] 
        #query_embs = inputs[1]
    
        #inputs0 --> 32 * d
        #inputs1 -- 32 * q
        docs = input[0] 
        ques = input[1]
        query_embs = []   #J * 32 * 100
        context_embs = [] #T * 32 * 100 

       
        for b in range(self.num_batch) :  #32
            q_embs = []
            c_embs = []
            for word in docs[b] : #J
                try:
                    q_embs.append(dy.inputTensor(word2vec[word.lower()]))
                except:
                    q_embs.append(dy.random_normal(100))
            
            
            for word in ques[b]: #T
                try:
                    c_embs.append(dy.inputTensor(word2vec[word.lower()]))
                except:
                    c_embs.append(dy.random_normal(100))
   
            query_embs.append(q_embs)   #32* J * 100
            context_embs.append(c_embs) #32 * T * 100
            
        #list -> array -> reshape -> list    
        
        query_embs = list(np.reshape(np.array(query_embs),(len(ques) ,self.num_batch ,len(ques[0] ))))
        context_embs = list(np.reshape(np.array(query_embs),(len(docs) ,self.num_batch ,len(docs[0] ))))
         
        self.T = len(context_embs) #length of doc
        self.J = len(query_embs) # length of query

        self.W_ss = dy.parameter(self.W_ss_)
        self.b_ss = dy.parameter(self.b_ss_)
        self.W_p1 = dy.parameter(self.W_p1_)
        self.b_p1 = dy.parameter(self.b_p1_)
        self.W_p2 = dy.parameter(self.W_p2_)
        self.b_p2 = dy.parameter(self.b_p2_)


        contextLSTM_init = self.contextLSTM.initial_state()
        queryLSTM_init = self.queryLSTM.initial_state()
        modellingLSTM_init = self.modellingLSTM.initial_state()
        outputLSTM_init = self.outputLSTM.initial_state()

        #context_states = contextLSTM_init.transduce(context_embs)
        #query_states = queryLSTM_init.transduce(query_embs)
        
        cstate = contextLSTM_init.add_input(context_embs[0]) #32 * 100 for word 1
        qstate = queryLSTM_init.add_input(query_embs[0])
    
        ##context LSTM  -->  T * 32 *  100
        for i in range(1 , self.T) :
            
            context_states.append(cstate.output())  # --> output at timestep i 
            cstate = cstate.add_input(context_embs[i])
            
        ### query LSTM    -->  J * 32 *  100
        for i in range(1 , self.J) :
            
            query_states.append(qstate.output())  # --> output at timestep i 
            qstate = qstate.add_input(query_embs[i])                
            

        sim_matrix = self.similarity_matrix(context_states, query_states)
        
        ######
        c2q = self.c2q_attention(sim_matrix, query_states) ## T  * 32 * 100
        q2c = self.q2c_attention(sim_matrix, context_states) ###T * 32 * 100
        #####
        
        
        #modelling_input = [dy.concatenate([context_states[i],c2q[i],
        #                   dy.cmult(context_states[i],c2q[i]),
        #                   dy.cmult(context_states[i],q2c[i])], d=0) for i in range(self.T)]
        
        #@TODO - better way 
        for j in range(self.T) :
            cstate = cotext_states[j]
            cq = c2q[j]
            qc = q2c[j]
            minput = [dy.concatenate([cstate[i],cq[i],
                           dy.cmult(cstate[i],cq[i]),
                           dy.cmult(cstate[i],qc[i])], d=0) for i in range(self.numbatch)]
                
            modelling_input.append(minput)   ### T * 32 * 100   
    
    

        m_output1 = modellingLSTM_init.add_input(modelling_input[0]) # 32* 100
               
        for i in range(self.T) :
            modelling_output1.append(m_output1.output())  # --> output at timestep i 
            m_output1  = m_output1.add_input(modelling_input[i]) 
        
        m_output2 = outputLSTM_init.add_input(modelling_output1[0])
        
        for i in range(self.T) :
            modelling_output2.append(m_output2.output())  # --> output at timestep i 
            m_output2  = m_output2.add_input(modelling_input[i]) 
        
        ###############       
        combined_input1 = [dy.concatenate([modelling_input[:,i],modelling_output1[:,i]], d=0) for i in range(self.T)]
        combined_input2 = [dy.concatenate([modelling_input[:,i],modelling_output2[:,i]], d=0) for i in range(self.T)]

        ##################
        p1, p2 = self.span_scores(combined_input1, combined_input2)

        return p1, p2

def loss_fn(p1, p2, gold_answer):
    ###@TODO  - masking  
    loss = dy.pickneglogsoftmax_batch(dy.inputTensor(p1), gold_answer[:,0]) + dy.pickneglogsoftmax(dy.inputTensor(p2), gold_answer[:,1])
    return loss

def predict_fn(model, shared_file, data_file, train=True):
    word2vec = shared_file['lower_word2vec']
    Questions = data_file['q']
    passage_id = data_file['*x']
    Passages = shared_file['x']
    if train:
        Answers = np.array(data_file['y'])[:,0,:,0]
    else:
        Answers = [np.array(a[0])[:,0] for a in data_file['y']]
    
    num_correct = 0

    for k in range(len(Answers)):
        answer = Answers[k]
        doc = Passages[passage_id[k][0]][passage_id[k][1]]      
        p1, p2 = model.complete_forward_pass((doc, Questions[k]))

        predict_start = np.argmax(p1)
        predict_end = np.argmax(p2)

        if predict_start == answer[0] and predict_end == answer[1]:
            num_correct += 1

    accuracy = num_correct/len(Answers) * 100
    return accuracy

def main():

    shared_train = json.load(open('shared_train.json'))
    data_train = json.load(open('data_train.json')) 

    shared_dev = json.load(open('shared_dev.json'))
    data_dev = json.load(open('data_dev.json')) 

    word2vec = shared_train['lower_word2vec']
    Questions = data_train['q']
    passage_id = data_train['*x']
    Passages = shared_train['x']
    Answers = np.array(data_train['y'])[:,0,:,0]  #answer spans
    pc = dy.Model()
    trainer = dy.AdamTrainer(pc)
    num_batch = len(Answers)
    
    model = BiDAF(pc,100,50, load_model = False, num_batch)

    for epoch in range(10):
        train_loss = 0
        for k in range(len(Answers)): #number of batches
            
            answer = np.array(Answers[k]) #32*2
            #doc = Passages[passage_id[k][0]][passage_id[k][1]] 
            doc = Passages[k] #32 * 
            
            
            p1, p2 = model.complete_forward_pass((doc, Questions[k]))
            loss = loss_fn(p1, p2, answer)
            train_loss += loss.scalar_value()
            loss.backward()
            trainer.update()
        print("Epoch: {} || Training Loss: {}".format(epoch+1,train_loss/len(Answers))) #len(Answers)
        
        train_accuracy = predict_fn(model, shared_train, data_train)
        print("Epoch: {} || Training accuracy: {}".format(epoch+1, train_accuracy))

        val_accuracy = predict_fn(model, shared_dev, data_dev, train=False)
        print("Epoch: {} || Validation accuracy: {}".format(epoch+1, val_accuracy))

        #saving the model at every epoch
        dy.save('model' + str(epoch), [model.W_ss_, model.b_ss_, model.W_p1_, model.b_p1_, model.W_p2_, model.b_p2_,
                                       model.contextLSTM, model.queryLSTM, model.modellingLSTM, model.outputLSTM])

if __name__ == '__main__':
    main()







