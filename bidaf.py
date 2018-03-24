#BIDAF Implementation
#BIDAF Implementation
from __future__ import division
import numpy as np
import dynet as dy
from time import time
import json
    
class BiDAF():
    def __init__(self, pc, word_emb_dim, hidden_dim, load_model=False):
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim

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
            self.modellingLSTM = dy.VanillaLSTMBuilder(2, 4 * hidden_dim, hidden_dim, pc) #TODO
            self.outputLSTM = dy.VanillaLSTMBuilder(1,hidden_dim,hidden_dim, pc)

        else:
            self.W_ss_, self.b_ss_, self.W_p1_, self.b_p1_, self.W_p2_, self.b_p2_, self.contextLSTM, self.queryLSTM, self.modellingLSTM, self.outputLSTM = dy.load('model9',pc)


    def similarity_score(self,h,u):
        concat = dy.concatenate([h,u,dy.cmult(h,u)],d=0)
        score = self.W_ss * concat + self.b_ss

        return score

    def c2q_attention(self, sim_matrix, query_states):
        attention_vector = dy.softmax(sim_matrix)
        c2q = [dy.esum([b * dy.select_cols(attention_vector,[i])[j]  for j,b in enumerate(query_states)]) for i in range(self.T)]

        return c2q

    def q2c_attention(self, sim_matrix, context_states):
        attention_vector = dy.softmax(dy.max_dim(sim_matrix))
        weighted_vectors = [b * a for a,b in zip(attention_vector, context_states)]

        return [dy.esum(weighted_vectors) for _ in range(self.T)]

    def similarity_matrix(self, context_states, query_states):
        rows = [None] * self.J
        for i in range(self.J):
            cols = [None] * self.T
            for j in range(self.T):
                cols[j] = self.similarity_score(context_states[j],query_states[i])
            rows[i] = dy.concatenate_cols(cols)
        sim_matrix = dy.concatenate(rows)
        
        return sim_matrix

    def span_scores(self,combined_input1, combined_input2):
        s1 = [self.W_p1*combined_input1[i] + self.b_p1 for i in range(self.T)]
        s2 = [self.W_p2*combined_input2[i] + self.b_p2 for i in range(self.T)]
#         p2 = self.W_p2*dy.inputTensor(combined_input2) + self.b_p2

        p1 = dy.concatenate(s1)
        p2 = dy.concatenate(s2)
        
        return p1, p2


    def complete_forward_pass(self, inputs, word2vec):
        dy.renew_cg()
        context_embs = inputs[0]
        query_embs = inputs[1]

        query_embs = []
        context_embs = []
        
        kw = 0
        uw = 0
        for word in inputs[1]:
            try:
                query_embs.append(dy.inputTensor(word2vec[word.lower()]))
            except:
                query_embs.append(dy.random_normal(100))

        for word in inputs[0]:
            try:
                context_embs.append(dy.inputTensor(word2vec[word.lower()]))
            except:
                context_embs.append(dy.random_normal(100))
                
        self.T = len(context_embs)
        self.J = len(query_embs)

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

        #context_embs = [dy.lookup[self.lookup_table,x] for x in context]
        #query_embs = [dy.lookup[self.lookup_table,x] for x in query]

        context_states = contextLSTM_init.transduce(context_embs)
        query_states = queryLSTM_init.transduce(query_embs)

        sim_matrix = self.similarity_matrix(context_states, query_states)
        c2q = self.c2q_attention(sim_matrix, query_states)
        q2c = self.q2c_attention(sim_matrix, context_states)

        modelling_input = [dy.concatenate([context_states[i],c2q[i],
                           dy.cmult(context_states[i],c2q[i]),
                           dy.cmult(context_states[i],q2c[i])], d=0) for i in range(self.T)]
        
        modelling_output1 = modellingLSTM_init.transduce(modelling_input)
        modelling_output2 = outputLSTM_init.transduce(modelling_output1)

        combined_input1 = [dy.concatenate([modelling_input[i],modelling_output1[i]], d=0) for i in range(self.T)]
        combined_input2 = [dy.concatenate([modelling_input[i],modelling_output2[i]], d=0) for i in range(self.T)]

        p1, p2 = self.span_scores(combined_input1, combined_input2)

        return p1, p2

def loss_fn(p1, p2, gold_answer):
    loss = dy.pickneglogsoftmax(p1, gold_answer[0]) + dy.pickneglogsoftmax(p2, gold_answer[1])
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
        p1, p2 = model.complete_forward_pass((doc, Questions[k]), word2vec)

        predict_start = np.argmax(p1.npvalue())
        predict_end = np.argmax(p2.npvalue())

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

    model = BiDAF(pc,100,50, load_model = False)

    for epoch in range(10):
        train_loss = 0
        for k in np.random.permutation(len(Answers)): #len(Answers)

            answer = Answers[k]
            doc = Passages[passage_id[k][0]][passage_id[k][1]]
            p1, p2 = model.complete_forward_pass((doc, Questions[k]), word2vec)
    #         print(answer)
    #         print(np.argmax(p1),np.argmax(p2))
    #         time.sleep(2)
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








