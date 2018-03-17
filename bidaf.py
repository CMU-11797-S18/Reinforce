#BIDAF Implementation
import dynet as dy, numpy as np
from time import time
	

class BiDAF():
	def __init__(self, pc, word_emb_dim, hidden_dim):
		self.pc = pc
		self.word_emb_dim = word_emb_dim
		self.hidden_dim = hidden_dim
		
		self.lookup_table = pc.add_lookup_parameters((100000, word_emb_dim)) # or use pretrained glove embeddings
		self.W_ss_ = dy.add_parameters((1, 3*hidden_dim))
		self.b_ss_ = dy.add_parameters((1))

		self.contextLSTM = dy.VannillaLSTMBuilder(1, word_emb_dim, hidden_dim, pc)
		self.queryLSTM = dy.VannillaLSTMBuilder(1, word_emb_dim, hidden_dim, pc)
		self.modellingLSTM = dy.VannillaLSTMBuilder(2, hidden_dim + some_dim + some_dim, hidden_dim, pc) #TODO
		self.outputLSTM = dy.VannillaLSTMBuilder(1,,hidden_dim, pc)

	def similarity_score(self,h,u):
		concat = dy.concatenate([h,u,dy.cmult(h,u)],d=0)
		score = self.W_ss * concat + self.b_ss

		return score

	def c2q_attention(self, sim_matrix, query_states):
		attention_vector = dy.softmax(sim_matrix)
		c2q = [dy.esum([a * b for a,b in zip(attention_vector[:,i], query_states)]) for i in range(self.T)]

		return c2q

	def q2c_attention(self, sim_matrix, context_states):
		attention_vector = dy.softmax(dy.max_dim(sim_matrix,d=1))
		weighted_vectors = [a * b for a,b in zip(attention_vector, context_states)]

		return [dy.esum(weighted_vectors) for _ in range(self.T)]

	def similarity_matrix(self, context_states, query_states):
		sim_matrix = dy.zeros((self.J,self.T))
		for i in range(len(query_states)):
			for j in range(len(context_states)):
				sim_matrix[i][j] = self.similarity_score(context_states[j],query_states[i])

		return sim_matrix

	def span_scores(self,combined_input1, combined_input2):
		p1 = dy.softmax(self.W_p1*combined_input1 + self.b_p1)
		p2 = dy.softmax(self.W_p2*combined_input2 + self.b_p2)

		return p1, p2


	def complete_forward_pass(self, inputs):
		dy.renew_cg()
		context = inputs[0]
		query = inputs[1]

		self.T = len(context)
		self.J = len(query)

		self.W_ss = dy.parameter(self.W_ss_)
		self.b_ss = dy.parameter(self.b_ss_)
		self.W_p1 = dy.parameter(self.W_p1_)
		self.b_p1 = dy.parameter(self.b_p1_)
		self.W_p2 = dy.parameter(self.W_p2_)
		self.b_p2 = dy.parameter(self.b_p2_)

		contextLSTM_init = self.contextLSTM.initial_state()
		queryLSTM_init = self.queryLSTM.initial_state()
		modellingLSTM_init = self.contextLSTM.initial_state()
		outputLSTM_init = self.queryLSTM.initial_state()

		context_embs = [dy.lookup[self.lookup_table,x] for x in context]
		query_embs = [dy.lookup[self.lookup_table,x] for x in query]

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

def main():

	# some data reader function
	
	pc = dy.Model()
	trainer = dy.AdamTrainer(pc)

	model = BiDAF(pc,300,150)

	for k in range(num_examples):
		question = Questions[k]
		passage = Passages[k]
		answer = Answers[k]

		p1, p2 = model.complete_forward_pass((passage, question))
		loss = loss_fn(p1, p2, answer)
		loss.backward()
		trainer.update()

	accuracy = predict('train')








