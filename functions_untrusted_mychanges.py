import tensorflow.compat.v1 as tf # change I made
tf.disable_v2_behavior() # change I made
#import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.model_selection import PredefinedSplit
import os
from os.path import join
import scipy
import matplotlib.pyplot as plt

#### ORIGINAL CODE BY ME ######
def show_differences(wordss, cock_data): #function for mini experiment described in paper
	votes = cock_data["word_ratings"]["votes"]
	round2_voters = sorted(votes[2][wordss[2][0]]["yes_votes"] + votes[2][wordss[2][0]]["no_votes"])
	round2_hashes = [{w: 1 if u in votes[2][w]["yes_votes"] else 0 for w in wordss[2]} for u in round2_voters]
	basis_voter = 11
	wordss_set = set(wordss[2])
	# select 10 random funny words
	funny_basis_words = []
	for word in wordss_set:	
		if len(funny_basis_words) < 10:
			if round2_hashes[0][word] == 1:
				funny_basis_words.append(word)
	# see if other voters found it funny
	yes_votes = dict()
	for word in funny_basis_words:
		yes_votes[word] = 0
	counts_same = []
	for voter in round2_voters:
		counts = 0
		for word in funny_basis_words:
			if round2_hashes[voter][word] == 1:
				yes_votes[word] = yes_votes[word] + 1
				counts = counts + 1
		counts_same.append(counts)
	for word in funny_basis_words:
		yes_votes[word] = yes_votes[word] / len(round2_voters)
	print(yes_votes)
	# find average number of same words
	mega_count = 0
	counts_identical = 0
	counts_null = 0
	for count in counts_same:
		mega_count = mega_count + count
		if count == 10:
			counts_identical = counts_identical + 1
		if count == 0:
			counts_null = counts_null + 1
	print(mega_count/ len(counts_same))
	print(counts_identical)
	print(counts_null)

def plot_test(wordss, cock_data, num_testers, E): # main test function described in paper
	voters_list = []
	errors_list = []
	voters = list(range(10)) #change this to make the "corrupted" data sets larger
	#print(voters)
	for j in range(num_testers):
		K = 1
		i = 1
		attack_type = 'none'
		CV_with_disc = False
		basis_voter = j
		train_words, test_words = get_test_train_words(10, 10, 20, 20, wordss, basis_voter, cock_data)
		m = len(train_words)
		errs, alphas, lambdas = test_funny(m, K, CV_with_disc, attack_type, i, wordss, cock_data, basis_voter, voters, E, test_words, train_words)
		print(errs)
		voters_list.append(j)
		errors_list.append(errs)
	plt.scatter(voters_list, errors_list)
	plt.xlabel("Voter id")
	plt.ylabel("Y-label")		
	plt.savefig("results.png")
	errors_list_np = np.array(errors_list)
	with open ("results.txt", "w") as f:
		f.write("mean error:")
		f.write(f"{np.mean(errors_list_np)}")
	print("done")


def get_test_train_words(num_funny_train_words, num_lame_train_words, num_funny_test_words, num_lame_test_words, wordss, basis_voter, cock_data):
	# get funny words
	train_words = []
	test_words = []
	funny_train_added = 0
	funny_test_added = 0
	lame_train_added = 0
	lame_test_added = 0
	votes = cock_data["word_ratings"]["votes"]
	# a list of voters, and a dictionary with each individual's votes for each word w 
	round2_voters = sorted(votes[2][wordss[2][0]]["yes_votes"] + votes[2][wordss[2][0]]["no_votes"])
	round3_voters = sorted(votes[3][wordss[2][0]]["yes_votes"] + votes[3][wordss[2][0]]["no_votes"])
	round4_voters = sorted(votes[4][wordss[2][0]]["yes_votes"] + votes[4][wordss[2][0]]["no_votes"])
	round2_hashes = [{w: 1 if u in votes[2][w]["yes_votes"] else 0 for w in wordss[2]} for u in round2_voters]
	round3_hashes = [{w: 2 if u in votes[2][w]["yes_votes"] else 0 for w in wordss[2]} for u in round3_voters]
	round4_hashes = [{w: 3 if u in votes[2][w]["yes_votes"] else 0 for w in wordss[2]} for u in round4_voters]
	
	wordss_set = set(wordss[2]) # to make it random
	
	# make train_words
	for word in wordss_set:
		if (round2_hashes[basis_voter][word] == 1):
			if (funny_train_added < num_funny_train_words):
				train_words.append(word)
				funny_train_added = funny_train_added + 1
			elif (funny_test_added < num_funny_test_words):
				test_words.append(word)
				funny_test_added = funny_test_added + 1
		elif (round2_hashes[basis_voter][word] == 0):
			if (lame_train_added < num_lame_train_words):
				train_words.append(word)
				lame_train_added = lame_train_added + 1
			elif (lame_test_added < num_lame_test_words):
				test_words.append(word)
				lame_test_added = lame_test_added + 1
	
	return train_words, test_words
# Loading the data


# training words: word embeddings of words to test
# cock_data: database returned by get_cockamanie in cockamanie.py
# basis_voter: person I'm basing alphas off of
# list of voters INCLUDING basis voter
# E: training embeddings set
# test_words: list of words to test
	
	
def get_data_fred(wordss, cock_data, basis_voter, voters, E, test_words, train_words):  #, train_subj, ):
	Inputs_train = []#np.ndarray() #[]
	Outputs_test = []#np.ndarray() #[]
	Outputs_train = []#np.ndarray() #[]
	Inputs_test = []#np.ndarray() #[]
	votes = cock_data["word_ratings"]["votes"]
	# a list of voters, and a dictionary with each individual's votes for each word w 
	round2_voters = sorted(votes[2][wordss[2][0]]["yes_votes"] + votes[2][wordss[2][0]]["no_votes"])
	round2_hashes = [{w: 1 if u in votes[2][w]["yes_votes"] else 0 for w in wordss[2]} for u in round2_voters]
	round2_voters == list(range(len(round2_voters)))
	

	for voter in voters:
		voter_train_inputs = []
		voter_train_outputs = []
		for word in train_words: 
			voter_train_inputs.append(E[word])
			voter_train_outputs.append(round2_hashes[voter][word])
		
		Inputs_train.append(voter_train_inputs)
		Outputs_train.append(voter_train_outputs)
		voter_test_inputs = []
		voter_test_outputs = []
		for word in test_words:
			voter_test_inputs.append(E[word])
			voter_test_outputs.append(round2_hashes[voter][word])
		Inputs_test.append(voter_test_inputs)
		Outputs_test.append(voter_test_outputs)
	#TODO: randomly generate words/subjects?	
	#for word in train_words:
	#	voter_train_inputs = []#np.ndarray()#[]
	#	voter_train_outputs = []#np.ndarray()#[]
	#	for subj in subjs:
	#		voter_train_inputs.append(E[word])
	#		voter_train_outputs.append(round2_hashes[subj][word])
			 		
	#for word in test_words:
	#	for subj in subjs:
	
	#for voter in round2_voters[0:5]: #start with these so that it takes less long
	#	if voter is not train_subj:
	#		voter_set_inputs = []#np.ndarray()#[]
	#		voter_set_outputs = []#np.ndarray()#[]
	#		for word in wordss[2]:
	#			if word not in test_words:
	#				#np.append(voter_set_inputs, E[word])
	#				#np.append(voter_set_outputs, round2_hashes[voter][word])
	#				voter_set_inputs.append(E[word])
	#				voter_set_outputs.append(round2_hashes[voter][word])
			#np.append(Inputs_train
	#		Inputs_train.append(voter_set_inputs)
	#		Outputs_train.append(voter_set_outputs)
	#print(Inputs_train)
	return (Inputs_train, Outputs_train, Inputs_test, Outputs_test)
	
### END COMPLETELY WRITTEN BY ME #####

### REST LIGHTLY MODIFIED BY ME ####
def test_funny(m, K, CV_with_disc, attack_type, i, wordss, cock_data, basis_voter, voters, E, test_words, train_words):
	
	# Get the data
	Inputs_train, Outputs_train, Inputs_test, Outputs_test = get_data_fred(wordss, cock_data, basis_voter, voters, E, test_words, train_words)
	Inputs_train = np.array(Inputs_train) # TODO: change I made 
	Inputs_test = np.array(Inputs_test) # TODO: change I made
	Outputs_train = np.array(Outputs_train) # TODO: change I made
	Outputs_test = np.array(Outputs_test) # TODO: change I made
	

	T = len(Inputs_train)
	d = Inputs_train[0].shape[1]
	#task_ind = i #TODO: removed this line
	# Store all data in big matrices
	X_train_all = np.zeros((1,d))
	Y_train_all = np.zeros(1)
	for i in range(T):
		X_train_all = np.concatenate((X_train_all, Inputs_train[i]), axis = 0)
		Y_train_all = np.concatenate((Y_train_all, Outputs_train[i]))
	X_train_all = X_train_all[1:,:]
	Y_train_all = Y_train_all[1:]

	#else: TODO: removed these next 4 lines
        
        # Get the data from the books and non-books. The target task has task_ind = 0
        #Inputs_train, Outputs_train, Inputs_test, Outputs_test, poisoned_tasks, task_ind \
        #    = get_data_specific(pathname, m, min_pos_reviews, n)
        
        #T = len(Inputs_train)
        #d = Inputs_train[0].shape[1]

	# Possible values for lambda
	lambdas = np.concatenate((np.array([0]), 2**(np.arange(-10,10, dtype=float)), np.array([10000])))

	#print("lambds\n\n\n", lambdas)
	# Select an optimal value of lambda by using K-fold validation. Calculate the
	# test error for this value of lambda and for all baselines
	# We also report the selected values of lambda and their corresponding alphas
	#print("Inputs_train")
	#print(Inputs_train)
	#print("Outputs_train")
	#print(Outputs_train)

	disc_for_task = get_disc_sq(Inputs_train, Outputs_train, basis_voter)
	disc_for_task = np.delete(disc_for_task, basis_voter) #TODO: CHANGE TO GET MY DATA SET TO WORK
	#print(disc_for_task)
	#np.delete(disc_for_task, basis_voter)

	val_errs = np.zeros((K, len(lambdas)))
	for k in range(K):
		# Create a training and validation set
		mask_val = np.all([np.arange(0, m*T) >= m*basis_voter + k*(m/K), np.arange(0, m*T) < m*basis_voter + (k+1)*(m/K)], axis = 0)
		#print(mask_val)
		mask_train = np.all([np.arange(0, m*T) >= m*basis_voter, np.arange(0, m*T) < m*(basis_voter+1), ~mask_val], axis = 0)
		X_val_now = X_train_all[mask_val, :]
		X_train_now = X_train_all[~mask_val, :]
		Y_val_now = Y_train_all[mask_val]
		Y_train_now = Y_train_all[~mask_val]

		ms = np.repeat(m, T)
		ms[basis_voter] = int(m - m/K) 
		#print(ms)
		ms = np.delete(ms, basis_voter) #TODO: CHANGE I MADE TO GET MY DATA SET TO WORK
		# If CV_with_disc == True, discrepancies are recalculated without the help of the validation set
		if CV_with_disc == True:
			
			Inputs_train2 = Inputs_train.copy()
			Inputs_train2[basis_voter] = X_train_all[mask_train,:]
			Outputs_train2 = Outputs_train.copy()
			Outputs_train2[basis_voter] = Y_train_all[mask_train]
			disc_for_task = get_disc_sq(Inputs_train2, Outputs_train2, basis_voter)
		
		# For each value of lambda, calculate the validation error
		for not_i, lamb in enumerate(lambdas):
			#print("disc!\n\n\n", disc_for_task)
			# For a given lambda, find optimal alphas
			#print("lamb\n\n", lamb)

			#disc_for_task = np.delete(disc_for_task, basis_voter)
			#np.delete(disc_for_task[basis_voter])
			#print("ms:", ms)

			

			alphas = optimize_weights(disc_for_task, ms, lamb, nepochs = 100, learn_rate = 0.01)
			
			# Compute the best predictor based on the training data only
			#print(Y_train_now)
			#print(alphas)
			best_w = get_lr_model_with_cv_on_clean(alphas, X_train_now, Y_train_now, ms, basis_voter, T)
			#best_w = get_lr_model(alphas, X_train_now, Y_train_now, ms, basis_voter, T)	
			# Evaluate the predictor on the validation set
			#print(1/(1+np.exp(-X_val_now@best_w)))			
			predictions = ((1/(1+np.exp(-X_val_now@best_w))) > 0.5).astype(int)

			val_errs[k, not_i] = np.mean(predictions != Y_val_now)


	# Select the optimal value of lambda
	lamb = lambdas[np.argmin(np.sum(val_errs, axis = 0))]

	selected_lambdas = lamb
	ms = np.repeat(m, T)


	# Obtain the best linear predictor based on the full training set, wiht the chosen value of lambda
	disc_for_task = get_disc_sq(Inputs_train, Outputs_train, basis_voter)
	alphas = optimize_weights(disc_for_task, ms, lamb, nepochs = 10001, learn_rate = 0.01)
	best_w = get_lr_model_with_cv_on_clean(alphas, X_train_all, Y_train_all, ms, basis_voter, T)
	#best_w = get_lr_model(alphas, X_train_all, Y_train_all, ms, basis_voter, T)
	# Evaluate the predictor on the test data from the task
	#print("Xtrain:")
	#print(X_train_all)
	#print("inputs test")	
	#print(Inputs_test)
	#print("inputs test basis")
	#print(Inputs_test[basis_voter])
	#print((1/(1+np.exp(-Inputs_test[basis_voter]@best_w))))
	#predictions = ((1/(1+np.exp(-Inputs_test[basis_voter]@best_w))) > 0.5).astype(int)
	predictions = ((1/(1+np.exp(-Inputs_test@best_w))) > 0.5).astype(int) #TODO: change I made to work with my data set
	print("Our Algo:", lamb, np.mean(predictions != Outputs_test)) 
	errs = np.mean(predictions != Outputs_test) #TODO: Change I made to work with my data set
	all_alphas = alphas
	#alphas = optimize_weights(disc_for_task, ms, 2, nepochs = 10001, learn_rate = 0.01)
	#best_w = get_lr_model_with_cv_on_clean(alphas, X_train_all, Y_train_all, ms, basis_voter, T)
	#print("lamb = 2:", np.mean(predictions != Outputs_test))
		
	return (errs, all_alphas, selected_lambdas)	
	
# end written by me


# Training (weighted) linear models

def get_lr_model(alphas, X_train_all, Y_train_all, ms, basis_voter, T):
    all_alphas = np.repeat(alphas, ms)
    #print(all_alphas)
    lr = LogisticRegression(C = 100, fit_intercept = False)
    lr.fit(X_train_all, Y_train_all, sample_weight=all_alphas)
    
    best_w = lr.coef_[0]
    return best_w


'''
get_lr_model_with_cv_on_clean minimizes an alpha-weighted log-loss and chooses a regularization
constant by cross-validation on the clean data only.
'''

def get_lr_model_with_cv_on_clean(alphas, X_train_all, Y_train_all, ms, clean_task, T):
    # Spread the weights on sample level and define the CV split
    all_alphas = np.repeat(alphas*T, ms)
    indexes_cv = (-1)*np.ones(X_train_all.shape[0])
    clean_begins = np.sum(ms[:clean_task])
    curr_m = ms[clean_task]
    all_alphas[clean_begins:(clean_begins + curr_m)] = all_alphas[clean_begins:(clean_begins + curr_m)]*(5/4)
    for l in range(5):
        indexes_cv[(clean_begins + l*(int(curr_m/5))):(clean_begins + (l+1)*int((curr_m/5)))] = l
    #print(indexes_cv)
    ps = PredefinedSplit(indexes_cv)
    #all_alphas = scipy.special.softmax(all_alphas)
    # Train on all data, with 5-fold CV on the clean data
    lr = LogisticRegressionCV(fit_intercept = False, cv = ps)
    #print("all_alphas:", all_alphas)
    #print("X_train:", X_train_all)
    #print("Y_train:", Y_train_all)
    lr.fit(X_train_all, Y_train_all, sample_weight=all_alphas)
    best_w = lr.coef_[0]
    return best_w


'''
get_model_with_cv finds the best linear predictor for a task based on the log loss, with cross-validation:
'''

def get_lr_model_with_cv(X_train_all, Y_train_all):
    
    lr = LogisticRegressionCV(fit_intercept = False)
    lr.fit(X_train_all, Y_train_all, sample_weight=all_alphas)
    best_w = lr.coef_[0]
    return best_w

############################################################################################
# Functions for estimating the discrepancies and optimizing the source weights

'''
get_disc_sq estimates the empirical discrepancies for a dataset, against all other sources, by finding a linear classifier
that performs well on one and badly on another. We use the linear classifier that minimises a SQUARE LOSS.
'''

def get_disc_sq(Inputs, Outputs, task_ind):
	# We store the estimates of the discrepancies in the Disc numpy array
	Disc = np.zeros((len(Inputs)))
	
	# We loop over all the other tasks
	X1 = Inputs[task_ind]
	for j in range(len(Inputs)):
		# We set up Inputs X and outputs Y as a classification problem
		X2 = Inputs[j]
		m1 = X1.shape[0]
		m2 = X2.shape[0]
		X = np.concatenate((X1, X2))
		Y = np.concatenate((Outputs[task_ind], -Outputs[j]+1))
		
		# We find the optimal parameter values and the fitted values using the usual
		# weighted linear regression formulas
		
		weights = np.concatenate((np.repeat(1/m1, m1), np.repeat(1/m2, m2)))
		bla = np.multiply(np.transpose(X), weights)
		w = np.linalg.solve(bla@X + 0.000001*np.eye(X.shape[1]),bla@Y)
		fitted = X@w
		
		# We estimate the discrepancy by evaluating the emperical discrepancy at this optimal
		# (for square loss) linear classifier
		loss = np.sum(((Y==1)*(fitted<0.5) + (Y==0)*(fitted>=0.5))*weights) - 1
		
		Disc[j] = np.abs(loss)

	return(Disc)
	
	
	
'''
optimize_weights finds the best values for the alphas, that minimize our bound. This is done for a particular
task index and discrepancies. lamb controls the strength of regularization.
'''
#https://stackoverflow.com/questions/55060736/tensorflow-2-api-regression-tensorflow-python-framework-ops-eagertensor-object
#def loss(lamb, alphas, ms, disc_for_task):#
#	return lamb*tf.sqrt(tf.reduce_sum(tf.square(alphas)*(1/ms))) + 2*tf.reduce_sum(alphas*(disc_for_task))

#def train(inputs, outputs):
#	with tf.GradientTape() as t:
#		current_loss = loss(inputs, outputs)
#	grads = t.gradient(current_loss, 

def optimize_weights(disc_for_task, ms, lamb, nepochs = 101, learn_rate = 0.01):
	tf.reset_default_graph()
	#ops.reset_default_graph()
	# Initiate the weights randomly
	T = disc_for_task.shape[0]
	with tf.name_scope('Parameters'):
		#logits = tf.Variable(tf.random_normal([T], stddev = 1/np.sqrt(T), dtype = tf.float32))
		logits = tf.Variable(tf.random.normal([T], stddev = 1/np.sqrt(T), dtype = tf.float32))
		alphas = tf.Variable(tf.nn.softmax(logits)) # Takes soft-max for every row
		#alphas = tf.Variable(tf.random_uniform([T], minval = 0, maxval = 1, dtype = tf.float32))
	# Find an expression for the optimisation target
	with tf.name_scope('loss'):
		loss = lamb*tf.sqrt(tf.reduce_sum(tf.square(alphas)*(1/ms))) + 2*tf.reduce_sum(alphas*(disc_for_task))
	
	# Run gradient descent to optimize the weights:
	optimizer = tf.train.AdamOptimizer(learn_rate)	
	#optimizer = tf.AdamOptimizer(learn_rate)
	#optimizer = tf.optimizers.Adam(learn_rate)
	train = optimizer.minimize(loss)
	#train = optimizer.minimize(loss, var_list=[tf.convert_to_tensor(lamb, dtype=tf.float32), tf.convert_to_tensor(alphas, dtype=tf.float32), tf.convert_to_tensor(ms, dtype = tf.float32), tf.convert_to_tensor(disc_for_task, dtype=tf.float32)]) #my change
			
	with tf.Session() as sess:	  
		sess.run(tf.global_variables_initializer()) # set initial random values	  
		# Train the linear model
		#print(sess.run(alphas))
		for i in range(nepochs):
			sess.run(train)
			#if i%100 == 0:
				#print(sess.run(loss))
		alph = sess.run(alphas)
		return scipy.special.softmax(alph) #TODO: change I made. alphas were becoming negative, so I used softmax to get it to get the alphas to reasonable values. This could be a source of my wrong results
	

