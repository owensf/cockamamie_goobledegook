import cockamamie as cock
import pickle as pkl
import functions_untrusted_mychanges as myf
import matplotlib.pyplot as plt
# WRITTEN BY ME
if __name__ == "__main__":
	cock_data = cock.get_cockamamie()
	votes = cock_data["word_ratings"]["votes"]
	#print(len(votes[4]["asshattery"]["yes_votes"]))
	EH_scores = cock.get_EH()
	wordss = [sorted(ws) for ws in votes]
	words_of_interest = [("EH", list(EH_scores)), ("120k", wordss[0])]
	#cock.vec_file2dict("wiki-news-300d-1M-subword.vec", "wiki-news-300d-120kEH-subword.pk1", words_of_interest) # Note: run this to generate WEs 
	#myf.show_differences(wordss, cock_data)
	filename = "wiki-news-300d-120kEH-subword.pk1"
	with open(filename, "rb") as f:
		E_wiki = pkl.load(f)
	myf.plot_test(wordss, cock_data, 5, E_wiki) #main test function
	#print(E_wiki)
	#(cock.scatter_embedding(EH_scores, E_wiki))
	#print(len(E_wiki["cockamamie"]))
	#print(f"Loaded {len(E_wiki):,} words from {filename}")
	#print(wordss[2])
	#basis_voter = 0#NOTE: Index of voter??
	#voters = [1, 2, 3, 4, 5, 6]
	#voters = list(range(0, 200))
	#round2_voters = sorted(votes[2][wordss[2][0]]["yes_votes"] + votes[2][wordss[2][0]]["no_votes"])
	#round2_hashes = [{w: 1 if u in votes[2][w]["yes_votes"] else 0 for w in wordss[2]} for u in round2_voters]

	#print(round2_hashes[199])
	#train_words, test_words = myf.get_test_train_words(10, 10, 20, 20, wordss, basis_voter, cock_data)
	#print(train_words, test_words)	
	#train_words = ["apeshit", "arseholes", "kerfuffle", "tootsies", "clusterfuck"]
	#test_words = ["cockamamie", "annulus", "dickheads", "twat", "nutted"]

	#Inputs_train, Outputs_train, Inputs_test, Outputs_test = myf.get_data_fred(wordss, cock_data, train_subj, E_wiki, test_words)
	#print(len(Outputs_train))
	# m: number of samples per task
	m = len(train_words)
	# K: parameter for the K-fold CV
	K = 1
	#CV_with_disc: if True, discrepancies are recomputed every time a part of the data is taken out for cross-validation
	CV_with_disc = False
	# n: desired values of non-books 
	#n = 100
	# p:
	#p = 
	# i:
	i = 1 
	attack_type = 'none'

	#myf.test_funny(m, K, CV_with_disc, attack_type, i, wordss, cock_data, basis_voter, voters, E_wiki, test_words, train_words)	all_errors = []
	all_lambdas = []
	all_alphas = []
	################ This was the old way I tested my results, and I didn't want to delete it. ##########
	for i in range(0): ############change this to number besides 0 for how many tests you want
		basis_voter = i
		train_words, test_words = myf.get_test_train_words(10, 10, 20, 20, wordss, basis_voter, cock_data)
		#print(train_words, test_words)
		errs, alphas, lambdas = myf.test_funny(m, K, CV_with_disc, attack_type, i, wordss, cock_data, basis_voter, voters, E_wiki, test_words, train_words)
		all_errors.append(errs)
		all_lambdas.append(lambdas)
		all_alphas.append(alphas)
	with open ("results.txt", "w") as f:
		f.write("errors:\n")
		for error in all_errors:
			f.write(f"{error}")
		#f.write(all_errors)
		f.write("\n")
		f.write("lambdas:")
		for l in all_lambdas:
			f.write(f"{l}")
		f.write("\n")
		f.write("alphas:")
		for alpha in all_alphas:
			f.write(f"{alpha}")
