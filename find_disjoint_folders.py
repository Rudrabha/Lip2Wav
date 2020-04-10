import pickle

# train = pickle.load(open('logs/filenames_train.pkl', 'rb'))
test = pickle.load(open('logs/filenames_test.pkl', 'rb'))

# train_vids = {}
# for x in train:
# 	train_vids[x[:x.rfind('/')]] = True

test_vids = {}
for x in test:
	test_vids[x[:x.rfind('/')]] = True

# for t in train_vids:
# 	if t in test_vids:
# 		del test_vids[t]

for t in test_vids:
	input(t)