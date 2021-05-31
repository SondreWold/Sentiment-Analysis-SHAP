make_folders:
	-v mkdir data
	-v mkdir models
	-v mkdir configs

get_train:
	wget https://raw.githubusercontent.com/ltgoslo/NorBERT/main/benchmarking/data/sentiment/no/train.csv -P ./data/

get_dev:
	wget https://raw.githubusercontent.com/ltgoslo/NorBERT/main/benchmarking/data/sentiment/no/dev.csv -P ./data/

get_test:
	wget https://raw.githubusercontent.com/ltgoslo/NorBERT/main/benchmarking/data/sentiment/no/test.csv -P ./data/

get_norec: get_train get_dev get_test


