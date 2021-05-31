SAGA_UNAME?=sondrewo

make_folders:
	-v mkdir data
	-v mkdir models
	-v mkdir configs
	-v mkdir saga_models

get_train:
	wget https://raw.githubusercontent.com/ltgoslo/NorBERT/main/benchmarking/data/sentiment/no/train.csv -P ./data/

get_dev:
	wget https://raw.githubusercontent.com/ltgoslo/NorBERT/main/benchmarking/data/sentiment/no/dev.csv -P ./data/

get_test:
	wget https://raw.githubusercontent.com/ltgoslo/NorBERT/main/benchmarking/data/sentiment/no/test.csv -P ./data/

get_norec: get_train get_dev get_test

download_saga_classifiers:
	scp -r $(SAGA_UNAME)@saga.sigma2.no:~/Sentiment-Analysis-SHAP/models/ ./saga_models/