train_data_path = 	"data/chunked/train/train_*"
eval_data_path = 	"data/chunked/valid/valid_*"
decode_data_path = 	"data/chunked/test/test_*"
vocab_path = 		"data/vocab"


# Hyperparameters
hidden_dim= 512
emb_dim= 256
batch_size=  200
max_enc_steps=55		#99% of the articles have no. of words within 55
max_dec_steps=15		#99% of the titles have no. of words within 15
max_rec_dec_steps =55 
beam_size=4
min_dec_steps=3
min_dec_rec_steps =10
vocab_size=50000

lr=0.001
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4

eps = 1e-12
max_iterations = 500000


save_model_path = "data/saved_models"
load_model_path = save_model_path + "/0185000.tar"

resume_training = False

intra_encoder = True
intra_decoder = True

