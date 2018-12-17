train_data_path = 	"data/chunked/train/train_*"
valid_data_path = 	"data/chunked/valid/valid_*"
test_data_path = 	"data/chunked/test/test_*"
vocab_path = 		"data/vocab"


# Hyperparameters
hidden_dim = 512
emb_dim = 256
batch_size = 200
max_enc_steps = 55		#99% of the articles are within length 55
max_dec_steps = 15		#99% of the titles are within length 15
beam_size = 4
min_dec_steps= 3
vocab_size = 50000

lr = 0.001
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

eps = 1e-12
max_iterations = 500000


save_model_path = "data/saved_models"

intra_encoder = True
intra_decoder = True