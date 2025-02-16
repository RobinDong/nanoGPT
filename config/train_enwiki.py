# train a miniature word-level enwiki model
# good for debugging and playing on macbooks and such

out_dir = 'out-enwiki'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 5000 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False
init_from = 'resume'

wandb_log = False # override via command line if you like
wandb_project = 'enwiki'
wandb_run_name = 'nano-gpt'

dataset = 'enwiki'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 512 # context of up to 256 previous words

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 400000
lr_decay_iters = 200000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 900 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
