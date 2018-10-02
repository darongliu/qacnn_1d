gpu_id=0

mode='train'

train_data='/home/darong/darong/qacnn_1d/data/movie_qa/train.json'
dev_data='/home/darong/darong/qacnn_1d/data/movie_qa/dev.json'

resume_dir=''
save_dir='./model/movie_qa'
log='./log/movie_wa_log'
test_result='./result'

CUDA_VISIBLE_DEVICES=$gpu_id python main.py $mode --train_data $train_data --dev_data $dev_data \
--save_dir $save_dir

