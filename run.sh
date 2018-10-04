gpu_id=0

mode='train'

train_data='/home/kgb/qacnn_1d/data/movie_qa/train.json'
dev_data='/home/kgb/qacnn_1d/data/movie_qa/dev.json'

dropout=0.2
cnn_layers=3

#resume_dir=''
save_dir='./model/movie_qa'
test_result='./result'

CUDA_VISIBLE_DEVICES=$gpu_id python main.py $mode --train_data $train_data --dev_data $dev_data \
--dropout $dropout --cnn_layers $cnn_layers \
--save_dir $save_dir

