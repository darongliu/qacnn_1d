gpu_id=3

mode='train'
batch_size=32

train_data='/home/kgb/qacnn_1d/data/movie_qa/train_part.json'
dev_data='/home/kgb/qacnn_1d/data/movie_qa/dev_part.json'

dropout=0.1
cnn_layers=4

#resume_dir='./model/movie_qa'
save_dir='./model/test'
test_result='./result'

CUDA_VISIBLE_DEVICES=$gpu_id python main.py $mode --batch_size $batch_size --train_data $train_data --dev_data $dev_data \
--dropout $dropout --cnn_layers $cnn_layers \
--save_dir $save_dir \
#--resume_dir $resume_dir \

