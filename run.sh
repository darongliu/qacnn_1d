gpu_id=3

mode='train'
batch_size=64

train_data='/home/kgb/qacnn_1d/data/train.json'
dev_data='/home/kgb/qacnn_1d/data/dev.json'

dropout=0.2
cnn_layers=4

#resume_dir='./model/movie_qa'
save_dir='./model/kgb_4'
test_result='./result'

CUDA_VISIBLE_DEVICES=$gpu_id python main.py $mode --batch_size $batch_size --train_data $train_data --dev_data $dev_data \
--dropout $dropout --cnn_layers $cnn_layers \
--save_dir $save_dir \
#--resume_dir $resume_dir \

