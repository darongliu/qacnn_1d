gpu_id=0

mode='train'
batch_size=10

train_data='/home/kgb/qacnn_1d/data/train.json'
dev_data='/home/kgb/qacnn_1d/data/dev.json'

dropout=0.1
cnn_layers=3

#resume_dir='./model/movie_qa'
save_dir='./model/kgb_4_nn'
test_result='./result'

CUDA_VISIBLE_DEVICES=$gpu_id python main.py $mode --batch_size $batch_size --train_data $train_data --dev_data $dev_data \
--dropout $dropout --cnn_layers $cnn_layers \
--save_dir $save_dir \
#--resume_dir $resume_dir \

