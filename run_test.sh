gpu_id=1

mode='test'
batch_size=32

test_data='/home/kgb/qacnn_1d/data/dev.json'

dropout=0.
cnn_layers=4

#resume_dir='./model/movie_qa'
resume_dir='./model/test_final'
test_result='./result'

CUDA_VISIBLE_DEVICES=$gpu_id python main.py $mode --batch_size $batch_size --test_data $test_data \
--dropout $dropout --cnn_layers $cnn_layers \
--resume_dir $resume_dir \
--test_result $test_result
