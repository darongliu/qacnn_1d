import torch
import tqdm
from utils.utils import * 

def inference(model, test_data_loader, output_path, output_prob='./prob'):
    all_predict = {}
    all_prob = {}
    #for i, data in enumerate(tqdm.tqdm(test_data_loader)):
    for i, data in enumerate(test_data_loader):
        context, question, option, question_id, _, useful_feat = data
        #context = torch.cat([context, torch.zeros([1,10,context.size()[-1]])], 1)
        context, question, option, useful_feat = put_to_cuda([context, question, option, useful_feat])

        output = model(context, question, option, useful_feat)
        _, predict = torch.max(output, 1)
        predict_answer = predict.cpu().detach().numpy()
        prob = output.cpu().detach().numpy()
        all_predict[question_id[0]] = predict_answer[0]
        all_prob[question_id[0]] = prob[0]
        
        #all_predict[question_id[1]] = predict_answer[1]
        #all_prob[question_id[1]] = prob[1]

    print('total question number: ', len(all_predict))

    with open(output_path, 'w') as f:
        f.write('ID,Answer\n')
        all_key = sorted(all_predict.keys())
        for key in all_key:
            ans = all_predict[key]+1
            f.write(str(key)+','+str(ans)+'\n')

    with open(output_prob, 'w') as f:
        all_key = sorted(all_prob.keys())
        for key in all_key:
            prob = all_prob[key]
            f.write(str(key))
            for p in prob:
                f.write(',')
                f.write(str(p))
            f.write('\n')
