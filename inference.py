import torch
import tqdm

def inference(model, data_loader, output_path):
    all_predict = {}
    for i, data in enumerate(tqdm(data_loader, ncol=70)):
        context, question, option, question_id, _ = data
        context, question, option = put_to_cuda([context, question, option])

        output = model(context, question, option)
        _, predict = torch.max(output)
        predict_answer = predict.cpu().numpy()[0]
        all_predict[question_id[0]] = predict_answer

    print('total question number: ', len(all_predict))

    with open(output_path, 'w') as f:
        f.write('ID,Answer\n')
        all_key = sorted(all_predict.keys())
        for key in all_key:
            ans = all_predict[key]+1
            f.write(str(key)+','+str(ans)+'\n')


