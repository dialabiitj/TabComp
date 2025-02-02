from .tools import llm_answer_eval, postprocess_llm_vqa, textcaps_textvqa_eval

if __name__ == '__main__':

    llm_answer_eval(metric_names=['RelaxedAccuracy'], result_path='evaluate_results/test.jsonl', save_each_eval=True)
    llm_answer_eval(metric_names=['ExactAccuracy'], result_path='evaluate_results/test.jsonl', save_each_eval=True)
    llm_answer_eval(metric_names=['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'Meteor', 'RougeL', 'CIDEr', 'Bertscore'], result_path='evaluate_results/test.jsonl', save_each_eval=True)


    # postprocess_llm_vqa(dataset_name='DocVQA', split='test',
    #                         llm_pred_path='./evaluate_results/test.jsonl',
    #                         eval_flag=True)
    
    # # need to submit evaluate_results/***_official_eval.json
    # textcaps_textvqa_eval(result_path='evaluate_results/test.jsonl', dataset='TextVQA', split='test')




