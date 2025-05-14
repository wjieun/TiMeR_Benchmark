import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftModel
import torch
import json
from tqdm import tqdm
from scoring import print_scores
from instruction import FINETUNE_INST

batch_size = 16
basemodel_name = "meta-llama/Llama-3.1-8B"
model_name = "" # YOUR MODEL

def format_data1(example):
    orig_conv = [e["speaker"] + ": " + e["text"] for e in example["orig_conv"]]
    res_conv = [e["speaker"] + ": " + e["text"] for e in example["res_conv"]]

    orig_conv = json.dumps(orig_conv)
    res_conv = json.dumps(res_conv)

    text = (
        "<s>[INST]"
        f"{FINETUNE_INST}"
        f"SPEECH TIME: {example['speech_time']} (Week {example['week_num']})\n\n"
        f"CONVERSATION: {orig_conv}\n"
        "[/INST]\n\n"
    )

    return {'text': text, 'orig': orig_conv, 'label': res_conv}

def format_data2(example):
    orig_conv = [f"{example['speech_time']} - {e['speaker']}: {e['text']}" for e in example["orig_conv"]]
    res_conv = [f"{example['speech_time']} - {e['speaker']}: {e['text']}" for e in example["res_conv"]]

    orig_conv = json.dumps(orig_conv)
    res_conv = json.dumps(res_conv)

    text = (
        "<s>[INST]"
        f"{FINETUNE_INST}"
        "SPEECH TIME: Inherent in the following CONVERSATION.\n\n"
        f"CONVERSATION: {orig_conv}\n"
        "[/INST]\n\n"
    )

    return {'text': text, 'orig': orig_conv, 'label': res_conv}

def decode_tokens(tokenized_output):
    return tokenizer.batch_decode(tokenized_output, skip_special_tokens=True)

def generate_predictions(dataset, model, tokenizer, batch_size, max_new_tokens=1024):
    model.eval()
    preds, refs, origs = [], [], []

    all_texts = [example["text"] for example in dataset]
    all_refs = [example["label"] for example in dataset]
    all_origs = [example["orig"] for example in dataset]

    for i in tqdm(range(0, len(all_texts), batch_size), desc='Generate Predictions'):
        batch_texts = all_texts[i:i+batch_size]
        batch_refs = all_refs[i:i+batch_size]
        batch_origs = all_origs[i:i+batch_size]

        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id
            )

        batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for pred, ref, orig in zip(batch_preds, batch_refs, batch_origs):
            preds.append(pred.split("[/INST]")[-1].split("</s>")[0].strip())
            refs.append(ref)
            origs.append(orig)

    return preds, refs, origs

if __name__ == "__main__":
    if model_name:
        base_model = AutoModelForCausalLM.from_pretrained(basemodel_name, device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    else:
        model = AutoModelForCausalLM.from_pretrained(basemodel_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(basemodel_name, padding_side='left')
        FINETUNE_INST += 'Follow the format of the CONVERSATION and output the response as a python list.\n\n'

    device = model.device
    tokenizer.pad_token = tokenizer.eos_token
    data = load_dataset("wjieun/TiMeR")["test"]

    results = []
    for idx, format_data in enumerate([format_data1, format_data2]):
        test_dataset = data.map(format_data)
        predictions, references, originals = generate_predictions(test_dataset, model, tokenizer, batch_size)
        results.append([{'pred': p, 'ref': r, 'orig': o} for p, r, o in zip(predictions, references, originals)])

        with open(f'prediction{idx+1}.json', 'w') as f:
            json.dump(results[idx], f, indent=4)

    print_scores(results)