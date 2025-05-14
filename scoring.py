import re
import regex
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.stem import PorterStemmer
ps = PorterStemmer()

temporal_pattern = re.compile(
    r'\b('
    r'\d{4}-Q[1-4]|'                         # Quarter: 2025-Q2
    r'\d{4}-(?:Spring|Summer|Fall|Winter)|'  # Season: 2025-Summer
    r'\d{4}-W\d{2}-WE|'                      # Weekend: 2025-W14-WE
    r'\d{4}-W\d{2}|'                         # Week: 2025-W14
    r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}|'        # Datetime: 2025-04-06T14:30
    r'\d{4}-\d{2}-\d{2}T(?:MO|AF|EV|NI)|'    # Daypart: 2025-04-06TMO
    r'\d{4}-\d{2}-\d{2}|'                    # Day: 2025-04-06
    r'\d{4}-\d{2}|'                          # Month: 2025-04
    r'\d{4}|'                                # Year: 2025
    r'\b\d{2}:\d{2}'                         # Time: 14:30
    r')\b'
)

def tokenize(sentence):
    return [ps.stem(w) for w in normalize_answer(sentence).split()]

def normalize_answer(s):
    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def json_to_list(json_text):
    match = re.search(r'\[\s*(?:"[^"]*"\s*,\s*)*"[^"]*"\s*\]', json_text)

    if match:
        try:
            json_text = match.group(0)
            json_list = eval(json_text)
            if type(json_list[0]) == dict:
                json_list = [j['resolved'] for j in json_list]
            return json_list
        except:
            print(json_text)
            return []
    
    print(json_text)
    return []

def extract_datetimes(text):
    return re.findall(temporal_pattern, text)

def get_datetimes(pred_json, ref_json):
    pred_list = json_to_list(pred_json)
    ref_list = json_to_list(ref_json)

    if not len(pred_list) == len(ref_list):
        pred_datetimes = [[] for _ in range(len(ref_list))]
    else:
        pred_datetimes = [extract_datetimes(line) for line in pred_list]
    ref_datetimes = [extract_datetimes(line) for line in ref_list]
    
    return pred_datetimes, ref_datetimes

def compute_f1(pred, ref):
    pred_set = set(pred)
    ref_set = set(ref)

    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return f1, precision, recall

def compute_exact_match(pred, ref):
    return int(pred == ref)

def print_scores(results):
    for data in results:
        ### Full Dialogue ###
        predictions, references = [], []
        for d in data:
            pred = json_to_list(d['pred'])
            ref = json_to_list(d['ref'])

            if not len(pred) == len(ref):
                pred = [''] * len(ref)

            predictions.extend(pred)
            references.extend(ref)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge1_scores, rougeL_scores = [], []
        bleu_scores, f1_scores, em_scores = [], [], []
        smoothie = SmoothingFunction().method4

        for pred_str, ref_str in zip(predictions, references):
            pred = tokenize(pred_str)
            ref = tokenize(ref_str)

            # BLEU
            bleu = sentence_bleu([ref], pred, smoothing_function=smoothie)
            bleu_scores.append(bleu)

            # ROUGE
            score = scorer.score(ref_str, pred_str)
            rouge1_scores.append(score['rouge1'].fmeasure)
            rougeL_scores.append(score['rougeL'].fmeasure)

            # F1
            f1, prec, rec = compute_f1(pred, ref)
            f1_scores.append(f1)

            # Exact Match
            em = compute_exact_match(pred_str, ref_str)
            em_scores.append(em)

        print(f'BLEU: {sum(bleu_scores)/len(bleu_scores)*100:.2f}')
        print(f'ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores)*100:.2f}')
        print(f'F1 Score: {sum(f1_scores)/len(f1_scores)*100:.2f}')
        print(f'Exact Match: {sum(em_scores)/len(em_scores)*100:.2f}')

        ### Datetime Only ###
        predictions, references = [], []
        for d in data:
            pred, ref = get_datetimes(d['pred'], d['ref'])
            predictions.extend(pred)
            references.extend(ref)
        
        em_scores = []
        for pred, ref in zip(predictions, references):
            em = compute_exact_match(pred, ref)
            em_scores.append(em)

        print(f'Exact Match (Datetime Only): {sum(em_scores)/len(em_scores)*100:.2f}')
        print()