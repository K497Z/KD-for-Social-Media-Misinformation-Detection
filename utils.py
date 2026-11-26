import ast
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def compute_mse_loss(student_f, teacher_f):
    return F.mse_loss(student_f, teacher_f)


def decode_sequence(value):
    if isinstance(value, (list, tuple, dict)):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text[0] in '[{(':
        for decoder in [json.loads, ast.literal_eval]:
            try:
                return decoder(text)
            except (ValueError, SyntaxError, TypeError):
                pass
    return [item.strip() for item in re.split(r'\[SEP\]|\|\|\||[\r\n]+', text) if item.strip()]


def parse_comments(value):
    items = decode_sequence(value)
    if isinstance(items, dict):
        items = items.get('comments', [items])
    if not isinstance(items, (list, tuple)):
        items = [items]
    comments, tags = [], []
    for item in items:
        if isinstance(item, dict):
            text = item.get('text', item.get('content', item.get('comment', '')))
            tag = item.get('sentiment', item.get('sent', item.get('polarity')))
        else:
            text, tag = str(item), None
        if str(text).strip():
            comments.append(str(text).strip())
            tags.append(tag)
    return comments, tags


def sentiment_index(value):
    key = str(value).strip().lower()
    labels = {'positive': 0, 'pos': 0, '正面': 0, '积极': 0,
              'neutral': 1, 'neu': 1, '中性': 1,
              'negative': 2, 'neg': 2, '负面': 2, '消极': 2}
    if key not in labels:
        raise ValueError('Sentiment tags must identify positive, neutral or negative: ' + key)
    return labels[key]


def extract_social_emotion(comments, extractor_func, target_dim=3):
    if target_dim != 3:
        raise ValueError('The sentiment feature has exactly three components')
    if not comments:
        return np.zeros(3, dtype=np.float32)
    tags = extractor_func(comments)
    if len(tags) != len(comments):
        raise ValueError('One sentiment tag is required for each comment')
    counts = np.bincount([sentiment_index(tag) for tag in tags], minlength=3)
    return (counts / len(comments)).astype(np.float32)


class CommentSentiment:
    def __init__(self, model_name, device='cpu', batch_size=32, offline=False):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=offline).to(device).eval()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.cache = {}

    def __call__(self, comments):
        missing = list(dict.fromkeys(text for text in comments if text not in self.cache))
        with torch.no_grad():
            for start in range(0, len(missing), self.batch_size):
                texts = missing[start:start + self.batch_size]
                encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
                encoded = {key: tensor.to(self.device) for key, tensor in encoded.items()}
                predictions = self.model(**encoded).logits.argmax(dim=-1).cpu().tolist()
                for text, index in zip(texts, predictions):
                    tag = self.model.config.id2label[index]
                    sentiment_index(tag)
                    self.cache[text] = tag
        return [self.cache[text] for text in comments]


def normalize_frame(frame):
    source = next((name for name in ['news_content', 'content', 'text'] if name in frame.columns), None)
    if source is None or 'label' not in frame.columns:
        raise ValueError('Data must contain text/content/news_content and label columns')
    result = frame.copy().reset_index(drop=True)
    result['content'] = result[source].fillna('').astype(str)
    labels = pd.to_numeric(result['label'], errors='raise')
    if not labels.isin([0, 1]).all():
        raise ValueError('Labels must be 0 for real and 1 for fake')
    result['label'] = labels.astype(int)
    if 'comments' not in result:
        result['comments'] = ''
    return result


def prepare_frame(frame, extractor=None, use_privileged=True, max_comments=None):
    result = normalize_frame(frame)
    comment_sets, emotions = [], []
    for _, row in result.iterrows():
        comments, tags = parse_comments(row['comments']) if use_privileged else ([], [])
        explicit = next((row[name] for name in ['comment_sentiments', 'sentiments'] if use_privileged and name in row.index and str(row[name]).strip() not in ['', 'nan']), None)
        if explicit is not None:
            tags = list(decode_sequence(explicit))
            if len(tags) != len(comments):
                raise ValueError('Comment and sentiment counts differ')
        if comments and all(tag is not None for tag in tags):
            emotion = extract_social_emotion(comments, lambda values: tags)
        elif comments:
            if extractor is None:
                raise ValueError('Comment sentiment tags or a sentiment classifier are required')
            emotion = extract_social_emotion(comments, extractor)
        else:
            emotion = np.zeros(3, dtype=np.float32)
        comment_sets.append(comments if max_comments is None else comments[:max_comments])
        emotions.append(emotion)
    result['_comments'] = comment_sets
    result['_sentiment'] = emotions
    return result


def fingerprint(frame):
    columns = [name for name in ['content', 'comments', 'label', 'comment_sentiments', 'sentiments'] if name in frame]
    content = frame[columns].to_json(orient='records', force_ascii=False)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def check_disjoint(frames):
    names = list(frames)
    keys = {name: set(frames[name]['content'].str.strip()) for name in names}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if keys[left] & keys[right]:
                raise ValueError('Source posts overlap across ' + left + ' and ' + right)


def load_splits(data_dir='.', data_path=None, seed=42):
    if data_path:
        frame = normalize_frame(pd.read_csv(data_path))
        train_val, test = train_test_split(frame, test_size=0.2, stratify=frame['label'], random_state=seed)
        train, val = train_test_split(train_val, test_size=0.125, stratify=train_val['label'], random_state=seed)
        frames = {'train': train, 'val': val, 'test': test}
    else:
        folder = Path(data_dir)
        frames = {name: normalize_frame(pd.read_csv(folder / (name + '.CSV'))) for name in ['train', 'val', 'test']}
    check_disjoint(frames)
    return {name: frame.reset_index(drop=True) for name, frame in frames.items()}


class RecordDataset(Dataset):
    def __init__(self, frame):
        self.records = frame.to_dict(orient='records')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class BatchCollator:
    def __init__(self, tokenizers, max_len=128, include_comments=True):
        self.tokenizers = tokenizers
        self.max_len = max_len
        self.include_comments = include_comments

    def __call__(self, rows):
        comments = [row.get('_comments', []) if self.include_comments else [] for row in rows]
        count = max(1, max(map(len, comments)))
        valid = torch.tensor([[index < len(items) for index in range(count)] for items in comments], dtype=torch.bool)
        batch = {'label': torch.tensor([row['label'] for row in rows], dtype=torch.long),
                 'source_ids': {}, 'source_masks': {}, 'comment_ids': {}, 'comment_masks': {},
                 'valid_comments': valid,
                 'sentiment': torch.tensor(np.stack([row.get('_sentiment', np.zeros(3)) for row in rows]), dtype=torch.float32)}
        texts = [row['content'] for row in rows]
        for name, tokenizer in self.tokenizers.items():
            encoded = tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
            batch['source_ids'][name] = encoded['input_ids']
            batch['source_masks'][name] = encoded['attention_mask']
            if name != 'student' and self.include_comments:
                flat = [items[index] if index < len(items) else '' for items in comments for index in range(count)]
                encoded = tokenizer(flat, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
                batch['comment_ids'][name] = encoded['input_ids'].reshape(len(rows), count, -1)
                batch['comment_masks'][name] = encoded['attention_mask'].reshape(len(rows), count, -1)
        return batch


def make_loader(frame, tokenizers, cfg, shuffle=False, include_comments=True):
    generator = torch.Generator().manual_seed(cfg.get('seed', 42))
    return DataLoader(RecordDataset(frame), batch_size=cfg.get('batch_size', 32), shuffle=shuffle,
                      num_workers=cfg.get('num_workers', 0), generator=generator,
                      collate_fn=BatchCollator(tokenizers, cfg.get('max_len', 128), include_comments))


def move_batch(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_batch(item, device) for key, item in value.items()}
    return value
