import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from evaluation import Evaluation, load_student
from model import Model, StudentModel, TeacherModel
from trainer import Trainer, train_teacher
from utils import CommentSentiment, check_disjoint, fingerprint, load_splits, make_loader, move_batch, prepare_frame, set_seed


def default_config():
    return {'student_model': 'hfl/chinese-roberta-wwm-ext',
            'teacher_models': {'bert': 'bert-base-chinese', 'roberta': 'hfl/chinese-roberta-wwm-ext'},
            'teacher_names': ['bert', 'roberta'], 'use_privileged': True,
            'sentiment_model': 'lxyuan/distilbert-base-multilingual-cased-sentiments-student',
            'epochs': 20, 'teacher_epochs': 20, 'batch_size': 32, 'max_len': 128,
            'lr': 2e-5, 'teacher_lr': 2e-5, 'agent_lr': 1e-4, 'alpha': 0.4,
            'temperature': 2.0, 'warmup': 1, 'entropy_coefficient': 0.02,
            'action_temperature': 1.0, 'baseline_decay': 0.9, 'max_grad_norm': 1.0,
            'comment_batch_size': 32, 'max_comments': None, 'num_workers': 0,
            'seed': 42, 'split_seed': 42, 'seeds': [42, 43, 44, 45, 46],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'offline': False,
            'mode': 'train', 'data_dir': str(Path(__file__).resolve().parent),
            'data_path': None, 'output_dir': 'outputs', 'checkpoint': None, 'folds': 5}


def parse_config(mode='train', cross_validation=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--mode', choices=['train', 'train_teachers', 'no_agent', 'no_pi', 'single_teacher', 'sensitivity', 'predict'], default=None)
    parser.add_argument('--data-dir')
    parser.add_argument('--data-path')
    parser.add_argument('--output-dir', '--save_dir', dest='output_dir')
    parser.add_argument('--student-model')
    parser.add_argument('--bert-model')
    parser.add_argument('--roberta-model')
    parser.add_argument('--sentiment-model')
    parser.add_argument('--device')
    parser.add_argument('--checkpoint')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--split-seed', type=int)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--teacher-epochs', type=int)
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int)
    parser.add_argument('--max-len', type=int)
    parser.add_argument('--lr', '--training_learning_rate_student', dest='lr', type=float)
    parser.add_argument('--teacher-lr', type=float)
    parser.add_argument('--agent-lr', type=float)
    parser.add_argument('--alpha', '--training_alpha', dest='alpha', type=float)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--entropy-coefficient', type=float)
    parser.add_argument('--action-temperature', type=float)
    parser.add_argument('--baseline-decay', type=float)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--max-comments', type=int)
    parser.add_argument('--comment-batch-size', type=int)
    parser.add_argument('--folds', type=int)
    parser.add_argument('--offline', action='store_true', default=None)
    parser.add_argument('--cross-validation', action='store_true', default=None)
    args = vars(parser.parse_args())
    cfg = default_config()
    cfg['mode'] = mode
    cfg['cross_validation'] = cross_validation
    loaded = {}
    if args.get('config'):
        import yaml
        loaded = yaml.safe_load(Path(args['config']).read_text(encoding='utf-8'))
        if not isinstance(loaded, dict):
            raise ValueError('Configuration must be a mapping')
        cfg.update(loaded)
    for key, value in args.items():
        if value is not None and key not in ['config', 'bert_model', 'roberta_model']:
            cfg[key] = value
    for name in ['bert', 'roberta']:
        if args.get(name + '_model'):
            cfg['teacher_models'][name] = args[name + '_model']
    if args.get('seed') is not None and args.get('seeds') is None:
        cfg['seeds'] = [args['seed']]
    cfg['use_privileged'] = cfg['mode'] != 'no_pi'
    if args.get('output_dir') is None and 'output_dir' not in loaded:
        cfg['output_dir'] = str(Path('outputs') / (cfg['mode'] + ('_5fold' if cfg['cross_validation'] else '')))
    if cfg['epochs'] < 1 or cfg['teacher_epochs'] < 1 or cfg['batch_size'] < 1:
        raise ValueError('Epochs and batch size must be positive')
    if cfg['max_len'] < 2 or cfg['comment_batch_size'] < 1 or cfg['warmup'] < 0:
        raise ValueError('Invalid sequence, comment batch or warm-up setting')
    if not 0 <= cfg['alpha'] <= 1 or cfg['temperature'] <= 0 or cfg['action_temperature'] <= 0:
        raise ValueError('Invalid distillation or action setting')
    if not 0 <= cfg['baseline_decay'] < 1 or cfg['entropy_coefficient'] < 0:
        raise ValueError('Invalid policy regularization setting')
    if cfg['max_comments'] is not None and cfg['max_comments'] < 1:
        raise ValueError('The comment limit must be positive')
    if cfg['num_workers'] < 0 or cfg['folds'] < 2:
        raise ValueError('Invalid worker or fold count')
    return cfg


def prepare_splits(frames, cfg):
    extractor = None

    def classify(comments):
        nonlocal extractor
        if extractor is None:
            extractor = CommentSentiment(cfg['sentiment_model'], cfg['device'], cfg['batch_size'], cfg['offline'])
        return extractor(comments)

    prepared = {name: prepare_frame(frame, classify, cfg['use_privileged'], cfg.get('max_comments'))
                for name, frame in frames.items() if frame is not None}
    del extractor
    if str(cfg['device']).startswith('cuda'):
        torch.cuda.empty_cache()
    return prepared


def summarize(rows):
    if not rows:
        return {}
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows if row.get(key) is not None]
        result[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'n': len(values)} if values else None
    return result


def run_experiment(cfg, frames):
    from transformers import AutoTokenizer
    set_seed(cfg['seed'])
    folder = Path(cfg['output_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    if '_comments' not in frames['train']:
        frames = prepare_splits(frames, cfg)
    names = cfg.get('teacher_names', ['bert', 'roberta'])
    tokenizers = {'student': AutoTokenizer.from_pretrained(cfg['student_model'], local_files_only=cfg['offline'])}
    for name in names:
        tokenizers[name] = AutoTokenizer.from_pretrained(cfg['teacher_models'][name], local_files_only=cfg['offline'])
    teacher_val_loader = make_loader(frames['val'], tokenizers, cfg, include_comments=cfg['use_privileged'])
    teachers = {}
    for name in names:
        teacher_cfg = dict(cfg, seed=cfg['seed'] + ['bert', 'roberta'].index(name))
        set_seed(teacher_cfg['seed'])
        teacher_train_loader = make_loader(frames['train'], tokenizers, teacher_cfg, shuffle=True,
                                          include_comments=cfg['use_privileged'])
        teacher = TeacherModel(cfg['teacher_models'][name], use_privileged=cfg['use_privileged'],
                               local_files_only=cfg['offline'], comment_batch_size=cfg['comment_batch_size'])
        metadata = {'backbone': cfg['teacher_models'][name], 'use_privileged': cfg['use_privileged'],
                    'train': fingerprint(frames['train']), 'validation': fingerprint(frames['val']),
                    'seed': teacher_cfg['seed'], 'lr': cfg['teacher_lr'], 'epochs': cfg['teacher_epochs'],
                    'max_len': cfg['max_len'], 'max_comments': cfg.get('max_comments'),
                    'sentiment_model': cfg['sentiment_model']}
        teacher = train_teacher(teacher, name, teacher_cfg, teacher_train_loader, teacher_val_loader,
                                folder / 'teachers' / (name + '.pt'), metadata)
        teachers[name] = teacher
    teacher_metrics = {}
    if frames.get('test') is not None:
        teacher_test_loader = make_loader(frames['test'], tokenizers, cfg, include_comments=cfg['use_privileged'])
        teacher_metrics = {name: Evaluation(teacher, cfg, teacher_test_loader, teacher_name=name).evaluate()
                           for name, teacher in teachers.items()}
    if cfg['mode'] == 'train_teachers':
        result = {'teacher_test': teacher_metrics}
    else:
        set_seed(cfg['seed'])
        train_loader = make_loader(frames['train'], tokenizers, cfg, shuffle=True, include_comments=cfg['use_privileged'])
        student = StudentModel(cfg['student_model'], local_files_only=cfg['offline'])
        model = Model(cfg, student=student, teachers=teachers)
        val_loader = make_loader(frames['val'], tokenizers, cfg, include_comments=cfg['use_privileged'])
        trainer = Trainer(model, cfg, train_loader, val_loader)
        validation = trainer.train()
        test = None
        if frames.get('test') is not None:
            test_loader = make_loader(frames['test'], {'student': tokenizers['student']}, cfg, include_comments=False)
            test = Evaluation(model.student, cfg, test_loader).evaluate()
        result = {'validation': validation, 'test': test, 'teacher_test': teacher_metrics}
    (folder / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def run_repeated(cfg, frames):
    prepared = prepare_splits(frames, cfg)
    runs = []
    for seed in cfg['seeds']:
        current = copy.deepcopy(cfg)
        current['seed'] = seed
        current['output_dir'] = str(Path(cfg['output_dir']) / ('seed_' + str(seed)))
        runs.append({'seed': seed, **run_experiment(current, prepared)})
    result = {'runs': runs, 'test_statistics': summarize([row['test'] for row in runs if row.get('test') is not None])}
    Path(cfg['output_dir']).mkdir(parents=True, exist_ok=True)
    (Path(cfg['output_dir']) / 'repeated_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['test_statistics'], indent=2))
    return result


def run_cross_validation(cfg, frames):
    pool = pd.concat([frames['train'], frames['val']], ignore_index=True)
    prepared = prepare_splits({'train': pool, 'val': frames['val']}, cfg)['train']
    splitter = StratifiedKFold(n_splits=cfg['folds'], shuffle=True, random_state=cfg['split_seed'])
    folds = []
    for index, (train_indices, test_indices) in enumerate(splitter.split(pool, pool['label'])):
        inner_train, inner_val = train_test_split(train_indices, test_size=0.2, random_state=cfg['split_seed'],
                                                 stratify=pool.iloc[train_indices]['label'])
        current = copy.deepcopy(cfg)
        current['seed'] = cfg['seed'] + index
        current['output_dir'] = str(Path(cfg['output_dir']) / ('fold_' + str(index + 1)))
        split = {'train': prepared.iloc[inner_train].reset_index(drop=True),
                 'val': prepared.iloc[inner_val].reset_index(drop=True),
                 'test': prepared.iloc[test_indices].reset_index(drop=True)}
        check_disjoint(split)
        result = run_experiment(current, split)
        folds.append({'fold': index + 1, 'train_indices': inner_train.tolist(),
                      'validation_indices': inner_val.tolist(), 'test_indices': test_indices.tolist(), **result})
    result = {'folds': folds, 'stats': summarize([row['test'] for row in folds])}
    Path(cfg['output_dir']).mkdir(parents=True, exist_ok=True)
    (Path(cfg['output_dir']) / '5fold_results_final.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['stats'], indent=2))
    return result


def run_sensitivity(cfg, frames):
    prepared = prepare_splits({'train': frames['train'], 'val': frames['val']}, cfg)
    experiments = {'lr': [0.8e-5, 1.2e-5, 1.6e-5, 2e-5, 2.4e-5, 2.8e-5, 3.2e-5],
                   'warmup': [1, 2, 3, 4, 5, 6, 7],
                   'alpha': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                   'temperature': [1., 2., 3., 4., 5., 6., 7.]}
    results = {}
    for parameter, values in experiments.items():
        results[parameter] = []
        for value in values:
            current = copy.deepcopy(cfg)
            current['mode'] = 'train'
            current[parameter] = value
            current['output_dir'] = str(Path(cfg['output_dir']) / (parameter + '_' + str(value)))
            result = run_experiment(current, prepared)
            results[parameter].append({'value': value, 'validation': result['validation']})
    Path(cfg['output_dir']).mkdir(parents=True, exist_ok=True)
    (Path(cfg['output_dir']) / 'pure_sensitivity_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def predict(cfg):
    from transformers import AutoTokenizer
    if not cfg['checkpoint']:
        raise ValueError('--checkpoint is required for prediction')
    student = load_student(cfg['checkpoint'], cfg['device'], cfg['offline'])
    frame = pd.read_csv(cfg['data_path'] or str(Path(cfg['data_dir']) / 'test.CSV'))
    if 'label' not in frame:
        frame['label'] = 0
    frame = prepare_frame(frame, use_privileged=False)
    checkpoint = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
    cfg['max_len'] = checkpoint.get('max_len', cfg['max_len'])
    tokenizer = AutoTokenizer.from_pretrained(student.model_name, local_files_only=cfg['offline'])
    loader = make_loader(frame, {'student': tokenizer}, cfg, include_comments=False)
    probabilities = []
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, cfg['device'])
            logits = student(batch['source_ids']['student'], batch['source_masks']['student'])
            probabilities.extend(logits.softmax(dim=-1).cpu().tolist())
    values = np.asarray(probabilities)
    result = pd.DataFrame({'prediction': values.argmax(axis=1), 'probability_real': values[:, 0], 'probability_fake': values[:, 1]})
    Path(cfg['output_dir']).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(cfg['output_dir']) / 'predictions.csv', index=False)
    return result


def cli(mode='train', cross_validation=False):
    cfg = parse_config(mode, cross_validation)
    if cfg['mode'] == 'predict':
        return predict(cfg)
    frames = load_splits(cfg['data_dir'], cfg.get('data_path'), cfg['split_seed'])
    if cfg['mode'] == 'sensitivity':
        return run_sensitivity(cfg, frames)
    if cfg['mode'] == 'single_teacher':
        results = {}
        for name in ['bert', 'roberta']:
            current = copy.deepcopy(cfg)
            current['teacher_names'] = [name]
            current['output_dir'] = str(Path(cfg['output_dir']) / name)
            results[name] = run_repeated(current, frames)
        return results
    if cfg.get('cross_validation'):
        return run_cross_validation(cfg, frames)
    return run_repeated(cfg, frames)


if __name__ == '__main__':
    cli()
