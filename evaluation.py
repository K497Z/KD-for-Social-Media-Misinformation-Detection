import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from utils import move_batch


def calculate_metrics(labels, predictions, probabilities):
    if not labels:
        raise ValueError('Evaluation data is empty')
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, labels=[0, 1], zero_division=0)
    auc = float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None
    return {'accuracy': float(accuracy_score(labels, predictions)), 'AUC': auc,
            'macro_f1': float(np.mean(f1)), 'precision_macro': float(np.mean(precision)),
            'recall_macro': float(np.mean(recall)), 'f1_real': float(f1[0]), 'f1_fake': float(f1[1]),
            'precision_real': float(precision[0]), 'precision_fake': float(precision[1]),
            'recall_real': float(recall[0]), 'recall_fake': float(recall[1])}


class Evaluation:
    def __init__(self, model, config, loader, teacher_name=None):
        self.model = model
        self.device = torch.device(config.get('device', 'cpu'))
        self.loader = loader
        self.teacher_name = teacher_name

    def evaluate(self):
        network = self.model.student if hasattr(self.model, 'student') else self.model
        network.eval()
        labels, predictions, probabilities = [], [], []
        with torch.no_grad():
            for raw in self.loader:
                batch = move_batch(raw, self.device)
                if self.teacher_name is None:
                    logits = network(batch['source_ids']['student'], batch['source_masks']['student'])
                else:
                    name = self.teacher_name
                    logits = network(batch['source_ids'][name], batch['source_masks'][name],
                                     batch['comment_ids'].get(name), batch['comment_masks'].get(name),
                                     batch['valid_comments'], batch['sentiment'])
                probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
                predictions.extend(logits.argmax(dim=-1).cpu().tolist())
                labels.extend(batch['label'].cpu().tolist())
        return calculate_metrics(labels, predictions, probabilities)


def load_student(checkpoint_path, device='cpu', offline=False):
    from model import StudentModel
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    student = StudentModel(checkpoint['model_name'], checkpoint.get('num_classes', 2), local_files_only=offline)
    student.load_state_dict(checkpoint['state_dict'], strict=True)
    return student.to(device).eval()
