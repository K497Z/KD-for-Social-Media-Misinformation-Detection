import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical


def load_encoder(name, local_files_only=False):
    from transformers import AutoModel
    return AutoModel.from_pretrained(name, local_files_only=local_files_only)


class TeacherModel(nn.Module):
    def __init__(self, model_name, num_classes=2, use_privileged=True, encoder=None,
                 local_files_only=False, comment_batch_size=32):
        super().__init__()
        self.model_name = model_name
        self.use_privileged = use_privileged
        self.encoder = encoder if encoder is not None else load_encoder(model_name, local_files_only)
        hidden = self.encoder.config.hidden_size
        self.comment_projection = nn.Linear(hidden, hidden, bias=False)
        self.attention_vector = nn.Linear(hidden, 1, bias=False)
        self.fusion = nn.Linear(2 * hidden + 3, hidden)
        self.fc = nn.Linear(hidden, num_classes, bias=False)
        self.comment_batch_size = comment_batch_size

    def forward(self, input_ids, attention_mask, comment_ids=None, comment_mask=None,
                valid_comments=None, sentiment=None, return_attention=False):
        source = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        batch, hidden = source.shape
        pooled = source.new_zeros(batch, hidden)
        emotion = source.new_zeros(batch, 3)
        attention = source.new_zeros(batch, 1)
        if self.use_privileged and comment_ids is not None:
            count, length = comment_ids.shape[1:]
            valid = valid_comments.bool() if valid_comments is not None else comment_mask.any(dim=-1)
            flat_valid = valid.reshape(-1)
            features = source.new_zeros(batch * count, hidden)
            if flat_valid.any():
                ids = comment_ids.reshape(-1, length)[flat_valid]
                masks = comment_mask.reshape(-1, length)[flat_valid]
                encoded = []
                for start in range(0, ids.size(0), self.comment_batch_size):
                    encoded.append(self.encoder(input_ids=ids[start:start + self.comment_batch_size],
                                                attention_mask=masks[start:start + self.comment_batch_size]).last_hidden_state[:, 0])
                features[flat_valid] = torch.cat(encoded, dim=0)
            features = features.reshape(batch, count, hidden)
            scores = self.attention_vector(torch.tanh(self.comment_projection(features))).squeeze(-1)
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            attention = F.softmax(scores, dim=1) * valid.to(scores.dtype)
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
            pooled = (attention.unsqueeze(-1) * features).sum(dim=1)
            if sentiment is not None:
                if sentiment.shape != (batch, 3):
                    raise ValueError('Sentiment features must have shape [batch, 3]')
                emotion = sentiment.to(source.dtype)
        logits = self.fc(torch.tanh(self.fusion(torch.cat([source, pooled, emotion], dim=-1))))
        return (logits, attention) if return_attention else logits


class StudentModel(nn.Module):
    def __init__(self, model_name, num_classes=2, encoder=None, local_files_only=False):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.encoder = encoder if encoder is not None else load_encoder(model_name, local_files_only)
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes, bias=False)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.fc(hidden)


class AgentPolicy(nn.Module):
    def __init__(self, num_teachers=2, num_classes=2, hidden_size=128):
        super().__init__()
        self.num_teachers = num_teachers
        self.net = nn.Sequential(nn.Linear((1 + 2 * num_teachers) * num_classes, hidden_size),
                                 nn.ReLU(), nn.Linear(hidden_size, num_teachers))

    def forward(self, student_logits, teacher_logits):
        student = student_logits.detach()
        teachers = [logits.detach() for logits in teacher_logits]
        differences = [(student - logits).abs() for logits in teachers]
        return self.net(torch.cat([student] + teachers + differences, dim=-1))

    def sample(self, student_logits, teacher_logits, temperature=1.0):
        if temperature <= 0:
            raise ValueError('Action temperature must be positive')
        logits = self(student_logits, teacher_logits).double()
        distribution = RelaxedOneHotCategorical(logits.new_tensor(temperature), logits=logits)
        action = distribution.rsample()
        log_probability = distribution.log_prob(action.detach())
        entropy = -distribution.log_prob(action)
        return action, log_probability, entropy

    def mean_action(self, student_logits, teacher_logits):
        return F.softmax(self(student_logits, teacher_logits), dim=-1)


def distillation_terms(student_logits, teacher_logits, weights, labels, temperature=2.0, alpha=0.4):
    if temperature <= 0 or not 0 <= alpha <= 1:
        raise ValueError('Invalid distillation temperature or weight')
    teacher_probabilities = torch.stack([F.softmax(logits.detach() / temperature, dim=-1)
                                         for logits in teacher_logits], dim=1)
    target = (weights.detach().to(teacher_probabilities.dtype).unsqueeze(-1) * teacher_probabilities).sum(dim=1)
    classification = F.cross_entropy(student_logits, labels, reduction='none')
    distillation = F.kl_div(F.log_softmax(student_logits / temperature, dim=-1),
                            target, reduction='none').sum(dim=-1) * temperature ** 2
    losses = (1 - alpha) * classification + alpha * distillation
    return losses, classification, distillation, target


class Model(nn.Module):
    def __init__(self, cfg, student=None, teachers=None):
        super().__init__()
        self.cfg = cfg
        self.student = student if student is not None else StudentModel(cfg['student_model'],
                                                                        local_files_only=cfg.get('offline', False))
        if teachers is None:
            teachers = {name: TeacherModel(cfg['teacher_models'][name], use_privileged=cfg.get('use_privileged', True),
                                           local_files_only=cfg.get('offline', False),
                                           comment_batch_size=cfg.get('comment_batch_size', 32))
                        for name in cfg.get('teacher_names', ['bert', 'roberta'])}
        self.teacher_encoders = nn.ModuleDict(teachers)
        self.agent = AgentPolicy(len(teachers))
        self.register_buffer('reward_baseline', torch.tensor(0.0))
        self.freeze_teachers()

    def freeze_teachers(self):
        for teacher in self.teacher_encoders.values():
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None

    def train(self, mode=True):
        super().train(mode)
        self.freeze_teachers()
        return self

    def forward(self, input_ids, attention_mask, k_ids=None, k_mask=None, return_feat=False):
        logits = self.student(input_ids, attention_mask)
        return {'logits': logits}

    def teacher_logits(self, batch):
        values = []
        with torch.no_grad():
            for name, teacher in self.teacher_encoders.items():
                values.append(teacher(batch['source_ids'][name], batch['source_masks'][name],
                                      batch['comment_ids'].get(name), batch['comment_masks'].get(name),
                                      batch['valid_comments'], batch['sentiment']))
        return values
