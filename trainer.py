import json
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluation import Evaluation
from model import distillation_terms
from utils import move_batch


class Trainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(model.student.parameters(), lr=config.get('lr', 2e-5))
        self.use_agent = config.get('mode', 'train') != 'no_agent' and len(model.teacher_encoders) > 1
        self.policy_optimizer = torch.optim.AdamW(model.agent.parameters(), lr=config.get('agent_lr', 1e-4)) if self.use_agent else None
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

    def train_step(self, raw_batch, epoch):
        batch = move_batch(raw_batch, self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.policy_optimizer is not None:
            self.policy_optimizer.zero_grad(set_to_none=True)
        teachers = self.model.teacher_logits(batch)
        student = self.model.student(batch['source_ids']['student'], batch['source_masks']['student'])
        active_policy = self.use_agent and epoch >= self.config.get('warmup', 1)
        if active_policy:
            action, log_probability, entropy = self.model.agent.sample(student, teachers, self.config.get('action_temperature', 1.0))
        else:
            action = student.new_full((student.size(0), len(teachers)), 1.0 / len(teachers))
        losses, classification, distillation, target = distillation_terms(student, teachers, action, batch['label'],
                                                                         self.config.get('temperature', 2.0), self.config.get('alpha', 0.4))
        student_loss = losses.mean()
        if not torch.isfinite(student_loss):
            raise FloatingPointError('Non-finite student loss')
        reward = -losses.detach()
        student_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.student.parameters(), self.config.get('max_grad_norm', 1.0))
        self.optimizer.step()
        policy_value, entropy_value = 0.0, 0.0
        if active_policy:
            with torch.no_grad():
                rho = self.config.get('baseline_decay', 0.9)
                self.model.reward_baseline.mul_(rho).add_(reward.mean() * (1 - rho))
                advantage = reward - self.model.reward_baseline
            policy_loss = -(advantage.double() * log_probability).mean() - self.config.get('entropy_coefficient', 0.02) * entropy.mean()
            if not torch.isfinite(policy_loss):
                raise FloatingPointError('Non-finite policy loss')
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.agent.parameters(), self.config.get('max_grad_norm', 1.0))
            self.policy_optimizer.step()
            policy_value, entropy_value = policy_loss.item(), entropy.mean().item()
        return {'student_loss': student_loss.item(), 'classification_loss': classification.mean().item(),
                'distillation_loss': distillation.mean().item(), 'policy_loss': policy_value,
                'reward': reward.mean().item(), 'baseline': self.model.reward_baseline.item(),
                'entropy': entropy_value, 'policy_active': active_policy}

    def train_epoch(self, epoch):
        values = [self.train_step(batch, epoch) for batch in self.train_loader]
        if not values:
            raise ValueError('Training data is empty')
        return {key: sum(row[key] for row in values) / len(values) for key in values[0]}

    def validation_objective(self, epoch):
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for raw in self.val_loader:
                batch = move_batch(raw, self.device)
                if any(name not in batch['source_ids'] for name in self.model.teacher_encoders):
                    return None
                teachers = self.model.teacher_logits(batch)
                student = self.model.student(batch['source_ids']['student'], batch['source_masks']['student'])
                if self.use_agent and epoch >= self.config.get('warmup', 1):
                    weights = self.model.agent.mean_action(student, teachers)
                else:
                    weights = student.new_full((student.size(0), len(teachers)), 1.0 / len(teachers))
                losses = distillation_terms(student, teachers, weights, batch['label'],
                                             self.config.get('temperature', 2.0), self.config.get('alpha', 0.4))[0]
                total += losses.sum().item()
                count += len(losses)
        return total / count if count else None

    def train(self):
        best_f1, best_metrics, best_state = -1.0, None, None
        for epoch in range(self.config.get('epochs', 20)):
            losses = self.train_epoch(epoch)
            metrics = Evaluation(self.model.student, self.config, self.val_loader).evaluate()
            objective = self.validation_objective(epoch)
            self.history.append({'epoch': epoch + 1, 'losses': losses, 'validation': metrics,
                                 'validation_objective': objective})
            print(f"Epoch {epoch + 1:02d} | Loss {losses['student_loss']:.4f} | Val F1 {metrics['macro_f1']:.4f}")
            if metrics['macro_f1'] > best_f1:
                best_f1, best_metrics = metrics['macro_f1'], metrics
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.student.state_dict().items()}
                torch.save({'state_dict': best_state, 'model_name': self.model.student.model_name,
                            'num_classes': self.model.student.num_classes, 'max_len': self.config.get('max_len', 128)},
                           self.output_dir / 'best_student.pt')
                torch.save({'state_dict': self.model.agent.state_dict(), 'reward_baseline': self.model.reward_baseline.detach().cpu(),
                            'config': self.config}, self.output_dir / 'best_policy.pt')
        if best_state is None:
            raise ValueError('At least one training epoch is required')
        self.model.student.load_state_dict(best_state, strict=True)
        self.model.student.eval()
        (self.output_dir / 'training_history.json').write_text(json.dumps(self.history, indent=2), encoding='utf-8')
        return best_metrics


def train_teacher(teacher, name, config, train_loader, val_loader, checkpoint_path, metadata):
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(config.get('device', 'cpu'))
    teacher.to(device)
    if checkpoint_path.exists():
        cached = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if cached.get('metadata') == metadata:
            teacher.load_state_dict(cached['state_dict'], strict=True)
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            return teacher
    for parameter in teacher.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=config.get('teacher_lr', 2e-5))
    best_f1, best_state = -1.0, None
    for epoch in range(config.get('teacher_epochs', 20)):
        teacher.train()
        for raw in train_loader:
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            logits = teacher(batch['source_ids'][name], batch['source_masks'][name],
                             batch['comment_ids'].get(name), batch['comment_masks'].get(name),
                             batch['valid_comments'], batch['sentiment'])
            loss = F.cross_entropy(logits, batch['label'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite teacher loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), config.get('max_grad_norm', 1.0))
            optimizer.step()
        metrics = Evaluation(teacher, config, val_loader, teacher_name=name).evaluate()
        print(f"Teacher {name} | Epoch {epoch + 1:02d} | Val F1 {metrics['macro_f1']:.4f}")
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            best_state = {key: value.detach().cpu().clone() for key, value in teacher.state_dict().items()}
    if best_state is None:
        raise ValueError('At least one teacher training epoch is required')
    teacher.load_state_dict(best_state, strict=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'metadata': metadata}, checkpoint_path)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return teacher
