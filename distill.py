# Pytorch codes which represent the basic ideas of distillation
# with KL Divergence Loss + Temperature 
# paper: https://arxiv.org/pdf/1503.02531.pdf

"""Explanation: In knowledge distillation, the student learns from the teacher by observing the relations between different classes. For example, a 3 will possibly more mistaken as a 2 rather than a 7. By dividing the output logits by a temperature, which then given softer distribution of softmax probabilities among the classes and later helps the students to learn to generalize in a similar way as the teacher. Then, KL loss is computed to find the similarities between 2 probability distributions. To offset the dividends, the KL loss is multiplied by temperature squared. Finally, we try to balanced between the original loss (cross entropy) and the distillation loss by a factor of alpha."""

import torch.nn as nn
import torch.nn.functional as F

softmax = nn.Softmax(dim=1)
mse = nn.MSELoss
cross_entropy = F.cross_entropy

class StudentNN(nn.Module):
    def __init__(self, alpha, temperature, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        
    def kd_loss_fn(self, student_outputs, teacher_outputs):
        soft_prediction = softmax(student_outputs / self.temperature)
        soft_target = softmax(teacher_outputs / self.temperature)
        loss = mse(soft_prediction, soft_target)
        return loss
    
    def distillation_loss_fn(self, kd_loss):
        return kd_loss * self.temperature ** 2

    def student_loss_fn(student_outputs, labels):
        return cross_entropy(student_outputs, labels)

    def loss(self, student_loss, distillation_loss):
        return self.alpha * student_loss + (1 - self.alpha) * distillation_loss

