import torch
from .cross_entropy_loss import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn
import inspect

def normalize(value):
    means = value.mean(dim=-1, keepdim=True)
    stds = value.std(dim=-1, keepdim=True)
    z_score_normalized_student = (value)/ (stds+0.0001)
    return z_score_normalized_student

def KL_wo(y_s, y_t,T=1):
    p_s = F.log_softmax(y_s/T, dim=-1)
    p_t = F.softmax(y_t/T, dim=-1)
    loss = -torch.sum(p_t * p_s, dim=-1).mean()
    return loss

class Sinkhorn_seq(nn.Module):
    def __init__(self, T=2):
        super(Sinkhorn_seq, self).__init__()
        self.T = 2   
    def sinkhorn_normalized(self,x, n_iters=20):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(self,x, y, epsilon=0.1, n_iters=10):
        Wxy = torch.cdist(x, y, p=1)  
        K = torch.exp(-Wxy / epsilon)  
        P = self.sinkhorn_normalized(K, n_iters)  
        return torch.sum(P * Wxy)  
    def forward(self, y_s, y_t):
        softmax = nn.Softmax(dim=-1)
        p_s = softmax(y_s/self.T)
        p_t = softmax(y_t/self.T)
        emd_loss = 0
        for i in range(p_s.shape[0]):
            emd_loss = 0.001*self.sinkhorn_loss(x=p_s[i],y=p_t[i])
        return emd_loss

def greedy_algorithm_adjust_s(t, s):
    batch_size, T, k = t.shape
    _, n, _ = s.shape
    
    # Initialize the adjusted source tensor
    s_adjusted = torch.zeros_like(t)
    
    for b in range(batch_size):
        # Initialize set of available source indices for each batch
        available_indices = list(range(n))
        
        for i in range(T):
            C_min = float('inf')
            j_star = -1
            
            for j in available_indices:
                # Compute cost as the sum of absolute differences for each batch
                C = torch.sum(torch.abs(t[b,:,i] - s[b,:,j]))
                
                if C < C_min:
                    C_min = C
                    j_star = j
            
            # Assign the best matching source vector to the adjusted tensor
            s_adjusted[b,:,i] = s[b,:,j_star]
            
            # Remove the selected index from available indices
            available_indices.remove(j_star)

    return s_adjusted

def improved_sort(value):
    sums = value.sum(dim=(0, 1))
    sorted_indices = torch.argsort(sums, descending=True)
    sorted_values = value[:, :, sorted_indices]
    return sorted_values

class MultiLevelOTDistillation(CrossEntropyLoss):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        self.kd_rate = args.kd_rate
        self.skip_student_eos = False
        self.skip_teacher_eos = False
        self.ignore_index = -100
        self.f = 1
        self.student_temperature = 1
        self.teacher_temperature = 1
    
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        
        if "position_ids" in inspect.signature(model.forward).parameters:
            outputs = model(
                input_ids=input_data["input_ids"],
                attention_mask=input_data["attention_mask"],
                position_ids=input_data.get("position_ids", None),
                output_hidden_states=True
            )
        else:
            outputs = model(
                input_ids=input_data["input_ids"],
                attention_mask=input_data["attention_mask"],
                output_hidden_states=True
            )
        logits = outputs.logits
        log = {}
        ce_loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]
        log["nll_loss"] = ce_loss
        
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True
            )
        
        kd_loss, log = self.compute_multi_level_OT_distillation_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        log["kd_loss"] = kd_loss

        loss = ce_loss + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"], 
        )
        log["accuracy"] = accuracy


        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss / batch_denom, logging_output

    def compute_multi_level_OT_distillation_loss(
        self, outputs, teacher_outputs, output_data, distiller, log
    ):
        student_target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        student_logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        # Get answer first token and answer size
        student_answer_index, student_answer_size = self.__get_start_and_size_answers(
            student_target)
        teacher_answer_index, teacher_answer_size = self.__get_start_and_size_answers(
            teacher_target)
        # Avoid eos token, if needed
        if self.skip_student_eos: student_answer_size = [size-1 for size in student_answer_size]
        if self.skip_teacher_eos: teacher_answer_size = [size-1 for size in teacher_answer_size]
        
        student = normalize(student_logits)      
        teacher = normalize(teacher_logits)
        
        # Align answer first token, pad to right and compute softmax
        for i in range(student.size(0)):
            shift = student_answer_index[i]
            size = student_answer_size[i]
            end_shift = shift+size
            student[i] = torch.cat((
                torch.nn.functional.softmax(student[i, shift:end_shift, :]/self.student_temperature, dim=-1),
                torch.zeros_like(student[i, :(student.size(1)-size), :])), dim=0
            )
            
        for i in range(teacher.size(0)):
            shift = teacher_answer_index[i]
            size = teacher_answer_size[i]
            end_shift = shift+size
            teacher[i] = torch.cat((
               torch.nn.functional.softmax(teacher[i, shift:end_shift, :]/self.teacher_temperature, dim=-1),
               torch.zeros_like(teacher[i, :(teacher.size(1)-size), :])), dim=0
            )
            
        # Cut to max answer length
        mex_length = max(max(student_answer_size), max(teacher_answer_size))

        student = student[:, :mex_length, :]
        teacher = teacher[:, :mex_length, :]
        
        sinkorn_loss = Sinkhorn_seq()

        # # Sort in descending order to align probabilities
        student = student.sort(dim=-1, descending=True).values
        teacher = teacher.sort(dim=-1, descending=True).values
        teacher = improved_sort(teacher)
        teacher = teacher[:,:,:50]
        if self.f == 1:
            student = improved_sort(student)
            student = student[:,:,:50]
        elif self.f == 2:
            student = greedy_algorithm_adjust_s(teacher,student)

        # Pad to get same vocabulary size
        diff_size = student.size(2) - teacher.size(2)
        if diff_size > 0:
            teacher = F.pad(teacher, (0, diff_size), value=0)
        elif diff_size < 0:
            student = F.pad(student, (0, abs(diff_size)), value=0)

        distillation_loss = torch.zeros(student.size(0), device=student.device) 
        for i in range(student.size(0)):
            size = min(student_answer_size[i], teacher_answer_size[i])
            distillation_loss[i] = abs(student[i][:size] - teacher[i][:size]).sum(-1).mean(-1) 

        distillation_loss = distillation_loss + KL_wo(teacher,student) # *0.1
        distillation_loss = distillation_loss.mean() + sinkorn_loss(teacher.float(),student.float()) # *0.1

        return distillation_loss, log

    def __get_start_and_size_answers(self, answer_tensors):
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            is_value = answer.eq(self.ignore_index)
            answers_size.append(len(answer) - int(is_value.sum()))
            indices = is_value.nonzero(as_tuple=True)[0]
            if len(indices) == 0 or indices[0] != 0:
                answers_index.append(0)
            else:
                diff_indices = indices[1:] - indices[:-1]
                break_index = (diff_indices != 1).nonzero()
                length = (break_index[0].item() +
                          1) if len(break_index) > 0 else len(indices)
                answers_index.append(length-1)
        return answers_index, answers_size
    