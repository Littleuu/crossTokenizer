from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .dual_space_kd import DualSpaceKD
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from .universal_logit_distillation import UniversalLogitDistillation
from .min_edit_dis_kld import MinEditDisForwardKLD
from .contextual_dynamic_mapping import CDMKLD
from .multi_level_OT import MultiLevelOTDistillation
from .sedi import SEDILogitDistillation
from .sft_sedi import SFTSEDILogitDistillation
from .sft_kl import SFTKLDistillation

criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "dual_space_kd": DualSpaceKD,
    "dual_space_kd_with_cma": DualSpaceKDWithCMA,
    "universal_logit_distillation": UniversalLogitDistillation,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "contextual_dynamic_mapping": CDMKLD,
    "multi_level_OT": MultiLevelOTDistillation,
    "SEDI_distillation": SEDILogitDistillation,
    "SFT_SEDI_distillation": SFTSEDILogitDistillation,
    "SFT_KL": SFTKLDistillation,
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")