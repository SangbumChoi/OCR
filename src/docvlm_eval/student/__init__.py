"""Native sub-1B document VLM student and selective initialization tools."""

from .config import ConnectorConfig, LanguageConfig, StudentConfig, TaskHeadConfig, VisionConfig
from .data import (
    BalancedGroupBatchSampler,
    DeterministicDistributedBatchSampler,
    StudentCollator,
    StudentCollatorConfig,
    StudentExample,
    UDDStudentDataset,
    student_model_inputs,
)
from .curriculum import CurriculumSchedule, CurriculumStage, planned_optimizer_steps
from .distillation import (
    DistillationConfig,
    DistillationLoss,
    NativeStudentTeacher,
    TeacherSignals,
)
from .evaluate import (
    StructuredEvalConfig,
    StructuredEvalResult,
    compare_split_summaries,
    evaluate_structured_student,
    wandb_metrics_for_split,
    write_split_comparison,
)
from .mixture import MixtureComponent, build_weighted_mixture
from .acquisition import HubComponentSpec, acquire_hub_component, materialize_component
from .teacher_targets import (
    apply_teacher_predictions,
    export_teacher_requests,
    generate_teacher_predictions,
)
from .tokenizer import DocumentTokenizer
from .pretrain import PretrainConfig, TrainingResult, train_student
from .posttrain import (
    RLVRConfig,
    RLVRResult,
    SFTConfig,
    StructuredPostTrainingDataset,
    completion_log_probs,
    group_relative_policy_loss,
    sample_completion_group,
    posttraining_prompt_batch,
    train_grpo,
    train_sft,
)
from .rewards import (
    RewardConfig,
    RewardContext,
    RewardResult,
    StructuredResponse,
    build_structured_target,
    parse_structured_response,
    score_structured_response,
)

__all__ = [
    "BalancedGroupBatchSampler",
    "ConnectorConfig",
    "CurriculumSchedule",
    "CurriculumStage",
    "DocumentTokenizer",
    "DistillationConfig",
    "DistillationLoss",
    "DeterministicDistributedBatchSampler",
    "LanguageConfig",
    "HubComponentSpec",
    "MixtureComponent",
    "NativeStudentTeacher",
    "PretrainConfig",
    "RLVRConfig",
    "RLVRResult",
    "RewardConfig",
    "RewardContext",
    "RewardResult",
    "StudentCollator",
    "StudentCollatorConfig",
    "StudentConfig",
    "StudentExample",
    "SFTConfig",
    "StructuredEvalConfig",
    "StructuredEvalResult",
    "StructuredPostTrainingDataset",
    "TaskHeadConfig",
    "TeacherSignals",
    "TrainingResult",
    "StructuredResponse",
    "UDDStudentDataset",
    "VisionConfig",
    "student_model_inputs",
    "build_structured_target",
    "build_weighted_mixture",
    "apply_teacher_predictions",
    "acquire_hub_component",
    "compare_split_summaries",
    "export_teacher_requests",
    "completion_log_probs",
    "group_relative_policy_loss",
    "generate_teacher_predictions",
    "materialize_component",
    "parse_structured_response",
    "planned_optimizer_steps",
    "sample_completion_group",
    "posttraining_prompt_batch",
    "score_structured_response",
    "train_grpo",
    "train_sft",
    "train_student",
    "evaluate_structured_student",
    "wandb_metrics_for_split",
    "write_split_comparison",
]
