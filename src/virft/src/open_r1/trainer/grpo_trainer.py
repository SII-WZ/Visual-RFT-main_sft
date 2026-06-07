# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tqdm import tqdm
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import copy


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        
    ):
        self._eval_metrics = defaultdict(list)
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        images = [x["image"] for x in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]
    #  #   print(pixel_values,"这是训练的")
    #     print(f"pixel_values 形状：{prompt_inputs['pixel_values'].shape}")#有问题

        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        #print(f"【真实损失值】loss: {loss.item():.8f}")

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["loss"].append(loss.item())

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        

        return loss
#这个是原来的
    # def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
    #     metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
    #     logs = {**logs, **metrics}
    #     if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
    #         super().log(logs, start_time)
    #     else:  # transformers<=4.46
    #         super().log(logs)
    #     self._metrics.clear()
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
    # 重构指标计算逻辑，保留小数点后 8 位
        metrics = {}
        for key, val in self._metrics.items():
            if len(val) > 0:  # 避免空列表除以零报错
                avg_value = sum(val) / len(val)
                # 格式化保留 8 位小数，再转为 float 类型（兼容后续日志系统）
                metrics[key] = float("{0:.8f}".format(avg_value))
            else:
                metrics[key] = 0.0  # 无数据时默认 0.0，保证程序健壮性
        
        # 合并原有 logs 和格式化后的 metrics（与原逻辑一致）
        logs = {**logs, **metrics}
        
        # 原有的版本兼容逻辑，完全保留
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        
        # 清空 metrics 缓存，准备下一轮记录（与原逻辑一致）
        self._metrics.clear()
        
        
        
    #下面是我写的

    def maybe_apply_chat_template(example, processing_class):
        return processing_class._maybe_apply_chat_template(example) if hasattr(processing_class, "_maybe_apply_chat_template") else {"prompt": example["prompt"]}

    def is_conversational(example):
        return "messages" in example or "role" in example

    def unwrap_model_for_generation(model, accelerator):
        return accelerator.unwrap_model(model)

    def apply_chat_template(example, processing_class):
        return processing_class.apply_chat_template(example, tokenize=False, add_generation_prompt=False)
 


    def fix_pixel_values_and_grid(self, prompt_inputs):
        """
        严格对齐 Qwen2-VL patch_embed 层要求（形状 [-1, 3, 2, 14, 14]）
        构建合法输入，确保元素总数能被 1176（3*2*14*14）整除
        """
        try:
            # 1. 模型内置固定参数（从 Qwen2-VL patch_embed 层提取，不可修改）
            patch_channel = 3
            patch_block = 2
            patch_size = 14
            valid_patch_factor = patch_channel * patch_block * patch_size * patch_size  # 1176，核心整除因子
            device = prompt_inputs['pixel_values'].device
            dtype = torch.bfloat16
            
            # 2. 提取原始输入，确保为 2 维张量（H, W）
            pv = prompt_inputs['pixel_values']
            if len(pv.shape) != 2:
                # 压缩多余维度，保留核心的高宽维度
                pv = pv.squeeze() if len(pv.shape) > 2 else pv
                if len(pv.shape) != 2:
                    # 兜底：重塑为 2 维，确保后续处理
                    pv = pv.view(-1, valid_patch_factor)
            
            # 3. 计算合法尺寸（确保总元素数能被 valid_patch_factor 整除）
            # 合法高宽：必须是 (patch_block * patch_size) 的整数倍（即 2*14=28 的整数倍）
            valid_side = patch_block * patch_size  # 28，最小合法边长
            # 调整输入高宽为 valid_side 的整数倍
            h, w = pv.shape
            new_h = ((h + valid_side - 1) // valid_side) * valid_side  # 向上取整到 28 的整数倍
            new_w = ((w + valid_side - 1) // valid_side) * valid_side  # 向上取整到 28 的整数倍
            
            # 4. 重塑输入为合法尺寸（补零或裁剪，优先补零避免信息丢失）
            pv_fixed = torch.zeros((new_h, new_w), device=device, dtype=dtype)
            h_min = min(h, new_h)
            w_min = min(w, new_w)
            pv_fixed[:h_min, :w_min] = pv[:h_min, :w_min].to(dtype=dtype)
            
            # 5. 验证总元素数是否合法（能被 valid_patch_factor 整除）
            total_elements = pv_fixed.numel()
            if total_elements % valid_patch_factor != 0:
                # 兜底调整：扩展高度，确保整除
                extra_h = (valid_patch_factor - (total_elements % valid_patch_factor)) // new_w
                new_h += extra_h
                pv_fixed = torch.zeros((new_h, new_w), device=device, dtype=dtype)
                pv_fixed[:h_min, :w_min] = pv[:h_min, :w_min].to(dtype=dtype)
            
            # 6. 构建合法的 image_grid_thw（与模型 patch 规则兼容）
            batch_size = 1
            num_grids = (new_h // valid_side) * (new_w // valid_side)  # 网格数 = 高宽方向网格数乘积
            grid_h = patch_block * patch_size
            grid_w = patch_block * patch_size
            new_grid = torch.tensor([[num_grids, grid_h, grid_w]], device=device, dtype=torch.long)
            new_grid = new_grid.repeat(batch_size, 1)
            
            # 7. 更新 prompt_inputs，返回合法输入
            prompt_inputs['pixel_values'] = pv_fixed
            prompt_inputs['image_grid_thw'] = new_grid
            
            # 8. 打印验证信息
            print(f"✅ 构建 Qwen2-VL 合法输入成功：")
            print(f"   - 合法尺寸：{pv_fixed.shape}（总元素数：{pv_fixed.numel()}，可被 {valid_patch_factor} 整除）")
            print(f"   - 网格参数：{new_grid[0].tolist()}")
            print(f"   - 模型 patch 因子匹配：{pv_fixed.numel() % valid_patch_factor == 0}")
            
            return prompt_inputs
        
        except Exception as e:
            print(f"❌ 尺寸对齐方法内部报错：{str(e)}")
            raise e
    # def evaluation_loop(
    #         self,
    #         dataloader,
    #         description,
    #         prediction_loss_only: Optional[bool] = None,
    #         ignore_keys: Optional[list[str]] = None,
    #         metric_key_prefix: str = "eval",
    #     ):
    #         prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
    #         ignore_keys = ignore_keys if ignore_keys is not None else []
            
    #         # 初始化评估指标容器
    #         self._eval_metrics = defaultdict(list)
    #         total_loss = 0.0
    #         total_samples = 0
            
    #         # 模型切换为评估模式
    #         model = self._wrap_model(self.model, training=False)
    #         #model.eval()

    #         if self.ref_model is not None:
    #             self.ref_model.eval()

            
    #         # ========== 新增：初始化 tqdm 进度条 ==========
    #         pbar = tqdm(dataloader, desc=f"Evaluation ({metric_key_prefix})", total=len(dataloader))
            
    #         # ========== 修改：遍历 pbar 而非 dataloader ==========
    #         for step, inputs in enumerate(pbar):
    #             print("begin")
    #             if len(inputs) == 0:
    #                 continue
                
    #             batch_size = len(inputs)
    #             total_samples += batch_size
    #             print("begin")
    #             # ========== 新增：开启混合精度推理 ==========
    #             with self.accelerator.autocast():
    #                 with torch.no_grad():  # 评估阶段禁用梯度
    #                     # -------------------------- 1. 复用compute_loss的输入预处理逻辑 --------------------------
    #                     prompts = [x["prompt"] for x in inputs]
    #                     prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
    #                     images = [x["image"] for x in inputs]
    #                     print(1)
    #                     prompt_inputs = self.processing_class(
    #                         text=prompts_text,
    #                         images=images,
    #                         return_tensors="pt",
    #                         padding=True,
    #                         padding_side="left",
    #                         add_special_tokens=False,
    #                     )
    #                     prompt_inputs = super()._prepare_inputs(prompt_inputs)

    #                     prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    #                     pixel_values = prompt_inputs["pixel_values"]
    #                     image_grid_thw = prompt_inputs["image_grid_thw"]
    #                     # print(prompt_inputs)
    #                     # print(f"input_ids 形状：{prompt_inputs['input_ids'].shape}")
    #                     # print(f"pixel_values 形状：{prompt_inputs['pixel_values'].shape}")#有问题

    #                     if self.max_prompt_length is not None:
    #                         prompt_ids = prompt_ids[:, -self.max_prompt_length :]
    #                         prompt_mask = prompt_mask[:, -self.max_prompt_length :]
    #                     print(2222)

    #                    # -------------------------- 2. 生成completions（评估阶段关闭采样） --------------------------
    #                     # eval_generation_config = copy.deepcopy(self.generation_config)
    #                     # eval_generation_config.do_sample = False  # 评估阶段禁用采样，保证结果可复现
    #                     # eval_generation_config.temperature = 1.0   # 固定温度
                        
    #                     # # ========== 核心修改1：强制设置合规生成参数，解决贪心搜索冲突 ==========
    #                     # eval_generation_config.num_return_sequences = 1  # 强制设为1，适配贪心搜索（必须）
    #                     # eval_generation_config.num_beams = 1            # 明确关闭beam search，对应贪心搜索（必须）
    #                     # eval_generation_config.early_stopping = False   # 贪心搜索关闭早停，避免额外冲突（可选）
                        
                        
    #                     eval_generation_config = GenerationConfig(
    #                         max_new_tokens=16,  # 极度缩短生成长度，最多生成 16 个 token，快速验证
    #                         do_sample=False,    # 关闭采样，使用最快的贪心搜索
    #                         num_return_sequences=1,  # 仅返回 1 个结果，避免张量复制开销
    #                         num_beams=1,        # 关闭束搜索，减少计算量
    #                         eos_token_id=self.processing_class.eos_token_id,  # 明确 EOS 标识，让模型知道何时停止
    #                         pad_token_id=self.processing_class.pad_token_id,  # 对齐 pad token，避免维度错误
    #                         max_time=60,        # 强制设置最大生成时间 60 秒，超时自动终止（Transformers 4.28+ 支持）
    #                         early_stopping=True,  # 遇到 EOS 立即停止，避免多余计算
    #                     )

                        
    #                     # ========== 核心修改2：同步self.num_generations为1，避免后续张量维度冲突 ==========
    #                     self.num_generations = 1
                        
    #                     prompt_inputs = self.fix_pixel_values_and_grid(prompt_inputs)
    #                     print(22)
    #                     with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
    
    #                         unwrapped_model.eval()
    #                     #    torch.cuda.empty_cache()
                            
    #                         # # 关键：修正 pixel_values 和 image_grid_thw，解决维度不匹配问题
    #                         # prompt_inputs = self.fix_pixel_values_and_grid(prompt_inputs)
    #                         print(11)
    #                         # 执行生成（此时已无维度不匹配问题）
    #                         prompt_completion_ids = unwrapped_model.generate(
    #                             **prompt_inputs,
    #                             generation_config=eval_generation_config
    #                           #  generation_config=self.generation_config
    #                         )

    #                         print(11)
    #                         prompt_length = prompt_ids.size(1)
    #                         completion_ids = prompt_completion_ids[:, prompt_length:]
    #                         # 修正：不再需要repeat_interleave，因为num_generations=1，张量维度一致
    #                         prompt_mask = prompt_mask  # 移除 repeat_interleave(self.num_generations, dim=0)
    #                         print(1111111)


    #                         print(11)
    #                         prompt_length = prompt_ids.size(1)
    #                         completion_ids = prompt_completion_ids[:, prompt_length:]
    #                         prompt_mask = prompt_mask
    #                         print(1111111)
    
    def evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[list[str]] = None,
            metric_key_prefix: str = "eval",
        ):
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        ignore_keys = ignore_keys if ignore_keys is not None else []
        
        # 初始化评估指标容器
        self._eval_metrics = defaultdict(list)
        total_loss = 0.0
        total_samples = 0
        
        # 模型切换为评估模式
        model = self._wrap_model(self.model, training=False)

        if self.ref_model is not None:
            self.ref_model.eval()
        
        # ========== 新增：初始化 tqdm 进度条 ==========
        pbar = tqdm(dataloader, desc=f"Evaluation ({metric_key_prefix})", total=len(dataloader))
        
        # ========== 修改：遍历 pbar 而非 dataloader ==========
        for step, inputs in enumerate(pbar):
            print("begin")
            if len(inputs) == 0:
                continue
            
            batch_size = len(inputs)
            total_samples += batch_size
            print("begin")
            
            # ========== 新增：开启混合精度推理 ==========
            with self.accelerator.autocast():
                with torch.no_grad():  # 评估阶段禁用梯度
                    # -------------------------- 1. 复用compute_loss的输入预处理逻辑 --------------------------
                    prompts = [x["prompt"] for x in inputs]
                    prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
                    images = [x["image"] for x in inputs]
                    print(1)
                    
                    # 核心优化：使用处理器原生输出，关闭自定义尺寸修正（避免位置编码错位）
                    prompt_inputs = self.processing_class(
                        text=prompts_text,
                        images=images,
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                        add_special_tokens=False,
                    )
                    prompt_inputs = super()._prepare_inputs(prompt_inputs)

                    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
                    # 移除自定义 fix_pixel_values_and_grid 调用（根源：手动修改导致位置编码错位）
                    # 直接使用处理器输出的合规张量
                    pixel_values = prompt_inputs["pixel_values"]
                    image_grid_thw = prompt_inputs["image_grid_thw"]

                    if self.max_prompt_length is not None:
                        # 优化：裁剪时保留维度一致性，避免张量形状突变
                        prompt_ids = prompt_ids[:, -self.max_prompt_length :]
                        prompt_mask = prompt_mask[:, -self.max_prompt_length :]
                        # 同步更新 prompt_inputs 中的 input_ids 和 attention_mask，保证生成时输入一致
                        prompt_inputs["input_ids"] = prompt_ids
                        prompt_inputs["attention_mask"] = prompt_mask
                    print(2222)

                    # -------------------------- 2. 生成completions（评估阶段关闭采样） --------------------------
                    eval_generation_config = GenerationConfig(
                        max_new_tokens=16,  # 极度缩短生成长度，最多生成 16 个 token，快速验证
                        do_sample=False,    # 关闭采样，使用最快的贪心搜索
                        num_return_sequences=1,  # 仅返回 1 个结果，避免张量复制开销
                        num_beams=1,        # 关闭束搜索，减少计算量
                        eos_token_id=self.processing_class.eos_token_id,  # 明确 EOS 标识，让模型知道何时停止
                        pad_token_id=self.processing_class.pad_token_id,  # 对齐 pad token，避免维度错误
                        max_time=60,        # 强制设置最大生成时间 60 秒，超时自动终止（Transformers 4.28+ 支持）
                        early_stopping=True,  # 遇到 EOS 立即停止，避免多余计算
                    )

                    # ========== 核心修改2：同步self.num_generations为1，避免后续张量维度冲突 ==========
                    self.num_generations = 1
                    
                    print(22)
                    with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                        unwrapped_model.eval()
                        
                        print(11)
                        # 执行生成（使用更新后的 prompt_inputs，保证输入合规）
                        prompt_completion_ids = unwrapped_model.generate(
                            **prompt_inputs,
                            generation_config=eval_generation_config
                        )

                        print(11)
                        prompt_length = prompt_ids.size(1)
                        # 优化：裁剪生成结果时，保证维度与批量大小匹配
                        completion_ids = prompt_completion_ids[:, prompt_length:]
                        prompt_mask = prompt_mask
                        print(1111111)

                        print(11)
                        # 重复代码移除：保留一次裁剪逻辑即可，避免冗余
                        prompt_length = prompt_ids.size(1)
                        completion_ids = prompt_completion_ids[:, prompt_length:]
                        prompt_mask = prompt_mask
                        print(1111111)
                        # -------------------------- 3. EOS token掩码处理 --------------------------
                        is_eos = completion_ids == self.processing_class.eos_token_id
                        device = self.accelerator.device
                        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
                        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
                        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
                        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

                        # -------------------------- 4. 拼接attention_mask --------------------------
                        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
                        # 修正：不再需要repeat，因为num_generations=1，张量维度一致
                        pixel_values = prompt_inputs["pixel_values"] if pixel_values is not None else None
                        image_grid_thw = prompt_inputs["image_grid_thw"] if image_grid_thw is not None else None

                        # -------------------------- 5. 计算策略模型/参考模型的逐token logps --------------------------
                        per_token_logps = self._get_per_token_logps(
                            model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                        )
                        per_token_logps = per_token_logps[:, prompt_length - 1 :]

                        if self.ref_model is not None:
                            ref_per_token_logps = self._get_per_token_logps(
                                self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                            )
                        else:
                            with self.accelerator.unwrap_model(model).disable_adapter():
                                ref_per_token_logps = self._get_per_token_logps(
                                    model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                                )
                        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

                        # -------------------------- 6. 计算KL散度 --------------------------
                        print("kl")
                        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

                        # -------------------------- 7. 计算
                        # 修正：不再需要repeat，因为num_generations=1，prompts和completions维度一致

                        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

                        # 步骤2：判断是否为对话格式，格式化 completions（if 分支内仅处理 completions，无其他无效代码）
                        if is_conversational(inputs[0]):
                            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

                        # 步骤3：定义 prompts_expanded，与 completions 维度对齐（num_generations=1，直接等于 prompts）
                        # 修正：不再需要repeat，因为num_generations=1，prompts和completions维度一致
                        prompts_expanded = prompts  # 移除多余的扩展逻辑，保持简洁

                        # 步骤4：初始化 rewards_per_func（依赖 prompts_expanded 的长度，必须放在其后）
                        rewards_per_func = torch.zeros(len(prompts_expanded), len(self.reward_funcs), device=device)
                        print("rw")
                        # 步骤5：遍历奖励函数，计算每个函数的奖励（原有逻辑，保持不变）
                        for i, (reward_func, reward_processing_class) in enumerate(
                            zip(self.reward_funcs, self.reward_processing_classes)
                        ):
                            if isinstance(reward_func, PreTrainedModel):
                                if is_conversational(inputs[0]):
                                    messages = [{"messages": p + c} for p, c in zip(prompts_expanded, completions)]
                                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                                else:
                                    texts = [p + c for p, c in zip(prompts_expanded, completions)]
                                
                                reward_inputs = reward_processing_class(
                                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                                )
                                reward_inputs = super()._prepare_inputs(reward_inputs)
                                
                                rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                            else:
                                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                                for key in reward_kwargs:
                                    for example in inputs:
                                        # 修正：不再需要extend多份，因为num_generations=1
                                        reward_kwargs[key].append(example[key])  # 移除 extend([example[key]] * self.num_generations)
                                output_reward_func = reward_func(prompts=prompts_expanded, completions=completions, **reward_kwargs)
                                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

                        # 汇总奖励
                        rewards = rewards_per_func.sum(dim=1)

                        # -------------------------- 8. 计算优势值和损失 --------------------------
                        # 修正：不再需要view/reshape，因为num_generations=1，rewards维度就是[batch_size]
                        mean_grouped_rewards = rewards  # 移除 view(-1, self.num_generations).mean(dim=1)
                        std_grouped_rewards = torch.zeros_like(rewards)  # 因为单样本，标准差为0，避免除零错误
                        
                        # 修正：不再需要repeat_interleave，维度一致
                        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

                        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
                        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
                        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

                        # -------------------------- 9. 收集评估指标 --------------------------
                        # 基础指标
                        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
                        self._eval_metrics[f"{metric_key_prefix}_completion_length"].append(completion_length)
                        self._eval_metrics[f"{metric_key_prefix}_loss"].append(loss.item())
                        
                        # 奖励相关指标
                        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
                        for i, reward_func in enumerate(self.reward_funcs):
                            if isinstance(reward_func, PreTrainedModel):
                                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                            else:
                                reward_func_name = reward_func.__name__
                            self._eval_metrics[f"{metric_key_prefix}_rewards/{reward_func_name}"].append(reward_per_func[i].item())
                        
                        self._eval_metrics[f"{metric_key_prefix}_reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
                        self._eval_metrics[f"{metric_key_prefix}_reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
                        
                        # KL散度指标
                        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
                        self._eval_metrics[f"{metric_key_prefix}_kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

                        total_loss += loss.item() * batch_size

                        # ========== 新增：更新进度条描述 ==========
                        current_loss = self._eval_metrics[f"{metric_key_prefix}_loss"][-1] if self._eval_metrics[f"{metric_key_prefix}_loss"] else 0.0
                        current_kl = self._eval_metrics[f"{metric_key_prefix}_kl"][-1] if self._eval_metrics[f"{metric_key_prefix}_kl"] else 0.0
                        pbar.set_postfix({
                            "Loss": f"{current_loss:.4f}",
                            "KL": f"{current_kl:.4f}",
                            "Step": f"{step}/{len(dataloader)}"
                        })

                        # 终止条件（如果设置了最大评估步数）
                        if self.args.max_eval_steps is not None and step >= self.args.max_eval_steps - 1:
                            break
            
            # ========== 新增：关闭进度条 ==========
            pbar.close()

            # -------------------------- 10. 计算最终平均指标 --------------------------
            avg_metrics = {}
            for key in self._eval_metrics:
                avg_metrics[key] = torch.tensor(self._eval_metrics[key]).mean().item() if self._eval_metrics[key] else 0.0
            avg_metrics[f"{metric_key_prefix}_total_samples"] = total_samples

            # -------------------------- 11. 保存指标到本地JSON文件 --------------------------
            # 确保输出目录存在
            output_dir = self.args.output_dir or "./eval_results"
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, "eval_metrics.json")
            
            # 优化：去掉 indent=4，减少 JSON 格式化耗时
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(avg_metrics, f, ensure_ascii=False)
            
            # -------------------------- 12. 返回评估结果 --------------------------
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            return (avg_loss, avg_metrics, total_samples)

    
    
    
    
    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
