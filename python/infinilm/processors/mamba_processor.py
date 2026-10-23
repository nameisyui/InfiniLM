from typing_extensions import override

from .. import _core as infinicore
from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .basic_llm_processor import BasicLLMProcessor


class MambaProcessor(BasicLLMProcessor):
    @override
    def build_model_inputs(
        self,
        scheduler_output: SchedulerOutput | StaticSchedulerOutput,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        **kwargs,
    ) -> dict:
        model_inputs = super().build_model_inputs(
            scheduler_output, temperature, top_p, top_k, **kwargs
        )

        if isinstance(scheduler_output, StaticSchedulerOutput):
            final_indices = scheduler_output.physical_rows
            # Static pools retain request row numbers and append a zero-state row.
            zero_state_index = scheduler_output.cache_capacity
        else:
            final_indices = []
            for req in scheduler_output.scheduled_requests:
                if req.mamba_cache_index is None:
                    raise RuntimeError(
                        f"Request {req.request_id} has no assigned mamba cache index"
                    )
                final_indices.append(req.mamba_cache_index)
            zero_state_index = 0

        model_inputs["mamba_init_state_indices"] = infinicore.from_list(
            [zero_state_index] * scheduler_output.num_requests
            if scheduler_output.is_prefill
            else final_indices,
            dtype=infinicore.int32,
        )
        model_inputs["mamba_final_state_indices"] = infinicore.from_list(
            final_indices, dtype=infinicore.int32
        )
        return model_inputs

    @override
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        normalized_conversation = []
        for message in conversation:
            if isinstance(message["content"], list):
                assert len(message["content"]) == 1, (
                    "Only one content item supported in list"
                )
                content_item = message["content"][0]
                assert "type" in content_item and "text" in content_item, (
                    "Content dict must have 'type' and 'text' keys"
                )
                normalized_conversation.append(
                    {"role": message["role"], "content": content_item["text"]}
                )
            else:
                normalized_conversation.append(message)

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                conversation=normalized_conversation,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )

        text = "\n".join(
            str(message.get("content", "")) for message in normalized_conversation
        )
        if tokenize:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return text
