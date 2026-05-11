import gc
import logging
import resource

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

VALID_SIZES = ("70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b")

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class PythiaModel:
    def __init__(self, size: str, revision: str = "main", dtype: str = "float16"):
        if size not in VALID_SIZES:
            raise ValueError(f"Invalid size '{size}'. Must be one of: {', '.join(VALID_SIZES)}")
        if dtype not in DTYPE_MAP:
            raise ValueError(f"Invalid dtype '{dtype}'. Must be one of: {', '.join(DTYPE_MAP)}")

        self.size = size
        self.revision = revision

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        torch_dtype = DTYPE_MAP[dtype]
        if self.device.type == "mps" and torch_dtype == torch.bfloat16:
            logger.warning("bfloat16 not reliably supported on MPS, falling back to float16.")
            torch_dtype = torch.float16

        model_name = f"EleutherAI/pythia-{size}"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()

        self._log_memory()

    def _log_memory(self):
        if self.device.type == "mps":
            allocated = torch.mps.current_allocated_memory()
            logger.info(f"MPS memory allocated: {allocated / 1024**2:.1f} MB")
        else:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            logger.info(f"Process RSS: {rss / 1024**2:.1f} MB")

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
        new_tokens = outputs[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.inference_mode()
    def loglikelihood(self, context: str, continuation: str) -> float:
        return self._loglikelihood_single(context, continuation)

    @torch.inference_mode()
    def loglikelihood_batch(
        self, pairs: list[tuple[str, str]], batch_size: int = 8
    ) -> list[float]:
        results = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            for context, continuation in chunk:
                results.append(self._loglikelihood_single(context, continuation))
        return results

    @torch.inference_mode()
    def _loglikelihood_single(self, context: str, continuation: str) -> float:
        context_ids = self.tokenizer(context, return_tensors="pt")["input_ids"]
        ctx_len = context_ids.shape[1]

        full_ids = self.tokenizer(
            context + continuation, return_tensors="pt"
        )["input_ids"].to(self.device)

        logits = self.model(full_ids).logits.float()
        log_probs = F.log_softmax(logits, dim=-1)

        cont_log_probs = log_probs[0, ctx_len - 1 : -1, :]
        target_tokens = full_ids[0, ctx_len:]

        token_log_probs = cont_log_probs[torch.arange(target_tokens.shape[0]), target_tokens]
        return token_log_probs.sum().item()

    def unload(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        if self.device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
        logger.info("Model unloaded and cache cleared.")
