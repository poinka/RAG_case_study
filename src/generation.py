import math
import time
from typing import Optional, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_generator(model_name: str, device: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device is None:
        device = get_best_device()

    # On CUDA we can safely use float16.
    # On MPS float32 is usually more stable for small experiments.
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device != "cuda":
        model = model.to(device)

    # Keep generation deterministic by default.
    # This also avoids accidental sampling settings from model.generation_config.
    model.generation_config.do_sample = False

    model.eval()
    return tokenizer, model, device


def build_parametric_prompt(question: str) -> str:
    return (
        "Answer the question concisely. "
        "If you do not know the answer, answer: I don't know.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_rag_prompt(question: str, context: str) -> str:
    return (
        "Use only the provided context to answer the question. "
        "If the answer is not supported by the context, answer: I don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


@torch.inference_mode()
def generate_answer(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 48,
    temperature: float = 0.0,
) -> tuple[str, float]:
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3072,
    ).to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if temperature and temperature > 0:
        # Sampling mode: not recommended for final evaluation,
        # but useful if you want qualitative exploration.
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 50,
            }
        )
    else:
        # Deterministic greedy decoding: better for reproducible evaluation.
        gen_kwargs.update(
            {
                "do_sample": False,
            }
        )

    start = time.perf_counter()

    output_ids = model.generate(
        **inputs,
        **gen_kwargs,
    )

    latency = time.perf_counter() - start

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Keep only the first line because HotpotQA answers are usually short.
    text = text.split("\n")[0].strip()

    return text, latency


@torch.inference_mode()
def answer_loss_and_perplexity(
    tokenizer,
    model,
    prompt: str,
    answer: str,
    max_length: int = 3072,
) -> Dict[str, float]:
    """
    Conditional negative log-likelihood and perplexity of gold answer tokens:

        loss = NLL(answer | prompt)
        ppl  = exp(loss)

    The prompt tokens are masked with -100 labels, so the loss is computed
    only on answer tokens.

    Important:
    Mean PPL can be dominated by extreme outliers. For reporting, prefer:
    - mean answer_loss
    - median answer_ppl
    - p90/p95 answer_ppl
    """
    device = next(model.parameters()).device

    answer = str(answer)

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"][0]

    # Leading space matters for causal LM tokenization.
    answer_ids = tokenizer(
        " " + answer,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    # If the sequence is too long, truncate the prompt but keep the answer.
    if prompt_ids.shape[0] + answer_ids.shape[0] > max_length:
        keep_prompt_len = max_length - answer_ids.shape[0]

        if keep_prompt_len <= 0:
            return {
                "answer_loss": math.nan,
                "answer_ppl": math.nan,
                "answer_n_tokens": 0,
            }

        prompt_ids = prompt_ids[-keep_prompt_len:]

    input_ids = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[0]] = -100

    n_answer_tokens = int((labels != -100).sum().item())

    if n_answer_tokens == 0:
        return {
            "answer_loss": math.nan,
            "answer_ppl": math.nan,
            "answer_n_tokens": 0,
        }

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = float(outputs.loss.detach().float().cpu().item())

    # Avoid numerical overflow. Very high PPL is still meaningful as "bad",
    # but exact exp(loss) is not important above this scale.
    if math.isnan(loss):
        ppl = math.nan
    elif loss > 50:
        ppl = float("inf")
    else:
        ppl = float(math.exp(loss))

    return {
        "answer_loss": loss,
        "answer_ppl": ppl,
        "answer_n_tokens": n_answer_tokens,
    }


def answer_perplexity(
    tokenizer,
    model,
    prompt: str,
    answer: str,
    max_length: int = 3072,
) -> float:
    """
    Backward-compatible wrapper.
    Prefer answer_loss_and_perplexity() in new notebooks.
    """
    return answer_loss_and_perplexity(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        answer=answer,
        max_length=max_length,
    )["answer_ppl"]


def estimate_model_size_mb(model) -> float:
    total = 0

    for p in model.parameters():
        total += p.numel() * p.element_size()

    return total / (1024**2)