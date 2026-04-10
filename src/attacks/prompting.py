from PIL import Image
import torch


TOKEN_TYPE_INPUT_KEYS = ("mm_token_type_ids", "token_type_ids")


def build_text_model_inputs(
    prompt_inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model_inputs = {
        "input_ids": prompt_inputs["input_ids"].to(device),
        "attention_mask": prompt_inputs["attention_mask"].to(device),
    }
    for token_type_key in TOKEN_TYPE_INPUT_KEYS:
        token_type_ids = prompt_inputs.get(token_type_key)
        if token_type_ids is not None:
            model_inputs[token_type_key] = token_type_ids.to(device)
    return model_inputs


def build_chat_prompt_inputs(
    processor,
    device: torch.device,
    prompt: str,
    dummy_image_size: tuple[int, int],
    **processor_kwargs,
) -> tuple[str, dict[str, torch.Tensor]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    dummy_image = Image.new("RGB", dummy_image_size, color="white")
    prompt_inputs = processor(
        text=[rendered_prompt],
        images=[dummy_image],
        return_tensors="pt",
        **processor_kwargs,
    )
    return rendered_prompt, build_text_model_inputs(prompt_inputs, device)


def build_teacher_forced_batch(
    tokenizer,
    prompt_model_inputs: dict[str, torch.Tensor],
    text: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    target_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and (target_ids.shape[1] == 0 or target_ids[0, -1].item() != eos_token_id):
        eos_tensor = torch.tensor([[eos_token_id]], device=device, dtype=target_ids.dtype)
        target_ids = torch.cat([target_ids, eos_tensor], dim=1)

    full_model_inputs = {
        "input_ids": torch.cat([prompt_model_inputs["input_ids"], target_ids], dim=1),
        "attention_mask": torch.cat([prompt_model_inputs["attention_mask"], torch.ones_like(target_ids)], dim=1),
    }
    for token_type_key in TOKEN_TYPE_INPUT_KEYS:
        token_type_ids = prompt_model_inputs.get(token_type_key)
        if token_type_ids is not None:
            full_model_inputs[token_type_key] = torch.cat(
                [
                    token_type_ids,
                    torch.zeros(target_ids.shape, device=device, dtype=token_type_ids.dtype),
                ],
                dim=1,
            )

    labels = full_model_inputs["input_ids"].clone()
    labels[:, : prompt_model_inputs["input_ids"].shape[1]] = -100
    return {
        "model_inputs": full_model_inputs,
        "labels": labels,
    }


def build_target_batches(
    tokenizer,
    prompt_model_inputs: dict[str, torch.Tensor],
    target_texts: list[str],
    device: torch.device,
) -> list[dict]:
    target_batches = []
    for target_text in target_texts:
        target_batch = build_teacher_forced_batch(
            tokenizer,
            prompt_model_inputs,
            target_text,
            device,
        )
        target_batches.append(
            {
                "target_text": target_text,
                **target_batch,
            }
        )

    return target_batches


def generate_greedy_text(
    model,
    processor,
    prompt_model_inputs: dict[str, torch.Tensor],
    prompt_token_count: int,
    vision_inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
) -> str:
    generated = model.generate(
        **prompt_model_inputs,
        **vision_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = generated[:, prompt_token_count:]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
