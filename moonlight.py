# moonlight_nodes.py
import comfy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class MoonlightModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["base", "instruct"], {"default": "base"}),
                "load_local_model": ("BOOLEAN", {"default": False}),
            }, "optional": {
                "local_base_model_path": ("STRING", {"default": "moonshotai/Moonlight-16B-A3B"}),
                "local_instruct_model_path": ("STRING", {"default": "moonshotai/Moonlight-16B-A3B-Instruct"}),
            }
        }

    RETURN_TYPES = ("MODEL", "TOKENIZER")
    FUNCTION = "load_model"

    def load_model(self, model_type, load_local_model, *args, **kwargs):
        if load_local_model:
            model_paths = {
                "base": kwargs.get("local_base_model_path", "moonshotai/Moonlight-16B-A3B"),
                "instruct": kwargs.get("local_instruct_model_path", "moonshotai/Moonlight-16B-A3B-Instruct"),
            }
        else:
            model_paths = {
                "base": "moonshotai/Moonlight-16B-A3B",
                "instruct": "moonshotai/Moonlight-16B-A3B-Instruct"
            }
            path = model_paths[model_type]

        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return (model, tokenizer)


class ChatTemplateProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"default": "You are a helpful assistant provided by Moonshot-AI."}),
                "user_prompt": ("STRING", {"default": "Is 123 a prime?"}),
                "tokenizer": ("TOKENIZER",),
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL_INPUTS",)
    FUNCTION = "process"

    def process(self, system_prompt, user_prompt, tokenizer, model):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        return ({"inputs": input_ids},)


# 保留原有的PromptProcessor和TextGenerator/TextDecoder
class PromptProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "1+1=2, 1+2="}),
                "tokenizer": ("TOKENIZER",),
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL_INPUTS",)
    FUNCTION = "process"

    def process(self, prompt, tokenizer, model):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        return (inputs,)


class TextGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "inputs": ("MODEL_INPUTS",),
                "max_tokens": ("INT", {"default": 100, "min": 1, "max": 2000}),
            }
        }

    RETURN_TYPES = ("TEXT",)
    FUNCTION = "generate"

    def generate(self, model, inputs, max_tokens):
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        return (generated_ids,)


class TextDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tokenizer": ("TOKENIZER",),
                "generated_ids": ("TEXT",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "decode"

    def decode(self, tokenizer, generated_ids):
        response = tokenizer.batch_decode(generated_ids)[0]
        return (response,)


NODE_CLASS_MAPPINGS = {
    "MoonlightModelLoader": MoonlightModelLoader,
    "PromptProcessor": PromptProcessor,
    "ChatTemplateProcessor": ChatTemplateProcessor,
    "TextGenerator": TextGenerator,
    "TextDecoder": TextDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonlightModelLoader": "CustomID Model Loader",
    "PromptProcessor": "ApplyCustomIDFlux",
    "ChatTemplateProcessor": "ChatTemplateProcessor",
    "TextGenerator": "TextGenerator",
    "TextDecoder": "TextDecoder"
}
