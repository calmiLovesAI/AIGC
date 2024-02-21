import math
import re
import torch

from typing import Union, List, Optional
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


class LongPromptWeightingAdapter:
    """
    Encode the prompt without token length limit.
    """

    def __init__(self,
                 pipeline: DiffusionPipeline,
                 max_embeddings_multiples: Optional[int] = 3):
        """
        :param pipeline: Different pipelines have different encoding methods for long prompts.
        :param max_embeddings_multiples: The max multiple length of prompt embeddings compared to the max output length of text encoder.
        """
        self.pipe = pipeline
        self.max_embeddings_multiples = max_embeddings_multiples

    def __call__(self,
                 prompt: Union[str, List[str]],
                 neg_prompt: Optional[Union[str, List[str]]] = None,
                 clip_skip: int = None):
        """
        :param prompt: The prompt or prompts to guide the image generation.
        :param neg_prompt: The prompt or prompts not to guide the image generation.
        :param clip_skip: Number of layers to be skipped from CLIP while computing the prompt embeddings.
                          A value of 1 means that the output of the pre-final layer will be used for computing the prompt embeddings.
        :return:
        """
        batch_size = 1
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        if isinstance(self.pipe, StableDiffusionPipeline):
            text_embedder = CLIPTextCustomEmbedder(pipe=self.pipe, clip_skip=clip_skip)
            cond_embeds = text_embedder(prompt, batch_size)
            if neg_prompt:
                uncond_embeds = text_embedder(neg_prompt, batch_size)
                cond_len = cond_embeds.shape[1]
                uncond_len = uncond_embeds.shape[1]
                if cond_len > uncond_len:
                    n = (cond_len - uncond_len) // self.pipe.tokenizer.model_max_length
                    uncond_embeds = torch.cat([uncond_embeds] + [text_embedder("", batch_size)] * n, dim=1)
                elif cond_len < uncond_len:
                    n = (uncond_len - cond_len) // self.pipe.tokenizer.model_max_length
                    cond_embeds = torch.cat([cond_embeds] + [text_embedder("", batch_size)] * n, dim=1)
                else:
                    pass
                return cond_embeds, uncond_embeds
            else:
                return cond_embeds, None
        elif isinstance(self.pipe, StableDiffusionXLPipeline):
            pass
        else:
            raise ValueError


class CLIPTextCustomEmbedder:
    """
    The code is derived from https://gist.github.com/takuma104/43552b8ec70b63323c57dc9c6fcb9b90
    """

    def __init__(self,
                 pipe: DiffusionPipeline,
                 clip_skip: int = 2):
        self.tokenizer = pipe.tokenizer
        vocab = self.tokenizer.get_vocab()
        self.chunk_length = pipe.tokenizer.model_max_length - 2
        self.text_encoder = pipe.text_encoder

        self.token_mults = {}
        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

        self.device = pipe.text_encoder.device
        self.clip_stop_at_last_layers = clip_skip

    def get_target_prompt_token_count(self, token_count):
        """returns the maximum number of tokens a prompt of a known length can have before it requires one more PromptChunk to be represented"""
        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize_line(self, line):
        id_end = self.tokenizer.eos_token_id
        parsed = parse_prompt_attention(line)
        tokenized = self.tokenizer(
            [text for text, _ in parsed], truncation=False,
            add_special_tokens=False)["input_ids"]

        fixes = []
        remade_tokens = []
        multipliers = []

        for tokens, (text, weight) in zip(tokenized, parsed):
            i = 0
            while i < len(tokens):
                token = tokens[i]
                remade_tokens.append(token)
                multipliers.append(weight)
                i += 1

        token_count = len(remade_tokens)
        prompt_target_length = self.get_target_prompt_token_count(token_count)
        tokens_to_add = prompt_target_length - len(remade_tokens)
        remade_tokens = remade_tokens + [id_end] * tokens_to_add
        multipliers = multipliers + [1.0] * tokens_to_add
        return remade_tokens, fixes, multipliers, token_count

    def process_text(self, texts, batch_size):
        if isinstance(texts, str):
            texts = [texts] * batch_size

        remade_batch_tokens = []
        cache = {}
        batch_multipliers = []
        for line in texts:
            if line in cache:
                remade_tokens, fixes, multipliers = cache[line]
            else:
                remade_tokens, fixes, multipliers, _ = self.tokenize_line(line)
                cache[line] = (remade_tokens, fixes, multipliers)

            remade_batch_tokens.append(remade_tokens)
            batch_multipliers.append(multipliers)

        return batch_multipliers, remade_batch_tokens

    def __call__(self, text, batch_size):
        batch_multipliers, remade_batch_tokens = self.process_text(text, batch_size)

        z = None
        i = 0
        while max(map(len, remade_batch_tokens)) != 0:
            rem_tokens = [x[self.chunk_length:] for x in remade_batch_tokens]
            rem_multipliers = [x[self.chunk_length:] for x in batch_multipliers]

            tokens = []
            multipliers = []
            for j in range(len(remade_batch_tokens)):
                if len(remade_batch_tokens[j]) > 0:
                    tokens.append(remade_batch_tokens[j][:self.chunk_length])
                    multipliers.append(batch_multipliers[j][:self.chunk_length])
                else:
                    tokens.append([self.tokenizer.eos_token_id] * self.chunk_length)
                    multipliers.append([1.0] * self.chunk_length)

            z1 = self.process_tokens(tokens, multipliers)
            z = z1 if z is None else torch.cat((z, z1), dim=-2)

            remade_batch_tokens = rem_tokens
            batch_multipliers = rem_multipliers
            i += 1

        return z

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        remade_batch_tokens = [[self.tokenizer.bos_token_id] + x[:self.chunk_length] +
                               [self.tokenizer.eos_token_id] for x in remade_batch_tokens]
        batch_multipliers = [[1.0] + x[:self.chunk_length] + [1.0] for x in batch_multipliers]

        tokens = torch.asarray(remade_batch_tokens).to(self.device)
        outputs = self.text_encoder(
            input_ids=tokens, output_hidden_states=True)

        if self.clip_stop_at_last_layers > 1:
            z = outputs.hidden_states[-self.clip_stop_at_last_layers]
            z = self.text_encoder.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state

        # restoring original mean is likely not correct, but it seems to work well
        # to prevent artifacts that happen otherwise
        batch_multipliers_of_same_length = [
            x + [1.0] * (self.chunk_length - len(x)) for x in batch_multipliers]
        batch_multipliers = torch.asarray(
            batch_multipliers_of_same_length).to(self.device)

        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape +
                                       (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z

    # def get_text_tokens(self, text):
    #     batch_multipliers, remade_batch_tokens = self.process_text(text)
    #     return [[self.tokenizer.bos_token_id] + remade_batch_tokens[0]], \
    #         [[1.0] + batch_multipliers[0]]
