"""Manual Torch inference path for Qwen3-ASR.

The official transformers backend is correct but routes decoding through
`GenerationMixin.generate()`. For single-stream greedy ASR we can do less work:
prepare the prompt/audio tensors once, prefill the decoder once, then feed one
token at a time with the returned KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .audio import is_batch_audio
from .model import ASRResult

_EOS_TOKEN_IDS = frozenset({151645, 151643})


@dataclass
class _Chunk:
    orig_index: int
    wav: Any
    offset_sec: float


class TorchQwenASRBackend:
    """Fast greedy Torch backend backed by the official Qwen3-ASR model."""

    def __init__(
        self,
        official_model: Any,
        *,
        max_new_tokens: int = 256,
        max_inference_batch_size: int = 32,
        use_cuda_graph: bool = True,
        cuda_graph_stride: int = 128,
        use_torch_compile: bool = True,
        attn_implementation: str | None = None,
    ) -> None:
        self.official_model = official_model
        self.model = official_model.model
        self.processor = official_model.processor
        self.max_new_tokens = int(max_new_tokens)
        self.max_inference_batch_size = int(max_inference_batch_size)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.cuda_graph_stride = max(1, int(cuda_graph_stride))
        self.use_torch_compile = bool(use_torch_compile)
        self.attn_implementation = attn_implementation
        self.device = getattr(official_model, "device", None)
        self.dtype = getattr(official_model, "dtype", None)
        self.eos_token_ids = _resolve_eos_token_ids(self.model)
        self._graph: _DecoderGraph | None = None
        self._graph_failed = False
        if attn_implementation:
            _set_attention_implementation(self.model, attn_implementation)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        max_new_tokens: int = 256,
        max_inference_batch_size: int = 32,
        use_cuda_graph: bool = True,
        cuda_graph_stride: int = 128,
        use_torch_compile: bool = True,
        attn_implementation: str | None = None,
        forced_aligner: str | None = None,
        forced_aligner_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "TorchQwenASRBackend":
        from qwen_asr import Qwen3ASRModel

        official_model = Qwen3ASRModel.from_pretrained(
            pretrained_model_name_or_path,
            forced_aligner=forced_aligner,
            forced_aligner_kwargs=forced_aligner_kwargs,
            max_new_tokens=max_new_tokens,
            max_inference_batch_size=max_inference_batch_size,
            **kwargs,
        )
        return cls(
            official_model,
            max_new_tokens=max_new_tokens,
            max_inference_batch_size=max_inference_batch_size,
            use_cuda_graph=use_cuda_graph,
            cuda_graph_stride=cuda_graph_stride,
            use_torch_compile=use_torch_compile,
            attn_implementation=attn_implementation,
        )

    def get_supported_languages(self) -> list[str]:
        return self.official_model.get_supported_languages()

    def transcribe(
        self,
        audio: Any,
        context: str | list[str] = "",
        language: str | list[str | None] | None = None,
        return_time_stamps: bool = False,
    ) -> list[ASRResult]:
        """Transcribe using manual greedy decode.

        Timestamp output still falls back to the official implementation because
        forced alignment is orthogonal to the decoder fast path.
        """
        if return_time_stamps:
            return self._transcribe_official(
                audio=audio,
                context=context,
                language=language,
                return_time_stamps=True,
            )

        if is_batch_audio(audio) and self.max_inference_batch_size != 1:
            return self._transcribe_official(
                audio=audio,
                context=context,
                language=language,
                return_time_stamps=False,
            )

        wavs = _normalize_audios(audio)
        count = len(wavs)
        contexts = _broadcast_context(context, count)
        languages = _normalize_languages(language, count)

        chunks: list[_Chunk] = []
        for index, wav in enumerate(wavs):
            for chunk_wav, offset_sec in _split_audio_into_chunks(wav):
                chunks.append(_Chunk(orig_index=index, wav=chunk_wav, offset_sec=offset_sec))

        if len(chunks) > 1 and self.max_inference_batch_size != 1:
            return self._transcribe_official(
                audio=audio,
                context=context,
                language=language,
                return_time_stamps=False,
            )

        chunk_outputs: list[tuple[int, str, str]] = []
        for chunk in chunks:
            raw = self._decode_one(
                wav=chunk.wav,
                context=contexts[chunk.orig_index],
                language=languages[chunk.orig_index],
            )
            parsed_language, parsed_text = _parse_asr_output(
                raw,
                user_language=languages[chunk.orig_index],
            )
            chunk_outputs.append((chunk.orig_index, parsed_language, parsed_text))

        per_audio_langs: list[list[str]] = [[] for _ in range(count)]
        per_audio_texts: list[list[str]] = [[] for _ in range(count)]
        for index, parsed_language, parsed_text in chunk_outputs:
            per_audio_langs[index].append(parsed_language)
            per_audio_texts[index].append(parsed_text)

        return [
            ASRResult(
                text="".join(text for text in per_audio_texts[index] if text is not None),
                language=_merge_languages(per_audio_langs[index]),
            )
            for index in range(count)
        ]

    def _transcribe_official(
        self,
        *,
        audio: Any,
        context: str | list[str],
        language: str | list[str | None] | None,
        return_time_stamps: bool,
    ) -> list[ASRResult]:
        previous_attn = self.attn_implementation
        _set_attention_implementation(self.model, "sdpa")
        try:
            return [
                ASRResult(text=r.text, language=r.language, time_stamps=r.time_stamps)
                for r in self.official_model.transcribe(
                    audio=audio,
                    context=context,
                    language=language,
                    return_time_stamps=return_time_stamps,
                )
            ]
        finally:
            if previous_attn:
                _set_attention_implementation(self.model, previous_attn)

    def _decode_one(self, *, wav: Any, context: str, language: str | None) -> str:
        import torch

        prompt = self.official_model._build_text_prompt(context=context, force_language=language)
        inputs = self.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device, self.model.dtype)

        with torch.inference_mode():
            generated = None
            if self._can_use_cuda_graph(torch):
                try:
                    generated = self._decode_with_graph(inputs, wav)
                except Exception:
                    self._graph_failed = True
                    generated = None

            if generated is None:
                last_logits, past_key_values, _ = self._prefill(inputs)
                generated = self._decode_dynamic(
                    last_logits, past_key_values, inputs["attention_mask"]
                )

        if not generated:
            return ""

        token_tensor = torch.tensor([generated], dtype=torch.long)
        return self.processor.batch_decode(
            token_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _can_use_cuda_graph(self, torch: Any) -> bool:
        return (
            self.use_cuda_graph
            and not self._graph_failed
            and torch.cuda.is_available()
            and str(self.model.device).startswith("cuda")
        )

    def _prefill(
        self,
        inputs: Any,
        *,
        past_key_values: Any = None,
        cache_position: Any = None,
    ) -> tuple[Any, Any, Any]:
        """Prefill the decoder, computing lm_head only for the last position.

        The thinker's forward runs lm_head over every prompt position (vocab
        ~151k) even though greedy decode only needs the final one, so this
        replicates its prompt preparation and calls the text model directly.
        Returns (last_position_logits, past_key_values, rope_deltas).
        """
        thinker = self.model.thinker
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        inputs_embeds = thinker.get_input_embeddings()(input_ids)
        audio_features = thinker.get_audio_features(
            inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
        ).to(inputs_embeds.device, inputs_embeds.dtype)
        audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
        position_ids, rope_deltas = thinker.get_rope_index(attention_mask)
        rope_deltas = rope_deltas - delta0
        # The per-step thinker() calls in the dynamic path read this state.
        thinker.rope_deltas = rope_deltas

        outputs = thinker.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            cache_position=cache_position,
        )
        last_logits = thinker.lm_head(outputs.last_hidden_state[:, -1:, :])
        return last_logits, outputs.past_key_values, rope_deltas

    def _decode_dynamic(self, last_logits: Any, past_key_values: Any, attention_mask: Any) -> list[int]:
        import torch

        thinker = self.model.thinker
        current_token = torch.argmax(last_logits[:, -1, :], dim=-1, keepdim=True)
        prompt_len = attention_mask.shape[1]
        device = attention_mask.device
        full_mask = torch.ones(
            (attention_mask.shape[0], prompt_len + self.max_new_tokens),
            dtype=attention_mask.dtype,
            device=device,
        )
        full_mask[:, :prompt_len] = attention_mask
        positions = torch.arange(prompt_len, prompt_len + self.max_new_tokens, device=device)
        generated: list[int] = []

        for step in range(self.max_new_tokens):
            token_id = int(current_token.item())
            if token_id in self.eos_token_ids:
                break
            generated.append(token_id)

            outputs = thinker(
                input_ids=current_token,
                attention_mask=full_mask[:, : prompt_len + step + 1],
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=positions[step : step + 1],
            )
            past_key_values = outputs.past_key_values
            current_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        return generated

    def _estimate_graph_max_new_tokens(self, wav: Any) -> int:
        try:
            duration_sec = max(0.0, float(len(wav)) / 16000.0)
        except Exception:
            return self.max_new_tokens
        estimated = int(duration_sec * 8.0) + 32
        estimated = max(32, estimated)
        return min(self.max_new_tokens, estimated)

    def _decode_with_graph(self, inputs: Any, wav: Any) -> list[int] | None:
        """Prefill into the graph's static cache, then replay the decode graph.

        Returns None when the duration-based token budget was exhausted so the
        caller can redo the request with the unbounded dynamic decoder.
        """
        import torch

        thinker = self.model.thinker
        input_len = int(inputs["input_ids"].shape[1])
        graph_max_new_tokens = self._estimate_graph_max_new_tokens(wav)
        required_cache_len = input_len + graph_max_new_tokens + 4

        graph = self._graph
        if graph is None or graph.max_cache_len < required_cache_len:
            graph = _DecoderGraph(
                thinker=thinker,
                max_cache_len=_round_up(required_cache_len, self.cuda_graph_stride),
                dtype=self.model.dtype,
                device=str(self.model.device),
                eos_token_ids=self.eos_token_ids,
                compile_decode=self.use_torch_compile,
            )
            self._graph = graph

        last_logits, _, rope_deltas = self._prefill(
            inputs,
            past_key_values=graph.static_cache,
            cache_position=torch.arange(input_len, device=self.model.device),
        )

        if not graph.captured:
            graph.capture(last_logits, rope_deltas, input_len=input_len)
        generated = graph.run(
            last_logits, rope_deltas, input_len=input_len, max_new_tokens=graph_max_new_tokens
        )
        if len(generated) >= graph_max_new_tokens < self.max_new_tokens:
            return None
        return generated


class _DecoderGraph:
    """Self-feeding CUDA graph for one-token Qwen3-ASR text decode.

    The captured graph contains the full token feedback loop: position/mask
    computation from `cache_position`, embedding lookup, decoder forward,
    argmax, the write of the new token into a ring buffer, and the advance of
    `cache_position`. Replays therefore need no host work at all; the host only
    syncs once every `eos_check_interval` replays to scan for EOS. Replays past
    EOS within a batch are wasted GPU work, and on GB10 one replay costs far
    more than the ~60us host readback, so the default interval is 1.

    Prefill happens directly into `static_cache` (the caller passes it as
    `past_key_values`), so no KV copy is needed between prefill and decode.
    Because `input_len` only enters the graph through tensor *values*, one
    captured graph serves any prompt length that fits in `max_cache_len`.
    """

    def __init__(
        self,
        *,
        thinker: Any,
        max_cache_len: int,
        dtype: Any,
        device: str,
        eos_token_ids: frozenset[int] = _EOS_TOKEN_IDS,
        eos_check_interval: int = 1,
        compile_decode: bool = True,
    ) -> None:
        import torch
        from transformers import StaticCache

        self.thinker = thinker
        self.eos_token_ids = eos_token_ids
        self.compile_decode = bool(compile_decode)
        self.text_model = thinker.model
        self.max_cache_len = int(max_cache_len)
        self.dtype = dtype
        self.device = device
        self.eos_check_interval = max(1, int(eos_check_interval))
        device_index = torch.device(device).index
        self.device_index = device_index if device_index is not None else torch.cuda.current_device()

        self.static_cache = StaticCache(config=self.text_model.config, max_cache_len=self.max_cache_len)
        self.input_id_buf = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self.rope_delta = torch.zeros(1, dtype=torch.long, device=device)
        self.token_ring = torch.zeros(self.max_cache_len, dtype=torch.long, device=device)
        self.step_buf = torch.zeros(1, dtype=torch.long, device=device)
        self.kv_arange = torch.arange(self.max_cache_len, device=device).view(1, 1, 1, -1)
        self.mask_keep = torch.zeros((), dtype=dtype, device=device)
        self.mask_drop = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device)
        self.graph = None
        self.captured = False

    def _decode_step(self) -> None:
        import torch

        position_ids = (self.cache_position + self.rope_delta).view(1, 1, 1).expand(3, 1, 1)
        attn_mask = torch.where(
            self.kv_arange <= self.cache_position.view(1, 1, 1, 1),
            self.mask_keep,
            self.mask_drop,
        )
        embeds = self.thinker.get_input_embeddings()(self.input_id_buf)
        outputs = self.text_model(
            inputs_embeds=embeds,
            attention_mask=attn_mask,
            past_key_values=self.static_cache,
            use_cache=True,
            cache_position=self.cache_position,
            position_ids=position_ids,
        )
        logits = self.thinker.lm_head(outputs.last_hidden_state[:, -1, :])
        next_token = torch.argmax(logits, dim=-1)
        self.token_ring.index_copy_(0, self.step_buf, next_token)
        self.input_id_buf.copy_(next_token.view(1, 1))
        self.cache_position.add_(1)
        self.step_buf.add_(1)

    def _seed(self, last_logits: Any, rope_deltas: Any, input_len: int) -> Any:
        import torch

        self.rope_delta.copy_(
            rope_deltas.to(device=self.device, dtype=torch.long).reshape(-1)[:1]
        )
        first_token = torch.argmax(last_logits[:, -1, :], dim=-1)
        self.input_id_buf.copy_(first_token.view(1, 1))
        self.cache_position.fill_(input_len)
        self.step_buf.zero_()
        return first_token

    def capture(self, last_logits: Any, rope_deltas: Any, *, input_len: int) -> None:
        import torch

        if self.compile_decode:
            try:
                # Fuses the step's elementwise soup (rmsnorm/rotary/silu chains)
                # and picks better matmul kernels before the graph freezes the
                # kernel sequence; "no-cudagraphs" because we capture ourselves.
                step = torch.compile(self._decode_step, mode="max-autotune-no-cudagraphs")
                self._capture_step(step, last_logits, rope_deltas, input_len)
                return
            except Exception:
                self.graph = None
                self.captured = False
        self._capture_step(self._decode_step, last_logits, rope_deltas, input_len)

    def _capture_step(self, step: Any, last_logits: Any, rope_deltas: Any, input_len: int) -> None:
        import torch

        self._seed(last_logits, rope_deltas, input_len)
        for _ in range(3):
            step()
        torch.cuda.synchronize()

        with torch.cuda.device(self.device_index):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._seed(last_logits, rope_deltas, input_len)
                step()
                torch.cuda.synchronize()

                self._seed(last_logits, rope_deltas, input_len)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    step()

            torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.captured = True

    def run(self, last_logits: Any, rope_deltas: Any, *, input_len: int, max_new_tokens: int) -> list[int]:
        first_token = self._seed(last_logits, rope_deltas, input_len)
        first = int(first_token.item())
        if first in self.eos_token_ids:
            return []
        generated = [first]

        # Step s decodes at cache position input_len + s, which must stay
        # inside the static cache even for replays wasted past EOS.
        budget = min(max_new_tokens - 1, self.max_cache_len - input_len)
        done = 0
        while done < budget:
            steps = min(self.eos_check_interval, budget - done)
            for _ in range(steps):
                self.graph.replay()
            for token_id in self.token_ring[done : done + steps].tolist():
                if token_id in self.eos_token_ids:
                    return generated
                generated.append(token_id)
            done += steps

        return generated


def _resolve_eos_token_ids(model: Any) -> frozenset[int]:
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        return _EOS_TOKEN_IDS
    if isinstance(eos, int):
        return frozenset({eos})
    ids = frozenset(int(token_id) for token_id in eos)
    return ids or _EOS_TOKEN_IDS


def _normalize_audios(audio: Any) -> list[Any]:
    from qwen_asr.inference.utils import normalize_audios

    return normalize_audios(audio)


def _round_up(value: int, stride: int) -> int:
    return ((int(value) + int(stride) - 1) // int(stride)) * int(stride)


def _set_attention_implementation(model: Any, attn_implementation: str) -> None:
    configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "thinker", None), "config", None),
        getattr(getattr(getattr(model, "thinker", None), "model", None), "config", None),
        getattr(getattr(getattr(model, "thinker", None), "audio_tower", None), "config", None),
    ]
    for config in configs:
        if config is not None:
            setattr(config, "_attn_implementation", attn_implementation)


def _split_audio_into_chunks(wav: Any) -> list[tuple[Any, float]]:
    from qwen_asr.inference.utils import MAX_ASR_INPUT_SECONDS, SAMPLE_RATE, split_audio_into_chunks

    return split_audio_into_chunks(wav=wav, sr=SAMPLE_RATE, max_chunk_sec=MAX_ASR_INPUT_SECONDS)


def _parse_asr_output(raw: str, user_language: str | None) -> tuple[str, str]:
    from qwen_asr.inference.utils import parse_asr_output

    return parse_asr_output(raw, user_language=user_language)


def _normalize_languages(language: str | list[str | None] | None, count: int) -> list[str | None]:
    from qwen_asr.inference.utils import normalize_language_name, validate_language

    if language is None:
        return [None] * count
    languages = language if isinstance(language, list) else [language]
    if len(languages) == 1 and count > 1:
        languages = languages * count
    if len(languages) != count:
        raise ValueError(f"Batch size mismatch: audio={count}, language={len(languages)}")

    normalized: list[str | None] = []
    for item in languages:
        if item is None or str(item).strip() == "":
            normalized.append(None)
        else:
            value = normalize_language_name(str(item))
            validate_language(value)
            normalized.append(value)
    return normalized


def _broadcast_context(context: str | list[str], count: int) -> list[str]:
    contexts = context if isinstance(context, list) else [context]
    if len(contexts) == 1 and count > 1:
        contexts = contexts * count
    if len(contexts) != count:
        raise ValueError(f"Batch size mismatch: audio={count}, context={len(contexts)}")
    return [item or "" for item in contexts]


def _merge_languages(languages: list[str]) -> str:
    from qwen_asr.inference.utils import merge_languages

    return merge_languages(languages)
