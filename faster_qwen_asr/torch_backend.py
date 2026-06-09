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
        attn_implementation: str | None = None,
    ) -> None:
        self.official_model = official_model
        self.model = official_model.model
        self.processor = official_model.processor
        self.max_new_tokens = int(max_new_tokens)
        self.max_inference_batch_size = int(max_inference_batch_size)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.cuda_graph_stride = max(1, int(cuda_graph_stride))
        self.attn_implementation = attn_implementation
        self.device = getattr(official_model, "device", None)
        self.dtype = getattr(official_model, "dtype", None)
        self._graph_cache: dict[int, _DecoderGraph] = {}
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
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        thinker = self.model.thinker

        with torch.inference_mode():
            thinker.rope_deltas = None
            outputs = thinker(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"],
                feature_attention_mask=inputs["feature_attention_mask"],
                use_cache=True,
            )

            if self._can_use_cuda_graph(torch):
                try:
                    graph_max_new_tokens = self._estimate_graph_max_new_tokens(wav)
                    generated = self._decode_with_graph(
                        outputs,
                        max_new_tokens=graph_max_new_tokens,
                    )
                    if len(generated) >= graph_max_new_tokens < self.max_new_tokens:
                        generated = self._decode_dynamic(outputs, inputs["attention_mask"])
                except Exception:
                    self._graph_failed = True
                    generated = self._decode_dynamic(outputs, inputs["attention_mask"])
            else:
                generated = self._decode_dynamic(outputs, inputs["attention_mask"])

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

    def _decode_dynamic(self, outputs: Any, attention_mask: Any) -> list[int]:
        import torch

        thinker = self.model.thinker
        past_key_values = outputs.past_key_values
        current_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        current_attention_mask = attention_mask
        one = torch.ones(
            (current_attention_mask.shape[0], 1),
            dtype=current_attention_mask.dtype,
            device=current_attention_mask.device,
        )
        generated: list[int] = []
        eos_token_ids = {151645, 151643}

        for _ in range(self.max_new_tokens):
            token_id = int(current_token.item())
            if token_id in eos_token_ids:
                break
            generated.append(token_id)

            current_attention_mask = torch.cat([current_attention_mask, one], dim=1)
            cache_position = torch.tensor(
                [current_attention_mask.shape[1] - 1],
                dtype=torch.long,
                device=current_attention_mask.device,
            )
            outputs = thinker(
                input_ids=current_token,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
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

    def _decode_with_graph(self, outputs: Any, *, max_new_tokens: int) -> list[int]:
        input_len = int(outputs.past_key_values.get_seq_length())
        required_cache_len = input_len + max_new_tokens + 4
        max_cache_len = _round_up(required_cache_len, self.cuda_graph_stride)
        graph = self._graph_cache.get(max_cache_len)
        if graph is None:
            graph = _DecoderGraph(
                thinker=self.model.thinker,
                max_cache_len=max_cache_len,
                dtype=self.model.dtype,
                device=str(self.model.device),
            )
            graph.capture(outputs, input_len=input_len)
            self._graph_cache[max_cache_len] = graph
        return graph.run(outputs, input_len=input_len, max_new_tokens=max_new_tokens)


class _DecoderGraph:
    """CUDA graph for one-token Qwen3-ASR text decode."""

    def __init__(self, *, thinker: Any, max_cache_len: int, dtype: Any, device: str) -> None:
        import torch
        from transformers import StaticCache

        self.thinker = thinker
        self.text_model = thinker.model
        self.max_cache_len = int(max_cache_len)
        self.dtype = dtype
        self.device = device
        device_index = torch.device(device).index
        self.device_index = device_index if device_index is not None else torch.cuda.current_device()

        self.static_cache = StaticCache(config=self.text_model.config, max_cache_len=self.max_cache_len)
        self.input_id_buf = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.output_id_buf = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self.position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
        self.rope_deltas = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.attn_mask = None
        self.attn_masks = None
        self.graph = None
        self.captured = False

    def _fill_static_cache(self, dynamic_cache: Any) -> None:
        import torch

        self.static_cache.reset()
        for layer_idx in range(len(dynamic_cache.layers)):
            key_states, value_states = dynamic_cache[layer_idx]
            seq_len = key_states.shape[2]
            if seq_len > self.max_cache_len:
                raise RuntimeError(
                    f"Prefill length {seq_len} exceeds CUDA graph cache length {self.max_cache_len}."
                )
            self.static_cache.update(
                key_states,
                value_states,
                layer_idx,
                {"cache_position": torch.arange(seq_len, device=key_states.device)},
            )

    def _build_attention_masks(self) -> None:
        import torch
        from transformers.masking_utils import create_causal_mask

        dummy = torch.zeros(
            1,
            1,
            self.text_model.config.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.attn_masks = [
            create_causal_mask(
                config=self.text_model.config,
                input_embeds=dummy,
                attention_mask=None,
                cache_position=torch.tensor([pos], device=self.device),
                past_key_values=self.static_cache,
            )
            for pos in range(self.max_cache_len)
        ]
        self.attn_mask = self.attn_masks[0].clone()

    def _set_position(self, position: int) -> None:
        self.cache_position[0] = position
        self.position_ids.copy_(
            (self.cache_position.view(1, 1, 1) + self.rope_deltas.view(1, 1, 1)).expand(3, 1, 1)
        )
        self.attn_mask.copy_(self.attn_masks[position])

    def _decode_step(self) -> None:
        import torch

        embeds = self.thinker.get_input_embeddings()(self.input_id_buf)
        outputs = self.text_model(
            inputs_embeds=embeds,
            attention_mask=self.attn_mask,
            past_key_values=self.static_cache,
            use_cache=True,
            cache_position=self.cache_position,
            position_ids=self.position_ids,
        )
        logits = self.thinker.lm_head(outputs.last_hidden_state)
        self.output_id_buf.copy_(torch.argmax(logits[:, -1, :], dim=-1, keepdim=True))

    def capture(self, prefill_outputs: Any, *, input_len: int) -> None:
        import torch

        self._fill_static_cache(prefill_outputs.past_key_values)
        self._build_attention_masks()
        self.rope_deltas.copy_(prefill_outputs.rope_deltas.to(device=self.device, dtype=torch.long))
        current_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1, keepdim=True)
        self.input_id_buf.copy_(current_token)
        self._set_position(input_len)

        for _ in range(3):
            self._decode_step()
        torch.cuda.synchronize()

        self._fill_static_cache(prefill_outputs.past_key_values)
        self.input_id_buf.copy_(current_token)
        self._set_position(input_len)

        with torch.cuda.device(self.device_index):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._decode_step()
                torch.cuda.synchronize()

                self._fill_static_cache(prefill_outputs.past_key_values)
                self.input_id_buf.copy_(current_token)
                self._set_position(input_len)

                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self._decode_step()

            torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.captured = True

    def run(self, prefill_outputs: Any, *, input_len: int, max_new_tokens: int) -> list[int]:
        import torch

        self._fill_static_cache(prefill_outputs.past_key_values)
        self.rope_deltas.copy_(prefill_outputs.rope_deltas.to(device=self.device, dtype=torch.long))
        current_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated: list[int] = []
        eos_token_ids = {151645, 151643}

        for step in range(max_new_tokens):
            token_id = int(current_token.item())
            if token_id in eos_token_ids:
                break
            generated.append(token_id)

            position = input_len + step
            if position >= self.max_cache_len:
                raise RuntimeError(
                    f"Decode position {position} exceeds CUDA graph cache length {self.max_cache_len}."
                )
            self.input_id_buf.copy_(current_token)
            self._set_position(position)
            self.graph.replay()
            current_token = self.output_id_buf.clone()

        return generated


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
