import unittest

from faster_qwen_asr import ASRResult, MODEL_ALIASES, resolve_model_id
from faster_qwen_asr.audio import is_batch_audio
from faster_qwen_asr.model import _select_backend
from faster_qwen_asr.torch_backend import TorchQwenASRBackend, _round_up


class PublicApiTest(unittest.TestCase):
    def test_model_aliases_cover_both_qwen3_asr_sizes(self):
        self.assertEqual(resolve_model_id(size="0.6B"), "Qwen/Qwen3-ASR-0.6B")
        self.assertEqual(resolve_model_id(size="1.7B"), "Qwen/Qwen3-ASR-1.7B")
        self.assertTrue(MODEL_ALIASES["small"].endswith("0.6B"))
        self.assertTrue(MODEL_ALIASES["large"].endswith("1.7B"))

    def test_explicit_model_id_passes_through(self):
        self.assertEqual(resolve_model_id(model="/models/qwen"), "/models/qwen")

    def test_model_and_size_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "either `model` or `size`"):
            resolve_model_id(model="x", size="0.6B")

    def test_audio_batch_detection(self):
        self.assertFalse(is_batch_audio("audio.wav"))
        self.assertFalse(is_batch_audio(("array", 16000)))
        self.assertTrue(is_batch_audio(["a.wav", "b.wav"]))

    def test_result_dataclass_defaults(self):
        result = ASRResult(text="hello")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.language, "")
        self.assertIsNone(result.time_stamps)

    def test_explicit_torch_backend_is_supported(self):
        self.assertEqual(_select_backend("torch", None), "torch")

    def test_cuda_graph_bucket_rounding(self):
        self.assertEqual(_round_up(293, 128), 384)
        self.assertEqual(_round_up(384, 128), 384)
        self.assertEqual(_round_up(1, 128), 128)

    def test_torch_backend_loader_accepts_forced_aligner(self):
        import inspect

        sig = inspect.signature(TorchQwenASRBackend.from_pretrained)
        self.assertIn("forced_aligner", sig.parameters)
        self.assertIn("forced_aligner_kwargs", sig.parameters)

    def test_public_loader_uses_auto_attention_default(self):
        import inspect
        from faster_qwen_asr import QwenASR

        sig = inspect.signature(QwenASR.from_pretrained)
        self.assertIsNone(sig.parameters["attn_implementation"].default)

    def test_torch_backend_routes_batch_to_official_backend(self):
        class RawResult:
            text = "ok"
            language = "English"
            time_stamps = None

        class Official:
            def __init__(self):
                self.calls = []
                self.model = object()
                self.processor = object()

            def transcribe(self, **kwargs):
                self.calls.append(kwargs)
                audio = kwargs["audio"]
                return [RawResult() for _ in audio]

        official = Official()
        backend = TorchQwenASRBackend(official, max_inference_batch_size=8)
        results = backend.transcribe(["a.wav", "b.wav"], language="English")
        self.assertEqual([r.text for r in results], ["ok", "ok"])
        self.assertEqual(len(official.calls), 1)
        self.assertEqual(official.calls[0]["audio"], ["a.wav", "b.wav"])

    def test_torch_backend_estimates_short_audio_graph_budget(self):
        class Official:
            model = object()
            processor = object()

        backend = TorchQwenASRBackend(Official(), max_new_tokens=256)
        self.assertEqual(backend._estimate_graph_max_new_tokens([0.0] * 16000), 40)
        self.assertEqual(backend._estimate_graph_max_new_tokens([0.0] * 176000), 120)

    def test_torch_backend_graph_budget_never_exceeds_limit(self):
        class Official:
            model = object()
            processor = object()

        backend = TorchQwenASRBackend(Official(), max_new_tokens=64)
        self.assertEqual(backend._estimate_graph_max_new_tokens([0.0] * 176000), 64)

    def test_torch_backend_sets_nested_attention_implementation(self):
        class Config:
            _attn_implementation = "sdpa"

        class Thinker:
            config = Config()

            class Text:
                config = Config()

            class Audio:
                config = Config()

            model = Text()
            audio_tower = Audio()

        class Model:
            config = Config()
            thinker = Thinker()

        class Official:
            model = Model()
            processor = object()

        backend = TorchQwenASRBackend(Official(), attn_implementation="eager")
        self.assertEqual(backend.model.config._attn_implementation, "eager")
        self.assertEqual(backend.model.thinker.config._attn_implementation, "eager")
        self.assertEqual(backend.model.thinker.model.config._attn_implementation, "eager")
        self.assertEqual(backend.model.thinker.audio_tower.config._attn_implementation, "eager")

    def test_torch_backend_uses_sdpa_for_official_fallback_then_restores(self):
        class Config:
            _attn_implementation = "eager"

        class Thinker:
            config = Config()

            class Text:
                config = Config()

            class Audio:
                config = Config()

            model = Text()
            audio_tower = Audio()

        class Model:
            config = Config()
            thinker = Thinker()

        class RawResult:
            text = "ok"
            language = "English"
            time_stamps = None

        class Official:
            model = Model()
            processor = object()

            def transcribe(self, **kwargs):
                assert self.model.thinker.model.config._attn_implementation == "sdpa"
                return [RawResult()]

        backend = TorchQwenASRBackend(Official(), attn_implementation="eager")
        result = backend._transcribe_official(
            audio="a.wav",
            context="",
            language="English",
            return_time_stamps=False,
        )
        self.assertEqual(result[0].text, "ok")
        self.assertEqual(backend.model.thinker.model.config._attn_implementation, "eager")


if __name__ == "__main__":
    unittest.main()
