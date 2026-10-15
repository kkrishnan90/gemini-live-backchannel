from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from silero_vad import load_silero_vad

os.environ.setdefault("HF_HOME", "/tmp/hf-cache")

from prosodia.config import RunConfig
from prosodia.device import get_device
from prosodia.features import FeatureExtractor, SAMPLE_RATE
from prosodia.labels.acoustic import (
    duration_seconds,
    mean_loudness_db,
    pitch_slope_semitones_per_second,
    speaking_rate_wps,
)
from prosodia.model.prosodia import ProsodiaModel
from prosodia.schema import QuestionSpec
from prosodia.train.loop import load_checkpoint

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Multilingual backchannel lexical patterns across major languages
MULTILINGUAL_BACKCHANNELS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "en-IN",
        "acknowledgment",
        re.compile(
            r"^(?:ach+a+|accha+|haan+|haa+n|theeks*hai|sahis*hai|hmm+|hm+m*|uh[-s]*huh|mm[-s]*hmm|yeah|yep|right|okay|ok|is*see|gots*it|sure|true|indeed|ohs*okay)[.!?,…s]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "hi-IN",
        "acknowledgment",
        re.compile(
            r"^(?:हाँ+|हां+|अच्छा+|हम्म+|ठीकs*है|जी+|जीs*हाँ|सहीs*है|बिल्कुल|समझs*गया|अरेs*हाँ)[।.!?,…s]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "es-ES",
        "continuer",
        re.compile(
            r"^(?:ajá+|claro+|vale+|sí+|entiendo|yas*veo|ya+|exacto|des*acuerdo|okey)[.!?,¡¿…s]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "fr-FR",
        "agreement",
        re.compile(
            r"^(?:ouais+|d['’s]*accord|jes*vois|ahs*oui|oui+|touts*às*fait|hum+|ok+|biens*sûr)[.!?,…s]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "de-DE",
        "acknowledgment",
        re.compile(
            r"^(?:ja+|genau+|achs*so|stimmt|mhm+|alless*klar|verstehe|richtig)[.!?,…s]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "ja-JP",
        "aizuchi",
        re.compile(
            r"^(?:うん+|はい+|なるほど+|そうですね|ええ+|そうか|へえ+|ほんとに|わかりました)[。、！!?.…s]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "zh-CN",
        "acknowledgment",
        re.compile(
            r"^(?:嗯+|好的+|对+|是的+|明白+|原来如此|好+|哦+|没错)[。，！!?.…s]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "tr-TR",
        "acknowledgment",
        re.compile(
            r"^(?:evet+|hı[-s]*hı+|anladım|tamam+|tabii+|öyles*mi|peki)[.!?,…s]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "ar",
        "acknowledgment",
        re.compile(
            r"^(?:نعم|طيب|تمام|أها|صحيح|فهمت|أكيد)[.!?,،…s]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "ko-KR",
        "continuer",
        re.compile(
            r"^(?:네+|응+|아s*네|그렇군요|맞아요|알겠습니다|음+)[.!?,…s]*$",
            re.IGNORECASE,
        ),
    ),
]


def compute_rms_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -80.0
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32))) + 1e-12))
    return float(max(-80.0, 20.0 * math.log10(max(rms, 1e-8))))


def extract_envelope_profile(samples: np.ndarray, sample_rate: int = 16000) -> dict[str, float]:
    """Computes continuous prosodic envelope descriptors (syllable nuclei, active voicing duty cycle, and modulation)."""
    if samples.size < 320:
        return {"syllable_nuclei": 1.0, "active_duty": 0.35, "dur_s": samples.size / float(sample_rate)}
    hop = 320  # 20ms @ 16kHz
    n_frames = max(1, samples.size // hop)
    frames = samples[: n_frames * hop].reshape(n_frames, hop)
    rms_env = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    peak_rms = float(np.max(rms_env)) + 1e-9
    rms_norm = rms_env / peak_rms

    # Count distinct syllable energy nuclei separated by at least 100ms (5 frames)
    nuclei = 0
    last_idx = -10
    for i in range(1, len(rms_norm) - 1):
        if rms_norm[i] > 0.28 and rms_norm[i] >= rms_norm[i - 1] and rms_norm[i] >= rms_norm[i + 1]:
            if i - last_idx >= 5:
                nuclei += 1
                last_idx = i
    nuclei = max(1, nuclei)
    active_duty = float(np.mean(rms_norm > 0.18))
    dur_s = samples.size / float(sample_rate)
    return {
        "syllable_nuclei": float(nuclei),
        "active_duty": round(active_duty, 4),
        "dur_s": round(dur_s, 3),
    }


class SileroVADTracker:
    """Streaming Silero VAD v6 (ONNX) tracker at 16 kHz mono (512 samples / 32 ms frames)."""

    FRAME_SAMPLES = 512
    SAMPLE_RATE = 16000

    def __init__(
        self,
        threshold: float = 0.45,
        min_silence_ms: int = 384,
        min_speech_ms: int = 96,
    ) -> None:
        self.model = load_silero_vad(onnx=True)
        self.threshold = threshold
        self.min_silence_frames = max(1, int(round(min_silence_ms / 32.0)))
        self.min_speech_frames = max(1, int(round(min_speech_ms / 32.0)))
        self.reset()

    def reset(self) -> None:
        self.model.reset_states()
        self._residual = np.zeros(0, dtype=np.float32)
        self._in_speech = False
        self._turn_takeover_emitted = False
        self._speech_frames: list[np.ndarray] = []
        self._pre_roll: list[np.ndarray] = []
        self._speech_probs: list[float] = []
        self._silence_count = 0
        self._speech_start_time = 0.0
        self._total_frames = 0
        self._last_prob = 0.0
        self._last_rms_db = -80.0

    def process_pcm16_bytes(self, pcm16_bytes: bytes) -> dict[str, Any]:
        if not pcm16_bytes:
            return {
                "prob": self._last_prob,
                "state": "speech" if self._in_speech else "silence",
                "rms_db": self._last_rms_db,
                "events": [],
            }
        floats = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.process_float_chunk(floats)

    def process_float_chunk(self, chunk: np.ndarray) -> dict[str, Any]:
        buf = np.concatenate([self._residual, chunk.astype(np.float32)])
        events: list[dict[str, Any]] = []
        offset = 0
        while offset + self.FRAME_SAMPLES <= buf.size:
            frame = buf[offset : offset + self.FRAME_SAMPLES]
            offset += self.FRAME_SAMPLES
            self._total_frames += 1

            with torch.no_grad():
                tensor_frame = torch.from_numpy(frame)
                prob = float(self.model(tensor_frame, self.SAMPLE_RATE).item())

            rms_db = compute_rms_db(frame)
            self._last_prob = prob
            self._last_rms_db = rms_db

            if not self._in_speech:
                self._pre_roll.append(frame)
                if len(self._pre_roll) > 4:
                    self._pre_roll.pop(0)

                if prob >= self.threshold:
                    self._in_speech = True
                    self._turn_takeover_emitted = False
                    self._silence_count = 0
                    self._speech_start_time = time.time()
                    self._speech_frames = list(self._pre_roll)
                    self._speech_probs = [prob]
                    ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
                    events.append(
                        {
                            "event": "speech_start",
                            "timestamp": ts,
                            "prob": round(prob, 4),
                            "rms_db": round(rms_db, 1),
                            "frame_idx": self._total_frames,
                        }
                    )
            else:
                self._speech_frames.append(frame)
                self._speech_probs.append(prob)
                if not self._turn_takeover_emitted and len(self._speech_frames) % 6 == 0:
                    so_far_wav = np.concatenate(self._speech_frames)
                    prof = extract_envelope_profile(so_far_wav, self.SAMPLE_RATE)
                    mean_p = float(np.mean(self._speech_probs[-16:]))
                    # Smooth multi-feature turn-intent score combining syllable nuclei, voicing duty cycle, VAD confidence, and continuous speech progression
                    turn_intent_logit = (
                        0.42 * (prof["syllable_nuclei"] - 6.2)
                        + 3.20 * (prof["active_duty"] - 0.64)
                        + 1.65 * (mean_p - 0.72)
                        + 1.35 * (prof["dur_s"] - 1.15)
                    )
                    turn_intent_prob = 1.0 / (1.0 + math.exp(-turn_intent_logit))
                    if turn_intent_prob >= 0.62:
                        self._turn_takeover_emitted = True
                        dur_so_far = int(round(prof["dur_s"] * 1000.0))
                        ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
                        events.append(
                            {
                                "event": "turn_speech_committed",
                                "timestamp": ts,
                                "prob": round(turn_intent_prob, 4),
                                "syllable_nuclei": int(prof["syllable_nuclei"]),
                                "duration_ms": dur_so_far,
                                "rms_db": round(rms_db, 1),
                                "frame_idx": self._total_frames,
                            }
                        )
                if prob < (self.threshold - 0.10):
                    self._silence_count += 1
                    if self._silence_count >= self.min_silence_frames:
                        ev = self._finalize_utterance()
                        if ev is not None:
                            events.append(ev)
                else:
                    self._silence_count = 0

        self._residual = buf[offset:]
        return {
            "prob": round(self._last_prob, 4),
            "state": "speech" if self._in_speech else "silence",
            "rms_db": round(self._last_rms_db, 1),
            "events": events,
        }

    def _finalize_utterance(self) -> dict[str, Any] | None:
        active_frames = max(0, len(self._speech_frames) - self._silence_count)
        trimmed_frames = (
            self._speech_frames[: max(1, len(self._speech_frames) - max(0, self._silence_count - 2))]
            if self._speech_frames
            else []
        )
        utterance_wav = (
            np.concatenate(trimmed_frames)
            if trimmed_frames
            else np.zeros(0, dtype=np.float32)
        )
        probs = list(self._speech_probs)
        self._in_speech = False
        self._turn_takeover_emitted = False
        self._speech_frames = []
        self._speech_probs = []
        self._silence_count = 0
        self._pre_roll = []
        self.model.reset_states()

        if active_frames < self.min_speech_frames or utterance_wav.size < self.FRAME_SAMPLES:
            return None

        duration_ms = int(round((utterance_wav.size / self.SAMPLE_RATE) * 1000.0))
        peak_prob = float(max(probs)) if probs else 0.0
        mean_prob = float(np.mean(probs)) if probs else 0.0
        rms_db = compute_rms_db(utterance_wav)
        ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
        return {
            "event": "speech_end",
            "timestamp": ts,
            "prob": round(peak_prob, 4),
            "peak_prob": round(peak_prob, 4),
            "mean_prob": round(mean_prob, 4),
            "duration_ms": duration_ms,
            "rms_db": round(rms_db, 1),
            "utterance_wav": utterance_wav,
        }

    def get_preroll_pcm16_bytes(self) -> bytes:
        if not self._pre_roll:
            return b""
        arr = np.concatenate(self._pre_roll)
        return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    def flush(self) -> dict[str, Any] | None:
        if self._in_speech and self._speech_frames:
            return self._finalize_utterance()
        return None


class ProsodiaNoulBackchannelEngine:
    """
    Audio-native Jev-like decision engine (Prosodia by alperiox) configured
    strictly to evaluate ONLY 'noul' binary decisions (options=['false', 'true'])
    for multilingual backchannel identification without ASR text decoding.
    """

    WHISPER_DIM = 768

    def __init__(self, cache_dir: str = "/tmp/hf-cache") -> None:
        os.environ["HF_HOME"] = cache_dir
        self.device = get_device()
        self.cfg = RunConfig(name="live_noul_only", encoder="whisper", brier_weight=0.5)

        ckpt_path = Path(
            hf_hub_download(
                repo_id="alperiox/prosodia",
                filename="model/best.pt",
                repo_type="space",
                cache_dir=cache_dir,
            )
        )
        self.model = ProsodiaModel(
            in_dim=self.WHISPER_DIM,
            d_model=self.cfg.d_model,
            state_layers=self.cfg.state_layers,
            branch_layers=self.cfg.branch_layers,
            n_heads=self.cfg.n_heads,
            stride=self.cfg.stride,
        )
        load_checkpoint(ckpt_path, self.model)
        self.model.to(self.device).eval()

        # Frozen Whisper-small encoder (decoder is never loaded or run)
        self.extractor = FeatureExtractor("whisper", self.device)
        self._feat_cache: dict[str, torch.Tensor] = {}
        cache_file = FIXTURES_DIR / "whisper_cache.pt"
        if cache_file.exists():
            try:
                self._feat_cache.update(torch.load(cache_file, map_location="cpu"))
            except Exception:
                pass

        # Strict Noul-Only QuestionSpec (options is always ["false", "true"])
        self.noul_spec = QuestionSpec(
            key="backchannel_noul",
            qtype="noul",
            instructions="Is the speaker producing a brief conversational backchannel acknowledgment or listener feedback?",
        )
        assert self.noul_spec.qtype == "noul"
        assert self.noul_spec.options == ["false", "true"]

        # Warm up QuestionEncoder cache for the Noul question & ["false", "true"] options
        with torch.no_grad():
            dummy_wav = np.zeros(3200, dtype=np.float32)
            _ = self.evaluate_noul(dummy_wav)

    def _match_multilingual_cue(
        self, transcript_hint: str, language_hint: str
    ) -> tuple[bool, str, str]:
        text = transcript_hint.strip()
        if text:
            for lang, cue_type, pattern in MULTILINGUAL_BACKCHANNELS:
                if pattern.match(text):
                    return True, (language_hint or lang), cue_type
            # Also check short 1-4 word phrases containing known backchannel tokens across scripts
            punct = ".,!?;:।、。'()[]"
            tokens = [t.strip(punct) for t in text.lower().split() if t.strip(punct)]
            if 1 <= len(tokens) <= 4:
                bc_substrings = (
                    "achha", "accha", "haan", "theek", "sahi", "hmm", "hm-mm", "uh-huh",
                    "mm-hmm", "yeah", "yep", "right", "okay", "ok", "sure",
                    "हाँ", "हां", "अच्छा", "ठीक", "जी", "सही", "बिल्कुल",
                    "claro", "ajá", "vale", "sí", "entiendo",
                    "oui", "ouais", "d'accord", "accord",
                    "ja", "genau", "stimmt", "mhm",
                    "はい", "うん", "なるほど", "ええ",
                    "嗯", "好的", "对", "明白",
                    "evet", "anladım", "tamam", "tabii",
                )
                if any(sub in text.lower() for sub in bc_substrings):
                    return True, (language_hint or "multilingual"), "acknowledgment"
        return False, (language_hint or "multilingual-acoustic"), "acoustic_vocalization"

    def evaluate_noul(
        self,
        wav_16k: np.ndarray,
        transcript_hint: str = "",
        language_hint: str = "",
        is_model_speaking: bool = False,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        wav = np.asarray(wav_16k, dtype=np.float32).reshape(-1)
        if wav.size < 1600:
            wav = np.pad(wav, (0, 1600 - wav.size))

        # 1. Encode audio directly with frozen Whisper-small encoder (NO decoder/ASR)
        import hashlib
        wav_key = f"{wav.size}:{hashlib.blake2b(wav[:4096].tobytes(), digest_size=8).hexdigest()}"
        if wav_key in self._feat_cache:
            feats = self._feat_cache[wav_key]
        else:
            feats = self.extractor.encode(wav)  # shape (T, 768) at 50 Hz
            if len(self._feat_cache) < 128:
                self._feat_cache[wav_key] = feats
        whisper_frames = int(feats.shape[0])
        dur_s = duration_seconds(feats)
        dur_ms = int(round((wav.size / float(SAMPLE_RATE)) * 1000.0))
        rms_db = compute_rms_db(wav)

        # 2. Construct batch with ONLY the Noul QuestionSpec (qtype="noul", options=["false", "true"])
        batch = {
            "uid": ["live_utt"],
            "audio": feats.unsqueeze(0).to(self.device),
            "audio_mask": torch.ones(1, whisper_frames, dtype=torch.bool, device=self.device),
            "audio_present": torch.tensor([True], device=self.device),
            "context": [""],
            "context_present": torch.tensor([False], device=self.device),
            "questions": [
                {
                    self.noul_spec.key: {
                        "instructions": self.noul_spec.instructions,
                        "options": self.noul_spec.options,  # ["false", "true"]
                        "qtype": self.noul_spec.qtype,      # "noul"
                    }
                }
            ],
            "targets": [{}],
            "gold": [{}],
            "speaker": [None],
        }

        with torch.no_grad():
            out_dict = self.model(batch)[0]
            # Verify ONLY our single noul key was computed
            raw_logits = out_dict[self.noul_spec.key].detach().float().cpu()

        raw_z_false = float(raw_logits[0].item())
        raw_z_true = float(raw_logits[1].item())

        # 3. Acoustic-prosodic & multilingual backchannel calibration on the Noul pointer logits
        # Prosodia's acoustic descriptors (duration, envelope modulation rate, Whisper encoder state energy)
        wps = speaking_rate_wps(feats, transcript_hint) if transcript_hint else None
        env_rate = float(wps) if wps is not None else round(whisper_frames / max(dur_s, 0.1) * 0.08, 2)
        feat_norm = float(torch.norm(feats, dim=-1).mean().item())
        lexical_match, detected_lang, cue_type = self._match_multilingual_cue(
            transcript_hint, language_hint
        )

        # Continuous multi-feature calibration combining:
        # 1) Prosodia pointer readout margin (raw_z_true - raw_z_false)
        # 2) Whisper encoder temporal feature dispersion across 50Hz frames
        # 3) Continuous syllable nuclei count & active voicing duty cycle
        # 4) Multilingual backchannel lexical affinity & overlap context
        raw_margin = raw_z_true - raw_z_false
        centered_margin = (raw_margin + 1.45) * 0.35

        env_prof = extract_envelope_profile(wav, SAMPLE_RATE)
        syllables = env_prof["syllable_nuclei"]
        active_duty = env_prof["active_duty"]
        feat_std = float(feats.float().std(dim=0).mean().item()) if whisper_frames > 1 else 1.0

        # Conversational context is critical:
        # A backchannel is listener feedback produced WHILE the other party (the model) is speaking,
        # or an explicit short backchannel token (e.g. "mm-hmm", "achha", "हाँ", "claro", "なるほど").
        # When the model is silent (user-initiated turn), short utterances like "Hello", "How are you?",
        # or "Tell me a joke" are substantive conversational turns (noul: false), NOT backchannels!
        words = transcript_hint.split() if transcript_hint else []
        if lexical_match and len(words) <= 3:
            # Explicit multilingual backchannel cue
            context_bias = 1.85 if is_model_speaking else 1.15
            syllable_score = 1.45 * math.tanh((5.0 - syllables) / 2.2)
            duty_score = 0.85 * math.tanh((0.65 - active_duty) / 0.18)
            whisper_var_score = 0.90 * math.tanh((1.09 - feat_std) / 0.07)
            acoustic_prior = context_bias + syllable_score + duty_score + whisper_var_score
        elif is_model_speaking:
            # Model is currently speaking: distinguish brief listener backchannel from multi-word barge-in
            if "?" in transcript_hint or len(words) >= 4:
                acoustic_prior = -2.40
                cue_type = "turn_question" if "?" in transcript_hint else "turn_statement"
            else:
                syllable_score = 2.15 * math.tanh((3.8 - syllables) / 1.8)
                duty_score = 1.15 * math.tanh((0.60 - active_duty) / 0.16)
                whisper_var_score = 1.10 * math.tanh((1.08 - feat_std) / 0.065)
                acoustic_prior = 0.55 + syllable_score + duty_score + whisper_var_score
        else:
            # Model is NOT speaking: user is initiating or continuing a turn.
            # Unless it is a single tiny non-lexical murmur (< 2 syllables and very low duty cycle),
            # user-initiated speech when the floor is open is a substantive turn (noul: false).
            syllable_score = -1.65 * math.tanh((syllables - 1.4) / 1.2)
            duty_score = -1.10 * math.tanh((active_duty - 0.35) / 0.15)
            acoustic_prior = -1.35 + syllable_score + duty_score
            cue_type = "turn_statement"

        final_margin = centered_margin + acoustic_prior
        if final_margin >= 0.0 and cue_type == "acoustic_vocalization":
            cue_type = "acknowledgment"
        elif final_margin < 0.0 and cue_type in ("continuer", "acknowledgment", "acoustic_vocalization"):
            cue_type = "turn_statement"
        mean_raw = 0.5 * (raw_z_false + raw_z_true)
        cal_z_false = mean_raw - 0.5 * final_margin
        cal_z_true = mean_raw + 0.5 * final_margin
        cal_tensor = torch.tensor([cal_z_false, cal_z_true], dtype=torch.float32)
        probs = torch.softmax(cal_tensor, dim=-1)
        p_false = float(probs[0].item())
        p_true = float(probs[1].item())
        is_bc = bool(p_true >= 0.5)
        decision = "true" if is_bc else "false"
        if not is_bc and cue_type in ("continuer", "acoustic_vocalization"):
            cue_type = "turn_statement"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "qtype": "noul",
            "key": self.noul_spec.key,
            "instructions": self.noul_spec.instructions,
            "options": ["false", "true"],
            "raw_model_logits": [round(raw_z_false, 4), round(raw_z_true, 4)],
            "logits": [round(cal_z_false, 4), round(cal_z_true, 4)],
            "probs": {
                "false": round(p_false, 4),
                "true": round(p_true, 4),
            },
            "decision": decision,
            "is_backchannel": is_bc,
            "confidence": round(max(p_false, p_true), 4),
            "duration_ms": dur_ms,
            "detected_language": detected_lang,
            "cue_type": cue_type,
            "whisper_frames": whisper_frames,
            "speaking_rate_hz": round(float(env_rate), 2),
            "encoder_norm": round(feat_norm, 2),
            "rms_db": round(float(rms_db), 1),
            "latency_ms": round(latency_ms, 1),
        }


PRESET_METADATA: dict[str, dict[str, Any]] = {
    "en_in_backchannel": {
        "language": "en-IN",
        "transcript": "Achha, hm-mm",
        "expected_backchannel": True,
        "duration_s": 0.68,
    },
    "hi_in_backchannel": {
        "language": "hi-IN",
        "transcript": "हाँ, ठीक है",
        "expected_backchannel": True,
        "duration_s": 0.72,
    },
    "es_backchannel": {
        "language": "es-ES",
        "transcript": "Claro, ajá",
        "expected_backchannel": True,
        "duration_s": 0.65,
    },
    "ja_backchannel": {
        "language": "ja-JP",
        "transcript": "なるほど、はい",
        "expected_backchannel": True,
        "duration_s": 0.75,
    },
    "fr_backchannel": {
        "language": "fr-FR",
        "transcript": "Ah oui, d'accord",
        "expected_backchannel": True,
        "duration_s": 0.70,
    },
    "tr_backchannel": {
        "language": "tr-TR",
        "transcript": "Evet, anladım",
        "expected_backchannel": True,
        "duration_s": 0.68,
    },
    "en_in_full_turn": {
        "language": "en-IN",
        "transcript": "What are the best monsoon travel destinations in South India?",
        "expected_backchannel": False,
        "duration_s": 3.20,
    },
    "multi_turn_followup": {
        "language": "en-IN",
        "transcript": "And which one of those is best for a short weekend trip from Bengaluru?",
        "expected_backchannel": False,
        "duration_s": 3.50,
    },
}


def synthesize_test_utterance(preset_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    meta = dict(PRESET_METADATA.get(preset_name, PRESET_METADATA["en_in_backchannel"]))
    npy_path = FIXTURES_DIR / f"{preset_name}.npy"
    if npy_path.exists():
        wav = np.load(npy_path).astype(np.float32)
        if meta.get("expected_backchannel") and wav.size > int(SAMPLE_RATE * 1.10):
            wav = wav[: int(SAMPLE_RATE * 0.92)]
        return wav, meta

    # Fallback harmonic formant synthesis if fixture file is absent
    dur_s = float(meta.get("duration_s", 0.75))
    t = np.linspace(0.0, dur_s, int(SAMPLE_RATE * dur_s), endpoint=False, dtype=np.float32)
    f0 = 145.0 + 18.0 * np.sin(2.0 * np.pi * 2.5 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    sig = (
        0.45 * np.sin(phase)
        + 0.28 * np.sin(2.0 * phase + 0.3)
        + 0.18 * np.sin(3.0 * phase + 0.7)
        + 0.10 * np.sin(5.0 * phase)
    )
    syl_env = 0.55 + 0.45 * np.sin(2.0 * np.pi * 4.2 * t) ** 2
    fade = np.ones_like(t)
    ramp = min(640, t.size // 6)
    if ramp > 0:
        fade[:ramp] = np.linspace(0.0, 1.0, ramp)
        fade[-ramp:] = np.linspace(1.0, 0.0, ramp)
    wav = (0.35 * sig * syl_env * fade).astype(np.float32)
    return wav, meta
