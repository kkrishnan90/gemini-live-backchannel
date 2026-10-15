from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

# Ensure HuggingFace cache uses /tmp/hf-cache
os.environ.setdefault("HF_HOME", "/tmp/hf-cache")

from audio_engine import (
    ProsodiaNoulBackchannelEngine,
    SileroVADTracker,
    synthesize_test_utterance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gemini-live-prosodia")

def _resolve_default_project_id() -> str:
    env_proj = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    if env_proj:
        return env_proj
    try:
        import google.auth
        _, default_proj = google.auth.default()
        if default_proj:
            return default_proj
    except Exception:
        pass
    return "your-gcp-project-id"


PROJECT_ID = _resolve_default_project_id()
_env_loc = os.environ.get("GEMINI_LIVE_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
LOCATION = "us-central1" if _env_loc in ("global", "", None) else _env_loc
MODEL_ID = "gemini-3.8-live"
VOICE_NAME = "Kore"
LANGUAGE_CODE = "en-IN"

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Global shared Prosodia Noul engine (loaded once at startup)
prosodia_engine: ProsodiaNoulBackchannelEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global prosodia_engine
    logger.info("Initializing Prosodia (audio-jevlike) Noul-Only Backchannel Engine...")
    prosodia_engine = ProsodiaNoulBackchannelEngine(cache_dir="/tmp/hf-cache")
    logger.info("Prosodia Noul-Only Engine initialized and ready.")
    yield


app = FastAPI(
    title="Prosodia × Gemini 3.8 Live API (Agent Platform)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_live_config_summary() -> dict[str, Any]:
    return {
        "platform": "Google Cloud Agent Platform (Vertex AI)",
        "auth": "Application Default Credentials (ADC)",
        "project": "<configured-via-adc>",
        "location": LOCATION,
        "model": MODEL_ID,
        "launch_stage": "GA (Latest)",
        "voice_name": VOICE_NAME,
        "language_code": LANGUAGE_CODE,
        "response_modalities": ["AUDIO"],
        "input_audio_format": "audio/pcm;rate=16000",
        "output_audio_format": "audio/pcm;rate=24000",
        "input_audio_transcription": True,
        "output_audio_transcription": True,
        "multi_turn": True,
        "vad": {
            "engine": "Silero VAD v6 (ONNX)",
            "sample_rate": 16000,
            "frame_samples": 512,
            "frame_ms": 32,
            "threshold": 0.45,
        },
        "prosodia": {
            "architecture": "alperiox/audio-jevlike (ProsodiaModel)",
            "encoder": "openai/whisper-small (Frozen Encoder, No Decoder/ASR)",
            "question_encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "primitive": "noul",
            "options": ["false", "true"],
            "mode": "ONLY backchannel noul decisions across multiple languages",
        },
    }


def build_gemini_live_connect_config() -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=VOICE_NAME
                )
            ),
            language_code=LANGUAGE_CODE,
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(
            parts=[
                types.Part.from_text(
                    text=(
                        "You are an articulate, warm, concise voice intelligence agent running on "
                        "Google Cloud Agent Platform powered by Gemini 3.8 Flash Live API (gemini-3.8-live) "
                        "using the Kore voice and en-IN (Indian English) locale. "
                        "Keep your spoken responses natural, crisp, and conversational (1 to 3 sentences). "
                        "You seamlessly handle multi-turn dialogue and multilingual listener backchannels."
                    )
                )
            ]
        ),
    )


@app.get("/api/config")
async def api_config():
    return JSONResponse(get_live_config_summary())


@app.get("/api/health")
async def api_health():
    return JSONResponse(
        {
            "status": "ok",
            "prosodia_loaded": prosodia_engine is not None,
            "config": get_live_config_summary(),
        }
    )


class LiveConversationCoordinator:
    """Manages a single client WebSocket session bridged to Gemini 3.8 Live API + Silero VAD + Prosodia Noul."""

    def __init__(self, ws: WebSocket, engine: ProsodiaNoulBackchannelEngine):
        self.ws = ws
        self.engine = engine
        self.vad = SileroVADTracker(threshold=0.45, min_silence_ms=384, min_speech_ms=96)
        # Ensure google-genai SDK uses regional endpoint (us-central1) for Live API WebSockets
        os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
        self.client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
        )
        self.gemini_session = None
        self.turn_counter = 1
        self.is_model_speaking = False
        self.current_user_transcript: list[str] = []
        self.current_model_transcript: list[str] = []
        self.last_noul_event: dict[str, Any] | None = None
        self.last_telemetry_sent = 0.0
        self._buffered_speech_chunks: list[bytes] = []
        self._streaming_direct_to_gemini = False
        self._closed = False
        self._send_lock = asyncio.Lock()

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            async with self._send_lock:
                await self.ws.send_text(json.dumps(payload))
        except Exception:
            pass

    async def evaluate_and_emit_noul(
        self,
        utterance_wav: np.ndarray,
        transcript_hint: str = "",
        language_hint: str = "",
        forward_to_gemini_if_turn: bool = False,
        raw_pcm16_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        noul_res = await loop.run_in_executor(
            None,
            lambda: self.engine.evaluate_noul(
                utterance_wav,
                transcript_hint=transcript_hint,
                language_hint=language_hint,
                is_model_speaking=self.is_model_speaking,
            ),
        )
        event_id = f"noul-{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
        noul_payload = {
            "type": "prosodia_noul",
            "id": event_id,
            "timestamp": ts,
            **noul_res,
            "transcript_hint": transcript_hint,
        }
        self.last_noul_event = noul_payload
        await self.send_json(noul_payload)

        if noul_res.get("is_backchannel"):
            # Do NOT interrupt Gemini Live when user produces a backchannel; log event to side panel only.
            await self.send_json(
                {
                    "type": "turn_event",
                    "event": "backchannel_no_interrupt",
                    "timestamp": ts,
                    "turn_id": self.turn_counter,
                    "noul_decision": "true",
                    "noul_prob_true": noul_res["probs"]["true"],
                    "detected_language": noul_res.get("detected_language", "multilingual"),
                    "duration_ms": noul_res.get("duration_ms", 0),
                    "transcript_hint": transcript_hint,
                }
            )
        elif forward_to_gemini_if_turn and self.gemini_session is not None and raw_pcm16_bytes:
            # Forward substantive long sentence audio to Gemini 3.8 Live API
            chunk_size = 3200
            for idx in range(0, len(raw_pcm16_bytes), chunk_size):
                await self.gemini_session.send_realtime_input(
                    audio=types.Blob(
                        data=raw_pcm16_bytes[idx : idx + chunk_size],
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                await asyncio.sleep(0.01)
            # Send trailing silence so Gemini Live server VAD immediately finalizes the turn
            silence = np.zeros(1600, dtype=np.int16).tobytes()
            for _ in range(12):
                await self.gemini_session.send_realtime_input(
                    audio=types.Blob(data=silence, mime_type="audio/pcm;rate=16000")
                )
                await asyncio.sleep(0.02)

        return noul_payload

    async def process_pcm16_stream(
        self,
        pcm16_bytes: bytes,
        stream_to_gemini_live: bool = True,
        transcript_hint: str = "",
        language_hint: str = "",
    ) -> None:
        # 1. Feed Silero VAD
        vad_result = self.vad.process_pcm16_bytes(pcm16_bytes)
        now = time.time()
        if now - self.last_telemetry_sent >= 0.06:
            self.last_telemetry_sent = now
            await self.send_json(
                {
                    "type": "vad_telemetry",
                    "prob": round(float(vad_result["prob"]), 4),
                    "state": vad_result["state"],
                    "rms_db": round(float(vad_result["rms_db"]), 1),
                }
            )

        for ev in vad_result.get("events", []):
            ev_name = ev.get("event")
            ev_copy = {k: v for k, v in ev.items() if k != "utterance_wav"}
            await self.send_json(
                {
                    "type": "vad_event",
                    **ev_copy,
                }
            )

            if ev_name == "speech_start":
                self._buffered_speech_chunks = []
                # When the model is NOT currently speaking, stream user speech immediately to Gemini Live
                # so even short greetings ("Hello", "Hi Kore") get instant sub-second responses!
                # When the model IS speaking, buffer incoming audio until prosodic turn-intent confirms
                # a substantive barge-in so backchannels NEVER interrupt the model.
                self._streaming_direct_to_gemini = (not self.is_model_speaking) and stream_to_gemini_live
                if self._streaming_direct_to_gemini and self.gemini_session is not None:
                    # Include Silero VAD pre-roll frames so the beginning consonant/vowel is never clipped
                    utt_pre = self.vad.get_preroll_pcm16_bytes()
                    if utt_pre:
                        try:
                            await self.gemini_session.send_realtime_input(
                                audio=types.Blob(data=utt_pre, mime_type="audio/pcm;rate=16000")
                            )
                        except Exception:
                            pass

            elif ev_name == "turn_speech_committed":
                # Multi-feature prosodic turn-intent score confirmed a substantive turn (not a backchannel).
                # If model was speaking, interrupt model playback now and flush buffered audio to Gemini Live.
                if self.is_model_speaking:
                    self.is_model_speaking = False
                    await self.send_json(
                        {
                            "type": "turn_event",
                            "event": "interrupted",
                            "timestamp": ev.get("timestamp", ""),
                            "turn_id": self.turn_counter,
                            "reason": "prosodic_turn_barge_in",
                        }
                    )
                if stream_to_gemini_live and self.gemini_session is not None and not self._streaming_direct_to_gemini:
                    self._streaming_direct_to_gemini = True
                    buffered_bytes = b"".join(self._buffered_speech_chunks)
                    self._buffered_speech_chunks = []
                    if buffered_bytes:
                        try:
                            chunk_sz = 3200
                            for idx in range(0, len(buffered_bytes), chunk_sz):
                                await self.gemini_session.send_realtime_input(
                                    audio=types.Blob(
                                        data=buffered_bytes[idx : idx + chunk_sz],
                                        mime_type="audio/pcm;rate=16000",
                                    )
                                )
                        except Exception as exc:
                            logger.warning(f"Error flushing buffered speech to Gemini Live: {exc}")

            elif ev_name == "speech_end" and ev.get("utterance_wav") is not None:
                utt_wav = ev["utterance_wav"]
                was_already_streamed = self._streaming_direct_to_gemini
                buffered_bytes = b"".join(self._buffered_speech_chunks)
                self._buffered_speech_chunks = []
                self._streaming_direct_to_gemini = False
                if was_already_streamed and stream_to_gemini_live and self.gemini_session is not None:
                    # Send trailing silence so Gemini Live server VAD immediately finalizes the turn
                    try:
                        silence = np.zeros(1600, dtype=np.int16).tobytes()
                        for _ in range(10):
                            await self.gemini_session.send_realtime_input(
                                audio=types.Blob(data=silence, mime_type="audio/pcm;rate=16000")
                            )
                    except Exception as exc:
                        logger.warning(f"Error sending trailing silence to Gemini Live: {exc}")
                hint = transcript_hint or "".join(self.current_user_transcript).strip()
                asyncio.create_task(
                    self.evaluate_and_emit_noul(
                        utt_wav,
                        transcript_hint=hint,
                        language_hint=language_hint,
                        forward_to_gemini_if_turn=(stream_to_gemini_live and not was_already_streamed),
                        raw_pcm16_bytes=buffered_bytes if not was_already_streamed else None,
                    )
                )

        # 2. Gate audio to Gemini 3.8 Live API:
        # - If _streaming_direct_to_gemini is True (floor is open OR substantive barge-in committed), stream immediately.
        # - If model is speaking and speech hasn't committed yet, buffer chunks so backchannels NEVER interrupt the model.
        if stream_to_gemini_live and self.gemini_session is not None:
            if self._streaming_direct_to_gemini:
                try:
                    await self.gemini_session.send_realtime_input(
                        audio=types.Blob(
                            data=pcm16_bytes,
                            mime_type="audio/pcm;rate=16000",
                        )
                    )
                except Exception as exc:
                    logger.warning(f"Error streaming audio to Gemini Live: {exc}")
            elif vad_result.get("state") == "speech":
                self._buffered_speech_chunks.append(pcm16_bytes)
            elif not self.is_model_speaking:
                try:
                    await self.gemini_session.send_realtime_input(
                        audio=types.Blob(
                            data=pcm16_bytes,
                            mime_type="audio/pcm;rate=16000",
                        )
                    )
                except Exception as exc:
                    logger.warning(f"Error streaming silence to Gemini Live: {exc}")

    async def _consume_gemini_messages(self, session) -> None:
        while not self._closed and self.gemini_session is session:
            async for message in session.receive():
                if self._closed or self.gemini_session is not session:
                    break
                sc = message.server_content
                if sc is None:
                    continue

                # 1. User Input Audio Transcription
                if sc.input_transcription and sc.input_transcription.text:
                    txt = sc.input_transcription.text
                    self.current_user_transcript.append(txt)
                    full_user_text = "".join(self.current_user_transcript).strip()
                    await self.send_json(
                        {
                            "type": "transcript",
                            "role": "user",
                            "text": full_user_text,
                            "delta": txt,
                            "is_final": bool(sc.input_transcription.finished),
                            "turn_id": self.turn_counter,
                            "is_backchannel": False,
                            "noul_decision": "false",
                        }
                    )

                # 2. Model Output Audio Transcription
                if sc.output_transcription and sc.output_transcription.text:
                    if self.current_user_transcript:
                        full_user_text = "".join(self.current_user_transcript).strip()
                        await self.send_json(
                            {
                                "type": "transcript",
                                "role": "user",
                                "text": full_user_text,
                                "is_final": True,
                                "turn_id": self.turn_counter,
                                "is_backchannel": False,
                                "noul_decision": "false",
                            }
                        )
                        self.current_user_transcript.clear()

                    self.is_model_speaking = True
                    txt = sc.output_transcription.text
                    self.current_model_transcript.append(txt)
                    full_model_text = "".join(self.current_model_transcript).strip()
                    await self.send_json(
                        {
                            "type": "transcript",
                            "role": "model",
                            "text": full_model_text,
                            "delta": txt,
                            "is_final": bool(sc.output_transcription.finished),
                            "turn_id": self.turn_counter,
                            "voice": VOICE_NAME,
                            "language_code": LANGUAGE_CODE,
                        }
                    )

                # 3. Model 24kHz Native Audio Output
                if sc.model_turn and sc.model_turn.parts:
                    self.is_model_speaking = True
                    for part in sc.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            b64_audio = base64.b64encode(part.inline_data.data).decode("ascii")
                            await self.send_json(
                                {
                                    "type": "model_audio",
                                    "pcm24_b64": b64_audio,
                                    "sample_rate": 24000,
                                    "turn_id": self.turn_counter,
                                }
                            )

                # 4. Interruption handling
                if sc.interrupted:
                    self.is_model_speaking = False
                    await self.send_json(
                        {
                            "type": "turn_event",
                            "event": "interrupted",
                            "turn_id": self.turn_counter,
                        }
                    )

                # 5. Turn completion
                if sc.turn_complete:
                    self.is_model_speaking = False
                    if self.current_model_transcript:
                        full_model_text = "".join(self.current_model_transcript).strip()
                        await self.send_json(
                            {
                                "type": "transcript",
                                "role": "model",
                                "text": full_model_text,
                                "is_final": True,
                                "turn_id": self.turn_counter,
                                "voice": VOICE_NAME,
                                "language_code": LANGUAGE_CODE,
                            }
                        )
                        self.current_model_transcript.clear()
                    await self.send_json(
                        {
                            "type": "turn_event",
                            "event": "turn_complete",
                            "turn_id": self.turn_counter,
                        }
                    )
                    self.turn_counter += 1

    async def gemini_session_manager(self) -> None:
        """Maintains a resilient Gemini 3.8 Live API session with automatic GoAway reconnection."""
        config = build_gemini_live_connect_config()
        while not self._closed:
            try:
                async with self.client.aio.live.connect(model=MODEL_ID, config=config) as session:
                    self.gemini_session = session
                    await self.send_json(
                        {
                            "type": "session_status",
                            "status": "connected",
                            "config": get_live_config_summary(),
                            "message": f"Connected · {MODEL_ID} · Voice: {VOICE_NAME} ({LANGUAGE_CODE}) · ADC",
                        }
                    )
                    await self._consume_gemini_messages(session)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.gemini_session = None
                if self._closed:
                    break
                logger.warning(f"Gemini Live session reconnecting after: {exc}")
                await asyncio.sleep(0.4)

    async def handle_simulate_utterance(self, preset: str) -> None:
        """Streams a real 16kHz spoken fixture through Silero VAD -> Prosodia Noul -> Gemini 3.8 Live."""
        wav_16k, meta = synthesize_test_utterance(preset)
        transcript_hint = meta.get("transcript", "")
        lang_hint = meta.get("language", "en-IN")
        is_bc_preset = meta.get("expected_backchannel", False)

        # Pad with 480ms of trailing silence so Silero VAD cleanly fires speech_end
        trailing_silence = np.zeros(int(16000 * 0.48), dtype=np.float32)
        full_wav = np.concatenate([wav_16k, trailing_silence])
        pcm16_all = (np.clip(full_wav, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        speech_pcm16 = (np.clip(wav_16k, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

        # Stream in 32ms (512 samples = 1024 bytes) frames through Silero VAD
        frame_bytes = 1024
        for offset in range(0, len(pcm16_all), frame_bytes):
            chunk = pcm16_all[offset : offset + frame_bytes]
            # Only forward substantive turns directly to Gemini Live so backchannels don't trigger unnecessary replies
            await self.process_pcm16_stream(
                chunk,
                stream_to_gemini_live=(not is_bc_preset),
                transcript_hint=transcript_hint,
                language_hint=lang_hint,
            )
            await asyncio.sleep(0.012)

        if not is_bc_preset and self.gemini_session is not None:
            # Send trailing silence to Gemini Live to finalize turn quickly
            silence_chunk = np.zeros(1600, dtype=np.int16).tobytes()
            for _ in range(12):
                await self.gemini_session.send_realtime_input(
                    audio=types.Blob(data=silence_chunk, mime_type="audio/pcm;rate=16000")
                )
                await asyncio.sleep(0.02)

    async def run(self) -> None:
        await self.ws.accept()
        await self.send_json(
            {
                "type": "session_status",
                "status": "connecting",
                "config": get_live_config_summary(),
                "message": "Connecting to Gemini 3.8 Live API on Agent Platform via ADC...",
            }
        )
        session_task = asyncio.create_task(self.gemini_session_manager())
        try:
            while True:
                raw_msg = await self.ws.receive_text()
                data = json.loads(raw_msg)
                msg_type = data.get("type")

                if msg_type == "audio_chunk":
                    b64 = data.get("pcm16_b64", "")
                    if b64:
                        pcm_bytes = base64.b64decode(b64)
                        await self.process_pcm16_stream(
                            pcm_bytes,
                            stream_to_gemini_live=True,
                            transcript_hint=data.get("transcript_hint", ""),
                            language_hint=data.get("language_hint", ""),
                        )

                elif msg_type == "stop_speaking":
                    was_already_streamed = self._streaming_direct_to_gemini
                    buffered_bytes = b"".join(self._buffered_speech_chunks)
                    self._buffered_speech_chunks = []
                    self._streaming_direct_to_gemini = False
                    flush_ev = self.vad.flush()
                    if flush_ev is not None:
                        ev_copy = {k: v for k, v in flush_ev.items() if k != "utterance_wav"}
                        await self.send_json({"type": "vad_event", **ev_copy})
                        if flush_ev.get("utterance_wav") is not None:
                            hint = "".join(self.current_user_transcript).strip()
                            asyncio.create_task(
                                self.evaluate_and_emit_noul(
                                    flush_ev["utterance_wav"],
                                    transcript_hint=hint,
                                    forward_to_gemini_if_turn=(not was_already_streamed),
                                    raw_pcm16_bytes=buffered_bytes if not was_already_streamed else None,
                                )
                            )
                    if self.gemini_session is not None:
                        silence = np.zeros(1600, dtype=np.int16).tobytes()
                        for _ in range(10):
                            await self.gemini_session.send_realtime_input(
                                audio=types.Blob(data=silence, mime_type="audio/pcm;rate=16000")
                            )
                            await asyncio.sleep(0.015)

                elif msg_type == "simulate_utterance":
                    preset = data.get("preset", "en_in_backchannel")
                    asyncio.create_task(self.handle_simulate_utterance(preset))

                elif msg_type == "ping":
                    await self.send_json({"type": "pong", "timestamp": time.time()})

        except WebSocketDisconnect:
            logger.info("Client WebSocket disconnected.")
        finally:
            self._closed = True
            session_task.cancel()


@app.websocket("/ws/live")
async def ws_live_endpoint(websocket: WebSocket):
    global prosodia_engine
    if prosodia_engine is None:
        prosodia_engine = ProsodiaNoulBackchannelEngine(cache_dir="/tmp/hf-cache")
    coordinator = LiveConversationCoordinator(websocket, prosodia_engine)
    await coordinator.run()


@app.get("/")
async def index_page():
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
