import os
import io
import math
import wave
import struct
import json
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

PORT = 8888
VOICEVOX_URL = "http://127.0.0.1:50121"
SPEAKER_ID = 10002

def extract_f0_track(samples, sr, frame_size=1024, hop_size=256, min_f0=65, max_f0=500):
    """Returns (f0_track, active_duration_sec).

    f0_track is a per-hop-frame array, in the original time order, covering the
    trimmed speech region only. Unvoiced/silent frames are NaN rather than
    being dropped, so the array's index still corresponds to real time -
    that's what lets us line it up against VOICEVOX's own mora durations.
    """
    min_lag = int(sr / max_f0)
    max_lag = int(sr / min_f0)

    # Trim silence (energy-based VAD)
    energy = samples ** 2
    window_len = max(1, int(sr * 0.05))
    rolling_energy = np.convolve(energy, np.ones(window_len)/window_len, mode='same')
    threshold = np.max(rolling_energy) * 0.03
    active_indices = np.where(rolling_energy > threshold)[0]

    if len(active_indices) > 0:
        start_idx = max(0, active_indices[0] - int(sr * 0.02))
        end_idx = min(len(samples), active_indices[-1] + int(sr * 0.02))
        samples = samples[start_idx:end_idx]

    if len(samples) < frame_size:
        return np.array([]), 0.0

    n_frames = max(1, (len(samples) - frame_size) // hop_size + 1)
    f0_track = np.full(n_frames, np.nan)

    for fi, i in enumerate(range(0, len(samples) - frame_size + 1, hop_size)):
        frame = samples[i:i + frame_size] * np.hanning(frame_size)

        rms = np.sqrt(np.mean(frame**2))
        if rms < 0.01:
            continue

        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]

        search_region = corr[min_lag:max_lag]
        if len(search_region) == 0:
            continue

        peak_idx = np.argmax(search_region) + min_lag
        peak_val = corr[peak_idx]
        zero_lag_val = corr[0]

        if peak_val > 0.35 * zero_lag_val:
            if 0 < peak_idx < len(corr) - 1:
                alpha = corr[peak_idx - 1]
                beta = corr[peak_idx]
                gamma = corr[peak_idx + 1]
                denom = 2 * (2 * beta - alpha - gamma)
                if denom != 0:
                    delta = (alpha - gamma) / denom
                    precise_lag = peak_idx + delta
                else:
                    precise_lag = peak_idx
            else:
                precise_lag = peak_idx

            f0 = sr / precise_lag
            if min_f0 <= f0 <= max_f0:
                f0_track[fi] = f0

    active_duration_sec = len(samples) / sr
    return f0_track, active_duration_sec

def map_pitch_to_query(query_data, f0_track, sensitivity=1.3):
    valid_f0 = f0_track[~np.isnan(f0_track)] if len(f0_track) else np.array([])
    if len(valid_f0) < 2:
        return query_data, 0

    accent_phrases = query_data.get("accent_phrases", [])

    all_moras = []
    for p_idx, phrase in enumerate(accent_phrases):
        for m_idx, mora in enumerate(phrase.get("moras", [])):
            all_moras.append((p_idx, m_idx, mora))

    num_moras = len(all_moras)
    if num_moras == 0:
        return query_data, 0

    # Use VOICEVOX's own predicted phoneme durations to decide how much of the
    # recorded time-track each mora "owns" - a plain equal split across moras
    # ignores that morae take different amounts of time to say.
    durations = []
    for _, _, mora in all_moras:
        d = (mora.get("consonant_length") or 0.0) + (mora.get("vowel_length") or 0.0)
        durations.append(max(d, 0.03))
    total_dur = sum(durations)

    n_frames = len(f0_track)
    boundaries = [0]
    acc = 0.0
    for d in durations:
        acc += d
        boundaries.append(int(round(acc / total_dur * n_frames)))

    user_median_log = np.median(np.log(valid_f0))

    mora_pitches = []
    for i in range(num_moras):
        seg = f0_track[boundaries[i]:boundaries[i + 1]]
        seg = seg[~np.isnan(seg)]
        mora_pitches.append(float(np.median(seg)) if len(seg) > 0 else None)

    # Voiceless-consonant-only moras (e.g. "し", "か") can land with zero
    # voiced frames in their slot - fill those from the nearest voiced mora
    # instead of silently defaulting to the phrase mean.
    fallback_f0 = float(np.exp(user_median_log))
    for i in range(num_moras):
        if mora_pitches[i] is not None:
            continue
        left = next((mora_pitches[j] for j in range(i - 1, -1, -1) if mora_pitches[j] is not None), None)
        right = next((mora_pitches[j] for j in range(i + 1, num_moras) if mora_pitches[j] is not None), None)
        mora_pitches[i] = left if left is not None else (right if right is not None else fallback_f0)

    default_pitches = [m[2].get("pitch", 5.0) for m in all_moras if m[2].get("pitch", 0) > 0]
    default_mean_pitch = np.mean(default_pitches) if default_pitches else 5.0

    detected_accent = 0
    mora_idx = 0
    for p_idx, phrase in enumerate(accent_phrases):
        phrase_f0s = []
        for m_idx, mora in enumerate(phrase.get("moras", [])):
            user_f0 = mora_pitches[mora_idx]
            phrase_f0s.append(user_f0)

            log_diff = np.log(user_f0) - user_median_log
            new_pitch = default_mean_pitch + (log_diff * sensitivity)
            new_pitch = float(np.clip(new_pitch, 3.5, 6.8))
            mora["pitch"] = new_pitch
            mora_idx += 1

        if len(phrase_f0s) > 0:
            peak_mora_in_phrase = int(np.argmax(phrase_f0s)) + 1
            if peak_mora_in_phrase == len(phrase_f0s):
                phrase["accent"] = 0
                detected_accent = 0
            else:
                phrase["accent"] = peak_mora_in_phrase
                detected_accent = peak_mora_in_phrase

    query_data["accent_phrases"] = accent_phrases
    return query_data, detected_accent


class TunerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            with open("/Users/ishiiisamu/Documents/GitHub/SNStheApp-Contents/voice_tuner/index.html", "rb") as f:
                self.wfile.write(f.read())
        elif self.path.startswith("/api/user_dict"):
            # Fetch user dict words
            req = urllib.request.Request(f"{VOICEVOX_URL}/user_dict", method="GET")
            with urllib.request.urlopen(req) as res:
                dict_data = json.loads(res.read())
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(dict_data, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/api/audio_query"):
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len).decode("utf-8"))
            text = body.get("text", "")
            
            url = f"{VOICEVOX_URL}/audio_query?text={urllib.parse.quote(text)}&speaker={SPEAKER_ID}"
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req) as res:
                query_data = json.loads(res.read())
                
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(query_data, ensure_ascii=False).encode("utf-8"))
            
        elif self.path.startswith("/api/process_voice"):
            content_len = int(self.headers.get("Content-Length", 0))
            raw_data = self.rfile.read(content_len)
            
            payload = json.loads(raw_data.decode("utf-8"))
            text = payload.get("text", "")
            wav_b64 = payload.get("wav_base64", "")
            sensitivity = float(payload.get("sensitivity", 1.3))
            
            import base64
            wav_bytes = base64.b64decode(wav_b64)
            
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                sampwidth = wf.getsampwidth()
                raw_frames = wf.readframes(n_frames)
                if sampwidth == 2:
                    samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    samples = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    samples = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                
            f0_track, active_duration_sec = extract_f0_track(samples, sr)

            url = f"{VOICEVOX_URL}/audio_query?text={urllib.parse.quote(text)}&speaker={SPEAKER_ID}"
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req) as res:
                query_data = json.loads(res.read())

            query_data["speedScale"] = 0.85
            query_data["prePhonemeLength"] = 0.1
            query_data["postPhonemeLength"] = 0.1

            updated_query, detected_accent = map_pitch_to_query(query_data, f0_track, sensitivity=sensitivity)
            
            # Extract suggested katakana pronunciation
            kana_moras = []
            for phrase in updated_query.get("accent_phrases", []):
                for m in phrase.get("moras", []):
                    kana_moras.append(m["text"])
            suggested_pronunciation = "".join(kana_moras)
            
            synth_url = f"{VOICEVOX_URL}/synthesis?speaker={SPEAKER_ID}"
            req_synth = urllib.request.Request(
                synth_url,
                data=json.dumps(updated_query).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req_synth) as res_synth:
                synth_wav = res_synth.read()
                
            synth_b64 = base64.b64encode(synth_wav).decode("ascii")

            valid_f0 = f0_track[~np.isnan(f0_track)] if len(f0_track) else np.array([])
            f0_points = valid_f0[::max(1, len(valid_f0)//30)].tolist() if len(valid_f0) else []

            resp = {
                "f0_count": int(len(valid_f0)),
                "f0_points": f0_points,
                "query": updated_query,
                "detected_accent": detected_accent,
                "suggested_pronunciation": suggested_pronunciation,
                "audio_base64": synth_b64
            }
            
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))
            
        elif self.path.startswith("/api/add_dict_word"):
            content_len = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(content_len).decode("utf-8"))
            surface = payload.get("surface", "").strip()
            pronunciation = payload.get("pronunciation", "").strip()
            accent_type = int(payload.get("accent_type", 0))
            word_type = payload.get("word_type", "PROPER_NOUN")
            priority = int(payload.get("priority", 10))
            
            params = urllib.parse.urlencode({
                "surface": surface,
                "pronunciation": pronunciation,
                "accent_type": accent_type,
                "word_type": word_type,
                "priority": priority
            })
            url = f"{VOICEVOX_URL}/user_dict_word?{params}"
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req) as res:
                word_uuid = res.read().decode("utf-8").strip('"')
                
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "uuid": word_uuid}, ensure_ascii=False).encode("utf-8"))

        elif self.path.startswith("/api/delete_dict_word"):
            content_len = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(content_len).decode("utf-8"))
            word_uuid = payload.get("uuid", "").strip()
            
            url = f"{VOICEVOX_URL}/user_dict_word/{word_uuid}"
            req = urllib.request.Request(url, method="DELETE")
            with urllib.request.urlopen(req) as res:
                pass
                
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), TunerHandler)
    print(f"Voice Tuner Server running on http://localhost:{PORT}")
    server.serve_forever()
