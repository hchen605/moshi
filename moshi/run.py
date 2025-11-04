#!/usr/bin/env python3
"""
Run Moshi+Mimi on a real audio file and save the generated audio.

Usage:
  python run_moshi_stream.py --in input.wav --out output.wav
"""

import argparse
import math
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders, LMGen
import soundfile as sf  # pip install soundfile
import numpy as np

TARGET_SR = 24000

def load_audio_24k_mono(path: str):
    """Load any audio -> [1,1,T] float32 at 24kHz, mono. Avoids TorchCodec requirement."""
    try:
        # Try torchaudio first (works if torchcodec is installed)
        wav, sr = torchaudio.load(path)  # [C, T]
        wav = wav.to(torch.float32)
    except ImportError:
        # Fallback to soundfile
        data, sr = sf.read(path, always_2d=True, dtype="float32")  # [T, C]
        wav = torch.from_numpy(data.T)  # -> [C, T]
    # mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # resample to 24k
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        sr = TARGET_SR
    # [B, C, T]
    return wav.unsqueeze(0), sr  # [1,1,T]

def save_wav_24k(path: str, wav_bcT: torch.Tensor):
    """Save [B,1,T] to 24kHz WAV via soundfile (no torchcodec)."""
    x = wav_bcT.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)  # [T]
    sf.write(path, x, TARGET_SR, subtype="PCM_16")

def pad_to_multiple(x: torch.Tensor, multiple: int):
    """x is [B, C, T]; returns padded x and pad amount."""
    T = x.shape[-1]
    pad = (multiple - (T % multiple)) % multiple
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x, pad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="Input audio file (any format torchaudio can read).")
    parser.add_argument("--out", dest="out", required=True, help="Output .wav path.")
    parser.add_argument("--temp", type=float, default=0.8, help="LM temperature for acoustic tokens.")
    parser.add_argument("--temp_text", type=float, default=0.7, help="LM temperature for text token.")
    parser.add_argument("--codebooks", type=int, default=8, help="# of codebooks for Mimi (max 8 for Moshi).")
    args = parser.parse_args()

    # --------- Load and prep audio ---------
    wav, sr = load_audio_24k_mono(args.inp)  # [1,1,T] at 24 kHz
    assert sr == 24000, "Resampling to 24 kHz failed."

    # --------- Load Mimi (codec) ---------
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device='cpu')
    mimi.set_num_codebooks(args.codebooks)  # up to 32 for mimi, Moshi limited to 8

    # Frame size for streaming with Mimi
    frame_size = mimi.frame_size  # 1920 samples @ 24 kHz
    wav, pad = pad_to_multiple(wav, frame_size)  # ensure full frames for streaming

    # --------- Encode to codes in streaming mode ---------
    all_codes = []
    with torch.no_grad(), mimi.streaming(batch_size=1):
        for offset in range(0, wav.shape[-1], frame_size):
            frame = wav[:, :, offset: offset + frame_size]  # [1,1,frame_size]
            codes = mimi.encode(frame)  # [B, K, 1] in streaming mode
            assert codes.shape[-1] == 1, f"Unexpected codes shape: {codes.shape}"
            all_codes.append(codes)

    # --------- Load Moshi LM ---------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        mimi = mimi.to(device)
    moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    moshi = loaders.get_moshi_lm(moshi_weight, device=device)
    lm_gen = LMGen(moshi, temp=args.temp, temp_text=args.temp_text)

    # --------- Run LM streaming + Mimi decode-on-the-fly ---------
    out_wav_chunks = []
    with torch.no_grad(), lm_gen.streaming(1), mimi.streaming(1):
        for idx, code in enumerate(all_codes):
            code = code.to(device)
            tokens_out = lm_gen.step(code)  # [B, 1+K, 1]; tokens_out[:, 1] == text token
            if tokens_out is not None:
                # Decode only the K acoustic token channels
                wav_chunk = mimi.decode(tokens_out[:, 1:])  # [B, 1, frame_size]
                out_wav_chunks.append(wav_chunk.to("cpu"))
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx+1}/{len(all_codes)} frames...", end="\r")

    if not out_wav_chunks:
        raise RuntimeError("No output audio chunks were produced.")

    out_wav = torch.cat(out_wav_chunks, dim=-1)  # [B, 1, T]
    # Remove any padding we introduced on input length to keep symmetry
    if pad:
        # Output is generated per input frame; keep full length (often desirable).
        # If you want to strictly remove trailing padding, uncomment below:
        # out_wav = out_wav[:, :, :-pad]
        pass

    # --------- Save to .wav (24 kHz) ---------
    out_path = args.out
    out_audio = out_wav.squeeze(0)  # [1, T]
    out_audio = torch.clamp(out_audio, -1.0, 1.0).to(torch.float32)
    #torchaudio.save(out_path, out_audio, sample_rate=24000)
    save_wav_24k(args.out, out_wav)
    print(f"\nFlow finished. Saved to: {out_path}")

if __name__ == "__main__":
    main()
