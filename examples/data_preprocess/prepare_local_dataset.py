"""
Prepare a local dataset for T5Gemma-TTS training using XCodec2 acoustic tokens.

This script converts local audio/transcript data into the format expected by
the training scripts.

Input structure:
    <input_dir>/
        transcript_utf8.txt   # Format: "utt_id:text" per line
        wav/<utt_id>.wav      # Audio files

Output structure:
    <output_dir>/
        text/<shard_id>/<utt_id>.txt
        xcodec2_1cb/<shard_id>/<utt_id>.txt
        manifest_final/<split>.txt
        neighbors/<utt_id>.txt

Usage (example):
    python examples/data_preprocess/prepare_local_dataset.py \\
        --input-dir raw_data/RISE \\
        --output-dir datasets/rise \\
        --speaker-name RISE \\
        --encoder-device cuda:0 \\
        --valid-ratio 0.05
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from tqdm import tqdm

# Project root (two levels up from this script)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.tokenizer import AudioTokenizer

LOGGER = logging.getLogger("prepare_local_dataset")


@dataclass
class Sample:
    utt_id: str
    text: str
    audio_path: Path
    duration_sec: float = 0.0
    token_len: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local audio dataset for T5Gemma-TTS training."
    )

    # Input/Output
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing transcript_utf8.txt and wav/ folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed data.",
    )

    # Dataset configuration
    parser.add_argument(
        "--transcript-file",
        type=str,
        default="transcript_utf8.txt",
        help="Name of the transcript file.",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="wav",
        help="Name of the audio subdirectory.",
    )
    parser.add_argument(
        "--audio-ext",
        type=str,
        default=".wav",
        help="Audio file extension.",
    )
    parser.add_argument(
        "--speaker-name",
        type=str,
        default="speaker",
        help="Speaker name for neighbor grouping.",
    )

    # Processing configuration
    parser.add_argument(
        "--codec-sample-rate",
        type=int,
        default=16000,
        help="Sample rate for audio encoding.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="NandemoGHS/Anime-XCodec2-44.1kHz-v2",
        help="XCodec2 model for acoustic tokenization.",
    )
    parser.add_argument(
        "--encoder-device",
        type=str,
        default=None,
        help="Device for encoding (e.g., 'cuda:0', 'cpu', or 'auto').",
    )

    # Split configuration
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.0,
        help="Ratio of samples for validation split.",
    )
    parser.add_argument(
        "--train-split-name",
        type=str,
        default="train",
        help="Name for training split.",
    )
    parser.add_argument(
        "--valid-split-name",
        type=str,
        default="valid",
        help="Name for validation split.",
    )

    # Filtering
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.1,
        help="Minimum audio duration in seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds.",
    )

    # Neighbor configuration
    parser.add_argument(
        "--neighbor-folder",
        type=str,
        default="neighbors",
        help="Folder name for neighbor files.",
    )
    parser.add_argument(
        "--max-neighbors-per-utt",
        type=int,
        default=None,
        help="Maximum number of neighbors per utterance.",
    )
    parser.add_argument(
        "--encodec-sr",
        type=float,
        default=50.0,
        help="Tokens per second for duration estimation.",
    )

    # Misc
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log progress every N samples.",
    )

    return parser.parse_args()


def load_transcript(transcript_path: Path) -> Dict[str, str]:
    """Load transcript file into a dictionary.
    
    Supports formats:
        - "utt_id:text"
        - "utt_id|text"
        - "utt_id\ttext"
    """
    transcript = {}
    with transcript_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Try different separators
            parts = None
            for sep in [":", "|", "\t"]:
                if sep in line:
                    parts = line.split(sep, 1)
                    break

            if parts is None or len(parts) != 2:
                LOGGER.warning(
                    "Skipping malformed line %d: %s",
                    line_num,
                    line[:50] + "..." if len(line) > 50 else line,
                )
                continue

            utt_id, text = parts
            utt_id = utt_id.strip()
            text = text.strip()

            if not utt_id or not text:
                LOGGER.warning("Skipping empty utt_id or text at line %d", line_num)
                continue

            transcript[utt_id] = text

    return transcript


def load_audio_tensor(
    audio_path: Path, target_sr: int
) -> Tuple[torch.Tensor, int, float]:
    """Load audio file and resample if needed.
    
    Returns:
        waveform: Audio tensor [1, samples]
        sample_rate: Sample rate
        duration_sec: Duration in seconds
    """
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    # Normalize
    waveform = waveform * 0.99

    duration_sec = waveform.shape[-1] / sr

    return waveform, sr, duration_sec


def get_shard_id(utt_id: str) -> str:
    """Get shard ID from utterance ID (first 2 hex digits of MD5 hash)."""
    return hashlib.md5(utt_id.encode("utf-8")).hexdigest()[:2]


def ensure_dirs(base: Path) -> Dict[str, Path]:
    """Create output directories."""
    dirs = {
        "text": base / "text",
        "codes": base / "xcodec2_1cb",
        "manifest": base / "manifest_final",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_outputs(
    dirs: Dict[str, Path],
    split: str,
    utt_id: str,
    tokens: torch.Tensor,
    text: str,
    overwrite: bool,
) -> int:
    """Write text, tokens, and manifest entry.
    
    Returns:
        Token length
    """
    shard_id = get_shard_id(utt_id)
    text_parent_dir = dirs["text"] / shard_id
    codes_parent_dir = dirs["codes"] / shard_id
    text_parent_dir.mkdir(exist_ok=True)
    codes_parent_dir.mkdir(exist_ok=True)

    text_path = text_parent_dir / f"{utt_id}.txt"
    codes_path = codes_parent_dir / f"{utt_id}.txt"

    if not overwrite and (text_path.exists() or codes_path.exists()):
        raise FileExistsError(
            f"Destination files already exist for {utt_id}; rerun with --overwrite."
        )

    # Write text
    text_path.write_text(text.strip() + "\n", encoding="utf-8")

    # Write tokens
    tokens_np = tokens.cpu().numpy()
    if tokens_np.ndim == 1:
        tokens_np = tokens_np[None, :]
    elif tokens_np.ndim == 2 and tokens_np.shape[0] > tokens_np.shape[1]:
        tokens_np = tokens_np.T

    lines = [" ".join(str(int(tok)) for tok in row.tolist()) for row in tokens_np]
    codes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Append to manifest
    manifest_entry = f"{shard_id}/{utt_id}\t{tokens_np.shape[-1]}\n"
    manifest_path = dirs["manifest"] / f"{split}.txt"
    with manifest_path.open("a", encoding="utf-8") as mf:
        mf.write(manifest_entry)

    return tokens_np.shape[-1]


def resolve_device(device_spec: Optional[str]) -> torch.device:
    """Resolve device specification."""
    if device_spec is None or device_spec.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    return torch.device(device_spec)


def generate_neighbors(
    output_root: Path,
    samples: List[Sample],
    speaker_name: str,
    neighbor_folder: str,
    max_neighbors: Optional[int],
    overwrite: bool,
    seed: int,
) -> None:
    """Generate neighbor files for voice prompting."""
    neighbor_dir = output_root / neighbor_folder
    if overwrite and neighbor_dir.exists():
        for file in neighbor_dir.glob("*.txt"):
            try:
                file.unlink()
            except OSError:
                LOGGER.warning("Failed to remove %s", file, exc_info=True)
    neighbor_dir.mkdir(parents=True, exist_ok=True)

    # Group by speaker (all samples belong to the same speaker in this case)
    groups: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        groups[speaker_name].append(sample)

    # Sort by utt_id within each group
    for group_samples in groups.values():
        group_samples.sort(key=lambda s: s.utt_id)

    rng = random.Random(seed)
    total_written = 0
    total_empty = 0

    for sample in tqdm(samples, desc="writing neighbors"):
        neighbors = [s for s in groups[speaker_name] if s.utt_id != sample.utt_id]
        neighbor_path = neighbor_dir / f"{sample.utt_id}.txt"

        if not neighbors:
            neighbor_path.touch(exist_ok=True)
            total_empty += 1
            continue

        # Sort by duration difference
        neighbors.sort(key=lambda n: abs(n.duration_sec - sample.duration_sec))

        # Subsample if needed
        if max_neighbors is not None and len(neighbors) > max_neighbors:
            limit = max_neighbors
            stride = len(neighbors) / float(limit)
            sampled: List[Sample] = []
            for i in range(limit):
                pos = int(rng.uniform(i * stride, (i + 1) * stride))
                pos = max(0, min(len(neighbors) - 1, pos))
                sampled.append(neighbors[pos])
            neighbors = sampled

        with neighbor_path.open("w", encoding="utf-8") as nf:
            for neighbor in neighbors:
                distance_val = abs(neighbor.duration_sec - sample.duration_sec)
                nf.write(
                    f"{neighbor.utt_id}.txt\t{distance_val:.3f}\t{neighbor.duration_sec:.3f}\n"
                )
        total_written += 1

    LOGGER.info(
        "Neighbor generation complete. Non-empty=%d, empty=%d, output_dir=%s",
        total_written,
        total_empty,
        neighbor_dir,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    LOGGER.info("Processing local dataset with args: %s", args)

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    # Load transcript
    transcript_path = input_dir / args.transcript_file
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    transcript = load_transcript(transcript_path)
    LOGGER.info("Loaded %d entries from transcript", len(transcript))

    # Find audio files
    audio_dir = input_dir / args.audio_dir
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    samples: List[Sample] = []
    for utt_id, text in transcript.items():
        audio_path = audio_dir / f"{utt_id}{args.audio_ext}"
        if not audio_path.exists():
            LOGGER.warning("Audio file not found for %s: %s", utt_id, audio_path)
            continue
        samples.append(Sample(utt_id=utt_id, text=text, audio_path=audio_path))

    LOGGER.info("Found %d samples with matching audio files", len(samples))

    if not samples:
        LOGGER.error("No valid samples found. Exiting.")
        return

    # Setup output directories
    dirs = ensure_dirs(output_root)

    # Clear manifests if overwriting
    if args.overwrite:
        for manifest_file in dirs["manifest"].glob("*.txt"):
            manifest_file.unlink()

    # Setup tokenizer
    device = resolve_device(args.encoder_device)
    LOGGER.info("Using device: %s", device)

    tokenizer = AudioTokenizer(
        device=device,
        backend="xcodec2",
        model_name=args.tokenizer_model,
    )

    default_encode_sr = getattr(
        tokenizer, "encode_sample_rate", tokenizer.sample_rate
    )
    encode_sr = args.codec_sample_rate or default_encode_sr
    if args.codec_sample_rate and args.codec_sample_rate != default_encode_sr:
        LOGGER.info(
            "Overriding codec input sample rate: tokenizer expects %d, using %d per argument.",
            default_encode_sr,
            args.codec_sample_rate,
        )
        tokenizer.encode_sample_rate = encode_sr
        tokenizer.sample_rate = encode_sr

    # Split samples
    rng = random.Random(args.seed)
    rng.shuffle(samples)

    target_splits = [args.train_split_name]
    if args.valid_ratio and args.valid_ratio > 0:
        target_splits.append(args.valid_split_name)

    # Process samples
    processed = 0
    skipped = 0
    split_counts = {split: 0 for split in target_splits}
    processed_samples: List[Sample] = []

    for sample in tqdm(samples, desc="Processing samples"):
        try:
            # Load audio
            waveform, sr, duration_sec = load_audio_tensor(
                sample.audio_path, encode_sr
            )

            # Filter by duration
            if duration_sec < args.min_duration or duration_sec > args.max_duration:
                LOGGER.debug(
                    "Skipping %s due to duration %.2f sec (min=%.2f, max=%.2f)",
                    sample.utt_id,
                    duration_sec,
                    args.min_duration,
                    args.max_duration,
                )
                skipped += 1
                continue

            # Determine split
            if args.valid_ratio and args.valid_ratio > 0:
                assign_valid = rng.random() < args.valid_ratio
                dest_split = args.valid_split_name if assign_valid else args.train_split_name
            else:
                dest_split = args.train_split_name

            # Encode audio
            waveform = waveform.to(device)
            with torch.no_grad():
                codes = tokenizer.encode(waveform)
            codes = codes.squeeze(0)

            if codes.numel() == 0:
                LOGGER.warning("Empty codes for %s, skipping", sample.utt_id)
                skipped += 1
                continue

            # Write outputs
            token_len = write_outputs(
                dirs=dirs,
                split=dest_split,
                utt_id=sample.utt_id,
                tokens=codes.cpu(),
                text=sample.text,
                overwrite=args.overwrite,
            )

            sample.duration_sec = duration_sec
            sample.token_len = token_len
            processed_samples.append(sample)

            split_counts[dest_split] += 1
            processed += 1

            if processed % args.log_every == 0:
                LOGGER.info(
                    "Processed %d samples (skipped %d) – last utt_id=%s len=%d tokens",
                    processed,
                    skipped,
                    sample.utt_id,
                    token_len,
                )

        except Exception as exc:
            LOGGER.warning(
                "Error processing %s: %s",
                sample.utt_id,
                exc,
                exc_info=True,
            )
            skipped += 1

    # Log results
    for split_name in target_splits:
        LOGGER.info(
            "Split %s: %d samples",
            split_name,
            split_counts[split_name],
        )
    LOGGER.info("Overall processed=%d, skipped=%d", processed, skipped)

    # Generate neighbors
    if processed_samples:
        generate_neighbors(
            output_root=output_root,
            samples=processed_samples,
            speaker_name=args.speaker_name,
            neighbor_folder=args.neighbor_folder,
            max_neighbors=args.max_neighbors_per_utt,
            overwrite=args.overwrite,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

