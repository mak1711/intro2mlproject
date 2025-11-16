"""Record from microphone and transcribe with OpenAI Whisper.

Usage examples:
  python -m text_classification.whisper_mic --model small --duration 5
  python -m text_classification.whisper_mic --model base --duration 10 --language ar

Notes:
 - Requires `whisper` and `sounddevice` and `soundfile` Python packages.
   Install with: pip install -U openai-whisper sounddevice soundfile
 - The script records a short clip (default 5s) and sends it to Whisper for transcription.
 - If --language is `auto` (default), Whisper will detect the language. Use `en` or `ar` to force.
"""

import argparse
import tempfile
import os
import sys

try:
    import whisper
except Exception as e:
    print("Missing required package 'whisper'. Install: pip install -U openai-whisper")
    raise

try:
    import sounddevice as sd
    import soundfile as sf
except Exception:
    print("Missing required audio packages. Install: pip install sounddevice soundfile")
    raise


def record(duration: float, out_path: str, samplerate: int = 16000):
    """Record `duration` seconds from default microphone and save to out_path (mono WAV)."""
    print(f"Recording {duration:.1f}s to {out_path} (samplerate={samplerate})...")
    try:
        data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
    except Exception as e:
        print("Microphone recording failed:", e)
        raise
    # data is Nx1 float32, soundfile accepts numpy array
    sf.write(out_path, data, samplerate)
    print("Recording finished.")


def transcribe(model_name: str, audio_path: str, language: str = 'auto', task: str = 'transcribe'):
    """Load Whisper model and transcribe the given audio file.

    language: 'auto' (detect), or language code like 'en' or 'ar'.
    task: 'transcribe' or 'translate'
    """
    print(f"Loading Whisper model '{model_name}' (this may take a bit)...")
    model = whisper.load_model(model_name)

    whisper_opts = {}
    if language is not None and language != 'auto':
        whisper_opts['language'] = language

    print(f"Transcribing {audio_path} (language={language})...")
    result = model.transcribe(audio_path, task=task, **whisper_opts)
    # result contains 'text' and 'segments' and possibly 'language'
    text = result.get('text', '').strip()
    detected_lang = result.get('language', None)
    return text, detected_lang, result


def main(argv=None):
    p = argparse.ArgumentParser(description='Record from mic and transcribe using Whisper')
    p.add_argument('--model', default='small', help='whisper model name (tiny, base, small, medium, large)')
    p.add_argument('--duration', type=float, default=5.0, help='seconds to record from mic')
    p.add_argument('--samplerate', type=int, default=16000, help='audio sampling rate')
    p.add_argument('--out', default=None, help='optional output wav path (if omitted a temp file is used)')
    p.add_argument('--language', default='auto', help="language code ('en','ar') or 'auto' to detect")
    p.add_argument('--task', default='transcribe', choices=['transcribe', 'translate'], help='whisper task')
    p.add_argument('--interactive', action='store_true', help='interactive start/stop recording (press Enter to start, Enter to stop)')
    p.add_argument('--ckpt', default=None, help='optional path to classifier checkpoint to run on the transcribed text')
    p.add_argument('--topk', type=int, default=3, help='how many top classifier predictions to show (when --ckpt is used)')
    p.add_argument('--save-text', default=None, help='optional path to save the transcribed text')
    args = p.parse_args(argv)

    out_path = args.out
    tmp_file = None
    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        out_path = tmp.name
        tmp.close()
        tmp_file = out_path

    try:
        if args.interactive:
            # Interactive mode: press Enter to start, Enter to stop
            input('Press Enter to START recording...')
            print('Opening microphone stream...')
            try:
                with sf.SoundFile(out_path, mode='w', samplerate=args.samplerate, channels=1) as f:
                    with sd.InputStream(samplerate=args.samplerate, channels=1, callback=lambda indata, frames, time, status: f.write(indata.copy())):
                        print('Recording... Press Enter to STOP')
                        input()
            except Exception as e:
                print('Interactive recording failed:', e)
                raise
        else:
            record(args.duration, out_path, samplerate=args.samplerate)

        text, detected_lang, raw = transcribe(args.model, out_path, language=args.language, task=args.task)
        print('\n=== Transcription ===')
        if detected_lang:
            print(f"Detected language: {detected_lang}")
        print(text)
        print('=== End ===\n')

        # Optionally save transcribed text for later inspection
        if args.save_text:
            try:
                with open(args.save_text, 'w', encoding='utf-8') as f:
                    f.write(text + '\n')
                print(f'Saved transcription to {args.save_text}')
            except Exception as e:
                print('Failed to save transcription:', e)

        # Optionally run classifier on the transcribed text
        if args.ckpt:
            try:
                from text_classification.predict import load_checkpoint, prepare_model_from_checkpoint, predict_text
                device = None
                # prefer cuda if available
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                cp = load_checkpoint(args.ckpt, device)
                model, vocab, id2label = prepare_model_from_checkpoint(cp, max_len=32, device=device)
                preds = predict_text(model, vocab, id2label, text, device, max_len=32, topk=args.topk)
                print('=== Classifier predictions ===')
                for lbl, prob in preds:
                    print(f"{lbl}\t{prob:.4f}")
                print('=== End classifier ===\n')
            except Exception as e:
                print('Failed to run classifier on transcription:', e)

    finally:
        if tmp_file is not None and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass


if __name__ == '__main__':
    main()
