"""
Interactive voice -> transcription -> ensemble prediction loop.

Press SPACE to record a short clip (fixed duration), Whisper will transcribe it,
then the ensemble predictor will run and print the voted command. Press 'q' to quit.

Usage:
  python -m text_classification.predict_voice_ensemble \
      --custom-model models/test_run/best.pth \
      --ensemble-dir models/ensemble \
      --whisper-model small \
      --duration 3
"""

import argparse
import sys
import tempfile
import os
import termios
import tty

from pathlib import Path
import torch

# Ensure package imports work when run from module
proj_root = str(Path(__file__).resolve().parent.parent)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from text_classification.predict_ensemble import load_custom_model, load_pretrained_models
from text_classification.ensemble import EnsembleClassifier
from text_classification.whisper_mic import record

try:
    import whisper
except Exception:
    whisper = None


def wait_for_space_or_q():
    """Wait for a single keypress: space or 'q'. Returns 'space' or 'quit'."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == ' ':
                return 'space'
            if ch in ('q', 'Q'):
                return 'quit'
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def transcribe_with_model(wmodel, audio_path: str, language: str = 'auto'):
    opts = {}
    if language and language != 'auto':
        opts['language'] = language
    # whisper returns dict with 'text'
    res = wmodel.transcribe(audio_path, **opts)
    text = res.get('text', '').strip()
    return text, res


def main():
    p = argparse.ArgumentParser(description='Voice-driven ensemble predictor (press SPACE to record)')
    p.add_argument('--custom-model', '-m', default='models/test_run/best.pth', help='Path to custom model checkpoint')
    p.add_argument('--ensemble-dir', '-e', default='models/ensemble', help='Directory containing fine-tuned pretrained models')
    p.add_argument('--device', default=None, help='torch device (cpu or cuda). Default: auto-detect')
    p.add_argument('--whisper-model', default='small', help='Whisper model id (tiny, base, small, medium, large)')
    p.add_argument('--duration', type=float, default=3.0, help='Recording duration in seconds')
    p.add_argument('--samplerate', type=int, default=16000, help='Recording samplerate')
    p.add_argument('--press-toggle', action='store_true', help='Use press-to-start/press-to-stop (press SPACE to start, SPACE to stop)')
    p.add_argument('--hold-to-record', action='store_true', help='Hold SPACE to record; release SPACE to stop (requires `keyboard` package and may need elevated permissions on Linux)')
    p.add_argument('--input-wav', default=None, help='Optional path to an existing wav file to transcribe (skip recording)')
    p.add_argument('--ros-publish', action='store_true', help='Publish predicted class id to a ROS topic (requires ROS/noetic and rospy)')
    p.add_argument('--ros-topic', default='command', help='ROS topic name to publish predicted class id (std_msgs/Int32)')
    p.add_argument('--ros-node-name', default='voice_command_node', help='ROS node name to use when publishing')
    p.add_argument('--language', default='auto', help="language code ('en','ar') or 'auto')")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    custom_model_path = Path(args.custom_model)
    if not custom_model_path.exists():
        raise FileNotFoundError(f'Custom model not found: {custom_model_path}')

    print('Loading custom model...')
    custom_model, vocab, id2label = load_custom_model(custom_model_path, device)

    print('Loading pretrained models from', args.ensemble_dir)
    pretrained_models = load_pretrained_models(args.ensemble_dir, device, len(id2label))
    print(f'Loaded {len(pretrained_models)} pretrained models')

    ensemble = EnsembleClassifier(
        custom_model=custom_model,
        custom_vocab=vocab,
        custom_id2label=id2label,
        pretrained_models=pretrained_models,
        device=device
    )

    # Prepare ROS publisher (optional)
    ros_pub = None
    label2id = {v: k for k, v in id2label.items()}
    if args.ros_publish:
        try:
            import rospy
            from std_msgs.msg import Int32

            # Initialize a ROS node in a way that doesn't override signal handlers in this script
            try:
                rospy.init_node(args.ros_node_name, anonymous=True, disable_signals=True)
            except Exception:
                # If node already initialized or master not available, continue and attempt to create publisher
                pass

            ros_pub = rospy.Publisher(args.ros_topic, Int32, queue_size=1)
            print(f'ROS publishing enabled -> topic: {args.ros_topic}')
        except Exception as e:
            print('Failed to enable ROS publishing (rospy missing or ROS master unreachable):', e)
            print('Continuing without ROS publishing.')
            args.ros_publish = False
            ros_pub = None

    # Load whisper model once
    if whisper is None:
        raise RuntimeError("Whisper package not available. Install with: pip install -U openai-whisper sounddevice soundfile")
    print(f"Loading Whisper model '{args.whisper_model}' (this may take a while)...")
    wmodel = whisper.load_model(args.whisper_model)

    print('\nInteractive voice mode:')
    print(" Press SPACE to record a short command (duration {:.1f}s).".format(args.duration))
    print(" Press 'q' to quit.")

    try:
        while True:
            # If input wav is provided, process it once and exit immediately (no waiting)
            if args.input_wav:
                if not os.path.exists(args.input_wav):
                    raise FileNotFoundError(args.input_wav)
                tmp_path = args.input_wav
                # Transcribe and predict directly
                print('Transcribing...')
                text, raw = transcribe_with_model(wmodel, tmp_path, language=args.language)
                if not text:
                    print('No transcription detected.')
                    break
                print(f"Transcribed: {text}")
                predictions, confidences, details = ensemble.predict_with_probabilities(
                    [text], use_custom=True, use_pretrained=len(pretrained_models) > 0,
                    custom_max_len=32, pretrained_max_len=128
                )
                pred = predictions[0]
                conf = confidences[0]
                print(f"Prediction: {pred} (confidence: {conf:.1%})")
                voting_detail = details['voting_details'][0]
                print('\nVoting breakdown:')
                votes = voting_detail['votes']
                for model_name in sorted(votes.keys()):
                    vote_class_id = votes[model_name]
                    vote_label = id2label[vote_class_id]
                    print(f"  {model_name:20s} -> {vote_label}")
                print(f"Agreement: {voting_detail['agreement_count']}/{voting_detail['total_models']} models")
                break

            process_once = False

            # Choose recording mode
            if args.hold_to_record:
                # Hold-to-record behavior using `keyboard`.
                try:
                    import keyboard
                except Exception:
                    print('Hold-to-record requires the `keyboard` package. Install with: pip install keyboard')
                    print('Falling back to press-toggle mode.')
                    args.press_toggle = True

            if args.input_wav is None and not args.hold_to_record:
                print('\nReady. Press SPACE to record (press-toggle) or q to quit.')

            # Wait for user action
            k = wait_for_space_or_q()
            if k == 'quit':
                print('\nQuitting.')
                break

            # Prepare temp file
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp_path = tmp.name
            tmp.close()

            try:
                # If input_wav was provided, we've already set tmp_path to that and skip recording
                if not args.input_wav:
                    if args.hold_to_record and not args.press_toggle:
                        # Use keyboard events to start/stop while holding SPACE
                        try:
                            import keyboard
                            import sounddevice as sd
                            import soundfile as sf

                            print('Hold SPACE to record, release SPACE to stop. Press q to quit.')

                            # Wait for key down event for space
                            while True:
                                ev = keyboard.read_event()
                                if ev.event_type == keyboard.KEY_DOWN and ev.name == 'space':
                                    break
                                if ev.event_type == keyboard.KEY_DOWN and ev.name in ('q', 'Q'):
                                    raise KeyboardInterrupt()

                            print('Recording... (hold SPACE)')
                            try:
                                with sf.SoundFile(tmp_path, mode='w', samplerate=args.samplerate, channels=1) as f:
                                    with sd.InputStream(samplerate=args.samplerate, channels=1, callback=lambda indata, frames, time, status: f.write(indata.copy())):
                                        # wait until space is released
                                        while True:
                                            ev2 = keyboard.read_event()
                                            if ev2.event_type == keyboard.KEY_UP and ev2.name == 'space':
                                                break
                                            if ev2.event_type == keyboard.KEY_DOWN and ev2.name in ('q', 'Q'):
                                                raise KeyboardInterrupt()
                            except Exception as e:
                                print('Recording failed:', e)
                                continue
                            print('Recording finished.')
                        except KeyboardInterrupt:
                            print('\nQuit requested while recording; exiting.')
                            break
                        except Exception as e:
                            print('Hold-to-record failed, falling back to press-toggle:', e)
                            args.press_toggle = True

                    if args.press_toggle and not args.hold_to_record:
                        # Press-to-start / press-to-stop behavior using sounddevice.InputStream
                        import sounddevice as sd
                        import soundfile as sf

                        print('Press SPACE to START recording, SPACE again to STOP')
                        print('Starting recording...')
                        try:
                            with sf.SoundFile(tmp_path, mode='w', samplerate=args.samplerate, channels=1) as f:
                                def callback(indata, frames, time, status):
                                    f.write(indata.copy())

                                with sd.InputStream(samplerate=args.samplerate, channels=1, callback=callback):
                                    print('Recording... press SPACE to STOP')
                                    k2 = wait_for_space_or_q()
                                    if k2 == 'quit':
                                        print('Quit requested while recording; stopping.')
                        except Exception as e:
                            print('Recording failed:', e)
                            continue
                        print('Recording finished.')
                    elif not args.press_toggle and not args.hold_to_record:
                        print(f'Recording {args.duration:.1f}s...')
                        record(args.duration, tmp_path, samplerate=args.samplerate)

                # Transcribe
                print('Transcribing...')
                text, raw = transcribe_with_model(wmodel, tmp_path, language=args.language)
                if not text:
                    print('No transcription detected.')
                    if args.input_wav:
                        break
                    continue
                print(f"Transcribed: {text}")

                # Run ensemble prediction
                predictions, confidences, details = ensemble.predict_with_probabilities(
                    [text], use_custom=True, use_pretrained=len(pretrained_models) > 0,
                    custom_max_len=32, pretrained_max_len=128
                )
                pred = predictions[0]
                conf = confidences[0]
                print(f"Prediction: {pred} (confidence: {conf:.1%})")

                # Print verbose voting breakdown
                voting_detail = details['voting_details'][0]
                print('\nVoting breakdown:')
                votes = voting_detail['votes']
                for model_name in sorted(votes.keys()):
                    vote_class_id = votes[model_name]
                    vote_label = id2label[vote_class_id]
                    print(f"  {model_name:20s} -> {vote_label}")
                print(f"Agreement: {voting_detail['agreement_count']}/{voting_detail['total_models']} models")

            finally:
                # remove temporary file if it was created by us and not the input_wav
                try:
                    if not args.input_wav and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

            if process_once:
                break

    except KeyboardInterrupt:
        print('\nInterrupted. Exiting.')


if __name__ == '__main__':
    main()
