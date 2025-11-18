"""
Interactive voice -> transcription -> ensemble prediction loop with ROS publishing.

Press and hold SPACE to record a clip. Whisper (small) will transcribe it,
then the ensemble predictor will run and print the voted command.
Press 'q' to quit.

Usage:
  python -m text_classification.predict_voice_ros \
      --custom-model models/test_run/best.pth \
      --ensemble-dir models/ensemble \
      --ros-publish --ros-topic command
"""

import argparse
import sys
import tempfile
import os
import torch
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf

# Project imports
proj_root = str(Path(__file__).resolve().parent.parent)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Press-and-hold recording
try:
    from pynput import keyboard
except ImportError:
    raise RuntimeError("Install pynput: pip install pynput")

# ROS optional
ros_pub = None


class PressHoldRecorder:
    """Records audio while space is held down."""
    def __init__(self, device=None, samplerate=None, channels=1):
        self.device = device if device is not None else sd.default.device[0]
        if samplerate is None:
            self.samplerate = int(sd.query_devices(self.device, 'input')['default_samplerate'])
        else:
            self.samplerate = samplerate
        self.channels = channels
        self.recording = False
        self.frames = []
        self.stream = None

    def start(self):
        self.recording = True
        self.frames = []

        def callback(indata, frames, time, status):
            if self.recording:
                self.frames.append(indata.copy())

        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.samplerate,
            channels=self.channels,
            callback=callback
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        data = np.concatenate(self.frames, axis=0)
        sf.write(tmp_file.name, data, self.samplerate)
        return tmp_file.name


def main():
    p = argparse.ArgumentParser(description='Interactive voice -> text -> ensemble with ROS')
    p.add_argument('--custom-model', '-m', default='models/test_run/best.pth', help='Custom model checkpoint')
    p.add_argument('--ensemble-dir', '-e', default='models/ensemble', help='Directory of pretrained models')
    p.add_argument('--device', default=None, help='torch device (cpu or cuda)')
    p.add_argument('--ros-publish', action='store_true', help='Publish predicted class id to ROS topic')
    p.add_argument('--ros-topic', default='command', help='ROS topic name')
    p.add_argument('--ros-node-name', default='voice_command_node', help='ROS node name')
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ROS publisher
    global ros_pub
    if args.ros_publish:
        try:
            import rospy
            from std_msgs.msg import Int32
            try:
                rospy.init_node(args.ros_node_name, anonymous=True, disable_signals=True)
            except Exception:
                pass  # node may already exist
            ros_pub = rospy.Publisher(args.ros_topic, Int32, queue_size=1)
            print(f'ROS publishing enabled -> topic: {args.ros_topic}')
        except Exception as e:
            print("ROS not available:", e)
            ros_pub = None
            args.ros_publish = False

    # Recorder
    recorder = PressHoldRecorder()
    print("\nInteractive voice mode (hold SPACE to record, release to predict, press 'q' to quit):")

    # Lazy-loaded objects
    wmodel = None
    ensemble = None
    label2id = None
    id2label = None

    def on_press(key):
        nonlocal wmodel
        if key == keyboard.Key.space and not recorder.recording:
            print("Recording... hold SPACE")
            recorder.start()

            # Lazy load Whisper only on first recording
            if wmodel is None:
                try:
                    import whisper
                except ImportError:
                    raise RuntimeError("Install whisper: pip install -U openai-whisper")
                print("Loading Whisper small model...")
                wmodel = whisper.load_model('small')

    def on_release(key):
        nonlocal ensemble, label2id, id2label, wmodel

        if key == keyboard.Key.space and recorder.recording:
            print("Recording stopped, processing...")
            wav_file = recorder.stop()

            # Transcribe (restricted to Arabic or English)
            from text_classification.whisper_mic import transcribe
            text, lang, _ = transcribe('small', wav_file)
            if lang not in ['ar', 'en'] or not text:
                print(f"Detected language '{lang}' is not Arabic or English, ignoring.")
                os.remove(wav_file)
                return
            print(f"Transcribed ({lang}): {text}")

            # Lazy load prediction models after transcription
            if ensemble is None:
                from text_classification.predict_ensemble import load_custom_model, load_pretrained_models
                from text_classification.ensemble import EnsembleClassifier

                print('Loading custom model...')
                custom_model, vocab, id2label = load_custom_model(Path(args.custom_model), device)
                print('Loading pretrained models...')
                pretrained_models = load_pretrained_models(args.ensemble_dir, device, len(id2label))
                ensemble = EnsembleClassifier(custom_model, vocab, id2label, pretrained_models, device)
                label2id = {v: k for k, v in id2label.items()}
                print(f'Ensemble ready with {len(ensemble.model_names)} models.')

            # Predict with ensemble
            predictions, confidences, details = ensemble.predict_with_probabilities(
                [text],
                use_custom=True,
                use_pretrained=len(ensemble.model_names) > 1,
                custom_max_len=32,
                pretrained_max_len=128
            )
            pred = predictions[0]
            conf = confidences[0]
            print(f"Prediction: {pred} (confidence: {conf:.1%})")

            # Voting breakdown
            voting_detail = details['voting_details'][0]
            votes = voting_detail['votes']
            print("Voting breakdown:")
            for model_name in sorted(votes.keys()):
                vote_label = id2label[votes[model_name]]
                print(f"  {model_name:20s} -> {vote_label}")
            print(f"Agreement: {voting_detail['agreement_count']}/{voting_detail['total_models']} models")

            # Publish to ROS
            if ros_pub:
                try:
                    ros_pub.publish(Int32(label2id[pred]))
                except Exception as e:
                    print("Failed to publish to ROS:", e)

            os.remove(wav_file)

        elif hasattr(key, 'char') and key.char in ('q', 'Q'):
            print("Exiting...")
            return False

    # Start listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == '__main__':
    main()

