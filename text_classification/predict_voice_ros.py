"""
Interactive voice -> transcription -> ensemble prediction loop with ROS publishing.

Press and hold SPACE to record a clip.
Whisper (small) will transcribe (restricted to English/Arabic only),
then the ensemble predictor will run and print the voted command.
Press 'q' to quit.
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

from pynput import keyboard

ros_pub = None


# -------------------------------------------------------
# High-accuracy EN/AR-only Whisper language selector
# -------------------------------------------------------
def transcribe_english_or_arabic(model, audio_path):
    """
    Transcribes audio twice: once as English, once as Arabic.
    Returns whichever transcription is stronger (longer non-empty text).
    """

    # --- English decode ---
    result_en = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        temperature=0,
        beam_size=5,
        best_of=5,
        fp16=False,
    )
    text_en = result_en["text"].strip()

    # --- Arabic decode ---
    result_ar = model.transcribe(
        audio_path,
        language="ar",
        task="transcribe",
        temperature=0,
        beam_size=5,
        best_of=5,
        fp16=False,
    )
    text_ar = result_ar["text"].strip()

    # Score = length of text
    score_en = len(text_en)
    score_ar = len(text_ar)

    print(f"English score: {score_en} | Arabic score: {score_ar}")

    if score_ar > score_en:
        print("Final: Arabic selected")
        return "ar", text_ar
    else:
        print("Final: English selected")
        return "en", text_en


# -------------------------------------------------------
# Recorder class
# -------------------------------------------------------
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


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='Interactive voice -> text -> ensemble with ROS')
    p.add_argument('--custom-model', '-m', default='models/mild_reg_3ep/best.pth', help='Custom model checkpoint')
    p.add_argument('--ensemble-dir', '-e', default='models/ensemble', help='Directory of pretrained models')
    p.add_argument('--device', default=None, help='torch device (cpu or cuda)')
    p.add_argument('--ros-publish', action='store_true', help='Publish predicted class to ROS topic')
    p.add_argument('--ros-topic', default='command', help='ROS topic name')
    p.add_argument('--ros-node-name', default='voice_command_node', help='ROS node name')
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ROS
    global ros_pub
    if args.ros_publish:
        try:
            import rospy
            from std_msgs.msg import Int32
            try:
                rospy.init_node(args.ros_node_name, anonymous=True, disable_signals=True)
            except Exception:
                pass
            ros_pub = rospy.Publisher(args.ros_topic, Int32, queue_size=1)
            print(f'ROS publishing enabled -> topic: {args.ros_topic}')
        except Exception as e:
            print("ROS not available:", e)
            ros_pub = None
            args.ros_publish = False

    recorder = PressHoldRecorder()
    print("\nInteractive voice mode (hold SPACE to record, release to predict, press 'q' to quit):")

    # Lazy load
    wmodel = None
    ensemble = None
    label2id = None
    id2label = None

    # ----------------------------------------------------
    # On key press
    # ----------------------------------------------------
    def on_press(key):
        nonlocal wmodel
        if key == keyboard.Key.space and not recorder.recording:
            print("Recording... hold SPACE")
            recorder.start()

            # Lazy Whisper load
            if wmodel is None:
                import whisper
                print("Loading Whisper small model...")
                wmodel = whisper.load_model("small")

    # ----------------------------------------------------
    # On key release
    # ----------------------------------------------------
    def on_release(key):
        nonlocal ensemble, label2id, id2label, wmodel

        # SPACE released
        if key == keyboard.Key.space and recorder.recording:
            print("Recording stopped, processing...")
            wav_file = recorder.stop()

            # -------- Restrict transcription to EN or AR --------
            print("Transcribing with EN/AR restricted mode...")
            lang, text = transcribe_english_or_arabic(wmodel, wav_file)
            print(f"Final Transcription ({lang}): {text}")

            if not text:
                print("Empty transcription, ignoring.")
                os.remove(wav_file)
                return

            # Lazy load ensemble
            if ensemble is None:
                from text_classification.predict_ensemble import load_custom_model, load_pretrained_models
                from text_classification.ensemble import EnsembleClassifier

                print("Loading custom model...")
                custom_model, vocab, id2label = load_custom_model(Path(args.custom_model), device)

                print("Loading pretrained models...")
                pretrained = load_pretrained_models(args.ensemble_dir, device, len(id2label))

                ensemble = EnsembleClassifier(custom_model, vocab, id2label, pretrained, device)
                label2id = {v: k for k, v in id2label.items()}

                print(f"Ensemble ready with {len(ensemble.model_names)} models.")

            # Predict text
            preds, confs, details = ensemble.predict_with_probabilities(
                [text],
                use_custom=True,
                use_pretrained=True,
                custom_max_len=32,
                pretrained_max_len=128,
            )

            pred = preds[0]
            conf = confs[0]

            print(f"\nPrediction: {pred} (confidence: {conf:.1%})")

            print("Voting breakdown:")
            vote_info = details["voting_details"][0]
            for m, v in vote_info["votes"].items():
                print(f"  {m:20s} -> {id2label[v]}")
            print(f"Agreement: {vote_info['agreement_count']}/{vote_info['total_models']} models\n")

            # ROS
            if ros_pub:
                from std_msgs.msg import Int32
                ros_pub.publish(Int32(label2id[pred]))

            os.remove(wav_file)

        # Quit
        elif hasattr(key, "char") and key.char.lower() == "q":
            print("Exiting...")
            return False

    # Key listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == '__main__':
    main()
