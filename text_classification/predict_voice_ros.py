"""
Interactive voice -> transcription -> SPM prediction loop with ROS publishing.

Press 'i' for Arabic or 'e' for English, then hold SPACE to record a clip.
Whisper (small) will transcribe (restricted to the chosen language),
then the SPM predictor will run and print the predicted command.
Press ENTER to publish to ROS, or 'n' to skip publishing.
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
from pynput import keyboard

# -----------------------------
# Project root
# -----------------------------
proj_root = str(Path(__file__).resolve().parent.parent)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

ros_pub = None
selected_lang = None  # 'en' or 'ar'

# -------------------------------------------------------
# Recorder class
# -------------------------------------------------------
class PressHoldRecorder:
    """Records audio while space is held down."""
    def __init__(self, device=None, samplerate=None, channels=1):
        self.device = device if device is not None else sd.default.device[0]
        self.samplerate = int(sd.query_devices(self.device, 'input')['default_samplerate']) if samplerate is None else samplerate
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

        self.stream = sd.InputStream(device=self.device, samplerate=self.samplerate,
                                     channels=self.channels, callback=callback)
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
# Load SPM model with correct hyperparameters
# -------------------------------------------------------
def load_spm_model(spm_weights_path, device):
    from text_classification.spmpredict import load_checkpoint, prepare_model_from_checkpoint
    cp = load_checkpoint(spm_weights_path, device)

    # Hyperparameters from config.txt
    model, vocab, id2label = prepare_model_from_checkpoint(
        cp,
        max_len=32,
        device=device,
        d_model=256,
        nhead=4,
        ff_dim=512,
        num_layers=4,
        dropout=0.2
    )
    return model, vocab, id2label

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    global selected_lang

    parser = argparse.ArgumentParser(description='Interactive voice -> text -> SPM with ROS')
    parser.add_argument('--device', default=None, help='torch device (cpu or cuda)')
    parser.add_argument('--ros-publish', action='store_true', help='Publish predicted class to ROS topic')
    parser.add_argument('--ros-topic', default='command', help='ROS topic name')
    parser.add_argument('--ros-node-name', default='voice_command_node', help='ROS node name')
    args = parser.parse_args()

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
    print("\nPress 'e' for English or 'i' for Arabic, then hold SPACE to record.")
    print("After prediction, press ENTER to publish or 'n' to skip. Press 'q' to quit.")

    # Lazy load models
    wmodel = None
    spm_model = None
    spm_vocab = None
    spm_id2label = None
    label2id = None

    # ----------------------------------------------------
    # Key press
    # ----------------------------------------------------
    def on_press(key):
        nonlocal wmodel
        global selected_lang
        if hasattr(key, "char"):
            if key.char.lower() == 'e':
                selected_lang = 'en'
                print("English selected. Hold SPACE to record.")
            elif key.char.lower() == 'i':
                selected_lang = 'ar'
                print("Arabic selected. Hold SPACE to record.")

        if key == keyboard.Key.space and not recorder.recording:
            if selected_lang is None:
                print("Select language first (e = English, i = Arabic).")
                return
            print("Recording... hold SPACE")
            recorder.start()
            # Lazy load Whisper
            if wmodel is None:
                import whisper
                print("Loading Whisper small model...")
                wmodel = whisper.load_model("small")

    # ----------------------------------------------------
    # Key release
    # ----------------------------------------------------
    def on_release(key):
        nonlocal spm_model, spm_vocab, spm_id2label, label2id, wmodel
        global selected_lang

        if key == keyboard.Key.space and recorder.recording:
            print("Recording stopped, processing...")
            wav_file = recorder.stop()

            # Transcribe
            print(f"Transcribing ({selected_lang})...")
            result = wmodel.transcribe(wav_file, language=selected_lang, task="transcribe",
                                       temperature=0, beam_size=5, best_of=5, fp16=False)
            text = result["text"].strip()
            print(f"Final Transcription ({selected_lang}): {text}")

            if not text:
                print("Empty transcription, ignoring.")
                os.remove(wav_file)
                return

            # Lazy load SPM model
            if spm_model is None:
                spm_weights_path = '/home/kan/ML/models/spmfinal/best.pth'
                print("Loading SPM model...")
                spm_model, spm_vocab, spm_id2label = load_spm_model(spm_weights_path, device)
                label2id = {v: k for k, v in spm_id2label.items()}

            # Prediction
            from text_classification.spmpredict import predict_text
            results = predict_text(spm_model, spm_vocab, spm_id2label, text, device, max_len=32, topk=1)
            pred, conf = results[0]
            print(f"\nPrediction: {pred} (confidence: {conf:.1%})")

            # Wait for confirmation to publish
            print("Press ENTER to publish or 'n' to skip...")
            while True:
                inp = input().strip().lower()
                if inp == "":
                    if ros_pub:
                        from std_msgs.msg import Int32
                        ros_pub.publish(Int32(label2id[pred]))
                        print("Published to ROS.")
                    else:
                        print("ROS not enabled, skipping publish.")
                    break
                elif inp == "n":
                    print("Skipping ROS publish.")
                    break
                else:
                    print("Press ENTER to publish or 'n' to skip...")

            os.remove(wav_file)

        elif hasattr(key, "char") and key.char.lower() == "q":
            print("Exiting...")
            return False

    # Start listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == '__main__':
    main()

