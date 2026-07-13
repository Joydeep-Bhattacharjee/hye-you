# Low-Cost AI Digital Stethoscope - Implementation Manual

Working prototype: contact chest-piece -> INMP441 I2S mic -> ESP32 -> USB serial ->
Python (bandpass + MFCC + RandomForest) -> live normal / crackle / wheeze / both +
"NEEDS REFERRAL" flag. Windows PC. Every script in `src/` is runnable.

## Key decisions (why this differs from the proposal)
- **INMP441 I2S mic** instead of piezo/electret + MAX9814 -> ESP32 ADC. The ADC is
  noisy and fights WiFi; piezo/MAX9814 is an impedance mismatch. INMP441 is clean
  24-bit digital, no analog stage. Mount it **sealed inside the chest-piece air chamber**.
- **USB wired serial first** (no packet loss, easy to debug). WiFi = later upgrade.
- **MFCC + RandomForest baseline is the primary deliverable.** CNN optional on Colab.
- Deferred for v1: LiPo, TP4056, OLED, SD, 3D enclosure. USB powers + carries data.

## Bill of materials (v1, wired, ~1,650 BDT)
| Part | Qty | ~BDT |
|---|---|---|
| INMP441 I2S mic module | 2 | 200 ea |
| ESP32 DevKit V1 (CP2102) | 1 | 600 |
| Donor acoustic stethoscope (chest-piece) | 1 | 300-600 |
| Female-female jumper wires | 1 set | 100 |
| USB DATA cable (not charge-only) | 1 | 150 |
| Breadboard (optional) | 1 | 100 |

## Wiring: INMP441 -> ESP32
| INMP441 | ESP32 | Purpose |
|---|---|---|
| VDD | 3V3 (NEVER 5V) | power |
| GND | GND | ground |
| SCK | GPIO26 (D26) | I2S bit clock |
| WS  | GPIO25 (D25) | I2S word select |
| SD  | GPIO33 (D33) | data (mic -> ESP32) |
| L/R | GND | channel = LEFT (essential) |

Multimeter (continuity, unplugged): each wire beeps end-to-end; 3V3<->GND = NO beep;
SD<->VDD = NO beep. Powered: 3V3 vs GND = 3.2-3.3 V.

## Software setup (Phase 0)
1. Install Python 3.11.9 (tick "Add to PATH"), Git, VS Code (+Python, Pylance).
2. In this folder:
   ```
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python src\verify_setup.py      # expect RESULT: ALL GOOD
   ```
Always activate venv (see `(venv)`) before running anything.

## Run order (each maps to a project phase)
```
# Phase 1 - dataset (download ICBHI into data\raw\icbhi\ first)
python src\parse_icbhi.py          # -> data\processed\cycles.csv (~6900 cycles)
python src\peek.py                 # sanity check one clip

# Phase 2 - features + baseline model
python src\build_features.py       # -> data\processed\features.npz
python src\train_baseline.py       # -> models\baseline.joblib + results\baseline_confusion.png

# Phase 3 - evaluation
python src\evaluate.py             # patient-grouped CV + binary sens/spec
python src\tune.py                 # optional grid search

# Phase 5 - firmware (flash firmware\stethoscope_esp32 via Arduino IDE, ESP32 core 2.0.17)
python src\serial_test.py          # list COM ports
python src\serial_test.py COM5     # tap mic -> RMS jumps

# Phase 6 - live prototype
python src\live_infer.py COM5      # headless predictions
python src\live_gui.py COM5        # waveform + spectrogram + label GUI
python src\record.py COM5 8 normal p01   # record field data

# Phase 7 - optional CNN
python src\build_spectrograms.py   # -> data\processed\specs.npz (upload to Colab)
# train on Colab GPU, download stetho_cnn.keras into models\, then:
pip install tensorflow==2.15.0
python src\live_gui_cnn.py COM5

# Phase 8 - latency
python src\benchmark.py            # expect < 50 ms/window
```

## Dataset (Phase 1)
Download ICBHI 2017 (Kaggle mirror: vbookshelf/respiratory-sound-database, ~3.7 GB).
Copy all `.wav` + `.txt` into `data\raw\icbhi\`. Labels come from each `.txt` line
`start end crackle wheeze`. Split is by PATIENT (no leakage) - handled in parse_icbhi.py.

## ESP32 firmware notes (Phase 5)
Arduino IDE + ESP32 core **2.0.17** (3.x breaks the legacy I2S API). Board "ESP32 Dev
Module", Upload Speed 921600. Install CP2102 and/or CH340 USB driver. If upload sticks
on "Connecting....", hold the BOOT button. Do not open Arduino Serial Monitor and a
Python script on the same COM port at once.

## Expected results (honest)
- 4-class CV macro-F1 ~0.40-0.55 (ICBHI is hard; normal strong, wheeze/both weaker).
- Binary normal-vs-abnormal: accuracy ~0.75-0.85, sensitivity ~0.75+ (the product metric).
- Latency < 50 ms/window. Report the baseline; CNN is a bonus.

## Troubleshooting quick hits
- `python not recognized` -> PATH unticked at install; reinstall + new terminal.
- pip build errors -> must be Python 3.11 inside venv.
- No COM port -> charge-only cable or missing CP2102/CH340 driver.
- RMS flat in serial_test -> L/R not tied to GND, or SCK/WS/SD swapped.
- RMS pinned at max -> change `>> 11` to `>> 13` in the .ino, re-upload.
- Feature shape error at inference -> you edited features.py after training; rerun
  build_features.py + train_baseline.py.
- No audio / weak signal -> reseat mic sealed into the chest-piece air chamber.

## Folder layout
```
stethoscope-ai\
  data\{raw\icbhi, processed, field}
  models\   baseline.joblib [, stetho_cnn.keras]
  results\  baseline_confusion.png
  src\      config, verify_setup, parse_icbhi, peek, features, build_features,
            train_baseline, evaluate, tune, serial_stream, serial_test,
            live_infer, live_gui, record, build_spectrograms, live_gui_cnn, benchmark
  firmware\stethoscope_esp32\stethoscope_esp32.ino
  requirements.txt
```
