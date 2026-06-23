# Digital Hoarding Detector

Digital Hoarding Detector is a computer vision web application that helps users review cluttered photo collections. A user can upload images from a desktop or mobile browser and receive suggestions for duplicates, blurry photos, screenshots, and groups of similar selfies.

The project is designed as a working prototype for small and medium-sized galleries. It processes images in memory and presents the results through a Streamlit interface. It does not delete or modify any uploaded file.

## Why this project exists

Phone galleries grow quickly, but most people do not regularly review them. Storage is often consumed by repeated photos, accidental blurry shots, old screenshots, and several nearly identical selfies.

This application reduces the manual work involved in finding those images. It combines several lightweight computer vision techniques into one cleanup report and estimates how much storage could be recovered.

## Features

- Upload multiple JPG, JPEG, or PNG images
- Detect exact and near-duplicate images
- Identify blurry images using Laplacian variance
- Estimate whether an image is a screenshot
- Detect faces and group similar selfies
- Display flagged images and related groups
- Estimate potential storage savings without double-counting images
- Adjust detector thresholds from the web interface
- Use the application from a desktop or mobile browser

## How the detection works

### Duplicate detection

Each image is converted into a perceptual difference hash. The application compares hashes using Hamming distance and groups images whose hashes are sufficiently close.

This method is fast and handles small changes such as resizing or recompression. It is not intended to replace a learned image-embedding model for difficult cases such as different crops or camera angles.

### Blur detection

The image is converted to grayscale and passed through a Laplacian filter. The variance of the filtered image is used as a sharpness score.

A low variance usually indicates that the image contains few strong edges and may be blurry. The correct threshold depends on the type and resolution of the uploaded images, so it can be adjusted in the interface.

### Screenshot detection

The screenshot detector uses lightweight visual heuristics:

- Text-like region count
- Text coverage
- Edge density
- Flat-color region ratio

These signals work reasonably well for chat, browser, document, and application screenshots. The detector is heuristic-based and may misclassify text-heavy photographs or screenshots with photographic backgrounds.

### Similar-selfie detection

OpenCV's frontal-face cascade detects faces in each image. The largest detected face is normalized, converted into a low-frequency DCT embedding, and compared with other faces using cosine similarity.

This approach keeps the project easy to install because it does not depend on large deep-learning models. It works best with frontal, well-lit faces and should be treated as prototype-level similarity detection rather than identity verification.

## System architecture

```text
Mobile or desktop browser
          |
          v
Streamlit upload interface
          |
          v
Image decoding and validation
          |
          v
Gallery analysis service
          |
          +-- Duplicate detection
          +-- Blur detection
          +-- Screenshot detection
          +-- Face and selfie grouping
          |
          v
Aggregated cleanup report
          |
          v
Metrics, groups, thumbnails, and storage estimate
```

The detector modules are independent from the Streamlit interface. This makes them easier to test and allows the web layer to be replaced later without rewriting the computer vision logic.

## Technology stack

- Python
- Streamlit
- OpenCV
- NumPy
- Pytest

## Project structure

```text
Digital-Hoarding-Detector/
├── app.py
├── pyproject.toml
├── requirements.txt
├── src/
│   └── digital_hoarding_detector/
│       ├── __init__.py
│       ├── blur.py
│       ├── duplicate.py
│       ├── gallery.py
│       ├── screenshot.py
│       └── selfie.py
└── tests/
    ├── test_blur.py
    ├── test_duplicate.py
    ├── test_gallery.py
    ├── test_screenshot.py
    └── test_selfie.py
```

## Local setup

### Requirements

- Python 3.10 or newer
- Git

### 1. Clone the repository

```bash
git clone https://github.com/mewta/Digital-Hoarding-Detector.git
cd Digital-Hoarding-Detector
```

### 2. Create a virtual environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

To run the application:

```bash
pip install -r requirements.txt
```

For editable development installation:

```bash
pip install -e ".[test]"
```

### 4. Start the application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Access from a phone on the same network

The computer and phone must be connected to the same Wi-Fi network.

Start Streamlit so it accepts connections from other devices:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Find the computer's local IP address and open the following address on the phone:

```text
http://YOUR_COMPUTER_IP:8501
```

Example:

```text
http://192.168.1.10:8501
```

On Android, open the address in Chrome. On iPhone, open it in Safari. If the page does not load, check the operating system firewall and confirm that both devices are on the same network.

## Using the application

1. Open the application in a browser.
2. Select **Upload images**.
3. Choose multiple JPG, JPEG, or PNG files.
4. Optionally adjust the detection settings.
5. Select **Analyze Gallery**.
6. Review the summary metrics and each result tab.

The cleanup-candidates tab combines all flagged categories. An image that appears in more than one category is counted only once in the storage estimate.

## Detection settings

| Setting | Effect |
| --- | --- |
| Blur threshold | Higher values mark more images as blurry. |
| Duplicate sensitivity | Higher values allow more perceptual-hash differences. |
| Screenshot confidence threshold | Lower values classify more images as screenshots. |
| Selfie similarity threshold | Higher values require faces to be more visually similar. |

The default values are starting points. Real photo collections vary by camera, resolution, lighting, and compression, so thresholds may require adjustment.

## Running tests

Install the test dependencies and run:

```bash
pytest
```

The tests cover detector behavior, input validation, grouping logic, storage formatting, and cleanup-candidate aggregation.

## Current limitations

- Images are processed in memory and are not saved permanently.
- Large uploads may use significant memory and take longer to analyze.
- Duplicate detection may miss heavily cropped, rotated, or edited versions.
- Blur thresholds are sensitive to image resolution and scene content.
- Screenshot detection is heuristic-based and can produce false positives.
- Face detection works best with frontal faces.
- Similar-selfie grouping is not an identity-recognition system.
- The application suggests cleanup candidates but does not delete files.

Users should review every result before removing anything from the original gallery.

## Possible improvements

- Replace perceptual hashes with CLIP or MobileNet embeddings
- Add OCR-based screenshot classification
- Use a modern face detector and learned face embeddings
- Process large galleries in background jobs
- Add progress reporting and result export
- Add user accounts and private cloud storage
- Store previous analyses in a database
- Add Docker and hosted deployment configuration
- Benchmark thresholds against a labeled image dataset

## Privacy

When run locally, image analysis happens on the user's computer. Uploaded images remain in the Streamlit process memory for the active session and are not intentionally written to disk by the application.

For a hosted deployment, the server operator is responsible for transport security, access control, retention rules, and privacy disclosures.

## License

No license has been added yet. Until a license is provided, the repository remains under standard copyright protection.
