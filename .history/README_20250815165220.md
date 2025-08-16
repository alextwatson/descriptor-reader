# Descriptor Reader

A desktop QR reader that scans **Bitcoin wallet descriptors** from your webcam and lets you copy the text into any wallet.

## Features
- Live camera preview with **auto-detect** (pauses when a QR is found)
- **Manual freeze + scan** workflow for tricky focus situations
- **Spacebar shortcut** to toggle pause and scan the frozen frame
- Works even when the QR isn’t inside the green guide box

## Requirements

This app uses OpenCV + GoCV.

**macOS (Homebrew):**
```bash
brew install opencv
GoCV:
go install gocv.io/x/gocv@latest
```
**Linux/Windows:** install OpenCV via your package manager or binaries, then go install GoCV as above.

Install

```bash
git clone https://github.com/alextwatson/Descriptor-reader.git
cd Descriptor-reader
go mod tidy
```

## Run
Direct run:
```bash
go run .
```
Build & run:
```bash
go build -o Descriptor-reader
./Descriptor-reader
```
## How to Use

- When the preview opens, hold the QR so it’s in focus.
- The green box is just a guide; the reader scans the entire frame.
- On auto-detect, the preview pauses and the descriptor appears in the text box.
- Click Copy to send the descriptor to your clipboard.
- Manual Pause + Scan (for small/far QRs)
- Pause button (or Spacebar) freezes the current frame.
- Click Scan Paused Frame (or press Spacebar again) to decode that frozen image.
- Resume to go back to live preview.

## Keyboard Shortcut

- Spacebar: toggle Pause/Resume and immediately scan when paused.

## Tips

If you have multiple cameras, change the camera index in code:

```bash
gocv.OpenVideoCapture(0) // try 1, 2, ... if needed
```

Higher resolution helps small QRs:

```bash
webcam.Set(gocv.VideoCaptureFrameWidth, 1920)
webcam.Set(gocv.VideoCaptureFrameHeight, 1080)
```

Good, even lighting and avoiding glare dramatically improves decoding.

## Troubleshooting

No camera / permission denied (macOS): grant Camera access to the binary in System Settings → Privacy & Security → Camera.

Can’t build GoCV: ensure OpenCV is installed and visible on PATH/PKG_CONFIG. On macOS:

```bash
echo 'export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"' >> ~/.zprofile
```

Not detecting: hold the code farther back to get sharp focus, then Pause → Scan.