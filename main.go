package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"sync"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"

	"github.com/liyue201/goqr"
	"gocv.io/x/gocv"
)

type uiState struct {
	mu          sync.Mutex
	previewOn   bool
	cancelPrev  context.CancelFunc
	lastFrame   image.Image
	detected    bool
}

func main() {
	a := app.New()
	w := a.NewWindow("Descriptor Reader (Live)")
	w.Resize(fyne.NewSize(800, 700))

	// UI
	img := canvas.NewImageFromImage(nil)
	img.FillMode = canvas.ImageFillContain
	img.SetMinSize(fyne.NewSize(800, 450))

	output := widget.NewMultiLineEntry()
	output.SetPlaceHolder("Descriptor will appear here…")
	output.Wrapping = fyne.TextWrapWord

	copyBtn := widget.NewButton("Copy", func() { a.Clipboard().SetContent(output.Text) })

	state := &uiState{}

	// Camera open once
	webcam, err := gocv.OpenVideoCapture(0)
	if err != nil {
		dialog.ShowError(fmt.Errorf("open camera: %w", err), w)
	} else {
		_ = webcam.Set(gocv.VideoCaptureFrameWidth, 1280)
		_ = webcam.Set(gocv.VideoCaptureFrameHeight, 720)
	}

	startPreview := func() {
		if webcam == nil {
			dialog.ShowError(fmt.Errorf("camera not available"), w)
			return
		}
		state.mu.Lock()
		if state.previewOn {
			state.mu.Unlock()
			return
		}
		ctx, cancel := context.WithCancel(context.Background())
		state.cancelPrev = cancel
		state.previewOn = true
		state.detected = false
		state.mu.Unlock()

		go runPreview(ctx, webcam, img, output, a, state)
	}

	stopPreview := func() {
		state.mu.Lock()
		if state.cancelPrev != nil {
			state.cancelPrev()
		}
		state.previewOn = false
		state.cancelPrev = nil
		state.mu.Unlock()
	}

	readBtn := widget.NewButton("Read Descriptor", func() {
		// Starts scanning (preview already running). When a QR is detected,
		// the preview loop will freeze the frame and set output text.
		if !state.previewOn {
			startPreview()
		}
	})

	resetBtn := widget.NewButton("Reset", func() {
		output.SetText("")
		state.mu.Lock()
		state.detected = false
		state.mu.Unlock()
		startPreview()
	})

	w.SetContent(container.NewVBox(
		img,
		container.NewHBox(readBtn, resetBtn, copyBtn),
		output,
	))

	// Start preview immediately
	startPreview()

	// Cleanup when window closes
	w.SetCloseIntercept(func() {
		stopPreview()
		if webcam != nil {
			webcam.Close()
		}
		w.Close()
	})

	w.ShowAndRun()
}

// Preview/scanner loop: grabs frames, updates UI, tries to decode QR.
// On first successful decode, annotates frame, freezes preview, and writes payload.
func runPreview(ctx context.Context, cam *gocv.VideoCapture, img *canvas.Image, out *widget.MultiLineEntry, a fyne.App, state *uiState) {
	frame := gocv.NewMat()
	defer frame.Close()

	ticker := time.NewTicker(33 * time.Millisecond) // ~30 FPS
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if ok := cam.Read(&frame); !ok || frame.Empty() {
				continue
			}

			// Convert to image.Image
			src := matToImage(frame)
			if src == nil {
				continue
			}

			// Try decode
			payload, bounds := tryDecode(src)
			if payload != "" {
				// Draw rectangle and freeze
				annotated := drawBoundingBox(src, bounds)
				a.Driver().RunOnMain(func() {
					img.Image = annotated
					img.Refresh()
					out.SetText(payload)
				})
				// Stop preview (freeze on detection)
				state.mu.Lock()
				if state.cancelPrev != nil {
					state.cancelPrev()
				}
				state.previewOn = false
				state.detected = true
				state.cancelPrev = nil
				state.mu.Unlock()
				return
			}

			// No detection: keep previewing
			a.Driver().RunOnMain(func() {
				img.Image = src
				img.Refresh()
			})
		}
	}
}

func tryDecode(src image.Image) (string, image.Rectangle) {
	codes, err := goqr.Recognize(src)
	if err != nil || len(codes) == 0 {
		return "", image.Rectangle{}
	}
	// Choose the longest payload (useful if multiple QRs present)
	best := codes[0]
	for _, c := range codes {
		if len(c.Payload) > len(best.Payload) {
			best = c
		}
	}
	return string(best.Payload), best.Bounds
}

func matToImage(m gocv.Mat) image.Image {
	if m.Empty() {
		return nil
	}
	buf, err := gocv.IMEncode(".png", m)
	if err != nil {
		return nil
	}
	defer buf.Close()
	img, err := png.Decode(bytes.NewReader(buf.GetBytes()))
	if err != nil {
		return nil
	}
	return img
}

func drawBoundingBox(src image.Image, r image.Rectangle) image.Image {
	dst := image.NewRGBA(src.Bounds())
	draw.Draw(dst, dst.Bounds(), src, image.Point{}, draw.Src)
	col := color.RGBA{0, 255, 0, 255}
	th := 4
	// Top
	for y := r.Min.Y; y < r.Min.Y+th; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			setRGBA(dst, x, y, col)
		}
	}
	// Bottom
	for y := r.Max.Y - th; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			setRGBA(dst, x, y, col)
		}
	}
	// Left
	for x := r.Min.X; x < r.Min.X+th; x++ {
		for y := r.Min.Y; y < r.Max.Y; y++ {
			setRGBA(dst, x, y, col)
		}
	}
	// Right
	for x := r.Max.X - th; x < r.Max.X; x++ {
		for y := r.Min.Y; y < r.Max.Y; y++ {
			setRGBA(dst, x, y, col)
		}
	}
	return dst
}

func setRGBA(img *image.RGBA, x, y int, c color.RGBA) {
	if image.Pt(x, y).In(img.Bounds()) {
		img.SetRGBA(x, y, c)
	}
}
