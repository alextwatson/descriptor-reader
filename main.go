package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
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

func runOnMain(f func()) { fyne.Do(f) }

type uiState struct {
	mu         sync.Mutex
	previewOn  bool
	paused     bool
	cancelPrev context.CancelFunc
	lastFrame  image.Image
	detected   bool
}

func main() {
	a := app.New()
	w := a.NewWindow("Descriptor Reader (Live)")
	w.Resize(fyne.NewSize(900, 740))

	img := canvas.NewImageFromImage(nil)
	img.FillMode = canvas.ImageFillContain
	img.SetMinSize(fyne.NewSize(900, 520))

	output := widget.NewMultiLineEntry()
	output.SetPlaceHolder("Descriptor will appear here…")
	output.Wrapping = fyne.TextWrapWord

	state := &uiState{}

	webcam, err := gocv.OpenVideoCapture(0)
	if err != nil {
		dialog.ShowError(fmt.Errorf("open camera: %w", err), w)
	} else {
		webcam.Set(gocv.VideoCaptureFrameWidth, 1920)
		webcam.Set(gocv.VideoCaptureFrameHeight, 1080)
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
		state.paused = false
		state.detected = false
		state.mu.Unlock()
		go runPreview(ctx, webcam, img, output, state)
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

	readBtn := widget.NewButton("Read (Auto)", func() {
		if !state.previewOn {
			startPreview()
		}
	})

	pauseBtn := widget.NewButton("Pause", func() {
		state.mu.Lock()
		state.paused = true
		state.mu.Unlock()
	})
	resumeBtn := widget.NewButton("Resume", func() {
		state.mu.Lock()
		state.paused = false
		state.mu.Unlock()
	})
	scanPausedBtn := widget.NewButton("Scan Paused Frame", func() {
		scanPausedFrame(state, output, w)
	})
	resetBtn := widget.NewButton("Reset", func() {
		output.SetText("")
		state.mu.Lock()
		state.detected = false
		state.paused = false
		state.mu.Unlock()
		startPreview()
	})
	copyBtn := widget.NewButton("Copy", func() { a.Clipboard().SetContent(output.Text) })

	w.SetContent(container.NewVBox(
		img,
		container.NewHBox(readBtn, pauseBtn, resumeBtn, scanPausedBtn, resetBtn, copyBtn),
		output,
	))

	w.Canvas().SetOnTypedKey(func(k *fyne.KeyEvent) {
		switch k.Name {
		case fyne.KeySpace:
			state.mu.Lock()
			if state.paused {
				state.paused = false
			} else {
				state.paused = true
			}
			state.mu.Unlock()
			scanPausedFrame(state, output, w)
		}
	})

	startPreview()

	w.SetCloseIntercept(func() {
		stopPreview()
		if webcam != nil {
			webcam.Close()
		}
		w.Close()
	})

	w.ShowAndRun()
}

func scanPausedFrame(state *uiState, output *widget.Entry, w fyne.Window) {
	state.mu.Lock()
	imgCopy := state.lastFrame
	state.mu.Unlock()
	if imgCopy == nil {
		dialog.ShowInformation("Scan", "No paused frame yet.", w)
		return
	}
	mat, err := gocv.ImageToMatRGB(imgCopy)
	if err != nil {
		log.Printf("ImageToMatRGB err: %v", err)
	}
	payload := ""
	if err == nil {
		payload = tryDecodeOpenCV(mat)
		mat.Close()
	}
	if payload == "" {
		if s, _ := tryDecode(imgCopy); s != "" {
			payload = s
		}
	}
	if payload != "" {
		runOnMain(func() { output.SetText(payload) })
	} else {
		dialog.ShowInformation("Scan", "No QR detected in paused frame.", w)
	}
}

func runPreview(ctx context.Context, cam *gocv.VideoCapture, img *canvas.Image, out *widget.Entry, state *uiState) {
	frame := gocv.NewMat()
	defer frame.Close()
	det := gocv.NewQRCodeDetector()
	defer det.Close()
	ticker := time.NewTicker(33 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			state.mu.Lock()
			paused := state.paused
			state.mu.Unlock()
			if paused {
				state.mu.Lock()
				lf := state.lastFrame
				state.mu.Unlock()
				if lf != nil {
					runOnMain(func() { img.Image = lf; img.Refresh() })
				}
				continue
			}
			if ok := cam.Read(&frame); !ok || frame.Empty() {
				continue
			}
			src := matToImage(frame)
			if src == nil {
				continue
			}
			state.mu.Lock()
			state.lastFrame = src
			state.mu.Unlock()
			payload := tryDecodeOpenCV(frame)
			var bounds image.Rectangle
			if payload == "" {
				payload, bounds = tryDecode(src)
			}
			if payload != "" {
				state.mu.Lock()
				if state.cancelPrev != nil {
					state.cancelPrev()
				}
				state.previewOn = false
				state.detected = true
				state.cancelPrev = nil
				state.mu.Unlock()
				annotated := drawBoundingBox(src, bounds)
				runOnMain(func() { img.Image = annotated; img.Refresh(); out.SetText(payload) })
				return
			}
			runOnMain(func() { img.Image = drawGuideBox(src); img.Refresh() })
		}
	}
}

func tryDecodeOpenCV(m gocv.Mat) string {
	det := gocv.NewQRCodeDetector()
	defer det.Close()
	try := func(src gocv.Mat) string {
		pts := gocv.NewMat()
		straight := gocv.NewMat()
		defer pts.Close()
		defer straight.Close()
		s := det.DetectAndDecode(src, &pts, &straight)
		if s != "" {
			log.Printf("OpenCV QR hit")
		}
		return s
	}
	if s := try(m); s != "" {
		return s
	}
	gray := gocv.NewMat()
	gocv.CvtColor(m, &gray, gocv.ColorBGRToGray)
	if s := try(gray); s != "" {
		gray.Close()
		return s
	}
	enlarged := gocv.NewMat()
	if m.Cols() < 1000 {
		newW := 1200
		newH := int(float64(m.Rows()) * float64(newW) / float64(m.Cols()))
		gocv.Resize(gray, &enlarged, image.Pt(newW, newH), 0, 0, gocv.InterpolationArea)
		if s := try(enlarged); s != "" {
			gray.Close()
			enlarged.Close()
			return s
		}
	}
	bin := gocv.NewMat()
	gocv.Threshold(gray, &bin, 0, 255, gocv.ThresholdBinary|gocv.ThresholdOtsu)
	if s := try(bin); s != "" {
		gray.Close()
		enlarged.Close()
		bin.Close()
		return s
	}
	inv := gocv.NewMat()
	gocv.BitwiseNot(bin, &inv)
	if s := try(inv); s != "" {
		gray.Close()
		enlarged.Close()
		bin.Close()
		inv.Close()
		return s
	}
	gray.Close()
	enlarged.Close()
	bin.Close()
	inv.Close()
	return ""
}

func tryDecode(src image.Image) (string, image.Rectangle) {
	b := src.Bounds()
	g := image.NewGray(b)
	draw.Draw(g, b, src, b.Min, draw.Src)
	codes, err := goqr.Recognize(g)
	log.Printf("goqr: recognized %d QR(s), err=%v", len(codes), err)
	if err != nil || len(codes) == 0 {
		return "", image.Rectangle{}
	}
	best := codes[0]
	for _, c := range codes {
		if len(c.Payload) > len(best.Payload) {
			best = c
		}
	}
	return string(best.Payload), image.Rectangle{}
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
	if r.Dx() <= 0 || r.Dy() <= 0 {
		return src
	}
	dst := image.NewRGBA(src.Bounds())
	draw.Draw(dst, dst.Bounds(), src, image.Point{}, draw.Src)
	col := color.RGBA{0, 255, 0, 255}
	th := 4
	for y := r.Min.Y; y < r.Min.Y+th; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			setRGBA(dst, x, y, col)
		}
	}
	for y := r.Max.Y - th; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			setRGBA(dst, x, y, col)
		}
	}
	for x := r.Min.X; x < r.Min.X+th; x++ {
		for y := r.Min.Y; y < r.Max.Y; y++ {
			setRGBA(dst, x, y, col)
		}
	}
	for x := r.Max.X - th; x < r.Max.X; x++ {
		for y := r.Min.Y; y < r.Max.Y; y++ {
			setRGBA(dst, x, y, col)
		}
	}
	return dst
}

func drawGuideBox(src image.Image) image.Image {
	b := src.Bounds()
	boxSize := int(float64(min(b.Dx(), b.Dy())) * 0.5)
	cx, cy := b.Dx()/2, b.Dy()/2
	r := image.Rect(cx-boxSize/2, cy-boxSize/2, cx+boxSize/2, cy+boxSize/2)
	return drawBoundingBox(src, r)
}

func setRGBA(img *image.RGBA, x, y int, c color.RGBA) {
	if image.Pt(x, y).In(img.Bounds()) {
		img.SetRGBA(x, y, c)
	}
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
