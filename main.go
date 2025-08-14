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
	"regexp"
	"strings"
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
	mu         sync.Mutex
	previewOn  bool
	paused     bool
	detected   bool
	lastFrame  image.Image
	cancelPrev context.CancelFunc
}

func runOnMain(f func()) { fyne.Do(f) }

func main() {
	a := app.New()
	w := a.NewWindow("Descriptor Reader (live)")
	w.Resize(fyne.NewSize(900, 740))

	img := canvas.NewImageFromImage(nil)
	img.FillMode = canvas.ImageFillContain

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

	pauseBtn := widget.NewButton("Pause Preview", func() {
		state.mu.Lock()
		state.paused = true
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
	copyBtn := widget.NewButton("Copy", func() {
		w.Clipboard().SetContent(output.Text)
	})

	controls := container.NewVBox(pauseBtn, scanPausedBtn, resetBtn, copyBtn)
	w.SetContent(container.NewBorder(nil, output, controls, nil, img))

	startPreview()
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

	// Convert paused frame to Mat once
	mat, err := gocv.ImageToMatRGB(imgCopy)
	if err != nil {
		log.Printf("ImageToMatRGB err: %v", err)
		dialog.ShowInformation("Scan", "Could not convert frame.", w)
		return
	}
	defer mat.Close()

	// Do an in-depth corner-first sweep on the paused frame
	payload := deepDetectAndDecode(mat)
	log.Printf("deep detect FINISH")

	if payload == "" {
		dialog.ShowInformation("Scan", "No QR detected in paused frame.", w)
		return
	}
	if IsLikelyDescriptor(payload) {
		runOnMain(func() { output.SetText(payload) })
	} else {
		dialog.ShowInformation("Scan", "QR detected, but it's not a Bitcoin output descriptor.", w)
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
				log.Printf("payload != \"\"")
				if IsLikelyDescriptor(payload) {
					log.Printf("is a descriptor")
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
				} else {
					log.Printf("not a descriptor")
				}
			}
			runOnMain(func() { img.Image = drawGuideBox(src); img.Refresh() })
		}
	}
}

func tryDecodeOpenCV(m gocv.Mat) string {
	det := gocv.NewQRCodeDetector()
	defer det.Close()

	// Attempt helper that also logs which path worked.
	try := func(src gocv.Mat, tag string) string {
		pts := gocv.NewMat()
		straight := gocv.NewMat()
		defer pts.Close()
		defer straight.Close()
		s := det.DetectAndDecode(src, &pts, &straight)
		if s != "" {
			log.Printf("QR hit via %s", tag)
		}
		return s
	}

	// 0) Raw frame
	if s := try(m, "raw"); s != "" {
		return s
	}

	// 1) Grayscale
	gray := gocv.NewMat()
	gocv.CvtColor(m, &gray, gocv.ColorBGRToGray)
	defer gray.Close()
	if s := try(gray, "gray"); s != "" {
		return s
	}

	// 2) Histogram equalization (helps low contrast)
	eq := gocv.NewMat()
	gocv.EqualizeHist(gray, &eq)
	if s := try(eq, "equalize"); s != "" {
		eq.Close()
		return s
	}
	eq.Close()

	// 3) Inverted gray (helps white-on-black)
	ginv := gocv.NewMat()
	gocv.BitwiseNot(gray, &ginv)
	if s := try(ginv, "gray_inverted"); s != "" {
		ginv.Close()
		return s
	}
	ginv.Close()

	// 4) Nearest-neighbor upscale (preserve edges for tiny modules)
	enlarged := gocv.NewMat()
	if m.Cols() < 1400 {
		newW := 1400
		newH := int(float64(m.Rows()) * float64(newW) / float64(m.Cols()))
		gocv.Resize(gray, &enlarged, image.Pt(newW, newH), 0, 0, gocv.InterpolationNearestNeighbor)
		if s := try(enlarged, "nn_upscale(gray)"); s != "" {
			enlarged.Close()
			return s
		}
	}
	enlarged.Close()

	// 5) Otsu binarization
	bin := gocv.NewMat()
	gocv.Threshold(gray, &bin, 0, 255, gocv.ThresholdBinary|gocv.ThresholdOtsu)
	if s := try(bin, "otsu"); s != "" {
		bin.Close()
		return s
	}

	// 6) Inverted Otsu
	inv := gocv.NewMat()
	gocv.BitwiseNot(bin, &inv)
	if s := try(inv, "otsu_inverted"); s != "" {
		inv.Close()
		bin.Close()
		return s
	}
	inv.Close()
	bin.Close()

	// 7) Adaptive threshold (robust to uneven lighting)
	adap := gocv.NewMat()
	gocv.AdaptiveThreshold(gray, &adap, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinary, 31, 5)
	if s := try(adap, "adaptive"); s != "" {
		adap.Close()
		return s
	}

	// 8) Inverted adaptive
	adinv := gocv.NewMat()
	gocv.BitwiseNot(adap, &adinv)
	if s := try(adinv, "adaptive_inverted"); s != "" {
		adinv.Close()
		adap.Close()
		return s
	}
	adinv.Close()
	adap.Close()

	return ""
}

func tryDecode(src image.Image) (string, image.Rectangle) {
	qrCodes, err := goqr.Recognize(src)
	if err != nil || len(qrCodes) == 0 {
		return "", image.Rectangle{}
	}
	return string(qrCodes[0].Payload), image.Rectangle{}
}

func matToImage(m gocv.Mat) image.Image {
	if m.Empty() {
		return nil
	}
	bgra := gocv.NewMat()
	gocv.CvtColor(m, &bgra, gocv.ColorBGRToBGRA)
	nbuf, err := gocv.IMEncode(".png", bgra)
	if err != nil || nbuf == nil {
		return nil
	}
	defer nbuf.Close()

	buf := nbuf.GetBytes()
	if len(buf) == 0 {
		return nil
	}

	img, err := png.Decode(bytes.NewReader(buf))
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

// IsLikelyDescriptor: filter for Bitcoin output descriptors
var (
	headRe     = regexp.MustCompile(`^(pkh|wpkh|sh|wsh|tr|combo)\(`)
	hasKeyRe   = regexp.MustCompile(`(xpub|xprv|tpub|tprv|[023][0-9A-Fa-f]{64})`)
	checksumRe = regexp.MustCompile(`#[-a-z0-9]{8}$`)
)

func IsLikelyDescriptor(s string) bool {
	s = strings.TrimSpace(s)
	if !headRe.MatchString(s) {
		return false
	}
	if !strings.Contains(s, ")") || len(s) < 20 {
		return false
	}
	if !hasKeyRe.MatchString(s) {
		return false
	}
	if strings.Contains(s, "#") && !checksumRe.MatchString(s) {
		return false
	}
	// Basic paren balance check.
	bal := 0
	for _, r := range s {
		if r == '(' {
			bal++
		}
		if r == ')' {
			bal--
		}
		if bal < 0 {
			return false
		}
	}
	return bal == 0
}

// deepDetectAndDecode runs a heavier multi-variant pass and rectifies detected QRs.
// It returns the first decoded payload found, or "".
func deepDetectAndDecode(m gocv.Mat) string {
	log.Printf("deep detect START")
	det := gocv.NewQRCodeDetector()
	defer det.Close()

	type variant struct {
		M   gocv.Mat
		tag string
	}
	variants := make([]variant, 0, 12)
	add := func(mat gocv.Mat, tag string) { variants = append(variants, variant{M: mat, tag: tag}) }
	// Keep track of mats to close (we don't close the original 'm')
	toClose := []gocv.Mat{}

	// Base variants
	add(m, "raw")
	gray := gocv.NewMat()
	gocv.CvtColor(m, &gray, gocv.ColorBGRToGray)
	add(gray, "gray")
	toClose = append(toClose, gray)

	eq := gocv.NewMat()
	gocv.EqualizeHist(gray, &eq)
	add(eq, "equalize")
	toClose = append(toClose, eq)

	ginv := gocv.NewMat()
	gocv.BitwiseNot(gray, &ginv)
	add(ginv, "gray_inverted")
	toClose = append(toClose, ginv)

	bin := gocv.NewMat()
	gocv.Threshold(gray, &bin, 0, 255, gocv.ThresholdBinary|gocv.ThresholdOtsu)
	add(bin, "otsu")
	toClose = append(toClose, bin)

	inv := gocv.NewMat()
	gocv.BitwiseNot(bin, &inv)
	add(inv, "otsu_inverted")
	toClose = append(toClose, inv)

	adap := gocv.NewMat()
	gocv.AdaptiveThreshold(gray, &adap, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinary, 31, 5)
	add(adap, "adaptive")
	toClose = append(toClose, adap)

	adinv := gocv.NewMat()
	gocv.BitwiseNot(adap, &adinv)
	add(adinv, "adaptive_inverted")
	toClose = append(toClose, adinv)

	// Add edge-preserving upscales for small frames (helps tiny modules)
	if m.Cols() < 1400 {
		base := append([]variant(nil), variants...) // copy current list
		for _, v := range base {
			up := gocv.NewMat()
			newW := 1600
			newH := int(float64(v.M.Rows()) * float64(newW) / float64(v.M.Cols()))
			gocv.Resize(v.M, &up, image.Pt(newW, newH), 0, 0, gocv.InterpolationNearestNeighbor)
			add(up, v.tag+"+nnx")
			toClose = append(toClose, up)
		}
	}

	// Try each variant: detect corners -> decode; if decode fails but corners exist, try rectified 'straight'
	for _, v := range variants {
		pts := gocv.NewMat()
		straight := gocv.NewMat()
		s := det.DetectAndDecode(v.M, &pts, &straight)

		if s != "" {
			log.Printf("QR hit via %s", v.tag)
			pts.Close()
			straight.Close()
			// return immediately; caller will gate on IsLikelyDescriptor
			// (If you prefer to continue searching for a descriptor specifically, move this return after a gate.)
			for _, c := range toClose {
				c.Close()
			}
			return s
		}

		// If corners found but decode empty, try rectified patch with goqr.
		if !pts.Empty() && !straight.Empty() {
			if img := matToImage(straight); img != nil {
				if codes, err := goqr.Recognize(img); err == nil && len(codes) > 0 {
					s2 := string(codes[0].Payload)
					log.Printf("QR rectified+goqr hit via %s", v.tag)
					pts.Close()
					straight.Close()
					for _, c := range toClose {
						c.Close()
					}
					return s2
				}
			}
		}
		pts.Close()
		straight.Close()
	}

	for _, c := range toClose {
		c.Close()
	}
	return ""
}
