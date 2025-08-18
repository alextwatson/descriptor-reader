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

// uistate struct helps control the UI state and synchronization
type uiState struct {
	mu         sync.Mutex
	previewOn  bool
	paused     bool
	detected   bool
	lastFrame  image.Image
	cancelPrev context.CancelFunc
}

func runOnMain(f func()) {
	fyne.Do(f)
}

func main() {
	// Initialize the Fyne app and main window
	a := app.New()
	w := a.NewWindow("Descriptor Reader (live)")
	w.Resize(fyne.NewSize(900, 740))

	// Create UI components
	img := canvas.NewImageFromImage(nil)
	img.FillMode = canvas.ImageFillContain

	// Placeholder for the output descriptor
	output := widget.NewMultiLineEntry()
	output.SetPlaceHolder("Descriptor will appear here…")
	output.Wrapping = fyne.TextWrapWord

	state := &uiState{}
	// Open the webcam
	webcam, _, err := openFirstWorkingCamera()
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
		go runPreview(ctx, webcam, img, output, state, w)
	}

	resetBtn := widget.NewButton("Reset", func() {
		output.SetText("")
		state.mu.Lock()
		state.detected = false
		state.paused = false
		state.mu.Unlock()
		startPreview()
	})

	copyBtn := widget.NewButton("Copy", func() {
		w.Clipboard().SetContent(output.Text)
	})

	// Layout the UI components
	controls := container.NewVBox(resetBtn, copyBtn)
	w.SetContent(container.NewBorder(img, output, nil, nil, controls))

	startPreview()
	w.ShowAndRun()
}

// scanPausedFrame processes the currently paused frame for QR codes

func runPreview(ctx context.Context, cam *gocv.VideoCapture, img *canvas.Image, out *widget.Entry, state *uiState, w fyne.Window) {
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
			payload, bounds := quickDetect(frame)

			if payload != "" {
				log.Printf("payload != \"\"")
				if IsLikelyDescriptor(payload) {
					log.Printf("is a descriptor")

					node, err := Parse(payload)
					if err != nil {
						fmt.Println("Parse error:", err)
						dialog.ShowInformation("Type", "QR detected, but descriptor parse error.", w)
					}

					desc := Explain(node)

					// crop QR region
					if payload != "" && !bounds.Empty() && src != nil {
						cropped := cropImage(src, bounds)

						qrImg := canvas.NewImageFromImage(cropped)
						qrImg.FillMode = canvas.ImageFillContain
						qrImg.SetMinSize(fyne.NewSize(200, 200))

						msg := widget.NewLabel(desc)
						msg.Alignment = fyne.TextAlignLeading
						msg.Wrapping = fyne.TextWrapWord

						content := container.NewVBox(qrImg, msg)
						d := dialog.NewCustom("Type", "OK", content, w)
						d.Resize(fyne.NewSize(450, 450))
						d.Show()
					}
					log.Printf(Explain(node))

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

func quickDetect(m gocv.Mat) (string, image.Rectangle) {
	det := gocv.NewQRCodeDetector()
	defer det.Close()

	pts := gocv.NewMat()
	straight := gocv.NewMat()
	defer pts.Close()
	defer straight.Close()

	s := det.DetectAndDecode(m, &pts, &straight)
	if s == "" {
		return "", image.Rectangle{}
	}

	rect := image.Rectangle{}
	if !pts.Empty() {
		arr, _ := pts.DataPtrFloat32()
		if len(arr) >= 8 {
			minX, minY := int(arr[0]), int(arr[1])
			maxX, maxY := minX, minY
			for i := 2; i < 8; i += 2 {
				x, y := int(arr[i]), int(arr[i+1])
				if x < minX {
					minX = x
				}
				if y < minY {
					minY = y
				}
				if x > maxX {
					maxX = x
				}
				if y > maxY {
					maxY = y
				}
			}
			rect = image.Rect(minX, minY, maxX, maxY)
		}
	}

	return s, rect
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

// drawBoundingBox draws a green bounding box around the specified rectangle on the source image.
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

// IsLikelyDescriptor checks if a string looks like a Bitcoin output descriptor.
var (
	headRe     = regexp.MustCompile(`^(pkh|wpkh|sh|wsh|tr|combo)\(`)
	hasKeyRe   = regexp.MustCompile(`(xpub|xprv|tpub|tprv|[023][0-9A-Fa-f]{6,})`)
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

// addRotations creates rotated variants of src at several angles and registers them via add.
// Angles chosen to cover slight skew + orthogonal cases without exploding the search.
func addRotations(src gocv.Mat, tag string, add func(gocv.Mat, string), toClose *[]gocv.Mat) {
	angles := []float64{5, -5, 10, -10, 15, -15, 90, 180, 270}
	center := image.Pt(src.Cols()/2, src.Rows()/2)
	for _, a := range angles {
		M := gocv.GetRotationMatrix2D(center, a, 1.0)
		dst := gocv.NewMat()
		gocv.WarpAffine(src, &dst, M, image.Pt(src.Cols(), src.Rows()))
		add(dst, fmt.Sprintf("%s+rot%.0f", tag, a))
		*toClose = append(*toClose, dst)
		M.Close()
	}
}

// cropImage crops the source image to the specified rectangle.
func cropImage(img image.Image, rect image.Rectangle) image.Image {
	if img == nil {
		return nil
	}
	if rect.Empty() {
		return img // or return nil if you prefer not to display anything
	}

	// Make sure the rect lies within the image
	rect = rect.Intersect(img.Bounds())
	if rect.Empty() {
		return img
	}

	cropped := image.NewRGBA(rect)
	draw.Draw(cropped, rect, img, rect.Min, draw.Src)
	return cropped
}

func openFirstWorkingCamera() (*gocv.VideoCapture, int, error) {
	for i := 0; i < 6; i++ {
		cap, err := gocv.OpenVideoCapture(i)
		if err != nil || cap == nil || !cap.IsOpened() {
			if cap != nil {
				cap.Close()
			}
			continue
		}
		mat := gocv.NewMat()
		ok := cap.Read(&mat)
		empty := !ok || mat.Empty()
		mat.Close()
		if empty {
			cap.Close()
			continue
		}
		return cap, i, nil
	}
	return nil, -1, fmt.Errorf("no usable camera found")
}
