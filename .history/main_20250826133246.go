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
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/storage"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"

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

type historyItem struct {
	When      time.Time
	Payload   string // raw descriptor text
	Explained string // human-readable Explain()
}

func runOnMain(f func()) {
	fyne.Do(f)
}

func main() {
	// Initialize the Fyne app and main window
	a := app.New()
	a.Settings().SetTheme(theme.DarkTheme()) // dark mode

	w := a.NewWindow("Descriptor Reader (live)")
	w.Resize(fyne.NewSize(1100, 760)) // a bit wider to make room for history

	// Create UI components
	img := canvas.NewImageFromImage(nil)
	img.FillMode = canvas.ImageFillContain

	// Descriptor output
	output := widget.NewMultiLineEntry()
	output.SetPlaceHolder("Descriptor will appear here…")
	output.Wrapping = fyne.TextWrapWord
	output.TextStyle = fyne.TextStyle{Monospace: true} // monospace for readability

	state := &uiState{}

	// History store + list widget
	var histMu sync.Mutex
	history := []historyItem{}

	histList := widget.NewList(
		func() int {
			histMu.Lock()
			defer histMu.Unlock()
			return len(history)
		},
		func() fyne.CanvasObject {
			return widget.NewLabel("item")
		},
		func(i widget.ListItemID, o fyne.CanvasObject) {
			histMu.Lock()
			defer histMu.Unlock()
			if i < 0 || i >= len(history) {
				return
			}
			h := history[i]
			preview := h.Payload
			if nl := strings.Index(preview, "\n"); nl > 0 {
				preview = preview[:nl]
			}
			o.(*widget.Label).SetText(h.When.Format("15:04:05") + " — " + preview)
		},
	)

	// On select, load the descriptor + explanation into the UI
	histList.OnSelected = func(i widget.ListItemID) {
		histMu.Lock()
		defer histMu.Unlock()
		if i < 0 || i >= len(history) {
			return
		}
		out := history[i]
		output.SetText(out.Payload)
		dialog.ShowInformation("Explanation", out.Explained, w)
	}

	// Open the webcam
	webcam, _, err := openFirstWorkingCamera()
	if err != nil {
		dialog.ShowError(fmt.Errorf("open camera: %w", err), w)
	} else {
		webcam.Set(gocv.VideoCaptureFrameWidth, 1920)
		webcam.Set(gocv.VideoCaptureFrameHeight, 1080)
	}

	// Start/Restart preview loop
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

		go runPreview(
			ctx,
			webcam,
			img,
			output,
			state,
			w,
			// onHit: record successful detection to history
			func(payload, explained string) {
				histMu.Lock()
				history = append([]historyItem{{
					When:      time.Now(),
					Payload:   payload,
					Explained: explained,
				}}, history...) // newest first
				histMu.Unlock()
				runOnMain(func() { histList.Refresh() })
			},
		)
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

	exportBtn := widget.NewButton("Export", func() {
		saver := dialog.NewFileSave(func(u fyne.URIWriteCloser, err error) {
			if err != nil || u == nil {
				return
			}
			defer u.Close()

			histMu.Lock()
			defer histMu.Unlock()

			var b strings.Builder
			for i := range history {
				h := history[i]
				b.WriteString(h.When.Format(time.RFC3339))
				b.WriteString("\n")
				b.WriteString("DESCRIPTOR:\n")
				b.WriteString(h.Payload)
				b.WriteString("\n\nEXPLAINED:\n")
				b.WriteString(h.Explained)
				b.WriteString("\n\n---\n\n")
			}
			_, _ = u.Write([]byte(b.String()))
		}, w)
		saver.SetFileName("descriptor_history.txt")
		saver.SetFilter(storage.NewExtensionFileFilter([]string{".txt"}))
		saver.Show()
	})

	// Buttons row (centered horizontally)
	buttons := container.NewHBox(
		layout.NewSpacer(),
		resetBtn,
		copyBtn,
		exportBtn,
		layout.NewSpacer(),
	)

	// Split the lower area into text + buttons
	lowerSplit := container.NewVSplit(output, buttons)
	lowerSplit.SetOffset(0.66) // mostly text

	// Split center into video (top) + lower area
	centerSplit := container.NewVSplit(img, lowerSplit)
	centerSplit.SetOffset(0.7)

	// Left history + main content
	mainSplit := container.NewHSplit(histList, centerSplit)
	mainSplit.SetOffset(0.28)

	w.SetContent(mainSplit)

	startPreview()
	w.ShowAndRun()
}

// Live preview + detection loop
func runPreview(
	ctx context.Context,
	cam *gocv.VideoCapture,
	img *canvas.Image,
	out *widget.Entry,
	state *uiState,
	w fyne.Window,
	onHit func(payload, explained string),
) {
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
				if IsLikelyDescriptor(payload) { // :contentReference[oaicite:2]{index=2}
					log.Printf("is a descriptor")

					// Parse + explain
					desc := ""
					if node, err := Parse(payload); err == nil { // :contentReference[oaicite:3]{index=3}
						desc = Explain(node) // :contentReference[oaicite:4]{index=4}
					} else {
						fmt.Println("Parse error:", err)
						desc = "QR detected, but descriptor parse error."
						dialog.ShowInformation("Type", "QR detected, but descriptor parse error.", w)
					}

					// crop QR region dialog (nice UX)
					if !bounds.Empty() && src != nil {
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

					// Stop this preview loop
					state.mu.Lock()
					if state.cancelPrev != nil {
						state.cancelPrev()
					}
					state.previewOn = false
					state.detected = true
					state.cancelPrev = nil
					state.mu.Unlock()

					annotated := drawBoundingBox(src, bounds)
					runOnMain(func() {
						img.Image = annotated
						img.Refresh()
						out.SetText(payload)
					})

					// record in history
					if onHit != nil {
						onHit(payload, desc)
					}
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
		return img
	}
	// Keep behavior consistent with original: use rect bounds
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
