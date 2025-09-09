package main

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
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
	frameRate  time.Duration // configurable frame rate
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

	state := &uiState{
		frameRate: 33 * time.Millisecond, // default 30fps, configurable
	}

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

	// Reuse one detector
	det := gocv.NewQRCodeDetector()
	defer det.Close()

	state.mu.Lock()
	frameRate := state.frameRate
	state.mu.Unlock()
	ticker := time.NewTicker(frameRate)
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

			// --- Preview frame (mirrored for user UI) ---
			previewFrame := gocv.NewMat()
			gocv.Flip(frame, &previewFrame, 1)

			src := matToImage(previewFrame)

			previewFrame.Close()

			state.mu.Lock()
			state.lastFrame = src
			state.mu.Unlock()

			// --- Detection uses raw (non-flipped) frame ---
			payload, bounds := quickDetect(det, frame)

			if payload != "" {
				log.Printf("payload detected: %q", payload)

				if IsLikelyDescriptor(payload) {
					log.Printf("is a descriptor")

					desc := ""
					if node, err := Parse(payload); err == nil {
						desc = Explain(node)
					} else {
						desc = "QR detected, but descriptor parse error."
						dialog.ShowInformation("Type", desc, w)
					}

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

					if onHit != nil {
						onHit(payload, desc)
					}
					return
				} else {
					log.Printf("not a descriptor, payload=%q", payload)
					// Add non-descriptor QR codes to history
					if onHit != nil {
						onHit(payload, "QR code detected (not a Bitcoin descriptor)")
					}
				}
			}

			runOnMain(func() { img.Image = drawGuideBox(src); img.Refresh() })
		}
	}
}

// quickDetect: more robust (multi + single + preprocessing)
// quickDetect: robust path (multi first, then single) for your GoCV version.
func quickDetect(det gocv.QRCodeDetector, m gocv.Mat) (string, image.Rectangle) {
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(m, &gray, gocv.ColorBGRToGray)
	
	// Create preprocessing variations optimized for different QR code types
	preprocessedImages := []gocv.Mat{}
	
	// Add standard approaches
	preprocessedImages = append(preprocessedImages, 
		// 1. Histogram equalization (original approach)
		func() gocv.Mat {
			hist := gocv.NewMat()
			gocv.EqualizeHist(gray, &hist)
			return hist
		}(),
		// 2. Adaptive thresholding - standard
		func() gocv.Mat {
			adaptive := gocv.NewMat()
			gocv.AdaptiveThreshold(gray, &adaptive, 255, gocv.AdaptiveThresholdMean, gocv.ThresholdBinary, 11, 2)
			return adaptive
		}(),
		// 3. Adaptive thresholding - fine details (for dense QR codes)
		func() gocv.Mat {
			adaptive := gocv.NewMat()
			gocv.AdaptiveThreshold(gray, &adaptive, 255, gocv.AdaptiveThresholdMean, gocv.ThresholdBinary, 7, 3)
			return adaptive
		}(),
		// 4. Sharpened with Gaussian blur reduction
		func() gocv.Mat {
			blurred := gocv.NewMat()
			defer blurred.Close()
			gocv.GaussianBlur(gray, &blurred, image.Pt(3, 3), 0, 0, gocv.BorderDefault)
			sharpened := gocv.NewMat()
			gocv.AddWeighted(gray, 1.5, blurred, -0.5, 0, &sharpened)
			return sharpened
		}(),
		// 5. Morphological operations for noise cleanup
		func() gocv.Mat {
			kernel := gocv.GetStructuringElement(gocv.MorphEllipse, image.Pt(2, 2))
			defer kernel.Close()
			opened := gocv.NewMat()
			gocv.MorphologyEx(gray, &opened, gocv.MorphOpen, kernel)
			return opened
		}(),
		// 6. Enhanced contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
		func() gocv.Mat {
			clahe := gocv.NewCLAHEWithParams(2.0, image.Pt(8, 8))
			defer clahe.Close()
			enhanced := gocv.NewMat()
			clahe.Apply(gray, &enhanced)
			return enhanced
		}(),
		// 7. Gamma correction for low contrast - use simple brightness/contrast adjustment
		func() gocv.Mat {
			gamma := gocv.NewMat()
			gray.ConvertTo(&gamma, gocv.MatTypeCV8U)
			// Brighten the image
			gocv.ConvertScaleAbs(gamma, &gamma, 1.2, 30)
			return gamma
		}(),
	)
	
	// Add inverted versions
	preprocessedImages = append(preprocessedImages,
		// 6. Inverted histogram equalization
		func() gocv.Mat {
			inverted := gocv.NewMat()
			defer inverted.Close()
			gocv.BitwiseNot(gray, &inverted)
			hist := gocv.NewMat()
			gocv.EqualizeHist(inverted, &hist)
			return hist
		}(),
		// 7. Inverted adaptive thresholding - standard
		func() gocv.Mat {
			inverted := gocv.NewMat()
			defer inverted.Close()
			gocv.BitwiseNot(gray, &inverted)
			adaptive := gocv.NewMat()
			gocv.AdaptiveThreshold(inverted, &adaptive, 255, gocv.AdaptiveThresholdMean, gocv.ThresholdBinary, 11, 2)
			return adaptive
		}(),
		// 8. Inverted adaptive thresholding - fine details
		func() gocv.Mat {
			inverted := gocv.NewMat()
			defer inverted.Close()
			gocv.BitwiseNot(gray, &inverted)
			adaptive := gocv.NewMat()
			gocv.AdaptiveThreshold(inverted, &adaptive, 255, gocv.AdaptiveThresholdMean, gocv.ThresholdBinary, 7, 3)
			return adaptive
		}(),
	)
	
	// ROI detection - try to focus on rectangular paper-like regions
	rois := detectPaperRegions(gray)
	for _, roi := range rois {
		roiMat := gray.Region(roi)
		defer roiMat.Close()
		
		// Apply basic preprocessing to ROI
		preprocessedImages = append(preprocessedImages,
			// ROI histogram equalization
			func() gocv.Mat {
				hist := gocv.NewMat()
				gocv.EqualizeHist(roiMat, &hist)
				return hist
			}(),
			// ROI adaptive threshold
			func() gocv.Mat {
				adaptive := gocv.NewMat()
				gocv.AdaptiveThreshold(roiMat, &adaptive, 255, gocv.AdaptiveThresholdMean, gocv.ThresholdBinary, 11, 2)
				return adaptive
			}(),
		)
	}
	
	// Multi-scale detection - try different image sizes (simplified)
	scales := []float64{1.0, 1.2, 0.8, 1.5}
	for _, scale := range scales {
		if scale != 1.0 {
			resized := gocv.NewMat()
			newSize := image.Pt(int(float64(gray.Cols())*scale), int(float64(gray.Rows())*scale))
			gocv.Resize(gray, &resized, newSize, 0, 0, gocv.InterpolationLinear)
			
			// Add scaled versions with basic preprocessing
			preprocessedImages = append(preprocessedImages,
				// Scaled histogram equalization
				func() gocv.Mat {
					hist := gocv.NewMat()
					gocv.EqualizeHist(resized, &hist)
					return hist
				}(),
				// Scaled adaptive threshold
				func() gocv.Mat {
					adaptive := gocv.NewMat()
					blockSize := int(11.0 / scale)
					if blockSize%2 == 0 {
						blockSize++
					}
					if blockSize < 3 {
						blockSize = 3
					}
					gocv.AdaptiveThreshold(resized, &adaptive, 255, gocv.AdaptiveThresholdMean, gocv.ThresholdBinary, blockSize, 2)
					return adaptive
				}(),
			)
			resized.Close()
		}
	}
	
	defer func() {
		for _, img := range preprocessedImages {
			img.Close()
		}
	}()
	
	// Try detection on each preprocessed image
	for _, processedImg := range preprocessedImages {
		pointsMulti := gocv.NewMat()
		defer pointsMulti.Close()
		var straightMulti []gocv.Mat
		defer func() {
			for _, mat := range straightMulti {
				mat.Close()
			}
		}()
		decoded := make([]string, 10)

		ok := det.DetectAndDecodeMulti(processedImg, decoded, &pointsMulti, straightMulti)
		if ok {
			for i, s := range decoded {
				if s != "" {
					return s, rectFromMultiPoints(pointsMulti, i)
				}
			}
		}

		// Fallback to single detection
		pts := gocv.NewMat()
		straight := gocv.NewMat()
		defer pts.Close()
		defer straight.Close()

		s := det.DetectAndDecode(processedImg, &pts, &straight)
		if s != "" {
			return s, rectFromSinglePoints(pts)
		}
	}
	
	return "", image.Rectangle{}
}

// Build a bounding rect from the "multi" points Mat (8 floats per QR: x0,y0,...,x3,y3)
func rectFromMultiPoints(points gocv.Mat, idx int) image.Rectangle {
	if points.Empty() {
		return image.Rectangle{}
	}
	arr, _ := points.DataPtrFloat32()
	// Expect 8 floats per code (4 corners)
	offset := 8 * idx
	if len(arr) < offset+8 {
		return image.Rectangle{}
	}
	minX, minY := int(arr[offset]), int(arr[offset+1])
	maxX, maxY := minX, minY
	for i := 2; i < 8; i += 2 {
		x, y := int(arr[offset+i]), int(arr[offset+i+1])
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
	return image.Rect(minX, minY, maxX, maxY)
}

// Build a bounding rect from the "single" points Mat (8 floats: x0,y0,...,x3,y3)
func rectFromSinglePoints(pts gocv.Mat) image.Rectangle {
	if pts.Empty() {
		return image.Rectangle{}
	}
	arr, _ := pts.DataPtrFloat32()
	if len(arr) < 8 {
		return image.Rectangle{}
	}
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
	return image.Rect(minX, minY, maxX, maxY)
}

// rectFromCorners builds a bounding rect from multi-detect points

// rectFromPts builds a bounding rect from single-detect points

func matToImage(m gocv.Mat) image.Image {
	if m.Empty() {
		return nil
	}
	bgra := gocv.NewMat()
	defer bgra.Close()
	gocv.CvtColor(m, &bgra, gocv.ColorBGRToBGRA)
	
	// Direct conversion without PNG encoding/decoding
	data := bgra.ToBytes()
	bounds := image.Rect(0, 0, bgra.Cols(), bgra.Rows())
	img := &image.RGBA{
		Pix:    data,
		Stride: 4 * bgra.Cols(),
		Rect:   bounds,
	}
	
	// Copy data to avoid sharing memory with OpenCV
	copyImg := image.NewRGBA(bounds)
	copy(copyImg.Pix, img.Pix)
	return copyImg
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
	aliasRe    = regexp.MustCompile(`@\d+=\[`) // Detect descriptor aliases like @1=[...]
)

func IsLikelyDescriptor(s string) bool {
	s = strings.TrimSpace(s)
	
	// Check for descriptor aliases (like @1=[...] syntax)
	hasAliases := aliasRe.MatchString(s)
	
	// For alias-based descriptors, look for the main descriptor part after semicolons
	if hasAliases {
		parts := strings.Split(s, ";")
		if len(parts) < 2 {
			return false
		}
		// Check the last part (should be the main descriptor)
		mainDesc := parts[len(parts)-1]
		return IsLikelyDescriptor(mainDesc) // Recursive check on the main part
	}
	
	// Standard descriptor checks
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

// detectPaperRegions finds rectangular regions that might contain QR codes
func detectPaperRegions(gray gocv.Mat) []image.Rectangle {
	regions := []image.Rectangle{}
	
	// Simple approach - divide image into regions and return center region
	centerX := gray.Cols() / 2
	centerY := gray.Rows() / 2
	size := min(gray.Cols(), gray.Rows()) / 2
	centerRect := image.Rect(
		max(0, centerX-size/2),
		max(0, centerY-size/2),
		min(gray.Cols(), centerX+size/2),
		min(gray.Rows(), centerY+size/2),
	)
	regions = append(regions, centerRect)
	
	return regions
}

// rotateImage rotates an image by the given angle (in degrees)
func rotateImage(src gocv.Mat, angle float64) gocv.Mat {
	if angle == 0 {
		rotated := gocv.NewMat()
		src.CopyTo(&rotated)
		return rotated
	}
	
	center := image.Pt(src.Cols()/2, src.Rows()/2)
	rotMat := gocv.GetRotationMatrix2D(center, angle, 1.0)
	defer rotMat.Close()
	
	rotated := gocv.NewMat()
	gocv.WarpAffine(src, &rotated, rotMat, image.Pt(src.Cols(), src.Rows()))
	return rotated
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
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
