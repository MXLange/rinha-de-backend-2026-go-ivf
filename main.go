package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/valyala/fasthttp"
)

const (
	vectorDimensions = 14
	blockSize        = 8
	blockStride      = vectorDimensions * blockSize
	maxRequestBody   = 64 * 1024
	vectorScale      = 0.0001
)

var (
	normalizationMagic = [8]byte{'R', '2', '6', 'N', 'O', 'R', 'M', '1'}
	mccRiskMagic       = [8]byte{'R', '2', '6', 'M', 'C', 'C', '1', '0'}
	indexMagic         = [4]byte{'I', 'V', 'F', '1'}
	response0          = []byte(`{"approved":true,"fraud_score":0.0}`)
	response1          = []byte(`{"approved":true,"fraud_score":0.2}`)
	response2          = []byte(`{"approved":true,"fraud_score":0.4}`)
	response3          = []byte(`{"approved":false,"fraud_score":0.6}`)
	response4          = []byte(`{"approved":false,"fraud_score":0.8}`)
	response5          = []byte(`{"approved":false,"fraud_score":1.0}`)
)

type normalization struct {
	MaxAmount            float32
	MaxInstallments      float32
	AmountVsAvgRatio     float32
	MaxMinutes           float32
	MaxKM                float32
	MaxTxCount24h        float32
	MaxMerchantAvgAmount float32
}

type model struct {
	normalization normalization
	mccRisk       [10000]float32
	centroids     []float32
	offsets       []uint32
	labels        []byte
	blocks        []int16
	k             int
	n             int
	paddedN       int
	nprobe        int
}

type server struct {
	model *model
}

type fraudRequest struct {
	Transaction     transaction      `json:"transaction"`
	Customer        customer         `json:"customer"`
	Merchant        merchant         `json:"merchant"`
	Terminal        terminal         `json:"terminal"`
	LastTransaction *lastTransaction `json:"last_transaction"`
}

type transaction struct {
	Amount       float32 `json:"amount"`
	Installments int     `json:"installments"`
	RequestedAt  string  `json:"requested_at"`
}

type customer struct {
	AvgAmount      float32  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type merchant struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float32 `json:"avg_amount"`
}

type terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KMFromHome  float32 `json:"km_from_home"`
}

type lastTransaction struct {
	Timestamp     string  `json:"timestamp"`
	KMFromCurrent float32 `json:"km_from_current"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "test-go-ivf: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	resourcesDir := getenv("RESOURCES_DIR", "/app/resources")
	addr := getenv("LISTEN_ADDR", ":8080")
	started := time.Now()

	m, err := loadModel(resourcesDir)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "test-go-ivf: startup loaded in %s\n", time.Since(started))

	srv := &server{model: m}
	app := fasthttp.Server{
		Handler:                       srv.handle,
		Name:                          "test-go-ivf",
		DisableHeaderNamesNormalizing: true,
		ReduceMemoryUsage:             true,
	}
	return app.ListenAndServe(addr)
}

func loadModel(resourcesDir string) (*model, error) {
	norm, err := loadNormalization(filepath.Join(resourcesDir, "normalization.bin"))
	if err != nil {
		return nil, err
	}
	mccRisk, err := loadMCCRisk(filepath.Join(resourcesDir, "mcc_risk.bin"))
	if err != nil {
		return nil, err
	}
	centroids, offsets, labels, blocks, n, k, paddedN, err := loadIndex(filepath.Join(resourcesDir, "index.bin"))
	if err != nil {
		return nil, err
	}
	nprobe := getenvInt("NPROBE", 24)
	if nprobe < 1 {
		nprobe = 1
	}
	if nprobe > k {
		nprobe = k
	}
	return &model{
		normalization: norm,
		mccRisk:       mccRisk,
		centroids:     centroids,
		offsets:       offsets,
		labels:        labels,
		blocks:        blocks,
		k:             k,
		n:             n,
		paddedN:       paddedN,
		nprobe:        nprobe,
	}, nil
}

func (s *server) handle(ctx *fasthttp.RequestCtx) {
	switch string(ctx.Path()) {
	case "/ready":
		if !ctx.IsGet() && !ctx.IsHead() {
			ctx.Error("method not allowed", fasthttp.StatusMethodNotAllowed)
			return
		}
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetBodyString("OK")
	case "/fraud-score":
		if !ctx.IsPost() {
			ctx.Error("method not allowed", fasthttp.StatusMethodNotAllowed)
			return
		}
		if len(ctx.PostBody()) > maxRequestBody {
			ctx.Error("invalid request payload", fasthttp.StatusBadRequest)
			return
		}
		var req fraudRequest
		if err := json.Unmarshal(ctx.PostBody(), &req); err != nil {
			ctx.Error("invalid request payload", fasthttp.StatusBadRequest)
			return
		}
		query, err := vectorize(&req, s.model.normalization, s.model.mccRisk)
		if err != nil {
			ctx.Error("invalid request payload", fasthttp.StatusBadRequest)
			return
		}
		votes := knn5FraudCountIVF(query, s.model)
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetContentType("application/json")
		switch votes {
		case 0:
			ctx.SetBody(response0)
		case 1:
			ctx.SetBody(response1)
		case 2:
			ctx.SetBody(response2)
		case 3:
			ctx.SetBody(response3)
		case 4:
			ctx.SetBody(response4)
		default:
			ctx.SetBody(response5)
		}
	default:
		ctx.Error("not found", fasthttp.StatusNotFound)
	}
}

func vectorize(req *fraudRequest, norm normalization, mccRisk [10000]float32) ([vectorDimensions]float32, error) {
	var out [vectorDimensions]float32

	requestedAt, err := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	if err != nil {
		return out, fmt.Errorf("parse requested_at: %w", err)
	}

	out[0] = round4(clamp01(req.Transaction.Amount / safeDivisor(norm.MaxAmount)))
	out[1] = round4(clamp01(float32(req.Transaction.Installments) / safeDivisor(norm.MaxInstallments)))
	if req.Customer.AvgAmount > 0 {
		out[2] = round4(clamp01((req.Transaction.Amount / req.Customer.AvgAmount) / safeDivisor(norm.AmountVsAvgRatio)))
	} else if req.Transaction.Amount > 0 {
		out[2] = 1
	}
	out[3] = round4(clamp01(float32(requestedAt.UTC().Hour()) / 23))
	out[4] = round4(clamp01(dayOfWeek(requestedAt) / 6))
	if req.LastTransaction != nil {
		lastAt, err := time.Parse(time.RFC3339, req.LastTransaction.Timestamp)
		if err != nil {
			return out, fmt.Errorf("parse last_transaction.timestamp: %w", err)
		}
		minutes := float32(requestedAt.Sub(lastAt).Minutes())
		if minutes < 0 {
			minutes = 0
		}
		out[5] = round4(clamp01(minutes / safeDivisor(norm.MaxMinutes)))
		out[6] = round4(clamp01(req.LastTransaction.KMFromCurrent / safeDivisor(norm.MaxKM)))
	} else {
		out[5] = -1
		out[6] = -1
	}
	out[7] = round4(clamp01(req.Terminal.KMFromHome / safeDivisor(norm.MaxKM)))
	out[8] = round4(clamp01(float32(req.Customer.TxCount24h) / safeDivisor(norm.MaxTxCount24h)))
	if req.Terminal.IsOnline {
		out[9] = 1
	}
	if req.Terminal.CardPresent {
		out[10] = 1
	}
	if !contains(req.Customer.KnownMerchants, req.Merchant.ID) {
		out[11] = 1
	}
	if idx, ok := parseMCC(req.Merchant.MCC); ok {
		out[12] = round4(mccRisk[idx])
	} else {
		out[12] = 0.5
	}
	out[13] = round4(clamp01(req.Merchant.AvgAmount / safeDivisor(norm.MaxMerchantAvgAmount)))
	return out, nil
}

func knn5FraudCountIVF(query [vectorDimensions]float32, m *model) int {
	probes := topNProbes(query, m)
	bestDistances := [5]float32{math.MaxFloat32, math.MaxFloat32, math.MaxFloat32, math.MaxFloat32, math.MaxFloat32}
	bestLabels := [5]byte{}
	worstIdx := 0

	for _, ci := range probes {
		startBlock := int(m.offsets[ci])
		endBlock := int(m.offsets[ci+1])
		for blockIdx := startBlock; blockIdx < endBlock; blockIdx++ {
			blockBase := blockIdx * blockStride
			labelBase := blockIdx * blockSize
			for slot := 0; slot < blockSize; slot++ {
				dist := float32(0)
				for d := 0; d < vectorDimensions; d++ {
					v := float32(m.blocks[blockBase+d*blockSize+slot]) * vectorScale
					diff := v - query[d]
					dist += diff * diff
				}
				if dist >= bestDistances[worstIdx] {
					continue
				}
				bestDistances[worstIdx] = dist
				bestLabels[worstIdx] = m.labels[labelBase+slot]
				worstIdx = worstDistanceIndex(bestDistances[:])
			}
		}
	}

	votes := 0
	for _, label := range bestLabels {
		if label == 1 {
			votes++
		}
	}
	return votes
}

func topNProbes(query [vectorDimensions]float32, m *model) []int {
	probes := make([]int, m.nprobe)
	probeDistances := make([]float32, m.nprobe)
	for i := range probeDistances {
		probeDistances[i] = math.MaxFloat32
	}
	worstIdx := 0

	for ci := 0; ci < m.k; ci++ {
		dist := float32(0)
		for d := 0; d < vectorDimensions; d++ {
			diff := m.centroids[d*m.k+ci] - query[d]
			dist += diff * diff
		}
		if dist >= probeDistances[worstIdx] {
			continue
		}
		probes[worstIdx] = ci
		probeDistances[worstIdx] = dist
		worstIdx = worstDistanceIndex(probeDistances)
	}

	for i := 1; i < len(probes); i++ {
		j := i
		for j > 0 && probeDistances[j] < probeDistances[j-1] {
			probeDistances[j], probeDistances[j-1] = probeDistances[j-1], probeDistances[j]
			probes[j], probes[j-1] = probes[j-1], probes[j]
			j--
		}
	}
	return probes
}

func worstDistanceIndex(distances []float32) int {
	worstIdx := 0
	worstValue := distances[0]
	for i := 1; i < len(distances); i++ {
		if distances[i] > worstValue {
			worstValue = distances[i]
			worstIdx = i
		}
	}
	return worstIdx
}

func loadNormalization(path string) (normalization, error) {
	var norm normalization
	file, err := os.Open(path)
	if err != nil {
		return norm, fmt.Errorf("open normalization: %w", err)
	}
	defer file.Close()
	var magic [8]byte
	if _, err := file.Read(magic[:]); err != nil {
		return norm, err
	}
	if magic != normalizationMagic {
		return norm, errors.New("invalid normalization magic")
	}
	var version uint32
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return norm, err
	}
	if version != 1 {
		return norm, fmt.Errorf("unsupported normalization version %d", version)
	}
	values := []*float32{
		&norm.MaxAmount,
		&norm.MaxInstallments,
		&norm.AmountVsAvgRatio,
		&norm.MaxMinutes,
		&norm.MaxKM,
		&norm.MaxTxCount24h,
		&norm.MaxMerchantAvgAmount,
	}
	for _, value := range values {
		if err := binary.Read(file, binary.LittleEndian, value); err != nil {
			return norm, err
		}
	}
	return norm, nil
}

func loadMCCRisk(path string) ([10000]float32, error) {
	var table [10000]float32
	file, err := os.Open(path)
	if err != nil {
		return table, fmt.Errorf("open mcc risk: %w", err)
	}
	defer file.Close()
	var magic [8]byte
	if _, err := file.Read(magic[:]); err != nil {
		return table, err
	}
	if magic != mccRiskMagic {
		return table, errors.New("invalid mcc risk magic")
	}
	var version uint32
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return table, err
	}
	if version != 1 {
		return table, fmt.Errorf("unsupported mcc risk version %d", version)
	}
	if err := binary.Read(file, binary.LittleEndian, table[:]); err != nil {
		return table, err
	}
	return table, nil
}

func loadIndex(path string) ([]float32, []uint32, []byte, []int16, int, int, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, nil, 0, 0, 0, fmt.Errorf("open index: %w", err)
	}
	defer file.Close()

	var magic [4]byte
	if _, err := file.Read(magic[:]); err != nil {
		return nil, nil, nil, nil, 0, 0, 0, err
	}
	if magic != indexMagic {
		return nil, nil, nil, nil, 0, 0, 0, errors.New("invalid index magic")
	}

	var n32, k32, d32 uint32
	if err := binary.Read(file, binary.LittleEndian, &n32); err != nil {
		return nil, nil, nil, nil, 0, 0, 0, err
	}
	if err := binary.Read(file, binary.LittleEndian, &k32); err != nil {
		return nil, nil, nil, nil, 0, 0, 0, err
	}
	if err := binary.Read(file, binary.LittleEndian, &d32); err != nil {
		return nil, nil, nil, nil, 0, 0, 0, err
	}
	if d32 != vectorDimensions {
		return nil, nil, nil, nil, 0, 0, 0, fmt.Errorf("unexpected dimensions %d", d32)
	}
	n := int(n32)
	k := int(k32)

	centroids := make([]float32, k*vectorDimensions)
	if err := binary.Read(file, binary.LittleEndian, centroids); err != nil {
		return nil, nil, nil, nil, 0, 0, 0, err
	}

	offsets := make([]uint32, k+1)
	if err := binary.Read(file, binary.LittleEndian, offsets); err != nil {
		return nil, nil, nil, nil, 0, 0, 0, err
	}

	totalBlocks := int(offsets[k])
	paddedN := totalBlocks * blockSize
	labels := make([]byte, paddedN)
	if _, err := io.ReadFull(file, labels); err != nil {
		return nil, nil, nil, nil, 0, 0, 0, err
	}
	blocks := make([]int16, totalBlocks*blockStride)
	if err := binary.Read(file, binary.LittleEndian, blocks); err != nil {
		return nil, nil, nil, nil, 0, 0, 0, err
	}
	return centroids, offsets, labels, blocks, n, k, paddedN, nil
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func round4(v float32) float32 {
	return float32(math.Round(float64(v)*10000) * 0.0001)
}

func safeDivisor(v float32) float32 {
	if v <= 0 {
		return 1
	}
	return v
}

func dayOfWeek(t time.Time) float32 {
	return float32((int(t.UTC().Weekday()) + 6) % 7)
}

func parseMCC(raw string) (int, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, false
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 || value >= 10000 {
		return 0, false
	}
	return value, true
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}
