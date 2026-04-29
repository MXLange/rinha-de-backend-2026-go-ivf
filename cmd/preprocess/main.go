package main

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
)

const (
	vectorDimensions = 14
	blockSize        = 8
	blockStride      = vectorDimensions * blockSize
	maxSampleSize    = 50000
)

var (
	normalizationMagic = [8]byte{'R', '2', '6', 'N', 'O', 'R', 'M', '1'}
	mccRiskMagic       = [8]byte{'R', '2', '6', 'M', 'C', 'C', '1', '0'}
	indexMagic         = [4]byte{'I', 'V', 'F', '1'}
)

type normalization struct {
	MaxAmount            float32 `json:"max_amount"`
	MaxInstallments      float32 `json:"max_installments"`
	AmountVsAvgRatio     float32 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float32 `json:"max_minutes"`
	MaxKM                float32 `json:"max_km"`
	MaxTxCount24h        float32 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float32 `json:"max_merchant_avg_amount"`
}

type referenceEntry struct {
	Vector [vectorDimensions]float32 `json:"vector"`
	Label  string                    `json:"label"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "preprocess: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	resourcesDir := getenv("RESOURCES_DIR", "/app/resources")
	referencesPath := getenv("REFERENCES_PATH", "references.json.gz")
	normalizationPath := getenv("NORMALIZATION_PATH", "normalization.json")
	mccRiskPath := getenv("MCC_RISK_PATH", "mcc_risk.json")
	k := getenvInt("KMEANS_K", 4096)
	maxIterations := getenvInt("KMEANS_ITERS", 25)

	if err := os.MkdirAll(resourcesDir, 0o755); err != nil {
		return fmt.Errorf("create resources dir: %w", err)
	}

	if err := writeNormalizationBin(normalizationPath, filepath.Join(resourcesDir, "normalization.bin")); err != nil {
		return err
	}
	if err := writeMCCRiskBin(mccRiskPath, filepath.Join(resourcesDir, "mcc_risk.bin")); err != nil {
		return err
	}
	if err := writeIndexBin(referencesPath, filepath.Join(resourcesDir, "index.bin"), k, maxIterations); err != nil {
		return err
	}
	return nil
}

func writeNormalizationBin(input, output string) error {
	var norm normalization
	if err := readJSONFile(input, &norm); err != nil {
		return fmt.Errorf("read normalization: %w", err)
	}
	file, err := os.Create(output)
	if err != nil {
		return fmt.Errorf("create normalization bin: %w", err)
	}
	defer file.Close()
	if _, err := file.Write(normalizationMagic[:]); err != nil {
		return err
	}
	if err := binary.Write(file, binary.LittleEndian, uint32(1)); err != nil {
		return err
	}
	values := [...]float32{
		norm.MaxAmount,
		norm.MaxInstallments,
		norm.AmountVsAvgRatio,
		norm.MaxMinutes,
		norm.MaxKM,
		norm.MaxTxCount24h,
		norm.MaxMerchantAvgAmount,
	}
	for _, value := range values {
		if err := binary.Write(file, binary.LittleEndian, value); err != nil {
			return err
		}
	}
	return nil
}

func writeMCCRiskBin(input, output string) error {
	var raw map[string]float32
	if err := readJSONFile(input, &raw); err != nil {
		return fmt.Errorf("read mcc risk: %w", err)
	}
	table := make([]float32, 10000)
	for i := range table {
		table[i] = 0.5
	}
	for key, value := range raw {
		var idx int
		if _, err := fmt.Sscanf(key, "%d", &idx); err == nil && idx >= 0 && idx < len(table) {
			table[idx] = value
		}
	}
	file, err := os.Create(output)
	if err != nil {
		return fmt.Errorf("create mcc risk bin: %w", err)
	}
	defer file.Close()
	if _, err := file.Write(mccRiskMagic[:]); err != nil {
		return err
	}
	if err := binary.Write(file, binary.LittleEndian, uint32(1)); err != nil {
		return err
	}
	for _, value := range table {
		if err := binary.Write(file, binary.LittleEndian, value); err != nil {
			return err
		}
	}
	return nil
}

func writeIndexBin(input, output string, k, maxIterations int) error {
	vectors, labels, err := loadDataset(input)
	if err != nil {
		return err
	}
	if len(vectors) == 0 {
		return errors.New("empty dataset")
	}
	if k <= 0 {
		k = 1
	}
	if k > len(vectors) {
		k = len(vectors)
	}
	centroids := kmeansPlusPlusInit(vectors, k, 0x5eadbeef)
	assignments := make([]uint16, len(vectors))
	for i := 0; i < maxIterations; i++ {
		changed := assignParallel(vectors, centroids, assignments)
		updateCentroids(vectors, assignments, centroids)
		if changed*1000 < len(vectors) {
			break
		}
	}

	clusterVecs := make([][]int, k)
	for i, assignment := range assignments {
		clusterVecs[int(assignment)] = append(clusterVecs[int(assignment)], i)
	}

	offsets := make([]uint32, k+1)
	for ci := 0; ci < k; ci++ {
		size := uint32(len(clusterVecs[ci]))
		offsets[ci+1] = offsets[ci] + (size+blockSize-1)/blockSize
	}
	totalBlocks := int(offsets[k])
	paddedN := totalBlocks * blockSize
	outLabels := make([]byte, paddedN)
	outBlocks := make([]int16, totalBlocks*blockStride)

	for ci := 0; ci < k; ci++ {
		blockStart := int(offsets[ci])
		vecs := clusterVecs[ci]
		nBlocks := int(offsets[ci+1] - offsets[ci])
		for bk := 0; bk < nBlocks; bk++ {
			blockBase := (blockStart + bk) * blockStride
			labelBase := (blockStart + bk) * blockSize
			for slot := 0; slot < blockSize; slot++ {
				pos := bk*blockSize + slot
				if pos >= len(vecs) {
					for d := 0; d < vectorDimensions; d++ {
						outBlocks[blockBase+d*blockSize+slot] = math.MaxInt16
					}
					continue
				}
				vi := vecs[pos]
				for d := 0; d < vectorDimensions; d++ {
					outBlocks[blockBase+d*blockSize+slot] = quantizeI16(vectors[vi][d])
				}
				outLabels[labelBase+slot] = labels[vi]
			}
		}
	}

	centroidsT := make([]float32, vectorDimensions*k)
	for ci := 0; ci < k; ci++ {
		for d := 0; d < vectorDimensions; d++ {
			centroidsT[d*k+ci] = centroids[ci][d]
		}
	}

	file, err := os.Create(output)
	if err != nil {
		return fmt.Errorf("create index: %w", err)
	}
	defer file.Close()
	if _, err := file.Write(indexMagic[:]); err != nil {
		return err
	}
	if err := binary.Write(file, binary.LittleEndian, uint32(len(vectors))); err != nil {
		return err
	}
	if err := binary.Write(file, binary.LittleEndian, uint32(k)); err != nil {
		return err
	}
	if err := binary.Write(file, binary.LittleEndian, uint32(vectorDimensions)); err != nil {
		return err
	}
	for _, value := range centroidsT {
		if err := binary.Write(file, binary.LittleEndian, value); err != nil {
			return err
		}
	}
	for _, value := range offsets {
		if err := binary.Write(file, binary.LittleEndian, value); err != nil {
			return err
		}
	}
	if _, err := file.Write(outLabels); err != nil {
		return err
	}
	for _, value := range outBlocks {
		if err := binary.Write(file, binary.LittleEndian, value); err != nil {
			return err
		}
	}
	return nil
}

func loadDataset(path string) ([][vectorDimensions]float32, []byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open references: %w", err)
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, nil, fmt.Errorf("open gzip: %w", err)
	}
	defer gz.Close()

	dec := json.NewDecoder(gz)
	token, err := dec.Token()
	if err != nil {
		return nil, nil, fmt.Errorf("read references start: %w", err)
	}
	if delim, ok := token.(json.Delim); !ok || delim != '[' {
		return nil, nil, errors.New("references root is not array")
	}

	vectors := make([][vectorDimensions]float32, 0, 200000)
	labels := make([]byte, 0, 200000)
	for dec.More() {
		var entry referenceEntry
		if err := dec.Decode(&entry); err != nil {
			return nil, nil, err
		}
		vectors = append(vectors, entry.Vector)
		if entry.Label == "fraud" || entry.Label == "Fraud" || entry.Label == "FRAUD" {
			labels = append(labels, 1)
		} else {
			labels = append(labels, 0)
		}
	}
	if _, err := dec.Token(); err != nil {
		return nil, nil, fmt.Errorf("read references end: %w", err)
	}
	return vectors, labels, nil
}

func kmeansPlusPlusInit(vectors [][vectorDimensions]float32, k int, seed int64) [][vectorDimensions]float32 {
	rng := rand.New(rand.NewSource(seed))
	n := len(vectors)
	sampleSize := n
	if sampleSize > maxSampleSize {
		sampleSize = maxSampleSize
	}
	sample := make([]int, sampleSize)
	for i := range sample {
		sample[i] = rng.Intn(n)
	}

	centroids := make([][vectorDimensions]float32, 0, k)
	centroids = append(centroids, vectors[sample[rng.Intn(sampleSize)]])
	minDists := make([]float32, sampleSize)
	for i := range minDists {
		minDists[i] = math.MaxFloat32
	}

	for len(centroids) < k {
		last := centroids[len(centroids)-1]
		total := float64(0)
		for i, idx := range sample {
			d := distSq(vectors[idx], last)
			if d < minDists[i] {
				minDists[i] = d
			}
			total += float64(minDists[i])
		}
		target := rng.Float64() * total
		acc := float64(0)
		chosen := sampleSize - 1
		for i, d := range minDists {
			acc += float64(d)
			if acc >= target {
				chosen = i
				break
			}
		}
		centroids = append(centroids, vectors[sample[chosen]])
	}
	return centroids
}

func assignParallel(vectors [][vectorDimensions]float32, centroids [][vectorDimensions]float32, assignments []uint16) int {
	nThreads := runtime.GOMAXPROCS(0)
	if nThreads < 1 {
		nThreads = 1
	}
	if nThreads > 16 {
		nThreads = 16
	}
	chunk := (len(vectors) + nThreads - 1) / nThreads
	changedCh := make(chan int, nThreads)
	var wg sync.WaitGroup
	for start := 0; start < len(vectors); start += chunk {
		end := start + chunk
		if end > len(vectors) {
			end = len(vectors)
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			changed := 0
			for i := start; i < end; i++ {
				best := nearestCentroid(vectors[i], centroids)
				if best != assignments[i] {
					assignments[i] = best
					changed++
				}
			}
			changedCh <- changed
		}(start, end)
	}
	wg.Wait()
	close(changedCh)
	total := 0
	for changed := range changedCh {
		total += changed
	}
	return total
}

func nearestCentroid(v [vectorDimensions]float32, centroids [][vectorDimensions]float32) uint16 {
	bestDist := float32(math.MaxFloat32)
	bestIdx := uint16(0)
	for i, c := range centroids {
		d := distSq(v, c)
		if d < bestDist {
			bestDist = d
			bestIdx = uint16(i)
		}
	}
	return bestIdx
}

func updateCentroids(vectors [][vectorDimensions]float32, assignments []uint16, centroids [][vectorDimensions]float32) {
	sums := make([][vectorDimensions]float64, len(centroids))
	counts := make([]uint32, len(centroids))
	for i, v := range vectors {
		ci := int(assignments[i])
		counts[ci]++
		for d := 0; d < vectorDimensions; d++ {
			sums[ci][d] += float64(v[d])
		}
	}
	for i := range centroids {
		if counts[i] == 0 {
			continue
		}
		inv := float64(1) / float64(counts[i])
		for d := 0; d < vectorDimensions; d++ {
			centroids[i][d] = float32(sums[i][d] * inv)
		}
	}
}

func distSq(a, b [vectorDimensions]float32) float32 {
	total := float32(0)
	for i := 0; i < vectorDimensions; i++ {
		diff := a[i] - b[i]
		total += diff * diff
	}
	return total
}

func quantizeI16(v float32) int16 {
	if v > 1 {
		v = 1
	} else if v < -1 {
		v = -1
	}
	scaled := math.Round(float64(v) * 10000)
	if scaled > 32767 {
		scaled = 32767
	} else if scaled < -32768 {
		scaled = -32768
	}
	return int16(scaled)
}

func readJSONFile(path string, dest any) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return json.NewDecoder(file).Decode(dest)
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
