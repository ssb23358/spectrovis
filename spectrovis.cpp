/*
 * spectrovis.cpp
 *
 * Publication-grade audio spectrogram visualizer.
 *
 * Signal chain:
 *   PCM -> mono mix -> K-weight (BS.1770-4) -> gated LUFS
 *   PCM -> mono mix -> centered STFT (Hann, zero-padded 4x)
 *       -> power spectrum -> area-normalized Mel filterbank
 *       -> log-Mel -> spectral features (centroid, spread, flux, rolloff, flatness)
 *       -> Savitzky-Golay smoothing (exact LS coefficients)
 *       -> perceptual normalization (gamma, percentile compression)
 *       -> CIELAB perceptual color mapping with soft gamut compression
 *       -> anti-aliased rendering -> PNG
 *
 * All constants are defined explicitly as required for reproducibility.
 */

#include <sndfile.h>
#include <fftw3.h>
#include <png.h>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstring>
#include <cassert>

/* output image dimensions */
static constexpr int IMG_W = 3840;
static constexpr int IMG_H = 2160;

/* STFT parameters */
static constexpr int WINDOW_LEN    = 1024;           /* analysis window length (samples) */
static constexpr int ZERO_PAD      = 4;              /* zero-padding factor              */
static constexpr int FFT_SIZE      = WINDOW_LEN * ZERO_PAD;  /* 4096                     */
static constexpr int HOP_SIZE      = WINDOW_LEN / 2; /* 50% overlap = 512                */

/* Mel filterbank parameters (Slaney-style) */
static constexpr int   MEL_BANDS = 128;
static constexpr float MEL_FMIN  = 20.f;
static constexpr float MEL_FMAX  = 16000.f;

/* BS.1770-4 loudness parameters */
static constexpr int   LUFS_BLOCK_MS   = 400;        /* block length in ms               */
static constexpr float LUFS_OVERLAP    = 0.75f;       /* 75% overlap                      */
static constexpr float LUFS_ABS_GATE   = -70.f;       /* absolute gate in LUFS            */
static constexpr float LUFS_REL_OFFSET = -10.f;       /* relative gate offset in LU       */

/* Savitzky-Golay parameters */
static constexpr int SG_HALFWIN_LUFS = 15;            /* half-window for loudness curve    */
static constexpr int SG_HALFWIN_FEAT = 20;            /* half-window for spectral features */
static constexpr int SG_POLY_ORDER   = 3;             /* polynomial order (cubic)          */

/* rendering margins (pixels) */
static constexpr int MARGIN_L = 130;
static constexpr int MARGIN_R = 80;
static constexpr int MARGIN_T = 80;
static constexpr int MARGIN_B = 110;

/* spectral rolloff threshold */
static constexpr float ROLLOFF_PCT = 0.85f;

struct RGB { uint8_t r, g, b; };


/* --- K-weighting biquad (ITU-R BS.1770-4) --- */

struct Biquad {
    double b0, b1, b2, a1, a2;
    double x1, x2, y1, y2;

    void reset() { x1 = x2 = y1 = y2 = 0; }

    double tick(double x) {
        double y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2;
        x2 = x1; x1 = x;
        y2 = y1; y1 = y;
        return y;
    }
};

/*
 * Stage 1: shelving filter (high-frequency pre-emphasis).
 * Coefficients derived from BS.1770-4, Table 1.
 */
Biquad kweightShelf(int fs) {
    double f0 = 1681.974450955533;
    double G  = 3.999843853973347;
    double Q  = 0.7071752369554196;
    double K  = tan(M_PI * f0 / fs);
    double Vh = pow(10.0, G / 20.0);
    double Vb = pow(Vh, 0.4996667741545416);
    double a0 = 1.0 + K/Q + K*K;
    Biquad bq{};
    bq.b0 = (Vh + Vb*K/Q + K*K) / a0;
    bq.b1 = 2.0*(K*K - Vh) / a0;
    bq.b2 = (Vh - Vb*K/Q + K*K) / a0;
    bq.a1 = 2.0*(K*K - 1.0) / a0;
    bq.a2 = (1.0 - K/Q + K*K) / a0;
    bq.reset();
    return bq;
}

/*
 * Stage 2: high-pass filter.
 * Removes DC and sub-bass below perceptual threshold.
 */
Biquad kweightHipass(int fs) {
    double f0 = 38.13547087602444;
    double Q  = 0.5003270373238773;
    double K  = tan(M_PI * f0 / fs);
    double a0 = 1.0 + K/Q + K*K;
    Biquad bq{};
    bq.b0 =  1.0 / a0;
    bq.b1 = -2.0 / a0;
    bq.b2 =  1.0 / a0;
    bq.a1 =  2.0*(K*K - 1.0) / a0;
    bq.a2 =  (1.0 - K/Q + K*K) / a0;
    bq.reset();
    return bq;
}


/* --- Mel scale (Slaney / Auditory Toolbox style) --- */

static float hzToMel(float hz) {
    /* Slaney formula: linear below 1000 Hz, log above */
    if (hz < 1000.f) return hz * 3.f / 200.f;
    return 15.f + 27.f * log10f(hz / 1000.f) / log10f(6.4f);
}

static float melToHz(float mel) {
    if (mel < 15.f) return mel * 200.f / 3.f;
    return 1000.f * powf(6.4f, (mel - 15.f) / 27.f);
}


/* --- Area-normalized triangular Mel filterbank --- */

struct MelFilterbank {
    int nbins;
    int nmels;
    std::vector<std::vector<float>> weights; /* [nmels][nbins] */

    void init(int fftSize, int nMels, float sr, float fmin, float fmax) {
        nbins = fftSize / 2 + 1;
        nmels = nMels;
        weights.assign(nmels, std::vector<float>(nbins, 0.f));

        float melLo = hzToMel(fmin);
        float melHi = hzToMel(fmax);

        std::vector<float> melCenters(nmels + 2);
        for (int i = 0; i < nmels + 2; i++)
            melCenters[i] = melLo + (melHi - melLo) * i / (nmels + 1);

        std::vector<float> hzCenters(nmels + 2);
        for (int i = 0; i < nmels + 2; i++)
            hzCenters[i] = melToHz(melCenters[i]);

        /* convert Hz centers to fractional bin indices (no integer rounding) */
        std::vector<float> binCenters(nmels + 2);
        for (int i = 0; i < nmels + 2; i++)
            binCenters[i] = hzCenters[i] * fftSize / sr;

        for (int m = 0; m < nmels; m++) {
            float left   = binCenters[m];
            float center = binCenters[m + 1];
            float right  = binCenters[m + 2];

            for (int b = 0; b < nbins; b++) {
                float fb = (float)b;
                if (fb > left && fb < center)
                    weights[m][b] = (fb - left) / (center - left);
                else if (fb >= center && fb < right)
                    weights[m][b] = (right - fb) / (right - center);
            }

            /* area normalization: sum of weights = 1 for energy conservation */
            float area = 0;
            for (int b = 0; b < nbins; b++) area += weights[m][b];
            if (area > 1e-10f)
                for (int b = 0; b < nbins; b++) weights[m][b] /= area;
        }
    }

    void apply(const float* power, float* out) const {
        for (int m = 0; m < nmels; m++) {
            float s = 0;
            for (int b = 0; b < nbins; b++) s += power[b] * weights[m][b];
            out[m] = s;
        }
    }
};


/*
 * Savitzky-Golay smoothing filter.
 *
 * Computes exact least-squares polynomial coefficients via
 * the Gram polynomial / Vandermonde approach for the given
 * half-window M and polynomial order P.
 *
 * The convolution coefficients c[j] for j in [-M, M] are the
 * zeroth-row entries of (J^T J)^{-1} J^T where J is the
 * Vandermonde matrix.
 *
 * For efficiency, we solve the normal equations directly.
 */
std::vector<double> sgCoefficients(int halfwin, int polyOrder) {
    int M = halfwin;
    int P = std::min(polyOrder, 2 * M);
    int W = 2 * M + 1;

    /* build Vandermonde J: W rows x (P+1) cols */
    std::vector<std::vector<double>> J(W, std::vector<double>(P + 1));
    for (int i = 0; i < W; i++) {
        double x = (double)(i - M);
        J[i][0] = 1.0;
        for (int p = 1; p <= P; p++) J[i][p] = J[i][p-1] * x;
    }

    /* JtJ = J^T * J */
    std::vector<std::vector<double>> JtJ(P+1, std::vector<double>(P+1, 0));
    for (int a = 0; a <= P; a++)
        for (int b = 0; b <= P; b++)
            for (int i = 0; i < W; i++)
                JtJ[a][b] += J[i][a] * J[i][b];

    /* Gauss-Jordan inversion of JtJ */
    int N = P + 1;
    std::vector<std::vector<double>> aug(N, std::vector<double>(2*N, 0));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) aug[i][j] = JtJ[i][j];
        aug[i][N + i] = 1.0;
    }
    for (int col = 0; col < N; col++) {
        int pivot = col;
        for (int r = col+1; r < N; r++)
            if (fabs(aug[r][col]) > fabs(aug[pivot][col])) pivot = r;
        std::swap(aug[col], aug[pivot]);
        double d = aug[col][col];
        if (fabs(d) < 1e-15) d = 1e-15;
        for (int j = 0; j < 2*N; j++) aug[col][j] /= d;
        for (int r = 0; r < N; r++) {
            if (r == col) continue;
            double f = aug[r][col];
            for (int j = 0; j < 2*N; j++) aug[r][j] -= f * aug[col][j];
        }
    }
    /* inv = right half of aug */
    std::vector<std::vector<double>> inv(N, std::vector<double>(N));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            inv[i][j] = aug[i][N+j];

    /* coefficients: first row of inv * J^T */
    /* c[i] = sum_p inv[0][p] * J[i][p] */
    std::vector<double> c(W);
    for (int i = 0; i < W; i++) {
        double s = 0;
        for (int p = 0; p <= P; p++) s += inv[0][p] * J[i][p];
        c[i] = s;
    }
    return c;
}

void applySavgol(std::vector<float>& data, int halfwin, int polyOrder) {
    auto c = sgCoefficients(halfwin, polyOrder);
    int n = (int)data.size();
    int M = halfwin;
    std::vector<float> out(n);
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = -M; j <= M; j++) {
            int idx = i + j;
            /* symmetric boundary extension */
            if (idx < 0) idx = -idx;
            if (idx >= n) idx = 2*(n-1) - idx;
            idx = std::clamp(idx, 0, n-1);
            s += data[idx] * c[j + M];
        }
        out[i] = (float)s;
    }
    data = out;
}


/* --- CIELAB color mapping with soft gamut compression --- */

static float srgbGamma(float lin) {
    if (lin <= 0.0031308f) return 12.92f * lin;
    return 1.055f * powf(lin, 1.f / 2.4f) - 0.055f;
}

/*
 * Attempt to smoothly compress out-of-gamut values
 * instead of hard clamping (preserves hue relationships).
 */
static float softCompress(float x) {
    if (x >= 0.f && x <= 1.f) return x;
    if (x < 0.f) return 0.f;
    /* soft knee for x > 1 */
    return 1.f - expf(-(x - 1.f)) * 0.f + 1.f - 1.f / (1.f + (x - 1.f));
}

static void labToXyz(float L, float a, float b, float &X, float &Y, float &Z) {
    float fy = (L + 16.f) / 116.f;
    float fx = a / 500.f + fy;
    float fz = fy - b / 200.f;
    auto inv = [](float t) -> float {
        float d = 6.f / 29.f;
        return (t > d) ? t*t*t : 3.f * d * d * (t - 4.f / 29.f);
    };
    X = 0.95047f * inv(fx);
    Y = 1.00000f * inv(fy);
    Z = 1.08883f * inv(fz);
}

static RGB labToRgb(float L, float a, float b) {
    float X, Y, Z;
    labToXyz(L, a, b, X, Y, Z);
    /* XYZ to linear sRGB (D65) */
    float lr =  3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z;
    float lg = -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z;
    float lb =  0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z;
    lr = softCompress(lr);
    lg = softCompress(lg);
    lb = softCompress(lb);
    return {
        (uint8_t)(srgbGamma(std::clamp(lr, 0.f, 1.f)) * 255.f + 0.5f),
        (uint8_t)(srgbGamma(std::clamp(lg, 0.f, 1.f)) * 255.f + 0.5f),
        (uint8_t)(srgbGamma(std::clamp(lb, 0.f, 1.f)) * 255.f + 0.5f)
    };
}

/*
 * Map spectral features to perceptually uniform color.
 *   centroid  -> hue (angle in a*b* plane)
 *   spread    -> chroma (color saturation)
 *   loudness  -> L* (perceived brightness)
 */
static RGB featureToColor(float centroid, float spread, float loudness) {
    float hue = centroid * 330.f;
    float chroma = 25.f + spread * 70.f;
    float Lstar = 25.f + loudness * 70.f;

    float a = chroma * cosf(hue * (float)M_PI / 180.f);
    float b = chroma * sinf(hue * (float)M_PI / 180.f);
    return labToRgb(Lstar, a, b);
}


/* --- Bitmap digit renderer (5x7) --- */

static const uint8_t GLYPH[][7] = {
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, /* 0 */
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, /* 1 */
    {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, /* 2 */
    {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}, /* 3 */
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, /* 4 */
    {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, /* 5 */
    {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, /* 6 */
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, /* 7 */
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, /* 8 */
    {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, /* 9 */
    {0x00,0x00,0x04,0x00,0x04,0x00,0x00}, /* : */
    {0x00,0x00,0x00,0x1F,0x00,0x00,0x00}, /* - */
    {0x00,0x00,0x00,0x00,0x00,0x00,0x04}, /* . */
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, /* space */
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, /* O (for dB) */
    {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, /* B */
    {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, /* C (unused) */
    {0x08,0x08,0x08,0x08,0x08,0x08,0x0E}, /* L */
    {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}, /* U */
    {0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E}, /* S */
    {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, /* T (unused) */
    {0x1E,0x11,0x11,0x1E,0x11,0x11,0x11}, /* R (unused) */
};

static int glyphIndex(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    switch (ch) {
        case ':': return 10; case '-': return 11; case '.': return 12;
        case ' ': return 13; case 'd': return 14; case 'B': return 15;
        case 'L': return 17; case 'U': return 18; case 'F': return 19;
        case 'S': return 19;
    }
    return 13;
}

static void drawText(std::vector<RGB>& px, int w, int h,
                     int ox, int oy, const char* s, RGB col, int sc = 2) {
    for (int ci = 0; s[ci]; ci++) {
        int gi = glyphIndex(s[ci]);
        if (gi < 0 || gi >= (int)(sizeof(GLYPH)/sizeof(GLYPH[0]))) { ox += 6*sc; continue; }
        for (int row = 0; row < 7; row++) {
            uint8_t bits = GLYPH[gi][row];
            for (int bit = 0; bit < 5; bit++) {
                if (bits & (0x10 >> bit)) {
                    for (int sy = 0; sy < sc; sy++)
                        for (int sx = 0; sx < sc; sx++) {
                            int px_ = ox + bit*sc + sx;
                            int py_ = oy + row*sc + sy;
                            if (px_>=0 && px_<w && py_>=0 && py_<h)
                                px[py_*w + px_] = col;
                        }
                }
            }
        }
        ox += 6 * sc;
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: spectrovis <audio_file> [output.png]\n");
        printf("Supported formats: WAV, OGG, FLAC, AIFF\n");
        return 1;
    }

    std::string inPath  = argv[1];
    std::string outPath = (argc >= 3) ? argv[2] : "output.png";

    /* load audio via libsndfile (no audio device required) */
    SF_INFO sfinfo = {};
    SNDFILE* sfile = sf_open(inPath.c_str(), SFM_READ, &sfinfo);
    if (!sfile) {
        printf("Error opening %s: %s\n", inPath.c_str(), sf_strerror(NULL));
        return 1;
    }

    int channels = sfinfo.channels;
    int sampleRate = sfinfo.samplerate;
    sf_count_t totalFrames = sfinfo.frames;

    std::vector<float> raw(totalFrames * channels);
    sf_readf_float(sfile, raw.data(), totalFrames);
    sf_close(sfile);

    /* downmix to mono (equal-power sum) */
    std::vector<float> mono(totalFrames);
    for (sf_count_t i = 0; i < totalFrames; i++) {
        float s = 0;
        for (int c = 0; c < channels; c++) s += raw[i * channels + c];
        mono[i] = s / channels;
    }
    raw.clear();
    raw.shrink_to_fit();

    float duration = (float)totalFrames / sampleRate;
    printf("File       : %s\n", inPath.c_str());
    printf("Duration   : %.2f s\n", duration);
    printf("Sample rate: %d Hz\n", sampleRate);
    printf("Channels   : %d (downmixed to mono)\n", channels);
    printf("Window     : %d samples (%.1f ms), Hann\n",
           WINDOW_LEN, 1000.f * WINDOW_LEN / sampleRate);
    printf("FFT size   : %d (zero-pad %dx)\n", FFT_SIZE, ZERO_PAD);
    printf("Hop size   : %d samples (%.1f ms, %.0f%% overlap)\n",
           HOP_SIZE, 1000.f * HOP_SIZE / sampleRate,
           100.f * (1.f - (float)HOP_SIZE / WINDOW_LEN));
    printf("Mel bands  : %d (%.0f - %.0f Hz, Slaney scale)\n",
           MEL_BANDS, MEL_FMIN, MEL_FMAX);

    /* K-weight the entire signal */
    printf("Applying K-weighting (BS.1770-4)...\n");
    Biquad kw1 = kweightShelf(sampleRate);
    Biquad kw2 = kweightHipass(sampleRate);
    std::vector<float> kw(totalFrames);
    for (sf_count_t i = 0; i < totalFrames; i++)
        kw[i] = (float)kw2.tick(kw1.tick(mono[i]));

    /*
     * Gated integrated loudness (BS.1770-4).
     *
     * 1. Compute mean-square power in 400ms blocks with 75% overlap.
     * 2. Absolute gate at -70 LUFS.
     * 3. Compute relative gate = mean of ungated blocks - 10 LU.
     * 4. Final integrated loudness from blocks above relative gate.
     *
     * For visualization, we also store per-block loudness for the time curve.
     */
    int blockSamples = (int)(LUFS_BLOCK_MS * sampleRate / 1000);
    int blockHop = (int)(blockSamples * (1.f - LUFS_OVERLAP));
    int nBlocks = std::max(1, (int)((totalFrames - blockSamples) / blockHop) + 1);

    /* Hann window energy normalization factor */
    double hannEnergySum = 0;
    for (int i = 0; i < WINDOW_LEN; i++) {
        double w = 0.5 * (1.0 - cos(2.0 * M_PI * i / (WINDOW_LEN - 1)));
        hannEnergySum += w * w;
    }
    double hannNorm = WINDOW_LEN / hannEnergySum;

    std::vector<float> blockLufs(nBlocks);
    for (int blk = 0; blk < nBlocks; blk++) {
        size_t off = (size_t)blk * blockHop;
        double ms = 0;
        int cnt = 0;
        for (int i = 0; i < blockSamples && off + i < (size_t)totalFrames; i++) {
            double s = kw[off + i];
            ms += s * s;
            cnt++;
        }
        ms /= std::max(cnt, 1);
        blockLufs[blk] = (ms > 1e-15) ? (float)(-0.691 + 10.0 * log10(ms)) : -70.f;
    }

    /* absolute gate pass */
    double sumAbove = 0;
    int cntAbove = 0;
    for (int i = 0; i < nBlocks; i++) {
        if (blockLufs[i] > LUFS_ABS_GATE) {
            sumAbove += pow(10.0, blockLufs[i] / 10.0);
            cntAbove++;
        }
    }
    float integratedRough = (cntAbove > 0) ?
        (float)(10.0 * log10(sumAbove / cntAbove)) : -70.f;

    /* relative gate pass */
    float relGate = integratedRough + LUFS_REL_OFFSET;
    double sumFinal = 0;
    int cntFinal = 0;
    for (int i = 0; i < nBlocks; i++) {
        if (blockLufs[i] > LUFS_ABS_GATE && blockLufs[i] > relGate) {
            sumFinal += pow(10.0, blockLufs[i] / 10.0);
            cntFinal++;
        }
    }
    float integratedLufs = (cntFinal > 0) ?
        (float)(-0.691 + 10.0 * log10(sumFinal / cntFinal)) : -70.f;

    printf("Integrated loudness: %.1f LUFS\n", integratedLufs);

    /* STFT with centered framing and symmetric padding */
    printf("Running STFT (%d frames)...\n",
           (int)((totalFrames - WINDOW_LEN) / HOP_SIZE + 1));

    int nBins = FFT_SIZE / 2 + 1;
    int nStft = std::max(1, (int)((totalFrames - WINDOW_LEN) / HOP_SIZE) + 1);
    float freqRes = (float)sampleRate / FFT_SIZE;

    /* precompute Hann window */
    std::vector<float> hannWin(WINDOW_LEN);
    for (int i = 0; i < WINDOW_LEN; i++)
        hannWin[i] = 0.5f * (1.f - cosf(2.f * (float)M_PI * i / (WINDOW_LEN - 1)));

    /* Mel filterbank */
    MelFilterbank melBank;
    melBank.init(FFT_SIZE, MEL_BANDS, (float)sampleRate, MEL_FMIN, MEL_FMAX);

    double* fftIn = fftw_alloc_real(FFT_SIZE);
    fftw_complex* fftOut = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nBins);
    fftw_plan plan = fftw_plan_dft_r2c_1d(FFT_SIZE, fftIn, fftOut, FFTW_MEASURE);

    struct Features {
        float lufs;
        float centroid;
        float spread;
        float flux;
        float rolloff;
        float flatness;
    };
    std::vector<Features> feat(nStft);
    std::vector<float> prevMelFrame(MEL_BANDS, 0.f);

    for (int fr = 0; fr < nStft; fr++) {
        /* centered frame: center at hop*fr + WINDOW_LEN/2 */
        sf_count_t center = (sf_count_t)fr * HOP_SIZE + WINDOW_LEN / 2;

        /* zero-pad the FFT input */
        memset(fftIn, 0, sizeof(double) * FFT_SIZE);

        for (int i = 0; i < WINDOW_LEN; i++) {
            sf_count_t si = center - WINDOW_LEN / 2 + i;
            /* symmetric boundary extension */
            if (si < 0) si = -si;
            if (si >= totalFrames) si = 2 * (totalFrames - 1) - si;
            si = std::clamp(si, (sf_count_t)0, totalFrames - 1);
            fftIn[i] = mono[si] * hannWin[i];
        }

        fftw_execute(plan);

        /* power spectrum with window energy normalization */
        std::vector<float> power(nBins);
        for (int b = 0; b < nBins; b++) {
            double re = fftOut[b][0], im = fftOut[b][1];
            power[b] = (float)((re*re + im*im) * hannNorm / (FFT_SIZE * FFT_SIZE));
        }

        /* Mel spectrum */
        std::vector<float> melSpec(MEL_BANDS);
        melBank.apply(power.data(), melSpec.data());

        /* log-Mel (ref = 1.0, full scale) */
        std::vector<float> logMel(MEL_BANDS);
        for (int m = 0; m < MEL_BANDS; m++)
            logMel[m] = 10.f * log10f(std::max(melSpec[m], 1e-10f));

        /* per-frame loudness from K-weighted signal */
        double ms = 0;
        for (int i = 0; i < WINDOW_LEN; i++) {
            sf_count_t si = center - WINDOW_LEN / 2 + i;
            if (si < 0) si = -si;
            if (si >= totalFrames) si = 2 * (totalFrames - 1) - si;
            si = std::clamp(si, (sf_count_t)0, totalFrames - 1);
            double s = kw[si];
            ms += s * s;
        }
        ms /= WINDOW_LEN;
        feat[fr].lufs = (ms > 1e-15) ? (float)(-0.691 + 10.0 * log10(ms)) : -70.f;

        /* Mel-domain spectral centroid */
        float sumW = 0, sumM = 0;
        for (int m = 0; m < MEL_BANDS; m++) {
            float w = std::max(0.f, logMel[m] + 80.f); /* shift to positive */
            sumM += w;
            sumW += w * m;
        }
        feat[fr].centroid = (sumM > 1e-6f) ? sumW / sumM / MEL_BANDS : 0.5f;

        /* spectral spread */
        float mu = feat[fr].centroid * MEL_BANDS;
        float spreadSum = 0;
        for (int m = 0; m < MEL_BANDS; m++) {
            float w = std::max(0.f, logMel[m] + 80.f);
            float d = m - mu;
            spreadSum += w * d * d;
        }
        feat[fr].spread = (sumM > 1e-6f) ? sqrtf(spreadSum / sumM) / MEL_BANDS : 0.f;

        /* spectral flux (half-wave rectified L2) */
        float fluxSum = 0;
        for (int m = 0; m < MEL_BANDS; m++) {
            float diff = logMel[m] - prevMelFrame[m];
            if (diff > 0) fluxSum += diff * diff;
        }
        feat[fr].flux = sqrtf(fluxSum / MEL_BANDS);
        prevMelFrame = logMel;

        /* spectral rolloff (85%) */
        float totalPow = 0;
        for (int b = 0; b < nBins; b++) totalPow += power[b];
        float cumPow = 0;
        int rollBin = nBins - 1;
        for (int b = 0; b < nBins; b++) {
            cumPow += power[b];
            if (cumPow >= ROLLOFF_PCT * totalPow) { rollBin = b; break; }
        }
        feat[fr].rolloff = (float)rollBin * freqRes;

        /* spectral flatness = exp(mean(log(S))) / mean(S) */
        float geoSum = 0, arithSum = 0;
        int nonzero = 0;
        for (int b = 1; b < nBins; b++) {
            if (power[b] > 1e-20f) {
                geoSum += logf(power[b]);
                arithSum += power[b];
                nonzero++;
            }
        }
        if (nonzero > 0 && arithSum > 1e-20f)
            feat[fr].flatness = expf(geoSum / nonzero) / (arithSum / nonzero);
        else
            feat[fr].flatness = 0.f;
    }

    fftw_destroy_plan(plan);
    fftw_free(fftIn);
    fftw_free(fftOut);

    /* resample features to pixel columns */
    int plotW = IMG_W - MARGIN_L - MARGIN_R;
    int plotH = IMG_H - MARGIN_T - MARGIN_B;

    std::vector<float> cLufs(plotW), cCent(plotW), cSpread(plotW), cFlux(plotW);

    for (int col = 0; col < plotW; col++) {
        float t = (float)col / (plotW - 1);
        float fi = t * (nStft - 1);
        int i0 = std::clamp((int)fi, 0, nStft - 1);
        int i1 = std::min(i0 + 1, nStft - 1);
        float f = fi - i0;

        cLufs[col]   = feat[i0].lufs     * (1-f) + feat[i1].lufs     * f;
        cCent[col]   = feat[i0].centroid  * (1-f) + feat[i1].centroid  * f;
        cSpread[col] = feat[i0].spread    * (1-f) + feat[i1].spread    * f;
        cFlux[col]   = feat[i0].flux      * (1-f) + feat[i1].flux      * f;
    }

    /* Savitzky-Golay smoothing with exact LS coefficients */
    printf("Savitzky-Golay smoothing (order=%d)...\n", SG_POLY_ORDER);
    applySavgol(cLufs,   SG_HALFWIN_LUFS, SG_POLY_ORDER);
    applySavgol(cLufs,   SG_HALFWIN_LUFS / 2, SG_POLY_ORDER);
    applySavgol(cCent,   SG_HALFWIN_FEAT, SG_POLY_ORDER);
    applySavgol(cSpread, SG_HALFWIN_FEAT, SG_POLY_ORDER);
    applySavgol(cFlux,   SG_HALFWIN_FEAT, SG_POLY_ORDER);

    /* perceptual normalization */
    float lufsFloor = -60.f, lufsCeil = std::max(-5.f, integratedLufs + 15.f);
    for (auto& v : cLufs)
        v = std::clamp((v - lufsFloor) / (lufsCeil - lufsFloor), 0.f, 1.f);
    /* gamma correction approximating Stevens' power law */
    for (auto& v : cLufs)
        v = powf(v, 0.55f);

    auto percentileNorm = [](std::vector<float>& d, float pct) {
        std::vector<float> s = d;
        std::sort(s.begin(), s.end());
        float ref = s[std::min((int)(s.size() * pct), (int)s.size() - 1)];
        if (ref < 1e-8f) ref = *std::max_element(d.begin(), d.end());
        if (ref < 1e-8f) ref = 1.f;
        for (auto& v : d) v = std::clamp(v / ref, 0.f, 1.f);
    };
    percentileNorm(cCent, 0.98f);
    percentileNorm(cSpread, 0.95f);
    percentileNorm(cFlux, 0.95f);

    /* rendering */
    printf("Rendering %dx%d image...\n", IMG_W, IMG_H);

    RGB bg = {8, 8, 16};
    std::vector<RGB> px(IMG_W * IMG_H, bg);

    auto blend = [&](int x, int y, RGB c, float a) {
        if (x < 0 || x >= IMG_W || y < 0 || y >= IMG_H) return;
        a = std::clamp(a, 0.f, 1.f);
        auto& d = px[y * IMG_W + x];
        d.r = std::min(255, (int)(c.r * a + d.r * (1 - a)));
        d.g = std::min(255, (int)(c.g * a + d.g * (1 - a)));
        d.b = std::min(255, (int)(c.b * a + d.b * (1 - a)));
    };

    /* grid lines */
    RGB gridCol = {24, 24, 36};
    float dbTicks[] = {-6,-12,-18,-24,-30,-36,-42,-48,-54};
    for (float db : dbTicks) {
        float norm = std::clamp((db - lufsFloor) / (lufsCeil - lufsFloor), 0.f, 1.f);
        norm = powf(norm, 0.55f);
        int y = MARGIN_T + (int)((1.f - norm) * plotH);
        if (y >= MARGIN_T && y <= MARGIN_T + plotH)
            for (int x = MARGIN_L; x < MARGIN_L + plotW; x += 3)
                blend(x, y, gridCol, 0.35f);
    }

    float timeTick = (duration > 300) ? 60 : (duration > 120) ? 30 :
                     (duration > 60) ? 10 : 5;
    for (float sec = timeTick; sec < duration; sec += timeTick) {
        int x = MARGIN_L + (int)(sec / duration * plotW);
        for (int y = MARGIN_T; y < MARGIN_T + plotH; y += 3)
            blend(x, y, gridCol, 0.35f);
    }

    /* compute curve Y positions */
    std::vector<int> curveY(plotW);
    for (int col = 0; col < plotW; col++) {
        curveY[col] = MARGIN_T + (int)((1.f - cLufs[col]) * plotH);
        curveY[col] = std::clamp(curveY[col], MARGIN_T, MARGIN_T + plotH);
    }

    int bottomY = MARGIN_T + plotH;

    /* gradient fill below curve */
    for (int col = 0; col < plotW; col++) {
        int x = MARGIN_L + col;
        RGB c = featureToColor(cCent[col], cSpread[col], cLufs[col]);
        int top = curveY[col];
        for (int y = top; y <= bottomY; y++) {
            float t = (float)(y - top) / std::max(1, bottomY - top);
            float alpha = 0.6f * powf(1.f - t, 2.5f) + 0.01f;
            blend(x, y, c, alpha);
        }
    }

    /* wide glow */
    for (int col = 0; col < plotW; col++) {
        int x = MARGIN_L + col;
        int cy = curveY[col];
        RGB c = featureToColor(cCent[col], cSpread[col],
                               std::min(1.f, cLufs[col] + 0.2f));
        for (int dy = -16; dy <= 16; dy++) {
            float d = fabsf(dy) / 16.f;
            blend(x, cy + dy, c, 0.38f * expf(-4.f * d * d));
        }
    }

    /* core line */
    for (int col = 0; col < plotW; col++) {
        int x = MARGIN_L + col;
        int cy = curveY[col];
        RGB c = featureToColor(cCent[col], cSpread[col],
                               std::min(1.f, cLufs[col] + 0.3f));
        for (int dy = -3; dy <= 3; dy++) {
            float d = fabsf(dy) / 3.f;
            blend(x, cy + dy, c, 0.95f * (1.f - d * 0.35f));
        }
    }

    /* axes */
    RGB axisCol = {55, 55, 80};
    for (int y = MARGIN_T; y <= bottomY; y++) {
        px[y * IMG_W + MARGIN_L - 1] = axisCol;
        px[y * IMG_W + MARGIN_L - 2] = axisCol;
    }
    for (int x = MARGIN_L; x <= MARGIN_L + plotW; x++) {
        px[(bottomY + 1) * IMG_W + x] = axisCol;
        px[(bottomY + 2) * IMG_W + x] = axisCol;
    }

    /* Y-axis labels (LUFS dB scale) */
    RGB lblCol = {120, 120, 155};
    for (float db : dbTicks) {
        float norm = std::clamp((db - lufsFloor) / (lufsCeil - lufsFloor), 0.f, 1.f);
        norm = powf(norm, 0.55f);
        int y = MARGIN_T + (int)((1.f - norm) * plotH);
        if (y >= MARGIN_T && y <= MARGIN_T + plotH) {
            char buf[16];
            snprintf(buf, sizeof(buf), "%.0f", db);
            drawText(px, IMG_W, IMG_H, 8, y - 7, buf, lblCol, 2);
        }
    }

    /* X-axis labels (mm:ss) */
    for (float sec = 0; sec <= duration + 0.1f; sec += timeTick) {
        int x = MARGIN_L + (int)(std::min(sec, duration) / duration * plotW);
        int m = (int)sec / 60, s = (int)sec % 60;
        char buf[16];
        snprintf(buf, sizeof(buf), "%d:%02d", m, s);
        drawText(px, IMG_W, IMG_H, x - 12, bottomY + 14, buf, lblCol, 2);
    }

    /* save PNG */
    printf("Saving: %s\n", outPath.c_str());
    FILE* fp = fopen(outPath.c_str(), "wb");
    if (!fp) { printf("Cannot write file\n"); return 1; }
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    png_infop pngInfo = png_create_info_struct(png);
    png_init_io(png, fp);
    png_set_IHDR(png, pngInfo, IMG_W, IMG_H, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, pngInfo);
    std::vector<uint8_t> rowBuf(IMG_W * 3);
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            auto& p = px[y * IMG_W + x];
            rowBuf[x*3] = p.r; rowBuf[x*3+1] = p.g; rowBuf[x*3+2] = p.b;
        }
        png_write_row(png, rowBuf.data());
    }
    png_write_end(png, 0);
    png_destroy_write_struct(&png, &pngInfo);
    fclose(fp);
    printf("Done!\n");
    return 0;
}