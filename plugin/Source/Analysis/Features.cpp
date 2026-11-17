#include "Features.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>

namespace drum::analysis
{
    namespace
    {
        constexpr double pi = 3.14159265358979323846;

        /** 第 1 種 0 次の変形ベッセル関数（べき級数。Kaiser 窓の β = 5 では 20 項ほどで収束する）。 */
        double besselI0 (double x)
        {
            const double q = 0.25 * x * x;
            double sum = 1.0, term = 1.0;
            for (int k = 1; k < 200; ++k)
            {
                term *= q / (static_cast<double> (k) * k);
                sum += term;
                if (term < 1e-17 * sum)
                    break;
            }
            return sum;
        }

        double sinc (double x)
        {
            if (std::fpclassify (x) == FP_ZERO)
                return 1.0;
            const double a = pi * x;
            return std::sin (a) / a;
        }

        /** `scipy.signal.firwin (numTaps, cutoff, window=("kaiser", beta))`: 窓付き sinc の低域通過。直流の利得を 1 にそろえる。 */
        std::vector<double> firwinKaiser (int numTaps, double cutoff, double beta)
        {
            std::vector<double> h (static_cast<size_t> (numTaps));
            const double alpha = 0.5 * (numTaps - 1);
            const double i0Beta = besselI0 (beta);
            double sum = 0.0;
            for (int n = 0; n < numTaps; ++n)
            {
                const double m = n - alpha;
                const double r = alpha > 0.0 ? m / alpha : 0.0;
                const double w = besselI0 (beta * std::sqrt (std::max (0.0, 1.0 - r * r))) / i0Beta;
                h[static_cast<size_t> (n)] = cutoff * sinc (cutoff * m) * w;
                sum += h[static_cast<size_t> (n)];
            }
            for (auto& v : h)
                v /= sum;
            return h;
        }

        /** 長さが 2 のべきの複素 FFT（その場で。倍精度）。 */
        void fftInPlace (std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& twiddles)
        {
            const size_t n = a.size();
            for (size_t i = 1, j = 0; i < n; ++i)
            {
                size_t bit = n >> 1;
                for (; j & bit; bit >>= 1)
                    j ^= bit;
                j ^= bit;
                if (i < j)
                    std::swap (a[i], a[j]);
            }
            for (size_t len = 2; len <= n; len <<= 1)
            {
                const size_t step = n / len;
                for (size_t i = 0; i < n; i += len)
                    for (size_t k = 0; k < len / 2; ++k)
                    {
                        const auto u = a[i + k];
                        const auto v = a[i + k + len / 2] * twiddles[k * step];
                        a[i + k] = u + v;
                        a[i + k + len / 2] = u - v;
                    }
            }
        }

        /** `numpy.linspace (start, stop, count)`。 */
        std::vector<double> linspace (double start, double stop, int count)
        {
            std::vector<double> out (static_cast<size_t> (count));
            if (count == 1)
            {
                out[0] = start;
                return out;
            }
            const double step = (stop - start) / (count - 1);
            for (int i = 0; i < count; ++i)
                out[static_cast<size_t> (i)] = start + i * step;
            out.back() = stop;
            return out;
        }

        /** 切り捨ての割り算（b > 0。負の数でも −∞ の向きに丸める）。 */
        long long floorDiv (long long a, long long b)
        {
            long long q = a / b;
            if ((a % b != 0) && (a < 0))
                --q;
            return q;
        }

        double hzToMel (double f) { return 2595.0 * std::log10 (1.0 + f / 700.0); }
        double melToHz (double m) { return 700.0 * (std::pow (10.0, m / 2595.0) - 1.0); }
    }

    std::vector<double> resamplePoly (const std::vector<double>& x, int up, int down)
    {
        const int g = std::gcd (up, down);
        up /= g;
        down /= g;
        if (up == 1 && down == 1)
            return x;

        const long long numIn = static_cast<long long> (x.size());
        const long long numOut = (numIn * up + down - 1) / down;
        const int maxRate = std::max (up, down);
        const int halfLen = 10 * maxRate;
        auto h = firwinKaiser (2 * halfLen + 1, 1.0 / maxRate, 5.0);
        for (auto& v : h)
            v *= up;

        // scipy は、出力の標本がフィルタの中心に来るようにフィルタの前に 0 を足し、出力の頭を捨てる。
        // 出力 i の中心は、0 を挟んで up 倍にした入力の位置 c = (i + preRemove)·down − prePad − halfLen
        const int prePad = down - halfLen % down;
        const int preRemove = (halfLen + prePad) / down;

        std::vector<double> y (static_cast<size_t> (numOut), 0.0);
        for (long long i = 0; i < numOut; ++i)
        {
            const long long centre = (i + preRemove) * down - prePad - halfLen;
            // 入力 x[k] は位置 k·up にある。フィルタの係数 h[halfLen + d] が掛かるのは位置 centre − d（|d| ≤ halfLen）
            const long long kLo = std::max (-floorDiv (-(centre - halfLen), up), 0LL); // ceil((centre − halfLen) / up)
            const long long kHi = std::min (floorDiv (centre + halfLen, up), numIn - 1);
            double acc = 0.0;
            for (long long k = kLo; k <= kHi; ++k)
            {
                const long long d = centre - k * up;
                acc += h[static_cast<size_t> (halfLen + d)] * x[static_cast<size_t> (k)];
            }
            y[static_cast<size_t> (i)] = acc;
        }
        return y;
    }

    std::vector<double> resampleTo (const std::vector<double>& x, int sampleRate, int targetRate)
    {
        if (sampleRate == targetRate)
            return x;
        const int g = std::gcd (sampleRate, targetRate);
        return resamplePoly (x, targetRate / g, sampleRate / g);
    }

    std::vector<double> melFilterbank (const FeatureConfig& config)
    {
        const int numBins = config.fftSize / 2 + 1;
        const auto f = linspace (0.0, config.sampleRate / 2.0, numBins);
        const auto melPoints = linspace (hzToMel (config.fMin), hzToMel (config.fMax), config.numMels + 2);
        std::vector<double> hzPoints (melPoints.size());
        for (size_t i = 0; i < melPoints.size(); ++i)
            hzPoints[i] = melToHz (melPoints[i]);

        std::vector<double> fb (static_cast<size_t> (config.numMels) * static_cast<size_t> (numBins), 0.0);
        for (int i = 0; i < config.numMels; ++i)
        {
            const double lo = hzPoints[static_cast<size_t> (i)], c = hzPoints[static_cast<size_t> (i) + 1], hi = hzPoints[static_cast<size_t> (i) + 2];
            for (int k = 0; k < numBins; ++k)
            {
                const double up = (f[static_cast<size_t> (k)] - lo) / std::max (c - lo, 1e-9);
                const double dn = (hi - f[static_cast<size_t> (k)]) / std::max (hi - c, 1e-9);
                fb[static_cast<size_t> (i) * static_cast<size_t> (numBins) + static_cast<size_t> (k)] = std::max (std::min (up, dn), 0.0);
            }
        }
        return fb;
    }

    LogMel logMel (const std::vector<double>& x, const FeatureConfig& config)
    {
        const int n = config.fftSize;
        const int hop = config.hop;
        const int numBins = n / 2 + 1;
        LogMel out;
        out.numMels = config.numMels;
        if (x.empty() || n < 2 || (n & (n - 1)) != 0 || hop < 1 || hop > n)
            return out; // fftSize は 2 のべき

        // scipy.signal.stft (boundary="zeros", padded=True): 両端に n/2 の 0 を足し、最後の枠が埋まるまで 0 を足す
        const long long extended = static_cast<long long> (x.size()) + 2LL * (n / 2);
        long long pad = (-(extended - n)) % hop;
        if (pad < 0)
            pad += hop;
        pad %= n;
        const long long total = extended + pad;
        const int numFrames = static_cast<int> ((total - (n - hop)) / hop);
        out.numFrames = numFrames;
        out.data.assign (static_cast<size_t> (numFrames) * static_cast<size_t> (config.numMels), 0.0f);

        // 周期的な Hann 窓（scipy.signal.get_window ("hann", n)）。倍率は 1 / 窓の和
        std::vector<double> window (static_cast<size_t> (n));
        double windowSum = 0.0;
        for (int i = 0; i < n; ++i)
        {
            window[static_cast<size_t> (i)] = 0.5 - 0.5 * std::cos (2.0 * pi * i / n);
            windowSum += window[static_cast<size_t> (i)];
        }
        std::vector<std::complex<double>> twiddles (static_cast<size_t> (n / 2));
        for (int k = 0; k < n / 2; ++k)
            twiddles[static_cast<size_t> (k)] = std::polar (1.0, -2.0 * pi * k / n);

        const auto fb = melFilterbank (config);
        // 各メルの帯域で 0 でない範囲だけを足す
        std::vector<int> first (static_cast<size_t> (config.numMels), 0), last (static_cast<size_t> (config.numMels), -1);
        for (int m = 0; m < config.numMels; ++m)
        {
            int lo = numBins, hi = -1;
            for (int k = 0; k < numBins; ++k)
                if (fb[static_cast<size_t> (m) * static_cast<size_t> (numBins) + static_cast<size_t> (k)] > 0.0)
                {
                    lo = std::min (lo, k);
                    hi = k;
                }
            first[static_cast<size_t> (m)] = lo;
            last[static_cast<size_t> (m)] = hi;
        }

        const long long numIn = static_cast<long long> (x.size());
        const double scale = 1.0 / (windowSum * windowSum);
        std::vector<std::complex<double>> frame (static_cast<size_t> (n));
        std::vector<double> power (static_cast<size_t> (numBins));
        for (int t = 0; t < numFrames; ++t)
        {
            const long long start = static_cast<long long> (t) * hop - n / 2; // 元の信号での位置
            for (int i = 0; i < n; ++i)
            {
                const long long j = start + i;
                const double v = (j >= 0 && j < numIn) ? x[static_cast<size_t> (j)] : 0.0;
                frame[static_cast<size_t> (i)] = { v * window[static_cast<size_t> (i)], 0.0 };
            }
            fftInPlace (frame, twiddles);
            for (int k = 0; k < numBins; ++k)
                power[static_cast<size_t> (k)] = std::norm (frame[static_cast<size_t> (k)]) * scale;
            float* row = out.data.data() + static_cast<size_t> (t) * static_cast<size_t> (config.numMels);
            for (int m = 0; m < config.numMels; ++m)
            {
                double acc = 0.0;
                const double* w = fb.data() + static_cast<size_t> (m) * static_cast<size_t> (numBins);
                for (int k = first[static_cast<size_t> (m)]; k <= last[static_cast<size_t> (m)]; ++k)
                    acc += w[k] * power[static_cast<size_t> (k)];
                row[m] = static_cast<float> (std::log1p (1000.0 * acc));
            }
        }
        return out;
    }

    std::vector<Onset> pickHits (const float* probabilities, int numFrames, int numClasses, const FeatureConfig& config,
                                 const std::vector<float>& thresholds, double minGapSeconds)
    {
        std::vector<Onset> hits;
        const double frameRate = config.frameRate();
        const int gap = std::max (static_cast<int> (minGapSeconds * frameRate), 1);
        for (int c = 0; c < numClasses; ++c)
        {
            const float threshold = static_cast<size_t> (c) < thresholds.size() ? thresholds[static_cast<size_t> (c)] : 0.5f;
            long long lastPicked = -1000000000LL;
            for (int i = 0; i < numFrames; ++i)
            {
                const float p = probabilities[static_cast<size_t> (i) * static_cast<size_t> (numClasses) + static_cast<size_t> (c)];
                if (! (p > threshold))
                    continue;
                // 前後 gap フレームの最大（端は端の値の繰り返し = 範囲を切り詰めるのと同じ）
                bool isLocalMax = true;
                const int lo = std::max (i - gap, 0), hi = std::min (i + gap, numFrames - 1);
                for (int j = lo; j <= hi && isLocalMax; ++j)
                    if (probabilities[static_cast<size_t> (j) * static_cast<size_t> (numClasses) + static_cast<size_t> (c)] > p)
                        isLocalMax = false;
                if (isLocalMax && i - lastPicked >= gap)
                {
                    hits.push_back ({ i / frameRate, c, p });
                    lastPicked = i;
                }
            }
        }
        std::stable_sort (hits.begin(), hits.end(), [] (const Onset& a, const Onset& b) { return a.time < b.time; });
        return hits;
    }
}
