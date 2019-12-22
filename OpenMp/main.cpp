#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;
typedef complex<double> base;

const int numThreads = 1; //  must be power of 2

int rev(int num, int lg_n)
{
    int res = 0;
    for (int i = 0; i < lg_n; ++i)
        if (num & (1 << i))
            res |= 1 << (lg_n - 1 - i);
    return res;
}

chrono::milliseconds fft(vector<base>& polynomial)
{
    int n = static_cast<int>(polynomial.size());
    int lg_n = 0;
    while ((1 << lg_n) < n)
        ++lg_n;

    for (int i = 0; i < n; ++i)
        if (i < rev(i, lg_n))
            swap(polynomial[i], polynomial[rev(i, lg_n)]);

    int degreeOfPolynomial = ceil(log2(n));
    int numParalleIterations = degreeOfPolynomial - ceil(log2(numThreads));
    auto start = std::chrono::system_clock::now();

#pragma omp parallel shared(polynomial)
#pragma omp for
    for (int z = 0; z < numParalleIterations; ++z) {
        int len = pow(2, z + 1);
        double ang = 2 * M_PI / len;
        base wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            base w(1);
            for (int j = 0; j < len / 2; ++j) {
                base u = polynomial[i + j], v = polynomial[i + j + len / 2] * w;
                polynomial[i + j] = u + v;
                polynomial[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    for (int z = numParalleIterations; z < degreeOfPolynomial; ++z) {
        int len = pow(2, z + 1);
        double ang = 2 * M_PI / len;
        base wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            base w(1);
            for (int j = 0; j < len / 2; ++j) {
                base u = polynomial[i + j], v = polynomial[i + j + len / 2] * w;
                polynomial[i + j] = u + v;
                polynomial[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    auto end = std::chrono::system_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(end - start);
}
int main()
{
    omp_set_num_threads(numThreads);
    chrono::milliseconds elapsed = chrono::milliseconds::zero();
    int numTries = 100;
    int numNumbers = pow(2, 20);
    vector<base> polynomial;
    polynomial.resize(numNumbers);
    for (int i = 0; i < numNumbers; ++i) {
        polynomial.push_back(rand() % 100);
    };
    for (int i = 0; i < numTries; ++i) {
        vector<base> polynomialCopy = polynomial;
        elapsed += fft(polynomialCopy);
        std::cout << "Iteration num " << i << ". Elapsed " << elapsed.count() << std::endl;
    }
    std::cout << "Average time ms for " << numTries << " tries ";
    std::cout << 1.0 * elapsed.count() / (1.0 * numTries) << std::endl;
    // for (auto elem : a) {
    //   std::cout << "result" << elem << std::endl;
    //}
}
