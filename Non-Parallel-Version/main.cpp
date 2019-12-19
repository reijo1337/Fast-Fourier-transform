#include <cmath>
#include <complex>
#include <iostream>
#include <vector>
using namespace std;
typedef complex<double> base;

int rev(int num, int lg_n)
{
    int res = 0;
    for (int i = 0; i < lg_n; ++i)
        if (num & (1 << i))
            res |= 1 << (lg_n - 1 - i);
    return res;
}

void fft(vector<base>& a, bool invert)
{
    int n = (int)a.size();
    int lg_n = 0;
    while ((1 << lg_n) < n)
        ++lg_n;

    for (int i = 0; i < n; ++i)
        if (i < rev(i, lg_n))
            swap(a[i], a[rev(i, lg_n)]);

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        base wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            base w(1);
            //#pragma omp parallel shared(a) private(j, w)
            //#pragma omp for
            for (int j = 0; j < len / 2; ++j) {
                base u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert)
        for (int i = 0; i < n; ++i)
            a[i] /= n;
}
int main()
{
    vector<base> a{ 1, 2, 6, 24, 120 };
    fft(a, false);
    for (auto elem : a) {
        std::cout << "result" << elem << std::endl;
    }
}
