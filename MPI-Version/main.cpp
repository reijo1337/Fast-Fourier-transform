#include <cmath>
#include <complex>
#include <iostream>
#include <vector>
#include "mpi.h"

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
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank,size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << rank << std::endl;
    int numNumbers = pow(2, 2);
    int numTries = 100;
    vector<base> polynomial;
    
    if (rank == 0) {
        for (int i = 0; i < numNumbers; ++i) {
            polynomial.push_back(rand() % 100);
        }
    } else {
        polynomial.resize(numNumbers);
    }
    std::cout << polynomial.size() * sizeof(decltype(polynomial)::value_type) << std::endl;
    MPI_Bcast(&polynomial.front(), polynomial.size() * sizeof(decltype(polynomial)::value_type), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    for (auto elem : polynomial) {
        std::cout << "rank: " << rank << " result: " << elem << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
