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

	MPI_Status status;

    int numNumbers = pow(2, 20);
    int numTries = 100;
    vector<base> polynomial, original;

    if (rank == 0) {
        for (int i = 0; i < numNumbers; ++i) {
            polynomial.push_back(rand() % 100);
        }
    } else {
        polynomial.resize(numNumbers);
    }


    // FFT
    double total = 0;
    double totalBCast = 0;
    double totalRecv = 0;
    double totalLast = 0;

    for (int iter = 0; iter < numTries; iter++) {
        double startBCast = MPI_Wtime();
        MPI_Bcast(&polynomial.front(), polynomial.size(), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
        double endBCast = MPI_Wtime();
        totalBCast += (endBCast - startBCast) * 1000;

        int n = static_cast<int>(polynomial.size());
        int lg_n = 0;
        while ((1 << lg_n) < n)
            ++lg_n;
        if (rank == 0) {
            for (int i = 0; i < n; ++i)
                if (i < rev(i, lg_n))
                    swap(polynomial[i], polynomial[rev(i, lg_n)]);
        }

        int degreeOfPolynomial = ceil(log2(n));
        int numParalleIterations = degreeOfPolynomial - ceil(log2(size));
        int partSize = n / size;
        
        double starttime=MPI_Wtime();
        for (int z = 0; z < numParalleIterations; ++z) {
            int len = pow(2, z + 1);
            double ang = 2 * M_PI / len;
            base wlen(cos(ang), sin(ang));
            base w(1);
            for (int offset = 1; offset < partSize; offset *= 2) {
                for (int i = partSize*rank; i < partSize / 2; i++) {
                    base u = polynomial[i], v = polynomial[i + offset] * w;
                    polynomial[i] = u + v;
                    polynomial[i+offset] = u - v;
                }
            }
        }

        if (rank == 0) {
            double startRecv = MPI_Wtime();
            for (int i = 1; i < size; i++) {
                MPI_Recv(&polynomial[i*partSize], partSize, MPI_C_DOUBLE_COMPLEX, i, 42, MPI_COMM_WORLD, &status);
            }
            double endRecv = MPI_Wtime();
            totalRecv += (endRecv - startRecv) * 1000;
        } else {
            MPI_Send(&polynomial[rank*partSize], partSize, MPI_C_DOUBLE_COMPLEX, 0, 42, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            double startLast = MPI_Wtime();
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
            double endLast = MPI_Wtime();
            totalLast += (endLast - startLast) * 1000;
        } 
        double endtime = MPI_Wtime();
        total += (endtime - starttime) * 1000;
    }
    if (rank == 0) {
        std::cout << "Average time ms for " << numTries << " tries ";
        std::cout << total / (1.0 * numTries) << "ms" << std::endl;
        std::cout << "Average BCast: " << totalBCast / (1.0 * numTries) << "ms" << std::endl;
        std::cout << "Average Recv: " << totalRecv / (1.0 * numTries) << "ms" << std::endl;
        std::cout << "Average Last: " << totalLast / (1.0 * numTries) << "ms" << endl;
    }
    MPI_Finalize();
    return 0;
}
