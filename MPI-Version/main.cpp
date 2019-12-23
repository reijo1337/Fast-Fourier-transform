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
    double starttime=MPI_Wtime();

    for (int iter = 0; iter < numTries; iter++) {
        MPI_Bcast(&polynomial.front(), polynomial.size(), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

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

        for (int z = 0; z < numParalleIterations; ++z) {
            int len = pow(2, z + 1);
            double ang = 2 * M_PI / len;
            base wlen(cos(ang), sin(ang));
            for (int i = 0; i < n; i += len) {
                base w(1);
                for (int j = 0; j < len / 2; ++j) {
                    int first = i+j, second = i + j + len / 2;
                    if ((partSize*rank <= first < partSize*(rank+1)) && (partSize*rank <= second < partSize*(rank+1))) {
                        base u = polynomial[first], v = polynomial[second] * w;
                        polynomial[first] = u + v;
                        polynomial[second] = u - v;
                    }

                    w *= wlen;
                }
            }
        }

        if (rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Recv(&polynomial[i*partSize], partSize, MPI_C_DOUBLE_COMPLEX, i, 42, MPI_COMM_WORLD, &status);
            }
        } else {
            MPI_Send(&polynomial[rank*partSize], partSize, MPI_C_DOUBLE_COMPLEX, 0, 42, MPI_COMM_WORLD);
        }

        if (rank == 0) {
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
        } 
    }
    double endtime = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Average time ms for " << numTries << " tries ";
        std::cout << 1.0 * (endtime-starttime)*1000 / (1.0 * numTries) << "ms" << std::endl;
    }
    MPI_Finalize();
    return 0;
}
