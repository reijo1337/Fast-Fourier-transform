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

vector<base> initWtable( int number_of_samples)
{
    vector<base> W;
    for (int z = 0; z < ceil(log2(number_of_samples)); ++z) {
        int len = pow(2, z + 1);
        double ang = 2 * M_PI / len;
        base wlen(cos(ang), sin(ang));
        for (int i = 0; i < number_of_samples; i += len) {
            base w(1);
            for (int j = 0; j < len / 2; ++j) {
                W.push_back(w);
                w *= wlen;
            }
        }
    }

    return W;
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

    vector<base> wTab = initWtable(numNumbers);

    // FFT
    double total = 0;

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
        
        double starttime=MPI_Wtime();
        int offset = 1;
        for (int z = 0; z < numParalleIterations; ++z) {
            for (int i = partSize*rank; i < partSize / 2; i++) {
                int ti = 0;
                base w = wTab[ti];
                base u = polynomial[i], v = polynomial[i + offset] * w;
                polynomial[i] = u + v;
                polynomial[i+offset] = u - v;
            }
            offset *= 2;
        }

        int offsetMax = ceil(pow(2, degreeOfPolynomial-1));
        for (;offset <= offsetMax;) {
            int flag = offset / partSize;
            int rankToSwap;
            if ((rank / flag) % 2 == 0 ) {
                rankToSwap = rank + flag;
                MPI_Send(&polynomial[rank*partSize + partSize / 2], partSize / 2, MPI_C_DOUBLE_COMPLEX, rankToSwap, 1337, MPI_COMM_WORLD);
                MPI_Recv(&polynomial[rankToSwap*partSize], partSize / 2, MPI_C_DOUBLE_COMPLEX, rankToSwap, 1337, MPI_COMM_WORLD, &status);
                for (int i = partSize*rank; i < partSize / 2; i++) {
                    int ti = 0;
                    base w = wTab[ti];
                    base u = polynomial[i], v = polynomial[i + offset] * w;
                    polynomial[i] = u + v;
                    polynomial[i+offset] = u - v;
                }
            } else {
                rankToSwap = rank - flag;
                MPI_Recv(&polynomial[rankToSwap*partSize + partSize / 2], partSize / 2, MPI_C_DOUBLE_COMPLEX, rankToSwap, 1337, MPI_COMM_WORLD, &status);
                MPI_Send(&polynomial[rank*partSize], partSize / 2, MPI_C_DOUBLE_COMPLEX, rankToSwap, 1337, MPI_COMM_WORLD);
                for (int i = partSize*rank + partSize / 2; i < partSize; i++) {
                    int ti = 0;
                    base w = wTab[ti];
                    base u = polynomial[i - offset], v = polynomial[i] * w;
                    polynomial[i-offset] = u + v;
                    polynomial[i] = u - v;
                }
            }
            offset *= 2;
        }

        if (rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Recv(&polynomial[i*partSize], partSize, MPI_C_DOUBLE_COMPLEX, i, 42, MPI_COMM_WORLD, &status);
            }
        } else {
            MPI_Send(&polynomial[rank*partSize], partSize, MPI_C_DOUBLE_COMPLEX, 0, 42, MPI_COMM_WORLD);
        }
        double endtime = MPI_Wtime();
        total += (endtime - starttime) * 1000;
    }
    if (rank == 0) {
        std::cout << "Average time ms for " << numTries << " tries ";
        std::cout << total / (1.0 * numTries) << "ms" << std::endl;
    }
    MPI_Finalize();
    return 0;
}
