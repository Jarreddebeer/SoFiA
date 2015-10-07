#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

void histogram(double* ary, double* bins, int* histo, int ary_len, int bin_len, int histo_len) {

    double min = bins[0];
    double bin_size = bins[1] - bins[0];

    for (int i = 0; i < ary_len; i++) {
        int bin = ((double)ary[i] - min) / bin_size;
        if (bin >= 0 && bin < histo_len) {
            histo[bin]++;
        }
    }


}
