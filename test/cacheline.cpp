#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define ARRAY_SIZE (1024 * 1024 * 8)

int main() {
    char *array = (char *)malloc(ARRAY_SIZE);
    struct timespec start, end;

    for (int stride = 4; stride <= 1024; stride *= 2) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int i = 0; i < ARRAY_SIZE; i += stride) {
            array[i]++;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time = (end.tv_sec - start.tv_sec) * 1e9 +
                      (end.tv_nsec - start.tv_nsec);
        printf("Stride: %d, Time: %f ns\n", stride, time);
    }

    free(array);
    return 0;
}
