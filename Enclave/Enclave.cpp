#include "Enclave_t.h" /* print_string */

#include <stdio.h> /* vsnprintf */
#include <string.h>
#include <algorithm>    // std::max
#include <sgx_trts.h>


int printf(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

void read_rand(float *r, int totalSize) {
    for (int i = 0; i < totalSize; i++) {
        r[i] = (float) i;
    }
}


// the actual buffer of *inp is in untrusted memory
// You can read from it, but never write to it
int ecall_compute_secrete_operation(int *inp, int size) {
    // decrypt inp
    // ....

    int res = 0;

    for (int i = 0; i < size; i++) {
        res += inp[i];
    }

    // encrypt res
    // ....

    printf("Returning to App.cpp\n");
    return res;
}

// Computes w * inp, and writes the data into buffer out
// float *w => 2d array of weights
// int *dimW => 2-element array defining the size of w
// float *inp => 2d array representing input vector
// int *dimInp => 2-element array defining the size of inp
// float *out => Output buffer
void ecall_nativeMatMul(float *w, int *dimW, float *inp, int *dimInp, float *out) {
    printf("nativeMatMul\n");
    // Copy the W array out of untrusted memory
    int w_rows = dimW[0];
    int w_cols = dimW[1];
    float *w_cpy = (float*) malloc(sizeof(float) * w_rows * w_cols);
    memcpy(w_cpy, w, sizeof(float) * w_rows * w_cols);
    int inp_rows = dimInp[0];
    int inp_cols = dimInp[1];
    float *inp_cpy = (float*) malloc(sizeof(float) * inp_rows * inp_cols);
    memcpy(inp_cpy, inp, sizeof(float) * inp_rows * inp_cols);

    // Perform matrix multiplication
    float *res = (float*) malloc(sizeof(float) * w_rows * inp_cols);
    printf("Dims: w=%dx%d, i=%dx%d, r=%dx%d\n", w_rows, w_cols, inp_rows, inp_cols, w_rows, inp_cols);
    for (int i = 0; i < w_rows; i++) {
        for (int j = 0; j < inp_cols; j++) {
            for (int k = 0; k < w_cols; k++) {
                // printf("%d = %d * %d\n", i * inp_cols + j, i * w_cols + k, k * inp_cols + j);
                res[i*inp_cols + j] += w_cpy[i*w_cols + k] * inp_cpy[k*inp_cols + j];
            }
        }
    }
    // Copy the result into the output buffer
    memcpy(out, res, sizeof(float) * w_rows * inp_cols);
    free(res);
}

// Generate an array of random floats r, then compute r * weight
// The results of this operation are stored inside the enclave
// float  *weight => 2d array of Weights
// int *dim => 2-element array defining the size of weight
// int batch => ???
float *r = nullptr;
float *w_pre = nullptr;
void ecall_precompute(float *weight, int *dim, int batch) {
    printf("precompute\n");
    // Copy weight out of untrusted memory
    int weight_rows = dim[0];
    int weight_cols = dim[1];
    float *weight_cpy = (float*) malloc(sizeof(float) * weight_rows * weight_cols);
    printf("1\n");
    memcpy(weight_cpy, weight, sizeof(float) * weight_rows * weight_cols);
    printf("1.1\n");
    // Generate random numbers in r
    if (r != nullptr) {
        free(r);
    }
    r = (float*) malloc(sizeof(float) * batch * weight_rows);
    printf("2\n");
    read_rand(r, sizeof(float) * batch * weight_rows);
    printf("3\n");

    // Perform matrix multiplication
    if (w_pre != nullptr) {
        free(w_pre);
    }
    printf("4\n");
    w_pre = (float*) malloc(sizeof(float) * batch * weight_cols);
    printf("5\n");
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < weight_cols; j++) {
            for (int k = 0; k < weight_rows; k++) {
                w_pre[i*weight_cols + j] += r[i*weight_rows + k] * weight_cpy[k*weight_cols + j];
            }
        }
    }
    printf("6\n");
    // TODO: Need to store res in enclave somehow
    // Maybe done by global pointers?
}

// Computes inp + r, where r is a random buffer that was populated
// by ecall_precompute
void ecall_addNoise(float *inp, int *dim, float *out) {
    printf("addNoise\n");
    printf("%x", dim);
    // Copy input out of untrusted memory
    int inp_rows = dim[0];
    printf("1.1\n");
    int inp_cols = dim[1];
    printf("1\n");
    float *inp_cpy = (float*) malloc(sizeof(float) * inp_rows * inp_cols);
    printf("2\n");
    memcpy(inp_cpy, inp, sizeof(float) * inp_rows * inp_cols);
    printf("3\n");

    // Perform matrix addition
    float *res = (float*) malloc(sizeof(float) * inp_rows * inp_cols);
    printf("4\n");
    for (int i = 0; i < inp_rows; i++) {
        for (int j = 0; j < inp_cols; j++) {
            res[i*inp_cols + j] = inp_cpy[i*inp_cols + j] + r[i*inp_cols + j];
        }
    }
    printf("5\n");
    memcpy(out, res, sizeof(float) * inp_rows * inp_cols);
    printf("6\n");
    free(res);
    printf("7\n");
}

// Computes inp - (r * w). r * w has been precomputed by ecall_precompute

void ecall_removeNoise(float *inp, int *dim, float *out) {
    printf("removeNoise\n");
    // Copy input out of untrusted memory
    int inp_rows = dim[0];
    int inp_cols = dim[1];
    float *inp_cpy = (float*) malloc(sizeof(float) * inp_rows * inp_cols);
    memcpy(inp_cpy, inp, sizeof(float) * inp_rows * inp_cols);

    // Perform matrix substraction
    float *res = (float*) malloc(sizeof(float) * inp_rows * inp_cols);
    for (int i = 0; i < inp_rows; i++) {
        for (int j = 0; j < inp_cols; j++) {
            res[i*inp_cols + j] = inp_cpy[i*inp_cols + j] - w_pre[i*inp_cols * j];
        }
    }
    memcpy(out, res, sizeof(float) * inp_rows * inp_cols);
    free(res);
}