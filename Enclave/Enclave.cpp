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

void print_mat(float* a, int a_rows, int a_cols) {
    return;
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < a_cols; j++) {
            printf("%f, ", a[a_cols*i + j]);
        }
        // printf("\n");
    }
}

// Row Major
void matrix_mult(float *a, int a_rows, int a_cols, float *b, int b_rows, int b_cols, float *out) {
    //printf("Dims: a=%dx%d, b=%dx%d, out=%dx%d\n", a_rows, a_cols, b_rows, b_cols, a_rows, b_cols);
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            out[b_cols*i + j] = 0.0f;
            for (int k = 0; k < a_cols; k++) {
                out[b_cols*i + j] += a[a_cols*i + k] * b[b_cols*k + j];
            }
            // printf("\t(%d, %d)[%d] = %f\n", i, j, b_cols*i + j, out[b_cols*i + j]);
        }
    }
}

void matrix_add(float *a, int a_rows, int a_cols, float *b, float *out) {
    // printf("ADD\n");
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < a_cols; j++) {
            out[a_cols*i + j] = a[a_cols*i + j] + b[a_cols*i + j];
            // printf("\t(%d,%d) %f = %f + %f\n", i, j, out[a_cols*i + j], a[a_cols*i + j], b[a_cols*i + j]);
        }
    }
}
void matrix_sub(float *a, int a_rows, int a_cols, float *b, float *out) {
    // printf("SUB\n");
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < a_cols; j++) {
            out[a_cols*i + j] = a[a_cols*i + j] - b[a_cols*i + j];
            // printf("\t(%d,%d) %f = %f - %f\n", i, j, out[a_cols*i + j], a[a_cols*i + j], b[a_cols*i + j]);
        }
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

// Computes inp * W, and writes the data into buffer out
// float *w => 2d array of weights
// int *dimW => 2-element array defining the size of w
// float *inp => 2d array representing input vector
// int *dimInp => 2-element array defining the size of inp
// float *out => Output buffer
void ecall_nativeMatMul(float *w, int *dimW, float *inp, int *dimInp, float *out) {
    //printf("nativeMatMul\n");
    // Copy the W array out of untrusted memory
    int w_rows = dimW[0];
    int w_cols = dimW[1];
    float *w_cpy = (float*) malloc(sizeof(float) * w_cols * w_rows);
    memcpy(w_cpy, w, sizeof(float) * w_cols * w_rows);
    //printf("W_copy (%dx%d):\n", w_rows, w_cols);
    print_mat(w_cpy, w_rows, w_cols);
    int inp_rows = dimInp[0];
    int inp_cols = dimInp[1];
    float *inp_cpy = (float*) malloc(sizeof(float) * inp_cols * inp_rows);
    memcpy(inp_cpy, inp, sizeof(float) * inp_cols * inp_rows);
    //printf("Inp_copy (%dx%d):\n", inp_cols, inp_rows);
    print_mat(inp_cpy, inp_rows, inp_cols);

    // Perform matrix multiplication
    float *res = (float*) malloc(sizeof(float) * w_cols * inp_rows);
    matrix_mult(inp, inp_rows, inp_cols, w_cpy, w_rows, w_cols, out);
    //printf("\n");
    //printf("Res (%dx%d):\n", inp_rows, w_cols);
    //print_mat(res, inp_rows, w_cols);
    // Copy the result into the output buffer
    memcpy(out, res, sizeof(float) * w_cols * inp_rows);
    free(res);
}

// Generate an array of random floats r, then compute r * weight
// The results of this operation are stored inside the enclave
// float  *weight => 2d array of Weights
// int *dim => 2-element array defining the size of weight
// int batch => ???
static float *r = nullptr;
static int r_rows = 0;
static int r_cols = 0;
static float *w_pre = nullptr;
static int w_pre_rows = 0;
static int w_pre_cols = 0;
void ecall_precompute(float *weight, int *dim, int batch) {
    //printf("precompute\n");
    // Copy weight out of untrusted memory
    int weight_rows = dim[0];
    int weight_cols = dim[1];
    float *weight_cpy = (float*) malloc(sizeof(float) * weight_cols * weight_rows);
    memcpy(weight_cpy, weight, sizeof(float) * weight_rows * weight_cols);
    //printf("Weight (%dx%d):\n", weight_rows, weight_cols);
    //print_mat(weight_cpy, weight_rows, weight_cols);
    // Generate random numbers in r
    if (r != nullptr) {
        free(r);
    }
    r = (float*) malloc(sizeof(float) * weight_rows * batch);
    r_rows = batch;
    r_cols = weight_rows;
    read_rand(r, sizeof(float) * weight_rows * batch);
    //printf("R (%dx%d):\n", batch, weight_rows);
    //print_mat(r, batch, weight_rows);

    // Perform matrix multiplication
    if (w_pre != nullptr) {
        free(w_pre);
    }
    w_pre = (float*) malloc(sizeof(float) * weight_cols * batch);
    w_pre_rows = batch;
    w_pre_cols = weight_cols;
    matrix_mult(r, batch, weight_rows, weight_cpy, weight_rows, weight_cols, w_pre);
    //printf("W_pre (%dx%d):\n", batch, weight_cols);
    //print_mat(w_pre, batch, weight_cols);
}

// Computes inp + r, where r is a random buffer that was populated
// by ecall_precompute
void ecall_addNoise(float *inp, int *dim, float *out) {
    //printf("addNoise\n");
    // Copy input out of untrusted memory
    int inp_rows = dim[0];
    int inp_cols = dim[1];
    float *inp_cpy = (float*) malloc(sizeof(float) * inp_cols * inp_rows);
    memcpy(inp_cpy, inp, sizeof(float) * inp_cols * inp_rows);
    //printf("Inp_copy (%dx%d):\n", inp_rows, inp_cols);
    //print_mat(inp_cpy, inp_rows, inp_cols);
    
    //printf("R (%dx%d):\n", r_rows, r_cols);
    //print_mat(r, r_rows, r_cols);

    // Perform matrix addition
    float *res = (float*) malloc(sizeof(float) * inp_cols * inp_rows);
    matrix_add(inp, inp_rows, inp_cols, r, res);
    //printf("Res (%dx%d):\n", inp_rows, inp_cols);
    //print_mat(res, inp_rows, inp_cols);
    memcpy(out, res, sizeof(float) * inp_cols * inp_rows);
    free(res);
}

// Computes inp - (r * w). r * w has been precomputed by ecall_precompute

void ecall_removeNoise(float *inp, int *dim, float *out) {
    //printf("removeNoise\n");
    // Copy input out of untrusted memory
    int inp_rows = dim[0];
    int inp_cols = dim[1];
    float *inp_cpy = (float*) malloc(sizeof(float) * inp_cols * inp_rows);
    memcpy(inp_cpy, inp, sizeof(float) * inp_cols * inp_rows);
    //printf("Inp_copy (%dx%d):\n", inp_rows, inp_cols);
    //print_mat(inp_cpy, inp_rows, inp_cols);

    //printf("W_pre (%dx%d):\n", w_pre_rows, w_pre_cols);
    //print_mat(w_pre, w_pre_rows, w_pre_cols);

    // Perform matrix substraction
    float *res = (float*) malloc(sizeof(float) * inp_cols * inp_rows);
    matrix_sub(inp, inp_rows, inp_cols, w_pre, res);
    //printf("Res (%dx%d):\n", inp_rows, inp_cols);
    //print_mat(res, inp_rows, inp_cols);
    memcpy(out, res, sizeof(float) * inp_cols * inp_rows);
    free(res);
}