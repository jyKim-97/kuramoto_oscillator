#include "kuramoto.h"
#include <time.h>
#include <string.h>

#define _PI 3.14159265359
#define get_rand() ((double) rand() / (RAND_MAX + 1))


Kuramoto::Kuramoto(int _N, double _K, int seed){
    N = _N;
    K = _K;
    w = (double*) malloc(sizeof(double) * N);
    theta = (double*) malloc(sizeof(double) * N);
    theta0 = (double*) malloc(sizeof(double) * N);
    tmp_dsin = (double**) malloc(sizeof(double) * N);
    for (int n=0; n<N; n++){
        tmp_dsin[n] = (double*) malloc(sizeof(double) * N);
    }
    if (seed == -1){
        srand(time(NULL));
    } else {
        srand(seed);
    }
    init();
}


void Kuramoto::init(void){
    for (int n=0; n<N; n++){
        w[n] = get_rand()*2 - 1;
        theta[n] = get_rand() * _PI * 2;
        // printf("%f-%f\n", w[n], theta[n]);
    }
}


void Kuramoto::update(void){
    copy(N, theta0, theta);
    double *d1 = get_dsin_arr();
    apply_dsin(d1, 0.5);
    double *d2 = get_dsin_arr();
    apply_dsin(d2, 0.5);
    double *d3 = get_dsin_arr();
    apply_dsin(d3, 1);
    double *d4 = get_dsin_arr();
    for (int n=0; n<N; n++){
        theta[n] = theta0[n] + (d1[n] + 2*d2[n] + 2*d3[n] + d4[n])/6.;
    }

    free(d1);
    free(d2);
    free(d3);
    free(d4);
}


void Kuramoto::run(double tmax, char fname[]){
    int nmax = tmax / _dt;

    FILE *fp = fopen(fname, "wb");

    // FILE *fp = fopen(fname, "r");
    // if (fp == NULL){
    //     fp = fopen(fname, "wb");
    // } else {
    //     printf("File exist\n");
    //     exit(1);
    // }

    float info[4] = {(float) N, (float) K, (float) _dt, (float) tmax};
    fwrite(info, sizeof(float), 4, fp);
    for (int n=0; n<nmax; n++){
        update();
        print_arr(fp, N, theta);
    }
    fclose(fp);
}


double* Kuramoto::get_dsin_arr(void){
    double *dtheta = (double*) malloc(sizeof(double*) * N);
    for (int n=0; n<N; n++){
        double d = get_dsin(n);
        dtheta[n] = d*_dt;
    }
    return dtheta;
}


double Kuramoto::get_dsin(int nid){
    double dsin = 0;
    for (int i=0; i<N; i++){
        if (i < nid){
            dsin -= tmp_dsin[nid][i];
        } else {
            double val = sin(theta[i]-theta[nid]);
            tmp_dsin[i][nid] = val;
            dsin += val;
        }
    }
    dsin *= K/N;
    dsin += w[nid];
    return dsin;
}


void Kuramoto::apply_dsin(double *dsin, double factor){
    for (int n=0; n<N; n++){
        theta[n] = theta0[n] + factor * dsin[n];
    }
}


void copy(int N, double *dst, double *src){
    memcpy(dst, src, sizeof(double)*N);
}


void print_arr(FILE *fp, int N, double *var){
    float *var_f = (float*) malloc(sizeof(float) * N);
    for (int n=0; n<N; n++) var_f[n] = (float) var[n];
    fwrite(var_f, sizeof(float), N, fp);
    delete[] var_f;
}
