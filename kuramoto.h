#include <iostream>
#include <cmath>
#include <stdio.h>


/*
g++ -Wall -o2 -std=c++11 -c kuramoto.cpp
*/


class Kuramoto{
    public:
        int N;
        double *w, *theta, *theta0, K;
        double **tmp_dsin;
        double _dt = 0.01;
    
    Kuramoto(int _N, double _K, int seed);
    void init(void);
    void update(void);
    double* get_dsin_arr(void);
    double get_dsin(int nid);
    void apply_dsin(double *dsin, double factor=1);
    void run(double tmax, char fname[]);
};


void copy(int N, double *dst, double *src);
void print_arr(FILE *fp, int N, double *var);