#include <iostream>
#include <string.h>
#include <time.h>
#include "kuramoto.h"

/*
g++ -Wall -O2 -std=c++11 -o main.out main.cpp kuramoto.o
*/


int main(int argc, char **argv){
    double tmax = 100;
    int N = 100;
    double K = 2;
    char fname[] = "./test.dat";

    for (int n=0; n<argc; n++){
        if (strcmp(argv[n], "-tmax") == 0){
            tmax = atof(argv[n+1]);
        }

        if (strcmp(argv[n], "-n") == 0){
            N = atoi(argv[n+1]);
        }

        if (strcmp(argv[n], "-k") == 0){
            K = atof(argv[n+1]);
        }

        if (strcmp(argv[n], "-f") == 0){
            sprintf(fname, "%s", argv[n+1]);
        }

    }

    Kuramoto obj_k(N, K, time(NULL));
    obj_k.run(tmax, fname);
}