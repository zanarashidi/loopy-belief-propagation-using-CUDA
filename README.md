# Loopy Belief Propagation using CUDA

## About

An implementation of the Sum-Product (Loopy Belief Propagation) Algorithm using CUDA on GPU

This code was written as part of my B.Sc. Project in Computer Engineering at the Computer Engineering Department, Sharif University of Technology, Iran

July 2017

## How to use this code:

1. Compile lattice.cpp: 
```bash
$ g++ lattice.cpp -o lattice.out
```
2. Run lattice.out: 
```bash
$ ./lattice.out
```
3. Enter grid dimension: 
```bash
"Enter Grid Dim: "
```
4. Compile sp.cu: 
```bash
$ nvcc sp.cu -o sp.out
```
5. Run sp.out: 
```bash
$ ./sp.out 
```
