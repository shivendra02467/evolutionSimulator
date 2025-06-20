# evolutionSimulator

This project demonstrates simple **Genetic Algorithm (GA)** through evolution of nnet, which is modelled using a directed graph.

## Features

- Outputs:
  - `.ncol` format text and image of sample nnet graphs of each generation
  - A full video of all steps of each generation
  - An image of last step of each generation\
- Minimal dependencies: just OpenCV and CMake
- CMake as build system
- OpenMP for multithreading

## Build Instructions

### Linux/macOS

install opencv cmake then
```bash
git clone https://github.com/shivendra02467/evolutionSimulator.git
cd evolutionSimulator
mkdir build && cd build
cmake ..
make
./evolutionSimulator
```

### Windows
(if using MinGW-w64, OpenCV build with MinGW)

```bash
git clone https://github.com/shivendra02467/evolutionSimulator.git
cd evolutionSimulator
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make
./evolutionSimulator.exe
```
