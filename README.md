# evolutionSimulator

This project demonstrates simple **Genetic Algorithm (GA)** through evolution of nnet, which is modelled using a directed graph.

## Features

- Obstacle-avoidance and goal-reaching behavior in a 2D grid
- Graph-based brain representation saved per individual per generation
- Outputs:
  - `.ncol` format text and image of sample nnet graphs of each generation
  - A full video of all steps of each generation
  - An image of last step of each generation
- Cross-platform: tested on **Windows (MinGW)** and **Arch Linux BTW**
- Minimal dependencies: just OpenCV and CMake

## Build Instructions

### Linux/macOS

install opencv cmake then
```bash
git clone https://github.com/yourusername/evolutionSimulator.git
cd evolutionSimulator
mkdir build && cd build
cmake ..
make
./evolutionSimulator
```

### Windows
(if using MinGW-w64, OpenCV build with MinGW)

```bash
git clone https://github.com/yourusername/evolutionSimulator.git
cd evolutionSimulator
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make
./evolutionSimulator.exe
```
