#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include "brain.hpp"

class Coord
{
public:
    Coord(int x0 = 0, int y0 = 0)
    {
        x = x0;
        y = y0;
    }
    int x;
    int y;
};

class Indiv
{
private:
    std::vector<float> forward_pass();
    float sensor_output(int sensor_num);

public:
    int index;
    Coord loc;
    int age;
    Genome genome;
    NNet nnet;
    Coord last_move_dir;
    void construct_nnet();
    void perform_action();
};

uint32_t random_uint();

constexpr int EMPTY = 0;
constexpr int BARRIER = 0xffff;

extern int grid[128][128];
extern std::vector<Coord> barrier_locations;
extern Indiv peeps[3001];

#endif