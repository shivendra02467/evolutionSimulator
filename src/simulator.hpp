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

uint32_t random_uint();

constexpr int EMPTY = 0;
constexpr int BARRIER = 0xffff;

extern int grid[128][128];
extern std::vector<Coord> barrier_locations;
extern Indiv peeps[3001];

#endif