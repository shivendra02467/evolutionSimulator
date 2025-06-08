#ifndef SIMULATOR_H_INCLUDED
#define SIMULATOR_H_INCLUDED

#include "brain.h"

enum class Compass : uint8_t
{
    E = 0,
    SE,
    S,
    SW,
    W,
    NW,
    N,
    NE,
    CENTER
};

struct Dir;
struct Coord;

struct Dir
{
    Coord asCoord();
    Compass dir_val;
};
struct Coord
{
    Coord(int16_t x0 = 0, int16_t y0 = 0)
    {
        x = x0;
        y = y0;
    }
    Dir asDir();
    int16_t x;
    int16_t y;
};

uint32_t randomUint();

const uint16_t EMPTY = 0;
const uint16_t BARRIER = 0xffff;

class Grid
{
public:
    struct Column
    {
        Column(uint16_t numRows) : data{std::vector<uint16_t>(numRows, 0)} {}
        void zeroFill() { std::fill(data.begin(), data.end(), 0); }
        uint16_t &operator[](uint16_t rowNum) { return data[rowNum]; }
        uint16_t operator[](uint16_t rowNum) const { return data[rowNum]; }
        size_t size() const { return data.size(); }

    private:
        std::vector<uint16_t> data;
    };
    void init(uint16_t sizeX, uint16_t sizeY)
    {
        auto col = Column(sizeY);
        data = std::vector<Column>(sizeX, col);
    }
    void zeroFill()
    {
        for (Column &column : data)
            column.zeroFill();
    }
    uint16_t sizeX() const { return data.size(); }
    uint16_t sizeY() const { return data[0].size(); }
    bool isInBounds(Coord loc) const { return loc.x >= 0 && loc.x < sizeX() && loc.y >= 0 && loc.y < sizeY(); }
    bool isEmptyAt(Coord loc) const { return at(loc) == EMPTY; }
    bool isBarrierAt(Coord loc) const { return at(loc) == BARRIER; }
    bool isOccupiedAt(Coord loc) const { return at(loc) != EMPTY && at(loc) != BARRIER; }
    bool isBorder(Coord loc) const { return loc.x == 0 || loc.x == sizeX() - 1 || loc.y == 0 || loc.y == sizeY() - 1; }
    uint16_t at(Coord loc) const { return data[loc.x][loc.y]; }
    uint16_t at(uint16_t x, uint16_t y) const { return data[x][y]; }
    void set(Coord loc, uint16_t val) { data[loc.x][loc.y] = val; }
    void set(uint16_t x, uint16_t y, uint16_t val) { data[x][y] = val; }
    Coord findEmptyLocation();
    void createBarrier();
    const std::vector<Coord> &getBarrierLocations() const { return barrierLocations; }
    Column &operator[](uint16_t columnXNum) { return data[columnXNum]; }
    const Column &operator[](uint16_t columnXNum) const { return data[columnXNum]; }

private:
    std::vector<Column> data;
    std::vector<Coord> barrierLocations;
};

struct Indiv;
extern Grid grid;

class Peeps
{
public:
    Peeps(){};
    void init(unsigned population)
    {
        individuals.resize(population + 1);
    }
    Indiv &getIndiv(Coord loc) { return individuals[grid.at(loc)]; }
    const Indiv &getIndiv(Coord loc) const { return individuals[grid.at(loc)]; }

    Indiv &operator[](uint16_t index) { return individuals[index]; }
    Indiv const &operator[](uint16_t index) const { return individuals[index]; }

private:
    std::vector<Indiv> individuals;
};

struct Indiv
{
    void createWiringFromGenome();
    void initialize(uint16_t index_, Coord loc_, Genome &&genome_)
    {
        index = index_;
        loc = loc_;
        age = 0;
        grid.set(loc_, index_);
        lastMoveDir.dir_val = (Compass)(randomUint() % 8);
        genome = std::move(genome_);
        createWiringFromGenome();
    }
    uint16_t index;
    Coord loc;
    unsigned age;
    Genome genome;
    NeuralNet nnet;
    Dir lastMoveDir;
    std::vector<float> feedForward();
    float getSensor(uint16_t sensorNum);
};

extern Peeps peeps;

#endif