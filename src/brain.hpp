#ifndef BRAIN_HPP
#define BRAIN_HPP

#include <vector>
#include <cstdint>

constexpr int SENSOR = 1;
constexpr int ACTION = 1;
constexpr int RELAY = 0;

typedef std::vector<uint16_t> Genome;

class Node
{
public:
    float output;
    bool driven;
};

class Edge
{
public:
    int source_type;
    int source_num;
    int sink_type;
    int sink_num;
    int weight;
};

class NNet
{
public:
    std::vector<Node> nodes;
    std::vector<Edge> edges;
};

Genome generate_child_genome(std::vector<Genome> &parent_genomes);
Genome make_random_genome();

#endif