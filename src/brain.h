#ifndef GENOME_H_INCLUDED
#define GENOME_H_INCLUDED

#include <vector>

constexpr uint8_t SENSOR = 1;
constexpr uint8_t ACTION = 1;
constexpr uint8_t NEURON = 0;

struct Gene
{
    uint16_t sourceType;
    uint16_t sourceNum;
    uint16_t sinkType;
    uint16_t sinkNum;
    int16_t weight;
    float weightAsFloat() const { return weight / 8192.0; };
};

typedef std::vector<Gene> Genome;

struct NeuralNet
{
    std::vector<Gene> connections;

    struct Neuron
    {
        float output;
        bool driven;
    };
    std::vector<Neuron> neurons;
};

constexpr float initialNeuronOutput() { return 0.5; }

Genome generateChildGenome(std::vector<Genome> &parentGenomes);
Gene makeRandomGene();
Genome makeRandomGenome();

#endif