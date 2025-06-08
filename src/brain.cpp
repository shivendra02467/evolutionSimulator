#include <map>
#include "simulator.h"

struct Node
{
    uint16_t remappedNumber;
    uint16_t numOutputs;
    uint16_t numSelfInputs;
    uint16_t numInputsFromSensorsOrOtherNeurons;
};

typedef std::map<uint16_t, Node> NodeMap;

typedef std::vector<Gene> ConnectionList;

Gene makeRandomGene()
{
    Gene gene;

    gene.sourceType = randomUint() & 1;
    gene.sourceNum = (uint16_t)(randomUint() % 0x8000);
    gene.sinkType = randomUint() & 1;
    gene.sinkNum = (uint16_t)(randomUint() % 0x8000);
    gene.weight = (randomUint() % 0x10000) - 0x8000;

    return gene;
}

Genome makeRandomGenome()
{
    Genome genome;
    for (unsigned n = 0; n < 24; ++n)
    {
        genome.push_back(makeRandomGene());
    }

    return genome;
}

void makeRenumberedConnectionList(ConnectionList &connectionList, const Genome &genome)
{
    connectionList.clear();
    for (auto const &gene : genome)
    {
        connectionList.push_back(gene);
        auto &conn = connectionList.back();

        if (conn.sourceType == NEURON)
        {
            conn.sourceNum %= 5;
        }
        else
        {
            conn.sourceNum %= 9;
        }

        if (conn.sinkType == NEURON)
        {
            conn.sinkNum %= 5;
        }
        else
        {
            conn.sinkNum %= 9;
        }
    }
}

void makeNodeList(NodeMap &nodeMap, const ConnectionList &connectionList)
{
    nodeMap.clear();

    for (const Gene &conn : connectionList)
    {
        if (conn.sinkType == NEURON)
        {
            auto it = nodeMap.find(conn.sinkNum);
            if (it == nodeMap.end())
            {
                nodeMap.insert(std::pair<uint16_t, Node>(conn.sinkNum, {}));
                it = nodeMap.find(conn.sinkNum);
                it->second.numOutputs = 0;
                it->second.numSelfInputs = 0;
                it->second.numInputsFromSensorsOrOtherNeurons = 0;
            }

            if (conn.sourceType == NEURON && (conn.sourceNum == conn.sinkNum))
            {
                ++(it->second.numSelfInputs);
            }
            else
            {
                ++(it->second.numInputsFromSensorsOrOtherNeurons);
            }
        }
        if (conn.sourceType == NEURON)
        {
            auto it = nodeMap.find(conn.sourceNum);
            if (it == nodeMap.end())
            {
                nodeMap.insert(std::pair<uint16_t, Node>(conn.sourceNum, {}));
                it = nodeMap.find(conn.sourceNum);
                it->second.numOutputs = 0;
                it->second.numSelfInputs = 0;
                it->second.numInputsFromSensorsOrOtherNeurons = 0;
            }
            ++(it->second.numOutputs);
        }
    }
}

void removeConnectionsToNeuron(ConnectionList &connections, NodeMap &nodeMap, uint16_t neuronNumber)
{
    for (auto itConn = connections.begin(); itConn != connections.end();)
    {
        if (itConn->sinkType == NEURON && itConn->sinkNum == neuronNumber)
        {
            if (itConn->sourceType == NEURON)
            {
                --(nodeMap[itConn->sourceNum].numOutputs);
            }
            itConn = connections.erase(itConn);
        }
        else
        {
            ++itConn;
        }
    }
}

void cullUselessNeurons(ConnectionList &connections, NodeMap &nodeMap)
{
    bool allDone = false;
    while (!allDone)
    {
        allDone = true;
        for (auto itNeuron = nodeMap.begin(); itNeuron != nodeMap.end();)
        {
            if (itNeuron->second.numOutputs == itNeuron->second.numSelfInputs)
            {
                allDone = false;
                removeConnectionsToNeuron(connections, nodeMap, itNeuron->first);
                itNeuron = nodeMap.erase(itNeuron);
            }
            else
            {
                ++itNeuron;
            }
        }
    }
}

void Indiv::createWiringFromGenome()
{
    NodeMap nodeMap;
    ConnectionList connectionList;

    makeRenumberedConnectionList(connectionList, genome);

    makeNodeList(nodeMap, connectionList);

    cullUselessNeurons(connectionList, nodeMap);

    uint16_t newNumber = 0;
    for (auto &node : nodeMap)
    {
        node.second.remappedNumber = newNumber++;
    }

    nnet.connections.clear();

    for (auto const &conn : connectionList)
    {
        if (conn.sinkType == NEURON)
        {
            nnet.connections.push_back(conn);
            auto &newConn = nnet.connections.back();
            newConn.sinkNum = nodeMap[newConn.sinkNum].remappedNumber;
            if (newConn.sourceType == NEURON)
            {
                newConn.sourceNum = nodeMap[newConn.sourceNum].remappedNumber;
            }
        }
    }

    for (auto const &conn : connectionList)
    {
        if (conn.sinkType == ACTION)
        {
            nnet.connections.push_back(conn);
            auto &newConn = nnet.connections.back();
            if (newConn.sourceType == NEURON)
            {
                newConn.sourceNum = nodeMap[newConn.sourceNum].remappedNumber;
            }
        }
    }

    nnet.neurons.clear();
    for (unsigned neuronNum = 0; neuronNum < nodeMap.size(); ++neuronNum)
    {
        nnet.neurons.push_back({});
        nnet.neurons.back().output = initialNeuronOutput();
        nnet.neurons.back().driven = (nodeMap[neuronNum].numInputsFromSensorsOrOtherNeurons != 0);
    }
}

void randomBitFlip(Genome &genome)
{
    unsigned Index = randomUint() % genome.size();
    uint64_t Bits = (genome[Index].weight << 32) | (genome[Index].sourceType << 31) | (genome[Index].sourceNum << 16) | (genome[Index].sinkType << 15) | (genome[Index].sinkNum);
    unsigned bitIndex = randomUint() % 48;
    Bits ^= 1 << bitIndex;
    genome[Index].sinkNum = Bits & 0x8000;
    Bits = Bits >> 15;
    genome[Index].sinkType = Bits & 1;
    Bits = Bits >> 1;
    genome[Index].sourceNum = Bits & 0x8000;
    Bits = Bits >> 15;
    genome[Index].sourceType = Bits & 1;
    Bits = Bits >> 1;
    genome[Index].weight = (Bits & 0x10000) - 0x8000;
}

void applyPointMutations(Genome &genome)
{
    unsigned numberOfGenes = genome.size();
    while (numberOfGenes-- > 0)
    {
        if ((randomUint() / (float)0xffffffff) < 0.001)
        {
            randomBitFlip(genome);
        }
    }
}

Genome generateChildGenome(std::vector<Genome> &parentGenomes)
{
    Genome genome;
    uint16_t parentIdx;
    parentIdx = randomUint() % parentGenomes.size();
    const Genome &genome_ = parentGenomes[parentIdx];
    genome = genome_;
    applyPointMutations(genome);
    return genome;
}
