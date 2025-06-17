#include <map>
#include <cmath>
#include "brain.hpp"
#include "simulator.hpp"

float Indiv::sensor_output(int sensor_num)
{
    float output = 0.0f;
    switch (sensor_num)
    {
    case 0:
    {
        output = age / 300.0f;
        break;
    }
    case 1:
    {
        output = loc.x / 127.0f;
        break;
    }
    case 2:
    {
        output = loc.y / 127.0f;
        break;
    }
    case 3:
    {
        int last_x = last_move_dir.x;
        output = last_x == 0 ? 0.5f : (last_x == -1 ? 0.0f : 1.0f);
        break;
    }
    case 4:
    {
        int last_y = last_move_dir.y;
        output = last_y == 0 ? 0.5f : (last_y == -1 ? 0.0f : 1.0f);
        break;
    }
    case 5:
    {
        int min_dist_x = std::min(loc.x, (127 - loc.x));
        output = min_dist_x / (128 / 2.0f);
        break;
    }
    case 6:
    {
        int min_dist_y = std::min(loc.y, (127 - loc.y));
        output = min_dist_y / (128 / 2.0f);
        break;
    }
    case 7:
    {
        int count = 0;
        Coord test_loc(loc.x + last_move_dir.x, loc.y + last_move_dir.y);
        int num_locs = 32;
        while (num_locs > 0 && test_loc.x < 128 && test_loc.x >= 0 && test_loc.y < 128 && test_loc.y >= 0 && grid[test_loc.x][test_loc.y] == EMPTY)
        {
            ++count;
            test_loc.x = test_loc.x + last_move_dir.x;
            test_loc.y = test_loc.y + last_move_dir.y;
            --num_locs;
        }
        if (num_locs > 0 && (!(test_loc.x < 128 && test_loc.x >= 0 && test_loc.y < 128 && test_loc.y >= 0) || grid[test_loc.x][test_loc.y] == BARRIER))
            output = 1.0f;
        else
            output = count / 32.0f;
        break;
    }
    }
    return output;
}

std::vector<float> Indiv::forward_pass()
{
    std::vector<float> action_output(8, 0.0);
    std::vector<float> relay_accumulators(nnet.nodes.size(), 0.0);
    bool relay_computed = false;
    for (Edge &edge : nnet.edges)
    {
        if (edge.sink_type == ACTION && !relay_computed)
        {
            for (int index = 0; index < nnet.nodes.size(); ++index)
                if (nnet.nodes[index].driven)
                    nnet.nodes[index].output = std::tanh(relay_accumulators[index]);
            relay_computed = true;
        }
        float input_val;
        if (edge.source_type == SENSOR)
            input_val = sensor_output(edge.source_num);
        else
            input_val = nnet.nodes[edge.source_num % nnet.nodes.size()].output;
        if (edge.sink_type == ACTION)
            action_output[edge.sink_num] += input_val * edge.weight / 128.0f;
        else
            relay_accumulators[edge.sink_num] += input_val * edge.weight / 128.0f;
    }
    return action_output;
}

void Indiv::perform_action()
{
    std::vector<float> action_output = forward_pass();
    float move_x = (random_uint() % 3 - 1) * action_output[0] + action_output[1] - action_output[2] - last_move_dir.x * action_output[5] + last_move_dir.y * action_output[6] - last_move_dir.y * action_output[7];
    float move_y = (random_uint() % 3 - 1) * action_output[0] + action_output[3] - action_output[4] - last_move_dir.y * action_output[5] - last_move_dir.x * action_output[6] + last_move_dir.x * action_output[7];
    move_x = std::tanh(move_x);
    move_y = std::tanh(move_y);
    int prob_x = ((random_uint() / (float)0xffffffff) < (std::abs(move_x)));
    int prob_y = ((random_uint() / (float)0xffffffff) < (std::abs(move_x)));
    int signum_x = move_x < 0.0 ? -1 : 1;
    int signum_y = move_y < 0.0 ? -1 : 1;
    Coord movement_offset(prob_x * signum_x, prob_y * signum_y);
    Coord new_loc(loc.x + movement_offset.x, loc.y + movement_offset.y);
    if (new_loc.x < 128 && new_loc.x >= 0 && new_loc.y < 128 && new_loc.y >= 0 && grid[new_loc.x][new_loc.y] == EMPTY)
    {
        if (grid[new_loc.x][new_loc.y] == EMPTY)
        {
            grid[loc.x][loc.y] = EMPTY;
            grid[new_loc.x][new_loc.y] = index;
            loc = new_loc;
            last_move_dir = Coord(new_loc.x - loc.x, new_loc.y - loc.y);
        }
    }
}

Genome make_random_genome()
{
    Genome genome;
    for (int n = 0; n < 24; ++n)
    {
        uint16_t gene = random_uint() % 0x10000;
        genome.push_back(gene);
    }
    return genome;
}

void Indiv::construct_nnet()
{
    std::vector<Edge> edges;
    for (const uint16_t &gene : genome)
    {
        Edge edge;
        edge.source_type = (gene & 1);
        edge.source_num = ((gene >> 1) & 7);
        edge.sink_type = ((gene >> 4) & 1);
        edge.sink_num = ((gene >> 5) & 7);
        edge.weight = (int8_t)(gene >> 8);
        edges.push_back(edge);
    }

    struct Relay
    {
        int new_number;
        int outputs;
        int self_inputs;
        int other_inputs;
    };

    std::map<int, Relay> relays;
    for (const Edge &edge : edges)
    {
        if (edge.sink_type == RELAY)
        {
            auto it = relays.find(edge.sink_num);
            if (it == relays.end())
            {
                relays.insert(std::pair<int, Relay>(edge.sink_num, Relay()));
                it = relays.find(edge.sink_num);
                it->second.outputs = 0;
                it->second.self_inputs = 0;
                it->second.other_inputs = 0;
            }
            if (edge.source_type == RELAY && edge.source_num == edge.sink_num)
                ++(it->second.self_inputs);
            else
                ++(it->second.other_inputs);
        }
        if (edge.source_type == RELAY)
        {
            auto it = relays.find(edge.source_num);
            if (it == relays.end())
            {
                relays.insert(std::pair<int, Relay>(edge.source_num, Relay()));
                it = relays.find(edge.source_num);
                it->second.outputs = 0;
                it->second.self_inputs = 0;
                it->second.other_inputs = 0;
            }
            ++(it->second.outputs);
        }
    }

    bool all_done = false;
    while (!all_done)
    {
        all_done = true;
        for (auto it = relays.begin(); it != relays.end();)
        {
            if (it->second.outputs == it->second.self_inputs)
            {
                all_done = false;
                for (auto it_edge = edges.begin(); it_edge != edges.end();)
                {
                    if (it_edge->sink_type == RELAY && it_edge->sink_num == it->first)
                    {
                        if (it_edge->source_type == RELAY)
                        {
                            --(relays[it_edge->source_num].outputs);
                        }
                        it_edge = edges.erase(it_edge);
                    }
                    else
                    {
                        ++it_edge;
                    }
                }
                it = relays.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    int new_num = 0;
    for (auto &relay : relays)
        relay.second.new_number = new_num++;

    nnet.edges.clear();
    for (const auto &edge : edges)
    {
        if (edge.sink_type == RELAY)
        {
            nnet.edges.push_back(edge);
            auto &new_edge = nnet.edges.back();
            new_edge.sink_num = relays[new_edge.sink_num].new_number;
            if (new_edge.source_type == RELAY)
                new_edge.source_num = relays[new_edge.source_num].new_number;
        }
    }

    for (const auto &edge : edges)
    {
        if (edge.sink_type == ACTION)
        {
            nnet.edges.push_back(edge);
            auto &new_edge = nnet.edges.back();
            if (new_edge.source_type == RELAY)
                new_edge.source_num = relays[new_edge.source_num].new_number;
        }
    }

    nnet.nodes.clear();
    for (int num = 0; num < relays.size(); ++num)
    {
        nnet.nodes.push_back(Node());
        nnet.nodes.back().output = 0.5f;
        nnet.nodes.back().driven = (relays[num].other_inputs != 0);
    }
}

Genome generate_child_genome(std::vector<Genome> &parent_genomes)
{
    int index = random_uint() % parent_genomes.size();
    Genome genome = parent_genomes[index];
    int num_genes = genome.size();
    while (num_genes--)
    {
        if ((random_uint() / (float)0xffffffff) < 0.01)
        {
            int gene_index = random_uint() % genome.size();
            int bit_index = random_uint() % 16;
            uint16_t &gene = genome[gene_index];
            gene = gene ^ (1 << bit_index);
        }
    }
    return genome;
}
