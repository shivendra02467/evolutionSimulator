#include <map>
#include "simulator.hpp"
#include "brain.hpp"

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
    std::vector<float> neuron_accumulators(nnet.neurons.size(), 0.0);
    bool neuron_outputs_computed = false;
    for (Gene &conn : nnet.connections)
    {
        if (conn.sink_type == ACTION && !neuron_outputs_computed)
        {
            for (int index = 0; index < nnet.neurons.size(); ++index)
                if (nnet.neurons[index].driven)
                    nnet.neurons[index].output = std::tanh(neuron_accumulators[index]);
            neuron_outputs_computed = true;
        }
        float input_val;
        if (conn.source_type == SENSOR)
            input_val = sensor_output(conn.source_num);
        else
            input_val = nnet.neurons[conn.source_num % nnet.neurons.size()].output;
        if (conn.sink_type == ACTION)
            action_output[conn.sink_num] += input_val * conn.weight_as_float();
        else
            neuron_accumulators[conn.sink_num] += input_val * conn.weight_as_float();
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
            last_move_dir = Coord(new_loc.x - indiv.loc.x, new_loc.y - indiv.loc.y);
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

void remove_connections_to_neuron(ConnectionList &connections, NodeMap &node_map, uint16_t neuron_number)
{
    for (auto it_conn = connections.begin(); it_conn != connections.end();)
    {
        if (it_conn->sink_type == RELAY && it_conn->sink_num == neuron_number)
        {
            if (it_conn->source_type == RELAY)
            {
                --(node_map[it_conn->source_num].num_outputs);
            }
            it_conn = connections.erase(it_conn);
        }
        else
        {
            ++it_conn;
        }
    }
}

void cullUselessNeurons(ConnectionList &connections, NodeMap &node_map)
{
    bool all_done = false;
    while (!all_done)
    {
        all_done = true;
        for (auto it_neuron = node_map.begin(); it_neuron != node_map.end();)
        {
            if (it_neuron->second.num_outputs == it_neuron->second.num_self_inputs)
            {
                all_done = false;
                remove_connections_to_neuron(connections, node_map, it_neuron->first);
                it_neuron = node_map.erase(it_neuron);
            }
            else
            {
                ++it_neuron;
            }
        }
    }
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
        edge.weight = (gene >> 8);
        edges.push_back(edge);
    }
    struct Relay
    {
        int new_index;
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

    cullUselessNeurons(connection_list, node_map);

    uint16_t new_number = 0;
    for (auto &node : node_map)
    {
        node.second.remapped_number = new_number++;
    }

    nnet.connections.clear();

    for (auto const &conn : connection_list)
    {
        if (conn.sink_type == RELAY)
        {
            nnet.connections.push_back(conn);
            auto &new_conn = nnet.connections.back();
            new_conn.sink_num = node_map[new_conn.sink_num].remapped_number;
            if (new_conn.source_type == RELAY)
            {
                new_conn.source_num = node_map[new_conn.source_num].remapped_number;
            }
        }
    }

    for (auto const &conn : connection_list)
    {
        if (conn.sink_type == ACTION)
        {
            nnet.connections.push_back(conn);
            auto &new_conn = nnet.connections.back();
            if (new_conn.source_type == RELAY)
            {
                new_conn.source_num = node_map[new_conn.source_num].remapped_number;
            }
        }
    }

    nnet.nodes.clear();
    for (unsigned neuron_num = 0; neuron_num < node_map.size(); ++neuron_num)
    {
        nnet.neurons.push_back({});
        nnet.neurons.back().output = 0.5f;
        nnet.neurons.back().driven = (node_map[neuron_num].num_inputs_from_sensors_or_other_neurons != 0);
    }
}

void random_bit_flip(Genome &genome)
{
    unsigned index = random_uint() % genome.size();
    uint64_t Bits = ((uint64_t)genome[index].weight << 32) | ((uint64_t)genome[index].source_type << 31) | ((uint64_t)genome[index].source_num << 16) | ((uint64_t)genome[index].sink_type << 15) | ((uint64_t)genome[index].sink_num);
    unsigned bit_index = random_uint() % 48;
    Bits ^= 1 << bit_index;
    genome[index].sink_num = Bits & 0x8000;
    Bits = Bits >> 15;
    genome[index].sink_type = Bits & 1;
    Bits = Bits >> 1;
    genome[index].source_num = Bits & 0x8000;
    Bits = Bits >> 15;
    genome[index].source_type = Bits & 1;
    Bits = Bits >> 1;
    genome[index].weight = (Bits & 0x10000) - 0x8000;
}

void apply_point_mutations(Genome &genome)
{
    unsigned number_of_genes = genome.size();
    while (number_of_genes-- > 0)
    {
        if ((random_uint() / (float)0xffffffff) < 0.001)
        {
            random_bit_flip(genome);
        }
    }
}

Genome generate_child_genome(std::vector<Genome> &parent_genomes)
{
    Genome genome;
    uint16_t index;
    index = random_uint() % parent_genomes.size();
    const Genome &genome_ = parent_genomes[index];
    genome = genome_;
    apply_point_mutations(genome);
    return genome;
}
