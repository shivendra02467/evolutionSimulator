#include <random>
#include "omp.h"
#include "simulator.hpp"
#include "video_writer.hpp"

int grid[128][128];
std::vector<Coord> barrier_locations;
Indiv peeps[3001];

unsigned int random_uint()
{
    static thread_local std::mt19937 generator(
        []
        {
            std::random_device seed;
            return seed();
        }());
    std::uniform_int_distribution<unsigned int> distribution(0, std::numeric_limits<unsigned int>::max());
    return distribution(generator);
}

Coord find_empty_location()
{
    Coord loc;
    while (true)
    {
        loc.x = random_uint() % 128;
        loc.y = random_uint() % 128;
        if (grid[loc.x][loc.y] == EMPTY)
            break;
    }
    return loc;
}

void set_grid()
{
    for (int i = 0; i < 128; i++)
        for (int j = 0; j < 128; j++)
            grid[i][j] = 0;

    int min_x = 128 / 2;
    int max_x = min_x + 1;
    int min_y = 128 / 4;
    int max_y = min_y + 128 / 2;

    for (int x = min_x; x <= max_x; ++x)
    {
        for (int y = min_y; y <= max_y; ++y)
        {
            grid[x][y] = BARRIER;
            barrier_locations.push_back(Coord(x, y));
        }
    }
    // int min_x = 128 / 4;
    // int max_x = min_x + 128 / 2;
    // int min_y = 128 / 2 + 128 / 4;
    // int max_y = min_y + 2;

    // for (int x = min_x; x <= max_x; ++x)
    // {
    //     for (int y = min_y; y <= max_y; ++y)
    //     {
    //         grid[x][y] = BARRIER;
    //         barrier_locations.push_back(Coord(x, y));
    //     }
    // }
}

void simulator()
{
    int generation = 0;
    set_grid();
    for (int index = 1; index <= 3000; ++index)
    {
        Coord loc = find_empty_location();
        peeps[index].index = index;
        peeps[index].loc = loc;
        peeps[index].age = 0;
        peeps[index].last_move_dir = Coord(random_uint() % 3 - 1, random_uint() % 3 - 1);
        peeps[index].genome = make_random_genome();
        peeps[index].construct_nnet();
        grid[loc.x][loc.y] = index;
    }
    std::vector<cv::Mat> image_list;
#pragma omp parallel num_threads(4) default(shared)
    {
        while (generation < 100)
        {
            for (int sim_step = 0; sim_step < 300; ++sim_step)
            {
#pragma omp for schedule(auto)
                for (int index = 1; index <= 3000; ++index)
                {
                    peeps[index].age++;
                    peeps[index].perform_action();
                }
#pragma omp single
                save_video_frame(sim_step, generation, image_list);
            }
#pragma omp single
            {
                save_generation_video(generation, image_list);
                save_generation_image(generation, image_list);
                draw_nnet(generation);
                image_list.clear();
                std::vector<Genome> parent_genomes;
                for (int index = 1; index <= 3000; ++index)
                {
                    bool passed = peeps[index].loc.x > 128 / 2;
                    // bool passed = peeps[index].loc.x < 128 / 2 && peeps[index].loc.y < 128 / 2;
                    // bool passed = peeps[index].loc.x < 128 - 32 && peeps[index].loc.x > 32 && peeps[index].loc.y < 128 - 32 && peeps[index].loc.y > 32;
                    if (passed && !peeps[index].nnet.connections.empty())
                        parent_genomes.push_back(peeps[index].genome);
                }
                if (!parent_genomes.empty())
                {
                    for (int index = 1; index <= 3000; ++index)
                    {
                        Coord loc = find_empty_location();
                        peeps[index].index = index;
                        peeps[index].loc = loc;
                        peeps[index].age = 0;
                        peeps[index].last_move_dir = Coord(random_uint() % 3 - 1, random_uint() % 3 - 1);
                        peeps[index].genome = generate_child_genome(parent_genomes);
                        peeps[index].construct_nnet();
                        grid[loc.x][loc.y] = index;
                    }
                }
                else
                {
                    for (int index = 1; index <= 3000; ++index)
                    {
                        Coord loc = find_empty_location();
                        peeps[index].index = index;
                        peeps[index].loc = loc;
                        peeps[index].age = 0;
                        peeps[index].last_move_dir = Coord(random_uint() % 3 - 1, random_uint() % 3 - 1);
                        peeps[index].genome = make_random_genome();
                        peeps[index].construct_nnet();
                        grid[loc.x][loc.y] = index;
                    }
                }
                set_grid();
                std::cout << (parent_genomes.size() * 100) / 3000.0f << std::endl;
                generation++;
            }
        }
    }
}

int main()
{
    simulator();
}