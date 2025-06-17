#include <fstream>
#include "simulator.hpp"
#include "video_writer.hpp"
#include <filesystem>
namespace fs = std::filesystem;

const fs::path BASE_DIR = fs::current_path().parent_path() / "visuals";

void draw_nnet(unsigned generation)
{
    for (uint16_t index = 1; index <= 3; ++index)
    {
        fs::path graph_path = BASE_DIR / "draw_nnet" / "nnetText" / ("gen-" + std::to_string(generation) + "-index-" + std::to_string(index) + ".txt");
        fs::create_directories(graph_path.parent_path());
        std::ofstream output_file(graph_path);
        for (auto &conn : peeps[index].nnet.connections)
        {
            if (conn.source_type == SENSOR)
            {
                output_file << "S" << std::to_string(conn.source_num) << " ";
            }
            else
            {
                output_file << "N" << std::to_string(conn.source_num) << " ";
            }

            if (conn.sink_type == ACTION)
            {
                output_file << "A" + std::to_string(conn.sink_num) << " ";
            }
            else
            {
                output_file << "N" << std::to_string(conn.sink_num) << " ";
            }
            output_file << std::to_string(conn.weight) << std::endl;
        }
        output_file.close();
    }
}

void save_one_frame_immed(const ImageFrameData &data, std::vector<cv::Mat> &image_list)
{
    uint8_t color[3];
    cv::Mat image(128 * 8, 128 * 8, CV_8UC3, cv::Scalar(255, 255, 255));

    for (Coord loc : data.barrier_locs)
    {
        cv::Point p1(loc.x * 8, ((128 - loc.y) - 1) * 8);
        cv::Point p2((loc.x + 1) * 8, ((128 - (loc.y - 0))) * 8);
        cv::rectangle(image, p1, p2, cv::Scalar(0x88, 0x88, 0x88), -1);
    }

    for (size_t i = 0; i < data.indiv_locs.size(); ++i)
    {
        int c = data.indiv_colors[i];

        color[0] = (c & 0xff);
        color[1] = ((c >> 8) & 0xff);
        color[2] = ((c >> 16) & 0xff);

        cv::Point p1((data.indiv_locs[i].x + 0.5) * 8, ((128 - data.indiv_locs[i].y + 0.5) - 1) * 8);
        cv::circle(image, p1, 4, cv::Scalar(color[0], color[1], color[2]), -1);
    }

    image_list.push_back(image);
}

uint32_t make_genetic_color(const Genome &genome)
{
    uint32_t hash = 0;
    for (Gene gene : genome)
    {
        hash = (hash * 31) + ((gene.source_num << 24) | (gene.sink_num << 16) | (gene.source_type << 8) | gene.sink_type);
    }
    return hash;
}

void save_video_frame(unsigned sim_step, unsigned generation, std::vector<cv::Mat> &image_list)
{
    ImageFrameData data;
    data.sim_step = sim_step;
    data.generation = generation;
    data.indiv_locs.clear();
    data.indiv_colors.clear();
    data.barrier_locs.clear();
    for (uint16_t index = 1; index <= 3000; ++index)
    {
        const Indiv &indiv = peeps[index];
        data.indiv_locs.push_back(indiv.loc);
        data.indiv_colors.push_back(make_genetic_color(indiv.genome));
    }
    auto const &barrier_locs = barrier_locations;
    for (Coord loc : barrier_locs)
    {
        data.barrier_locs.push_back(loc);
    }
    save_one_frame_immed(data, image_list);
}

void save_generation_video(unsigned generation, std::vector<cv::Mat> &image_list)
{
    if (image_list.size() > 0)
    {
        fs::path video_path = BASE_DIR / "videos" / ("gen-" + std::to_string(generation) + ".avi");
        fs::create_directories(video_path.parent_path());
        cv::VideoWriter save_video(video_path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, image_list[0].size(), true);
        for (cv::Mat &frame : image_list)
        {
            save_video.write(frame);
        }
        save_video.release();
    }
}

void save_generation_image(unsigned generation, std::vector<cv::Mat> &image_list)
{
    fs::path image_path = BASE_DIR / "images" / ("gen-" + std::to_string(generation) + ".jpg");
    fs::create_directories(image_path.parent_path());
    cv::imwrite(image_path.string(), image_list.back());
}
