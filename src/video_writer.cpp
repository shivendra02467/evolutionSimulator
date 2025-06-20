#include <fstream>
#include "simulator.hpp"
#include "video_writer.hpp"
#include <filesystem>

namespace fs = std::filesystem;

const fs::path BASE_DIR = fs::current_path().parent_path() / "visuals";

void draw_nnet(int generation)
{
    for (int index = 1; index <= 3; ++index)
    {
        fs::path graph_path = BASE_DIR / "draw_nnet" / "nnet_text" / ("gen-" + std::to_string(generation) + "-index-" + std::to_string(index) + ".txt");
        fs::create_directories(graph_path.parent_path());
        std::ofstream output_file(graph_path);
        for (auto &edge : peeps[index].nnet.edges)
        {
            if (edge.source_type == SENSOR)
                output_file << "S" << std::to_string(edge.source_num) << " ";
            else
                output_file << "R" << std::to_string(edge.source_num) << " ";

            if (edge.sink_type == ACTION)
                output_file << "A" + std::to_string(edge.sink_num) << " ";
            else
                output_file << "R" << std::to_string(edge.sink_num) << " ";
            output_file << std::to_string(edge.weight) << std::endl;
        }
        output_file.close();
    }
    const int graph_width = 1600;
    const int graph_height = 1200;
    cv::Mat full_img(graph_height, graph_width * 3, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int index = 0; index < 3; ++index)
    {
        const auto &edges = peeps[1 + index].nnet.edges;
        std::set<std::string> sensors, relays, actions;
        std::map<std::string, cv::Point> positions;
        for (const auto &e : edges)
        {
            std::string src = (e.source_type == 1 ? "S" : "R") + std::to_string(e.source_num);
            std::string dst = (e.sink_type == 1 ? "A" : "R") + std::to_string(e.sink_num);
            if (e.source_type == 1)
                sensors.insert(src);
            else
                relays.insert(src);

            if (e.sink_type == 1)
                actions.insert(dst);
            else
                relays.insert(dst);
        }
        int offset_x = index * graph_width;
        int center_x = offset_x + graph_width / 2;
        int center_y = graph_height / 2;
        auto assign_positions = [&](const std::set<std::string> &nodes, int y_pos, bool circular = false)
        {
            int count = nodes.size();
            int idx = 0;
            for (const auto &label : nodes)
            {
                cv::Point pt;
                if (circular)
                {
                    double angle = 2 * M_PI * idx / count;
                    pt = cv::Point(center_x + 500 * std::cos(angle), center_y + 500 * std::sin(angle));
                }
                else
                {
                    pt = cv::Point(offset_x + 400 + (idx + 1) * graph_width / (2 * count + 1), y_pos);
                }
                positions[label] = pt;
                idx++;
            }
        };
        assign_positions(sensors, 500);
        assign_positions(relays, 0, true);
        assign_positions(actions, graph_height - 500);
        for (const auto &e : edges)
        {
            std::string src = (e.source_type == 1 ? "S" : "R") + std::to_string(e.source_num);
            std::string dst = (e.sink_type == 1 ? "A" : "R") + std::to_string(e.sink_num);
            const cv::Point &p1 = positions[src];
            const cv::Point &p2 = positions[dst];
            if (src == dst)
            {
                cv::ellipse(full_img, p1 + cv::Point(0, -25), cv::Size(25, 25), 0, 0, 360, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
            }
            else
            {
                cv::line(full_img, p1, p2, cv::Scalar(0, 0, 0), 2, cv::LINE_AA, 0);
            }
        }
        auto draw_nodes = [&](const std::set<std::string> &node_set, const cv::Scalar &color)
        {
            for (const auto &label : node_set)
            {
                const cv::Point &pos = positions[label];
                cv::circle(full_img, pos, 18, color, -1);
                cv::putText(full_img, label, pos + cv::Point(-10, 6), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            }
        };
        draw_nodes(sensors, cv::Scalar(255, 0, 0)); // Blue = Sensor
        draw_nodes(relays, cv::Scalar(0, 255, 0));  // Green = Relay
        draw_nodes(actions, cv::Scalar(0, 0, 255)); // Red = Action
    }
    fs::path graph_path1 = BASE_DIR / "draw_nnet" / ("gen-" + std::to_string(generation) + ".png");
    fs::create_directories(graph_path1.parent_path());
    cv::imwrite(graph_path1.string(), full_img);
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
    for (uint16_t gene : genome)
    {
        hash = (hash * 31) + ((((gene >> 1) & 7) << 24) | (((gene >> 5) & 7) << 16) | ((gene & 1) << 8) | ((gene >> 4) & 1));
    }
    return hash;
}

void save_video_frame(int sim_step, int generation, std::vector<cv::Mat> &image_list)
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

void save_generation_video(int generation, std::vector<cv::Mat> &image_list)
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

void save_generation_image(int generation, std::vector<cv::Mat> &image_list)
{
    fs::path image_path = BASE_DIR / "images" / ("gen-" + std::to_string(generation) + ".jpg");
    fs::create_directories(image_path.parent_path());
    cv::imwrite(image_path.string(), image_list.back());
}
