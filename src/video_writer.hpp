#ifndef VIDEO_WRITER_HPP
#define VIDEO_WRITER_HPP

#include <opencv2/opencv.hpp>

struct ImageFrameData
{
    unsigned sim_step;
    unsigned generation;
    std::vector<Coord> indiv_locs;
    std::vector<uint32_t> indiv_colors;
    std::vector<Coord> barrier_locs;
};
void draw_nnet(int generation);
void save_video_frame(int sim_step, int generation, std::vector<cv::Mat> &image_list);
void save_generation_video(int generation, std::vector<cv::Mat> &image_list);
void save_generation_image(int generation, std::vector<cv::Mat> &image_list);

#endif