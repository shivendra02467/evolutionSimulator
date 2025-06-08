#ifndef IMAGEWRITER_H_INCLUDED
#define IMAGEWRITER_H_INCLUDED

#include <opencv2/opencv.hpp>

struct ImageFrameData
{
    unsigned simStep;
    unsigned generation;
    std::vector<Coord> indivLocs;
    std::vector<uint32_t> indivColors;
    std::vector<Coord> barrierLocs;
};
void nnetGraph(unsigned generation);
void saveVideoFrame(unsigned simStep, unsigned generation, std::vector<cv::Mat> &imageList);
void saveGenerationVideo(unsigned generation, std::vector<cv::Mat> &imageList);
void saveGenerationImage(unsigned generation, std::vector<cv::Mat> &imageList);

#endif