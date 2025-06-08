#include <fstream>
#include "simulator.h"
#include "videoWriter.h"

void nnetGraph(unsigned generation)
{
    for (uint16_t index = 1; index <= 3; ++index)
    {
        std::stringstream graphFilename;
        graphFilename << "C:\\Users\\Kush\\Documents\\My Files\\MY PROJECTS\\evolution_simulator\\images\\nnetGraph\\nnetText\\gen-" << std::setw(6) << std::setfill('0') << generation << "-index-" << index << ".txt ";
        std::ofstream outputFile(graphFilename.str());
        for (auto &conn : peeps[index].nnet.connections)
        {
            if (conn.sourceType == SENSOR)
            {
                outputFile << "S" << std::to_string(conn.sourceNum) << " ";
            }
            else
            {
                outputFile << "N" << std::to_string(conn.sourceNum) << " ";
            }

            if (conn.sinkType == ACTION)
            {
                outputFile << "A" + std::to_string(conn.sinkNum) << " ";
            }
            else
            {
                outputFile << "N" << std::to_string(conn.sinkNum) << " ";
            }
            outputFile << std::to_string(conn.weight) << std::endl;
        }
        outputFile.close();
    }
}

void saveOneFrameImmed(const ImageFrameData &data, std::vector<cv::Mat> &imageList)
{
    uint8_t color[3];
    cv::Mat image(128 * 8, 128 * 8, CV_8UC3, cv::Scalar(255, 255, 255));

    for (Coord loc : data.barrierLocs)
    {
        cv::Point p1(loc.x * 8, ((128 - loc.y) - 1) * 8);
        cv::Point p2((loc.x + 1) * 8, ((128 - (loc.y - 0))) * 8);
        cv::rectangle(image, p1, p2, cv::Scalar(0x88, 0x88, 0x88), -1);
    }

    for (size_t i = 0; i < data.indivLocs.size(); ++i)
    {
        int c = data.indivColors[i];

        color[0] = (c & 0xff);
        color[1] = ((c >> 8) & 0xff);
        color[2] = ((c >> 16) & 0xff);

        cv::Point p1((data.indivLocs[i].x + 0.5) * 8, ((128 - data.indivLocs[i].y + 0.5) - 1) * 8);
        cv::circle(image, p1, 4, cv::Scalar(color[0], color[1], color[2]), -1);
    }

    imageList.push_back(image);
}

uint32_t makeGeneticColor(const Genome &genome)
{
    uint32_t hash = 0;
    for (Gene gene : genome)
    {
        hash = (hash * 31) + ((gene.sourceNum << 24) | (gene.sinkNum << 16) | (gene.sourceType << 8) | gene.sinkType);
    }
    return hash;
}

void saveVideoFrame(unsigned simStep, unsigned generation, std::vector<cv::Mat> &imageList)
{
    ImageFrameData data;
    data.simStep = simStep;
    data.generation = generation;
    data.indivLocs.clear();
    data.indivColors.clear();
    data.barrierLocs.clear();
    for (uint16_t index = 1; index <= 3000; ++index)
    {
        const Indiv &indiv = peeps[index];
        data.indivLocs.push_back(indiv.loc);
        data.indivColors.push_back(makeGeneticColor(indiv.genome));
    }
    auto const &barrierLocs = grid.getBarrierLocations();
    for (Coord loc : barrierLocs)
    {
        data.barrierLocs.push_back(loc);
    }
    saveOneFrameImmed(data, imageList);
}

void saveGenerationVideo(unsigned generation, std::vector<cv::Mat> &imageList)
{
    if (imageList.size() > 0)
    {
        std::stringstream videoFilename;
        videoFilename << "C:\\Users\\Kush\\Documents\\My Files\\MY PROJECTS\\evolution_simulator\\images\\gen-" << std::setw(6) << std::setfill('0') << generation << ".avi";
        cv::VideoWriter saveVideo(videoFilename.str(), cv::VideoWriter::fourcc('X', '2', '6', '4'), 25, imageList[0].size(), true);
        for (cv::Mat &frame : imageList)
        {
            saveVideo.write(frame);
        }
        saveVideo.release();
    }
}

void saveGenerationImage(unsigned generation, std::vector<cv::Mat> &imageList)
{
    std::stringstream imageFilename;
    imageFilename << "C:\\Users\\Kush\\Documents\\My Files\\MY PROJECTS\\evolution_simulator\\images\\gen-" << std::setw(6) << std::setfill('0') << generation << ".jpg";
    cv::imwrite(imageFilename.str(), imageList.back());
}
