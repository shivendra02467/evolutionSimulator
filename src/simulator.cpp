#include <random>
#include "simulator.h"
#include "videoWriter.h"

Grid grid;
Peeps peeps;

Coord Dir::asCoord()
{
    const Coord DirCoords[9] = {
        Coord(1, 0),
        Coord(1, -1),
        Coord(0, -1),
        Coord(-1, -1),
        Coord(-1, 0),
        Coord(-1, 1),
        Coord(0, 1),
        Coord(1, 1),
        Coord(0, 0)};
    return DirCoords[(uint8_t)dir_val];
}

Dir Coord::asDir()
{
    constexpr uint16_t tanN = 13860;
    constexpr uint16_t tanD = 33461;
    const Dir conversion[16]{
        Compass::S,
        Compass::CENTER,
        Compass::SW,
        Compass::N,
        Compass::SE,
        Compass::E,
        Compass::N,
        Compass::N,
        Compass::N,
        Compass::N,
        Compass::W,
        Compass::NW,
        Compass::N,
        Compass::NE,
        Compass::N,
        Compass::N};
    const int32_t xp = x * tanD + y * tanN;
    const int32_t yp = y * tanD - x * tanN;
    return conversion[(yp > 0) * 8 + (xp > 0) * 4 + (yp > xp) * 2 + (yp >= -xp)];
}

uint32_t randomUint()
{
    std::random_device seed;
    std::mt19937 generator(seed());
    std::uniform_int_distribution<uint32_t> distribution(0, std::numeric_limits<uint32_t>::max());
    return distribution(generator);
}

Coord Grid::findEmptyLocation()
{
    Coord loc;
    while (true)
    {
        loc.x = randomUint() % 128;
        loc.y = randomUint() % 128;
        if (grid.isEmptyAt(loc))
        {
            break;
        }
    }
    return loc;
}

void Grid::createBarrier()
{
    // ##############################################################################################################################
    // int16_t minX = 128 / 2;
    // int16_t maxX = minX + 1;
    // int16_t minY = 128 / 4;
    // int16_t maxY = minY + 128 / 2;

    // for (int16_t x = minX; x <= maxX; ++x)
    // {
    //     for (int16_t y = minY; y <= maxY; ++y)
    //     {
    //         grid.set(x, y, BARRIER);
    //         barrierLocations.push_back({x, y});
    //     }
    // }
    // ##############################################################################################################################
    // int16_t minX = 128 / 4;
    // int16_t maxX = minX + 128 / 2;
    // int16_t minY = 128 / 2 + 128 / 4;
    // int16_t maxY = minY + 2;

    // for (int16_t x = minX; x <= maxX; ++x)
    // {
    //     for (int16_t y = minY; y <= maxY; ++y)
    //     {
    //         grid.set(x, y, BARRIER);
    //         barrierLocations.push_back({x, y});
    //     }
    // }
}

float Indiv::getSensor(uint16_t sensorNum)
{
    float sensorVal = 0.0;

    switch (sensorNum)
    {
    case 0:
    {
        sensorVal = (float)age / 300;
        break;
    }
    case 1:
    {
        sensorVal = (float)loc.x / (128 - 1);
        break;
    }
    case 2:
    {
        sensorVal = (float)loc.y / (128 - 1);
        break;
    }
    case 3:
    {
        auto lastX = lastMoveDir.asCoord().x;
        sensorVal = lastX == 0 ? 0.5 : (lastX == -1 ? 0.0 : 1.0);
        break;
    }
    case 4:
    {
        auto lastY = lastMoveDir.asCoord().y;
        sensorVal = lastY == 0 ? 0.5 : (lastY == -1 ? 0.0 : 1.0);
        break;
    }
    case 5:
    {
        int minDistX = std::min<int>(loc.x, (128 - loc.x) - 1);
        sensorVal = minDistX / (128 / 2.0);
        break;
    }
    case 6:
    {
        int minDistY = std::min<int>(loc.y, (128 - loc.y) - 1);
        sensorVal = minDistY / (128 / 2.0);
        break;
    }
    case 7:
    {
        unsigned count = 0;
        Coord testLoc;
        testLoc.x = loc.x + lastMoveDir.asCoord().x;
        testLoc.y = loc.y + lastMoveDir.asCoord().y;
        unsigned numLocsToTest = 32;
        while (numLocsToTest > 0 && grid.isInBounds(testLoc) && grid.isEmptyAt(testLoc))
        {
            ++count;
            testLoc.x = testLoc.x + lastMoveDir.asCoord().x;
            testLoc.y = testLoc.y + lastMoveDir.asCoord().y;
            --numLocsToTest;
        }
        if (numLocsToTest > 0 && (!grid.isInBounds(testLoc) || grid.isBarrierAt(testLoc)))
        {
            sensorVal = (float)1;
        }
        else
        {
            sensorVal = count / (float)32;
        }
        break;
    }
    case 8:
    {
        unsigned count = 0;
        Coord testLoc;
        testLoc.x = loc.x + lastMoveDir.asCoord().x;
        testLoc.y = loc.y + lastMoveDir.asCoord().y;
        unsigned numLocsToTest = 32;
        while (numLocsToTest > 0 && grid.isInBounds(testLoc) && !grid.isBarrierAt(testLoc))
        {
            ++count;
            testLoc.x = testLoc.x + lastMoveDir.asCoord().x;
            testLoc.y = testLoc.y + lastMoveDir.asCoord().y;
            --numLocsToTest;
        }
        if (numLocsToTest > 0 && !grid.isInBounds(testLoc))
        {
            sensorVal = (float)1;
        }
        else
        {
            sensorVal = count / (float)32;
        }
        break;
    }
        return sensorVal;
    }
}

std::vector<float> Indiv::feedForward()
{
    std::vector<float> actionLevels(9, 0.0);

    std::vector<float> neuronAccumulators(nnet.neurons.size(), 0.0);

    bool neuronOutputsComputed = false;
    for (Gene &conn : nnet.connections)
    {
        if (conn.sinkType == ACTION && !neuronOutputsComputed)
        {
            for (unsigned neuronIndex = 0; neuronIndex < nnet.neurons.size(); ++neuronIndex)
            {
                if (nnet.neurons[neuronIndex].driven)
                {
                    nnet.neurons[neuronIndex].output = std::tanh(neuronAccumulators[neuronIndex]);
                }
            }
            neuronOutputsComputed = true;
        }
        float inputVal;
        if (conn.sourceType == SENSOR)
        {
            inputVal = getSensor(conn.sourceNum);
        }
        else
        {
            inputVal = nnet.neurons[conn.sourceNum % nnet.neurons.size()].output;
        }

        if (conn.sinkType == ACTION)
        {
            actionLevels[conn.sinkNum] += inputVal * conn.weightAsFloat();
        }
        else
        {
            neuronAccumulators[conn.sinkNum] += inputVal * conn.weightAsFloat();
        }
    }
    return actionLevels;
}

void simStepAction(Indiv &indiv)
{
    auto actionLevels = indiv.feedForward();

    Coord lastMoveOffset = indiv.lastMoveDir.asCoord();
    float moveX = 0;
    float moveY = 0;

    float level;
    Coord offset;
    Dir temp;

    level = actionLevels[0];
    temp.dir_val = (Compass)(randomUint() % 8);
    offset = temp.asCoord();
    moveX += offset.x * level;
    moveY += offset.y * level;

    moveX += actionLevels[1];

    moveX -= actionLevels[2];

    moveY += actionLevels[3];

    moveY -= actionLevels[4];

    level = actionLevels[5];
    moveX += lastMoveOffset.x * level;
    moveY += lastMoveOffset.y * level;

    level = actionLevels[6];
    moveX -= lastMoveOffset.x * level;
    moveY -= lastMoveOffset.y * level;

    level = actionLevels[7];
    temp.dir_val = (Compass)(((uint8_t)indiv.lastMoveDir.dir_val + 1) % 8);
    offset = temp.asCoord();
    moveX += offset.x * level;
    moveY += offset.y * level;

    level = actionLevels[8];
    temp.dir_val = (Compass)(((uint8_t)indiv.lastMoveDir.dir_val - 1) % 8);
    offset = temp.asCoord();
    moveX += offset.x * level;
    moveY += offset.y * level;

    moveX = std::tanh(moveX);
    moveY = std::tanh(moveY);
    int16_t probX = (int16_t)((randomUint() / (float)0xffffffff) < (std::abs(moveX)));
    int16_t probY = (int16_t)((randomUint() / (float)0xffffffff) < (std::abs(moveX)));
    int16_t signumX = moveX < 0.0 ? -1 : 1;
    int16_t signumY = moveY < 0.0 ? -1 : 1;
    Coord movementOffset = {(int16_t)(probX * signumX), (int16_t)(probY * signumY)};
    Coord newLoc;
    newLoc.x = indiv.loc.x + movementOffset.x;
    newLoc.y = indiv.loc.y + movementOffset.y;
    if (grid.isInBounds(newLoc) && grid.isEmptyAt(newLoc))
    {
        Coord temp;
        temp.x = newLoc.x - indiv.loc.x;
        temp.y = newLoc.y - indiv.loc.y;
        Dir moveDir = temp.asDir();
        if (grid.isEmptyAt(newLoc))
        {
            grid.set(indiv.loc, 0);
            grid.set(newLoc, indiv.index);
            indiv.loc = newLoc;
            indiv.lastMoveDir = moveDir;
        }
    }
}

void simulator()
{
    grid.init(128, 128);
    peeps.init(3000);
    unsigned generation = 0;

    grid.zeroFill();
    grid.createBarrier();

    for (uint16_t index = 1; index <= 3000; ++index)
    {
        peeps[index].initialize(index, grid.findEmptyLocation(), makeRandomGenome());
        nnetGraph(generation);
    }

    while (generation < 100)
    {
        std::vector<cv::Mat> imageList;
        for (unsigned simStep = 0; simStep < 300; ++simStep)
        {
            for (unsigned indivIndex = 1; indivIndex <= 3000; ++indivIndex)
            {
                peeps[indivIndex].age++;
                simStepAction(peeps[indivIndex]);
            }
            saveVideoFrame(simStep, generation, imageList);
        }
        if (generation % 20 == 0 || generation == 99)
        {
            saveGenerationVideo(generation, imageList);
        }
        saveGenerationImage(generation, imageList);

        std::vector<uint16_t> parents;
        std::vector<Genome> parentGenomes;
        for (uint16_t index = 1; index <= 3000; ++index)
        {
            // #####################################################################################################################
            bool passed = peeps[index].loc.x > 128 / 2;
            // bool passed = peeps[index].loc.x < 128 / 2 && peeps[index].loc.y < 128 / 2;
            // bool passed = peeps[index].loc.x < 128 - 32 && peeps[index].loc.x > 32 && peeps[index].loc.y < 128 - 32 && peeps[index].loc.y > 32;

            if (passed && !peeps[index].nnet.connections.empty())
            {
                parents.push_back(index);
            }
        }
        for (const uint16_t &parent : parents)
        {
            parentGenomes.push_back(peeps[parent].genome);
        }
        if (!parentGenomes.empty())
        {
            grid.zeroFill();
            grid.createBarrier();

            for (uint16_t index = 1; index <= 3000; ++index)
            {
                peeps[index].initialize(index, grid.findEmptyLocation(), generateChildGenome(parentGenomes));
                nnetGraph(generation);
            }
        }
        else
        {
            grid.zeroFill();
            grid.createBarrier();

            for (uint16_t index = 1; index <= 3000; ++index)
            {
                peeps[index].initialize(index, grid.findEmptyLocation(), makeRandomGenome());
                nnetGraph(generation);
            }
        }
        unsigned numberSurvivors = parentGenomes.size();
        std::cout << (numberSurvivors * 100) / (float)3000 << std::endl;
        generation++;
    }
}