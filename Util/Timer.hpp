/*
 * Timer.hpp
 *
 *  Created on: Nov 17, 2017
 *      Author: wyan
 *
 *      Reference: https://gist.github.com/jtilly/a423be999929d70406489a4103e67453
 */

#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <cstdio>
#include <chrono>
#include <sstream>
#include <iostream>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point stopTime;
    std::stringstream logfile;
    bool work = true;

public:
    explicit Timer() = default;

    explicit Timer(bool work_) :
            Timer() {
        work = work_;
    }

    ~Timer() = default;

    void start() {
        if (work)
            this->startTime = std::chrono::high_resolution_clock::now();
    }

    void stop(const std::string &s) {
        if (work) {
            this->stopTime = std::chrono::high_resolution_clock::now();
            logfile << s.c_str() << " Time elapsed = "
                    << std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime).count() / 1e6
                    << std::endl;
        }
    }

    void dump() {
        if (work)
            std::cout << logfile.rdbuf();
    }
};

#endif
