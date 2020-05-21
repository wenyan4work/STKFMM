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

#include <chrono>
#include <iostream>
#include <string>

class Timer {
  private:
    bool work = true;
    struct Block {
        std::chrono::high_resolution_clock::time_point startTime;
        std::chrono::high_resolution_clock::time_point stopTime;
        std::string message;
    };
    std::vector<Block> timing;

  public:
    explicit Timer() = default;

    explicit Timer(bool work_) : Timer() { work = work_; }

    ~Timer() = default;

    void enable() { work = true; }
    void disable() { work = false; }

    void tick() {
        if (work) {
            timing.emplace_back();
            auto &recording = timing.back();
            recording.startTime = std::chrono::high_resolution_clock::now();
        }
    }

    void tock(const std::string &s) {
        if (work) {
            auto &recording = timing.back();
            recording.stopTime = std::chrono::high_resolution_clock::now();
            recording.message = s;
        }
    }

    void dump() {
        for (const auto &event : timing) {
            std::cout
                << event.message << " :time "
                << std::chrono::duration_cast<std::chrono::microseconds>(event.stopTime - event.startTime).count() / 1e6
                << " seconds." << std::endl;
        }
    }

    std::vector<double> getTime() {
        std::vector<double> time;
        for (const auto &event : timing) {
            time.push_back(
                std::chrono::duration_cast<std::chrono::microseconds>(event.stopTime - event.startTime).count() / 1e6);
        }
        return time;
    }
};

#endif
