//
// Created by Hashara Kumarasinghe on 23/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_LOGGER_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_LOGGER_H

#pragma once

#include <string>

void initLogger(const std::string &filename);

void logInfo(const std::string &message);

void logResult(const std::string &backend, int num_taxa, int num_sites, int num_patterns, double time_sec,
               double likelihood);

void closeLogger();


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_LOGGER_H
