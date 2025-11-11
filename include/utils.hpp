#pragma once
#include <string>
#include <vector>
#include <cstdint>

void write_ppm(const std::string& path, const std::vector<uint8_t>& rgb, int w, int h);
