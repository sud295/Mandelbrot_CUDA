#include "utils.hpp"
#include <cstdio>
#include <cstdlib>

void write_ppm(const std::string& path, const std::vector<uint8_t>& rgb, int w, int h) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { perror("fopen"); exit(EXIT_FAILURE); }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb.data(), 1, (size_t)w*h*3, f);
    fclose(f);
}
