// ppm2png.cpp
// Usage: ./ppm2png input.ppm output.png
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct PPMImage {
    int width = 0;
    int height = 0;
    int maxval = 255;            // 0–255 が一般的
    std::vector<uint8_t> pixels; // RGBRGB… (8‑bit/chan)
};

static void skip_comments(std::istream& is) {
    char ch;
    while (is >> std::ws && is.peek() == '#') {
        std::string dummy;
        std::getline(is, dummy);
    }
}

PPMImage load_ppm(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Cannot open " + filename);

    std::string magic;
    ifs >> magic;
    if (magic != "P3" && magic != "P6")
        throw std::runtime_error("Unsupported PPM type: " + magic);

    skip_comments(ifs);
    int w, h, maxv;
    ifs >> w;
    skip_comments(ifs);
    ifs >> h;
    skip_comments(ifs);
    ifs >> maxv;
    ifs.get(); // consume single whitespace (newline)

    if (maxv <= 0 || maxv > 65535)
        throw std::runtime_error("Invalid maxval");

    const size_t num_pixels = static_cast<size_t>(w) * h;
    std::vector<uint8_t> data(num_pixels * 3);

    if (magic == "P6") {
        // binary
        if (maxv < 256) {
            ifs.read(reinterpret_cast<char *>(data.data()), data.size());
        } else {
            // 16‑bit -> 8‑bit down‑sample
            for (size_t i = 0; i < num_pixels * 3; ++i) {
                uint16_t v =
                    (static_cast<uint16_t>(ifs.get()) << 8) | ifs.get();
                data[i] = static_cast<uint8_t>(v * 255 / maxv);
            }
        }
    } else {
        // ASCII
        for (size_t i = 0; i < num_pixels * 3; ++i) {
            int v;
            ifs >> v;
            if (v < 0 || v > maxv)
                throw std::runtime_error("Pixel value out of range");
            data[i] = static_cast<uint8_t>(v * 255 / maxv);
        }
    }

    if (!ifs)
        throw std::runtime_error("Unexpected EOF while reading pixel data");

    PPMImage img;
    img.width = w;
    img.height = h;
    img.maxval = maxv;
    img.pixels = std::move(data);
    return img;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input.ppm output.png\n";
        return EXIT_FAILURE;
    }
    const std::string in = argv[1];
    const std::string out = argv[2];

    try {
        PPMImage img = load_ppm(in);

        if (stbi_write_png(out.c_str(), img.width, img.height,
                           3, // RGB
                           img.pixels.data(), img.width * 3) == 0) {
            throw std::runtime_error("stbi_write_png failed");
        }
        std::cout << "Converted " << in << " → " << out << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
