#ifndef MAIN_IMAGE_H
#define MAIN_IMAGE_H

#include "memman.cuh"

#include <string>
#include <stdint.h>

struct Pixel
{
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t a;
};

struct Image
{
	explicit Image(const std::string& a_path);
	Image(int a_width, int a_height, int a_channels, bool initialize = true);
	Image();

	int Save(const std::string& a_path);
	void Renew(const std::string& a_path);

	int Width()    const { return width; }
	int Height()   const { return height; }
	int Channels() const { return channels; }
	size_t Size()  const { return size; }
	Pixel*& Data() { return  data; }

	Pixel GetPixel(int x, int y) { return data[width * y + x]; }
	void  PutPixel(int x, int y, const Pixel& pix) { data[width * y + x] = pix; }

	~Image();

private:
	int width;
	int height;
	int channels;
	size_t size;
	Pixel* data;
	bool self_allocated;
	bool allocated;
};

#endif //MAIN_IMAGE_H
