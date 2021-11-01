#include "Image.h"
#include "memman.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>

Image::Image() {
    allocated = false;
    width = 0;
    height = 0;
    size = 0;
    channels = 0;
    self_allocated = false;
}

Image::Image(const std::string &a_path)
{
  if((data = (Pixel*)stbi_load(a_path.c_str(), &width, &height, &channels, 4)) != nullptr)
  {
    size = width * height * 4;
  }
  else
  {
      printf(stbi_failure_reason());
  }
  allocated = true;
}

Image::Image(int a_width, int a_height, int a_channels, bool initialize)
{
    //data = new Pixel[a_width * a_height ]{};
    if (initialize) {
        data = (Pixel*)STBI_MALLOC(a_width * a_height * a_channels);
        allocated = true;
    }
    else {
        data = NULL;
        allocated = false;
    }

    width = a_width;
    height = a_height;
    size = a_width * a_height * a_channels;
    channels = a_channels;
    self_allocated = true;
}


void Image::Renew(const std::string& a_path)
{
    if (allocated) {
        if (self_allocated)
            STBI_FREE(data);
        else
        {
            stbi_image_free(data);
        }
    }

    if ((data = (Pixel*)stbi_load(a_path.c_str(), &width, &height, &channels, 4)) != nullptr)
    {
        size = width * height * 4;
    }
    else
    {
        printf(stbi_failure_reason());
    }
    allocated = true;
}


int Image::Save(const std::string &a_path)
{
  auto extPos = a_path.find_last_of('.');
  if(a_path.substr(extPos, std::string::npos) == ".png" || a_path.substr(extPos, std::string::npos) == ".PNG")
  {
    stbi_write_png(a_path.c_str(), width, height, channels, data, width * channels);
  }
  else if(a_path.substr(extPos, std::string::npos) == ".jpg" || a_path.substr(extPos, std::string::npos) == ".JPG" ||
          a_path.substr(extPos, std::string::npos) == ".jpeg" || a_path.substr(extPos, std::string::npos) == ".JPEG")
  {
    stbi_write_jpg(a_path.c_str(), width, height, channels, data, 100);
  }
  else
  {
    std::cerr << "Unknown file extension: " << a_path.substr(extPos, std::string::npos) << "in file name" << a_path << "\n";
    return 1;
  }

  return 0;
}

Image::~Image()
{
    if (!allocated) return;
    if (self_allocated)
        STBI_FREE(data);
        //delete[] data;
    else
    {
        stbi_image_free(data);
    }
}