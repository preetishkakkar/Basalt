// The one translation unit that compiles the header-only third-party libraries.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>
