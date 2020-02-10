#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor MaskProb_forward(
		const at::Tensor& embed_pixel,
    const at::Tensor& embed_center,
    const at::Tensor& sigma_center,
    const at::Tensor& boxes,
    const at::Tensor& box_areas,
    const int area_sum,
    const int mask_width) {
  if (embed_pixel.type().is_cuda()) {
#ifdef WITH_CUDA
    return MaskProb_forward_cuda(embed_pixel, embed_center, sigma_center, boxes, box_areas, area_sum, mask_width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
