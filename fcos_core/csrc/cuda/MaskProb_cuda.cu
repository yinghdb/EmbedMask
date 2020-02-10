// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
// This file is modified from  https://github.com/pytorch/pytorch/blob/master/modules/detectron/sigmoid_focal_loss_op.cu
// Cheng-Yang Fu
// cyfu@cs.unc.edu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cfloat>

__global__ void MaskProbForward(
    const float* embed_pixel,
    const float* embed_center,
    const float* sigma_center,
    const int* boxes,
    const int* box_areas,
    const int area_sum, 
    const int num_pixel,
    const int mask_width,
    const int dim,
    float* probs) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= area_sum)
    return;

  int center_id = 0;
  int cur_area_sum = box_areas[0];
  int last_area_sum = 0;
  while(i >= cur_area_sum){
    center_id+=1;
    last_area_sum = cur_area_sum;
    cur_area_sum += box_areas[center_id];
  }
  int pixel_in_id = i - last_area_sum;

  const int* cur_box = &boxes[4*center_id];
  int box_width = cur_box[2] - cur_box[0];
  int x = pixel_in_id % box_width + cur_box[0];
  int y = pixel_in_id / box_width + cur_box[1];
  int pixel_id = y * mask_width + x;

  const float* p_ep = embed_pixel + pixel_id*dim;
  const float* p_ec = embed_center + center_id*dim;
  float norm2 = 0.0;

  for (int d = 0; d < dim; ++d){
    norm2 = norm2 + (*p_ep - *p_ec) * (*p_ep - *p_ec);
    p_ep++;
    p_ec++;
  }

  float p = expf(-norm2*sigma_center[center_id]);
  probs[center_id*num_pixel+pixel_id] = p;
} 

at::Tensor MaskProb_forward_cuda(
  const at::Tensor& embed_pixel,
  const at::Tensor& embed_center,
  const at::Tensor& sigma_center,
  const at::Tensor& boxes,
  const at::Tensor& box_areas,
  const int area_sum,
  const int mask_width) {
  AT_ASSERTM(embed_pixel.type().is_cuda(), "embed_pixel must be a CUDA tensor");
  AT_ASSERTM(embed_center.type().is_cuda(), "embed_center must be a CUDA tensor");
  AT_ASSERTM(sigma_center.type().is_cuda(), "sigma_center must be a CUDA tensor");
  AT_ASSERTM(embed_pixel.dim() == 2, "embed_pixel should be MxDim");
  AT_ASSERTM(embed_center.dim() == 2, "embed_center should be NxDim");
  AT_ASSERTM(sigma_center.dim() == 1, "sigma_center should be N");
  AT_ASSERTM(embed_pixel.size(1) == embed_center.size(1), "Dim should the same");
  AT_ASSERTM(embed_center.size(0) == sigma_center.size(0), "center number should be the same");
  AT_ASSERTM(embed_center.size(0) == boxes.size(0), "center number and box number should be the same");

  const int num_pixel = embed_pixel.size(0);
  const int num_center = embed_center.size(0);
  const int dim = embed_pixel.size(1);

  auto prob = at::empty({num_pixel, num_center}, embed_pixel.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(THCCeilDiv((long)area_sum, 512L));
  dim3 threads(512);

  if (prob.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return prob;
  }

  MaskProbForward<<<blocks, threads>>>(
      embed_pixel.contiguous().data<float>(),
      embed_center.contiguous().data<float>(),
      sigma_center.contiguous().data<float>(),
      boxes.contiguous().data<int>(),
      box_areas.contiguous().data<int>(),
      area_sum,
      num_pixel,
      mask_width,
      dim,
      prob.data<float>()
    );

  THCudaCheck(cudaGetLastError());
  return prob;   
}	

