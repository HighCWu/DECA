#include <torch/torch.h>
#include <vector>
#include <iostream>

std::vector<at::Tensor> forward_rasterize_cuda(
        at::Tensor face_vertices,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor baryw_buffer,
        int64_t h,
        int64_t w);

std::vector<at::Tensor> standard_rasterize(
        at::Tensor face_vertices,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor baryw_buffer,
        int64_t height, int64_t width
        ) {
    return forward_rasterize_cuda(face_vertices, depth_buffer, triangle_buffer, baryw_buffer, height, width);
}

std::vector<at::Tensor> forward_rasterize_colors_cuda(
        at::Tensor face_vertices,
        at::Tensor face_colors,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor images,
        int64_t h,
        int64_t w);

std::vector<at::Tensor> standard_rasterize_colors(
        at::Tensor face_vertices,
        at::Tensor face_colors,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor images,
        int64_t height, int64_t width
        ) {
    return forward_rasterize_colors_cuda(face_vertices, face_colors, depth_buffer, triangle_buffer, images, height, width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("standard_rasterize", &standard_rasterize, "RASTERIZE (CUDA)");
    m.def("standard_rasterize_colors", &standard_rasterize_colors, "RASTERIZE COLORS (CUDA)");
}

TORCH_LIBRARY(rasterize_ops, m) {
  m.def("standard_rasterize", standard_rasterize);
  m.def("standard_rasterize_colors", standard_rasterize_colors);
}

// TODO: backward