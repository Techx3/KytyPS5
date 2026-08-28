#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_

#include <cstdint>

namespace Libs::Graphics {

struct ShaderVertexInputInfo;

[[nodiscard]] int32_t ResolveVertexOffset(uint32_t                     index_offset,
                                          const ShaderVertexInputInfo& vs_input_info);

// GE_CNTL stores both subgroup sizes in 9-bit fields. Values larger than a wave are valid guest
// scheduling hints and do not change Vulkan primitive assembly.
[[nodiscard]] bool IsGeControlGroupSizeValid(uint16_t primitive_group_size,
                                             uint16_t vertex_group_size);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
