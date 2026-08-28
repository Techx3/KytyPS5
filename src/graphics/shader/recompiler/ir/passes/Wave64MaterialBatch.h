#pragma once

#include "graphics/shader/recompiler/ir/ValueProgram.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

struct Wave64MaterialBatchStats {
	uint32_t rewritten_masks = 0;
};

// Some RDNA2 pixel shaders collect the material IDs used by a wave into a scalar bit mask,
// then sample one descriptor at a time. A native subgroup32 cannot read guest lane 63, but the
// two 32-lane halves can execute this read-only batching loop independently without changing any
// per-lane result. Only the complete compiler idiom is rewritten; arbitrary Wave64 lane traffic is
// deliberately left untouched.
[[nodiscard]] Wave64MaterialBatchStats SplitWave64MaterialBatches(ValueProgram& program,
                                                                  uint32_t      guest_wave_size,
                                                                  uint32_t      host_subgroup_size);

} // namespace Libs::Graphics::ShaderRecompiler::IR
