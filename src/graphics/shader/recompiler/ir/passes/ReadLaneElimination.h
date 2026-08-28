#pragma once

#include "graphics/shader/recompiler/ir/ValueProgram.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

struct ReadLaneStats {
	uint32_t rewritten_reads = 0;
};

[[nodiscard]] ReadLaneStats EliminateReadLane(ValueProgram& program, uint32_t wave_size,
                                              const Program* runtime_program = nullptr);

} // namespace Libs::Graphics::ShaderRecompiler::IR
