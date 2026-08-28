#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions {
	ShaderType                  stage              = ShaderType::Compute;
	uint32_t                    wave_size          = 64;
	uint32_t                    host_subgroup_size = 0;
	uint32_t                    user_data_base     = 0;
	uint32_t                    user_data_count    = 64;
	uint32_t                    scratch_dwords     = 0;
	uint32_t                    terminal_external_setpc_sgpr = UINT32_MAX;
	uint64_t                    shader_hash        = 0;
	uint64_t                    shader_base        = 0;
	std::optional<uint64_t>     flat_memory_base;
	uint32_t                    push_constant_offset      = 0;
	bool                        dump_ir                    = true;
	bool                        early_dump                 = false;
	const char*                 dump_label                 = nullptr;
	const uint32_t*             user_data                  = nullptr;
	IR::SrtMemoryReader         read_memory                = nullptr;
	IR::SrtMemoryReader         read_specialization_memory = nullptr;
	void*                       read_memory_data           = nullptr;
	const IR::ResourceSnapshot* resource_snapshot          = nullptr;
	ShaderStageInputInfo        input_info;
};

struct CompileResult {
	std::vector<uint32_t> spirv;
	std::string           decoded_dump;
	std::string           ir_dump;
	IR::Program           program;
	IR::ResourceSnapshot  resources;
};

bool TryRecompile(std::span<const uint32_t> code, const CompileOptions& options,
                  CompileResult& result, std::string* error);

// Joins a compiler-generated GsFront/GsBack pair only when the LDS hand-off and the final
// position export can be proven. The resulting stream is suitable for the vertex fallback.
bool TryLinkSplitGeometryPrograms(std::span<const uint32_t> front,
                                  std::span<const uint32_t> back,
                                  std::vector<uint32_t>& linked, std::string* error);

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_ */
