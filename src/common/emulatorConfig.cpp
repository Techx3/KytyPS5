#include "common/emulatorConfig.h"

#include "common/assert.h"

#include <algorithm>
#include <memory>

namespace Config {

static std::unique_ptr<ConfigOptions> g_config;

void Initialize() {
	EXIT_IF(g_config != nullptr);

	g_config = std::make_unique<ConfigOptions>();
}

void Shutdown() {
	g_config.reset();
}

void Load(const ConfigOptions& cfg) {
	EXIT_IF(g_config == nullptr);

	*g_config = cfg;
}

uint32_t GetScreenWidth() {
	return g_config->screen_width;
}

uint32_t GetScreenHeight() {
	return g_config->screen_height;
}

bool FullscreenEnabled() {
	return g_config->fullscreen_enabled;
}

PresentMode GetPresentMode() {
	return g_config->present_mode;
}

uint32_t GetVblankFrequency() {
	return std::clamp(g_config->vblank_frequency, 30u, 360u);
}

uint32_t GetConsoleLanguage() {
	return g_config->console_language;
}

bool VulkanValidationEnabled() {
	return g_config->vulkan_validation_enabled;
}

bool VulkanDebugMarkersEnabled() {
	return g_config->vulkan_debug_markers_enabled;
}

bool ShaderValidationEnabled() {
	return g_config->shader_validation_enabled;
}

ShaderOptimizationType GetShaderOptimizationType() {
	return g_config->shader_optimization_type;
}

ShaderLogDirection GetShaderLogDirection() {
	return g_config->shader_log_direction;
}

std::filesystem::path GetShaderLogFolder() {
	return g_config->shader_log_folder;
}

bool CommandBufferDumpEnabled() {
	return g_config->command_buffer_dump_enabled;
}

std::filesystem::path GetCommandBufferDumpFolder() {
	return g_config->command_buffer_dump_folder;
}

bool GraphicsDebugDumpEnabled() {
	return g_config->graphics_debug_dump_enabled;
}

bool GpuPerformanceMetricsEnabled() {
	return g_config->gpu_performance_metrics_enabled;
}

OutputDirection GetPrintfDirection() {
	return g_config->printf_direction;
}

std::filesystem::path GetPrintfOutputFile() {
	return g_config->printf_output_file;
}

ProfilerDirection GetProfilerDirection() {
	return g_config->profiler_direction;
}

bool SpirvDebugPrintfEnabled() {
	return g_config->spirv_debug_printf_enabled;
}

bool GpuAssistedValidationEnabled() {
	return g_config->gpu_assisted_validation_enabled && g_config->vulkan_validation_enabled;
}

bool RenderDocEnabled() {
	return g_config->renderdoc_enabled;
}

bool ReadbackLinearImagesEnabled() {
	return g_config->readback_linear_images;
}

bool PlayGoHackEnabled() {
	return g_config->playgo_hack_enabled;
}

bool StrictUnresolvedImportsEnabled() {
	return g_config->strict_unresolved_imports;
}

std::filesystem::path GetUnresolvedImportReport() {
	return g_config->unresolved_import_report;
}

int32_t GetUserId() {
	return g_config->user_id;
}

uint32_t GetLocalPlayerCount() {
	return std::clamp(g_config->local_player_count, 1u, MAX_LOCAL_USERS);
}

int32_t GetLocalUserId(uint32_t player_index) {
	EXIT_IF(player_index >= MAX_LOCAL_USERS);

	return g_config->user_id + static_cast<int32_t>(player_index);
}

int32_t GetLocalUserIndex(int32_t user_id) {
	const int64_t index = static_cast<int64_t>(user_id) - g_config->user_id;

	return index >= 0 && index < GetLocalPlayerCount() ? static_cast<int32_t>(index) : -1;
}

const std::string& GetControllerGuid(uint32_t player_index) {
	EXIT_IF(player_index >= MAX_LOCAL_USERS);

	return g_config->controller_guids[player_index];
}

const std::string& GetUserName() {
	return g_config->user_name;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled() {
	return g_config->red_zone_protection_enabled;
}
#endif

const Keymap& GetKeymap() {
	return g_config->keymap;
}

} // namespace Config
