#ifndef GRAPHICS_GUEST_GPU_PM4_INSPECTOR_H
#define GRAPHICS_GUEST_GPU_PM4_INSPECTOR_H

#include "graphics/guest_gpu/resourceRegistry.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::Pm4 {

enum class InspectionStatus : uint8_t {
	Known,
	Partial,
	Inferred,
	Unknown,
	Unsupported,
	UnreadableAtCaptureTime,
	Count
};

enum class SubmissionQueueKind : uint8_t { Graphics, AsyncCompute };
enum class RegisterSpace : uint8_t { Context, Shader, Uconfig };
enum class MemoryReferenceKind : uint8_t { CommandBuffer, RegisterPairs, CompareValue };

struct ResourceAnnotation {
	uint32_t    handle       = 0;
	uint32_t    owner_handle = 0;
	uint32_t    type         = 0;
	uint64_t    base_address = 0;
	uint64_t    size_bytes   = 0;
	uint64_t    offset_bytes = 0;
	std::string name;
};

struct RegisterWrite {
	RegisterSpace space  = RegisterSpace::Context;
	uint32_t      offset = 0;
	uint32_t      value  = 0;
};

struct MemoryReference {
	MemoryReferenceKind             kind            = MemoryReferenceKind::CommandBuffer;
	uint64_t                        address         = 0;
	uint32_t                        size_dw         = 0;
	InspectionStatus                snapshot_status = InspectionStatus::UnreadableAtCaptureTime;
	std::optional<size_t>           nested_buffer_id;
	std::vector<uint32_t>           snapshot_words;
	std::vector<ResourceAnnotation> resources;
	std::string                     note;
};

struct PacketInspection {
	uint32_t                     offset_dw     = 0;
	uint32_t                     size_dw       = 0;
	uint32_t                     header        = 0;
	uint32_t                     type          = 0;
	uint8_t                      opcode        = 0;
	uint32_t                     custom_opcode = 0;
	bool                         predicate     = false;
	InspectionStatus             status        = InspectionStatus::Unknown;
	std::string                  name;
	std::vector<uint32_t>        payload;
	std::vector<RegisterWrite>   register_writes;
	std::vector<MemoryReference> references;
};

struct BufferInspection {
	size_t                          id = 0;
	std::string                     role;
	uint64_t                        address          = 0;
	uint32_t                        declared_size_dw = 0;
	uint32_t                        depth            = 0;
	std::vector<uint32_t>           words;
	std::vector<PacketInspection>   packets;
	std::vector<ResourceAnnotation> resources;
};

struct DescriptorInspection {
	std::string                     stage;
	uint32_t                        source_register = 0;
	uint64_t                        address         = 0;
	InspectionStatus                status          = InspectionStatus::Inferred;
	std::string                     kind;
	std::vector<uint32_t>           words;
	std::vector<ResourceAnnotation> resources;
};

struct StateView {
	uint32_t context_registers = 0;
	uint32_t shader_registers  = 0;
	uint32_t uconfig_registers = 0;
	uint64_t hash              = 0;
};

class QueueRegisterState {
public:
	void                                  Reset();
	void                                  Set(RegisterSpace space, uint32_t offset, uint32_t value);
	[[nodiscard]] std::optional<uint32_t> Get(RegisterSpace space, uint32_t offset) const;
	[[nodiscard]] StateView               View() const;
	[[nodiscard]] const std::map<uint32_t, uint32_t>& Entries(RegisterSpace space) const;

private:
	[[nodiscard]] std::map<uint32_t, uint32_t>&       Select(RegisterSpace space);
	[[nodiscard]] const std::map<uint32_t, uint32_t>& Select(RegisterSpace space) const;

	std::map<uint32_t, uint32_t> m_context;
	std::map<uint32_t, uint32_t> m_shader;
	std::map<uint32_t, uint32_t> m_uconfig;
};

struct SubmissionMetadata {
	uint64_t            capture_id  = 0;
	uint32_t            frame       = 0;
	uint32_t            queue       = 0;
	SubmissionQueueKind queue_kind  = SubmissionQueueKind::Graphics;
	bool                reset_state = false;
};

struct InspectionLimits {
	uint32_t max_depth              = 8;
	uint32_t max_single_snapshot_dw = 256 * 1024;
	uint32_t max_total_snapshot_dw  = 1024 * 1024;
};

using MemoryReader =
    std::function<bool(uint64_t address, uint32_t size_dw, std::vector<uint32_t>* words)>;

struct SubmissionInspection {
	SubmissionMetadata                                                 metadata;
	StateView                                                          state_before_reset;
	StateView                                                          state_at_submit;
	StateView                                                          state_after;
	std::array<uint32_t, static_cast<size_t>(InspectionStatus::Count)> status_counts {};
	std::vector<BufferInspection>                                      buffers;
	ResourceRegistration::Snapshot                                     registered_resources;
	std::vector<DescriptorInspection>                                  descriptors;
	struct Comparison {
		uint64_t                 previous_capture_id = 0;
		bool                     state_changed       = false;
		std::vector<std::string> added_packets;
		std::vector<std::string> removed_packets;
		std::vector<std::string> added_descriptors;
		std::vector<std::string> removed_descriptors;
		std::vector<std::string> added_resources;
		std::vector<std::string> removed_resources;
	};
	std::optional<Comparison> comparison;
};

[[nodiscard]] SubmissionInspection
InspectSubmission(const SubmissionMetadata& metadata, const char* root_role, uint64_t root_address,
                  std::span<const uint32_t> root_words, QueueRegisterState* state,
                  const MemoryReader& reader = {}, const InspectionLimits& limits = {},
                  const ResourceRegistration::Snapshot& resources = {});

[[nodiscard]] SubmissionInspection::Comparison
CompareSubmissionInspections(const SubmissionInspection& previous,
                             const SubmissionInspection& current);

[[nodiscard]] std::string SerializeSubmissionInspection(const SubmissionInspection& inspection);

class SubmissionInspector final {
public:
	using ResourceSnapshotProvider = std::function<ResourceRegistration::Snapshot()>;

	explicit SubmissionInspector(std::filesystem::path folder, MemoryReader reader = {},
	                             InspectionLimits         limits            = {},
	                             ResourceSnapshotProvider resource_provider = {});

	bool CaptureGraphics(uint32_t frame, bool reset_state, uint64_t dcb_address,
	                     std::span<const uint32_t> dcb_words);
	bool CaptureAsyncCompute(uint32_t frame, uint32_t queue, uint64_t acb_address,
	                         std::span<const uint32_t> acb_words);

private:
	bool Capture(SubmissionQueueKind queue_kind, uint32_t frame, uint32_t queue, bool reset_state,
	             const char* role, uint64_t address, std::span<const uint32_t> words);

	std::filesystem::path                              m_folder;
	MemoryReader                                       m_reader;
	InspectionLimits                                   m_limits;
	ResourceSnapshotProvider                           m_resource_provider;
	std::unordered_map<uint32_t, QueueRegisterState>   m_queue_states;
	std::unordered_map<uint32_t, SubmissionInspection> m_previous_inspections;
	std::mutex                                         m_mutex;
	uint64_t                                           m_next_capture_id = 0;
};

[[nodiscard]] const char* InspectionStatusName(InspectionStatus status) noexcept;

} // namespace Libs::Graphics::Pm4

#endif // GRAPHICS_GUEST_GPU_PM4_INSPECTOR_H
