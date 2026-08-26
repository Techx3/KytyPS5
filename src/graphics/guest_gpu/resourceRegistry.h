#ifndef GRAPHICS_GUEST_GPU_RESOURCE_REGISTRY_H
#define GRAPHICS_GUEST_GPU_RESOURCE_REGISTRY_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics::ResourceRegistration {

using OwnerHandle    = uint32_t;
using ResourceHandle = uint32_t;

inline constexpr OwnerHandle    InvalidOwnerHandle    = UINT32_MAX;
inline constexpr ResourceHandle InvalidResourceHandle = UINT32_MAX;

struct OwnerRecord {
	OwnerHandle handle = InvalidOwnerHandle;
	std::string name;
};

struct ResourceRecord {
	ResourceHandle handle       = InvalidResourceHandle;
	OwnerHandle    owner_handle = InvalidOwnerHandle;
	uint64_t       address      = 0;
	uint64_t       size_bytes   = 0;
	uint32_t       type         = 0;
	uint64_t       user_data    = 0;
	std::string    name;
};

struct Snapshot {
	bool                        initialized     = false;
	uint32_t                    max_name_length = 0;
	std::vector<OwnerRecord>    owners;
	std::vector<ResourceRecord> resources;
};

class Registry final {
public:
	[[nodiscard]] static bool QueryMemoryRequirement(size_t*  size_bytes,
	                                                 uint32_t max_owners_and_resources,
	                                                 uint32_t max_name_length) noexcept;

	bool Initialize(void* memory, size_t size_bytes, uint32_t max_name_length);
	void Reset();

	[[nodiscard]] bool GetDefaultOwner(OwnerHandle* owner_handle) const;
	[[nodiscard]] bool GetMaxNameLength(uint32_t* max_name_length) const;
	[[nodiscard]] bool RegisterOwner(OwnerHandle* owner_handle, const char* name);
	[[nodiscard]] bool RegisterResource(ResourceHandle* resource_handle, OwnerHandle owner_handle,
	                                    const void* memory, size_t size_bytes, const char* name,
	                                    uint32_t type, uint64_t user_data);
	[[nodiscard]] bool UnregisterOwnerAndResources(OwnerHandle owner_handle);
	[[nodiscard]] bool UnregisterResource(ResourceHandle resource_handle);

	[[nodiscard]] Snapshot GetSnapshot() const;

private:
	[[nodiscard]] bool HasOwner(OwnerHandle owner_handle) const;
	[[nodiscard]] bool ValidName(const char* name) const;
	[[nodiscard]] bool AtCapacity() const;

	mutable std::mutex          m_mutex;
	bool                        m_initialized          = false;
	uint32_t                    m_max_name_length      = 0;
	uint32_t                    m_capacity             = 0;
	OwnerHandle                 m_default_owner        = InvalidOwnerHandle;
	OwnerHandle                 m_next_owner_handle    = 1;
	ResourceHandle              m_next_resource_handle = 1;
	std::vector<OwnerRecord>    m_owners;
	std::vector<ResourceRecord> m_resources;
};

[[nodiscard]] Registry&   GetRegistry();
[[nodiscard]] const char* ResourceTypeName(uint32_t type) noexcept;

} // namespace Libs::Graphics::ResourceRegistration

#endif // GRAPHICS_GUEST_GPU_RESOURCE_REGISTRY_H
