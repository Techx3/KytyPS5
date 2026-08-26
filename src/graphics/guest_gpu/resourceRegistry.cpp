#include "graphics/guest_gpu/resourceRegistry.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace Libs::Graphics::ResourceRegistration {

namespace {

constexpr uint32_t MinimumNameLength = 16;
constexpr uint32_t NameAlignment     = 8;
constexpr size_t   HeaderBytes       = 64;
constexpr size_t   RecordBytes       = 64;

bool ValidNameLength(uint32_t value) {
	return value >= MinimumNameLength && value % NameAlignment == 0;
}

} // namespace

bool Registry::QueryMemoryRequirement(size_t* size_bytes, uint32_t max_owners_and_resources,
                                      uint32_t max_name_length) noexcept {
	if (size_bytes == nullptr || max_owners_and_resources == 0 ||
	    !ValidNameLength(max_name_length)) {
		return false;
	}
	const size_t bytes_per_record = RecordBytes + max_name_length;
	if (max_owners_and_resources >
	    (std::numeric_limits<size_t>::max() - HeaderBytes) / bytes_per_record) {
		return false;
	}
	*size_bytes = HeaderBytes + static_cast<size_t>(max_owners_and_resources) * bytes_per_record;
	return true;
}

bool Registry::Initialize(void* memory, size_t size_bytes, uint32_t max_name_length) {
	if (memory == nullptr || !ValidNameLength(max_name_length) || size_bytes < HeaderBytes) {
		return false;
	}
	const size_t bytes_per_record = RecordBytes + max_name_length;
	const size_t capacity         = (size_bytes - HeaderBytes) / bytes_per_record;
	if (capacity == 0 || capacity > std::numeric_limits<uint32_t>::max()) {
		return false;
	}

	std::lock_guard lock(m_mutex);
	m_initialized          = true;
	m_max_name_length      = max_name_length;
	m_capacity             = static_cast<uint32_t>(capacity);
	m_default_owner        = 1;
	m_next_owner_handle    = 2;
	m_next_resource_handle = 1;
	m_owners               = {{m_default_owner, "Default"}};
	m_resources.clear();
	return true;
}

void Registry::Reset() {
	std::lock_guard lock(m_mutex);
	m_initialized          = false;
	m_max_name_length      = 0;
	m_capacity             = 0;
	m_default_owner        = InvalidOwnerHandle;
	m_next_owner_handle    = 1;
	m_next_resource_handle = 1;
	m_owners.clear();
	m_resources.clear();
}

bool Registry::HasOwner(OwnerHandle owner_handle) const {
	return std::any_of(m_owners.begin(), m_owners.end(),
	                   [owner_handle](const auto& owner) { return owner.handle == owner_handle; });
}

bool Registry::ValidName(const char* name) const {
	return name != nullptr && std::strlen(name) < m_max_name_length;
}

bool Registry::AtCapacity() const {
	return m_owners.size() + m_resources.size() >= m_capacity;
}

bool Registry::GetDefaultOwner(OwnerHandle* owner_handle) const {
	if (owner_handle == nullptr) {
		return false;
	}
	std::lock_guard lock(m_mutex);
	if (!m_initialized) {
		return false;
	}
	*owner_handle = m_default_owner;
	return true;
}

bool Registry::GetMaxNameLength(uint32_t* max_name_length) const {
	if (max_name_length == nullptr) {
		return false;
	}
	std::lock_guard lock(m_mutex);
	if (!m_initialized) {
		return false;
	}
	*max_name_length = m_max_name_length;
	return true;
}

bool Registry::RegisterOwner(OwnerHandle* owner_handle, const char* name) {
	if (owner_handle == nullptr) {
		return false;
	}
	std::lock_guard lock(m_mutex);
	if (!m_initialized || !ValidName(name) || AtCapacity() ||
	    m_next_owner_handle == InvalidOwnerHandle) {
		return false;
	}
	const auto handle = m_next_owner_handle++;
	m_owners.push_back({handle, name});
	*owner_handle = handle;
	return true;
}

bool Registry::RegisterResource(ResourceHandle* resource_handle, OwnerHandle owner_handle,
                                const void* memory, size_t size_bytes, const char* name,
                                uint32_t type, uint64_t user_data) {
	std::lock_guard lock(m_mutex);
	if (!m_initialized || !HasOwner(owner_handle) || memory == nullptr || size_bytes == 0 ||
	    !ValidName(name) || AtCapacity() || m_next_resource_handle == InvalidResourceHandle) {
		return false;
	}
	const auto handle = m_next_resource_handle++;
	m_resources.push_back({handle, owner_handle, reinterpret_cast<uint64_t>(memory), size_bytes,
	                       type, user_data, name});
	if (resource_handle != nullptr) {
		*resource_handle = handle;
	}
	return true;
}

bool Registry::UnregisterOwnerAndResources(OwnerHandle owner_handle) {
	std::lock_guard lock(m_mutex);
	const auto      owner =
	    std::find_if(m_owners.begin(), m_owners.end(),
	                 [owner_handle](const auto& o) { return o.handle == owner_handle; });
	if (!m_initialized || owner == m_owners.end() || owner_handle == m_default_owner) {
		return false;
	}
	m_resources.erase(std::remove_if(m_resources.begin(), m_resources.end(),
	                                 [owner_handle](const auto& resource) {
		                                 return resource.owner_handle == owner_handle;
	                                 }),
	                  m_resources.end());
	m_owners.erase(owner);
	return true;
}

bool Registry::UnregisterResource(ResourceHandle resource_handle) {
	std::lock_guard lock(m_mutex);
	const auto      resource =
	    std::find_if(m_resources.begin(), m_resources.end(), [resource_handle](const auto& value) {
		    return value.handle == resource_handle;
	    });
	if (!m_initialized || resource == m_resources.end()) {
		return false;
	}
	m_resources.erase(resource);
	return true;
}

Snapshot Registry::GetSnapshot() const {
	std::lock_guard lock(m_mutex);
	Snapshot        snapshot;
	snapshot.initialized     = m_initialized;
	snapshot.max_name_length = m_max_name_length;
	snapshot.owners          = m_owners;
	snapshot.resources       = m_resources;
	std::sort(snapshot.owners.begin(), snapshot.owners.end(),
	          [](const auto& lhs, const auto& rhs) { return lhs.handle < rhs.handle; });
	std::sort(snapshot.resources.begin(), snapshot.resources.end(),
	          [](const auto& lhs, const auto& rhs) { return lhs.handle < rhs.handle; });
	return snapshot;
}

Registry& GetRegistry() {
	static Registry registry;
	return registry;
}

const char* ResourceTypeName(uint32_t type) noexcept {
	switch (type) {
		case 0: return "invalid";
		case 1: return "shader";
		case 3: return "texture";
		case 4: return "buffer";
		case 5: return "render_target";
		case 6: return "render_target_cmask";
		case 7: return "render_target_fmask";
		case 8: return "depth_target";
		case 9: return "depth_target_stencil";
		case 10: return "depth_target_htile";
		case 11: return "constant_buffer";
		case 12: return "vertex_buffer";
		case 13: return "index_buffer";
		case 14: return "draw_command_buffer";
		case 15: return "async_command_buffer";
		case 17: return "label";
		case 18: return "generic_buffer";
		case 19: return "render_target_dcc";
		case 20: return "vertex_attributes";
		case 21: return "indirect_state";
		case 22: return "indirect_draw";
		case 23: return "indirect_dispatch";
		case 24: return "shader_resource_table";
		case 25: return "shader_user_data";
		case 26: return "shader_extended_user_data";
		case 27: return "command_buffer_branch_condition";
		case 28: return "texture_metadata";
		case 29: return "top_level_bvh";
		case 30: return "bottom_level_bvh";
		default: return "unknown";
	}
}

} // namespace Libs::Graphics::ResourceRegistration
