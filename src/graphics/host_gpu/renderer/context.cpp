#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vma.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstring>
namespace Libs::Graphics {

struct OcclusionQueryScope {
	std::atomic<uint64_t> samples {0};
	std::atomic<uint32_t> pending_segments {0};
	std::atomic<uint64_t> end_address {0};
	std::atomic_bool      closed {false};
	std::atomic_bool      conservative_visible {false};
	std::atomic_bool      published {false};
};

namespace {

constexpr uint64_t OcclusionReadyBit = 1ull << 63u;
constexpr uint64_t OcclusionCounterMask = OcclusionReadyBit - 1u;

void WriteOcclusionCounters(uint64_t address, uint64_t samples) {
	if (address == 0 || (address & 0x7u) != 0) {
		return;
	}

	samples &= OcclusionCounterMask;
	const auto per_db   = samples / 16u;
	const auto remainder = samples % 16u;
	for (uint32_t db = 0; db < 16u; db++) {
		const auto value = OcclusionReadyBit | per_db | (db < remainder ? 1u : 0u);
		if (!LibKernel::Memory::TryWriteBacking(address + static_cast<uint64_t>(db) * 16u,
		                                        &value, sizeof(value))) {
			static std::atomic_bool warning_once {false};
			if (!warning_once.exchange(true, std::memory_order_relaxed)) {
				LOGF("occlusion query result address is not writable: 0x%016" PRIx64 "\n",
				     address);
			}
			return;
		}
	}
}

void TryPublishOcclusionScope(const std::shared_ptr<OcclusionQueryScope>& scope) {
	if (!scope || !scope->closed.load(std::memory_order_acquire) ||
	    scope->pending_segments.load(std::memory_order_acquire) != 0u ||
	    scope->published.exchange(true, std::memory_order_acq_rel)) {
		return;
	}

	auto samples = scope->samples.load(std::memory_order_relaxed);
	if (scope->conservative_visible.load(std::memory_order_relaxed)) {
		samples = std::max<uint64_t>(samples, 1u);
	}
	WriteOcclusionCounters(scope->end_address.load(std::memory_order_relaxed), samples);
}

void ReportVulkanFatal(const char* what, vk::Result result, uint32_t slot, uint64_t submit_seq,
                       uint32_t debug_op, uint64_t debug_submit, uint32_t arg0, uint32_t arg1,
                       uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), slot=%u submit_seq=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, VulkanToString(result).c_str(), static_cast<int>(result), slot, submit_seq, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), slot=%u submit_seq=%" PRIu64
	            " debug_op=%u debug_submit=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, VulkanToString(result).c_str(), static_cast<int>(result), slot, submit_seq,
	            debug_op, debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

} // namespace

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_scheduler(scheduler), m_graphics(scheduler.Graphics()),
      m_slot(scheduler.AllocateCommandBuffer()) {
	vk::QueryPoolCreateInfo create {};
	create.sType      = vk::StructureType::eQueryPoolCreateInfo;
	create.queryType  = vk::QueryType::eOcclusion;
	create.queryCount = OcclusionQueryCapacity;
	if (m_graphics.device.createQueryPool(&create, nullptr, &m_occlusion_query_pool) !=
	    vk::Result::eSuccess) {
		m_occlusion_query_pool = nullptr;
	}
}

CommandBuffer::~CommandBuffer() {
	Release();
	if (m_occlusion_query_pool != nullptr) {
		m_graphics.device.destroyQueryPool(m_occlusion_query_pool, nullptr);
		m_occlusion_query_pool = nullptr;
	}
}

bool CommandBuffer::IsInvalid() const {
	return m_slot == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());

	const auto handle = m_slot->buffer;
	EXIT_IF(handle == nullptr);
	return handle;
}

void CommandBuffer::Release() {
	EXIT_IF(IsInvalid());

	Common::LockGuard lock(*m_slot->pool_mutex);

	WaitForFence();

	m_slot->busy = false;
	m_slot->Reset();
	m_slot = nullptr;

	EXIT_NOT_IMPLEMENTED(!IsInvalid());
}

void CommandBuffer::Begin() const {
	EXIT_IF(m_rendering);
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.sType            = vk::StructureType::eCommandBufferBeginInfo;
	begin_info.pNext            = nullptr;
	begin_info.flags            = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	begin_info.pInheritanceInfo = nullptr;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	m_occlusion_query_cursor = 0;
	m_active_occlusion_query.reset();
	m_occlusion_scope.reset();
	if (m_occlusion_query_pool != nullptr) {
		buffer.resetQueryPool(m_occlusion_query_pool, 0, OcclusionQueryCapacity);
	}
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
}

void CommandBuffer::Execute(const SubmitInfo& submit) {
	EXIT_IF(IsInvalid());
	EXIT_IF(m_execute);
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores > SubmitInfo::MaxSemaphores);

	auto buffer = Handle();
	auto fence  = m_slot->fence;

	vk::TimelineSemaphoreSubmitInfo timeline_info {};
	timeline_info.sType                     = vk::StructureType::eTimelineSemaphoreSubmitInfo;
	timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
	timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
	timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
	timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

	vk::SubmitInfo submit_info {};
	submit_info.sType                = vk::StructureType::eSubmitInfo;
	submit_info.pNext                = &timeline_info;
	submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
	submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
	submit_info.pWaitDstStageMask    = submit.wait_stages.data();
	submit_info.commandBufferCount   = 1;
	submit_info.pCommandBuffers      = &buffer;
	submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
	submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

	auto& graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	auto result = graphics.device.resetFences(1, &fence);
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkResetFences (before submit)", result, m_slot->id, m_submit_seq,
		                  m_debug_op, m_debug_submit_id, m_debug_arg0, m_debug_arg1, m_debug_arg2,
		                  m_debug_arg3, m_debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	if (Config::GraphicsDebugDumpEnabled()) {
		LOGF("vkQueueSubmit begin: slot=%u waits=%u signals=%u debug_op=%u debug_submit=%" PRIu64
		     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
		     m_slot->id, submit.num_wait_semaphores, submit.num_signal_semaphores, m_debug_op,
		     m_debug_submit_id, m_debug_arg0, m_debug_arg1, m_debug_arg2, m_debug_arg3,
		     m_debug_arg4);
	}

	{
		Common::LockGuard lock(graphics.queue_mutex);
		m_submit_seq = m_scheduler.NextSubmitSequence();
		result       = graphics.queue.submit(1, &submit_info, fence);
	}

	m_execute      = true;
	m_fence_waited = false;

	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, m_slot->id, m_submit_seq, m_debug_op,
		                  m_debug_submit_id, m_debug_arg0, m_debug_arg1, m_debug_arg2, m_debug_arg3,
		                  m_debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::WaitForFence() {
	FinalizeFence(false);
}

void CommandBuffer::WaitForFenceOnly() {
	EXIT_IF(IsInvalid());
	if (!m_execute || m_fence_waited) {
		return;
	}
	auto       device        = m_graphics.device;
	const bool collect_stats = Config::GpuPerformanceMetricsEnabled();
	const auto wait_start = collect_stats ? Common::Timer::QueryPerformanceCounter() : uint64_t {0};
	auto       result     = device.waitForFences(1, &m_slot->fence, VK_TRUE, UINT64_MAX);
	if (collect_stats) {
		m_scheduler.RecordFenceWait(Common::Timer::QueryPerformanceCounter() - wait_start);
	}
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkWaitForFences", result, m_slot->id, m_submit_seq, m_debug_op,
		                  m_debug_submit_id, m_debug_arg0, m_debug_arg1, m_debug_arg2, m_debug_arg3,
		                  m_debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	m_fence_waited = true;
}

void CommandBuffer::WaitForFenceAndReset() {
	FinalizeFence(true);
}

void CommandBuffer::FinalizeFence(bool reset_recording) {
	const bool was_executed = m_execute;
	WaitForFenceOnly();
	if (was_executed) {
		m_execute      = false;
		m_fence_waited = false;
		if (reset_recording) {
			Common::LockGuard lock(*m_slot->pool_mutex);
			m_slot->Reset();
		}
	}
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	if (m_rendering && m_render_state == state) {
		return;
	}
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].sType        = vk::StructureType::eRenderingAttachmentInfo;
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.sType       = vk::StructureType::eRenderingAttachmentInfo;
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.sType       = vk::StructureType::eRenderingAttachmentInfo;
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.sType                = vk::StructureType::eRenderingInfo;
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	m_render_state = state;
	m_rendering    = true;
	BeginOcclusionSegment();
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	auto query = m_active_occlusion_query;
	auto scope = m_occlusion_scope;
	if (query.has_value()) {
		Handle().endQuery(m_occlusion_query_pool, *query);
		m_active_occlusion_query.reset();
	}
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
	if (query.has_value() && scope) {
		CompleteOcclusionSegment(*query, scope);
	}
}

void CommandBuffer::BeginOcclusionQuery(uint64_t begin_address) {
	if (m_occlusion_scope) {
		m_occlusion_scope->conservative_visible.store(true, std::memory_order_relaxed);
		m_occlusion_scope->end_address.store(begin_address + 8u, std::memory_order_relaxed);
		m_occlusion_scope->closed.store(true, std::memory_order_release);
		TryPublishOcclusionScope(m_occlusion_scope);
	}

	WriteOcclusionCounters(begin_address, 0);
	m_occlusion_scope = std::make_shared<OcclusionQueryScope>();
	if (m_rendering) {
		BeginOcclusionSegment();
	}
}

void CommandBuffer::EndOcclusionQuery(uint64_t end_address) {
	if (!m_occlusion_scope) {
		WriteOcclusionCounters(end_address, 1);
		return;
	}

	// Occlusion queries must end inside dynamic rendering; copying their result must occur after
	// rendering has ended.
	EndRendering();
	auto scope = std::move(m_occlusion_scope);
	scope->end_address.store(end_address, std::memory_order_relaxed);
	scope->closed.store(true, std::memory_order_release);
	TryPublishOcclusionScope(scope);
}

void CommandBuffer::MarkOcclusionConservativeVisible() const {
	if (m_occlusion_scope) {
		m_occlusion_scope->conservative_visible.store(true, std::memory_order_relaxed);
	}
}

void CommandBuffer::BeginOcclusionSegment() const {
	if (!m_occlusion_scope || m_active_occlusion_query.has_value()) {
		return;
	}
	if (m_occlusion_query_pool == nullptr ||
	    m_occlusion_query_cursor >= OcclusionQueryCapacity) {
		m_occlusion_scope->conservative_visible.store(true, std::memory_order_relaxed);
		return;
	}

	const auto query = m_occlusion_query_cursor++;
	Handle().beginQuery(m_occlusion_query_pool, query, {});
	m_active_occlusion_query = query;
}

void CommandBuffer::CompleteOcclusionSegment(
	uint32_t query, const std::shared_ptr<OcclusionQueryScope>& scope) const {
	auto& download = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Download);
	auto [mapped, offset] = download.Map(sizeof(uint64_t), alignof(uint64_t), false);
	if (mapped == nullptr) {
		scope->conservative_visible.store(true, std::memory_order_relaxed);
		return;
	}
	download.Commit();
	scope->pending_segments.fetch_add(1u, std::memory_order_relaxed);

	Handle().copyQueryPoolResults(
	    m_occlusion_query_pool, query, 1u, download.Handle(), offset, sizeof(uint64_t),
	    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
	vk::BufferMemoryBarrier barrier {};
	barrier.sType               = vk::StructureType::eBufferMemoryBarrier;
	barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download.Handle();
	barrier.offset              = offset;
	barrier.size                = sizeof(uint64_t);
	Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                         vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &barrier, 0,
	                         nullptr);

	m_scheduler.DeferPriorityOperation([scope, &download, mapped, offset] {
		download.Invalidate(offset, sizeof(uint64_t));
		uint64_t samples = 0;
		std::memcpy(&samples, mapped, sizeof(samples));
		scope->samples.fetch_add(samples & OcclusionCounterMask, std::memory_order_relaxed);
		if (scope->pending_segments.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
			TryPublishOcclusionScope(scope);
		}
	});
}

} // namespace Libs::Graphics
