#include "graphics/guest_gpu/pm4.h"
#include "graphics/guest_gpu/pm4Inspector.h"
#include "graphics/guest_gpu/resourceRegistry.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

namespace {

namespace Pm4 = Libs::Graphics::Pm4;

using Libs::Graphics::Pm4::InspectionStatus;
using Libs::Graphics::Pm4::InspectSubmission;
using Libs::Graphics::Pm4::MemoryReader;
using Libs::Graphics::Pm4::QueueRegisterState;
using Libs::Graphics::Pm4::RegisterSpace;
using Libs::Graphics::Pm4::SerializeSubmissionInspection;
using Libs::Graphics::Pm4::SubmissionMetadata;
namespace ResourceRegistration = Libs::Graphics::ResourceRegistration;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "Pm4InspectorTests: failed: %s\n", message);
    std::abort();
  }
}

SubmissionMetadata Metadata(bool reset = false) {
  SubmissionMetadata metadata;
  metadata.capture_id = 17;
  metadata.frame = 3;
  metadata.queue = 0;
  metadata.reset_state = reset;
  return metadata;
}

void TestDirectRegistersAndStatePersistence() {
  QueueRegisterState state;
  const std::vector<uint32_t> first = {
      KYTY_PM4(4, Libs::Graphics::Pm4::IT_SET_CONTEXT_REG, 0),
      0x70000020u,
      0x11223344u,
      0x55667788u,
  };
  auto result = InspectSubmission(Metadata(true), "dcb", 0x1000, first, &state);
  Check(result.buffers.size() == 1 && result.buffers[0].packets.size() == 1,
        "direct register packet was not captured");
  Check(result.buffers[0].packets[0].status == InspectionStatus::Known,
        "direct register packet was not classified as known");
  Check(state.Get(RegisterSpace::Context, 0x20).value_or(0) == 0x11223344u &&
            state.Get(RegisterSpace::Context, 0x21).value_or(0) == 0x55667788u,
        "direct context writes were not normalized or persisted");

  const std::vector<uint32_t> second = {
      KYTY_PM4(3, Libs::Graphics::Pm4::IT_SET_SH_REG, 0), 7, 0xaabbccddu};
  result = InspectSubmission(Metadata(false), "dcb", 0x2000, second, &state);
  Check(result.state_at_submit.context_registers == 2,
        "queue state did not persist between submissions");
  Check(state.Get(RegisterSpace::Shader, 7).value_or(0) == 0xaabbccddu,
        "shader register write was not tracked");

  const std::vector<uint32_t> empty;
  result = InspectSubmission(Metadata(true), "dcb", 0x3000, empty, &state);
  Check(result.state_before_reset.context_registers == 2 &&
            result.state_before_reset.shader_registers == 1,
        "state before reset was not reported");
  Check(result.state_at_submit.context_registers == 0 &&
            result.state_after.shader_registers == 0,
        "submission reset did not clear tracked queue state");
}

void TestIndirectCommandBufferSnapshot() {
  constexpr uint64_t nested_address = 0x00400000u;
  const std::vector<uint32_t> nested = {
      KYTY_PM4(3, Libs::Graphics::Pm4::IT_SET_SH_REG, 0), 4, 0x12345678u};
  const MemoryReader reader = [&nested](uint64_t address, uint32_t size_dw,
                                        std::vector<uint32_t> *words) {
    if (address != nested_address || size_dw != nested.size()) {
      return false;
    }
    *words = nested;
    return true;
  };
  const std::vector<uint32_t> root = {
      KYTY_PM4(4, Libs::Graphics::Pm4::IT_INDIRECT_BUFFER, 0),
      static_cast<uint32_t>(nested_address),
      static_cast<uint32_t>(nested_address >> 32u),
      static_cast<uint32_t>(nested.size()),
  };
  QueueRegisterState state;
  auto result =
      InspectSubmission(Metadata(true), "dcb", 0x1000, root, &state, reader);
  Check(result.buffers.size() == 2,
        "nested indirect command buffer was not copied");
  const auto &reference = result.buffers[0].packets[0].references[0];
  Check(reference.snapshot_status == InspectionStatus::Known &&
            reference.nested_buffer_id.value_or(0) == 1,
        "nested buffer reference was not linked to its snapshot");
  Check(state.Get(RegisterSpace::Shader, 4).value_or(0) == 0x12345678u,
        "nested command buffer did not update tracked state");
}

void TestIndirectRegisterSnapshot() {
  constexpr uint64_t pairs_address = 0x00500000u;
  const std::vector<uint32_t> pairs = {0x70000010u, 0x10101010u, 0x70000011u,
                                       0x20202020u};
  const MemoryReader reader = [&pairs](uint64_t address, uint32_t size_dw,
                                       std::vector<uint32_t> *words) {
    if (address != pairs_address || size_dw != pairs.size()) {
      return false;
    }
    *words = pairs;
    return true;
  };
  const std::vector<uint32_t> root = {
      KYTY_PM4(5, Libs::Graphics::Pm4::IT_SET_CONTEXT_REG_INDIRECT, 0),
      static_cast<uint32_t>(pairs_address),
      static_cast<uint32_t>(pairs_address >> 32u),
      0,
      2,
  };
  QueueRegisterState state;
  const auto result =
      InspectSubmission(Metadata(true), "dcb", 0x1000, root, &state, reader);
  Check(result.buffers[0].packets[0].register_writes.size() == 2,
        "indirect register pairs were not decoded");
  Check(state.Get(RegisterSpace::Context, 0x10).value_or(0) == 0x10101010u &&
            state.Get(RegisterSpace::Context, 0x11).value_or(0) == 0x20202020u,
        "indirect register state was not normalized");
}

void TestMalformedAndUnknownPacketsRemainDiagnostic() {
  QueueRegisterState state;
  const std::vector<uint32_t> truncated = {
      KYTY_PM4(4, Libs::Graphics::Pm4::IT_SET_CONTEXT_REG, 0), 0x20};
  auto result = InspectSubmission(Metadata(), "dcb", 0x1000, truncated, &state);
  Check(result.buffers[0].packets[0].status ==
            InspectionStatus::UnreadableAtCaptureTime,
        "truncated packet was not marked unreadable");

  const std::vector<uint32_t> unknown = {KYTY_PM4(2, 0xfe, 0), 0};
  result = InspectSubmission(Metadata(), "dcb", 0x2000, unknown, &state);
  Check(result.buffers[0].packets[0].status == InspectionStatus::Unknown,
        "unknown Type-3 opcode was misclassified");

  const std::vector<uint32_t> unsupported = {0x00000000u, 0};
  result = InspectSubmission(Metadata(), "dcb", 0x3000, unsupported, &state);
  Check(result.buffers[0].packets[0].status == InspectionStatus::Unsupported,
        "unsupported packet type was misclassified");
}

void TestResourceRegistryAndAnnotations() {
  ResourceRegistration::Registry registry;
  size_t required = 0;
  Check(
      ResourceRegistration::Registry::QueryMemoryRequirement(&required, 8, 32),
      "resource-registration memory requirement failed");
  std::vector<uint8_t> memory(required);
  Check(registry.Initialize(memory.data(), memory.size(), 32),
        "resource registry did not initialize");

  uint32_t owner = 0;
  Check(registry.RegisterOwner(&owner, "Renderer"),
        "resource owner was not registered");
  uint32_t resource = 0;
  Check(registry.RegisterResource(&resource, owner,
                                  reinterpret_cast<const void *>(0x1000), 0x100,
                                  "Main DCB", 14, 0x55),
        "resource was not registered");

  QueueRegisterState state;
  const std::vector<uint32_t> commands = {0x80000000u};
  const auto result = InspectSubmission(Metadata(), "dcb", 0x1040, commands,
                                        &state, {}, {}, registry.GetSnapshot());
  Check(result.buffers[0].resources.size() == 1 &&
            result.buffers[0].resources[0].handle == resource &&
            result.buffers[0].resources[0].offset_bytes == 0x40,
        "command buffer was not resolved through the registration map");
  Check(registry.UnregisterResource(resource) &&
            registry.UnregisterOwnerAndResources(owner),
        "registered resource or owner was not removed");
}

void TestDescriptorCapture() {
  constexpr uint64_t descriptor_address = 0x00600000u;
  const std::vector<uint32_t> descriptor(8, 0);
  const MemoryReader reader = [&descriptor](uint64_t address, uint32_t size_dw,
                                            std::vector<uint32_t> *words) {
    if (address != descriptor_address || size_dw != descriptor.size()) {
      return false;
    }
    *words = descriptor;
    return true;
  };
  const std::vector<uint32_t> commands = {
      KYTY_PM4(4, Libs::Graphics::Pm4::IT_SET_SH_REG, 0),
      Libs::Graphics::Pm4::SPI_SHADER_USER_DATA_PS_0,
      static_cast<uint32_t>(descriptor_address),
      static_cast<uint32_t>(descriptor_address >> 32u),
  };
  QueueRegisterState state;
  const auto result = InspectSubmission(Metadata(true), "dcb", 0x1000, commands,
                                        &state, reader);
  Check(result.descriptors.size() == 1 &&
            result.descriptors[0].stage == "pixel" &&
            result.descriptors[0].kind == "buffer" &&
            result.descriptors[0].words == descriptor,
        "shader user-data descriptor was not captured and classified");
}

void TestAutomaticComparison() {
  QueueRegisterState first_state;
  QueueRegisterState second_state;
  const std::vector<uint32_t> first = {0x80000000u};
  const std::vector<uint32_t> second = {KYTY_PM4(2, 0xfe, 0), 0};
  auto first_metadata = Metadata();
  auto second_metadata = Metadata();
  second_metadata.capture_id = 18;
  const auto previous =
      InspectSubmission(first_metadata, "dcb", 0x1000, first, &first_state);
  const auto current =
      InspectSubmission(second_metadata, "dcb", 0x2000, second, &second_state);
  const auto comparison =
      Libs::Graphics::Pm4::CompareSubmissionInspections(previous, current);
  Check(comparison.previous_capture_id == 17 &&
            comparison.added_packets.size() == 1 &&
            comparison.removed_packets.size() == 1,
        "trace comparison did not report changed packet signatures");
}

void TestJsonSchema() {
  QueueRegisterState state;
  const std::vector<uint32_t> commands = {0x80000000u};
  const auto result =
      InspectSubmission(Metadata(true), "dcb", 0x1234, commands, &state);
  const auto json =
      nlohmann::json::parse(SerializeSubmissionInspection(result));
  Check(json.at("schema") == "kyty.pm4.submission.v2",
        "JSON schema identifier is missing");
  Check(json.at("capture").at("queue_kind") == "graphics",
        "JSON queue kind is incorrect");
  Check(json.at("buffers").at(0).at("packets").at(0).at("status") == "Known",
        "JSON packet classification is incorrect");
  Check(!json.at("resource_registration").at("initialized") &&
            json.at("descriptors").is_array(),
        "JSON does not expose resource and descriptor diagnostics");
}

} // namespace

int main() {
  TestDirectRegistersAndStatePersistence();
  TestIndirectCommandBufferSnapshot();
  TestIndirectRegisterSnapshot();
  TestMalformedAndUnknownPacketsRemainDiagnostic();
  TestResourceRegistryAndAnnotations();
  TestDescriptorCapture();
  TestAutomaticComparison();
  TestJsonSchema();
  return 0;
}
