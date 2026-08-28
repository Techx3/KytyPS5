#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/ir/ValueProgram.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/ShaderInfoCollection.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderComputeInputInfo;
using Libs::Graphics::ShaderType;
namespace CFG = Libs::Graphics::ShaderRecompiler::CFG;
namespace Decoder = Libs::Graphics::ShaderRecompiler::Decoder;

void Check(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct Fixture {
  Program program;
  Block *block = nullptr;

  explicit Fixture(ShaderType stage = ShaderType::Compute) {
    program.stage = stage;
    program.values = std::make_shared<ValueProgram>();
    program.user_data_count = 64;
    block = AddBlock();
  }

  Block *AddBlock() {
    auto storage = std::make_unique<Block>();
    auto *result = storage.get();
    program.values->block_storage.push_back(std::move(storage));
    program.values->blocks.push_back(result);
    program.values->block_info.push_back(
        {.id = static_cast<uint32_t>(program.values->block_info.size())});
    return result;
  }

  Value Emit(ValueOpcode opcode, std::initializer_list<Value> args = {},
             uint64_t flags = 0, Block *destination = nullptr) {
    if (NumArgsOf(opcode) != std::numeric_limits<size_t>::max() &&
        NumArgsOf(opcode) != args.size()) {
      throw std::runtime_error(std::string(ValueOpcodeName(opcode)) +
                               " argument count");
    }
    auto &inst = (destination != nullptr ? destination : block)
                     ->AppendNewInst(opcode, args, flags);
    return Value(&inst);
  }

  template <typename T>
  Value Emit(ValueOpcode opcode, std::initializer_list<Value> args, T flags,
             Block *destination = nullptr) {
    uint64_t bits = 0;
    std::memcpy(&bits, &flags, sizeof(flags));
    return Emit(opcode, args, bits, destination);
  }

  Value UserData(uint32_t index) {
    return Emit(ValueOpcode::GetUserData,
                {Value(static_cast<ScalarReg>(index))});
  }

  MemoryFlags AddMemory(MemoryInfo memory, uint32_t pc) {
    const auto index =
        static_cast<uint32_t>(program.values->memory_info.size());
    program.values->memory_info.push_back(memory);
    return {index, pc};
  }

  Value Buffer(std::array<Value, 4> dwords, uint32_t pc = 0) {
    return Emit(ValueOpcode::GetBufferResource,
                {dwords[0], dwords[1], dwords[2], dwords[3]},
                MemoryFlags{0, pc});
  }

  Value Address(Value low, Value high, uint32_t pc = 0) {
    return Emit(ValueOpcode::GetAddressResource, {low, high},
                MemoryFlags{0, pc});
  }

  Value Image(std::array<Value, 8> dwords, uint32_t pc = 0) {
    return Emit(ValueOpcode::GetImageResource,
                {dwords[0], dwords[1], dwords[2], dwords[3], dwords[4],
                 dwords[5], dwords[6], dwords[7]},
                MemoryFlags{0, pc});
  }

  Value Sampler(std::array<Value, 4> dwords, uint32_t pc = 0) {
    return Emit(ValueOpcode::GetSamplerResource,
                {dwords[0], dwords[1], dwords[2], dwords[3]},
                MemoryFlags{0, pc});
  }

  Value ImageAddress() {
    return Emit(ValueOpcode::MakeImageAddress,
                {Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
                 Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
                 Value(0u), Value(0u), Value(0u)});
  }

  void PlanAndTrack() {
    std::string error;
    if (!BuildSrtPlan(program, &error) || !TrackResources(program, &error)) {
      throw std::runtime_error(error);
    }
  }
};

struct TestMemory {
  uint64_t base = 0x1000;
  std::array<uint32_t, 8> words{};
  uint32_t reads = 0;
  uint32_t fail_after = UINT32_MAX;
};

bool ReadTestMemory(void *userdata, uint64_t address, uint32_t *value) {
  auto *memory = static_cast<TestMemory *>(userdata);
  if (memory == nullptr || value == nullptr || address < memory->base ||
      address - memory->base >= memory->words.size() * sizeof(uint32_t) ||
      memory->reads >= memory->fail_after) {
    return false;
  }
  *value = memory->words[(address - memory->base) / sizeof(uint32_t)];
  memory->reads++;
  return true;
}

struct LinearTestMemory {
  uint64_t base = 0x1000;
  std::vector<uint32_t> words = std::vector<uint32_t>(0x2200 / 4);
  uint64_t fail_address = UINT64_MAX;
};

bool ReadLinearTestMemory(void *userdata, uint64_t address, uint32_t *value) {
  auto *memory = static_cast<LinearTestMemory *>(userdata);
  if (memory == nullptr || value == nullptr || address < memory->base ||
      address - memory->base >= memory->words.size() * sizeof(uint32_t) ||
      (address & 3u) != 0u || address == memory->fail_address) {
    return false;
  }
  *value = memory->words[(address - memory->base) / sizeof(uint32_t)];
  return true;
}

std::unique_ptr<Fixture>
MakeIndirectImageFixture(bool malformed, uint32_t material_immediate = 0,
                         bool memory_backed_material = false) {
  auto fixture = std::make_unique<Fixture>();
  std::array<Value, 4> material_words;
  std::array<Value, 4> heap_words;
  for (uint32_t dword = 0; dword < 4; dword++) {
    material_words[dword] = fixture->UserData(dword);
    heap_words[dword] = fixture->UserData(dword + 4u);
  }
  if (memory_backed_material) {
    const auto pointer_address =
        fixture->Address(fixture->UserData(9), fixture->UserData(10), 0x10b0);
    MemoryInfo pointer_word;
    pointer_word.kind = ResourceKind::ScalarAddress;
    const auto pointer =
        fixture->Emit(ValueOpcode::LoadAddressU32,
                      {pointer_address, Value(0u), Value(0u), Value(true)},
                      fixture->AddMemory(pointer_word, 0x10b0));
    const auto address = fixture->Address(pointer, Value(0u), 0x10c0);
    MemoryInfo descriptor_word;
    descriptor_word.kind = ResourceKind::ScalarAddress;
    material_words[0] =
        fixture->Emit(ValueOpcode::LoadAddressU32,
                      {address, Value(0u), Value(0u), Value(true)},
                      fixture->AddMemory(descriptor_word, 0x10c0));
  }
  const auto material = fixture->Buffer(material_words, 0x10d8);
  const auto heap = fixture->Buffer(heap_words, 0x10d8);
  if (memory_backed_material) {
    MemoryInfo shared_buffer;
    shared_buffer.kind = ResourceKind::Buffer;
    const auto load =
        fixture->Emit(ValueOpcode::LoadBufferU32,
                      {material, Value(0u), Value(0u), Value(0u), Value(true)},
                      fixture->AddMemory(shared_buffer, 0x10d8));
    fixture->Emit(ValueOpcode::ReferenceU32, {load});
  }
  const auto selector = fixture->Emit(ValueOpcode::ReadFirstLane,
                                      {fixture->UserData(8), Value(true)});
  const auto record =
      fixture->Emit(ValueOpcode::IMul32, {selector, Value(224u)});
  const auto member = fixture->Emit(ValueOpcode::IAdd32, {record, Value(4u)});
  fixture->Emit(ValueOpcode::ReferenceU32, {record});
  fixture->Emit(ValueOpcode::ReferenceU32, {member});
  MemoryInfo material_scalar;
  material_scalar.kind = ResourceKind::ScalarBuffer;
  material_scalar.offset = material_immediate;
  const auto key =
      fixture->Emit(ValueOpcode::ReadConstBuffer, {material, member},
                    fixture->AddMemory(material_scalar, 0x10d8));
  const auto heap_offset =
      fixture->Emit(ValueOpcode::ShiftLeftLogical32, {key, Value(5u)});
  std::array<Value, 8> image_words;
  MemoryInfo heap_scalar;
  heap_scalar.kind = ResourceKind::ScalarBuffer;
  for (uint32_t dword = 0; dword < image_words.size(); dword++) {
    auto component = heap_scalar;
    component.offset = dword * sizeof(uint32_t);
    if (malformed && dword == image_words.size() - 1u) {
      component.offset += sizeof(uint32_t);
    }
    image_words[dword] =
        fixture->Emit(ValueOpcode::ReadConstBuffer, {heap, heap_offset},
                      fixture->AddMemory(component, 0x10d8));
  }
  const auto image = fixture->Image(image_words, 0x10f0);
  const auto sampler =
      fixture->Sampler({Value(0u), Value(0u), Value(0u), Value(0u)}, 0x10f0);
  MemoryInfo sample;
  sample.kind = ResourceKind::Image;
  sample.image_dimension = Decoder::ImageDimension::Dim2D;
  const auto sampled = fixture->Emit(ValueOpcode::ImageSampleRaw,
                                     {image, sampler, fixture->ImageAddress()},
                                     fixture->AddMemory(sample, 0x10f0));
  const auto sampled_x =
      fixture->Emit(ValueOpcode::CompositeExtractU32x4, {sampled, Value(0u)});
  fixture->Emit(ValueOpcode::ReferenceU32, {sampled_x});
  return fixture;
}

std::unique_ptr<Fixture> MakeDirectIndirectImageFixture(bool read_lane_selector = false,
                                                        bool malformed_read_lane = false,
                                                        bool material_key_table = false) {
  auto fixture = std::make_unique<Fixture>();
  auto *alternate = fixture->AddBlock();
  auto *sample = fixture->AddBlock();
  fixture->block->AddBranch(sample);
  alternate->AddBranch(sample);

  const auto low = fixture->Emit(ValueOpcode::BitwiseAnd32,
                                 {fixture->UserData(2), Value(0xffffu)});
  const auto high = fixture->Emit(ValueOpcode::BitwiseAnd32,
                                  {fixture->UserData(3), Value(0xffffu)}, 0,
                                  alternate);
  auto &mask = sample->AppendNewInst(ValueOpcode::Phi, {},
                                     static_cast<uint64_t>(Type::U32));
  mask.AddPhiOperand(fixture->block, low);
  mask.AddPhiOperand(alternate, high);
  const auto table = fixture->Emit(
      ValueOpcode::GetAddressResource,
      {fixture->UserData(0), fixture->UserData(1)}, MemoryFlags{0, 0x104},
      sample);
  Value selector;
  if (read_lane_selector) {
    Value material = fixture->UserData(4);
    if (material_key_table) {
      const auto active = fixture->Emit(
          ValueOpcode::IEqual32, {fixture->UserData(7), Value(1u)}, 0, sample);
      const auto scaled16 = fixture->Emit(
          ValueOpcode::ShiftLeftLogical32,
          {fixture->UserData(6), Value(4u)}, 0, sample);
      const auto selected16 = fixture->Emit(
          ValueOpcode::SelectU32,
          {active, scaled16, fixture->UserData(4)}, 0, sample);
      const auto scaled8 = fixture->Emit(
          ValueOpcode::ShiftLeftLogical32, {selected16, Value(3u)}, 0,
          sample);
      const auto record = fixture->Emit(
          ValueOpcode::IAdd32, {scaled8, selected16}, 0, sample);
      const auto selected_record = fixture->Emit(
          ValueOpcode::SelectU32, {active, record, selected16}, 0, sample);
      const auto based_record = fixture->Emit(
          ValueOpcode::IAdd32, {Value(0xc00u), selected_record}, 0, sample);
      const auto material_offset = fixture->Emit(
          ValueOpcode::SelectU32,
          {active, based_record, selected_record}, 0, sample);
      MemoryInfo material_memory;
      material_memory.kind = ResourceKind::ScalarAddress;
      material = fixture->Emit(
          ValueOpcode::LoadAddressU32,
          {table, material_offset, Value(0u), Value(true)},
          fixture->AddMemory(material_memory, 0x108u), sample);
    }
    const auto lane = fixture->Emit(
        ValueOpcode::BitwiseAnd32,
        {fixture->UserData(5), Value(malformed_read_lane ? 0xffu : 0x3fu)},
        0, sample);
    selector = fixture->Emit(ValueOpcode::ReadLane, {material, lane}, 0, sample);
    if (!malformed_read_lane) {
      const auto grouped = fixture->Emit(ValueOpcode::IEqual32,
                                         {selector, material}, 0, sample);
      fixture->Emit(ValueOpcode::Reference, {grouped}, 0, sample);
    }
  } else {
    selector = fixture->Emit(ValueOpcode::FindILsb32, {Value(&mask)}, 0, sample);
  }
  const auto scaled = fixture->Emit(ValueOpcode::ShiftLeftLogical32,
                                    {selector, Value(5u)}, 0, sample);
  const auto first = fixture->Emit(ValueOpcode::IAdd32,
                                   {scaled, Value(0x158u)}, 0, sample);
  const auto second = fixture->Emit(ValueOpcode::IAdd32,
                                    {Value(0x10u), first}, 0, sample);
  std::array<Value, 8> words;
  for (uint32_t dword = 0; dword < words.size(); dword++) {
    MemoryInfo memory;
    memory.kind = ResourceKind::ScalarAddress;
    memory.offset = (dword & 3u) * sizeof(uint32_t);
    words[dword] = fixture->Emit(
        ValueOpcode::LoadAddressU32,
        {table, dword < 4u ? first : second, Value(0u), Value(true)},
        fixture->AddMemory(memory, dword < 4u ? 0x104u : 0x10cu), sample);
  }
  const auto image = fixture->Emit(
      ValueOpcode::GetImageResource,
      {words[0], words[1], words[2], words[3], words[4], words[5], words[6],
       words[7]},
      MemoryFlags{0, 0x118}, sample);
  const auto sampler = fixture->Emit(
      ValueOpcode::GetSamplerResource,
      {Value(0u), Value(0u), Value(0u), Value(0u)}, MemoryFlags{0, 0x118},
      sample);
  const auto address = fixture->Emit(
      ValueOpcode::MakeImageAddress,
      {Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
       Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
       Value(0u)},
      0, sample);
  MemoryInfo memory;
  memory.kind = ResourceKind::Image;
  memory.image_dimension = Decoder::ImageDimension::Dim2DArray;
  const auto result = fixture->Emit(ValueOpcode::ImageSampleRaw,
                                    {image, sampler, address},
                                    fixture->AddMemory(memory, 0x118), sample);
  fixture->Emit(ValueOpcode::ReferenceU32,
                {fixture->Emit(ValueOpcode::CompositeExtractU32x4,
                               {result, Value(0u)}, 0, sample)},
                0, sample);
  return fixture;
}

void TestReadLaneDirectIndirectImageTracking() {
  auto fixture = MakeDirectIndirectImageFixture(true);
  fixture->program.wave_size = 64u;
  fixture->PlanAndTrack();
  const auto source_index = fixture->program.info.images[0].source;
  const auto &source = fixture->program.values->descriptor_sources[source_index];
  Check(source.indirect_image.has_value() &&
            source.indirect_image->direct_address &&
            source.indirect_image->entry_count == ShaderInfo::MaxImages &&
            fixture->program.values->dynamic_reads.empty(),
        "grouped ReadLane image selector was not lowered to a direct table");

  auto malformed = MakeDirectIndirectImageFixture(true, true);
  malformed->program.wave_size = 64u;
  std::string error;
  Check(BuildSrtPlan(malformed->program, &error) &&
            !TrackResources(malformed->program, &error) &&
            error.find("ReadLane") != std::string::npos,
        "unbounded or ungrouped ReadLane image selector was accepted");

  auto material_table = MakeDirectIndirectImageFixture(true, false, true);
  material_table->program.wave_size = 64u;
  material_table->PlanAndTrack();
  const auto material_source_index = material_table->program.info.images[0].source;
  const auto &material_source =
      material_table->program.values->descriptor_sources[material_source_index];
  Check(material_source.indirect_image.has_value() &&
            material_source.indirect_image->direct_key_stride == 144u &&
            material_source.indirect_image->direct_key_offset == 0xc00u &&
            material_source.indirect_image->direct_key_count == 64u,
        "wave64 material records were not retained as direct image keys");

}

void TestReadLaneDirectMaterialKeyMaterialization() {
  auto fixture = MakeDirectIndirectImageFixture(true, false, true);
  fixture->program.wave_size = 64u;
  fixture->PlanAndTrack();
  EliminateDeadCode(fixture->program.values->blocks);

  LinearTestMemory memory;
  memory.words.resize(0x3000u / sizeof(uint32_t));
  for (uint32_t entry = 0; entry < 64u; entry++) {
    memory.words[(0xc00u + entry * 144u) / 4u] = UINT32_MAX;
  }
  memory.words[0xc00u / 4u] = 5u;
  memory.words[(0xc00u + 144u) / 4u] = 17u;
  memory.words[(0xc00u + 288u) / 4u] = 63u;

  std::array<uint32_t, 8> descriptor{};
  descriptor[0] = 0x20u;
  descriptor[1] = static_cast<uint32_t>(
                      Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float)
                  << 20u;
  descriptor[2] = 3u | (3u << 14u);
  descriptor[3] =
      Libs::Graphics::DstSel(4, 5, 6, 7) |
      (static_cast<uint32_t>(
           Libs::Graphics::Prospero::ImageType::kColor2DArray)
       << 28u);
  for (const auto key : {5u, 17u, 63u}) {
    for (uint32_t dword = 0; dword < descriptor.size(); dword++) {
      memory.words[(0x158u + key * 32u) / 4u + dword] = descriptor[dword];
    }
  }
  memory.words[(0x158u + 17u * 32u) / 4u] ^= 1u;

  std::array<uint32_t, 8> user_data{
      static_cast<uint32_t>(memory.base), 0u, 1u, 2u, 5u, 0u, 0u, 1u};
  SrtRuntime runtime{.user_data = user_data,
                     .userdata = &memory,
                     .read_specialization_memory = ReadLinearTestMemory};
  ResourceSnapshot snapshot;
  std::string error;
  Check(MaterializeResources(fixture->program, runtime, snapshot, &error),
        error.empty() ? "direct material keys could not be read" : error.c_str());
  Check(snapshot.indirect_images.size() == 1u,
        "direct material keys did not produce an indirect image table");
  Check(snapshot.indirect_images[0].capacity == 64u,
        "direct material key capacity did not preserve wave64");
  Check(snapshot.indirect_images[0].keys ==
            std::vector<uint32_t>({5u, 17u, 63u}),
        "direct material key values were not preserved");
  Check(snapshot.indirect_images[0].descriptors.size() == 2u,
        "direct material descriptors were not deduplicated");
}

void TestDirectIndirectImageMaterialization() {
  auto fixture = MakeDirectIndirectImageFixture();
  fixture->PlanAndTrack();
  EliminateDeadCode(fixture->program.values->blocks);
  Check(fixture->program.info.images.size() == 1 &&
            fixture->program.values->dynamic_reads.empty(),
        "direct indirect image loads remained ordinary address resources");
  const auto source = fixture->program.info.images[0].source;
  Check(source < fixture->program.values->descriptor_sources.size() &&
            fixture->program.values->descriptor_sources[source]
                .indirect_image.has_value() &&
            fixture->program.values->descriptor_sources[source]
                .indirect_image->direct_address,
        "direct indirect image source was not retained");

  LinearTestMemory memory;
  std::array<uint32_t, 8> descriptor{};
  descriptor[0] = 0x20u;
  descriptor[1] = static_cast<uint32_t>(
                      Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float)
                  << 20u;
  descriptor[2] = 3u | (3u << 14u);
  descriptor[3] =
      Libs::Graphics::DstSel(4, 5, 6, 7) |
      (static_cast<uint32_t>(
           Libs::Graphics::Prospero::ImageType::kColor2DArray)
       << 28u);
  for (uint32_t entry = 0; entry < 32u; entry++) {
    for (uint32_t dword = 0; dword < descriptor.size(); dword++) {
      memory.words[(0x158u + entry * 32u) / 4u + dword] =
          descriptor[dword];
    }
  }
  memory.words[(0x158u + 32u) / 4u] ^= 1u;
  memory.words[(0x158u + 64u) / 4u + 4u] = 0x83900000u;
  memory.words[(0x158u + 96u) / 4u + 4u] = 1u << 13u;
  auto invalid_layout = descriptor;
  constexpr uint32_t captured_width = 14630u;
  constexpr uint32_t captured_height = 237u;
  const auto encoded_width = captured_width - 1u;
  invalid_layout[1] = (8u << 20u) | ((encoded_width & 3u) << 30u);
  invalid_layout[2] = ((encoded_width >> 2u) & 0xfffu) |
                      ((captured_height - 1u) << 14u);
  invalid_layout[3] = Libs::Graphics::DstSel(4, 5, 6, 7) | (28u << 20u) |
                      (static_cast<uint32_t>(
                           Libs::Graphics::Prospero::ImageType::kColor2D)
                       << 28u);
  invalid_layout[4] = 0u;
  invalid_layout[5] = 0x00700000u | (14u << 4u);
  invalid_layout[6] = 0u;
  invalid_layout[7] = 0u;
  for (uint32_t dword = 0; dword < invalid_layout.size(); dword++) {
    memory.words[(0x158u + 128u) / 4u + dword] = invalid_layout[dword];
  }
  constexpr std::array<uint32_t, 8> captured_float_data{
      0x3f352000u, 0xc121a9c6u, 0x40d09321u, 0xc097c45au,
      0x00000000u, 0xbf351708u, 0xbf2a653bu, 0xbe73915au};
  for (uint32_t dword = 0; dword < captured_float_data.size(); dword++) {
    memory.words[(0x158u + 160u) / 4u + dword] = captured_float_data[dword];
  }
  std::array<uint32_t, 4> user_data{static_cast<uint32_t>(memory.base), 0u, 1u,
                                    2u};
  SrtRuntime runtime{.user_data = user_data,
                     .userdata = &memory,
                     .read_specialization_memory = ReadLinearTestMemory};
  ResourceSnapshot snapshot;
  std::string error;
  Check(
      MaterializeResources(fixture->program, runtime, snapshot, &error) &&
          snapshot.indirect_images.size() == 1u &&
          snapshot.indirect_images[0].keys.size() == 32u &&
          snapshot.indirect_images[0].descriptors.size() == 4u &&
          snapshot.indirect_images[0].candidates[5] == 3u &&
          std::ranges::all_of(snapshot.indirect_images[0].descriptors[2].dwords,
                              [](uint32_t value) { return value == 0u; }) &&
          SpecializeResources(fixture->program, snapshot, &error) &&
          fixture->program.info.images.size() == 4u &&
          fixture->program.info.images[3].kind == ResourceKind::ImageUint &&
          fixture->program.info.images[3].dimension ==
              Decoder::ImageDimension::Dim1DArray,
      error.empty() ? "direct indirect image table did not materialize"
                    : error.c_str());
  ShaderComputeInputInfo compute{};
  Check(CollectShaderInfo(fixture->program, {.compute = &compute}, &error) &&
            AllocateBindings(fixture->program, 0, &error) &&
            FindBinding(fixture->program.bindings,
                        DescriptorBindingKind::Sampled2DArray) != nullptr &&
            FindBinding(fixture->program.bindings,
                        DescriptorBindingKind::SampledUint1DArray) != nullptr,
        error.empty()
            ? "captured mixed numeric image bindings were not allocated"
            : error.c_str());
}

void TestLoopIndexedDirectImageMaterialization() {
  Fixture fixture(ShaderType::Vertex);
  auto *entry = fixture.block;
  auto *header = fixture.AddBlock();
  auto *guard = fixture.AddBlock();
  auto *sample = fixture.AddBlock();
  auto *latch = fixture.AddBlock();
  auto *exit = fixture.AddBlock();
  entry->AddBranch(header);
  header->AddBranch(guard);
  guard->AddBranch(sample);
  guard->AddBranch(exit);
  sample->AddBranch(latch);
  latch->AddBranch(header);

  const auto table_low = fixture.UserData(0);
  const auto table_high = fixture.UserData(1);
  const auto runtime_count = fixture.UserData(2);
  auto &index = header->AppendNewInst(ValueOpcode::Phi, {},
                                      static_cast<uint64_t>(Type::U32));
  index.AddPhiOperand(entry, Value(0u));
  const auto increment = fixture.Emit(ValueOpcode::IAdd32,
                                      {Value(&index), Value(1u)}, 0, latch);
  index.AddPhiOperand(latch, increment);
  const auto condition = fixture.Emit(ValueOpcode::SLessThan32,
                                      {Value(&index), runtime_count}, 0, guard);

  auto &blocks = fixture.program.values->block_info;
  blocks[0].terminator.kind = CFG::TerminatorKind::Branch;
  blocks[0].terminator.true_block = blocks[1].id;
  blocks[1].terminator.kind = CFG::TerminatorKind::Branch;
  blocks[1].terminator.true_block = blocks[2].id;
  blocks[2].condition = condition;
  blocks[2].terminator.kind = CFG::TerminatorKind::ConditionalBranch;
  blocks[2].terminator.true_block = blocks[3].id;
  blocks[2].terminator.false_block = blocks[5].id;
  blocks[2].terminator.merge_block = blocks[5].id;
  blocks[3].terminator.kind = CFG::TerminatorKind::Branch;
  blocks[3].terminator.true_block = blocks[4].id;
  blocks[4].terminator.kind = CFG::TerminatorKind::Branch;
  blocks[4].terminator.true_block = blocks[1].id;
  blocks[5].terminator.kind = CFG::TerminatorKind::Return;

  const auto scaled = fixture.Emit(ValueOpcode::ShiftLeftLogical32,
                                   {Value(&index), Value(5u)}, 0, sample);
  const auto offset = fixture.Emit(ValueOpcode::IAdd32,
                                   {scaled, Value(0x6b0u)}, 0, sample);
  const auto table = fixture.Emit(ValueOpcode::GetAddressResource,
                                  {table_low, table_high},
                                  MemoryFlags{0, 0x284u}, sample);
  std::array<Value, 8> words;
  for (uint32_t dword = 0; dword < words.size(); dword++) {
    MemoryInfo memory;
    memory.kind = ResourceKind::ScalarAddress;
    memory.offset = dword * sizeof(uint32_t);
    words[dword] = fixture.Emit(
        ValueOpcode::LoadAddressU32,
        {table, offset, Value(0u), Value(true)},
        fixture.AddMemory(memory, 0x284u + dword * sizeof(uint32_t)), sample);
  }
  const auto image = fixture.Emit(
      ValueOpcode::GetImageResource,
      {words[0], words[1], words[2], words[3], words[4], words[5], words[6],
       words[7]},
      MemoryFlags{0, 0x29cu}, sample);
  const auto sampler = fixture.Emit(
      ValueOpcode::GetSamplerResource,
      {Value(0u), Value(0u), Value(0u), Value(0u)},
      MemoryFlags{0, 0x29cu}, sample);
  const auto address = fixture.Emit(
      ValueOpcode::MakeImageAddress,
      {Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
       Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
       Value(0u)},
      0, sample);
  MemoryInfo image_memory;
  image_memory.kind = ResourceKind::Image;
  image_memory.image_dimension = Decoder::ImageDimension::Dim2D;
  fixture.Emit(ValueOpcode::ImageSampleRaw, {image, sampler, address},
               fixture.AddMemory(image_memory, 0x29cu), sample);

  fixture.PlanAndTrack();
  const auto source_index = fixture.program.info.images[0].source;
  const auto &source = fixture.program.values->descriptor_sources[source_index];
  Check(source.indirect_image.has_value() &&
            source.indirect_image->direct_address &&
            source.indirect_image->entry_count == 0u &&
            source.indirect_image->entry_count_source !=
                DescriptorSource::IndirectImage::NoEntryCountSource &&
            fixture.program.values->dynamic_reads.empty(),
        "loop-indexed descriptor loads were not lowered to an indirect table");

  LinearTestMemory memory;
  std::array<uint32_t, 8> descriptor{};
  descriptor[0] = 0x20u;
  descriptor[1] =
      static_cast<uint32_t>(
          Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float)
      << 20u;
  descriptor[2] = 3u | (3u << 14u);
  descriptor[3] =
      Libs::Graphics::DstSel(4, 5, 6, 7) |
      (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D)
       << 28u);
  for (uint32_t entry_index = 0; entry_index < 3u; entry_index++) {
    for (uint32_t dword = 0; dword < descriptor.size(); dword++) {
      memory.words[(0x6b0u + entry_index * 32u) / 4u + dword] =
          descriptor[dword];
    }
  }
  memory.words[(0x6b0u + 32u) / 4u] ^= 1u;

  std::array<uint32_t, 3> user_data{
      static_cast<uint32_t>(memory.base), 0u, 3u};
  SrtRuntime runtime{.user_data = user_data,
                     .userdata = &memory,
                     .read_specialization_memory = ReadLinearTestMemory};
  ResourceSnapshot snapshot;
  std::string error;
  Check(MaterializeResources(fixture.program, runtime, snapshot, &error) &&
            snapshot.indirect_images.size() == 1u &&
            snapshot.indirect_images[0].capacity == 3u &&
            snapshot.indirect_images[0].keys ==
                std::vector<uint32_t>({0u, 1u, 2u}) &&
            snapshot.indirect_images[0].descriptors.size() == 2u,
        error.empty() ? "runtime loop bound did not size the descriptor table"
                      : error.c_str());

  user_data[2] = ShaderInfo::MaxImages + 1u;
  ResourceSnapshot rejected;
  Check(!MaterializeResources(fixture.program, runtime, rejected, &error) &&
            error.find("invalid layout") != std::string::npos,
        "an out-of-range loop-indexed descriptor table was silently accepted");
}

void TestInvariantIndirectImageMaterialization() {
  auto fixture = MakeIndirectImageFixture(false);
  fixture->PlanAndTrack();
  EliminateDeadCode(fixture->program.values->blocks);
  std::string validation_error;
  Check(
      ValidateValueProgram(*fixture->program.values, true, &validation_error),
      "post-tracking dead-code elimination invalidated descriptor provenance");

  Check(fixture->program.info.buffers.size() == 1 &&
            fixture->program.info.images.size() == 1 &&
            fixture->program.values->dynamic_reads.size() == 1,
        "indirect image key was not retained as a scalar-buffer read");
  const auto source = fixture->program.info.images[0].source;
  Check(source < fixture->program.values->descriptor_sources.size() &&
            fixture->program.values->descriptor_sources[source]
                .indirect_image.has_value(),
        "indirect image source was not retained for runtime proof");
  const auto image_handle =
      std::ranges::find_if(*fixture->block, [](const Inst &inst) {
        return inst.GetOpcode() == ValueOpcode::GetImageResource;
      });
  Check(image_handle != fixture->block->end() &&
            image_handle->Arg(0).ResolveInstruction() != nullptr &&
            image_handle->Arg(0).ResolveInstruction()->GetOpcode() ==
                ValueOpcode::ReadConstBuffer,
        "indirect image handle discarded the live material key");

  std::array<uint32_t, 9> user_data{0x1000u,    224u << 16u, 2u, 0u, 0x2000u,
                                    16u << 16u, 4u,          0u, 7u};
  LinearTestMemory memory;
  std::array<uint32_t, 8> image_descriptor{};
  image_descriptor[0] = 0x20u;
  image_descriptor[1] =
      static_cast<uint32_t>(
          Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float)
      << 20u;
  image_descriptor[2] = 3u | (3u << 14u);
  image_descriptor[3] =
      Libs::Graphics::DstSel(4, 5, 6, 7) |
      (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D)
       << 28u);
  for (uint32_t dword = 0; dword < image_descriptor.size(); dword++) {
    memory.words[(0x2000u - memory.base) / 4u + dword] =
        image_descriptor[dword];
    memory.words[(0x2020u - memory.base) / 4u + dword] =
        image_descriptor[dword];
  }
  memory.words[(0x2020u - memory.base) / 4u] ^= 1u;

  SrtRuntime runtime{.user_data = user_data,
                     .userdata = &memory,
                     .read_specialization_memory = ReadLinearTestMemory};
  ResourceSnapshot snapshot;
  std::string error;
  const auto same_snapshot = [](const ResourceSnapshot &lhs,
                                const ResourceSnapshot &rhs) {
    return lhs.buffers == rhs.buffers && lhs.images == rhs.images &&
           lhs.samplers == rhs.samplers && lhs.addresses == rhs.addresses &&
           lhs.flattened_srt == rhs.flattened_srt &&
           lhs.user_data == rhs.user_data &&
           lhs.indirect_images.empty() == rhs.indirect_images.empty();
  };
  Check(MaterializeResources(fixture->program, runtime, snapshot, &error) &&
            snapshot.images.size() == 1 &&
            std::equal(image_descriptor.begin(), image_descriptor.end(),
                       snapshot.images[0].dwords.begin()),
        "invariant indirect image table did not materialize");

  const auto prior_snapshot = snapshot;
  memory.fail_address = 0x1004u;
  Check(!MaterializeResources(fixture->program, runtime, snapshot, &error) &&
            error.find("scalar read") != std::string::npos &&
            same_snapshot(snapshot, prior_snapshot),
        "rejected planning memory read mutated the snapshot");
  memory.fail_address = UINT64_MAX;

  memory.words[(0x1000u - memory.base + 36u) / 4u] = 1u;
  for (uint32_t dword = 0; dword < image_descriptor.size(); dword++) {
    memory.words[(0x2000u - memory.base) / 4u + dword] = 0u;
    memory.words[(0x2020u - memory.base) / 4u + dword] = 0u;
  }
  memory.words[(0x2000u - memory.base) / 4u + 1u] = image_descriptor[1];
  memory.words[(0x2000u - memory.base) / 4u + 3u] = image_descriptor[3];
  memory.words[(0x2020u - memory.base) / 4u + 1u] = image_descriptor[1];
  memory.words[(0x2020u - memory.base) / 4u + 3u] =
      image_descriptor[3] ^ (1u << 28u);
  ResourceSnapshot null_snapshot;
  Check(
      MaterializeResources(fixture->program, runtime, null_snapshot, &error) &&
          null_snapshot.indirect_images.empty() &&
          std::ranges::all_of(null_snapshot.images[0].dwords,
                              [](uint32_t dword) { return dword == 0u; }),
      "stale typed null image descriptors were not canonicalized");

  for (uint32_t dword = 0; dword < image_descriptor.size(); dword++) {
    memory.words[(0x2000u - memory.base) / 4u + dword] =
        image_descriptor[dword];
    memory.words[(0x2020u - memory.base) / 4u + dword] =
        image_descriptor[dword];
  }
  memory.words[(0x2020u - memory.base) / 4u] ^= 1u;
  memory.words[(0x1000u - memory.base + 36u) / 4u] = 1u;
  ResourceSnapshot dynamic_snapshot;
  Check(MaterializeResources(fixture->program, runtime, dynamic_snapshot,
                             &error) &&
            dynamic_snapshot.images.size() == 1 &&
            dynamic_snapshot.indirect_images.size() == 1 &&
            dynamic_snapshot.indirect_images[0].descriptors.size() == 2 &&
            SpecializeResources(fixture->program, dynamic_snapshot, &error) &&
            fixture->program.info.images.size() == 2 &&
            fixture->program.info.images[0].indirect_root == 0 &&
            fixture->program.info.images[0].indirect_mapping_capacity != 0 &&
            fixture->program.info.images[0].indirect_resources.size() == 2 &&
            dynamic_snapshot.images.size() == 2 &&
            dynamic_snapshot.indirect_images.empty(),
        "dynamic indirect image table was not specialized transactionally");

  for (uint32_t dword = 0; dword < image_descriptor.size(); dword++) {
    memory.words[(0x2000u - memory.base) / 4u + dword] =
        image_descriptor[dword];
    memory.words[(0x2020u - memory.base) / 4u + dword] =
        image_descriptor[dword];
  }
  memory.words[(0x2000u - memory.base) / 4u] += 0x100u;
  memory.words[(0x2020u - memory.base) / 4u] += 0x101u;
  ResourceSnapshot rebound_snapshot;
  Check(MaterializeResources(fixture->program, runtime, rebound_snapshot,
                             &error) &&
            ValidateResourceSpecialization(fixture->program, rebound_snapshot,
                                           &error),
        "stable indirect key mapping did not accept changed image addresses");
  memory.words[(0x2020u - memory.base) / 4u] =
      memory.words[(0x2000u - memory.base) / 4u];
  Check(MaterializeResources(fixture->program, runtime, rebound_snapshot,
                             &error) &&
            ValidateResourceSpecialization(fixture->program, rebound_snapshot,
                                           &error),
        "runtime indirect key mapping did not accept collapsed candidates");
  const auto collapsed_snapshot = rebound_snapshot;
  ResourceSnapshot capacity_snapshot;
  for (const uint32_t records : {1u, 3u}) {
    user_data[2] = records;
    Check(
        MaterializeResources(fixture->program, runtime, capacity_snapshot,
                             &error) &&
            ValidateResourceSpecialization(fixture->program, capacity_snapshot,
                                           &error),
        "runtime indirect key mapping rejected a fitting material-table size");
  }
  user_data[2] = 2u;
  memory.words[(0x2020u - memory.base) / 4u] =
      memory.words[(0x2000u - memory.base) / 4u] + 1u;
  memory.words[(0x2040u - memory.base) / 4u] =
      memory.words[(0x2000u - memory.base) / 4u] + 2u;
  for (uint32_t dword = 1; dword < image_descriptor.size(); dword++) {
    memory.words[(0x2040u - memory.base) / 4u + dword] =
        image_descriptor[dword];
  }
  memory.words[(0x1000u - memory.base + 68u) / 4u] = 2u;
  Check(
      !MaterializeResources(fixture->program, runtime, rebound_snapshot,
                            &error) &&
          error.find("candidate topology") != std::string::npos &&
          same_snapshot(rebound_snapshot, collapsed_snapshot),
      "larger indirect candidate topology reused or mutated a cached snapshot");

  auto memory_backed = MakeIndirectImageFixture(false, 0u, true);
  memory_backed->PlanAndTrack();
  EliminateDeadCode(memory_backed->program.values->blocks);
  std::array<uint32_t, 11> memory_backed_user_data{0x1000u, 224u << 16u, 2u, 0u,
                                                   0x2000u, 16u << 16u,  4u, 0u,
                                                   7u,      0x3100u,     0u};
  memory.words[(0x3100u - memory.base) / 4u] = 0x3000u;
  memory.words[(0x3000u - memory.base) / 4u] = 0x1000u;
  memory.fail_address = 0x3100u;
  SrtRuntime memory_backed_runtime{.user_data = memory_backed_user_data,
                                   .userdata = &memory,
                                   .read_specialization_memory =
                                       ReadLinearTestMemory};
  Check(!MaterializeResources(memory_backed->program, memory_backed_runtime,
                              snapshot, &error) &&
            error.find("constant read failed") != std::string::npos &&
            same_snapshot(snapshot, prior_snapshot),
        "rejected indirect table descriptor read mutated the snapshot");
  memory.fail_address = UINT64_MAX;

  auto malformed = MakeIndirectImageFixture(true);
  Check(BuildSrtPlan(malformed->program, &error) &&
            !TrackResources(malformed->program, &error) &&
            error.find("ReadFirstLane") != std::string::npos &&
            !malformed->program.resource_tracking_complete &&
            malformed->program.info.images.empty() &&
            malformed->program.values->descriptor_sources.empty(),
        "malformed indirect image pattern was partially accepted");

  auto wrapped_immediate = MakeIndirectImageFixture(false, 4u);
  Check(BuildSrtPlan(wrapped_immediate->program, &error) &&
            !TrackResources(wrapped_immediate->program, &error) &&
            error.find("ReadFirstLane") != std::string::npos &&
            !wrapped_immediate->program.resource_tracking_complete,
        "wrapped scalar immediate entered the invariant image proof");
}

void TestMixedDimensionIndirectImageSpecialization() {
  auto fixture = MakeIndirectImageFixture(false);
  fixture->PlanAndTrack();
  EliminateDeadCode(fixture->program.values->blocks);

  std::array<uint32_t, 9> user_data{0x1000u,    224u << 16u, 2u, 0u, 0x2000u,
                                    16u << 16u, 4u,          0u, 7u};
  LinearTestMemory memory;
  std::array<uint32_t, 8> descriptor{};
  descriptor[0] = 0x20u;
  descriptor[1] = static_cast<uint32_t>(
                      Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float)
                  << 20u;
  descriptor[2] = 3u | (3u << 14u);
  descriptor[3] =
      Libs::Graphics::DstSel(4, 5, 6, 7) |
      (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kCube)
       << 28u);
  for (uint32_t dword = 0; dword < descriptor.size(); dword++) {
    memory.words[(0x2000u - memory.base) / 4u + dword] = descriptor[dword];
    memory.words[(0x2020u - memory.base) / 4u + dword] = descriptor[dword];
  }
  memory.words[(0x2020u - memory.base) / 4u] = 0x40u;
  memory.words[(0x2020u - memory.base) / 4u + 1u] =
      static_cast<uint32_t>(
          Libs::Graphics::Prospero::BufferFormat::k32_32_32_32UInt)
      << 20u;
  memory.words[(0x2020u - memory.base) / 4u + 3u] =
      Libs::Graphics::DstSel(4, 5, 6, 7) |
      (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D)
       << 28u);
  memory.words[(0x1000u - memory.base + 36u) / 4u] = 1u;

  SrtRuntime runtime{.user_data = user_data,
                     .userdata = &memory,
                     .read_specialization_memory = ReadLinearTestMemory};
  ResourceSnapshot snapshot;
  std::string error;
  Check(MaterializeResources(fixture->program, runtime, snapshot, &error) &&
            snapshot.indirect_images.size() == 1 &&
            SpecializeResources(fixture->program, snapshot, &error) &&
            fixture->program.info.images.size() == 2 &&
            fixture->program.info.images[0].dimension ==
                Decoder::ImageDimension::Dim2DArray &&
            fixture->program.info.images[0].cube &&
            fixture->program.info.images[1].dimension ==
                Decoder::ImageDimension::Dim2D &&
            fixture->program.info.images[1].kind == ResourceKind::ImageUint &&
            !fixture->program.info.images[1].cube,
        error.empty()
            ? "mixed float-cube/uint-2D indirect image table was rejected"
            : error.c_str());

  ShaderComputeInputInfo compute{};
  Check(CollectShaderInfo(fixture->program, {.compute = &compute}, &error) &&
            AllocateBindings(fixture->program, 0, &error) &&
            FindBinding(fixture->program.bindings,
                        DescriptorBindingKind::Sampled2DArray) != nullptr &&
            FindBinding(fixture->program.bindings,
                        DescriptorBindingKind::SampledUint2D) != nullptr,
        error.empty() ? "mixed float-cube/uint-2D bindings were not allocated"
                      : error.c_str());
}

void TestDenseBufferTracking() {
  Fixture fixture;
  std::array<Value, 8> userdata;
  for (uint32_t index = 0; index < userdata.size(); index++) {
    userdata[index] = fixture.UserData(index);
  }
  const auto first =
      fixture.Buffer({userdata[0], userdata[1], userdata[2], userdata[3]}, 4);
  const auto second =
      fixture.Buffer({userdata[4], userdata[5], userdata[6], userdata[7]}, 28);

  MemoryInfo load_info;
  load_info.kind = ResourceKind::Buffer;
  load_info.offset = 4;
  load_info.formatted = true;
  const auto load_flags = fixture.AddMemory(load_info, 4);
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {first, Value(0u), Value(0u), Value(0u), Value(true)},
               load_flags);

  auto store_info = load_info;
  store_info.offset = 12;
  const auto store_flags = fixture.AddMemory(store_info, 8);
  fixture.Emit(ValueOpcode::StoreBufferU32,
               {first, Value(0u), Value(0u), Value(0u), Value(7u), Value(true)},
               store_flags);

  auto atomic_info = load_info;
  atomic_info.offset = 0;
  const auto atomic_flags = fixture.AddMemory(atomic_info, 12);
  fixture.Emit(ValueOpcode::BufferAtomicIAdd32,
               {first, Value(0u), Value(0u), Value(1u), Value(0u), Value(true)},
               atomic_flags);

  const auto other_flags = fixture.AddMemory(load_info, 28);
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {second, Value(0u), Value(0u), Value(0u), Value(true)},
               other_flags);
  fixture.PlanAndTrack();

  Check(fixture.program.info.buffers.size() == 2,
        "typed buffer sources were not densely interned");
  Check(fixture.program.values->descriptor_sources.size() == 2,
        "descriptor source table did not match dense topology");
  const auto &resource = fixture.program.info.buffers[0];
  Check(resource.read && resource.written && resource.atomic &&
            resource.formatted && resource.max_byte_extent == 16 &&
            resource.first_use_pc == 4,
        "buffer access facts were not merged");
  Check(first.Instruction()->Flags<uint32_t>() == 0 &&
            second.Instruction()->Flags<uint32_t>() == 1,
        "typed handles were not assigned dense indices");
  Check(fixture.program.values->memory_info[load_flags.index].resource == 0 &&
            fixture.program.values->memory_info[store_flags.index].resource ==
                0 &&
            fixture.program.values->memory_info[other_flags.index].resource ==
                1,
        "typed memory metadata was not patched to dense indices");

  std::string error;
  Check(!TrackResources(fixture.program, &error) &&
            error.find("already tracked") != std::string::npos,
        "resource tracking allowed a second mutation pass");
}

void TestScalarAndVectorBufferAlias() {
  Fixture fixture;
  const auto d0 = fixture.UserData(0);
  const auto d1 = fixture.UserData(1);
  const auto d2 = fixture.UserData(2);
  const auto d3 = fixture.UserData(3);
  const auto descriptor = fixture.Buffer({d0, d1, d2, d3}, 4);

  MemoryInfo scalar;
  scalar.kind = ResourceKind::ScalarBuffer;
  const auto scalar_flags = fixture.AddMemory(scalar, 4);
  fixture.Emit(ValueOpcode::ReadConstBuffer, {descriptor, fixture.UserData(4)},
               scalar_flags);
  MemoryInfo vector;
  vector.kind = ResourceKind::Buffer;
  const auto vector_flags = fixture.AddMemory(vector, 8);
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
               vector_flags);
  fixture.PlanAndTrack();

  Check(fixture.program.info.buffers.size() == 1 &&
            fixture.program.info.buffers[0].scalar,
        "typed scalar and vector uses of one descriptor were split");
  Check(fixture.program.values->memory_info[scalar_flags.index].resource == 0 &&
            fixture.program.values->memory_info[vector_flags.index].resource ==
                0,
        "scalar/vector alias did not share a dense index");
}

void TestRuntimeUnsignedMinDescriptor() {
  Fixture fixture;
  const auto word3 =
      fixture.Emit(ValueOpcode::UMin32, {fixture.UserData(0), Value(0x100u)});
  const auto descriptor =
      fixture.Buffer({Value(0u), Value(0u), Value(64u), word3}, 0x330);
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(memory, 0x330));
  fixture.PlanAndTrack();

  std::array<uint32_t, 1> user_data{0xffffffffu};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue value;
  std::string error;
  const auto source = fixture.program.info.buffers[0].source;
  Check(EvaluateDescriptorSource(fixture.program, source, 0x330, runtime, value,
                                 &error) &&
            value.dwords[3] == 0x100u,
        "runtime descriptor unsigned minimum did not clamp its first operand");
  user_data[0] = 0x80u;
  Check(
      EvaluateDescriptorSource(fixture.program, source, 0x330, runtime, value,
                               &error) &&
          value.dwords[3] == 0x80u,
      "runtime descriptor unsigned minimum did not preserve its first operand");
}

void TestUniformWqmBufferDescriptor() {
  Fixture fixture(ShaderType::Pixel);
  const auto condition = fixture.Emit(
      ValueOpcode::WqmMask,
      {fixture.Emit(ValueOpcode::IEqual32, {fixture.UserData(4), Value(1u)})});
  const auto word0 = fixture.Emit(
      ValueOpcode::SelectU32,
      {condition, fixture.UserData(0), fixture.UserData(1)});
  const auto descriptor = fixture.Buffer(
      {word0, fixture.UserData(2), fixture.UserData(3), Value(0u)}, 0x248);
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(memory, 0x248));
  fixture.PlanAndTrack();

  std::array<uint32_t, 5> user_data{0x11111111u, 0x22222222u, 0u, 0u, 1u};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue value;
  std::string error;
  const auto source = fixture.program.info.buffers[0].source;
  Check(EvaluateDescriptorSource(fixture.program, source, 0x248, runtime, value,
                                 &error) &&
            value.dwords[0] == 0x11111111u,
        "uniform WqmMask descriptor selected the wrong true value");
  user_data[4] = 0u;
  Check(EvaluateDescriptorSource(fixture.program, source, 0x248, runtime, value,
                                 &error) &&
            value.dwords[0] == 0x22222222u,
        "uniform WqmMask descriptor selected the wrong false value");
}

void TestUniformFloatingWqmBufferDescriptor() {
  Fixture fixture(ShaderType::Pixel);
  const auto scalar_float = fixture.Emit(ValueOpcode::BitCastF32U32,
                                         {fixture.UserData(4)});
  const auto condition = fixture.Emit(
      ValueOpcode::WqmMask,
      {fixture.Emit(ValueOpcode::FPOrdGreaterThan32,
                    {scalar_float, Value::F32(0.0f)})});
  const auto word0 = fixture.Emit(
      ValueOpcode::SelectU32,
      {condition, fixture.UserData(0), fixture.UserData(1)});
  const auto descriptor = fixture.Buffer(
      {word0, fixture.UserData(2), fixture.UserData(3), Value(0u)}, 0x248);
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(memory, 0x248));
  fixture.PlanAndTrack();

  std::array<uint32_t, 5> user_data{
      0x11111111u, 0x22222222u, 0u, 0u, std::bit_cast<uint32_t>(1.0f)};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue value;
  std::string error;
  const auto source = fixture.program.info.buffers[0].source;
  Check(EvaluateDescriptorSource(fixture.program, source, 0x248, runtime, value,
                                 &error) &&
            value.dwords[0] == 0x11111111u,
        "uniform floating WqmMask selected the wrong true descriptor");
  user_data[4] = std::bit_cast<uint32_t>(-1.0f);
  Check(EvaluateDescriptorSource(fixture.program, source, 0x248, runtime, value,
                                 &error) &&
            value.dwords[0] == 0x22222222u,
        "uniform floating WqmMask selected the wrong false descriptor");
  user_data[4] = 0x7fc00000u;
  Check(EvaluateDescriptorSource(fixture.program, source, 0x248, runtime, value,
                                 &error) &&
            value.dwords[0] == 0x22222222u,
        "ordered floating WqmMask treated NaN as a true comparison");

  Fixture rejected(ShaderType::Pixel);
  const auto lane_float = rejected.Emit(
      ValueOpcode::BitCastF32U32, {rejected.Emit(ValueOpcode::LaneId)});
  const auto lane_condition = rejected.Emit(
      ValueOpcode::WqmMask,
      {rejected.Emit(ValueOpcode::FPOrdGreaterThan32,
                     {lane_float, Value::F32(0.0f)})});
  const auto lane_word = rejected.Emit(
      ValueOpcode::SelectU32,
      {lane_condition, rejected.UserData(0), rejected.UserData(1)});
  const auto lane_descriptor = rejected.Buffer(
      {lane_word, Value(0u), Value(0u), Value(0u)}, 0x248);
  rejected.Emit(ValueOpcode::LoadBufferU32,
                {lane_descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
                rejected.AddMemory(memory, 0x248));
  Check(BuildSrtPlan(rejected.program, &error) &&
            !TrackResources(rejected.program, &error) &&
            error.find("LaneId") != std::string::npos,
        "lane-dependent floating comparison entered descriptor tracking");
}

void TestImagesSamplersAndAliases() {
  Fixture fixture;
  std::array<Value, 8> image_words;
  for (uint32_t index = 0; index < image_words.size(); index++) {
    image_words[index] = fixture.UserData(index);
  }
  const auto image_address = fixture.ImageAddress();
  const std::array<Value, 4> sampler0{Value(0u), Value(1u), Value(2u),
                                      Value(0x1111u)};
  const std::array<Value, 4> sampler1{Value(0u), Value(1u), Value(2u),
                                      Value(0x2222u)};

  auto AddSample = [&](uint32_t pc, uint32_t sample_flags,
                       const auto &sampler_words) {
    const auto image = fixture.Image(image_words, pc);
    const auto sampler = fixture.Sampler(sampler_words, pc);
    MemoryInfo memory;
    memory.kind = ResourceKind::Image;
    memory.image_dimension = Decoder::ImageDimension::Dim2D;
    memory.image_sample_flags = sample_flags;
    fixture.Emit(ValueOpcode::ImageSampleRaw, {image, sampler, image_address},
                 fixture.AddMemory(memory, pc));
    return std::pair{image, sampler};
  };
  const auto normal = AddSample(4, 0, sampler0);
  const auto repeated = AddSample(8, 0, sampler1);
  const auto compare = AddSample(12, Decoder::ImageSampleFlagCompare, sampler0);

  const auto storage = fixture.Image(image_words, 16);
  MemoryInfo storage_memory;
  storage_memory.kind = ResourceKind::StorageImageUint;
  storage_memory.image_dimension = Decoder::ImageDimension::Dim2D;
  fixture.Emit(ValueOpcode::ImageAtomicIAdd32,
               {storage, image_address, Value(1u), Value(true)},
               fixture.AddMemory(storage_memory, 16));

  const auto buffer = fixture.Buffer(
      {image_words[0], image_words[1], image_words[2], image_words[3]}, 20);
  MemoryInfo buffer_memory;
  buffer_memory.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {buffer, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(buffer_memory, 20));
  fixture.PlanAndTrack();

  Check(fixture.program.info.images.size() == 3 &&
            fixture.program.info.samplers.size() == 1 &&
            fixture.program.info.sampled_pairs.size() == 2,
        "typed image view classes or samplers were deduplicated incorrectly");
  Check(normal.first.Instruction()->Flags<uint32_t>() ==
                repeated.first.Instruction()->Flags<uint32_t>() &&
            compare.first.Instruction()->Flags<uint32_t>() !=
                normal.first.Instruction()->Flags<uint32_t>(),
        "image handles did not receive view-class indices");
  Check(normal.second.Instruction()->Flags<uint32_t>() == 0 &&
            repeated.second.Instruction()->Flags<uint32_t>() == 0,
        "unused sampler border colors prevented source interning");
  const auto sampler_source = fixture.program.info.samplers[0].source;
  Check(fixture.program.values->descriptor_sources[sampler_source]
                .dwords[3]
                .U32() == 0,
        "unused sampler border color was not canonicalized");
  Check(fixture.program.info.buffers[0].image_alias == 0,
        "buffer/image descriptor alias was not linked");
}

void TestSampleAdjustSamplerScratch() {
  Fixture fixture(ShaderType::Pixel);
  const auto active = fixture.Emit(ValueOpcode::WqmMask, {Value(true)});
  const auto lane =
      fixture.Emit(ValueOpcode::SelectU32, {active, Value(1u), Value(0u)});
  const auto low =
      fixture.Emit(ValueOpcode::BitwiseAnd32, {lane, Value(0xffu)});
  const auto high =
      fixture.Emit(ValueOpcode::BitwiseAnd32, {lane, Value(0xffu)});
  const auto quads = fixture.Emit(
      ValueOpcode::BitwiseOr32,
      {low, fixture.Emit(ValueOpcode::ShiftLeftLogical32, {high, Value(8u)})});
  const auto scratch =
      fixture.Emit(ValueOpcode::ShiftLeftLogical32, {quads, Value(12u)});
  const auto word3 =
      fixture.Emit(ValueOpcode::BitwiseOr32, {fixture.UserData(3), scratch});
  const auto image = fixture.Image({Value(0u), Value(0u), Value(0u), Value(0u),
                                    Value(0u), Value(0u), Value(0u), Value(0u)},
                                   0x1ec);
  const auto sampler = fixture.Sampler(
      {fixture.UserData(0), fixture.UserData(1), fixture.UserData(2), word3},
      0x1ec);
  MemoryInfo memory;
  memory.kind = ResourceKind::Image;
  memory.image_dimension = Decoder::ImageDimension::Dim2D;
  memory.image_sample_flags = Decoder::ImageSampleFlagAdjust;
  fixture.Emit(ValueOpcode::ImageSampleRaw,
               {image, sampler, fixture.ImageAddress()},
               fixture.AddMemory(memory, 0x1ec));
  fixture.PlanAndTrack();

  const auto source = fixture.program.info.samplers[0].source;
  const auto stored = fixture.program.values->descriptor_sources[source]
                          .dwords[3]
                          .Resolve()
                          .TryInstruction();
  Check(stored != nullptr && stored->GetOpcode() == ValueOpcode::GetUserData,
        "SampleAdjust reserved scratch remained in sampler identity");
  std::array<uint32_t, 4> user_data{4u, 1u, 2u, 0x80000abcu};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue descriptor;
  std::string error;
  Check(EvaluateDescriptorSource(fixture.program, source, 0x1ec, runtime,
                                 descriptor, &error) &&
            descriptor.dwords[3] == 0x80000abcu,
        "SampleAdjust canonicalization lost sampler border fields");

  const auto CheckRejected = [](uint32_t flags, uint32_t shift,
                                const char *message) {
    Fixture rejected(ShaderType::Pixel);
    const auto lane_condition = rejected.Emit(
        ValueOpcode::IEqual32,
        {rejected.Emit(ValueOpcode::LaneId), Value(0u)});
    const auto condition =
        rejected.Emit(ValueOpcode::WqmMask, {lane_condition});
    const auto bit = rejected.Emit(ValueOpcode::SelectU32,
                                   {condition, Value(1u), Value(0u)});
    const auto dynamic =
        rejected.Emit(ValueOpcode::ShiftLeftLogical32, {bit, Value(shift)});
    const auto dynamic_word3 = rejected.Emit(ValueOpcode::BitwiseOr32,
                                             {rejected.UserData(3), dynamic});
    const auto rejected_image =
        rejected.Image({Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
                        Value(0u), Value(0u), Value(0u)},
                       0x200);
    const auto rejected_sampler =
        rejected.Sampler({rejected.UserData(0), rejected.UserData(1),
                          rejected.UserData(2), dynamic_word3},
                         0x200);
    MemoryInfo rejected_memory;
    rejected_memory.kind = ResourceKind::Image;
    rejected_memory.image_dimension = Decoder::ImageDimension::Dim2D;
    rejected_memory.image_sample_flags = flags;
    rejected.Emit(ValueOpcode::ImageSampleRaw,
                  {rejected_image, rejected_sampler, rejected.ImageAddress()},
                  rejected.AddMemory(rejected_memory, 0x200));
    std::string rejected_error;
    Check(BuildSrtPlan(rejected.program, &rejected_error) &&
              !TrackResources(rejected.program, &rejected_error) &&
              rejected_error.find("LaneId") != std::string::npos,
          message);
  };
  CheckRejected(0u, 12u,
                "ordinary sampling accepted SampleAdjust reserved scratch");
  CheckRejected(Decoder::ImageSampleFlagAdjust, 30u,
                "SampleAdjust canonicalization discarded border-mode bits");
}

void TestDynamicStorageMipTracking() {
  Fixture fixture;
  std::array<Value, 8> image_words;
  for (uint32_t index = 0; index < image_words.size(); index++) {
    image_words[index] = fixture.UserData(index);
  }
  const auto data = fixture.Emit(ValueOpcode::CompositeConstructU32x4,
                                 {Value(1u), Value(2u), Value(3u), Value(4u)});
  const auto AddStore = [&](uint32_t pc, bool has_mip, Value lod) {
    const auto handle = fixture.Image(image_words, pc);
    const auto address = fixture.Emit(
        ValueOpcode::MakeImageAddress,
        {Value(0u), Value(0u), lod, Value(0u), Value(0u), Value(0u), Value(0u),
         Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u)});
    MemoryInfo memory;
    memory.kind = ResourceKind::StorageImage;
    memory.image_dimension = Decoder::ImageDimension::Dim2D;
    memory.image_address_components = has_mip ? 3u : 2u;
    memory.image_has_mip = has_mip;
    const auto flags = fixture.AddMemory(memory, pc);
    fixture.Emit(ValueOpcode::ImageWrite, {handle, address, data, Value(true)},
                 flags);
    return std::pair{handle, flags.index};
  };

  const auto plain = AddStore(4, false, Value(0u));
  const auto mip1 = AddStore(8, true, Value(1u));
  const auto mip2 = AddStore(12, true, Value(2u));
  const auto dynamic = AddStore(16, true, fixture.UserData(8));
  fixture.PlanAndTrack();

  const auto &images = fixture.program.info.images;
  Check(images.size() == 2 && images[0].mip_mode == ImageMipMode::None &&
            images[0].mip_count == 1 &&
            images[1].mip_mode == ImageMipMode::DynamicStorage &&
            images[1].mip_count == 1,
        "storage mip writes did not share one dynamic logical resource");
  Check(plain.first.Instruction()->Flags<uint32_t>() == 0 &&
            mip1.first.Instruction()->Flags<uint32_t>() == 1 &&
            mip2.first.Instruction()->Flags<uint32_t>() == 1 &&
            dynamic.first.Instruction()->Flags<uint32_t>() == 1 &&
            fixture.program.values->memory_info[plain.second].resource == 0 &&
            fixture.program.values->memory_info[mip1.second].resource == 1 &&
            fixture.program.values->memory_info[mip2.second].resource == 1 &&
            fixture.program.values->memory_info[dynamic.second].resource == 1,
        "dynamic storage mip handles and memory metadata were not patched");

  DescriptorValue descriptor{};
  descriptor.dwords[0] = 0x1000u;
  descriptor.dwords[1] =
      static_cast<uint32_t>(
          Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float)
      << 20u;
  descriptor.dwords[2] = 3u | (3u << 14u);
  descriptor.dwords[3] =
      Libs::Graphics::DstSel(4, 5, 6, 7) | (1u << 12u) | (3u << 16u) |
      (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D)
       << 28u);
  descriptor.dwords[5] = 3u << 4u;
  descriptor.dword_count = 8;
  ResourceSnapshot snapshot;
  snapshot.images.assign(images.size(), descriptor);
  std::string error;
  Check(SpecializeResources(fixture.program, snapshot, &error) &&
            fixture.program.info.images[1].mip_count == 3 &&
            ValidateResourceSpecialization(fixture.program, snapshot, &error),
        "base-1 through last-3 dynamic storage range was not specialized");
  ShaderComputeInputInfo compute{};
  Check(CollectShaderInfo(fixture.program, {.compute = &compute}, &error) &&
            AllocateBindings(fixture.program, 0, &error),
        "dynamic storage mip bindings were not allocated");
  const auto *storage_binding =
      FindBinding(fixture.program.bindings, DescriptorBindingKind::Storage2D);
  Check(storage_binding != nullptr &&
            storage_binding->resources == std::vector<uint32_t>({0, 1, 1, 1}),
        "dynamic storage mip descriptors were not expanded consecutively");

  Program null_program;
  null_program.values = std::make_shared<ValueProgram>();
  null_program.resource_tracking_complete = true;
  ImageResource null_image;
  null_image.kind = ResourceKind::StorageImage;
  null_image.dimension = Decoder::ImageDimension::Dim2D;
  null_image.mip_mode = ImageMipMode::DynamicStorage;
  null_image.written = true;
  null_program.info.images.push_back(null_image);
  ResourceSnapshot null_snapshot;
  DescriptorValue null_descriptor{};
  null_descriptor.dword_count = 8;
  null_snapshot.images.push_back(null_descriptor);
  Check(SpecializeResources(null_program, null_snapshot, &error) &&
            null_program.info.images[0].mip_count == 1 &&
            ValidateResourceSpecialization(null_program, null_snapshot, &error),
        "canonical null dynamic storage image did not retain one descriptor");

  snapshot.images[1].dwords[3] =
      (snapshot.images[1].dwords[3] & ~(0xfu << 16u)) | (2u << 16u);
  Check(!ValidateResourceSpecialization(fixture.program, snapshot, &error),
        "a changed dynamic storage mip count reused the specialization");
  snapshot.images[1].dwords[3] =
      (snapshot.images[1].dwords[3] & ~((0xfu << 12u) | (0xfu << 16u))) |
      (4u << 12u) | (3u << 16u);
  Check(!ValidateResourceSpecialization(fixture.program, snapshot, &error),
        "an inverted dynamic storage mip range was accepted");
}

void TestSrtFlatteningAndRuntimeMemoization() {
  Fixture fixture;
  const auto base =
      fixture.Address(fixture.UserData(0), fixture.UserData(1), 4);
  MemoryInfo scalar;
  scalar.kind = ResourceKind::ScalarAddress;
  scalar.offset = 4;
  const auto read0 = fixture.Emit(ValueOpcode::LoadAddressU32,
                                  {base, Value(0u), Value(0u), Value(true)},
                                  fixture.AddMemory(scalar, 4));
  const auto descriptor0 =
      fixture.Buffer({read0, Value(0u), Value(64u), Value(0u)}, 12);
  const auto descriptor1 =
      fixture.Buffer({read0, Value(0u), Value(64u), Value(0u)}, 16);
  MemoryInfo buffer;
  buffer.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {descriptor0, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(buffer, 12));
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {descriptor1, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(buffer, 16));
  fixture.PlanAndTrack();

  Check(fixture.program.values->srt_reads.size() == 1,
        "shared typed scalar read did not receive one flat SRT slot");
  Check(fixture.program.info.buffers.size() == 1 &&
            fixture.program.info.addresses.empty(),
        "planning-only scalar reads leaked into resource topology");
  Check(fixture.program.values->memory_info[0].planning_only,
        "canonical runtime scalar read was not marked planning-only");

  std::array<uint32_t, 2> user_data{0x1000u, 0u};
  TestMemory memory;
  memory.words[1] = 0xdeadbeefu;
  SrtRuntime runtime{.user_data = user_data,
                     .read_memory = ReadTestMemory,
                     .userdata = &memory};
  std::vector<DescriptorValue> descriptors;
  std::vector<uint32_t> flat;
  const DescriptorSourceRequest request{fixture.program.info.buffers[0].source,
                                        12};
  std::string error;
  Check(EvaluateRuntimeSources(fixture.program, std::span{&request, 1}, runtime,
                               descriptors, flat, {}, &error),
        "typed runtime source evaluation failed");
  Check(descriptors.size() == 1 && descriptors[0].dwords[0] == 0xdeadbeefu &&
            flat == std::vector<uint32_t>{0xdeadbeefu} && memory.reads == 1,
        "descriptor and flat SRT evaluation did not share one memoized read");

  memory.reads = 0;
  memory.fail_after = 0;
  descriptors = {{{1u}, 1u}};
  flat = {2u};
  Check(!EvaluateRuntimeSources(fixture.program, std::span{&request, 1},
                                runtime, descriptors, flat, {}, &error) &&
            descriptors == std::vector<DescriptorValue>{{{1u}, 1u}} &&
            flat == std::vector<uint32_t>{2u},
        "runtime evaluation failure was not transactional");

  ShaderComputeInputInfo compute{};
  Check(CollectShaderInfo(fixture.program, {.compute = &compute}, &error) &&
            AllocateBindings(fixture.program, 0, &error) &&
            FindBinding(fixture.program.bindings,
                        DescriptorBindingKind::FlattenedSrt) != nullptr,
        "flattened typed SRT reads did not receive a binding");
}

void TestDynamicSrtReadRemainsExplicit() {
  Fixture fixture;
  const auto base =
      fixture.Address(fixture.UserData(0), fixture.UserData(1), 4);
  MemoryInfo scalar;
  scalar.kind = ResourceKind::ScalarAddress;
  const auto read =
      fixture.Emit(ValueOpcode::LoadAddressU32,
                   {base, fixture.UserData(2), Value(0u), Value(true)},
                   fixture.AddMemory(scalar, 4));
  const auto descriptor =
      fixture.Buffer({read, Value(0u), Value(64u), Value(0u)}, 8);
  MemoryInfo buffer;
  buffer.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(buffer, 8));
  fixture.PlanAndTrack();

  Check(fixture.program.values->srt_reads.empty() &&
            fixture.program.values->dynamic_reads.size() == 1 &&
            fixture.program.info.addresses.size() == 1,
        "dynamic scalar read was incorrectly flattened or lost");
  std::array<uint32_t, 3> user_data{0x1000u, 0u, 4u};
  TestMemory memory;
  memory.words[1] = 0xabcdef01u;
  SrtRuntime runtime{.user_data = user_data,
                     .read_memory = ReadTestMemory,
                     .userdata = &memory};
  DescriptorValue value;
  std::string error;
  Check(EvaluateDescriptorSource(fixture.program,
                                 fixture.program.info.buffers[0].source, 8,
                                 runtime, value, &error) &&
            value.dwords[0] == 0xabcdef01u && memory.reads == 1,
        "dynamic typed scalar descriptor source was not evaluated");

  ShaderComputeInputInfo compute{};
  Check(CollectShaderInfo(fixture.program, {.compute = &compute}, &error) &&
            AllocateBindings(fixture.program, 0, &error) &&
            FindBinding(fixture.program.bindings,
                        DescriptorBindingKind::FlattenedSrt) == nullptr &&
            FindBinding(fixture.program.bindings,
                        DescriptorBindingKind::AddressMemory) != nullptr,
        "dynamic scalar read received the wrong resource bindings");
  Check(fixture.program.bindings.memory_offset_dword ==
                fixture.program.bindings.user_data_registers.size() &&
            fixture.program.bindings.memory_offset_count == 2u &&
            fixture.program.bindings.ShaderDataDwords() ==
                fixture.program.bindings.memory_offset_dword + 1u,
        "unified memory-offset layout is inconsistent");
}

void TestPhiValidation() {
  Fixture fixture;
  auto *entry = fixture.block;
  auto *left = fixture.AddBlock();
  auto *right = fixture.AddBlock();
  auto *merge = fixture.AddBlock();
  const auto condition = fixture.Emit(
      ValueOpcode::INotEqual32, {fixture.UserData(0), Value(0u)}, 0, entry);
  const auto left_word = fixture.UserData(1);
  const auto right_word = fixture.UserData(2);
  entry->AddBranch(left);
  entry->AddBranch(right);
  left->AddBranch(merge);
  right->AddBranch(merge);
  auto &blocks = fixture.program.values->block_info;
  blocks[0].condition = condition;
  blocks[0].terminator.kind = CFG::TerminatorKind::ConditionalBranch;
  blocks[0].terminator.true_block = blocks[1].id;
  blocks[0].terminator.false_block = blocks[2].id;
  blocks[0].terminator.merge_block = blocks[3].id;
  blocks[1].terminator.kind = CFG::TerminatorKind::Branch;
  blocks[1].terminator.true_block = blocks[3].id;
  blocks[2].terminator.kind = CFG::TerminatorKind::Branch;
  blocks[2].terminator.true_block = blocks[3].id;
  auto &phi = merge->AppendNewInst(ValueOpcode::Phi, {},
                                   static_cast<uint64_t>(Type::U32));
  phi.AddPhiOperand(left, left_word);
  phi.AddPhiOperand(right, right_word);
  const auto word3 =
      fixture.Emit(ValueOpcode::UMin32, {Value(&phi), Value(0x100u)}, 0, merge);
  const auto handle = fixture.Emit(ValueOpcode::GetBufferResource,
                                   {Value(0u), Value(0u), Value(0u), word3},
                                   MemoryFlags{0, 20}, merge);
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {handle, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(memory, 20), merge);

  std::string error;
  Check(BuildSrtPlan(fixture.program, &error) &&
            TrackResources(fixture.program, &error) &&
            fixture.program.resource_tracking_complete &&
            fixture.program.info.buffers.size() == 1u,
        error.empty() ? "runtime-selectable descriptor phi was rejected"
                      : error.c_str());

  std::array<uint32_t, 3> user_data{1u, 0x80u, 0x40u};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue descriptor;
  const auto source = fixture.program.info.buffers[0].source;
  Check(EvaluateDescriptorSource(fixture.program, source, 20, runtime,
                                 descriptor, &error) &&
            descriptor.dwords[3] == 0x80u,
        error.empty()
            ? "true CFG edge did not select its descriptor phi operand"
            : error.c_str());
  user_data[0] = 0u;
  Check(EvaluateDescriptorSource(fixture.program, source, 20, runtime,
                                 descriptor, &error) &&
            descriptor.dwords[3] == 0x40u,
        error.empty()
            ? "false CFG edge did not select its descriptor phi operand"
            : error.c_str());
}

void TestControlDependentImagePhi() {
  Fixture fixture(ShaderType::Vertex);
  auto *entry = fixture.block;
  auto *left = fixture.AddBlock();
  auto *right = fixture.AddBlock();
  auto *merge = fixture.AddBlock();
  const auto condition = fixture.Emit(
      ValueOpcode::INotEqual32, {fixture.UserData(0), Value(0u)}, 0, entry);
  const auto left_address = fixture.UserData(1);
  const auto right_address = fixture.UserData(2);
  entry->AddBranch(left);
  entry->AddBranch(right);
  left->AddBranch(merge);
  right->AddBranch(merge);
  auto &blocks = fixture.program.values->block_info;
  blocks[0].condition = condition;
  blocks[0].terminator.kind = CFG::TerminatorKind::ConditionalBranch;
  blocks[0].terminator.true_block = blocks[1].id;
  blocks[0].terminator.false_block = blocks[2].id;
  blocks[0].terminator.merge_block = blocks[3].id;
  blocks[1].terminator.kind = CFG::TerminatorKind::Branch;
  blocks[1].terminator.true_block = blocks[3].id;
  blocks[2].terminator.kind = CFG::TerminatorKind::Branch;
  blocks[2].terminator.true_block = blocks[3].id;

  auto &phi = merge->AppendNewInst(ValueOpcode::Phi, {},
                                   static_cast<uint64_t>(Type::U32));
  phi.AddPhiOperand(left, left_address);
  phi.AddPhiOperand(right, right_address);
  const auto image = fixture.Image(
      {Value(&phi),
       Value(static_cast<uint32_t>(
                 Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float)
             << 20u),
       Value(3u | (3u << 14u)),
       Value(
           Libs::Graphics::DstSel(4, 5, 6, 7) |
           (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D)
            << 28u)),
       Value(0u), Value(0u), Value(0u), Value(0u)},
      0x29cu);
  const auto sampler =
      fixture.Sampler({Value(0u), Value(0u), Value(0u), Value(0u)}, 0x29cu);
  MemoryInfo memory;
  memory.kind = ResourceKind::Image;
  memory.image_dimension = Decoder::ImageDimension::Dim2D;
  fixture.Emit(ValueOpcode::ImageSampleRaw,
               {image, sampler, fixture.ImageAddress()},
               fixture.AddMemory(memory, 0x29cu), merge);

  std::string error;
  Check(BuildSrtPlan(fixture.program, &error) &&
            TrackResources(fixture.program, &error) &&
            fixture.program.info.images.size() == 1u,
        error.empty() ? "vertex image descriptor phi was rejected"
                      : error.c_str());

  std::array<uint32_t, 3> user_data{1u, 0x1000u, 0x2000u};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue descriptor;
  const auto source = fixture.program.info.images[0].source;
  Check(EvaluateDescriptorSource(fixture.program, source, 0x29cu, runtime,
                                 descriptor, &error) &&
            descriptor.dwords[0] == 0x1000u,
        error.empty() ? "vertex image phi did not select its true descriptor"
                      : error.c_str());
  user_data[0] = 0u;
  Check(EvaluateDescriptorSource(fixture.program, source, 0x29cu, runtime,
                                 descriptor, &error) &&
            descriptor.dwords[0] == 0x2000u,
        error.empty() ? "vertex image phi did not select its false descriptor"
                      : error.c_str());
}

void TestZeroInitializedDescriptorPhi() {
  Fixture fixture;
  auto *live = fixture.block;
  auto *terminated = fixture.AddBlock();
  auto *merge = fixture.AddBlock();
  live->AddBranch(merge);
  terminated->AddBranch(merge);
  const auto descriptor_word = fixture.UserData(0);
  auto &phi = merge->AppendNewInst(ValueOpcode::Phi, {},
                                   static_cast<uint64_t>(Type::U32));
  phi.AddPhiOperand(live, descriptor_word);
  phi.AddPhiOperand(terminated, Value(0u));
  const auto handle =
      fixture.Emit(ValueOpcode::GetBufferResource,
                   {Value(&phi), Value(0u), Value(64u), Value(0u)},
                   MemoryFlags{0, 0x1b0}, merge);
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {handle, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(memory, 0x1b0), merge);
  fixture.PlanAndTrack();

  std::array<uint32_t, 1> user_data{0x12345678u};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue descriptor;
  std::string error;
  Check(EvaluateDescriptorSource(fixture.program,
                                 fixture.program.info.buffers[0].source, 4,
                                 runtime, descriptor, &error) &&
            descriptor.dwords[0] == user_data[0],
        "zero-initialized descriptor phi did not retain its live source");
}

void TestInactiveDescriptorPhiEdge() {
  Fixture fixture;
  auto *terminated = fixture.block;
  auto *live = fixture.AddBlock();
  auto *merge = fixture.AddBlock();
  auto *resource = fixture.AddBlock();
  auto *exit = fixture.AddBlock();
  terminated->AddBranch(merge);
  live->AddBranch(merge);
  merge->AddBranch(resource);
  merge->AddBranch(exit);

  const auto terminated_word = fixture.UserData(0);
  const auto live_word = fixture.Emit(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(1))}, 0, live);
  auto &condition = merge->AppendNewInst(ValueOpcode::Phi, {},
                                         static_cast<uint64_t>(Type::U1));
  condition.AddPhiOperand(terminated, Value(false));
  condition.AddPhiOperand(live, Value(true));
  auto &descriptor_phi = merge->AppendNewInst(ValueOpcode::Phi, {},
                                              static_cast<uint64_t>(Type::U32));
  descriptor_phi.AddPhiOperand(terminated, terminated_word);
  descriptor_phi.AddPhiOperand(live, live_word);

  const auto merge_index = static_cast<size_t>(
      std::distance(fixture.program.values->blocks.begin(),
                    std::ranges::find(fixture.program.values->blocks, merge)));
  const auto resource_index = static_cast<size_t>(std::distance(
      fixture.program.values->blocks.begin(),
      std::ranges::find(fixture.program.values->blocks, resource)));
  const auto exit_index = static_cast<size_t>(
      std::distance(fixture.program.values->blocks.begin(),
                    std::ranges::find(fixture.program.values->blocks, exit)));
  auto &merge_info = fixture.program.values->block_info[merge_index];
  merge_info.terminator.kind = CFG::TerminatorKind::ConditionalBranch;
  merge_info.terminator.true_block =
      fixture.program.values->block_info[resource_index].id;
  merge_info.terminator.false_block =
      fixture.program.values->block_info[exit_index].id;
  merge_info.condition = Value(&condition);

  const auto handle =
      fixture.Emit(ValueOpcode::GetBufferResource,
                   {Value(&descriptor_phi), Value(0u), Value(64u), Value(0u)},
                   MemoryFlags{0, 0x3d0}, resource);
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {handle, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(memory, 0x3d0), resource);

  fixture.PlanAndTrack();

  std::array<uint32_t, 2> user_data{0xdeadbeefu, 0x12345678u};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue descriptor;
  std::string error;
  Check(EvaluateDescriptorSource(fixture.program,
                                 fixture.program.info.buffers[0].source, 4,
                                 runtime, descriptor, &error) &&
            descriptor.dwords[0] == user_data[1],
        "descriptor phi retained a value from the terminating branch");
}

void TestLoopCycleEnteredThroughRuntimeValue() {
  Fixture fixture;
  auto *entry = fixture.block;
  auto *loop = fixture.AddBlock();
  const auto initial = fixture.UserData(0);
  entry->AddBranch(loop);
  loop->AddBranch(loop);
  auto &phi = loop->AppendNewInst(ValueOpcode::Phi, {},
                                  static_cast<uint64_t>(Type::U32));
  const auto carried = fixture.Emit(ValueOpcode::BitwiseAnd32,
                                    {Value(&phi), Value(0xffffffffu)}, 0, loop);
  phi.AddPhiOperand(entry, initial);
  phi.AddPhiOperand(loop, carried);
  fixture.Emit(ValueOpcode::GetBufferResource,
               {carried, Value(0u), Value(0u), Value(0u)}, MemoryFlags{0, 12},
               loop);

  std::string error;
  Check(BuildSrtPlan(fixture.program, &error),
        "SRT planning rejected a valid loop entered through a runtime value");
}

void TestInvariantLoopPhi() {
  Fixture fixture;
  auto *entry = fixture.block;
  auto *loop = fixture.AddBlock();
  entry->AddBranch(loop);
  loop->AddBranch(loop);
  const auto invariant = fixture.UserData(0);
  auto &phi = loop->AppendNewInst(ValueOpcode::Phi, {},
                                  static_cast<uint64_t>(Type::U32));
  phi.AddPhiOperand(entry, invariant);
  phi.AddPhiOperand(loop, Value(&phi));
  const auto handle = fixture.Emit(
      ValueOpcode::GetBufferResource,
      {Value(&phi), Value(0u), Value(0u), Value(0u)}, MemoryFlags{0, 4}, loop);
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {handle, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(memory, 4), loop);
  fixture.PlanAndTrack();

  std::array<uint32_t, 1> user_data{0x12345678u};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue descriptor;
  std::string error;
  Check(EvaluateDescriptorSource(fixture.program,
                                 fixture.program.info.buffers[0].source, 4,
                                 runtime, descriptor, &error) &&
            descriptor.dwords[0] == user_data[0],
        "loop-invariant descriptor phi was not evaluated through typed SSA");
}

void TestAddressMaterializationAndSpecialization() {
  Fixture fixture;
  const auto based =
      fixture.Address(fixture.UserData(0), fixture.UserData(1), 4);
  MemoryInfo global;
  global.kind = ResourceKind::Global;
  global.offset = static_cast<uint32_t>(-8);
  fixture.Emit(ValueOpcode::LoadAddressU32,
               {based, Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(global, 4));

  const auto undef = fixture.Emit(ValueOpcode::UndefU32);
  const auto unbased = fixture.Address(undef, undef, 8);
  MemoryInfo flat;
  flat.kind = ResourceKind::Flat;
  flat.address_is_full = true;
  fixture.Emit(ValueOpcode::StoreAddressU32,
               {unbased, Value(0u), Value(0u), Value(9u), Value(true)},
               fixture.AddMemory(flat, 8));
  fixture.PlanAndTrack();

  Check(fixture.program.info.addresses.size() == 2 &&
            !fixture.program.info.addresses[0].unbased &&
            fixture.program.info.addresses[0].min_offset == -8 &&
            fixture.program.info.addresses[1].unbased,
        "typed based and unbased addresses were classified incorrectly");
  std::array<uint32_t, 2> user_data{0x2008u, 0u};
  SrtRuntime runtime{.user_data = user_data, .flat_memory_base = 0x9000u};
  ResourceSnapshot snapshot;
  std::string error;
  Check(MaterializeResources(fixture.program, runtime, snapshot, &error),
        "address resources did not materialize");
  Check(snapshot.addresses.size() == 2 &&
            snapshot.addresses[0].guest_base == 0x2008u &&
            snapshot.addresses[0].binding_base == 0x2000u &&
            snapshot.addresses[1].binding_base == 0x9000u,
        "materialized address windows are incorrect");
  Check(SpecializeResources(fixture.program, snapshot, &error) &&
            fixture.program.info.addresses[0].specialized_base == 8u &&
            fixture.program.info.addresses[1].specialized_base == 0x9000u,
        "typed address specialization was not applied");
}

void TestExecMaskedFlatAddressProvenance() {
  Fixture fixture;
  const auto low_root = fixture.UserData(0);
  const auto high_root = fixture.UserData(1);
  const auto active =
      fixture.Emit(ValueOpcode::INotEqual32, {fixture.UserData(2), Value(0u)});
  const auto inactive_low = fixture.Emit(ValueOpcode::UndefU32);
  const auto inactive_high = fixture.Emit(ValueOpcode::UndefU32);
  const auto low =
      fixture.Emit(ValueOpcode::SelectU32, {active, low_root, inactive_low});
  const auto high =
      fixture.Emit(ValueOpcode::SelectU32, {active, high_root, inactive_high});
  const auto address = fixture.Address(low, high, 0xa4);
  MemoryInfo flat;
  flat.kind = ResourceKind::Flat;
  flat.address_is_full = true;
  fixture.Emit(ValueOpcode::LoadAddressU8, {address, low, high, active},
               fixture.AddMemory(flat, 0xa4));
  fixture.PlanAndTrack();

  Check(fixture.program.info.addresses.size() == 1 &&
            !fixture.program.info.addresses[0].unbased,
        "exec-masked FLAT address lost its active user-data root");
  std::array<uint32_t, 3> user_data{0x23456780u, 1u, 1u};
  SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  std::string error;
  Check(MaterializeResources(fixture.program, runtime, snapshot, &error) &&
            snapshot.addresses.size() == 1 &&
            snapshot.addresses[0].guest_base == 0x0000000123456780ull &&
            snapshot.addresses[0].binding_base == 0x0000000123450000ull,
        "exec-masked FLAT address materialized the wrong user-data root");

  Fixture mismatch;
  const auto mismatch_active = mismatch.Emit(ValueOpcode::INotEqual32,
                                             {mismatch.UserData(2), Value(0u)});
  const auto other_active =
      mismatch.Emit(ValueOpcode::LogicalNot, {mismatch_active});
  const auto mismatch_low = mismatch.Emit(
      ValueOpcode::SelectU32, {mismatch_active, mismatch.UserData(0),
                               mismatch.Emit(ValueOpcode::UndefU32)});
  const auto mismatch_high = mismatch.Emit(
      ValueOpcode::SelectU32, {mismatch_active, mismatch.UserData(1),
                               mismatch.Emit(ValueOpcode::UndefU32)});
  const auto mismatch_address =
      mismatch.Address(mismatch_low, mismatch_high, 0xa4);
  mismatch.Emit(ValueOpcode::LoadAddressU8,
                {mismatch_address, mismatch_low, mismatch_high, other_active},
                mismatch.AddMemory(flat, 0xa4));
  mismatch.PlanAndTrack();
  Check(mismatch.program.info.addresses.size() == 1 &&
            mismatch.program.info.addresses[0].unbased,
        "FLAT address used a select arm guarded by a different active mask");
}

void TestBufferSwizzleSpecialization() {
  Fixture fixture;
  const auto handle = fixture.Buffer({fixture.UserData(0), fixture.UserData(1),
                                      fixture.UserData(2), fixture.UserData(3)},
                                     4);
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  memory.formatted = true;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {handle, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(memory, 4));
  fixture.PlanAndTrack();

  constexpr auto swizzle = Libs::Graphics::DstSel(4, 5, 0, 1);
  std::array<uint32_t, 4> user_data{
      0, 16u << 16u, 1,
      swizzle |
          (static_cast<uint32_t>(
               Libs::Graphics::Prospero::BufferFormat::k32_32Float)
           << 12u) |
          (1u << 24u)};
  SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  std::string error;
  Check(MaterializeResources(fixture.program, runtime, snapshot, &error) &&
            SpecializeResources(fixture.program, snapshot, &error) &&
            fixture.program.info.buffers[0].descriptor_swizzle == swizzle &&
            ValidateResourceSpecialization(fixture.program, snapshot, &error),
        "buffer destination selectors were not specialized");

  snapshot.buffers[0].dwords[3] ^= 1u << 9u;
  Check(!ValidateResourceSpecialization(fixture.program, snapshot, &error),
        "buffer swizzle change did not invalidate specialization");
}

void TestShaderInfoAndBindingLayout() {
  Fixture fixture;
  const auto handle = fixture.Buffer(
      {fixture.UserData(3), fixture.UserData(4), Value(64u), Value(0u)}, 4);
  MemoryInfo buffer;
  buffer.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {handle, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(buffer, 4));
  fixture.Emit(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::GlobalInvocationId)),
       Value(2u)});
  fixture.Emit(ValueOpcode::BitwiseXor32, {Value(1u), Value(2u)});
  MemoryInfo gds;
  gds.kind = ResourceKind::Gds;
  fixture.Emit(ValueOpcode::WriteSharedU32, {Value(0u), Value(1u), Value(true)},
               fixture.AddMemory(gds, 8));
  fixture.PlanAndTrack();

  ShaderComputeInputInfo compute{};
  compute.dispatch_thread_dimensions = true;
  std::string error;
  Check(CollectShaderInfo(fixture.program, {.compute = &compute}, &error),
        "typed shader info collection failed");
  Check(fixture.program.info.has_bitwise_xor &&
            !fixture.program.info.inputs.empty() &&
            fixture.program.info.inputs[0].kind ==
                StageInputKind::GlobalInvocationId,
        "typed shader values were not reflected in shader info");

  Check(AllocateBindings(fixture.program, 0, &error),
        "typed binding allocation failed");
  Check(FindBinding(fixture.program.bindings, DescriptorBindingKind::Buffers) !=
                nullptr &&
            FindBinding(fixture.program.bindings, DescriptorBindingKind::Gds) !=
                nullptr &&
            FindBinding(fixture.program.bindings,
                        DescriptorBindingKind::UserData) == nullptr &&
            fixture.program.bindings.push_constant_size ==
                fixture.program.bindings.ShaderDataDwords() * sizeof(uint32_t),
        "typed resources were not assigned native bindings");
  Check(NativeBinding(ShaderType::Compute, DescriptorBindingKind::Buffers) ==
                static_cast<uint32_t>(DescriptorBindingKind::Buffers) &&
            NativeBinding(ShaderType::Vertex, DescriptorBindingKind::Buffers) ==
                static_cast<uint32_t>(DescriptorBindingKind::Buffers) &&
            NativeBinding(ShaderType::Pixel, DescriptorBindingKind::Buffers) ==
                static_cast<uint32_t>(DescriptorBindingKind::Count) +
                    static_cast<uint32_t>(DescriptorBindingKind::Buffers),
        "fixed stage binding ranges are inconsistent");
  Check(fixture.program.bindings.user_data_registers ==
            std::vector<uint32_t>({3u, 4u}),
        "binding layout did not collect live typed user-data values");
}

void TestGraphicsPushConstantLayout() {
  const auto AddUserData = [](Fixture &fixture, uint32_t count) {
    for (uint32_t index = 0; index < count; index++) {
      fixture.Emit(ValueOpcode::ReferenceU32, {fixture.UserData(index)});
    }
    fixture.program.shader_info_complete = true;
  };

  std::string error;
  Fixture vertex(ShaderType::Vertex);
  AddUserData(vertex, 9);
  Check(AllocateBindings(vertex.program, 0, &error) &&
            vertex.program.bindings.push_constant_offset == 0 &&
            vertex.program.bindings.push_constant_size == 9 * sizeof(uint32_t),
        "vertex shader was not placed at the start of the graphics push bank");

  const auto pixel_offset = vertex.program.bindings.push_constant_size;
  Fixture pixel(ShaderType::Pixel);
  AddUserData(pixel, 4);
  Check(
      AllocateBindings(pixel.program, pixel_offset, &error) &&
          pixel.program.bindings.push_constant_offset == pixel_offset &&
          pixel.program.bindings.push_constant_size == 4 * sizeof(uint32_t) &&
          FindBinding(pixel.program.bindings,
                      DescriptorBindingKind::UserData) == nullptr,
      "pixel shader did not follow the vertex data in the graphics push bank");

  Fixture edge(ShaderType::Pixel);
  AddUserData(edge, 1);
  Check(AllocateBindings(edge.program,
                         NativePushConstantSize - sizeof(uint32_t), &error) &&
            edge.program.bindings.push_constant_size == sizeof(uint32_t),
        "last aligned push-constant dword did not fit in the graphics bank");

  Fixture spill(ShaderType::Pixel);
  AddUserData(spill, 32);
  Check(
      AllocateBindings(spill.program, pixel_offset, &error) &&
          spill.program.bindings.push_constant_size == 0 &&
          FindBinding(spill.program.bindings,
                      DescriptorBindingKind::UserData) != nullptr,
      "pixel shader overlapping the vertex push data did not spill to storage");

  Fixture full(ShaderType::Pixel);
  AddUserData(full, 1);
  Check(AllocateBindings(full.program, NativePushConstantSize, &error) &&
            full.program.bindings.push_constant_size == 0 &&
            FindBinding(full.program.bindings,
                        DescriptorBindingKind::UserData) != nullptr,
        "full graphics push bank did not spill pixel user data to storage");

  Fixture invalid(ShaderType::Pixel);
  AddUserData(invalid, 1);
  Check(!AllocateBindings(invalid.program,
                          NativePushConstantSize + sizeof(uint32_t), &error) &&
            !invalid.program.binding_layout_complete,
        "push-constant placement beyond the graphics bank was accepted");
}

void TestResourceLimitIsTransactional() {
  Fixture fixture;
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  for (uint32_t index = 0; index <= ShaderInfo::MaxBuffers; index++) {
    const auto handle = fixture.Buffer(
        {Value(index), Value(index + 1u), Value(index + 2u), Value(index + 3u)},
        index * 4u);
    fixture.Emit(ValueOpcode::LoadBufferU32,
                 {handle, Value(0u), Value(0u), Value(0u), Value(true)},
                 fixture.AddMemory(memory, index * 4u));
  }
  std::string error;
  Check(BuildSrtPlan(fixture.program, &error),
        "SRT plan failed before resource-limit test");
  Check(!TrackResources(fixture.program, &error) &&
            error.find("buffer resource limit exceeded") != std::string::npos &&
            !fixture.program.resource_tracking_complete &&
            fixture.program.info.buffers.empty() &&
            fixture.program.values->descriptor_sources.empty(),
        "resource-limit failure partially mutated typed resource state");
}

void TestMalformedMemoryKindsRejected() {
  {
    Fixture fixture;
    const auto address = fixture.Address(Value(0u), Value(0u), 4);
    MemoryInfo memory;
    memory.kind = ResourceKind::Buffer;
    fixture.Emit(ValueOpcode::StoreAddressU32,
                 {address, Value(0u), Value(0u), Value(1u), Value(true)},
                 fixture.AddMemory(memory, 4));
    std::string error;
    Check(BuildSrtPlan(fixture.program, &error) &&
              !TrackResources(fixture.program, &error) &&
              error.find("address operation has invalid resource kind") !=
                  std::string::npos,
          "resource tracking accepted an address opcode with buffer metadata");
  }
  {
    Fixture fixture;
    const auto image =
        fixture.Image({Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
                       Value(0u), Value(0u), Value(0u)},
                      8);
    MemoryInfo memory;
    memory.kind = ResourceKind::Flat;
    fixture.Emit(ValueOpcode::ImageRead,
                 {image, fixture.ImageAddress(), Value(true)},
                 fixture.AddMemory(memory, 8));
    std::string error;
    Check(BuildSrtPlan(fixture.program, &error) &&
              !TrackResources(fixture.program, &error) &&
              error.find("image operation has invalid resource kind") !=
                  std::string::npos,
          "resource tracking accepted an image opcode with address metadata");
  }
  {
    Fixture fixture;
    const auto image =
        fixture.Image({Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
                       Value(0u), Value(0u), Value(0u)},
                      12);
    MemoryInfo memory;
    memory.kind = ResourceKind::StorageImage;
    fixture.Emit(ValueOpcode::ImageRead,
                 {image, fixture.ImageAddress(), Value(true)},
                 fixture.AddMemory(memory, 12));
    std::string error;
    Check(BuildSrtPlan(fixture.program, &error) &&
              !TrackResources(fixture.program, &error) &&
              error.find("image operation has invalid resource kind") !=
                  std::string::npos,
          "resource tracking accepted a sampled read with storage metadata");
  }
  {
    Fixture fixture;
    const auto image =
        fixture.Image({Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
                       Value(0u), Value(0u), Value(0u)},
                      16);
    MemoryInfo memory;
    memory.kind = ResourceKind::StorageImage;
    fixture.Emit(ValueOpcode::ImageAtomicIAdd32,
                 {image, fixture.ImageAddress(), Value(1u), Value(true)},
                 fixture.AddMemory(memory, 16));
    std::string error;
    Check(
        BuildSrtPlan(fixture.program, &error) &&
            !TrackResources(fixture.program, &error) &&
            error.find("image operation has invalid resource kind") !=
                std::string::npos,
        "resource tracking accepted a uint atomic with float storage metadata");
  }
}

} // namespace

int main() {
  try {
    const auto Run = [](const char *name, auto test) {
      try {
        test();
      } catch (const std::exception &exception) {
        throw std::runtime_error(std::string(name) + ": " + exception.what());
      }
    };
    Run("dense buffers", TestDenseBufferTracking);
    Run("scalar/vector alias", TestScalarAndVectorBufferAlias);
    Run("runtime unsigned min", TestRuntimeUnsignedMinDescriptor);
    Run("uniform WQM buffer descriptor", TestUniformWqmBufferDescriptor);
    Run("uniform floating WQM buffer descriptor",
        TestUniformFloatingWqmBufferDescriptor);
    Run("images and samplers", TestImagesSamplersAndAliases);
    Run("SampleAdjust sampler scratch", TestSampleAdjustSamplerScratch);
    Run("dynamic storage mips", TestDynamicStorageMipTracking);
    Run("invariant indirect images", TestInvariantIndirectImageMaterialization);
    Run("direct indirect images", TestDirectIndirectImageMaterialization);
    Run("grouped ReadLane direct images", TestReadLaneDirectIndirectImageTracking);
    Run("ReadLane direct material keys",
        TestReadLaneDirectMaterialKeyMaterialization);
    Run("loop-indexed direct images",
        TestLoopIndexedDirectImageMaterialization);
    Run("mixed-dimension indirect images",
        TestMixedDimensionIndirectImageSpecialization);
    Run("SRT runtime", TestSrtFlatteningAndRuntimeMemoization);
    Run("dynamic SRT", TestDynamicSrtReadRemainsExplicit);
    Run("phi validation", TestPhiValidation);
    Run("control-dependent image phi", TestControlDependentImagePhi);
    Run("zero-initialized descriptor phi", TestZeroInitializedDescriptorPhi);
    Run("inactive descriptor phi edge", TestInactiveDescriptorPhiEdge);
    Run("runtime-rooted loop", TestLoopCycleEnteredThroughRuntimeValue);
    Run("invariant loop phi", TestInvariantLoopPhi);
    Run("address materialization", TestAddressMaterializationAndSpecialization);
    Run("exec-masked FLAT address", TestExecMaskedFlatAddressProvenance);
    Run("buffer swizzle specialization", TestBufferSwizzleSpecialization);
    Run("shader info and bindings", TestShaderInfoAndBindingLayout);
    Run("graphics push constants", TestGraphicsPushConstantLayout);
    Run("resource limit", TestResourceLimitIsTransactional);
    Run("malformed memory kinds", TestMalformedMemoryKindsRejected);
  } catch (const std::exception &exception) {
    std::cerr << "resource tracking test failed: " << exception.what() << '\n';
    return 1;
  }
  std::cout << "resource tracking tests passed\n";
  return 0;
}

// The full emulator supplies these assertion hooks through common. This focused
// target links only fmt; keep assertion failures observable without widening
// its legacy build manifest.
namespace Common {
int DbgExitIfHandler(const char *expression, const char *file, int line) {
  throw std::runtime_error(std::string("typed IR assertion: ") + expression +
                           " at " + file + ':' + std::to_string(line));
}

void DbgExit(int) { throw std::runtime_error("typed IR assertion failed"); }
} // namespace Common

namespace Libs::Graphics {
bool ShaderPixelParameterIsFlat(const ShaderPixelInputInfo &info,
                                uint32_t input) {
  constexpr uint32_t FlatShade = 0x00000400u;
  const bool custom =
      input < 32u && (info.custom_interpolation_mask & (1u << input)) != 0;
  return input < info.input_num &&
         (info.interpolator_settings[input] & FlatShade) != 0 && !custom;
}
} // namespace Libs::Graphics

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/ValueProgram.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.cpp"
