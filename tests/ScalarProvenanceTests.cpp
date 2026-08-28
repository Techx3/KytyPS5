#include "graphics/shader/recompiler/ir/ValueProgram.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/recompiler/ir/passes/Wave64MaterialBatch.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderType;

void Check(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct Fixture {
  Program program;

  explicit Fixture(uint32_t block_count = 1) {
    program.stage = ShaderType::Compute;
    program.shader_hash = 0x12345678u;
    program.user_data_base = 2;
    program.values = std::make_shared<ValueProgram>();
    for (uint32_t index = 0; index < block_count; index++) {
      program.values->block_storage.push_back(std::make_unique<Block>());
      auto *block = program.values->block_storage.back().get();
      program.values->blocks.push_back(block);
      program.values->block_info.push_back({.id = index});
    }
  }

  Block &BlockAt(uint32_t index = 0) { return *program.values->blocks[index]; }

  Value Emit(ValueOpcode opcode, std::initializer_list<Value> args = {},
             uint64_t flags = 0, uint32_t block = 0) {
    return Value(&BlockAt(block).AppendNewInst(opcode, args, flags));
  }

  Value EmitMemory(ValueOpcode opcode, std::initializer_list<Value> args,
                   uint32_t memory, uint32_t pc = 0x40, uint32_t block = 0) {
    MemoryFlags flags{.index = memory, .pc = pc};
    uint64_t bits = 0;
    std::memcpy(&bits, &flags, sizeof(flags));
    return Emit(opcode, args, bits, block);
  }

  uint32_t AddMemory(ResourceKind kind, int32_t offset = 0) {
    MemoryInfo info;
    info.kind = kind;
    info.offset = static_cast<uint32_t>(offset);
    program.values->memory_info.push_back(info);
    return static_cast<uint32_t>(program.values->memory_info.size() - 1u);
  }

  void Plan() {
    std::string error;
    Check(BuildSrtPlan(program, &error), error.c_str());
  }
};

struct TestMemory {
  std::unordered_map<uint64_t, uint32_t> words;
  uint32_t reads = 0;
};

bool ReadMemory(void *userdata, uint64_t address, uint32_t *value) {
  auto &memory = *static_cast<TestMemory *>(userdata);
  const auto it = memory.words.find(address);
  if (it == memory.words.end()) {
    return false;
  }
  *value = it->second;
  memory.reads++;
  return true;
}

Value Address(Fixture &fixture, Value low, Value high, uint32_t block = 0) {
  return fixture.Emit(ValueOpcode::GetAddressResource, {low, high}, 0, block);
}

Value RawRead(Fixture &fixture, Value address, Value offset, uint32_t memory,
              uint32_t block = 0) {
  return fixture.EmitMemory(ValueOpcode::LoadAddressU32,
                            {address, offset, Value(0u), Value(true)}, memory,
                            0x80, block);
}

void TestImmediateFlatteningAndGvn() {
  Fixture fixture;
  const auto memory = fixture.AddMemory(ResourceKind::ScalarAddress, 0x20);
  const auto first = RawRead(
      fixture, Address(fixture, Value(0x1000u), Value(0u)), Value(0u), memory);
  const auto second = RawRead(
      fixture, Address(fixture, Value(0x1000u), Value(0u)), Value(0u), memory);
  fixture.Emit(ValueOpcode::GetBufferResource,
               {first, second, Value(16u), Value(0u)});

  fixture.Plan();
  Check(fixture.program.values->srt_reads.size() == 1,
        "equivalent typed scalar reads were not coalesced");
  Check(fixture.program.values->dynamic_reads.empty(),
        "immediate scalar read was classified as dynamic");
  Check(fixture.program.values->memory_info[memory].planning_only,
        "flattened raw read was not kept as a planning-only root");

  TestMemory memory_image{{{0x1020u, 0xfeedbeefu}}};
  SrtRuntime runtime{.read_memory = ReadMemory, .userdata = &memory_image};
  std::vector<uint32_t> flat;
  std::string error;
  Check(WalkSrt(fixture.program, runtime, flat, &error), error.c_str());
  Check(flat == std::vector<uint32_t>{0xfeedbeefu} && memory_image.reads == 1,
        "flattened SRT did not evaluate its canonical read once");
}

void TestRawScalarComponentAlignment() {
  Fixture fixture;
  const auto memory = fixture.AddMemory(ResourceKind::ScalarAddress, 2);
  const auto read = RawRead(
      fixture, Address(fixture, Value(0x1003u), Value(0u)), Value(2u), memory);
  fixture.Emit(ValueOpcode::GetBufferResource,
               {read, Value(0u), Value(16u), Value(0u)});
  fixture.Plan();

  TestMemory memory_image{{{0x1000u, 0x12345678u}}};
  SrtRuntime runtime{.read_memory = ReadMemory, .userdata = &memory_image};
  std::vector<uint32_t> flat;
  std::string error;
  Check(WalkSrt(fixture.program, runtime, flat, &error), error.c_str());
  Check(
      flat == std::vector<uint32_t>{0x12345678u} && memory_image.reads == 1,
      "raw scalar base, immediate, and offset were not aligned independently");
}

void TestScalarMemoryDomainMismatchFails() {
  Fixture raw;
  const auto raw_memory = raw.AddMemory(ResourceKind::ScalarBuffer);
  RawRead(raw, Address(raw, Value(0x1000u), Value(0u)), Value(0u), raw_memory);
  std::string error;
  Check(!BuildSrtPlan(raw.program, &error) &&
            error.find("incompatible scalar memory metadata") !=
                std::string::npos,
        "raw scalar load accepted descriptor-buffer metadata");

  Fixture buffer;
  const auto buffer_memory = buffer.AddMemory(ResourceKind::ScalarAddress);
  const auto resource =
      buffer.Emit(ValueOpcode::GetBufferResource,
                  {Value(0x1000u), Value(0u), Value(16u), Value(0u)});
  buffer.EmitMemory(ValueOpcode::ReadConstBuffer, {resource, Value(0u)},
                    buffer_memory);
  error.clear();
  Check(!BuildSrtPlan(buffer.program, &error) &&
            error.find("incompatible scalar memory metadata") !=
                std::string::npos,
        "descriptor scalar load accepted raw-address metadata");
}

void TestDynamicReadRemainsTyped() {
  Fixture fixture;
  const auto memory = fixture.AddMemory(ResourceKind::ScalarAddress);
  const auto offset = fixture.Emit(ValueOpcode::GetUserData,
                                   {Value(static_cast<ScalarReg>(2))});
  const auto read = RawRead(
      fixture, Address(fixture, Value(0x1000u), Value(0u)), offset, memory);
  fixture.Emit(ValueOpcode::GetBufferResource,
               {read, Value(0u), Value(16u), Value(0u)});

  fixture.Plan();
  Check(fixture.program.values->srt_reads.empty() &&
            fixture.program.values->dynamic_reads == std::vector<Value>{read},
        "dynamic scalar read received a fake flattened slot");
}

void TestNestedSrtWalk() {
  Fixture fixture;
  const auto memory = fixture.AddMemory(ResourceKind::ScalarAddress);
  const auto pointer = RawRead(
      fixture, Address(fixture, Value(0x1000u), Value(0u)), Value(0u), memory);
  const auto value =
      RawRead(fixture, Address(fixture, pointer, Value(0u)), Value(0u), memory);
  fixture.Emit(ValueOpcode::GetBufferResource,
               {value, Value(0u), Value(16u), Value(0u)});
  fixture.Plan();

  TestMemory memory_image{{{0x1000u, 0x2000u}, {0x2000u, 0xabcdef01u}}};
  SrtRuntime runtime{.read_memory = ReadMemory, .userdata = &memory_image};
  std::vector<uint32_t> flat;
  std::string error;
  Check(WalkSrt(fixture.program, runtime, flat, &error), error.c_str());
  Check(flat == std::vector<uint32_t>({0x2000u, 0xabcdef01u}),
        "nested typed SRT reads were not evaluated in dependency order");
}

void TestShaderBaseAndUserData() {
  Fixture fixture;
  const auto base = fixture.Emit(ValueOpcode::GetShaderBase);
  const auto low =
      fixture.Emit(ValueOpcode::CompositeExtractU64, {base, Value(0u)});
  const auto high =
      fixture.Emit(ValueOpcode::CompositeExtractU64, {base, Value(1u)});
  const auto user = fixture.Emit(ValueOpcode::GetUserData,
                                 {Value(static_cast<ScalarReg>(2))});
  const auto sum = fixture.Emit(ValueOpcode::IAdd32, {user, Value(4u)});
  fixture.Plan();
  fixture.program.values->descriptor_sources.push_back(
      {.dwords = {low, high, sum}, .dword_count = 3});

  const std::array user_data{0x20u};
  SrtRuntime runtime{.user_data = user_data,
                     .shader_base = 0x12345678abcdef00ull};
  DescriptorValue result;
  std::string error;
  Check(EvaluateDescriptorSource(fixture.program, 0, 0x90, runtime, result,
                                 &error),
        error.c_str());
  Check(result.dword_count == 3 && result.dwords[0] == 0xabcdef00u &&
            result.dwords[1] == 0x12345678u && result.dwords[2] == 0x24u,
        "shader-relative typed descriptor expression evaluated incorrectly");
}

void TestCarryAndBitFields() {
  Fixture fixture;
  const auto carry =
      fixture.Emit(ValueOpcode::IAddCarry32, {Value(0xffffffffu), Value(2u)});
  const auto low =
      fixture.Emit(ValueOpcode::CompositeExtractU32x2, {carry, Value(0u)});
  const auto high =
      fixture.Emit(ValueOpcode::CompositeExtractU32x2, {carry, Value(1u)});
  const auto inserted =
      fixture.Emit(ValueOpcode::BitFieldInsert,
                   {Value(0u), Value(0x89abcdefu), Value(0u), Value(32u)});
  const auto sign = fixture.Emit(ValueOpcode::BitFieldSExtract,
                                 {Value(0x000000f0u), Value(4u), Value(4u)});
  fixture.Plan();
  fixture.program.values->descriptor_sources.push_back(
      {.dwords = {low, high, inserted, sign}, .dword_count = 4});

  DescriptorValue result;
  std::string error;
  Check(EvaluateDescriptorSource(fixture.program, 0, 0, {}, result, &error),
        error.c_str());
  Check(result.dwords[0] == 1u && result.dwords[1] == 1u &&
            result.dwords[2] == 0x89abcdefu && result.dwords[3] == 0xffffffffu,
        "typed carry or bit-field runtime evaluation is incorrect");
}

void TestInvariantAndDivergentPhi() {
  Fixture fixture(3);
  auto &invariant = fixture.BlockAt(2).AppendNewInst(ValueOpcode::Phi);
  invariant.SetFlags(Type::U32);
  invariant.AddPhiOperand(&fixture.BlockAt(0), Value(7u));
  invariant.AddPhiOperand(&fixture.BlockAt(1), Value(7u));
  auto &divergent = fixture.BlockAt(2).AppendNewInst(ValueOpcode::Phi);
  divergent.SetFlags(Type::U32);
  divergent.AddPhiOperand(&fixture.BlockAt(0), Value(7u));
  divergent.AddPhiOperand(&fixture.BlockAt(1), Value(9u));
  fixture.Plan();
  fixture.program.values->descriptor_sources.push_back(
      {.dwords = {Value(&invariant)}, .dword_count = 1});
  fixture.program.values->descriptor_sources.push_back(
      {.dwords = {Value(&divergent)}, .dword_count = 1});

  DescriptorValue result;
  std::string error;
  Check(
      EvaluateDescriptorSource(fixture.program, 0, 0x100, {}, result, &error) &&
          result.dwords[0] == 7u,
      "loop-invariant typed phi was rejected");
  result.dword_count = 4;
  result.dwords[0] = 0xdeadbeefu;
  Check(!EvaluateDescriptorSource(fixture.program, 1, 0x104, {}, result,
                                  &error) &&
            result.dword_count == 4 && result.dwords[0] == 0xdeadbeefu &&
            error.find("control-dependent phi") != std::string::npos,
        "divergent phi did not fail transactionally");
}

void TestControlDependentStandaloneLoadStaysTyped() {
  Fixture fixture(3);
  const auto memory = fixture.AddMemory(ResourceKind::ScalarAddress);
  auto &base = fixture.BlockAt(2).AppendNewInst(ValueOpcode::Phi);
  base.SetFlags(Type::U32);
  base.AddPhiOperand(&fixture.BlockAt(0), Value(0x1000u));
  base.AddPhiOperand(&fixture.BlockAt(1), Value(0x2000u));
  const auto read =
      RawRead(fixture, Address(fixture, Value(&base), Value(0u), 2), Value(0u),
              memory, 2);
  fixture.Plan();
  Check(fixture.program.values->srt_reads.empty() &&
            read.ResolveInstruction()->GetOpcode() ==
                ValueOpcode::LoadAddressU32 &&
            !fixture.program.values->memory_info[memory].planning_only,
        "control-dependent standalone scalar load was flattened into a host "
        "snapshot");
}

void TestRuntime64BitDescriptorOps() {
  Fixture fixture;
  const auto shifted = fixture.Emit(ValueOpcode::ShiftLeftLogical64,
                                    {Value(uint64_t{0x1234u}), Value(32u)});
  const auto masked =
      fixture.Emit(ValueOpcode::BitwiseAnd64,
                   {shifted, Value(uint64_t{0x0000ffff00000000ull})});
  const auto combined = fixture.Emit(
      ValueOpcode::IAdd64, {masked, Value(uint64_t{0x000000010000abcdull})});
  const auto low =
      fixture.Emit(ValueOpcode::CompositeExtractU64, {combined, Value(0u)});
  const auto high =
      fixture.Emit(ValueOpcode::CompositeExtractU64, {combined, Value(1u)});
  fixture.Plan();
  fixture.program.values->descriptor_sources.push_back(
      {.dwords = {low, high}, .dword_count = 2});

  DescriptorValue result;
  std::string error;
  Check(EvaluateDescriptorSource(fixture.program, 0, 0, {}, result, &error) &&
            result.dwords[0] == 0xabcdu && result.dwords[1] == 0x1235u,
        "64-bit typed descriptor arithmetic evaluation is incorrect");
}

void TestConstantBufferBounds() {
  Fixture fixture;
  const auto memory = fixture.AddMemory(ResourceKind::ScalarBuffer);
  const auto buffer =
      fixture.Emit(ValueOpcode::GetBufferResource,
                   {Value(0x3000u), Value(0u), Value(16u), Value(0u)});
  const auto read = fixture.EmitMemory(ValueOpcode::ReadConstBuffer,
                                       {buffer, Value(12u)}, memory);
  fixture.Emit(ValueOpcode::GetBufferResource,
               {read, Value(0u), Value(16u), Value(0u)});
  fixture.Plan();

  TestMemory memory_image{{{0x300cu, 0xa5a5a5a5u}}};
  SrtRuntime runtime{.read_memory = ReadMemory, .userdata = &memory_image};
  std::vector<uint32_t> flat;
  std::string error;
  Check(WalkSrt(fixture.program, runtime, flat, &error) &&
            flat == std::vector<uint32_t>{0xa5a5a5a5u},
        error.c_str());

  Fixture overflow;
  const auto overflow_memory = overflow.AddMemory(ResourceKind::ScalarBuffer);
  const auto overflow_buffer =
      overflow.Emit(ValueOpcode::GetBufferResource,
                    {Value(0x3000u), Value(0u), Value(16u), Value(0u)});
  const auto overflow_read =
      overflow.EmitMemory(ValueOpcode::ReadConstBuffer,
                          {overflow_buffer, Value(16u)}, overflow_memory);
  overflow.Emit(ValueOpcode::GetBufferResource,
                {overflow_read, Value(0u), Value(16u), Value(0u)});
  overflow.Plan();
  flat = {0x55u};
  Check(!WalkSrt(overflow.program, runtime, flat, &error) &&
            flat == std::vector<uint32_t>{0x55u} &&
            error.find("exceeds size") != std::string::npos,
        "out-of-bounds constant-buffer walk was not transactional");
}

void TestReadLaneElimination() {
  Fixture fixture;
  const auto undef = fixture.Emit(ValueOpcode::UndefU32);
  const auto write = fixture.Emit(ValueOpcode::WriteLane,
                                  {undef, Value(0xdeadbeefu), Value(5u)});
  const auto read = fixture.Emit(ValueOpcode::ReadLane, {write, Value(5u)});
  const auto use = fixture.Emit(ValueOpcode::IAdd32, {read, Value(1u)});
  const auto stats = EliminateReadLane(*fixture.program.values, 64);
  Check(stats.rewritten_reads == 1 &&
            use.ResolveInstruction()->Arg(0).Resolve() == Value(0xdeadbeefu),
        "fixed-lane typed read was not rewritten from its SSA write chain");

  const auto selector = fixture.Emit(ValueOpcode::GetUserData,
                                     {Value(static_cast<ScalarReg>(2))});
  const auto dynamic = fixture.Emit(ValueOpcode::ReadLane, {write, selector});
  const auto dynamic_use = fixture.Emit(ValueOpcode::IAdd32, {dynamic, Value(1u)});
  Check(EliminateReadLane(*fixture.program.values, 64).rewritten_reads == 0,
        "dynamic-lane read was rewritten unsafely");

  fixture.program.user_data_count = 1;
  const auto uniform = fixture.Emit(ValueOpcode::IAdd32, {selector, Value(7u)});
  const auto uniform_read =
      fixture.Emit(ValueOpcode::ReadLane, {uniform, selector});
  const auto uniform_use =
      fixture.Emit(ValueOpcode::IAdd32, {uniform_read, Value(1u)});
  Check(EliminateReadLane(*fixture.program.values, 64, &fixture.program)
                .rewritten_reads == 1 &&
            uniform_use.ResolveInstruction()->Arg(0).Resolve() == uniform.Resolve() &&
            dynamic_use.ResolveInstruction()->Arg(0).Resolve() == dynamic.Resolve(),
        "runtime-uniform read-lane elimination was not conservative");

  const auto lane_id = fixture.Emit(ValueOpcode::LaneId);
  const auto divergent_read =
      fixture.Emit(ValueOpcode::ReadLane, {lane_id, Value(0u)});
  const auto divergent_use =
      fixture.Emit(ValueOpcode::IAdd32, {divergent_read, Value(1u)});
  Check(EliminateReadLane(*fixture.program.values, 64, &fixture.program)
                .rewritten_reads == 0 &&
            divergent_use.ResolveInstruction()->Arg(0).Resolve() ==
                divergent_read.Resolve(),
        "lane-divergent read was rewritten as a scalar value");
}

template <typename T>
uint64_t FlagBits(T flags) {
  uint64_t bits = 0;
  std::memcpy(&bits, &flags, sizeof(flags));
  return bits;
}

struct MaterialBatchValues {
  Value aggregate;
  Value local_read;
  Value group_bits;
  Inst *mask_phi = nullptr;
};

MaterialBatchValues BuildWave64MaterialBatch(Fixture &fixture,
                                              bool wrong_dpp_control = false,
                                              bool per_invocation_bit = false) {
  fixture.program.stage = ShaderType::Pixel;
  auto *preheader = fixture.program.values->blocks[0];
  auto *header = fixture.program.values->blocks[1];
  auto *select = fixture.program.values->blocks[2];
  auto *sample_block = fixture.program.values->blocks[3];
  auto *latch = fixture.program.values->blocks[4];
  auto *exit = fixture.program.values->blocks[5];
  preheader->AddBranch(header);
  header->AddBranch(select);
  header->AddBranch(exit);
  select->AddBranch(sample_block);
  select->AddBranch(latch);
  sample_block->AddBranch(latch);
  latch->AddBranch(header);

  const auto seed = fixture.Emit(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(2))}, 0, 0);
  const auto enabled = fixture.Emit(ValueOpcode::IEqual32,
                                    {seed, Value(0u)}, 0, 0);
  const auto disabled =
      fixture.Emit(ValueOpcode::LogicalNot, {enabled}, 0, 0);
  const auto active =
      fixture.Emit(ValueOpcode::LogicalOr, {disabled, enabled}, 0, 0);
  Value row = seed;
  const std::array<uint16_t, 4> controls = {
      0x111u, 0x112u, wrong_dpp_control ? uint16_t{0x113u} : uint16_t{0x114u},
      0x118u};
  for (const auto control : controls) {
    const DppMoveFlags flags{.control = control,
                             .row_mask = 0xfu,
                             .bank_mask = 0xfu,
                             .fetch_inactive = false,
                             .bound_control = false};
    const auto moved = fixture.Emit(ValueOpcode::DppMoveU32, {row, active},
                                    FlagBits(flags), 0);
    const auto merged =
        fixture.Emit(ValueOpcode::BitwiseOr32, {moved, row}, 0, 0);
    row = fixture.Emit(ValueOpcode::DppUpdateU32,
                       {merged, row, active}, FlagBits(flags), 0);
  }
  const PermlaneFlags permute_flags{.x16 = true,
                                    .fetch_inactive = false,
                                    .bound_control = true};
  const auto permuted = fixture.Emit(
      ValueOpcode::Permlane16U32,
      {row, Value(0xffffffffu), Value(0xffffffffu), active},
      FlagBits(permute_flags), 0);
  const auto selected_permute =
      fixture.Emit(ValueOpcode::SelectU32, {active, permuted, seed}, 0, 0);
  const auto combined =
      fixture.Emit(ValueOpcode::BitwiseOr32, {row, selected_permute}, 0, 0);
  const auto reduction =
      fixture.Emit(ValueOpcode::SelectU32, {active, combined, row}, 0, 0);
  const auto read31 = fixture.Emit(ValueOpcode::ReadLane,
                                   {reduction, Value(31u)}, 0, 0);
  const auto read63 = fixture.Emit(ValueOpcode::ReadLane,
                                   {reduction, Value(63u)}, 0, 0);
  const auto aggregate = per_invocation_bit
                             ? fixture.Emit(ValueOpcode::SelectU32,
                                            {enabled, Value(1u), Value(0u)}, 0, 0)
                             : fixture.Emit(ValueOpcode::BitwiseOr32,
                                            {read31, read63}, 0, 0);

  auto &mask_phi_inst =
      fixture.BlockAt(1).AppendNewInst(ValueOpcode::Phi);
  mask_phi_inst.SetFlags(Type::U32);
  const Value mask_phi(&mask_phi_inst);
  const auto nonzero = fixture.Emit(ValueOpcode::INotEqual32,
                                    {Value(0u), mask_phi}, 0, 1);
  fixture.Emit(ValueOpcode::Reference, {nonzero}, 0, 1);

  Inst *high_mask_phi_inst = nullptr;
  Value selected_bit;
  const auto bit = fixture.Emit(ValueOpcode::FindILsb32, {mask_phi}, 0, 2);
  if (per_invocation_bit) {
    high_mask_phi_inst =
        &fixture.BlockAt(1).AppendNewInst(ValueOpcode::Phi);
    high_mask_phi_inst->SetFlags(Type::U32);
    const Value high_mask(high_mask_phi_inst);
    const auto high_bit =
        fixture.Emit(ValueOpcode::FindILsb32, {high_mask}, 0, 2);
    const auto guest_high_bit = fixture.Emit(ValueOpcode::IAdd32,
                                             {high_bit, Value(32u)}, 0, 2);
    const auto high_nonzero = fixture.Emit(ValueOpcode::INotEqual32,
                                           {high_mask, Value(0u)}, 0, 2);
    const auto fallback = fixture.Emit(ValueOpcode::SelectU32,
                                       {high_nonzero, guest_high_bit,
                                        Value(UINT32_MAX)}, 0, 2);
    selected_bit = fixture.Emit(ValueOpcode::SelectU32,
                                {nonzero, bit, fallback}, 0, 2);
  } else {
    selected_bit = bit;
  }

  const auto material = fixture.Emit(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(3))}, 0, 2);
  const auto lane_index = per_invocation_bit
                              ? fixture.Emit(ValueOpcode::BitwiseAnd32,
                                             {selected_bit, Value(63u)}, 0, 2)
                              : Value{};
  const auto selected_material =
      per_invocation_bit
          ? fixture.Emit(ValueOpcode::ReadLane, {material, lane_index}, 0, 2)
          : bit;
  const auto material_match =
      fixture.Emit(ValueOpcode::IEqual32,
                   {selected_material, material}, 0, 2);
  const auto descriptor_key = fixture.Emit(
      ValueOpcode::ShiftLeftLogical32,
      {selected_material, Value(5u)}, 0, 2);
  const auto descriptor_offset = fixture.Emit(
      ValueOpcode::IAdd32, {descriptor_key, Value(0x158u)}, 0, 2);
  const auto address = fixture.Emit(ValueOpcode::GetAddressResource,
                                    {Value(0x1000u), Value(0u)}, 0, 2);
  const auto descriptor = fixture.Emit(
      ValueOpcode::LoadAddressU32,
      {address, descriptor_offset, Value(0u), Value(true)}, 0, 2);
  const auto image = fixture.Emit(
      ValueOpcode::GetImageResource,
      {descriptor, descriptor, descriptor, descriptor, descriptor, descriptor,
       descriptor, descriptor},
      0, 2);
  const auto sampler = fixture.Emit(
      ValueOpcode::GetSamplerResource,
      {Value(0u), Value(0u), Value(0u), Value(0u)}, 0, 2);
  const auto image_address = fixture.Emit(
      ValueOpcode::MakeImageAddress,
      {Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
       Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
       Value(0u)},
      0, 2);
  const auto guard = fixture.Emit(ValueOpcode::LogicalAnd,
                                  {active, material_match}, 0, 2);
  const auto group_bits =
      per_invocation_bit
          ? fixture.Emit(ValueOpcode::SelectU32,
                         {guard, Value(1u), Value(0u)}, 0, 2)
          : Value{};
  const auto sampled = fixture.Emit(ValueOpcode::ImageSampleRaw,
                                    {image, sampler, image_address}, 0, 3);
  const auto component = fixture.Emit(
      ValueOpcode::CompositeExtractU32x4, {sampled, Value(0u)}, 0, 3);
  const auto merged = fixture.Emit(ValueOpcode::SelectU32,
                                   {guard, component, Value(0u)}, 0, 3);
  fixture.Emit(ValueOpcode::ReferenceU32, {merged}, 0, 3);

  Value updated;
  if (per_invocation_bit) {
    const auto inverted_group = fixture.Emit(ValueOpcode::BitwiseNot32,
                                             {group_bits}, 0, 4);
    updated = fixture.Emit(ValueOpcode::BitwiseAnd32,
                           {mask_phi, inverted_group}, 0, 4);
  } else {
    const auto bit_index = fixture.Emit(ValueOpcode::BitwiseAnd32,
                                        {bit, Value(31u)}, 0, 4);
    const auto clear_bit = fixture.Emit(ValueOpcode::ShiftLeftLogical32,
                                        {Value(1u), bit_index}, 0, 4);
    updated = fixture.Emit(ValueOpcode::BitwiseXor32,
                           {clear_bit, mask_phi}, 0, 4);
  }
  mask_phi_inst.AddPhiOperand(preheader, aggregate);
  mask_phi_inst.AddPhiOperand(latch, updated);
  if (high_mask_phi_inst != nullptr) {
    high_mask_phi_inst->AddPhiOperand(preheader, Value(0u));
    high_mask_phi_inst->AddPhiOperand(latch, Value(high_mask_phi_inst));
  }

  fixture.Emit(ValueOpcode::ReferenceU32, {seed}, 0, 5);
  return {.aggregate = aggregate,
          .local_read = read31,
          .group_bits = group_bits,
          .mask_phi = &mask_phi_inst};
}

void TestWave64MaterialBatchSplit() {
  Fixture positive(6);
  const auto values = BuildWave64MaterialBatch(positive);
  const auto stats =
      SplitWave64MaterialBatches(*positive.program.values, 64u, 32u);
  Check(stats.rewritten_masks == 1u &&
            values.mask_phi->Arg(0).Resolve() == values.local_read.Resolve(),
        "canonical read-only Wave64 material batch was not split for subgroup32");
  EliminateDeadCode(positive.program.values->blocks);
  uint32_t lane63_reads = 0;
  for (const auto *block : positive.program.values->blocks) {
    for (const auto &inst : *block) {
      lane63_reads +=
          inst.GetOpcode() == ValueOpcode::ReadLane &&
          inst.Arg(1).Resolve().IsImmediate() && inst.Arg(1).Resolve().U32() == 63u;
    }
  }
  Check(lane63_reads == 0u,
        "split material batch retained an out-of-range subgroup32 lane read");

  Fixture invocation_mask(6);
  const auto invocation_values =
      BuildWave64MaterialBatch(invocation_mask, false, true);
  const auto invocation_stats =
      SplitWave64MaterialBatches(*invocation_mask.program.values, 64u, 32u);
  Check(invocation_stats.rewritten_masks == 2u &&
            invocation_values.mask_phi->Arg(0).Resolve().TryInstruction() != nullptr &&
            invocation_values.mask_phi->Arg(0).Resolve().TryInstruction()->GetOpcode() ==
                ValueOpcode::CompositeExtractU32x4,
        "per-invocation material bit was not reconstructed as a subgroup ballot");
  const auto ballot_extract =
      invocation_values.mask_phi->Arg(0).Resolve().TryInstruction();
  Check(ballot_extract->Arg(0).Resolve().TryInstruction() != nullptr &&
            ballot_extract->Arg(0).Resolve().TryInstruction()->GetOpcode() ==
                ValueOpcode::Ballot,
        "material bit reconstruction did not emit a subgroup ballot");
  Check(invocation_values.group_bits.Resolve().TryInstruction() != nullptr &&
            invocation_values.group_bits.Resolve().TryInstruction()->GetOpcode() ==
                ValueOpcode::CompositeExtractU32x4 &&
            invocation_values.group_bits.Resolve().TryInstruction()
                    ->Arg(0)
                    .Resolve()
                    .TryInstruction() != nullptr &&
            invocation_values.group_bits.Resolve().TryInstruction()
                    ->Arg(0)
                    .Resolve()
                    .TryInstruction()
                    ->GetOpcode() == ValueOpcode::Ballot,
        "material group mask was not reconstructed as a subgroup ballot");

  Fixture native64(6);
  BuildWave64MaterialBatch(native64);
  Check(SplitWave64MaterialBatches(*native64.program.values, 64u, 64u)
                .rewritten_masks == 0u,
        "native subgroup64 material batch was rewritten unnecessarily");

  Fixture escaped(6);
  const auto escaped_values = BuildWave64MaterialBatch(escaped);
  escaped.Emit(ValueOpcode::ReferenceU32, {escaped_values.aggregate}, 0, 0);
  Check(SplitWave64MaterialBatches(*escaped.program.values, 64u, 32u)
                .rewritten_masks == 0u,
        "material mask escaping its batching loop was split unsafely");

  Fixture side_effect(6);
  BuildWave64MaterialBatch(side_effect);
  side_effect.Emit(ValueOpcode::WriteSharedU32,
                   {Value(0u), Value(1u), Value(true)}, 0, 3);
  Check(SplitWave64MaterialBatches(*side_effect.program.values, 64u, 32u)
                .rewritten_masks == 0u,
        "material loop with observable writes was split unsafely");

  Fixture loop_counter(6);
  BuildWave64MaterialBatch(loop_counter);
  auto &counter_phi = *loop_counter.BlockAt(1).PrependNewInst(
      loop_counter.BlockAt(1).begin(), ValueOpcode::Phi);
  counter_phi.SetFlags(Type::U32);
  const Value counter(&counter_phi);
  const auto updated_counter = loop_counter.Emit(
      ValueOpcode::IAdd32, {counter, Value(1u)}, 0, 4);
  counter_phi.AddPhiOperand(loop_counter.program.values->blocks[0], Value(0u));
  counter_phi.AddPhiOperand(loop_counter.program.values->blocks[4],
                            updated_counter);
  loop_counter.Emit(ValueOpcode::ReferenceU32, {counter}, 0, 5);
  Check(SplitWave64MaterialBatches(*loop_counter.program.values, 64u, 32u)
                .rewritten_masks == 0u,
        "material loop with an iteration-count result was split unsafely");

  Fixture noncanonical(6);
  BuildWave64MaterialBatch(noncanonical, true);
  Check(SplitWave64MaterialBatches(*noncanonical.program.values, 64u, 32u)
                .rewritten_masks == 0u,
        "noncanonical DPP reduction was accepted as a material batch");

}

void TestOptimizationPipeline() {
  Fixture fixture;
  const auto sum = fixture.Emit(ValueOpcode::IAdd32, {Value(40u), Value(2u)});
  const auto kept = fixture.Emit(ValueOpcode::BitwiseOr32, {sum, Value(0u)});
  fixture.Emit(ValueOpcode::ReferenceU32, {kept});
  fixture.Emit(ValueOpcode::IMul32, {Value(6u), Value(7u)});

  ConstantPropagationPass(fixture.program.values->blocks);
  RemoveIdentities(fixture.program.values->blocks);
  EliminateDeadCode(fixture.program.values->blocks);

  const auto &instructions = fixture.BlockAt().Instructions();
  Check(instructions.size() == 1 &&
            instructions.front().GetOpcode() == ValueOpcode::ReferenceU32 &&
            instructions.front().Arg(0).Resolve() == Value(42u),
        "typed constant propagation, identity folding, or dead-code "
        "elimination regressed");
}

void TestControlFlowValueSurvivesReadLaneFolding() {
  Fixture fixture(3);
  auto *entry = fixture.program.values->blocks[0];
  auto *taken = fixture.program.values->blocks[1];
  auto *other = fixture.program.values->blocks[2];
  entry->AddBranch(taken);
  entry->AddBranch(other);

  auto &entry_info = fixture.program.values->block_info[0];
  entry_info.terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::ConditionalBranch;
  entry_info.terminator.true_block = 1;
  entry_info.terminator.false_block = 2;
  fixture.program.values->block_info[1].terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  fixture.program.values->block_info[2].terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;

  const auto undef = fixture.Emit(ValueOpcode::UndefU32);
  const auto write = fixture.Emit(
      ValueOpcode::WriteLane, {undef, Value(42u), Value(5u)});
  const auto read =
      fixture.Emit(ValueOpcode::ReadLane, {write, Value(5u)});
  entry_info.condition =
      fixture.Emit(ValueOpcode::IEqual32, {read, Value(42u)});
  fixture.Emit(ValueOpcode::Reference, {entry_info.condition});

  Check(EliminateReadLane(*fixture.program.values, 64).rewritten_reads == 1,
        "control-flow fixture did not eliminate its fixed-lane read");
  ConstantPropagationPass(fixture.program.values->blocks);
  ResolveControlFlowIdentities(*fixture.program.values);
  RemoveIdentities(fixture.program.values->blocks);
  EliminateDeadCode(fixture.program.values->blocks);

  std::string error;
  Check(entry_info.condition == Value(true) &&
            ValidateValueProgram(*fixture.program.values, true, &error),
        error.empty() ? "folded branch condition did not survive identity removal"
                      : error.c_str());
}

void TestUndefinedRuntimeValueFails() {
  Fixture fixture;
  const auto undef = fixture.Emit(ValueOpcode::UndefU32);
  fixture.Plan();
  fixture.program.values->descriptor_sources.push_back(
      {.dwords = {undef}, .dword_count = 1});
  DescriptorValue result;
  result.dword_count = 3;
  std::string error;
  Check(!EvaluateDescriptorSource(fixture.program, 0, 0x200, {}, result,
                                  &error) &&
            result.dword_count == 3 &&
            error.find("undefined typed runtime value") != std::string::npos,
        "undefined typed descriptor source did not fail transactionally");
}

} // namespace

namespace Common {

int DbgExitIfHandler(const char *, const char *, int) { return 1; }

void DbgExit(int) { std::abort(); }

} // namespace Common

int main() {
  try {
    TestImmediateFlatteningAndGvn();
    TestRawScalarComponentAlignment();
    TestScalarMemoryDomainMismatchFails();
    TestDynamicReadRemainsTyped();
    TestNestedSrtWalk();
    TestShaderBaseAndUserData();
    TestCarryAndBitFields();
    TestInvariantAndDivergentPhi();
    TestControlDependentStandaloneLoadStaysTyped();
    TestRuntime64BitDescriptorOps();
    TestConstantBufferBounds();
    TestReadLaneElimination();
    TestWave64MaterialBatchSplit();
    TestOptimizationPipeline();
    TestControlFlowValueSurvivesReadLaneFolding();
    TestUndefinedRuntimeValueFails();
    std::cout << "TypedValuePlanningTests: all cases passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "TypedValuePlanningTests: failed: " << e.what() << '\n';
    return 1;
  }
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "../src/graphics/shader/recompiler/ir/Block.cpp"
#include "../src/graphics/shader/recompiler/ir/Type.cpp"
#include "../src/graphics/shader/recompiler/ir/Value.cpp"
#include "../src/graphics/shader/recompiler/ir/ValueProgram.cpp"
#include "../src/graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
#include "../src/graphics/shader/recompiler/ir/passes/ConstantPropagation.cpp"
#include "../src/graphics/shader/recompiler/ir/passes/DeadCodeElimination.cpp"
#include "../src/graphics/shader/recompiler/ir/passes/Wave64MaterialBatch.cpp"
