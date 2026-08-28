#include "graphics/shader/recompiler/ir/passes/Wave64MaterialBatch.h"

#include <array>

#include <queue>

#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

Inst* Match(Value value, ValueOpcode opcode) {
	value      = value.Resolve();
	auto* inst = value.TryInstruction();
	return inst != nullptr && inst->GetOpcode() == opcode ? inst : nullptr;
}

bool Same(Value left, Value right) {
	return left.Resolve() == right.Resolve();
}

bool ImmediateU32(Value value, uint32_t expected) {
	value = value.Resolve();
	return value.IsImmediate() && value.GetType() == Type::U32 && value.U32() == expected;
}

bool MatchPerInvocationBit(const Inst& inst) {
	return inst.GetOpcode() == ValueOpcode::SelectU32 && ImmediateU32(inst.Arg(1), 1u) &&
	       ImmediateU32(inst.Arg(2), 0u);
}

bool ExactDppFlags(const DppMoveFlags& flags, uint16_t control) {
	return flags.control == control && flags.row_mask == 0xfu && flags.bank_mask == 0xfu &&
	       !flags.fetch_inactive && !flags.bound_control;
}

bool MatchDppOrStage(Value current, uint16_t control, Value active, Value& previous) {
	auto* update = Match(current, ValueOpcode::DppUpdateU32);
	if (update == nullptr || !ExactDppFlags(update->Flags<DppMoveFlags>(), control) ||
	    !Same(update->Arg(2), active)) {
		return false;
	}

	previous     = update->Arg(1).Resolve();
	auto* merged = Match(update->Arg(0), ValueOpcode::BitwiseOr32);
	if (merged == nullptr) {
		return false;
	}
	Value moved;
	if (Same(merged->Arg(0), previous)) {
		moved = merged->Arg(1);
	} else if (Same(merged->Arg(1), previous)) {
		moved = merged->Arg(0);
	} else {
		return false;
	}

	auto* move = Match(moved, ValueOpcode::DppMoveU32);
	return move != nullptr && ExactDppFlags(move->Flags<DppMoveFlags>(), control) &&
	       Same(move->Arg(0), previous) && Same(move->Arg(1), active);
}

bool MatchRowPairOrReduction(Value source) {
	auto* final_select = Match(source, ValueOpcode::SelectU32);
	if (final_select == nullptr) {
		return false;
	}
	const auto active      = final_select->Arg(0).Resolve();
	const auto combined    = final_select->Arg(1).Resolve();
	const auto row_reduced = final_select->Arg(2).Resolve();
	auto*      combined_or = Match(combined, ValueOpcode::BitwiseOr32);
	if (combined_or == nullptr) {
		return false;
	}

	Value selected_permute;
	if (Same(combined_or->Arg(0), row_reduced)) {
		selected_permute = combined_or->Arg(1);
	} else if (Same(combined_or->Arg(1), row_reduced)) {
		selected_permute = combined_or->Arg(0);
	} else {
		return false;
	}

	auto* permute_select = Match(selected_permute, ValueOpcode::SelectU32);
	if (permute_select == nullptr || !Same(permute_select->Arg(0), active)) {
		return false;
	}
	auto* permute = Match(permute_select->Arg(1), ValueOpcode::Permlane16U32);
	if (permute == nullptr) {
		return false;
	}
	const auto permute_flags = permute->Flags<PermlaneFlags>();
	if (!permute_flags.x16 || permute_flags.fetch_inactive || !permute_flags.bound_control ||
	    !Same(permute->Arg(0), row_reduced) || !ImmediateU32(permute->Arg(1), 0xffffffffu) ||
	    !ImmediateU32(permute->Arg(2), 0xffffffffu) || !Same(permute->Arg(3), active)) {
		return false;
	}

	Value current = row_reduced;
	for (const auto control: std::array<uint16_t, 4> {0x118u, 0x114u, 0x112u, 0x111u}) {
		Value previous;
		if (!MatchDppOrStage(current, control, active, previous)) {
			return false;
		}
		current = previous;
	}
	return true;
}

bool MatchReadLane(Value value, uint32_t lane, Value& source, Inst*& read) {
	read = Match(value, ValueOpcode::ReadLane);
	if (read == nullptr || !ImmediateU32(read->Arg(1), lane)) {
		return false;
	}
	source = read->Arg(0).Resolve();
	return true;
}

bool IsZeroComparison(const Inst& inst, Value value) {
	return inst.GetOpcode() == ValueOpcode::INotEqual32 &&
	       ((Same(inst.Arg(0), value) && ImmediateU32(inst.Arg(1), 0u)) ||
	        (Same(inst.Arg(1), value) && ImmediateU32(inst.Arg(0), 0u)));
}

bool MatchBinaryImmediate(const Inst& inst, Value value, uint32_t immediate) {
	return (Same(inst.Arg(0), value) && ImmediateU32(inst.Arg(1), immediate)) ||
	       (Same(inst.Arg(1), value) && ImmediateU32(inst.Arg(0), immediate));
}

bool ContainsPositiveAndTerm(Value value, Value term, uint32_t depth = 0) {
	value = value.Resolve();
	if (Same(value, term)) {
		return true;
	}
	if (depth >= 16u) {
		return false;
	}
	auto* inst = Match(value, ValueOpcode::LogicalAnd);
	return inst != nullptr && (ContainsPositiveAndTerm(inst->Arg(0), term, depth + 1u) ||
	                           ContainsPositiveAndTerm(inst->Arg(1), term, depth + 1u));
}

bool CollectNaturalLoop(Block* header, Block* backedge, std::unordered_set<Block*>& loop) {
	if (header == nullptr || backedge == nullptr) {
		return false;
	}
	bool closes_loop = false;
	for (auto* successor: backedge->ImmSuccessors()) {
		closes_loop |= successor == header;
	}
	if (!closes_loop) {
		return false;
	}

	std::queue<Block*> pending;
	loop.insert(header);
	loop.insert(backedge);
	pending.push(backedge);
	while (!pending.empty()) {
		auto* block = pending.front();
		pending.pop();
		for (auto* predecessor: block->ImmPredecessors()) {
			if (predecessor != header && loop.insert(predecessor).second) {
				pending.push(predecessor);
			}
		}
	}

	for (auto* block: loop) {
		if (block == header) {
			continue;
		}
		for (auto* predecessor: block->ImmPredecessors()) {
			if (!loop.contains(predecessor)) {
				return false;
			}
		}
	}
	return true;
}

bool DescriptorIndexFeedsSample(Inst& shift, Inst& sample) {
	std::queue<Inst*>         pending;
	std::unordered_set<Inst*> visited;
	std::unordered_set<Inst*> images;
	bool                      found_load = false;
	pending.push(&shift);
	visited.insert(&shift);
	while (!pending.empty()) {
		auto* current = pending.front();
		pending.pop();
		for (const auto& use: current->Uses()) {
			auto* user = use.user;
			if (user->GetOpcode() == ValueOpcode::IAdd32) {
				if (visited.insert(user).second) {
					pending.push(user);
				}
				continue;
			}
			if ((user->GetOpcode() != ValueOpcode::LoadAddressU8 &&
			     user->GetOpcode() != ValueOpcode::LoadAddressU16 &&
			     user->GetOpcode() != ValueOpcode::LoadAddressU32) ||
			    use.operand != 1u || user->Uses().empty()) {
				return false;
			}
			found_load = true;
			for (const auto& load_use: user->Uses()) {
				if (load_use.user->GetOpcode() != ValueOpcode::GetImageResource) {
					return false;
				}
				images.insert(load_use.user);
			}
		}
	}
	return found_load && images.contains(sample.Arg(0).ResolveInstruction());
}

bool SampleMergesUnderMaterialPredicate(Inst& sample, Value material_match) {
	if (sample.Uses().empty()) {
		return false;
	}
	for (const auto& sample_use: sample.Uses()) {
		auto* extract = sample_use.user;
		if (sample_use.operand != 0u ||
		    extract->GetOpcode() != ValueOpcode::CompositeExtractU32x4 ||
		    extract->UseCount() != 1u) {
			return false;
		}
		const auto& extract_use = extract->Uses().front();
		auto*       select      = extract_use.user;
		if (select->GetOpcode() != ValueOpcode::SelectU32 || extract_use.operand != 1u ||
		    !ContainsPositiveAndTerm(select->Arg(0), material_match)) {
			return false;
		}
	}
	return true;
}

bool IsSampleComponent(Value value, Inst& sample) {
	auto* extract = Match(value, ValueOpcode::CompositeExtractU32x4);
	return extract != nullptr && Same(extract->Arg(0), Value(&sample));
}

bool IsPerLaneSampleAccumulator(Value value, Inst& header_phi, Value material_match, Inst& sample,
                                std::unordered_set<Inst*>& visited) {
	value = value.Resolve();
	if (Same(value, Value(&header_phi))) {
		return true;
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (!visited.insert(inst).second) {
		return false;
	}
	if (inst->GetOpcode() == ValueOpcode::Phi) {
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			if (!IsPerLaneSampleAccumulator(inst->Arg(index), header_phi, material_match, sample,
			                                visited)) {
				return false;
			}
		}
		return true;
	}
	if (inst->GetOpcode() != ValueOpcode::SelectU32 ||
	    !ContainsPositiveAndTerm(inst->Arg(0), material_match) ||
	    !IsSampleComponent(inst->Arg(1), sample)) {
		return false;
	}
	return IsPerLaneSampleAccumulator(inst->Arg(2), header_phi, material_match, sample, visited);
}

bool HeaderPhisArePerLaneAccumulators(Block& header, Inst& mask_phi, size_t initial_index,
                                      Inst& material_match, Inst& sample,
                                      const std::unordered_set<Block*>& loop) {
	for (auto& inst: header) {
		if (inst.GetOpcode() != ValueOpcode::Phi || &inst == &mask_phi) {
			continue;
		}
		bool escapes_loop = false;
		for (const auto& use: inst.Uses()) {
			escapes_loop |= !loop.contains(use.user->Parent());
		}
		if (!escapes_loop) {
			continue;
		}
		if (inst.NumArgs() != 2u || inst.NumPhiBlocks() != 2u ||
		    inst.PhiBlock(initial_index) != mask_phi.PhiBlock(initial_index) ||
		    inst.PhiBlock(1u - initial_index) != mask_phi.PhiBlock(1u - initial_index)) {
			return false;
		}
		std::unordered_set<Inst*> visited;
		if (!IsPerLaneSampleAccumulator(inst.Arg(1u - initial_index), inst, Value(&material_match),
		                                sample, visited)) {
			return false;
		}
	}
	return true;
}

bool LoopValuesDoNotEscape(const std::unordered_set<Block*>& loop, Block& header,
                           const Inst& mask_phi) {
	for (auto* block: loop) {
		for (auto& inst: *block) {
			for (const auto& use: inst.Uses()) {
				if (loop.contains(use.user->Parent())) {
					continue;
				}
				if (inst.Parent() != &header || inst.GetOpcode() != ValueOpcode::Phi ||
				    &inst == &mask_phi) {
					return false;
				}
			}
		}
	}
	return true;
}

bool MatchReadOnlyMaterialLoop(Inst& aggregate) {
	if (aggregate.UseCount() != 1u) {
		return false;
	}
	auto* mask_phi = aggregate.Uses().front().user;
	if (mask_phi->GetOpcode() != ValueOpcode::Phi || mask_phi->NumArgs() != 2u ||
	    mask_phi->NumPhiBlocks() != 2u) {
		return false;
	}

	size_t initial_index = 2u;
	for (size_t index = 0; index < 2u; index++) {
		if (Same(mask_phi->Arg(index), Value(&aggregate))) {
			initial_index = index;
		}
	}
	if (initial_index == 2u || aggregate.Parent() != mask_phi->PhiBlock(initial_index)) {
		return false;
	}
	const auto update_index = 1u - initial_index;
	auto*      update       = Match(mask_phi->Arg(update_index), ValueOpcode::BitwiseXor32);
	if (update == nullptr) {
		return false;
	}
	Value one_hot;
	if (Same(update->Arg(0), Value(mask_phi))) {
		one_hot = update->Arg(1);
	} else if (Same(update->Arg(1), Value(mask_phi))) {
		one_hot = update->Arg(0);
	} else {
		return false;
	}

	Inst*    find_lsb = nullptr;
	Inst*    nonzero  = nullptr;
	uint32_t phi_uses = 0;
	for (const auto& use: mask_phi->Uses()) {
		phi_uses++;
		if (use.user == update) {
			continue;
		}
		if (use.user->GetOpcode() == ValueOpcode::FindILsb32 && find_lsb == nullptr) {
			find_lsb = use.user;
			continue;
		}
		if (IsZeroComparison(*use.user, Value(mask_phi)) && nonzero == nullptr) {
			nonzero = use.user;
			continue;
		}
		return false;
	}
	if (phi_uses != 3u || find_lsb == nullptr || nonzero == nullptr || find_lsb->UseCount() != 3u) {
		return false;
	}

	Inst* material_match = nullptr;
	Inst* descriptor_key = nullptr;
	Inst* bit_index      = nullptr;
	for (const auto& use: find_lsb->Uses()) {
		auto* user = use.user;
		switch (user->GetOpcode()) {
			case ValueOpcode::IEqual32:
				if (material_match != nullptr) {
					return false;
				}
				material_match = user;
				break;
			case ValueOpcode::ShiftLeftLogical32:
				if (descriptor_key != nullptr || !Same(user->Arg(0), Value(find_lsb)) ||
				    !ImmediateU32(user->Arg(1), 5u)) {
					return false;
				}
				descriptor_key = user;
				break;
			case ValueOpcode::BitwiseAnd32:
				if (bit_index != nullptr || !MatchBinaryImmediate(*user, Value(find_lsb), 31u)) {
					return false;
				}
				bit_index = user;
				break;
			default: return false;
		}
	}
	if (material_match == nullptr || material_match->UseCount() != 1u ||
	    descriptor_key == nullptr || bit_index == nullptr || bit_index->UseCount() != 1u) {
		return false;
	}
	auto* clear_bit = bit_index->Uses().front().user;
	if (clear_bit->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
	    !ImmediateU32(clear_bit->Arg(0), 1u) || !Same(clear_bit->Arg(1), Value(bit_index)) ||
	    !Same(one_hot, Value(clear_bit))) {
		return false;
	}

	std::unordered_set<Block*> loop;
	if (!CollectNaturalLoop(mask_phi->Parent(), mask_phi->PhiBlock(update_index), loop)) {
		return false;
	}
	Inst* sample = nullptr;
	for (auto* block: loop) {
		for (auto& inst: *block) {
			if (inst.MayHaveSideEffects() && inst.GetOpcode() != ValueOpcode::Reference &&
			    inst.GetOpcode() != ValueOpcode::ReferenceU32) {
				return false;
			}
			if (inst.GetOpcode() == ValueOpcode::ImageSampleRaw) {
				if (sample != nullptr) {
					return false;
				}
				sample = &inst;
			}
		}
	}
	return sample != nullptr && loop.contains(sample->Parent()) &&
	       DescriptorIndexFeedsSample(*descriptor_key, *sample) &&
	       SampleMergesUnderMaterialPredicate(*sample, Value(material_match)) &&
	       HeaderPhisArePerLaneAccumulators(*mask_phi->Parent(), *mask_phi, initial_index,
	                                        *material_match, *sample, loop) &&
	       LoopValuesDoNotEscape(loop, *mask_phi->Parent(), *mask_phi);
}

struct PerInvocationMaterialLoop {
	Inst* group_bits = nullptr;
};

bool FindMaterialReadLane(Inst& find_lsb, Inst*& lane_index, Inst*& read_lane) {
	std::queue<Inst*>         pending;
	std::unordered_set<Inst*> visited;
	pending.push(&find_lsb);
	visited.insert(&find_lsb);
	while (!pending.empty()) {
		auto* current = pending.front();
		pending.pop();
		for (const auto& use: current->Uses()) {
			auto* user = use.user;
			if (user->GetOpcode() == ValueOpcode::ReadLane && use.operand == 1u) {
				if (read_lane != nullptr) {
					return false;
				}
				lane_index = current;
				read_lane  = user;
				continue;
			}
			const bool select_index = user->GetOpcode() == ValueOpcode::SelectU32 &&
			                          (use.operand == 1u || use.operand == 2u);
			const bool masked_index = user->GetOpcode() == ValueOpcode::BitwiseAnd32 &&
			                          (MatchBinaryImmediate(*user, Value(current), 31u) ||
			                           MatchBinaryImmediate(*user, Value(current), 63u));
			if ((select_index || masked_index) && visited.insert(user).second) {
				pending.push(user);
			}
		}
	}
	return lane_index != nullptr && read_lane != nullptr;
}

bool MatchPerInvocationMaterialLoop(Inst& initial_bits, PerInvocationMaterialLoop& result) {
	if (!MatchPerInvocationBit(initial_bits) || initial_bits.UseCount() != 1u) {
		return false;
	}
	auto* mask_phi = initial_bits.Uses().front().user;
	if (mask_phi->GetOpcode() != ValueOpcode::Phi || mask_phi->NumArgs() != 2u ||
	    mask_phi->NumPhiBlocks() != 2u) {
		return false;
	}

	size_t initial_index = 2u;
	for (size_t index = 0; index < 2u; index++) {
		if (Same(mask_phi->Arg(index), Value(&initial_bits))) {
			initial_index = index;
		}
	}
	if (initial_index == 2u || initial_bits.Parent() != mask_phi->PhiBlock(initial_index)) {
		return false;
	}
	const auto update_index = 1u - initial_index;
	auto*      update       = Match(mask_phi->Arg(update_index), ValueOpcode::BitwiseAnd32);
	if (update == nullptr) {
		return false;
	}
	Value cleared;
	if (Same(update->Arg(0), Value(mask_phi))) {
		cleared = update->Arg(1);
	} else if (Same(update->Arg(1), Value(mask_phi))) {
		cleared = update->Arg(0);
	} else {
		return false;
	}
	auto* inverted_group = Match(cleared, ValueOpcode::BitwiseNot32);
	auto* group_bits = inverted_group != nullptr
	                       ? Match(inverted_group->Arg(0), ValueOpcode::SelectU32)
	                       : nullptr;
	if (group_bits == nullptr || !MatchPerInvocationBit(*group_bits) ||
	    group_bits->UseCount() != 1u) {
		return false;
	}

	Inst*    find_lsb = nullptr;
	Inst*    nonzero  = nullptr;
	uint32_t phi_uses = 0u;
	for (const auto& use: mask_phi->Uses()) {
		phi_uses++;
		if (use.user == update) {
			continue;
		}
		if (use.user->GetOpcode() == ValueOpcode::FindILsb32 && find_lsb == nullptr) {
			find_lsb = use.user;
			continue;
		}
		if (IsZeroComparison(*use.user, Value(mask_phi)) && nonzero == nullptr) {
			nonzero = use.user;
			continue;
		}
		return false;
	}
	if (phi_uses != 3u || find_lsb == nullptr || nonzero == nullptr) {
		return false;
	}

	Inst* lane_index = nullptr;
	Inst* read_lane  = nullptr;
	if (!FindMaterialReadLane(*find_lsb, lane_index, read_lane)) {
		return false;
	}

	Inst* material_match = nullptr;
	Inst* descriptor_key = nullptr;
	for (const auto& use: read_lane->Uses()) {
		auto* user = use.user;
		if (user->GetOpcode() == ValueOpcode::IEqual32) {
			const auto other = user->Arg(use.operand == 0u ? 1u : 0u).Resolve();
			if (material_match != nullptr || !Same(other, read_lane->Arg(0))) {
				return false;
			}
			material_match = user;
		} else if (user->GetOpcode() == ValueOpcode::ShiftLeftLogical32 &&
		           use.operand == 0u && ImmediateU32(user->Arg(1), 5u)) {
			if (descriptor_key != nullptr) {
				return false;
			}
			descriptor_key = user;
		} else {
			return false;
		}
	}
	if (material_match == nullptr || descriptor_key == nullptr ||
	    !ContainsPositiveAndTerm(group_bits->Arg(0), Value(material_match))) {
		return false;
	}

	std::unordered_set<Block*> loop;
	if (!CollectNaturalLoop(mask_phi->Parent(), mask_phi->PhiBlock(update_index), loop)) {
		return false;
	}
	Inst* sample = nullptr;
	for (auto* block: loop) {
		for (auto& inst: *block) {
			if (inst.MayHaveSideEffects() && inst.GetOpcode() != ValueOpcode::Reference &&
			    inst.GetOpcode() != ValueOpcode::ReferenceU32) {
				return false;
			}
			if (inst.GetOpcode() == ValueOpcode::ImageSampleRaw) {
				if (sample != nullptr) {
					return false;
				}
				sample = &inst;
			}
		}
	}
	if (sample == nullptr || !loop.contains(sample->Parent()) ||
	    !DescriptorIndexFeedsSample(*descriptor_key, *sample) ||
	    !SampleMergesUnderMaterialPredicate(*sample, Value(material_match)) ||
	    !HeaderPhisArePerLaneAccumulators(*mask_phi->Parent(), *mask_phi, initial_index,
	                                      *material_match, *sample, loop) ||
	    !LoopValuesDoNotEscape(loop, *mask_phi->Parent(), *mask_phi)) {
		return false;
	}

	result.group_bits = group_bits;
	return true;
}

Value InsertSubgroupBallotLow(Inst& invocation_bits) {
	auto* block = invocation_bits.Parent();
	auto  where = block->begin();
	while (where != block->end() && &*where != &invocation_bits) {
		++where;
	}
	if (where == block->end()) {
		return {};
	}
	++where;
	auto& ballot = *block->PrependNewInst(where, ValueOpcode::Ballot, {invocation_bits.Arg(0)});
	auto& low = *block->PrependNewInst(where, ValueOpcode::CompositeExtractU32x4,
	                                  {Value(&ballot), Value(0u)});
	return Value(&low);
}

} // namespace

Wave64MaterialBatchStats SplitWave64MaterialBatches(ValueProgram& program, uint32_t guest_wave_size,
                                                    uint32_t host_subgroup_size) {
	Wave64MaterialBatchStats stats;
	if (guest_wave_size != 64u || host_subgroup_size != 32u) {
		return stats;
	}

	std::vector<Inst*> candidates;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() == ValueOpcode::BitwiseOr32) {
				candidates.push_back(&inst);
			}
		}
	}
	for (auto* aggregate: candidates) {
		Value source31;
		Value source63;
		Inst* read31  = nullptr;
		Inst* read63  = nullptr;
		bool  matched = MatchReadLane(aggregate->Arg(0), 31u, source31, read31) &&
		                MatchReadLane(aggregate->Arg(1), 63u, source63, read63);
		if (!matched) {
			matched = MatchReadLane(aggregate->Arg(1), 31u, source31, read31) &&
			          MatchReadLane(aggregate->Arg(0), 63u, source63, read63);
		}
		if (!matched || !Same(source31, source63) || !MatchRowPairOrReduction(source31) ||
		    !MatchReadOnlyMaterialLoop(*aggregate)) {
			continue;
		}

		// Local lane 31 is guest lane 31 in the first subgroup and guest lane 63 in the second.
		// The strict proof above establishes that the full-wave mask is only a read-only batching
		// optimization, so each subgroup may consume its own 32-lane reduction.
		aggregate->ReplaceUsesWith(Value(read31));
		stats.rewritten_masks++;
	}

	// The per-invocation mask model intentionally represents a VALU comparison as 0/1 in each
	// invocation. That representation is normally sufficient, but material batching shaders consume
	// the numeric SGPR mask with FindLSB/ReadLane. Reconstruct a real host-subgroup ballot only for
	// this proven read-only idiom. Each subgroup32 then processes its own half of the guest Wave64.
	std::vector<Inst*> invocation_bits;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (MatchPerInvocationBit(inst)) {
				invocation_bits.push_back(&inst);
			}
		}
	}
	for (auto* invocation_bit: invocation_bits) {
		PerInvocationMaterialLoop match;
		if (!MatchPerInvocationMaterialLoop(*invocation_bit, match)) {
			continue;
		}
		const auto initial_ballot = InsertSubgroupBallotLow(*invocation_bit);
		const auto group_ballot   = InsertSubgroupBallotLow(*match.group_bits);
		if (initial_ballot.IsEmpty() || group_ballot.IsEmpty()) {
			continue;
		}
		invocation_bit->ReplaceUsesWith(initial_ballot);
		match.group_bits->ReplaceUsesWith(group_ballot);
		stats.rewritten_masks += 2u;
	}

	return stats;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
