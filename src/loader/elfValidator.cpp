#include "loader/elfValidator.h"

#include "common/file.h"
#include "loader/elf.h"

#include <cstring>
#include <limits>

namespace Loader {
namespace {

constexpr Elf64_Word SHT_NOBITS          = 8;
constexpr uint32_t   PROCESS_PARAM_MAGIC = 0x4942524fu;

bool Fail(std::string* error, const char* message) {
	if (error != nullptr) {
		*error = message;
	}
	return false;
}

bool RangeFits(uint64_t offset, uint64_t count, uint64_t element_size, uint64_t total) {
	return offset <= total &&
	       (element_size == 0 || count <= (total - offset) / element_size);
}

template <typename T>
bool ReadStruct(std::span<const uint8_t> image, uint64_t offset, T* value) {
	if (value == nullptr || !RangeFits(offset, 1, sizeof(T), image.size())) {
		return false;
	}
	std::memcpy(value, image.data() + offset, sizeof(T));
	return true;
}

bool IsSelfMagic(std::span<const uint8_t> image) {
	if (image.size() < 4) {
		return false;
	}
	return (image[0] == 0x4f && image[1] == 0x15 && image[2] == 0x3d &&
	        image[3] == 0x1d) ||
	       (image[0] == 0x54 && image[1] == 0x14 && image[2] == 0xf5 &&
	        image[3] == 0xee);
}

bool ValidateHeader(const Elf64_Ehdr& header, std::string* error) {
	if (header.e_ident[EI_MAG0] != 0x7f || header.e_ident[EI_MAG1] != 'E' ||
	    header.e_ident[EI_MAG2] != 'L' || header.e_ident[EI_MAG3] != 'F') {
		return Fail(error, "missing ELF magic");
	}
	if (header.e_ident[EI_CLASS] != ELFCLASS64 ||
	    header.e_ident[EI_DATA] != ELFDATA2LSB ||
	    header.e_ident[EI_VERSION] != EV_CURRENT) {
		return Fail(error, "unsupported ELF class, byte order, or identification version");
	}
	if (header.e_ident[EI_OSABI] != ELFOSABI_FREEBSD ||
	    (header.e_ident[EI_ABIVERSION] != 0 && header.e_ident[EI_ABIVERSION] != 2)) {
		return Fail(error, "unsupported ELF ABI");
	}
	if ((header.e_type != ET_DYNEXEC && header.e_type != ET_DYNAMIC) ||
	    header.e_machine != EM_X86_64 || header.e_version != EV_CURRENT) {
		return Fail(error, "unsupported ELF type, machine, or version");
	}
	if (header.e_ehsize != sizeof(Elf64_Ehdr) ||
	    header.e_phentsize != sizeof(Elf64_Phdr) ||
	    (header.e_shnum != 0 && header.e_shentsize != sizeof(Elf64_Shdr))) {
		return Fail(error, "invalid ELF header entry sizes");
	}
	return true;
}

} // namespace

bool ValidateElfImage(std::span<const uint8_t> image, ElfValidationReport* report,
	                  std::string* error) {
	ElfValidationReport local;
	if (image.empty()) {
		return Fail(error, "empty ELF/SELF image");
	}

	SelfHeader self_header {};
	if (IsSelfMagic(image)) {
		local.self = true;
		if (!ReadStruct(image, 0, &self_header) ||
		    !RangeFits(sizeof(SelfHeader), self_header.segments_num, sizeof(SelfSegment),
		               image.size())) {
			return Fail(error, "truncated SELF segment table");
		}
		local.elf_offset = sizeof(SelfHeader) +
		                   static_cast<uint64_t>(self_header.segments_num) * sizeof(SelfSegment);
		for (uint16_t index = 0; index < self_header.segments_num; index++) {
			SelfSegment segment {};
			const auto offset = sizeof(SelfHeader) +
			                    static_cast<uint64_t>(index) * sizeof(SelfSegment);
			if (!ReadStruct(image, offset, &segment) ||
			    !RangeFits(segment.offset, segment.compressed_size, 1, image.size())) {
				return Fail(error, "SELF segment payload is outside the file");
			}
		}
	}

	Elf64_Ehdr header {};
	if (!ReadStruct(image, local.elf_offset, &header)) {
		return Fail(error, "truncated ELF header");
	}
	if (!ValidateHeader(header, error)) {
		return false;
	}
	if (!RangeFits(local.elf_offset + header.e_phoff, header.e_phnum,
	               sizeof(Elf64_Phdr), image.size())) {
		return Fail(error, "ELF program header table is outside the file");
	}
	if (!local.self && header.e_shnum != 0 &&
	    !RangeFits(local.elf_offset + header.e_shoff, header.e_shnum,
	               sizeof(Elf64_Shdr), image.size())) {
		return Fail(error, "ELF section header table is outside the file");
	}

	uint32_t process_param_count = 0;
	for (Elf64_Half index = 0; index < header.e_phnum; index++) {
		Elf64_Phdr program {};
		const auto offset = local.elf_offset + header.e_phoff +
		                    static_cast<uint64_t>(index) * sizeof(Elf64_Phdr);
		if (!ReadStruct(image, offset, &program)) {
			return Fail(error, "truncated ELF program header");
		}
		if (!local.self && !RangeFits(program.p_offset, program.p_filesz, 1, image.size())) {
			return Fail(error, "ELF segment payload is outside the file");
		}
		if (program.p_type == PT_OS_PROCPARAM) {
			process_param_count++;
			if (process_param_count != 1) {
				return Fail(error, "ELF contains multiple process-parameter segments");
			}
			if (program.p_filesz < 0x18u) {
				return Fail(error, "ELF process-parameter segment is truncated");
			}
			if (!local.self && RangeFits(program.p_offset, 0x18u, 1, image.size())) {
				uint32_t magic = 0;
				std::memcpy(&magic, image.data() + program.p_offset + 8u, sizeof(magic));
				if (magic == PROCESS_PARAM_MAGIC) {
					std::memcpy(&local.sdk_version, image.data() + program.p_offset + 0x14u,
					            sizeof(local.sdk_version));
					local.has_sdk_version = true;
				}
			}
		}
	}

	if (local.self) {
		for (uint16_t index = 0; index < self_header.segments_num; index++) {
			SelfSegment segment {};
			ReadStruct(image,
			           sizeof(SelfHeader) + static_cast<uint64_t>(index) * sizeof(SelfSegment),
			           &segment);
			if ((segment.type & 0x800u) != 0u &&
			    ((segment.type >> 20u) & 0xfffu) >= header.e_phnum) {
				return Fail(error, "SELF segment references an invalid ELF program header");
			}
		}
	} else if (header.e_shnum != 0) {
		for (Elf64_Half index = 0; index < header.e_shnum; index++) {
			Elf64_Shdr section {};
			const auto offset = header.e_shoff +
			                    static_cast<uint64_t>(index) * sizeof(Elf64_Shdr);
			if (!ReadStruct(image, offset, &section)) {
				return Fail(error, "truncated ELF section header");
			}
			if (section.sh_type != SHT_NOBITS &&
			    !RangeFits(section.sh_offset, section.sh_size, 1, image.size())) {
				return Fail(error, "ELF section payload is outside the file");
			}
		}
	}

	if (report != nullptr) {
		*report = local;
	}
	if (error != nullptr) {
		error->clear();
	}
	return true;
}

bool ValidateElfFile(const std::filesystem::path& file_name, ElfValidationReport* report,
	                 std::string* error) {
	Common::File file(file_name, Common::File::Mode::Read);
	if (file.IsInvalid()) {
		return Fail(error, "cannot open ELF/SELF image");
	}
	if (file.Size() > std::numeric_limits<uint32_t>::max()) {
		file.Close();
		return Fail(error, "ELF/SELF image exceeds the supported validator size");
	}
	auto image = file.ReadWholeBuffer();
	file.Close();
	return ValidateElfImage(
	    {reinterpret_cast<const uint8_t*>(image.GetDataConst()), image.Size()}, report, error);
}

} // namespace Loader
