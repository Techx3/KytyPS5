#include "common/file.h"
#include "loader/elf.h"
#include "loader/elfValidator.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace Loader;

template <typename T>
void Store(std::vector<uint8_t>& image, size_t offset, const T& value) {
	if (image.size() < offset + sizeof(T)) {
		image.resize(offset + sizeof(T));
	}
	std::memcpy(image.data() + offset, &value, sizeof(T));
}

Elf64_Ehdr Header(uint16_t program_count = 1) {
	Elf64_Ehdr header {};
	header.e_ident[EI_MAG0]       = 0x7f;
	header.e_ident[EI_MAG1]       = 'E';
	header.e_ident[EI_MAG2]       = 'L';
	header.e_ident[EI_MAG3]       = 'F';
	header.e_ident[EI_CLASS]      = ELFCLASS64;
	header.e_ident[EI_DATA]       = ELFDATA2LSB;
	header.e_ident[EI_VERSION]    = EV_CURRENT;
	header.e_ident[EI_OSABI]      = ELFOSABI_FREEBSD;
	header.e_ident[EI_ABIVERSION] = 2;
	header.e_type                 = ET_DYNEXEC;
	header.e_machine              = EM_X86_64;
	header.e_version              = EV_CURRENT;
	header.e_phoff                = sizeof(Elf64_Ehdr);
	header.e_ehsize               = sizeof(Elf64_Ehdr);
	header.e_phentsize            = sizeof(Elf64_Phdr);
	header.e_phnum                = program_count;
	return header;
}

bool Expect(bool condition, const char* message) {
	if (!condition) {
		std::cerr << message << '\n';
	}
	return condition;
}

bool TestRawElfAndSdk() {
	std::vector<uint8_t> image(0x180);
	const auto header = Header();
	Store(image, 0, header);
	Elf64_Phdr process {};
	process.p_type   = PT_OS_PROCPARAM;
	process.p_offset = 0x100;
	process.p_filesz = 0x18;
	process.p_memsz  = 0x18;
	Store(image, header.e_phoff, process);
	const uint32_t magic = 0x4942524fu;
	const uint32_t sdk   = 0x10000040u;
	Store(image, process.p_offset + 8u, magic);
	Store(image, process.p_offset + 0x14u, sdk);

	ElfValidationReport report;
	std::string error;
	return Expect(ValidateElfImage(image, &report, &error), error.c_str()) &&
	       Expect(!report.self && report.elf_offset == 0 && report.has_sdk_version &&
	                  report.sdk_version == sdk,
	              "raw ELF SDK metadata was not reported");
}

bool TestFileValidationClosesHandle() {
	std::vector<uint8_t> image(0x180);
	const auto           header = Header();
	Store(image, 0, header);
	Elf64_Phdr load {};
	load.p_type   = PT_LOAD;
	load.p_offset = 0x100;
	load.p_filesz = 0x20;
	Store(image, header.e_phoff, load);

	const auto path = std::filesystem::temp_directory_path() / "kyty_elf_validator_test.bin";
	Common::File output(path);
	if (!Expect(!output.IsInvalid(), "could not create temporary ELF file")) {
		return false;
	}
	output.Write(image.data(), static_cast<uint32_t>(image.size()));
	output.Close();

	ElfValidationReport report;
	std::string         error;
	const bool          valid = ValidateElfFile(path, &report, &error);
	Common::File::DeleteFile(path);
	return Expect(valid, error.c_str());
}

bool TestTruncatedProgramTable() {
	std::vector<uint8_t> image(sizeof(Elf64_Ehdr) + sizeof(Elf64_Phdr) - 1u);
	Store(image, 0, Header());
	std::string error;
	return Expect(!ValidateElfImage(image, nullptr, &error) &&
	                  error.find("program header table") != std::string::npos,
	              "truncated ELF program table was accepted");
}

bool TestSegmentOutsideFile() {
	std::vector<uint8_t> image(0x100);
	const auto header = Header();
	Store(image, 0, header);
	Elf64_Phdr load {};
	load.p_type   = PT_LOAD;
	load.p_offset = 0xf0;
	load.p_filesz = 0x20;
	Store(image, header.e_phoff, load);
	std::string error;
	return Expect(!ValidateElfImage(image, nullptr, &error) &&
	                  error.find("segment payload") != std::string::npos,
	              "out-of-range ELF segment was accepted");
}

bool TestSelfContainer() {
	std::vector<uint8_t> image(0x180);
	SelfHeader self {};
	self.ident[0]    = 0x4f;
	self.ident[1]    = 0x15;
	self.ident[2]    = 0x3d;
	self.ident[3]    = 0x1d;
	self.segments_num = 1;
	Store(image, 0, self);
	SelfSegment segment {};
	segment.offset            = 0x100;
	segment.compressed_size   = 0x20;
	segment.decompressed_size = 0x20;
	Store(image, sizeof(SelfHeader), segment);
	const uint64_t elf_offset = sizeof(SelfHeader) + sizeof(SelfSegment);
	const auto header = Header();
	Store(image, elf_offset, header);
	Elf64_Phdr load {};
	load.p_type   = PT_LOAD;
	load.p_offset = 0x2000;
	load.p_filesz = 0x20;
	Store(image, elf_offset + header.e_phoff, load);

	ElfValidationReport report;
	std::string error;
	return Expect(ValidateElfImage(image, &report, &error), error.c_str()) &&
	       Expect(report.self && report.elf_offset == elf_offset,
	              "SELF embedded ELF offset was not reported");
}

bool TestSelfSegmentOverflow() {
	std::vector<uint8_t> image(sizeof(SelfHeader) + sizeof(SelfSegment));
	SelfHeader self {};
	self.ident[0]     = 0x54;
	self.ident[1]     = 0x14;
	self.ident[2]     = 0xf5;
	self.ident[3]     = 0xee;
	self.segments_num = 1;
	Store(image, 0, self);
	SelfSegment segment {};
	segment.offset          = UINT64_MAX - 3u;
	segment.compressed_size = 8u;
	Store(image, sizeof(SelfHeader), segment);
	std::string error;
	return Expect(!ValidateElfImage(image, nullptr, &error) &&
	                  error.find("SELF segment payload") != std::string::npos,
	              "overflowing SELF segment was accepted");
}

} // namespace

int main() {
	if (!TestRawElfAndSdk() || !TestFileValidationClosesHandle() ||
	    !TestTruncatedProgramTable() || !TestSegmentOutsideFile() ||
	    !TestSelfContainer() || !TestSelfSegmentOverflow()) {
		return 1;
	}
	std::cout << "ELF validator tests passed\n";
	return 0;
}
