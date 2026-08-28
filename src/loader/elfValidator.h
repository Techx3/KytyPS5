#ifndef KYTY_LOADER_ELF_VALIDATOR_H_
#define KYTY_LOADER_ELF_VALIDATOR_H_

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

namespace Loader {

struct ElfValidationReport {
	bool     self            = false;
	uint64_t elf_offset      = 0;
	uint32_t sdk_version     = 0;
	bool     has_sdk_version = false;
};

[[nodiscard]] bool ValidateElfImage(std::span<const uint8_t> image,
	                                ElfValidationReport* report,
	                                std::string* error = nullptr);

[[nodiscard]] bool ValidateElfFile(const std::filesystem::path& file_name,
	                               ElfValidationReport* report,
	                               std::string* error = nullptr);

} // namespace Loader

#endif /* KYTY_LOADER_ELF_VALIDATOR_H_ */
