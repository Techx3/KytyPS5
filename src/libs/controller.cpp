#include "libs/controller.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/padData.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace Libs::Controller {

LIB_NAME("Pad", "Pad");

constexpr int PAD_ERROR_INVALID_ARG    = -2137915391; /* 0x80920001 */
constexpr int PAD_ERROR_INVALID_HANDLE = -2137915389; /* 0x80920003 */

struct PadControllerInformation {
	float    touch_pixel_density;
	uint16_t touch_resolution_x;
	uint16_t touch_resolution_y;
	uint8_t  stick_dead_zone_left;
	uint8_t  stick_dead_zone_right;
	uint8_t  connection_type;
	uint8_t  connected_count;
	bool     connected;
	int      device_class;
	uint8_t  reserve[8];
};

struct PadVibrationParam {
	uint8_t large_motor;
	uint8_t small_motor;
};

struct ControllerState {
	uint64_t time                                  = 0;
	uint32_t buttons                               = 0;
	int      axes[static_cast<int>(Axis::AxisMax)] = {128, 128, 128, 128, 0, 0};
};

class GameController {
public:
	GameController()          = default;
	virtual ~GameController() = default;

	KYTY_CLASS_NO_COPY(GameController);

	void     Connect(int id, const char* controller_guid);
	void     Disconnect(int id);
	void     Button(int id, uint32_t button, bool down);
	void     Axis(int id, Axis axis, int value);
	void     ResetInputState();
	int      Open(int user_id, int type, int index, int player_index);
	int      GetHandle(int user_id, int type, int index);
	bool     Close(int handle);
	int      GetPlayerIndex(int handle);
	uint32_t GetConnectedPlayerMask();
	void     GetConnectionInfo(int player_index, bool* flag, int* count);
	void     ReadState(int player_index, ControllerState* state, bool* flag, int* count);
	int      ReadStates(int player_index, ControllerState* states, int states_num, bool* flag,
	                    int* count);

private:
	static constexpr uint32_t STATES_MAX = 64;

	struct StatePrivate {
		bool obtained = false;
	};

	struct PlayerPad {
		int             controller_id   = -1;
		bool            keyboard        = false;
		bool            connected       = false;
		int             connected_count = 0;
		ControllerState states[STATES_MAX];
		StatePrivate    state_private[STATES_MAX];
		ControllerState last_state;
		uint32_t        states_num  = 0;
		uint32_t        first_state = 0;
	};

	struct OpenPad {
		bool opened       = false;
		int  user_id      = -1;
		int  type         = -1;
		int  index        = -1;
		int  player_index = -1;
	};

	static constexpr size_t OPEN_PADS_MAX = Config::MAX_LOCAL_USERS * 2 + 1;

	static void               ResetState(PlayerPad& player);
	static ControllerState    GetLastState(const PlayerPad& player);
	static void               AddState(PlayerPad& player, const ControllerState& state);
	[[nodiscard]] int         FindController(int id) const;
	[[nodiscard]] int         FindAvailablePlayer(const char* controller_guid) const;
	[[nodiscard]] int         FindOpenPad(int user_id, int type, int index) const;
	[[nodiscard]] static bool PlayerIndexIsValid(int player_index);

	Common::Mutex                                  m_mutex;
	std::array<PlayerPad, Config::MAX_LOCAL_USERS> m_players;
	std::array<OpenPad, OPEN_PADS_MAX>             m_open_pads;
};

static GameController* g_controller = nullptr;

static uint8_t pad_connected_count_to_u8(int connected_count) {
	return static_cast<uint8_t>(connected_count > 255 ? 255 : connected_count);
}

static void pad_fill_data(PadData* data, const ControllerState& state, bool connected,
                          int connected_count) {
	EXIT_IF(data == nullptr);

	std::memset(data, 0, sizeof(*data));

	data->buttons                = state.buttons;
	data->left_stick_x           = state.axes[static_cast<int>(Axis::LeftX)];
	data->left_stick_y           = state.axes[static_cast<int>(Axis::LeftY)];
	data->right_stick_x          = state.axes[static_cast<int>(Axis::RightX)];
	data->right_stick_y          = state.axes[static_cast<int>(Axis::RightY)];
	data->analog_buttons_l2      = state.axes[static_cast<int>(Axis::TriggerLeft)];
	data->analog_buttons_r2      = state.axes[static_cast<int>(Axis::TriggerRight)];
	data->orientation_w          = 1.0f;
	data->touch_data_touch0_id   = 1;
	data->touch_data_touch1_id   = 2;
	data->connected              = connected;
	data->timestamp              = state.time;
	data->connected_count        = pad_connected_count_to_u8(connected_count);
	data->device_unique_data_len = 0;
}

void Initialize() {
	EXIT_IF(g_controller != nullptr);

	g_controller = new GameController;
	g_controller->Connect(HOST_INPUT_CONTROLLER_ID, nullptr);
}

void Shutdown() {
	delete g_controller;
	g_controller = nullptr;
}

void GameController::Connect(int id, const char* controller_guid) {
	Common::LockGuard lock(m_mutex);

	if (id == HOST_INPUT_CONTROLLER_ID) {
		auto& player = m_players[0];
		if (player.keyboard) {
			return;
		}
		player.keyboard = true;
		if (!player.connected) {
			player.connected = true;
			player.connected_count++;
		}
		ResetState(player);
		return;
	}

	if (FindController(id) >= 0) {
		return;
	}

	const int player_index = FindAvailablePlayer(controller_guid);
	if (player_index >= 0) {
		auto&      player        = m_players[static_cast<size_t>(player_index)];
		const bool was_connected = player.connected;
		player.controller_id     = id;
		player.connected         = true;
		if (!was_connected) {
			player.connected_count++;
		}
		ResetState(player);
		LOGF("Controller %d (GUID %s) assigned to local player %d\n", id,
		     controller_guid != nullptr && controller_guid[0] != '\0' ? controller_guid : "unknown",
		     player_index + 1);
		return;
	}

	LOGF("Ignoring controller %d (GUID %s): no matching local player slot is available\n", id,
	     controller_guid != nullptr && controller_guid[0] != '\0' ? controller_guid : "unknown");
}

void GameController::Disconnect(int id) {
	Common::LockGuard lock(m_mutex);

	if (id == HOST_INPUT_CONTROLLER_ID) {
		auto& player     = m_players[0];
		player.keyboard  = false;
		player.connected = player.controller_id >= 0;
		ResetState(player);
		return;
	}

	const int player_index = FindController(id);
	if (player_index < 0) {
		return;
	}

	auto& player         = m_players[static_cast<size_t>(player_index)];
	player.controller_id = -1;
	player.connected     = player_index == 0 && player.keyboard;
	ResetState(player);
}

void GameController::ResetState(PlayerPad& player) {
	player.states_num  = 0;
	player.first_state = 0;
	player.last_state  = ControllerState();
}

ControllerState GameController::GetLastState(const PlayerPad& player) {
	if (player.states_num == 0) {
		return player.last_state;
	}

	auto last = (player.first_state + player.states_num - 1) % STATES_MAX;

	return player.states[last];
}

void GameController::AddState(PlayerPad& player, const ControllerState& state) {
	if (player.states_num >= STATES_MAX) {
		player.states_num  = STATES_MAX - 1;
		player.first_state = (player.first_state + 1) % STATES_MAX;
	}

	auto index = (player.first_state + player.states_num) % STATES_MAX;

	player.states[index] = state;
	player.last_state    = state;

	player.state_private[index].obtained = false;

	player.states_num++;
}

int GameController::FindController(int id) const {
	for (uint32_t i = 0; i < Config::MAX_LOCAL_USERS; i++) {
		if (m_players[i].controller_id == id) {
			return static_cast<int>(i);
		}
	}
	return -1;
}

int GameController::FindAvailablePlayer(const char* controller_guid) const {
	const std::string_view guid             = controller_guid != nullptr ? controller_guid : "";
	const uint32_t         player_count     = Config::GetLocalPlayerCount();
	bool                   guid_is_reserved = false;

	if (!guid.empty()) {
		for (uint32_t i = 0; i < player_count; i++) {
			const auto& configured_guid = Config::GetControllerGuid(i);
			if (!configured_guid.empty() && Common::EqualNoCase(configured_guid, guid)) {
				guid_is_reserved = true;
				if (m_players[i].controller_id < 0) {
					return static_cast<int>(i);
				}
			}
		}
	}

	if (guid_is_reserved) {
		return -1;
	}

	for (uint32_t i = 0; i < player_count; i++) {
		if (m_players[i].controller_id < 0 && Config::GetControllerGuid(i).empty()) {
			return static_cast<int>(i);
		}
	}

	return -1;
}

void GameController::Button(int id, uint32_t button, bool down) {
	Common::LockGuard lock(m_mutex);

	const int player_index = id == HOST_INPUT_CONTROLLER_ID ? 0 : FindController(id);
	if (player_index >= 0) {
		auto& player = m_players[static_cast<size_t>(player_index)];
		auto  state  = GetLastState(player);

		state.time = LibKernel::KernelGetProcessTime();

		if (down) {
			state.buttons |= button;
		} else {
			state.buttons &= ~button;
		}

		AddState(player, state);
	}
}

void GameController::Axis(int id, Controller::Axis axis, int value) {
	Common::LockGuard lock(m_mutex);

	const int player_index = id == HOST_INPUT_CONTROLLER_ID ? 0 : FindController(id);
	if (player_index >= 0) {
		auto& player = m_players[static_cast<size_t>(player_index)];
		auto  state  = GetLastState(player);

		state.time = LibKernel::KernelGetProcessTime();

		int axis_id = static_cast<int>(axis);

		EXIT_IF(axis_id < 0 || axis_id >= static_cast<int>(Controller::Axis::AxisMax));

		state.axes[axis_id] = value;

		if (axis == Controller::Axis::TriggerLeft) {
			if (value > 0) {
				state.buttons |= PAD_BUTTON_L2;
			} else {
				state.buttons &= ~PAD_BUTTON_L2;
			}
		}

		if (axis == Controller::Axis::TriggerRight) {
			if (value > 0) {
				state.buttons |= PAD_BUTTON_R2;
			} else {
				state.buttons &= ~PAD_BUTTON_R2;
			}
		}

		AddState(player, state);
	}
}

void GameController::ResetInputState() {
	Common::LockGuard lock(m_mutex);
	for (auto& player: m_players) {
		ControllerState state {};
		state.time = LibKernel::KernelGetProcessTime();
		ResetState(player);
		AddState(player, state);
	}
}

bool GameController::PlayerIndexIsValid(int player_index) {
	return player_index >= 0 && player_index < static_cast<int>(Config::MAX_LOCAL_USERS);
}

int GameController::FindOpenPad(int user_id, int type, int index) const {
	for (uint32_t i = 0; i < m_open_pads.size(); i++) {
		const auto& pad = m_open_pads[i];
		if (pad.opened && pad.user_id == user_id && pad.type == type && pad.index == index) {
			return static_cast<int>(i);
		}
	}
	return -1;
}

int GameController::Open(int user_id, int type, int index, int player_index) {
	Common::LockGuard lock(m_mutex);
	EXIT_IF(!PlayerIndexIsValid(player_index));

	if (FindOpenPad(user_id, type, index) >= 0) {
		return -2137915388; /* SCE_PAD_ERROR_ALREADY_OPENED */
	}

	for (uint32_t i = 0; i < m_open_pads.size(); i++) {
		auto& pad = m_open_pads[i];
		if (!pad.opened) {
			pad.opened       = true;
			pad.user_id      = user_id;
			pad.type         = type;
			pad.index        = index;
			pad.player_index = player_index;
			return static_cast<int>(i + 1);
		}
	}

	return -2137915384; /* SCE_PAD_ERROR_NO_HANDLE */
}

int GameController::GetHandle(int user_id, int type, int index) {
	Common::LockGuard lock(m_mutex);

	const int pad_index = FindOpenPad(user_id, type, index);
	return pad_index >= 0 ? pad_index + 1 : -2137915384; /* SCE_PAD_ERROR_NO_HANDLE */
}

bool GameController::Close(int handle) {
	Common::LockGuard lock(m_mutex);
	const int         pad_index = handle - 1;
	if (pad_index < 0 || pad_index >= static_cast<int>(m_open_pads.size()) ||
	    !m_open_pads[static_cast<size_t>(pad_index)].opened) {
		return false;
	}

	m_open_pads[static_cast<size_t>(pad_index)] = {};
	return true;
}

int GameController::GetPlayerIndex(int handle) {
	Common::LockGuard lock(m_mutex);
	const int         pad_index = handle - 1;
	if (pad_index < 0 || pad_index >= static_cast<int>(m_open_pads.size()) ||
	    !m_open_pads[static_cast<size_t>(pad_index)].opened) {
		return -1;
	}

	return m_open_pads[static_cast<size_t>(pad_index)].player_index;
}

uint32_t GameController::GetConnectedPlayerMask() {
	Common::LockGuard lock(m_mutex);
	uint32_t          mask = 0;
	for (uint32_t i = 0; i < Config::GetLocalPlayerCount(); i++) {
		if (m_players[i].connected) {
			mask |= 1u << i;
		}
	}
	return mask;
}

void GameController::GetConnectionInfo(int player_index, bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);
	EXIT_IF(!PlayerIndexIsValid(player_index));

	Common::LockGuard lock(m_mutex);
	const auto&       player = m_players[static_cast<size_t>(player_index)];

	*flag  = player.connected;
	*count = player.connected_count;
}

void GameController::ReadState(int player_index, ControllerState* state, bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);
	EXIT_IF(state == nullptr);
	EXIT_IF(!PlayerIndexIsValid(player_index));

	Common::LockGuard lock(m_mutex);
	const auto&       player = m_players[static_cast<size_t>(player_index)];

	*flag  = player.connected;
	*count = player.connected_count;
	*state = GetLastState(player);
}

int GameController::ReadStates(int player_index, ControllerState* states, int states_num,
                               bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);
	EXIT_IF(states == nullptr);
	EXIT_IF(states_num < 1 || states_num > STATES_MAX);
	EXIT_IF(!PlayerIndexIsValid(player_index));

	Common::LockGuard lock(m_mutex);
	auto&             player = m_players[static_cast<size_t>(player_index)];

	*flag  = player.connected;
	*count = player.connected_count;

	int ret_num = 0;

	if (player.connected) {
		if (player.states_num != 0) {
			for (uint32_t i = 0; i < player.states_num; i++) {
				if (ret_num >= states_num) {
					break;
				}
				auto index = (player.first_state + i) % STATES_MAX;
				if (!player.state_private[index].obtained) {
					player.state_private[index].obtained = true;

					states[ret_num++] = player.states[index];
				}
			}
		}
	}

	return ret_num;
}

void ControllerConnect(int id, const char* controller_guid) {
	EXIT_IF(g_controller == nullptr);

	g_controller->Connect(id, controller_guid);
}

void ControllerDisconnect(int id) {
	EXIT_IF(g_controller == nullptr);

	g_controller->Disconnect(id);
}

void ControllerButton(int id, uint32_t button, bool down) {
	EXIT_IF(g_controller == nullptr);

	g_controller->Button(id, button, down);
}

void ControllerAxis(int id, Axis axis, int value) {
	EXIT_IF(g_controller == nullptr);

	g_controller->Axis(id, axis, value);
}

void ControllerResetInputState() {
	EXIT_IF(g_controller == nullptr);
	g_controller->ResetInputState();
}

uint32_t ControllerGetConnectedPlayerMask() {
	return g_controller != nullptr ? g_controller->GetConnectedPlayerMask() : 0;
}

int KYTY_SYSV_ABI PadInit() {
	PRINT_NAME();

	return OK;
}

static int PadOpenPlayerIndex(int user_id, int type, int index) {
	constexpr int user_id_system     = 0xff;
	constexpr int port_type_standard = 0;
	constexpr int port_type_special  = 2;
	constexpr int port_type_remote   = 16;

	if (index != 0) {
		return -1;
	}
	if (user_id == user_id_system && type == port_type_remote) {
		return 0;
	}
	if (type == port_type_standard || type == port_type_special) {
		return Config::GetLocalUserIndex(user_id);
	}

	return -1;
}

int KYTY_SYSV_ABI PadOpen(int user_id, int type, int index, const void* param) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n"
	     "\t param   = 0x%016" PRIx64 "\n",
	     user_id, type, index, reinterpret_cast<uint64_t>(param));

	constexpr int pad_error_invalid_arg = -2137915391; /* 0x80920001 */

	const int player_index = PadOpenPlayerIndex(user_id, type, index);
	if (player_index < 0) {
		return pad_error_invalid_arg;
	}

	EXIT_IF(g_controller == nullptr);

	return g_controller->Open(user_id, type, index, player_index);
}

int KYTY_SYSV_ABI PadGetHandle(int user_id, int type, int index) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n",
	     user_id, type, index);

	constexpr int pad_error_device_no_handle = -2137915384; /* 0x80920008 */

	const int player_index = PadOpenPlayerIndex(user_id, type, index);
	if (player_index < 0) {
		return pad_error_device_no_handle;
	}

	EXIT_IF(g_controller == nullptr);

	return g_controller->GetHandle(user_id, type, index);
}

bool PadHandleIsValid(int handle) {
	return g_controller != nullptr && g_controller->GetPlayerIndex(handle) >= 0;
}

int KYTY_SYSV_ABI PadClose(int handle) {
	PRINT_NAME();

	LOGF("\t handle = %d\n", handle);

	if (g_controller == nullptr || !g_controller->Close(handle)) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	return OK;
}

int KYTY_SYSV_ABI PadSetMotionSensorState(int handle, bool enable) {
	PRINT_NAME();

	if (!PadHandleIsValid(handle)) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	LOGF("\t enable = %s\n", (enable ? "true" : "false"));

	return OK;
}

int KYTY_SYSV_ABI PadSetAngularVelocityDeadbandState(int handle, bool enable) {
	PRINT_NAME();

	if (!PadHandleIsValid(handle)) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	LOGF("\t enable = %s\n", (enable ? "true" : "false"));

	return OK;
}

int KYTY_SYSV_ABI PadResetOrientation(int handle) {
	PRINT_NAME();

	if (!PadHandleIsValid(handle)) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	return OK;
}

int KYTY_SYSV_ABI PadGetControllerInformation(int handle, PadControllerInformation* info) {
	PRINT_NAME();

	EXIT_IF(g_controller == nullptr);

	const int player_index = g_controller->GetPlayerIndex(handle);
	if (player_index < 0) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (info == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	int  connected_count = 0;
	bool connected       = false;

	g_controller->GetConnectionInfo(player_index, &connected, &connected_count);

	std::memset(info, 0, sizeof(*info));

	info->touch_pixel_density   = 44.86f;
	info->touch_resolution_x    = 1920;
	info->touch_resolution_y    = 943;
	info->stick_dead_zone_left  = controller_get_axis(-32768, 32767, 8000) - 128;
	info->stick_dead_zone_right = controller_get_axis(-32768, 32767, 8000) - 128;
	info->connection_type       = 0;
	info->connected_count       = pad_connected_count_to_u8(connected_count);
	info->connected             = connected;
	info->device_class          = 0;

	return OK;
}

int KYTY_SYSV_ABI PadReadState(int handle, PadData* data) {
	PRINT_NAME();

	EXIT_IF(g_controller == nullptr);
	const int player_index = g_controller->GetPlayerIndex(handle);
	if (player_index < 0) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (data == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	int             connected_count = 0;
	bool            connected       = false;
	ControllerState state;

	g_controller->ReadState(player_index, &state, &connected, &connected_count);

	pad_fill_data(data, state, connected, connected_count);

	return OK;
}

int KYTY_SYSV_ABI PadRead(int handle, PadData* data, int num) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(num < 1 || num > 64);
	EXIT_IF(g_controller == nullptr);
	const int player_index = g_controller->GetPlayerIndex(handle);
	if (player_index < 0) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (data == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	std::memset(data, 0, sizeof(PadData) * static_cast<size_t>(num));

	int             connected_count = 0;
	bool            connected       = false;
	ControllerState states[64]      = {};

	int ret_num = g_controller->ReadStates(player_index, states, num, &connected, &connected_count);

	if (!connected || ret_num == 0) {
		if (connected) {
			g_controller->ReadState(player_index, &states[0], &connected, &connected_count);
		}
		ret_num = 1;
	}

	for (int i = 0; i < ret_num; i++) {
		pad_fill_data(&data[i], states[i], connected, connected_count);
	}

	return ret_num;
}

int KYTY_SYSV_ABI PadSetVibration(int handle, const PadVibrationParam* param) {
	PRINT_NAME();

	if (!PadHandleIsValid(handle)) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	LOGF("\t large_motor = %d\n"
	     "\t small_motor = %d\n",
	     static_cast<int>(param->large_motor), static_cast<int>(param->small_motor));

	return OK;
}

int KYTY_SYSV_ABI PadResetLightBar(int handle) {
	PRINT_NAME();

	if (!PadHandleIsValid(handle)) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	return OK;
}

int KYTY_SYSV_ABI PadSetLightBar(int handle, const PadLightBarParam* param) {
	PRINT_NAME();

	if (!PadHandleIsValid(handle)) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	return OK;
}

} // namespace Libs::Controller
