#include "libs/audio.h"
#include "libs/audio_internal.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace {

namespace AudioOut2 = Libs::Audio::AudioOut2;

std::mutex g_device_mutex;
std::condition_variable g_device_cv;
std::vector<int> g_live_devices;
std::vector<int> g_device_backed_handles;
std::vector<bool> g_output_blocking;
std::vector<float> g_captured_float_output;
uint32_t g_captured_output_count = 0;
bool g_capture_float_output = false;
int g_next_device = 1;
int g_open_waiters = 0;
bool g_block_opens = false;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "AudioOut2PortTests: failed: %s\n", text);
    std::abort();
  }
}

struct PortParam {
  uint16_t port_type;
  uint16_t pad;
  uint32_t data_format;
  uint32_t sampling_freq;
  uint32_t flags;
  uint64_t user_handle;
  uint32_t reserved[10];
};

struct ContextParam {
  uint32_t max_ports;
  uint32_t max_object_ports;
  uint32_t guarantee_object_ports;
  uint32_t queue_depth;
  uint32_t num_grains;
  uint32_t flags;
  uint32_t reserved[10];
};

struct PortState {
  uint16_t output;
  uint8_t num_channels;
  uint8_t pad1;
  int16_t volume;
  uint16_t reroute_counter;
  uint32_t flags;
  uint32_t pad2;
  uint64_t reserved[6];
};

struct Attribute {
  uint32_t attribute_id;
  int32_t reserved;
  const void *value;
  size_t value_size;
};

struct Pcm {
  const void *data;
};

const auto *AsParam(const PortParam *param) {
  return reinterpret_cast<const AudioOut2::AudioOut2PortParam *>(param);
}

const auto *AsParam(const ContextParam *param) {
  return reinterpret_cast<const AudioOut2::AudioOut2ContextParam *>(param);
}

auto *AsState(PortState *state) {
  return reinterpret_cast<AudioOut2::AudioOut2PortState *>(state);
}

const auto *AsAttribute(const Attribute *attribute) {
  return reinterpret_cast<const AudioOut2::AudioOut2Attribute *>(attribute);
}

PortParam MakeParam(uint32_t data_format = 0x200) {
  PortParam param{};
  param.data_format = data_format;
  param.sampling_freq = 48000;
  return param;
}

AudioOut2::AudioOut2ContextHandle CreateContext(uint32_t queue_depth = 4) {
  ContextParam param{};
  param.queue_depth = queue_depth;
  param.num_grains = 512;
  AudioOut2::AudioOut2ContextHandle context = 0;
  Check(AudioOut2::AudioOut2ContextCreate(AsParam(&param), nullptr, 0,
                                          &context) == OK,
        "context create failed");
  return context;
}

void BlockDeviceOpens() {
  std::lock_guard lock(g_device_mutex);
  g_open_waiters = 0;
  g_block_opens = true;
}

void WaitForDeviceOpens(int count) {
  std::unique_lock lock(g_device_mutex);
  g_device_cv.wait(lock, [count]() { return g_open_waiters >= count; });
}

void ReleaseDeviceOpens() {
  std::lock_guard lock(g_device_mutex);
  g_block_opens = false;
  g_device_cv.notify_all();
}

int LiveDeviceCount() {
  std::lock_guard lock(g_device_mutex);
  return static_cast<int>(g_live_devices.size());
}

void SetPcm(AudioOut2::AudioOut2PortHandle port, const void *data) {
  const Pcm pcm{data};
  const Attribute attribute{0, 0, &pcm, sizeof(pcm)};
  Check(AudioOut2::AudioOut2PortSetAttributes(port, AsAttribute(&attribute),
                                              1) == OK,
        "setting PCM failed");
}

void ResetOutputCalls() {
  std::lock_guard lock(g_device_mutex);
  g_output_blocking.clear();
  g_captured_float_output.clear();
  g_captured_output_count = 0;
}

std::vector<bool> OutputCalls() {
  std::lock_guard lock(g_device_mutex);
  return g_output_blocking;
}

std::vector<float> CapturedFloatOutput() {
  std::lock_guard lock(g_device_mutex);
  return g_captured_float_output;
}

uint32_t CapturedOutputCount() {
  std::lock_guard lock(g_device_mutex);
  return g_captured_output_count;
}

void TestSlotReuse() {
  const auto context = CreateContext();
  const auto param = MakeParam();
  for (int i = 0; i < 300; i++) {
    AudioOut2::AudioOut2PortHandle port = 0;
    Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
          "port slot was not reusable");
    Check(port != 0, "port handle is zero");
    AudioOut2::AudioOut2PortDestroy(port);
  }
  AudioOut2::AudioOut2ContextDestroy(context);
}

void TestFullTableRecovers() {
  const auto context = CreateContext();
  const auto param = MakeParam();
  std::vector<AudioOut2::AudioOut2PortHandle> ports;
  ports.reserve(256);

  for (int i = 0; i < 256; i++) {
    AudioOut2::AudioOut2PortHandle port = 0;
    Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
          "port table filled early");
    ports.push_back(port);
  }

  AudioOut2::AudioOut2PortHandle overflow = 0;
  Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &overflow) !=
            OK,
        "full port table accepted another port");

  for (auto port : ports) {
    AudioOut2::AudioOut2PortDestroy(port);
  }

  AudioOut2::AudioOut2PortHandle port = 0;
  Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
        "port table did not recover");
  AudioOut2::AudioOut2PortDestroy(port);
  AudioOut2::AudioOut2ContextDestroy(context);
}

void TestConcurrentCreates() {
  constexpr int thread_count = 8;
  const auto context = CreateContext();
  const auto param = MakeParam(0x800);
  std::vector<AudioOut2::AudioOut2PortHandle> ports(thread_count);
  std::vector<int> results(thread_count);
  std::vector<std::thread> threads;

  BlockDeviceOpens();
  for (int i = 0; i < thread_count; i++) {
    threads.emplace_back([&, i]() {
      results[i] =
          AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &ports[i]);
    });
  }
  WaitForDeviceOpens(thread_count);
  ReleaseDeviceOpens();
  for (auto &thread : threads) {
    thread.join();
  }

  for (int i = 0; i < thread_count; i++) {
    Check(results[i] == OK, "concurrent port create failed");
    PortState state{};
    AudioOut2::AudioOut2PortGetState(ports[i], AsState(&state));
    Check(state.num_channels == 8, "concurrent create lost its reserved slot");
    AudioOut2::AudioOut2PortDestroy(ports[i]);
  }
  Check(LiveDeviceCount() == 0, "concurrent create leaked a device");
  AudioOut2::AudioOut2ContextDestroy(context);
}

void TestContextDestroyCancelsPendingCreate() {
  const auto context = CreateContext();
  const auto param = MakeParam();
  AudioOut2::AudioOut2PortHandle port = 0;
  int result = OK;

  BlockDeviceOpens();
  std::thread creator([&]() {
    result = AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port);
  });
  WaitForDeviceOpens(1);
  AudioOut2::AudioOut2ContextDestroy(context);
  ReleaseDeviceOpens();
  creator.join();

  Check(result != OK, "destroyed context retained a pending port create");
  Check(LiveDeviceCount() == 0, "cancelled port create leaked a device");
}

void TestSynchronousDevicePushBypassesModelledQueue() {
  const auto context = CreateContext(1);
  const auto param = MakeParam();
  AudioOut2::AudioOut2PortHandle port = 0;
  Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
        "device port create failed");

  uint32_t pcm[512]{};
  SetPcm(port, pcm);
  ResetOutputCalls();

  Check(AudioOut2::AudioOut2ContextPush(context, 1) == OK,
        "first sync push failed");
  Check(AudioOut2::AudioOut2ContextPush(context, 1) == OK,
        "device-paced sync push was blocked by modelled queue");
  const auto calls = OutputCalls();
  Check(calls.size() == 2, "sync pushes did not reach the device backend");
  Check(calls[0] && calls[1], "sync pushes lost their blocking mode");

  AudioOut2::AudioOut2PortDestroy(port);
  AudioOut2::AudioOut2ContextDestroy(context);
}

void TestAsynchronousDevicePushKeepsQueueBounded() {
  const auto context = CreateContext(1);
  const auto param = MakeParam();
  AudioOut2::AudioOut2PortHandle port = 0;
  Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
        "device port create failed");

  uint32_t pcm[512]{};
  SetPcm(port, pcm);
  ResetOutputCalls();

  Check(AudioOut2::AudioOut2ContextPush(context, 0) == OK,
        "first async push failed");
  Check(AudioOut2::AudioOut2ContextPush(context, 0) != OK,
        "full async queue accepted another buffer");
  const auto calls = OutputCalls();
  Check(calls.size() == 1 && !calls[0],
        "rejected async push reached the device backend");

  uint32_t queued = 0;
  uint32_t available = 0;
  Check(AudioOut2::AudioOut2ContextGetQueueLevel(context, &queued,
                                                 &available) == OK,
        "queue-level query failed");
  Check(queued == 1 && available == 0,
        "async queue level does not match accepted pushes");

  AudioOut2::AudioOut2PortDestroy(port);
  AudioOut2::AudioOut2ContextDestroy(context);
}

void TestHandleWithoutPcmDoesNotBypassQueue() {
  const auto context = CreateContext(1);
  const auto param = MakeParam();
  AudioOut2::AudioOut2PortHandle port = 0;
  Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
        "device port create failed");
  ResetOutputCalls();

  Check(AudioOut2::AudioOut2ContextPush(context, 1) == OK,
        "empty sync push failed");
  Check(AudioOut2::AudioOut2ContextPush(context, 0) != OK,
        "handle without PCM bypassed queue backpressure");
  Check(OutputCalls().empty(), "empty push reached the device backend");

  AudioOut2::AudioOut2PortDestroy(port);
  AudioOut2::AudioOut2ContextDestroy(context);
}

void TestObjectPortsAreMixedIntoMainOutput() {
  const auto context = CreateContext();

  auto main_param = MakeParam(0x880);
  main_param.port_type = 0;
  AudioOut2::AudioOut2PortHandle main_port = 0;
  Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&main_param),
                                       &main_port) == OK,
        "main mix port create failed");
  std::array<float, 512 * 8> main_pcm{};
  main_pcm[0] = 0.2f;
  main_pcm[1] = -0.1f;
  SetPcm(main_port, main_pcm.data());

  auto object_param = MakeParam(0x100);
  object_param.port_type = 0x0100;
  std::vector<AudioOut2::AudioOut2PortHandle> object_ports;
  std::array<std::array<float, 512>, 5> object_pcm{};
  size_t object_pcm_index = 0;

  const auto add_ambisonics_port = [&](uint32_t channel, float sample) {
    AudioOut2::AudioOut2PortHandle port = 0;
    Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&object_param),
                                         &port) == OK,
          "ambisonics port create failed");
    object_ports.push_back(port);
    auto &samples = object_pcm[object_pcm_index++];
    samples[0] = sample;
    const float gain = 1.0f;
    const uint8_t restricted = 0;
    const Pcm pcm{samples.data()};
    const Attribute attributes[] = {{8, 0, &channel, sizeof(channel)},
                                    {1, 0, &gain, sizeof(gain)},
                                    {9, 0, &restricted, sizeof(restricted)},
                                    {0, 0, &pcm, sizeof(pcm)}};
    Check(AudioOut2::AudioOut2PortSetAttributes(port, AsAttribute(attributes),
                                                std::size(attributes)) == OK,
          "ambisonics attributes failed");
  };

  const auto add_passthrough_port = [&](uint32_t channel, float sample) {
    AudioOut2::AudioOut2PortHandle port = 0;
    Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&object_param),
                                         &port) == OK,
          "passthrough port create failed");
    object_ports.push_back(port);
    auto &samples = object_pcm[object_pcm_index++];
    samples[0] = sample;
    const float gain = 1.0f;
    const Pcm pcm{samples.data()};
    const Attribute attributes[] = {{5, 0, &channel, sizeof(channel)},
                                    {1, 0, &gain, sizeof(gain)},
                                    {0, 0, &pcm, sizeof(pcm)}};
    Check(AudioOut2::AudioOut2PortSetAttributes(port, AsAttribute(attributes),
                                                std::size(attributes)) == OK,
          "passthrough attributes failed");
  };

  add_ambisonics_port(64, 0.1f);
  add_ambisonics_port(65, 0.02f);
  add_ambisonics_port(67, 0.04f);
  add_passthrough_port(1, 0.01f);
  add_passthrough_port(2, 0.02f);

  ResetOutputCalls();
  {
    std::lock_guard lock(g_device_mutex);
    g_capture_float_output = true;
  }
  Check(AudioOut2::AudioOut2ContextPush(context, 1) == OK, "mixed push failed");
  {
    std::lock_guard lock(g_device_mutex);
    g_capture_float_output = false;
  }

  const auto output = CapturedFloatOutput();
  Check(CapturedOutputCount() == 1,
        "object ports opened separate output streams");
  Check(output.size() == main_pcm.size(), "mixed output size is incorrect");
  const auto front = 0.70710678f * 0.1f + 0.86602540f * 0.04f;
  Check(std::abs(output[0] - (0.2f + front + 0.5f * 0.02f + 0.01f)) < 0.00001f,
        "left object mix is incorrect");
  Check(std::abs(output[1] - (-0.1f + front - 0.5f * 0.02f + 0.02f)) < 0.00001f,
        "right object mix is incorrect");
  Check(main_pcm[0] == 0.2f && main_pcm[1] == -0.1f,
        "guest PCM was modified in place");

  for (const auto port : object_ports) {
    AudioOut2::AudioOut2PortDestroy(port);
  }
  AudioOut2::AudioOut2PortDestroy(main_port);
  AudioOut2::AudioOut2ContextDestroy(context);
}

} // namespace

namespace Libs::Audio::AudioInternal {

int AudioOutOpen(int type, uint32_t /*samples_num*/, uint32_t /*freq*/,
                 Format /*format*/) {
  std::unique_lock lock(g_device_mutex);
  const int handle = g_next_device++;
  g_live_devices.push_back(handle);
  if (type != 10) {
    g_device_backed_handles.push_back(handle);
  }
  g_open_waiters++;
  g_device_cv.notify_all();
  g_device_cv.wait(lock, []() { return !g_block_opens; });
  return handle;
}

void AudioOutClose(int handle) {
  std::lock_guard lock(g_device_mutex);
  const auto it =
      std::find(g_live_devices.begin(), g_live_devices.end(), handle);
  if (it != g_live_devices.end()) {
    g_live_devices.erase(it);
  }
  const auto device_it = std::find(g_device_backed_handles.begin(),
                                   g_device_backed_handles.end(), handle);
  if (device_it != g_device_backed_handles.end()) {
    g_device_backed_handles.erase(device_it);
  }
}

bool AudioOutHasDevice(int handle) {
  std::lock_guard lock(g_device_mutex);
  return std::find(g_device_backed_handles.begin(),
                   g_device_backed_handles.end(),
                   handle) != g_device_backed_handles.end();
}

uint32_t AudioOutOutputs(const OutputParam *params, uint32_t num,
                         bool blocking) {
  std::lock_guard lock(g_device_mutex);
  g_output_blocking.push_back(blocking);
  if (g_capture_float_output && params != nullptr && num != 0 &&
      params[0].data != nullptr) {
    const auto *samples = static_cast<const float *>(params[0].data);
    g_captured_float_output.assign(samples, samples + 512 * 8);
    g_captured_output_count = num;
  }
  return 0;
}

} // namespace Libs::Audio::AudioInternal

namespace Libs::LibKernel {

uint64_t KYTY_SYSV_ABI KernelGetProcessTime() {
  static std::atomic_uint64_t now{0};
  return now.fetch_add(1000);
}

} // namespace Libs::LibKernel

int main() {
  TestSlotReuse();
  TestFullTableRecovers();
  TestConcurrentCreates();
  TestContextDestroyCancelsPendingCreate();
  TestSynchronousDevicePushBypassesModelledQueue();
  TestAsynchronousDevicePushKeepsQueueBounded();
  TestHandleWithoutPcmDoesNotBypassQueue();
  TestObjectPortsAreMixedIntoMainOutput();
  std::printf("AudioOut2PortTests: all cases passed\n");
  return 0;
}
