# Project Memory — KytyPS5

Updated: 2026-08-28

## Goal

Improve general PS5 emulation compatibility without per-game hacks. Current validation titles:

- Scars Above (`PPSA08597`, `01.003.000`): gameplay world is black while HUD and markers render.
- Demon's Souls (`PPSA01342`, `01.004.000`): reaches gameplay, but 3D textures/colors remain incorrect.

## Working rules

- Batch related fixes; do not build after every edit.
- Before a build, close only `kyty_emulator.exe` and this project's `launcher.exe` if running.
- Preserve the dirty worktree and unrelated user changes.
- Prefer general, SDK-backed behavior over title-specific branches.
- Validate GPU behavior primarily against the SDK 10 corpus; keep SDK 12 evidence separate.

## Current Scars Above findings

- RenderDoc capture: `_Build/windows/install-scars-pm4-snapshot-20260827/_RenderDoc/kyty_1787875625248664_capture.rdc`.
- The scene geometry exists before final composition; the first known corruption is compute event `47947`.
- Event `47947` writes image `164608` (`R11G11B10_FLOAT`) with NaNs and blue values up to `64512`; its three sampled inputs are healthy.
- The bad path contains FP16 reciprocal/mixed multiply-add operations. This points to FP16/HDR eye-adaptation semantics, not occlusion queries or the HTTP import stubs.
- SDK 10 exposes compute-shader `float_mode`, `ieee_mode`, `dx10_clamp`, and `fp16_overflow`.
- The captured shader used `float_mode=0xc0`, `dx10_clamp=1`, `ieee=0`, and `fp16_overflow=0`.
- The first confirmed semantic mismatch was DX10 min/max: guest NaNs must become `+0` before the operation.
- The second mismatch was host-dependent F32-to-F16 packing. Ordinary half conversion now uses deterministic round-to-nearest-even; the explicit `PKRTZ` path remains round-toward-zero.

## Current texture/stencil finding

- Error `legacy texture upload format unsupported: fmt=0 tile=0 ... 2560x1440` is an internal stencil association proxy being mistaken for a real texture.
- A stencil proxy has no usable guest image layout and must never enter generic upload/reinterpret paths. A proxy promoted from an existing sampled owner may retain a dormant Vulkan backing until retirement.
- Sampled R8 stencil access may resolve through the depth/stencil owner when ranges and geometry match.
- Storage-image R8 access must not redirect to a depth image lacking Vulkan storage usage.
- A shared sampled R8 owner can become a stencil proxy after descriptor discovery. Every prepared
  occurrence must then be rediscovered from its original descriptor; `binding.needs_rebind` alone
  is insufficient because resolving the first occurrence clears that shared flag while later
  occurrences still hold the proxy `ImageId`.

## Verified texture rediscovery fix — 2026-08-28

- `RenderExecutor::RebindImages` now treats any `depth_id` record as a non-acquirable proxy and
  resolves the original descriptor again before calling `FindTexture`.
- The regression uses two sampled descriptors sharing one R8 owner, promotes it to a stencil proxy,
  keeps a storage alias independent, and verifies that both sampled entries reach the depth owner.
- SDK 10 corroborates that a depth target may alias depth/stencil textures only when their format,
  dimensions, array view, tile mode, fragments, and texture type are compatible; the cache keeps
  those compatibility checks separate from final Vulkan view acquisition.
- Focused tests passed: `--image-overlap-only`, `--fp-state-only`, and
  `--byte-geometry-alias-only`.
- Full Release build succeeded.
- Test package: `_Build/windows/install-scars-rediscovery-20260828`.

## Verified 2026-08-28 batch

- Compute FP state is carried into translation, dumps, and compute shader cache identity.
- DX10 NaN classification is bitwise IEEE-754, avoiding driver-dependent unordered comparisons.
- FP16 overflow mode is implemented for half arithmetic: disabled produces infinity; enabled clamps finite overflow to `±65504` while preserving genuine infinity/NaN.
- F32-to-F16 conversion and ordinary half packing use deterministic RNE, preserve half subnormals, signed zero, infinity, and quiet NaN payloads.
- Stencil association is restricted to compatible sampled `k8UInt/eR8Uint`; storage aliases remain independent and cannot mask later sampled lookup.
- Focused tests passed: `--fp-state-only`, `--image-overlap-only`, and `--byte-geometry-alias-only`.
- Full Release build succeeded.
- Test package: `_Build/windows/install-scars-fp16-rne-stencil-20260828`.

## Implemented foundations in the dirty tree

- GPU-native image backing reinterpretation for byte-compatible guest aliases.
- DCC-aware alias handling for the Scars 1920x1080 RGBA8 ↔ 3840x2160 R8 range.
- Stencil association proxy filtering in texture-cache overlap discovery.
- Broad shader/PM4/format/resource-tracking work accumulated during Demon's Souls and Scars testing.

## Next verification batch

1. Run Scars Above from `install-scars-rediscovery-20260828`.
2. Confirm that `TextureCache: texture requires rediscovery before final acquisition` is gone and
   whether the gameplay world renders.
3. If the world remains black, capture a new frame and compare event `47947`: DX10 min/max must show
   NaN-to-`+0`, the old `PackHalf2x16` extinst must be absent, and output image `164608` must be
   checked again for NaNs.

## Memory mechanism

- The requested `$maintain-project-memory` skill was not installed or available in this session.
- `PROJECT_MEMORY.md` is the repository-local fallback and must be updated after each verified debugging batch.
