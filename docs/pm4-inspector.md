# PM4 submission inspector

Enable **Command Buffer Dump** in the launcher, or pass:

```text
--command-buffer-dump true --command-buffer-dump-folder _Buffers
```

Kyty continues to create its existing text dumps and additionally writes one
`*.json` file per submitted graphics DCB or asynchronous-compute ACB.

Each JSON trace contains:

- The queue, frame, source address, and copied root packet stream.
- Readable nested indirect command buffers captured at submission time.
- Readable indirect context, shader, and uconfig register pairs.
- Register-state deltas and a persistent tracked-state hash per queue.
- The local AGC resource-registration map and address-to-resource annotations.
- Inferred buffer, texture, and sampler snapshots reached through shader user-data pairs.
- An automatic `comparison_to_previous` section for each queue after its first capture.
- Packet classifications: `Known`, `Partial`, `Inferred`, `Unknown`, `Unsupported`, or
  `UnreadableAtCaptureTime`.

`Known` describes the inspector's structural understanding. It does not claim that every
runtime behavior or register semantic is implemented. Descriptor candidates are deliberately
classified as `Inferred`: unreadable candidates are omitted instead of being treated as valid
resources.

The comparison is queue-local and reports added or removed packet, descriptor, and registered
resource signatures plus changes to the tracked register-state hash. This keeps captures
self-contained while making two consecutive submissions directly comparable.

The older internal `constant_commands` stream is not labeled as an AGC CCB. Official AGC
terminology uses Draw Command Buffer (DCB) and Async Command Buffer (ACB).
