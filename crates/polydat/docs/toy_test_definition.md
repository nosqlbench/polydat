# A toy test definition in one grammar file

This is a capsule-form example: one Polydat source that carries its
own parameters, its own description, a hierarchic dataset, a reusable
model, and the load, read, and verify statements of a test flow. Every
row of the dataset and every statement of the flow is a pure function
of one coordinate, so any of them can be regenerated on any fiber or
host without state.

The file is
[`examples/toy_test_definition.polydat`](../examples/toy_test_definition.polydat)
and the runner is
[`examples/toy_test_definition.rs`](../examples/toy_test_definition.rs).
Run it with:

```text
cargo run -p polydat --example toy_test_definition
```

## The grammar

```text
// Toy test definition: a self-describing, hierarchic, parameterized
// dataset and test flow in one Polydat grammar file.
//
// Hierarchy:  cycle -> (tenant, device, reading)
// Parameters: externs with defaults; a host may override any of them.
// Flow:       schema, load, read, and verify statements derived from
//             the same coordinate, so any row can be regenerated.

// ---- Parameters -------------------------------------------------------

// Runtime parameters are externs with defaults; a host may override
// them per scope activation without recompiling.
extern base_epoch_ms: u64 = 1700000000000
extern interval_ms: u64 = 1000

// ---- Self-description ------------------------------------------------

const dataset := "iot-readings-toy"
const keyspace := "toy"
const table := "readings"
const schema_stmt := "CREATE TABLE {keyspace}.{table} (tenant_id bigint, device_id text, ts bigint, temp_c double, humidity double, status text, PRIMARY KEY ((tenant_id, device_id), ts))"

// ---- Coordinates and hierarchy ---------------------------------------

input cycle: u64

// A fiber owns one slice of the row domain. The host resolves the
// partition spec and narrows this cursor at scope setup; the fiber's
// local cycle then maps onto an absolute row ordinal in its slice.
cursor rows = range(0, 1000000) over "*/4"
row := mod_in(cycle, rows.cursor)

// Structural parameters are literal constants in the decomposition.
// The first two dimensions are bounded; the third is unbounded, so
// the dataset grows with the row count.
(tenant, device, reading) := mixed_radix(row, 20, 50, 0)
const shape := "tenants=20 devices_per_tenant=50 readings=unbounded"

// ---- Reusable model ---------------------------------------------------

reading_model(seed: u64) -> (temp_c: f64, humidity: f64, status: String) := {
    temp_c   := normal_sample(input: seed, mean: 21.5, stddev: 2.0)
    humidity := uniform_sample(input: hash(seed), min: 30.0, max: 70.0)
    status   := weighted_strings(hash(hash(seed)), "ok:0.97;degraded:0.02;error:0.01")
}

// ---- Entities ---------------------------------------------------------

tenant_id   := hashed_id(input: tenant, bound: 1000000)
device_key  := interleave(tenant, device)
device_id   := hashed_uuid(device_key)
device_kind := weighted_strings(device_key, "sensor:0.7;gateway:0.2;controller:0.1")

reading_seed := hash(interleave(device_key, reading))
(temp_c, humidity, status) := reading_model(reading_seed)
ts := base_epoch_ms + reading * interval_ms

// ---- Test flow ----------------------------------------------------------

load_stmt   := "INSERT INTO {keyspace}.{table} (tenant_id, device_id, ts, temp_c, humidity, status) VALUES ({tenant_id}, '{device_id}', {ts}, {temp_c}, {humidity}, '{status}')"
read_stmt   := "SELECT temp_c, humidity, status FROM {keyspace}.{table} WHERE tenant_id = {tenant_id} AND device_id = '{device_id}' AND ts = {ts}"
verify_stmt := "expect temp_c = {temp_c}, humidity = {humidity}, status = '{status}'"

// A coarse health signal derived from the same coordinate: nonzero when
// the reading should be flagged by a verifier.
flagged := if temp_c > 26.0 { 1 } else { 0 }
```

## What each part does

| Lines | Feature | Role in the definition |
| --- | --- | --- |
| `extern ... = default` | Runtime parameters | Typed slots with defaults. A host can override them per scope activation without recompiling. |
| `const ... :=` | Self-description | Values computed once at scope init: the dataset name, the keyspace and table, the schema statement, and the shape of the coordinate space. |
| `input cycle: u64` | Coordinate | The one value a fiber advances. Everything else derives from it. |
| `cursor rows = ... over "*/4"` | Partitioned cursor | Names the row domain and declares that a fiber owns one of four slices. The host resolves the spec at scope setup. |
| `mod_in(cycle, rows.cursor)` | Slice projection | Maps the fiber's local cycle onto an absolute row ordinal inside its slice. |
| `mixed_radix(row, 20, 50, 0)` | Hierarchy | Unwinds one ordinal into tenant, device, and reading. The trailing `0` leaves readings unbounded. |
| `reading_model(...) := { ... }` | Module | A typed, reusable computation with three outputs. Resolved by the module loader from the same file. |
| `hashed_id`, `normal_sample`, `uniform_sample` | Standard library | Modules from the embedded `.polydat` library, called with named arguments. |
| `interleave`, `hash`, `hashed_uuid`, `weighted_strings` | Entity derivation | Independent, collision-resistant identity and attribute streams from the same coordinate. |
| `"... {expr} ..."` | String interpolation | The schema, load, read, and verify statements are ordinary bindings that embed typed wires. |
| `if temp_c > 26.0 { 1 } else { 0 }` | Conditional | A derived signal a verifier can act on, computed on the same substrate. |

## Running it

The runner plays the host. It compiles the file with the examples
directory as a module library path so the in-file module resolves,
resolves the `*/4` partition spec and writes fiber 1's slice into the
cursor slots, overrides `interval_ms` to one minute, and pulls a few
rows.

```text
dataset: iot-readings-toy
shape:   tenants=20 devices_per_tenant=50 readings=unbounded
fiber:   partition 1 of 4 = [250000, 500000)
schema:  CREATE TABLE toy.readings (tenant_id bigint, device_id text, ts bigint, temp_c double, humidity double, status text, PRIMARY KEY ((tenant_id, device_id), ts))

cycle 0: row=250000 tenant=0 device=0 reading=250 kind=sensor
  load:   INSERT INTO toy.readings (tenant_id, device_id, ts, temp_c, humidity, status) VALUES (607535, 'd9ac876f-bb3a-4bc7-b9f8-382893178079', 1700015000000, 19.322731887641915, 49.560474909266304, 'ok')
  read:   SELECT temp_c, humidity, status FROM toy.readings WHERE tenant_id = 607535 AND device_id = 'd9ac876f-bb3a-4bc7-b9f8-382893178079' AND ts = 1700015000000
  verify: expect temp_c = 19.322731887641915, humidity = 49.560474909266304, status = 'ok'  flagged=0
cycle 1: row=250001 tenant=1 device=0 reading=250 kind=gateway
  load:   INSERT INTO toy.readings (tenant_id, device_id, ts, temp_c, humidity, status) VALUES (822465, '2e79db64-3559-4c2f-8afc-353405f2fcd0', 1700015000000, 19.83699105568422, 38.567674388524416, 'ok')
  read:   SELECT temp_c, humidity, status FROM toy.readings WHERE tenant_id = 822465 AND device_id = '2e79db64-3559-4c2f-8afc-353405f2fcd0' AND ts = 1700015000000
  verify: expect temp_c = 19.83699105568422, humidity = 38.567674388524416, status = 'ok'  flagged=0
cycle 1000: row=251000 tenant=0 device=0 reading=251 kind=sensor
  load:   INSERT INTO toy.readings (tenant_id, device_id, ts, temp_c, humidity, status) VALUES (607535, 'd9ac876f-bb3a-4bc7-b9f8-382893178079', 1700015060000, 19.30569487327788, 41.76980748475056, 'ok')
  read:   SELECT temp_c, humidity, status FROM toy.readings WHERE tenant_id = 607535 AND device_id = 'd9ac876f-bb3a-4bc7-b9f8-382893178079' AND ts = 1700015060000
  verify: expect temp_c = 19.30569487327788, humidity = 41.76980748475056, status = 'ok'  flagged=0
cycle 12345: row=262345 tenant=5 device=17 reading=262 kind=sensor
  load:   INSERT INTO toy.readings (tenant_id, device_id, ts, temp_c, humidity, status) VALUES (358618, '4376abb8-fee3-41db-82ca-4a82f7b6acc9', 1700015720000, 17.70320094594486, 57.480835928562946, 'ok')
  read:   SELECT temp_c, humidity, status FROM toy.readings WHERE tenant_id = 358618 AND device_id = '4376abb8-fee3-41db-82ca-4a82f7b6acc9' AND ts = 1700015720000
  verify: expect temp_c = 17.70320094594486, humidity = 57.480835928562946, status = 'ok'  flagged=0
```

## Reading the output

- **Fiber 1 starts at row 250000.** Cycle 0 on this fiber is reading
  250 of device 0 in tenant 0, because the 1000-row stride of the
  first two dimensions has already been walked 250 times by the
  ordinals below its slice. Fiber 0 would produce reading 0 for the
  same cycle. The slices never overlap.
- **Identity is stable across rows.** Cycles 0 and 1000 land on the
  same tenant and device, and they get the same tenant id and device
  uuid. Only the reading and its derived values change.
- **The override took effect.** Timestamps step by 60000 between
  readings 250 and 251 rather than the declared 1000.
- **Load, read, and verify agree by construction.** All three
  statements pull the same wires. A verifier that regenerates the row
  from its ordinal gets the expected values without consulting the
  load phase.

## What this does not show

The `over` clause and the `for` comprehension surface are driven by a
host such as nmbrs, which resolves partitions, activates scopes, and
streams coordinate tuples into kernels. This file declares those
contracts; the runner simulates the host's part of them in a few
lines. The draft [`for` construct](design/for_traversal.md) moves that
work into Polydat, and its worked example is this definition in the
target form: the flow declared as a comprehension producer, and the
body traversed once per tuple with `phase`, `interval_ms`, and `p` as
wires. See [Cursor Partitions](design/cursor_partitions.md) and
[Comprehension Forms](design/comprehension_forms.md) for the full
host-facing contracts, and [Illustrations](illustrations.md) for each
feature shown on its own.
