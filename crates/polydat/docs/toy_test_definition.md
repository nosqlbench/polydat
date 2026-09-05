# A toy test definition in one grammar file

This is a capsule-form example: one Polydat source that carries its
own parameters, its own description, a hierarchic dataset, a reusable
model, and the load, read, and verify statements of a test flow. Every
row of the dataset and every statement of the flow is a pure function
of one coordinate, so any of them can be regenerated on any fiber or
host without state.

The file is
[`examples/toy_test_definition.polydat`](../examples/toy_test_definition.polydat).
It is driven by the `polydat` binary with no host code. From the repository:

```text
cargo run -p polydat -- run crates/polydat/examples/toy_test_definition.polydat --partition 1 --cycles 3 --emit map
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

The binary plays the host. It compiles the file, resolving the in-file
module from the file's own directory, then resolves the cursor's `*/4`
spec into four partitions. A run represents one partition with
`--partition`, or spreads them across fibers when `--fibers` matches the
partition count. Externs and inputs are assigned with bare `name=value`
arguments. Emission is a graph
transform: `--emit` appends an `emit_row` binding naming the selected
wires, so the rows below come from a node inside the kernel.

One fiber on partition 1, the second quarter of the row domain:

```text
$ polydat run toy_test_definition.polydat --partition 1 --cycles 3 --emit map \
    --outputs row,tenant,device,reading,device_kind,ts,status
cursor rows: partition 2/4 [250000, 500000)
row=250000 tenant=0 device=0 reading=250 device_kind=sensor ts=1700000250000 status=ok
row=250001 tenant=1 device=0 reading=250 device_kind=gateway ts=1700000250000 status=ok
row=250002 tenant=2 device=0 reading=250 device_kind=gateway ts=1700000250000 status=ok
```

Four fibers, one per partition, in the verify phase with one-minute
intervals, one cycle each:

```text
$ polydat run toy_test_definition.polydat phase=verify interval_ms=60000 \
    --fibers 4 --cycles 1 --emit csv --outputs row,tenant_id,stmt -q
row,tenant_id,stmt
0,607535,"expect temp_c = 22.282993976163535, humidity = 35.548376405822175, status = 'ok'"
250000,607535,"expect temp_c = 19.322731887641915, humidity = 49.560474909266304, status = 'ok'"
500000,607535,"expect temp_c = 21.696623542008687, humidity = 60.56241450142866, status = 'ok'"
750000,607535,"expect temp_c = 20.750799765843645, humidity = 67.01162199294797, status = 'ok'"
```

`polydat check --stats`, `polydat explain`, and `polydat viz` all accept
the same file.

## Reading the output

- **Each fiber starts at its own slice.** Cycle 0 on partition 1 is row
  250000, which is reading 250 of device 0 in tenant 0: the 1000-row
  stride of the first two dimensions has been walked 250 times by the
  ordinals below the slice. The four fibers above produce rows 0, 250000,
  500000, and 750000 for the same local cycle. The slices never overlap.
- **Identity is stable across slices.** All four fibers land on tenant 0
  and device 0 for cycle 0, and all four get the same tenant id. Only the
  reading and its derived values differ.
- **The override took effect.** With `phase=verify` the selected
  statement is the verify form, and `interval_ms=60000` spaces
  readings a minute apart.
- **Load, read, and verify agree by construction.** All three
  statements pull the same wires. A verifier that regenerates the row
  from its ordinal gets the expected values without consulting the
  load phase.

## What this does not show

The `over` clause and the `for` comprehension surface are driven by a
host such as nmbrs, which resolves partitions, activates scopes, and
streams coordinate tuples into kernels. This file declares those
contracts; the binary performs the host's part of them. The draft [`for` construct](design/for_traversal.md) moves that
work into Polydat, and its worked example is this definition in the
target form: the flow declared as a comprehension producer, and the
body traversed once per tuple with `phase`, `interval_ms`, and `p` as
wires. See [Cursor Partitions](design/cursor_partitions.md) and
[Comprehension Forms](design/comprehension_forms.md) for the full
host-facing contracts, and [Illustrations](illustrations.md) for each
feature shown on its own.
