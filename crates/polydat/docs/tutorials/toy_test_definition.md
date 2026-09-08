# A toy test definition in one grammar file

This is a capsule-form example: one Polydat source that carries its
own parameters, its own description, a reusable model, a test flow
bound as a comprehension, and a traversal that unwinds a hierarchic
dataset and derives the load, read, and verify statements of that flow
in every activation. Every row and every statement is a pure function
of one coordinate, so any of them can be regenerated on any fiber or
host without state.

The file is
[`examples/toy_test_definition.polydat`](../../examples/toy_test_definition.polydat).
It is driven by the `polydat` binary with no host code. From the repository:

```text
cargo run -p polydat -- run crates/polydat/examples/toy_test_definition.polydat --cycles 2 --emit map
```

## The grammar

```text
// Toy test definition: a self-describing, hierarchic, parameterized
// dataset and test flow in one Polydat grammar file.
//
// Hierarchy:  cycle -> row -> (tenant, device, reading)
// Parameters: externs with defaults; a host may override any of them.
// Traversal:  a comprehension over phases, intervals, and partitions of
//             the row domain, bound as a producer and traversed below.
// Flow:       schema, load, read, and verify statements derived from
//             the same coordinate, so any row can be regenerated. The
//             load statement carries a JSON document rendered by a tile.

// ---- Coordinates and parameters ---------------------------------------

// The root coordinate. Each traversal activation below has its own
// `cycle`, local to its slice; this one positions the root scope.
input cycle: u64

// Runtime parameters are externs with defaults; a host may assign them
// on the command line without recompiling.
extern base_epoch_ms: u64 = 1700000000000
extern rows_total: u64 = 1000000

// ---- Self-description ------------------------------------------------

const dataset := "iot-readings-toy"
const keyspace := "toy"
const table := "readings"
const schema_stmt := "CREATE TABLE {keyspace}.{table} (tenant_id bigint, device_id text, ts bigint, doc text, PRIMARY KEY ((tenant_id, device_id), ts))"
const shape := "tenants=20 devices_per_tenant=50 readings=unbounded"

// ---- Reusable model ---------------------------------------------------

reading_model(seed: u64) -> (temp_c: f64, humidity: f64, status: str) := {
    temp_c   := normal_sample(input: seed, mean: 21.5, stddev: 2.0)
    humidity := uniform_sample(input: hash(seed), min: 30.0, max: 70.0)
    status   := weighted_strings(hash(hash(seed)), "ok:0.97;degraded:0.02;error:0.01")
}

// ---- Traversal ----------------------------------------------------------

// The test flow is a comprehension bound as a producer: every phase runs
// over every quarter of the row domain, at two reading intervals. Its
// element names are wires inside the traversal below.
flow := for phase in load,verify, interval_ms in 1000,60000, p in partitions("*/4", {rows_total})

for flow {
    // Each activation owns the slice `p` of the row domain. The cursor is
    // narrowed to `p` at activation, and the local cycle maps onto an
    // absolute row ordinal inside the slice.
    cursor rows = range(0, rows_total) over p
    row := mod_in(cycle, rows.cursor)

    // Structural parameters are literal constants in the decomposition.
    // The first two dimensions are bounded; the third is unbounded, so
    // the dataset grows with the row count.
    (tenant, device, reading) := mixed_radix(row, 20, 50, 0)

    // ---- Entities -----------------------------------------------------

    tenant_id   := hashed_id(input: tenant, bound: 1000000)
    device_key  := interleave(tenant, device)
    device_id   := hashed_uuid(device_key)
    device_kind := weighted_strings(device_key, "sensor:0.7;gateway:0.2;controller:0.1")

    reading_seed := hash(interleave(device_key, reading))
    (temp_c, humidity, status) := reading_model(reading_seed)
    ts := base_epoch_ms + reading * interval_ms

    // A coarse health signal derived from the same coordinate: nonzero
    // when the reading should be flagged by a verifier.
    flagged := if temp_c > 26.0 { 1 } else { 0 }

    // ---- Document -----------------------------------------------------

    // The reading as a JSON document. A tile is a template whose holes
    // are wires: numbers render bare and strings quoted by their types,
    // the `meta` arm is one static copy, and `samples` repeats its body
    // over a comprehension. The tile is a wire like any other.
    tile doc : json := {
        "meta": { "schema": 3, "source": "polydat", "units": { "temp": "C", "rh": "%" } },
        "tenant": ${tenant_id},
        "device": ${device_id},
        "kind": ${device_kind},
        "ts": ${ts},
        "reading": { "temp": ${temp_c | .2}, "rh": ${humidity | .1}, "status": ${status} },
        "samples": [ @for s in 0..4 { { "n": ${s}, "temp": ${temp_c + s | .2} } } ],
        "flagged": ${flagged: bool}
    }

    // ---- Test flow ----------------------------------------------------

    // The load statement carries the document raw: it is already JSON.
    tile load := <<<
INSERT INTO ${keyspace}.${table} (tenant_id, device_id, ts, doc) VALUES (${tenant_id}, '${device_id}', ${ts}, '${doc!}')
>>>
    read_stmt   := "SELECT doc FROM {keyspace}.{table} WHERE tenant_id = {tenant_id} AND device_id = '{device_id}' AND ts = {ts}"
    verify_stmt := "expect temp_c = {temp_c}, humidity = {humidity}, status = '{status}'"

    // The statement this activation executes is selected by its phase.
    stmt := select_str(str_eq(phase, "load"), load, verify_stmt)

}
```

## What each part does

| Construct | Feature | Role in the definition |
| --- | --- | --- |
| `input cycle: u64` | Root coordinate | Positions the root scope. Each activation has its own local `cycle`. |
| `extern ... = default` | Runtime parameters | Typed slots with defaults. A bare `name=value` argument to the binary rewrites the default before compilation. |
| `const ... :=` | Self-description | Values computed once at scope init: the dataset name, keyspace, table, schema statement, and the shape of the coordinate space. |
| `reading_model(...) := { ... }` | Module | A typed, reusable computation with three outputs, resolved from the same file. |
| `flow := for ...` | Producer | Binds the test flow as a comprehension value: two phases, two intervals, and four partitions of the row domain. `{rows_total}` reads the extern when the traversal opens. |
| `for flow { ... }` | Traversal | Compiles the body once into a child program and activates it once per tuple, with `phase`, `interval_ms`, and `p` as typed wires and `base_epoch_ms`, `keyspace`, `table`, and `rows_total` cascaded from the root. |
| `cursor rows = ... over p` | Partitioned cursor | Narrowed to the activation's partition. Its slice sets how many cycles the activation runs. |
| `mod_in(cycle, rows.cursor)` | Slice projection | Maps the activation's local cycle onto an absolute row ordinal inside its slice. |
| `mixed_radix(row, 20, 50, 0)` | Hierarchy | Unwinds one ordinal into tenant, device, and reading. The trailing `0` leaves readings unbounded. |
| `hashed_id`, `normal_sample`, `uniform_sample` | Standard library | Modules from the embedded `.polydat` library, called with named arguments. |
| `"... {expr} ..."` | String interpolation | The schema, read, and verify statements are ordinary bindings that embed typed wires. |
| `tile doc : json := { ... }` | Tile | The reading as a JSON document. Holes are wires: `u64` and `f64` render bare, `Str` quoted, `${flagged: bool}` as a boolean; `\| .2` is a format; the `meta` arm is one static copy; `@for s in 0..4` repeats its body over a comprehension. |
| `tile load := <<< ... >>>` | Tile carrying a tile | The load statement, with the document inlined raw through `${doc!}`. Both tiles are wires like any other. |
| `select_str(str_eq(phase, "load"), ...)` | Phase selection | The statement an activation executes follows its `phase` element. |

## Running it

The binary plays the host. It compiles the file, resolving the in-file
module from the file's own directory, and runs it in traversal mode:
the emit binding is inserted into the `for` body, the sixteen
activations are taken by index across fibers, and each activation runs
its cycles under the traversal rule, capped by `--cycles`. Externs are
assigned with bare `name=value` arguments.

Two cycles per activation, showing the hierarchy each slice starts from:

```text
$ polydat run toy_test_definition.polydat --cycles 2 --emit map \
    --outputs phase,interval_ms,row,tenant,device,reading,ts,status
traversal 0: for flow  (16 activations)
phase=load interval_ms=1000 row=0 tenant=0 device=0 reading=0 ts=1700000000000 status=ok
phase=load interval_ms=1000 row=1 tenant=1 device=0 reading=0 ts=1700000000000 status=ok
phase=load interval_ms=1000 row=250000 tenant=0 device=0 reading=250 ts=1700000250000 status=ok
phase=load interval_ms=1000 row=250001 tenant=1 device=0 reading=250 ts=1700000250000 status=ok
phase=load interval_ms=1000 row=500000 tenant=0 device=0 reading=500 ts=1700000500000 status=ok
phase=load interval_ms=1000 row=500001 tenant=1 device=0 reading=500 ts=1700000500000 status=ok
```

The document each reading carries, one per activation with one cycle
each. Naming the tile in `--emit` writes its rendered text and nothing
else; the first of the sixteen:

```text
$ polydat run toy_test_definition.polydat --cycles 1 --emit tile:doc -q
{
    "meta": { "schema": 3, "source": "polydat", "units": { "temp": "C", "rh": "%" } },
    "tenant": 607535,
    "device": "d9ac876f-bb3a-4bc7-b9f8-382893178079",
    "kind": "sensor",
    "ts": 1700000000000,
    "reading": { "temp": 22.28, "rh": 35.5, "status": "ok" },
    "samples": [ { "n": 0, "temp": 22.28 },{ "n": 1, "temp": 23.28 },{ "n": 2, "temp": 24.28 },{ "n": 3, "temp": 25.28 } ],
    "flagged": false
}
```

The load statement carries that document inline. One activation's
statement, as a CSV row:

```text
$ polydat run toy_test_definition.polydat --cycles 1 --emit csv --outputs phase,stmt -q
phase,stmt
load,"INSERT INTO toy.readings (tenant_id, device_id, ts, doc) VALUES (607535, 'd9ac876f-bb3a-4bc7-b9f8-382893178079', 1700000000000, '{
    ""meta"": { ""schema"": 3, ""source"": ""polydat"", ""units"": { ""temp"": ""C"", ""rh"": ""%"" } },
    ...
    ""flagged"": false
}')"
```

Assigning the externs reshapes the run without recompiling anything but
the two rewritten declarations:

```text
$ polydat run toy_test_definition.polydat base_epoch_ms=5 rows_total=400 --cycles 1 --emit map --outputs row,ts -q
row=0 ts=5
row=100 ts=5
row=200 ts=5
row=300 ts=5
```

`polydat check --stats` reports two programs, the root and the one
traversal body; `polydat explain traversals` prints the body's elements
and cascade, and `polydat explain tiles` prints each tile's skeleton and
how every hole was typed and encoded.

## Reading the output

- **Each activation starts at its own slice.** The traversal's tuples
  are ordered by phase, then interval, then partition, so the first four
  activations are the four quarters of the row domain in the load phase
  at a one-second interval. Row 250000 is reading 250 of device 0 in
  tenant 0, because the 1000-row stride of the first two dimensions has
  been walked 250 times by the ordinals below the slice. Slices never
  overlap.
- **Identity is stable across slices.** All four quarters land on tenant
  0 and device 0 for their first row and get the same tenant id and
  device uuid. Only the reading and its derived values differ.
- **Elements drive the body.** The fifth activation is the same load
  phase and first quarter at a one-minute interval; `ts` would step by
  60000 between its readings. The verify-phase activations select the
  verify statement through `phase`.
- **Externs reach the traversal.** `rows_total` sizes both the
  partition spec and the cursor, so assigning it to 400 makes each
  quarter 100 rows. `base_epoch_ms` cascades into the body as the
  timestamp base.
- **The document is a wire.** `doc` renders once per cycle from the
  same coordinate as everything else, so the load statement, the
  verify statement, and the document can never disagree about a
  reading. `${temp_c | .2}` and `${humidity | .1}` fix the precision
  the document carries; the verify statement keeps the full values.

## What this shows and what it does not

The whole flow is declared in the grammar: the comprehension, the
partitioning, the per-activation cursor, and the statements. Polydat
compiles the body once and the binary activates it sixteen times; the
build counter stays flat while it does. What the file does not decide is
scheduling: how many fibers, how many cycles per activation, and what to
do with each statement are the host's choices, and the binary exposes
them as options. The contract is [The `for` Construct](../design/for_traversal.md);
the document tiles are [Polytile](../design/polytile.md), walked through in
[the Polytile tutorial](polytile_tutorial.md); each feature is shown on
its own in [Illustrations](illustrations.md).
