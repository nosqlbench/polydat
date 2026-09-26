# Guides

* [Compilation Levels](compilation.md) - The interpreter, the closure tier, the hybrid kernel, and native code: features and trade-offs.
* [Embedding Polydat](embedding.md) - What a host owns and what polydat owns, the APIs for compiling, driving, sharing, and extending a kernel, and the extension points.
* [Engine-ladder performance](performance.md) - One typed graph and one tile measured on all four engines, with the measurement contract, correctness gate, and benchmark commands.
* [Porting to 0.3.2](porting_to_0_3_2.md) - Every surface change from 0.3.1: the renames a host applies, the type change that fails elsewhere, and the decisions polydat stopped making for a host.
* [Porting to 0.5.0](porting_to_0_5_0.md) - The 0.5.0 release notes as a host reads them: breaking changes and fixes, deprecated healing writes and their replacements, quiet corrections, and known issues.
* [Releasing polydat](releasing.md) - The steps of a release, including raising the internal dependency requirements so a host's lockfile picks up every crate the release publishes.
