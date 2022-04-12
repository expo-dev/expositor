# Expositor

Expositor is a UCI-conforming chess engine for AMD64 / Intel 64 systems.[^1] You can [play against her on Lichess](https://lichess.org/@/expositor) or download a copy for local use.

[^1]: This is because the network code makes use of inline assembly and some multithreaded code assumes total store order. In practice, you could probably run the engine proper without significant problems on an architecture with a weaker memory model (e.g. ARM), but in theory, this could cause incorrect behavior.

Expositor has a [CCRL Blitz](https://www.computerchess.org.uk/ccrl/404/index.html) rating of 3119.

There are no command line options, but the engine does support some nonstandard commands; for more information, start Expositor and enter `help`.

**Note** that the transposition table is effectively cleared between searches.[^2]

[^2]: This is currently not an option and cannot be disabled. It's an intentional choice – it means singlythreaded searches are deterministic regardless of the state of the transposition table. The effect is achieved by tagging each table entry with a generation and does not incur the penalty of actually zeroing the table.

## Releases

If you know the microarchitecture of your processor, try using a binary from the appropriate `specific/` directory of a release archive.[^3][^4] If you don't know the microarchitecture of your processor but you do know which features it supports, try using a binary from the appropriate `generic/` directory of a release archive. See the file named "extensions" in this repository for more information.

The binaries include the default network, so you do not need to download a separate copy and do not need to set the `EvalFile` UCI option.

[^3]: I'm not completely confident that I matched the proper cpu-target for some AMD microarchitectures; please let me know if I've made any mistakes.

[^4]: I've attempted to include binaries for most consumer desktop hardware but not server or mobile platforms. If you have a Xeon or Atom processor, for example, and want a targeted binary, you'll need to compile from source. Feel free to reach out to me for help or if you'd like me to include a binary for your platform in releases.

## Building

If you'd like to compile Expositor from source, you will need to use a recent nightly toolchain.

To build Expositor on Windows, run the `build.bat` script. (This sets the `VERSION`, `BUILD`, and `RUSTFLAGS` environment variables and then invokes `cargo build --release`.)

To build Expositor on Linux, run the `build` script. (This needs to be done from within the repository, since it uses Git to automatically determine the version number.)

## Issues

If you find any bugs or have any questions, please file an issue on Github or send me a message.

<!-- to be added once more than version has been released
## Versions

| Version | Release Date |        Notes         |
|---------|--------------|----------------------|
|  2WQ23  |  23 Feb 2022 | First public release |
-->

## Pending

The to-do list has gotten to be rather long and variegated. My current plan is to tackle the smaller items first (aiming for a release in March or April) and then to begin work on some larger projects:

**Tuning Search**&ensp;None of the search constants have been tuned! I expect to be able to wring a fair amount of playing strength out of that.

**HCE Bootstrapping**&ensp;The neural network is currently trained from positions scored with Stockfish. I'd like to write an evaluator that replicates the personality of early versions of Expositor, train a network from positions scored by Expositor using that evaluator, then train another network from positions scored using the previous network, and so on.

**Experimental Network Architectures**&ensp;I'd like to try using different input features and play around with small convolutional networks.

**Error Tracking Search**&ensp;I've been reading and thinking about this since I started chess programming and at one point it was my primary focus. I'd like to pick it up again, deliver a working proof of concept of my ideas, and then write up my findings.

**Infrastructure**&ensp;I'd like to write my own automation system (à la OpenBench) that will kick off SPRT testing whenever I push a commit.

## Acknowledgments

Three sources are currently used to generate training positions:
- [the Lichess database](https://database.lichess.org/)
- [old Ethereal datasets](http://talkchess.com/forum3/viewtopic.php?t=75350)
- [the TCEC games archive](https://github.com/TCEC-Chess/tcecgames)

These are processed by Expositor with her quiescing search and the leaves from those searches are then scored with [Stockfish](https://github.com/official-stockfish/Stockfish).

Most of the positions used for perft tests are from the [Zahak](https://github.com/amanjpro/zahak) repository.

Particular thanks to [Jeremy Wright](https://github.com/jtheardw/mantissa) for help with main search. Many techniques were implemented from his descriptions and use parameter values he suggested.

## Terms of Use

Expositor is free and distributed under the terms of version 3 of the GNU Affero General Public License. You are welcome to run the program, modify it, copy it, sell it, or use it in a project of your own.

If you distribute the program, verbatim or modified, you must provide the source and extend to anyone who obtains a copy the same license that I am granting you.

If users can interact with a modified version of the program (or a work based on the program) remotely through a computer network, you must provide a way for users to obtain its source.

For more details, see the file named "license" in this repository.

### Note

Although the GNU Affero General Public License is a libre ("free") license in spirit, it is debatable whether programs licensed under its terms meet [the usual definition of libre software](http://www.gnu.org/philosophy/free-sw.html). Two of the freedoms that users of libre software have are the freedom to run the software as they wish, for any purpose, and the freedom to change it. However, the GNU Affero General Public License restricts these freedoms, requiring "the operator of a network server to provide the source code of [a] modified version [...] to the users of that server" (as summarized informally in the preamble). For example, your rights under this license are terminated if you violate section 13 ("Remote Network Interaction"), including (but not limited to) your right to modify the program.
