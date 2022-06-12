<img src="pic/nameplate-2WN29.png" style="width: 16em;">

Expositor is a UCI-conforming chess engine for AMD&thinsp;64 / Intel&thinsp;64 systems. You can [play against her on Lichess](https://lichess.org/@/expositor) or download a copy for local use.

Expositor currently has a [CCRL Blitz](https://www.computerchess.org.uk/ccrl/404/index.html) rating of 3210. You can read about her background on the [TalkChess forums](http://talkchess.com/forum3/viewtopic.php?f=2&t=79407).

## Releases

If you know the microarchitecture of your processor, try using a binary from the appropriate `specific/` directory of a release archive.[^2][^3] If you don't know the microarchitecture of your processor but you do know which features it supports, try using a binary from the appropriate `generic/` directory of a release archive. See the file named `extensions` in this repository for more information about AMD and Intel microarchitectures and the features they support.

The binaries include the default network, so you do not need to download a separate copy.

[^2]: I'm not completely confident that I matched the proper cpu-target for some AMD microarchitectures; please let me know if I've made any mistakes.

[^3]: I've attempted to include binaries for most consumer desktop hardware but not server or mobile platforms. If you have a Xeon or Atom processor, for example, and want a targeted binary, you'll need to compile from source. Feel free to reach out to me for help or if you'd like me to include a binary for your platform in releases.

## Building

If you'd like to compile Expositor from source, you will need to use a recent nightly toolchain.

To build Expositor on Windows, run the `build.bat` script. (This sets the `VERSION`, `BUILD`, and `RUSTFLAGS` environment variables and then invokes `cargo build --release`.)

To build Expositor on Linux, run the `build` script. (This needs to be done from within the repository, since it uses Git to automatically determine the version number.)

## Usage

There are no command line options but the engine can be configured through these UCI options:
```
setoption name Hash value <num>
  Set the size of the transposition (in MiB) to the largest power of two
  less than or equal to <num>.

setoption name Threads value <num>
  Use <num> search threads. Performance will suffer if this is set larger
  than the number of logical cores on your machine, and depending on your
  processor, may suffer if this is set larger than the number of physical
  cores.

setoption name Overhead value <num>
  Set the move overhead (used in time control calculations) to <num>
  milliseconds. "Overhead" refers to the time spent per move on I/O
  operations between the engine and your client or user interface, any
  time spent on network requests that is not corrected by the server
  (if the engine is playing online), and any other latency that uses
  time on the clock. It is important that this is not set to a value
  less than the true overhead, or the engine will have a increased
  risk of flagging.

setoption name Persist value <bool>
  Allow the engine to reuse transposition table entries from previous
  searches. When set to false, singlythreaded searches are deterministic
  and repeatable, regardless of the state of the transposition table.
  When set to true, search results may depend upon previous searches.
  (This is achieved by tagging each table entry with a generation and
  does not incur the penalty of actually zeroing the table.) Setting
  this option to true generally increases playing strength.
```
As well as some nonstandard commands:
```
flip          switch side to move in the current position
eval          print the static evaluation of the current position
load <file>   load a set of neural network weights and begin using them

help          prints this help message
license       prints information about the license
exit          alias for the UCI command "quit"
```
These commands are also available when stderr is a terminal:
```
stat        displays cumulative statistics related to move ordering
trace       displays cumulative statistics related to main search
reset       resets statistics

show        displays a human readable board with the current position
eval        displays NNUE-derived piece values and the static evaluation
clear       clears the terminal display
```
Expositor will automatically detect whether stderr and stdout are connected to a terminal when running on a Linux system, but assumes when running on Windows that neither stderr nor stdout are connected to a terminal. This can, however, be explicitly overridden with the following commands:
```
stderr-isatty <bool>
  Inform the engine that stderr is (or is not) connected to a terminal,
  or to behave as if stderr is (or is not) connected to a terminal.

stdout-isatty <bool>
  Inform the engine that stdout is (or is not) connected to a terminal,
  or to behave as if stdout is (or is not) connected to a terminal.
```
Expositor is lenient when reading moves &ndash; short algebraic notation can be used wherever long algebraic notation is expected. The current position can also be set by entering FEN directly (without being prefaced by `position fen `) or by entering a PGN movelist (movetext without comments or evaluation annotations).

## Issues

If you find any bugs or have any questions, please file an issue on Github or send me a message.

## Versions

| Version | Release Date | CCRL Blitz | Notes                                                  |
|:-------:|:------------:|:----------:|:-------------------------------------------------------|
|  2WN29  |  29 May 2022 |    3210    | Various fixes, better time control, other improvements |
|  2WQ23  |  23 Feb 2022 |    3118    | First public release                                   |

## In Progress

- **HCE Bootstrapping**&ensp;The neural network is currently trained from positions scored with Stockfish. I'd like to write an evaluator that replicates the personality of early versions of Expositor, train a network from positions scored by Expositor using that evaluator, then train another network from positions scored using the previous network, and so on.

## Planned

- **Infrastructure**&ensp;I'd like to write my own automation system (à la OpenBench) that will kick off SPRT testing whenever I push a commit.
- **Tuning Search**&ensp;None of the search constants have been tuned! I expect to be able to wring a fair amount of playing strength out of that.
- **Experimental Network Architectures**&ensp;I'd like to try using different input features and play around with small convolutional networks.
- **Error Tracking Search**&ensp;I've been reading and thinking about this since I started chess programming and at one point it was my primary focus. I'd like to pick it up again, deliver a working proof of concept of my ideas, and then write up my findings.

## Acknowledgments

Four sources are currently used to generate training positions:
- [the Lichess database](https://database.lichess.org/)
- [the Lichess elite database](https://database.nikonoel.fr/)
- [old Ethereal datasets](http://talkchess.com/forum3/viewtopic.php?t=75350)
- [the TCEC games archive](https://github.com/TCEC-Chess/tcecgames)

These are processed by Expositor with her quiescing search and the leaves from those searches are then scored with a gold-standard engine, usually some version of [Stockfish](https://github.com/official-stockfish/Stockfish).

Most of the positions used for perft tests are from the [Zahak](https://github.com/amanjpro/zahak) repository.

Particular thanks to [Jeremy Wright](https://github.com/jtheardw/mantissa) for help with main search. Many techniques were implemented from his descriptions and use parameter values he suggested.

## Terms of Use

Expositor is free and distributed under the terms of version 3 of the GNU Affero General Public License. You are welcome to run the program, modify it, copy it, sell it, or use it in a project of your own.

If you distribute the program, verbatim or modified, you must provide the source and extend to anyone who obtains a copy the same license that I am granting you.

If users can interact with a modified version of the program (or a work based on the program) remotely through a computer network, you must provide a way for users to obtain its source.

For more details, see the file named "license" in this repository.

### Note

Although the GNU Affero General Public License is a libre ("free") license in spirit, it is debatable whether programs licensed under its terms meet [the usual definition of libre software](http://www.gnu.org/philosophy/free-sw.html). Two of the freedoms that users of libre software have are the freedom to run the software as they wish, for any purpose, and the freedom to change it. However, the GNU Affero General Public License restricts these freedoms, requiring "the operator of a network server to provide the source code of [a] modified version [...] to the users of that server" (as summarized informally in the preamble). For example, your rights under this license are terminated if you violate section 13 ("Remote Network Interaction"), including (but not limited to) your right to modify the program.
