<img src="pic/nameplate-2BR17.png" style="width: 16em;">

_For the version currently playing in the Top Chess Engine Championship, see the [`tcec`](https://github.com/expo-dev/expositor/tree/tcec) branch._

Expositor is a UCI-conforming chess engine for AMD&thinsp;64 / Intel&thinsp;64 systems. You can [play against her on Lichess](https://lichess.org/@/expositor) or download a copy for local use.

Expositor currently has a [CCRL Blitz](https://www.computerchess.org.uk/ccrl/404) rating of about 3300 running singlythreaded (about #55) and a [CCRL 40/15](https://www.computerchess.org.uk/ccrl/4040) rating of about 3310 running with four threads (about #40). You can read about her background on the [TalkChess forums](http://talkchess.com/forum3/viewtopic.php?f=2&t=79407).

The name _Expositor_ comes from _exponent_ &ndash; the root of both is [_expōnēns_](https://en.wiktionary.org/wiki/exponent#Etymology). An exponent is one part of a floating-point number (along with a sign, significand, and radix). My friend and I started chess programming at the same time; his engine is named _Mantissa,_ which is another name for a significand. The name _Expositor_ also contains the word _posit,_ which is the name for an interesting and rather lovely [alternative format](https://posithub.org/docs/posit_standard-2.pdf) to the usual IEEE 754.

## Status

It's been many months since the last release and, aside from some time management adjustments for TCEC, quite a while since I pushed any commits to the public repository, so I want to say a word about the status of the project.

- The project hasn't been abandoned. Bootstrapping is still my first priority and, although I've taken some breaks, I've continued to work on the HCE.

- For a little while, I was instead working on a better [specification of the Universal Chess Interface](https://expositor.dev/uci-2022-12-29.pdf). Several people were supportive (and I'm very grateful for that &ndash; thank you!) but there were also some very discouraging responses, enough so that I put it aside. Hopefully, with my stamina recovered, I'll eventually pick this up again and publish a second draft. If you have any comments on the first draft, I'd love to hear them; you can contact me at `expositor` at `fastmail.com`.

- I've relented and decided to include Syzygy support through the use of Fathom, since that's a valuable feature and I won't get around to writing my own implementation as soon as I had hoped.

- Some of the core classes have been rewritten and a few things have been cleaned up, and I've made several improvements to the tooling I use for developing Expositor.

- There are a handful of other, small features that I've worked out, such as support for huge pages and setting processor affinity.

I expect it will be several months more before the next major release of Expositor: a few months to finish writing the HCE and a few months to then train two or three iterations of the NNUE. There are particular goals and constraints that I have for the HCE and, since I expect that I won't ever touch the HCE again after I've finished it, I want to get it right. I've started over multiple times and thrown away most of what I've written, and I've lost count of the number of versions there have been, but I do feel I've been making progress. It's unclear whether it will ultimately resemble traditional HCEs in any meaningful way; there's a decent chance I will end up calling it something else to communicate that fact.

There may be a minor release of Expositor next month. I do not expect it to be stronger than 2BR17 at standard chess, but it will be significantly stronger at [chess 324](https://talkchess.com/forum3/viewtopic.php?f=2&t=80482) &ndash; the most recent network I trained performed 100 Elo stronger at bullet time controls.

As ever, stay safe, and thank you for your interest in Expositor!

&ndash; the Expositor dev<br>&ensp;&ensp;15 May 2023

## Releases

If you know the microarchitecture of your processor, try using a binary from the appropriate `specific/` directory of a release archive.[^1][^2] If you don't know the microarchitecture of your processor but you do know which features it supports, try using a binary from the appropriate `generic/` directory of a release archive. See the file named `extensions` in this repository for more information about AMD and Intel microarchitectures and the features they support.

The binaries include the default network, so you do not need to download a separate copy.

[^1]: I'm not completely confident that I matched the proper cpu-target for some AMD microarchitectures; please let me know if I've made any mistakes.

[^2]: I've attempted to include binaries for most consumer desktop hardware but not server or mobile platforms. If you have a Xeon or Atom processor, for example, and want a targeted binary, you'll need to compile from source. Feel free to reach out to me for help or if you'd like me to include a binary for your platform in releases.

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

isatty <bool>
  Shorthand to set both stderr-isatty and stdout-isatty.
```
Expositor is lenient when reading moves &ndash; short algebraic notation can be used wherever long algebraic notation is expected. The current position can also be set by entering FEN directly (without being prefaced by `position fen `) or by entering a PGN movelist (movetext without comments or evaluation annotations).

## Issues

If you find any bugs or have any questions, please file an issue on Github or send a message to `expositor` at `fastmail.com`.

## Versions

| Version | Release Date | CCRL Blitz |    CCRL 40/15     | Notes                                         |
|:-------:|:------------:|:----------:|:-----------------:|:----------------------------------------------|
|  2BR17  |  17 Sep 2022 |    3304    | 3211 &ndash; 3312 | fixes, tuned search, internal 3-man tablebase |
|  2WN29  |  29 May 2022 |    3212    | 3148 &ndash; 3241 | fixes, better time control, cache persistence |
|  2WQ23  |  23 Feb 2022 |    3142    | 3077 &ndash; unkn | first public release                          |

The two ratings listed under _CCRL 40/15_ in each row are for 1- and 4-thread performance.

## In Progress

- **HCE Bootstrapping**&ensp;The neural network is currently trained from positions scored with Stockfish. I'm nearly done writing and training an evaluator with a personality that is true to her existing versions, past and present; then I'll train a network from positions scored by Expositor using that evaluator, then train another network from positions scored using the previous network, and so on.

## Planned

- **Experimental Network Architectures**&ensp;I'd like to try using different input features, play around with small convolutional networks, and test novel evaluator ideas.

- **Error Distribution Search**&ensp;I've been reading and thinking about this since I started chess programming and at one point it was my primary focus. I'd like to pick it up again, deliver a working proof of concept, and then write up my findings.

- **Syzygy Tablebase Support**&ensp;I'd like to write my own implementation of Syzygy tablebase support (so that Expositor continues to have no dependencies and includes no external code). The `tcec` branch currently has Syzygy tablebase support, however, by means of the Fathom library.

## Acknowledgments

Four sources were used to generate training positions:
- [the Lichess database](https://database.lichess.org/)
- [the Lichess elite database](https://database.nikonoel.fr/)
- [old Ethereal datasets](http://talkchess.com/forum3/viewtopic.php?t=75350)
- [the TCEC games archive](https://github.com/TCEC-Chess/tcecgames)

These were processed by Expositor with her quiescing search and the leaves from those searches were then scored with a gold-standard engine, usually a version of [Stockfish](https://github.com/official-stockfish/Stockfish). Those scored leaves were then filtered and the network was trained on the selected positions. (Note that the network and training code is original &ndash; network weights for other engines are not compatible with Expositor.)

Most of the positions used for perft tests were from the [Zahak](https://github.com/amanjpro/zahak) repository.

Particular thanks to [Jeremy Wright](https://github.com/jtheardw/mantissa) for help with main search. Many techniques were implemented from his verbal descriptions during our conversations.

## Terms of Use

Expositor is free and distributed under the terms of version 3 of the GNU Affero General Public License. You are welcome to run the program, modify it, copy it, sell it, or use it in a project of your own.

If you distribute the program, verbatim or modified, you must provide the source and extend to anyone who obtains a copy the same license that I am granting you.

If users can interact with a modified version of the program (or a work based on the program) remotely through a computer network, you must provide a way for users to obtain its source.

For more details, see the file named "license" in this repository.
