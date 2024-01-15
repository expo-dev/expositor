## List of AMD&thinsp;64 and Intel&thinsp;64 Extensions

| Extension |    Year   | Description                                                                                                                                                     |
|:---------:|:---------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    MMX    |    1997   | reuses floating point registers for mmx0–mmx7 registers (64 bits wide)                                                                                          |
|    SSE    |    1999   | adds the xmm0–xmm15 registers (128 bits wide)                                                                                                                   |
|    SSE2   | 2000–2003 | effectively replaces the MMX extension                                                                                                                          |
|    SSE3   | 2004–2005 | adds a few instructions (mostly horizontal operations)                                                                                                          |
|   SSE4.2  |    2007   | adds many instructions, adds the non-SIMD popcnt instruction (see below)                                                                                        |
|    AVX    |    2011   | extends register width to 256 bits (ymm0–ymm15 registers), adds the VEX coding scheme, adds instructions (e.g. vzeroupper and the memory version of vbroadcast) |
|    BMI    | 2012–2013 | also known as BMI1, adds non-SIMD instructions (e.g. tzcnt and blsr)                                                                                            |
|    BMI2   |    2013   | adds non-SIMD instructions (e.g. pdep and pext)                                                                                                                 |
|    AVX2   |    2013   | adds instructions (e.g. the register version of vbroadcast)                                                                                                     |
|  AVX-512  |    2017   | extends register width to 512 bits (zmm0-zmm15 registers)                                                                                                       |

Omitted from this list are, among others: 3DNow!; SSSE3; distinctions between
SSE4, SSE4.1, and SSE4.2; SSE4a, which is not supported by Intel; SSE5; TBM =
_trailing bit manipulation,_ which was not supported by Intel and is no longer
supported by AMD; ADX, which is Intel’s arbitrary-precision arithmetic
extension; and FMA = _fused multiply-add._


## Popcnt and Lzcnt

Intel considers popcnt part of SSE4.2 and lzcnt part of BMI. Intel has
supported popcnt since Nehalem (ca 2007) and lzcnt since Haswell (ca 2013).

AMD’s ABM instruction set was introduced alongside the SSE4a instruction set
(ca 2007). ABM is only implemented by AMD in its entirety: all AMD processors
support both popcnt and lzcnt or neither.

Support for popcnt is indicated by its own cpuid flag. Intel uses AMD’s
flag for ABM to indicate support of lzcnt (since lzcnt completes ABM).


## List of Intel Microarchitectures

| Year | Gen | μarch        | Step         | SIMD Extensions          | Non-SIMD Extensions |
|:----:|:---:|:-------------|:-------------|:-------------------------|---------------------|
| 2000 |     | NetBurst     | Willamette   | MMX, SSE, SSE2           |                     |
| 2004 |     | &nbsp;. . .  | Prescott     | MMX–SSE2 + SSE3, SSSE3   |                     |
| 2006 |     | Core         |              | MMX–SSSE3                |                     |
| 2007 |     | &nbsp;. . .  | Penryn       | MMX–SSSE3 + SSE4, SSE4.1 |                     |
| 2008 |  1  | Nehalem      | Nehalem      | MMX–SSE4.1 + SSE4.2      | SSE4.2¹             |
| 2010 |  1  | &nbsp;. . .  | Westmere     | MMX–SSE4.2               | SSE4.2              |
| 2011 |  2  | Sandy Bridge | Sandy Bridge | MMX–SSE4.2 + AVX         | SSE4.2              |
| 2012 |  3  | &nbsp;. . .  | Ivy Bridge   | MMX–AVX                  | SSE4.2              |
| 2013 |  4  | Haswell      | Haswell      | MMX–AVX + AVX2           | SSE4.2 + BMI, BMI2  |
| 2014 |  5  | &nbsp;. . .  | Broadwell    | MMX–AVX2                 | SSE4.2–BMI2         |
| 2015 |  6  | Skylake      | Skylake      | MMX–AVX2                 | SSE4.2–BMI2         |
| 2016 |  7  | &nbsp;. . .  | Kaby Lake    | MMX–AVX2                 | SSE4.2–BMI2         |
| 2017 |  7  | &nbsp;. . .  | Skylake-X    | MMX–AVX2 + AVX-512       | SSE4.2–BMI2         |
| 2017 |  8  | &nbsp;. . .  | Coffee Lake  | MMX–AVX2                 | SSE4.2–BMI2         |
| 2017 |  8  | &nbsp;. . .  | Kaby Lake²   | MMX–AVX2                 | SSE4.2–BMI2         |
| 2019 |  9  | &nbsp;. . .  | Coffee Lake² | MMX–AVX2                 | SSE4.2–BMI2         |
| 2019 | 10  | &nbsp;. . .  | Cascade Lake | MMX–AVX2, AVX-512        | SSE4.2–BMI2         |
| 2019 | 10  | &nbsp;. . .  | Comet Lake   | MMX–AVX2                 | SSE4.2–BMI2         |
| 2019 | 10  | Sunny Cove   | Ice Lake     | MMX–AVX2, AVX-512        | SSE4.2–BMI2         |
| 2021 | 11  | Cypress Cove | Rocket Lake³ | MMX–AVX2, AVX-512        | SSE4.2–BMI2         |
| 2021 | 12  | Golden Cove  | Alder Lake   | MMX–AVX2                 | SSE4.2–BMI2         |

¹ See the section on popcnt above \
² Kaby Lake or Coffee Lake refresh \
³ Sunny Cove backport to 14nm

Note that the year given is the year a model was first released from the family;
products of each generation may be introduced or sold for several years after
that (e.g. some Skylake-X models were introduced in 2019).

Note that some of the years may be off by one, since it was not always clear
from sources whether the year listed referred to announcement, launch date or
release date, or when products became available to customers or to consumers.


### List of AMD Microarchitectures

| Year | Fam | μarch       | SIMD Extensions | Non-SIMD Extensions |
|:----:|:---:|:------------|:----------------|:--------------------|
| 2003 | 0Fh | K8          | MMX–SSE2        |                     |
| 2004 | 0Fh | K8          | MMX–SSE2 + SSE3 |                     |
| 2007 | 10h | K10         | MMX–SSE3¹       | ABM                 |
| 2011 | 15h | Bulldozer   | MMX–SSE3 + AVX  | ABM                 |
| 2012 | 15h | Piledriver  | MMX–AVX         | ABM + BMI           |
| 2014 | 15h | Steamroller | MMX–AVX         | ABM, BMI            |
| 2015 | 15h | Excavator   | MMX–AVX + AVX2  | ABM, BMI + BMI2     |
| 2017 | 17h | Zen         | MMX–AVX2        | ABM–BMI2            |
| 2018 | 17h | Zen+        | MMX–AVX2        | ABM–BMI2            |
| 2019 | 17h | Zen 2       | MMX–AVX2        | ABM–BMI2            |
| 2020 | 19h | Zen 3       | MMX–AVX2        | ABM–BMI2            |

¹ Actually up to SSE4a

Note that, in architectures before Zen 3, the pdep and pext instructions (from
the BMI2 extension) are implemented in microcode and have high latency.
