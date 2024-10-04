#include "tbprobe.h"

unsigned tb_probe_wdl(
    uint64_t _white,
    uint64_t _black,
    uint64_t _kings,
    uint64_t _queens,
    uint64_t _rooks,
    uint64_t _bishops,
    uint64_t _knights,
    uint64_t _pawns,
    unsigned _rule50,
    unsigned _castling,
    unsigned _ep,
    bool     _turn)
{
    return _tb_probe_wdl(
        _white,
        _black,
        _kings,
        _queens,
        _rooks,
        _bishops,
        _knights,
        _pawns,
        _rule50,
        _castling,
        _ep,
        _turn
    );
}

unsigned tb_probe_root(
    uint64_t _white,
    uint64_t _black,
    uint64_t _kings,
    uint64_t _queens,
    uint64_t _rooks,
    uint64_t _bishops,
    uint64_t _knights,
    uint64_t _pawns,
    unsigned _rule50,
    unsigned _castling,
    unsigned _ep,
    bool     _turn)
{
    return _tb_probe_root(
        _white,
        _black,
        _kings,
        _queens,
        _rooks,
        _bishops,
        _knights,
        _pawns,
        _rule50,
        _castling,
        _ep,
        _turn,
        0
    );
}
