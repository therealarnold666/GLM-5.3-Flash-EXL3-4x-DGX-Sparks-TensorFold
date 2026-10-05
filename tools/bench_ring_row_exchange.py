"""Four-rank NCCL microbenchmark for GLM-5.3-Flash SPLIT=1 prompt rows.

Run one process per Spark with the serving image, patched NCCL and identical
NCCL ring environment. It does not load weights or start the API.
"""

import argparse
import ctypes
import json
import statistics
import time

import torch

from tensorfold.cuda.comm import NCCL, _DTYPES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--master", default="192.0.2.10")
    parser.add_argument("--port", type=int, default=29673)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reverse", action="store_true", help="measure the modes in reverse order")
    args = parser.parse_args()
    rank = args.rank
    world = 4
    assert args.rows % world == 0
    share = args.rows // world
    prev, nxt, opp = (rank - 1) % world, (rank + 1) % world, (rank + 2) % world
    nccl = NCCL(rank, world, args.master, args.port)
    part = torch.empty((world, share, args.width), dtype=torch.float32, device="cuda")
    for dest in range(world):
        part[dest].fill_(rank * 10 + dest)
    full = torch.empty((world, world, share, args.width), dtype=torch.float32, device="cuda")
    from_prev = torch.empty_like(part[0])
    from_next = torch.empty_like(part[0])
    forward_from_prev = torch.empty_like(part[0])
    from_opp = torch.empty_like(part[0])
    bidir_forward = torch.empty_like(part[0])
    bidir_from_opp = torch.empty_like(part[0])
    lib = nccl.lib
    stream = torch.cuda.current_stream().cuda_stream
    datatype = _DTYPES[torch.float32]
    elements = share * args.width

    def send(tensor: torch.Tensor, peer: int) -> None:
        nccl._check(lib.ncclSend(tensor.data_ptr(), elements, datatype, peer, nccl.comm, stream))

    def recv(tensor: torch.Tensor, peer: int) -> None:
        nccl._check(lib.ncclRecv(tensor.data_ptr(), elements, datatype, peer, nccl.comm, stream))

    def group(actions) -> None:
        nccl._check(lib.ncclGroupStart())
        try:
            for fn, tensor, peer in actions:
                fn(tensor, peer)
        finally:
            nccl._check(lib.ncclGroupEnd())

    def gather() -> None:
        nccl.all_gather(part.reshape(-1), full.reshape(-1))

    def owner_ring() -> None:
        # First hop: direct destination blocks go to both neighbors; the
        # opposite destination block travels clockwise through the next rank.
        group(((send, part[nxt], nxt), (send, part[opp], nxt),
               (recv, from_prev, prev), (recv, forward_from_prev, prev),
               (send, part[prev], prev), (recv, from_next, nxt)))
        # Second hop: forward the opposite block received from the previous
        # rank. Every send/receive here touches a physically adjacent rank.
        group(((send, forward_from_prev, nxt), (recv, from_opp, prev)))

    def owner_ring_bidirectional() -> None:
        # Even ranks route the opposite block clockwise; odd ranks route it
        # counterclockwise. Each rank then sends two blocks over each edge.
        relay = nxt if rank % 2 == 0 else prev
        other = prev if rank % 2 == 0 else nxt
        group(((send, part[nxt], nxt), (send, part[prev], prev),
               (send, part[opp], relay), (recv, from_prev, prev),
               (recv, from_next, nxt), (recv, bidir_forward, relay)))
        group(((send, bidir_forward, other), (recv, bidir_from_opp, other)))

    torch.cuda.synchronize()
    gather()
    owner_ring()
    owner_ring_bidirectional()
    torch.cuda.synchronize()
    for origin in range(world):
        assert full[origin, rank, 0, 0].item() == origin * 10 + rank
        assert full[origin, rank, -1, -1].item() == origin * 10 + rank
    for origin, tensor in ((prev, from_prev), (nxt, from_next), (opp, from_opp)):
        assert tensor[0, 0].item() == origin * 10 + rank
        assert tensor[-1, -1].item() == origin * 10 + rank
    for origin, tensor in ((prev, from_prev), (nxt, from_next), (opp, bidir_from_opp)):
        assert torch.equal(tensor, full[origin, rank])
    nccl.barrier()

    def measure(label: str, fn) -> dict:
        samples = []
        for repeat in range(args.repeats):
            fn()
            torch.cuda.synchronize()
            nccl.barrier()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start_wall = time.monotonic()
            begin.record()
            for _ in range(args.iterations):
                fn()
            end.record()
            end.synchronize()
            wall = time.monotonic() - start_wall
            samples.append({"cuda_ms": begin.elapsed_time(end), "wall_ms": wall * 1000})
            nccl.barrier()
        return {"label": label, "samples": samples,
                "median_cuda_ms_per_exchange": statistics.median(s["cuda_ms"] for s in samples) / args.iterations,
                "median_wall_ms_per_exchange": statistics.median(s["wall_ms"] for s in samples) / args.iterations}

    modes = [("full_fp32_all_gather", gather), ("owner_rows_neighbor_forward", owner_ring),
             ("owner_rows_bidirectional", owner_ring_bidirectional)]
    if args.reverse:
        modes.reverse()
    output = {"rank": rank, "rows": args.rows, "width": args.width, "world": world,
              "fp32_partial_mib": part.numel() * part.element_size() / 1048576,
              "owner_block_mib": part[0].numel() * part[0].element_size() / 1048576,
              "iterations": args.iterations, "repeats": args.repeats, "reverse": args.reverse,
              "correctness": "passed",
              "results": [measure(name, fn) for name, fn in modes]}
    print("ROWBENCH_RESULT=" + json.dumps(output, sort_keys=True), flush=True)
    nccl.barrier()


if __name__ == "__main__":
    main()
