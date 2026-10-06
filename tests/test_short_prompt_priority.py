"""Run inside a TensorFold image with patch 0072 installed."""

from types import SimpleNamespace as NS

from tensorfold.families.glm5_next.cuda.multi import MultiDecoder, short_prompt_rows


def lane(sid: int, rows: int):
    return NS(
        sid=sid, order=sid, decoding=False, paused=False, feed=None, head=None,
        stops=[], st=NS(pos=0), s=NS(sid=sid, prompt=[1] * rows, background=False, done=False),
    )


def decoder(rows: list[int], threshold: int):
    d = object.__new__(MultiDecoder)
    d.lanes = {i: lane(i, n) for i, n in enumerate(rows)}
    d.partial = None
    d.fill_budget = 200
    d.decode_due = False
    d.short_prompt_rows = threshold
    d.fill_rows = 2048
    d.unit = 64
    d.e = NS(prefill_rows=8192)
    return d


assert short_prompt_rows(8192, "0") == 0
assert short_prompt_rows(8192, "512") == 512
for invalid in ("-1", "8193", "oops"):
    try:
        short_prompt_rows(8192, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {invalid!r}")

mixed = decoder([32768, 128], 512)
first = mixed._fill_turn()
assert first.sid == 1, "a later short prompt should go first"
assert mixed._group(first) is None, "it should not wait for the long prompt's grouped forward"

shorts = decoder([128, 130, 132, 134], 512)
first = shorts._fill_turn()
assert first.sid == 0
assert len(shorts._group(first)) == 4, "short requests should still share a prompt chunk"

disabled = decoder([32768, 128], 0)
assert disabled._fill_turn().sid == 0, "the default must preserve FIFO"
assert len(disabled._group(disabled._fill_turn())) == 2
print("short-prompt scheduling tests passed")
