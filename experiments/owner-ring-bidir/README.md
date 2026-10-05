# Experimental TP4 bidirectional owner-row relay

This experiment routes opposite-owner blocks clockwise on even ranks and
counterclockwise on odd ranks. It applies on top of the qualified
`tensorfold-glm53:owner-ring-20261005` image. The modified `comm.py` reads
`TF_GLM_OWNER_BIDIR=1` at NCCL communicator construction. With `0` or when
unset, it runs the previous clockwise code path.

On each Spark with that base image:

```bash
docker build -t tensorfold-glm53:owner-ring-bidir-20261005 \
  --build-arg PATCHES_HASH=976a1c53276b \
  -f experiments/owner-ring-bidir/Dockerfile experiments/owner-ring-bidir
```

For a four-rank test, set `IMAGE` to that tag in the head's site file and
`export TF_GLM_OWNER_BIDIR=0` or `1` there so `start.sh` propagates it to
every rank. Keep `SPLIT=1`, `TF_GLM_HC_EXCHANGE=ring`, `COMM=nccl`, and the
qualified ring topology. The production launcher does not automatically
select this image or flag.

The [matched four-Spark A/B](../../docs/bidir-owner-ring-20261005/README.md)
found an isolated communication gain but no end-to-end serving gain, so the
original image remains the recommended deployment.
