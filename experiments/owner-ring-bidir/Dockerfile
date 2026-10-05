# Build on every Spark from the qualified owner-ring image. The original tag
# remains available; the candidate flag defaults to the original route.
ARG BASE_IMAGE=tensorfold-glm53:owner-ring-20261005
FROM ${BASE_IMAGE}
COPY 0072-glm-tp4-owner-ring-bidir.patch /tmp/0072-glm-tp4-owner-ring-bidir.patch
RUN cd /usr/local/lib/python3.12/dist-packages && \
    patch -p0 --forward < /tmp/0072-glm-tp4-owner-ring-bidir.patch && \
    python3 -c 'from tensorfold.cuda.comm import NCCL; assert hasattr(NCCL, "exchange_owner_ring")'
ARG PATCHES_HASH
LABEL tf.owner_bidir=${PATCHES_HASH}
