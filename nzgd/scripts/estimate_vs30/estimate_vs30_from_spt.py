"""Calculate checkpointed SPT Vs30 estimates; publish only by explicit command."""

from nzgd.scripts.estimate_vs30 import batch

if __name__ == "__main__":
    batch.main(default_kind="spt")
