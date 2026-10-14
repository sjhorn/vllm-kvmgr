# Contributing

Issues and pull requests are welcome. Before opening a pull request:

1. **Run the tests with `python3.12`** and read the exit status, not the last
   line:

   ```sh
   MODES=pristine bash tests/run-all.sh; echo $?
   ```

   No GPU or cluster is needed. The `dspark` mode needs a checkout of the
   DSpark recipe (`KVMGR_DSPARK_REPO`); CI runs `pristine` only. See
   `tests/README.md`.
2. **When you add a guard or a test, break the code once to see it fail.**
3. **If you add or remove a module**, regenerate the dist-info with
   `python3.12 deploy/gen-dist-info.py` (the tests run `--check`).

Some things must not change without a measurement on real hardware, because
they are part of the on-disk format or were found the hard way:

- The names `DSparkDiskOffloadingSpec`, `DSPARK_DISK` and the `dspark-kv-v1`
  root, the on-disk layout (pinned by a hash in the tests; never re-pin it to
  silence a failure), and `DISK_FORMAT_VERSION`. Changing any of them orphans
  every existing disk tier.
- The defaults `load_ops_max=2048` and `load_inflight_max=1`.

kvmgr wraps private vLLM internals (README "What it touches in vLLM"). A port
to another vLLM build starts by re-vendoring `tests/upstream/` from it.

By contributing you agree that your contribution is licensed under the
Apache License 2.0 (see `LICENSE`).
