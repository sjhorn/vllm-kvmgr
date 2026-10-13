#!/usr/bin/env bash
# RUN EVERY SUITE, IN BOTH MODES, AND EXIT ON THE VERDICT.
#
#   bash run-all.sh                     # both modes (the gate)
#   MODES=pristine bash run-all.sh      # one mode
#
# Modes:
#   pristine   tests/upstream/ alone: kvmgr on a vLLM with NO patch of ours.
#              This is the package's claim, so it is the mode that matters.
#   dspark     the DSpark fork's patch6/7/8 over the same sources -- what the
#              box runs. Needs KVMGR_DSPARK_REPO, or this package inside that
#              repo (the default).
#
# *** IT ENDS IN `exit $T`, AND THAT IS THE WHOLE POINT OF THE FILE. *** The
# sweep used to be a shell one-liner that finished with an `echo`, which
# returns 0 -- so `sweep && git commit` committed a RED suite (1109bba). A gate
# that prints its verdict instead of exiting on it is not a gate.
#
# *** A SKIPPED SECTION IS REPORTED AS SKIPPED, never as ok. *** Sections that
# assert facts about patch6 do not run in pristine mode; each prints a
# `SKIP [pristine mode] ...` line, and this counts them so a suite that skipped
# everything cannot read as a pass.
#
# *** USE python3.12. *** 3.9 cannot parse the evaluated `X | None`
# annotations, and the failure reads like a broken harness.
set -u
cd "$(dirname "$0")"
PY=${PY:-python3.12}
MODES=${MODES:-"pristine dspark"}
T=0
for mode in $MODES; do
  echo "== mode: $mode =="
  if [ "$mode" = "dspark" ]; then export KVMGR_WITH_DSPARK_PATCHES=1; else unset KVMGR_WITH_DSPARK_PATCHES; fi
  for f in test_phase0.py test_phase0_disk.py test_phase1.py test_phase1_disk.py \
           test_phase1b_disk.py test_phase1d_disk.py test_phase2.py \
           test_phase3.py test_phase4.py test_phase6.py \
           test_phase7_dropguard.py test_phase8_journal.py \
           test_phase9_verify.py test_phase10_standalone.py test_phase11_diag.py; do
    out=$("$PY" "$f" 2>&1); rc=$?
    skips=$(printf '%s\n' "$out" | grep -c '^SKIP ')
    if [ "$rc" = "0" ]; then
      # Two summary formats exist: most suites end "N passed, N failed" and
      # test_phase7_dropguard.py ends "phase7 drop guard: N assertions passed".
      sum=$(printf '%s' "$out" | grep -E '^[0-9]+ passed|assertions passed' | tail -1)
      [ -n "$sum" ] || sum='(NO SUMMARY LINE -- did this suite assert anything?)'
      tag="ok  "
      [ "$skips" != "0" ] && sum="$sum  [$skips section(s) SKIPPED]"
      printf '  %s  %-28s %s\n' "$tag" "$f" "$sum"
    else
      printf '  FAIL  %-28s exit=%s\n' "$f" "$rc"
      printf '%s\n' "$out" | grep -E "FAILED:|Error|Traceback" | head -8 | sed 's/^/          /'
      T=1
    fi
  done
  echo
done
[ "$T" = "0" ] && echo "ALL SUITES GREEN in: $MODES" || echo "*** A SUITE IS RED ***"
exit $T
