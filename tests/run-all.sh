#!/usr/bin/env bash
# RUN EVERY HARNESS SUITE AND EXIT ON THE VERDICT.
#
# *** IT ENDS IN `exit $T`, AND THAT IS THE WHOLE POINT OF THE FILE. *** The
# sweep used to be a shell one-liner that finished with an `echo`, which
# returns 0 -- so `sweep && git commit` committed a RED suite (1109bba). Same
# shape as this repo's `curl | python3 && deploy` and `gated-deploy.sh | tail`
# traps: a gate that prints its verdict instead of exiting on it is not a gate.
#
# *** USE python3.12. *** 3.9 cannot parse the patches' evaluated `X | None`
# annotations, and the failure reads like a broken harness.
set -u
cd "$(dirname "$0")"
PY=${PY:-python3.12}
T=0
for f in test_phase0.py test_phase0_disk.py test_phase1.py test_phase1_disk.py \
         test_phase1b_disk.py test_phase1d_disk.py test_phase2.py \
         test_phase3.py test_phase4.py test_phase6.py \
         test_phase7_dropguard.py test_phase8_journal.py \
         test_phase9_verify.py; do
  out=$("$PY" "$f" 2>&1); rc=$?
  # The suites do not agree on a summary line, so report the EXIT STATUS and
  # print the tail only when it is non-zero -- a sweep whose output has to be
  # read to find the verdict gets `| tail`ed, and a 2026-10-02 run lost its one
  # failing check exactly that way.
  if [ "$rc" = "0" ]; then
    # Two summary formats exist: most suites end "N passed, N failed" and
    # test_phase7_dropguard.py ends "phase7 drop guard: N assertions passed".
    # Matching only the first left phase7's cell BLANK, which reads as "no
    # assertions ran" beside an `ok` -- absent rendered as zero, in the sweep
    # whose whole job is saying whether anything was checked. The suite was
    # always sound (it exits 1 on the first failure); the SUMMARY was lying.
    sum=$(printf '%s' "$out" | grep -E '^[0-9]+ passed|assertions passed' | tail -1)
    [ -n "$sum" ] || sum='(NO SUMMARY LINE -- did this suite assert anything?)'
    printf '  ok    %-26s %s\n' "$f" "$sum"
  else
    printf '  FAIL  %-26s exit=%s\n' "$f" "$rc"
    printf '%s\n' "$out" | grep -E "FAILED:|Error|Traceback" | head -8 | sed 's/^/          /'
    T=1
  fi
done
echo
[ "$T" = "0" ] && echo "ALL SUITES GREEN" || echo "*** A SUITE IS RED ***"
exit $T
