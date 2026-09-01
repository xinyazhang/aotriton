#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 3 ]; then
  echo 'Missing arguments. Usage: run-test.sh <pass#> <test_level> <split/fused/aiter/flyc/v3>' >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"

# The dispatch index a backend NAME resolves to, or non-zero if this build has
# no such backend on that operator. $1 is OpAttnFwdBackend / OpAttnBwdBackend,
# $2 the @ati.backend name. Reads PYTHONPATH at call time, so it must be called
# after that is exported.
backend_index_of() {
  python -c "
import torch, pyaotriton
from pyaotriton.v3.flash import $1 as B
print({v: k for k, v in B.by_index.items()}['$2'])
" 2>/dev/null
}
add_torch_ldconfig
add_rocm_sdk_ldconfig

pass=$1
test_level="$2"
backend="$3"
if [ -n "${AOTRITON_TEST_LIBDIR:-}" ]; then
  bdir=""
else
  mapfile -d '' bdir_cans < <(find . -maxdepth 1 -type d -name "build-${aotriton_major}.${aotriton_minor}-test-*${native_arch}*" -print0)
  if [ ${#bdir_cans[@]} -gt 1 ]; then
    echo "There are multiple build directory candidates matching pattern 'build-${aotriton_major}.${aotriton_minor}-test-*${native_arch}*' for testing: ${bdir_cans[@]}. Please keep one only"
    exit 1
  fi
  bdir="${bdir_cans[0]}"
fi

small_vram=$(amd-smi static -g 0 -v --json|grep -v '^WARNING:'| python -c 'import json, sys; j = json.load(sys.stdin); print(int(j["gpu_data"][0]["vram"]["size"]["value"] / 1024.0 < 60))')

# Output directory: use $OUTPUT_DIR if set, otherwise current directory
outdir="${OUTPUT_DIR:-.}"
mkdir -p "$outdir"

# Partial test mode: if PARTIAL_INFO_DIR is set, use the sel file as a pytest selector
SELECT_FROM=""
if [ -n "${PARTIAL_INFO_DIR:-}" ]; then
  src="${PARTIAL_INFO_DIR}/sel${pass}.txt"
  dst="${outdir}/pytest-select-${pass}.txt"
  if [ -f "$src" ]; then
    # Remove "path/to/file.py::" prefix (first occurrence per line only)
    sed 's|[^:]*\.py::||' "$src" > "$dst"
    SELECT_FROM="--select-from-file $dst"
  fi
fi

# Resume mode: skip<pass>.txt names tests to EXCLUDE, which is the shape you want
# after a long pass that did not finish -- list what already passed and re-run
# the rest, rather than enumerating the rest. The two are not the same thing: a
# torn-down session leaves tests that were never dispatched and therefore appear
# in no outcome line at all, so an exclude list covers them and an include list
# cannot without also knowing the full collection.
#
# Passed through UNMODIFIED, unlike sel above: pytest-select matches an entry
# against `item.nodeid` OR `item.name`, so full "path.py::test[params]" ids work
# as-is, and they are what a `grep` over a .out file produces. Prefer them --
# a bare `item.name` can collide between two test files, a nodeid cannot.
#
# Missing entries (a test that passed once and no longer exists) are a warning,
# not an error. Note pytest-select builds that warning by joining every missing
# name into one string, so a skip file written against a different FOR_RELEASE
# can print a very large warning; the run is still correct.
DESELECT_FROM=""
if [ -n "${PARTIAL_INFO_DIR:-}" ]; then
  skipsrc="${PARTIAL_INFO_DIR}/skip${pass}.txt"
  if [ -f "$skipsrc" ]; then
    DESELECT_FROM="--deselect-from-file $skipsrc"
    echo "run-test.sh: excluding $(wc -l < "$skipsrc") test(s) listed in ${skipsrc}"
  fi
fi

if [ -n "${USE_ADIFFS_TXT:-}" ]; then
  if [ -f "$USE_ADIFFS_TXT" ]; then
    echo "USE_ADIFFS_TXT: $USE_ADIFFS_TXT ($(wc -l < "$USE_ADIFFS_TXT") lines)"
  else
    echo "USE_ADIFFS_TXT: $USE_ADIFFS_TXT does not exist, unsetting"
    unset USE_ADIFFS_TXT
  fi
fi

(
  ulimit -c 0
  cd ${SCRIPT_DIR}/..;
  export SMALL_VRAM=${small_vram};
  export COLUMNS=400;
  export FOR_RELEASE=${test_level};
  if [[ "$backend" == "split" ]]; then
    export BWD_IMPL=0
    fnprefix="ut_pass"
  fi
  if [[ "$backend" == "fused" ]]; then
    export V3_API=1
    export BWD_IMPL=1
    fnprefix="fused_pass"
  fi
  if [[ "$backend" == "aiter" ]]; then
    export V3_API=1
    export BWD_IMPL=2
    fnprefix="aiter_pass"
  fi
  if [[ "$backend" == "flyc" ]]; then
    export V3_API=1
    fnprefix="flyc_pass"
  fi
  if [[ "$backend" == "v3" ]]; then
    export V3_API=1
    fnprefix="oput_pass"
  fi
  set -v
  export PYTHONPATH="${AOTRITON_TEST_LIBDIR:-${bdir}/install_dir/lib}"
  # flyc pins BOTH directions, unlike the three above which pin the backward one
  # only. Indices are looked up rather than written down: they are internal
  # numbers that already moved once (flyc taking 2 on op_attn_fwd), whereas
  # 'flyc' is the name @ati.backend declares and the library publishes. Sits here
  # rather than beside the other backends because it needs PYTHONPATH.
  #
  # The backward half is conditional ON THE BUILD, not on a flag here: the flyc
  # backward kernel is being wired now, and this pass should start exercising it
  # the moment it lands rather than waiting for someone to remember this file.
  # Until then SKIP_BWD keeps the pass to the forward half, because otherwise
  # every case fails in .backward() for a reason that says nothing about the
  # forward kernel under test.
  if [[ "$backend" == "flyc" ]]; then
    FWD_IMPL=$(backend_index_of OpAttnFwdBackend flyc) || {
      echo "run-test.sh: this build publishes no 'flyc' forward backend" >&2
      exit 1
    }
    export FWD_IMPL
    if BWD_IMPL=$(backend_index_of OpAttnBwdBackend flyc); then
      export BWD_IMPL
      echo "run-test.sh: flyc FWD_IMPL=${FWD_IMPL} BWD_IMPL=${BWD_IMPL} (forward and backward)"
    else
      export SKIP_BWD=1
      echo "run-test.sh: flyc FWD_IMPL=${FWD_IMPL}; no flyc backward backend in this build, SKIP_BWD=1"
    fi
  fi
  _sig=$(ls "$PYTHONPATH/aotriton.images/"*"/__signature__" 2>/dev/null | head -n 1)
  {
    [ -n "$_sig" ] && cat "$_sig" \
      || echo "NO __signature__ file at $PYTHONPATH/aotriton.images/"
  } > "${outdir}/${fnprefix}${pass}.out"
  # One invocation over the whole suite dir (conftest.py sets up sys.path); pytest
  # collects test_backward / test_varlen together (test_forward.py is excluded via
  # conftest.py's collect_ignore - its coverage is a subset of test_backward.py's).
  pytest --tb=line -n ${ngpus} --max-worker-restart 9999 -rfEsx \
    --timeout=300 \
    -p no:cacheprovider \
    ${SELECT_FROM} \
    ${DESELECT_FROM} \
    modules/flash/tests \
    -v \
    1>>"${outdir}/${fnprefix}${pass}.out" \
    2>"${outdir}/${fnprefix}${pass}.err" || true
  _out="${outdir}/${fnprefix}${pass}.out"
  # Per-test ids by outcome, read from pytest's VERBOSE stream
  # (`[gw12] [ 43%] PASSED <nodeid>`), not from the short summary at the end.
  #
  # That distinction is the whole point. A session that xdist tears down --
  # which it does when two workers crash close enough together, and GPU faults
  # make that a matter of time on a run this long -- never prints a summary. Of
  # the four Level-3 passes so far, three left a 0-byte sel<N>.txt for exactly
  # that reason, on the runs that most needed resuming.
  _ids_by_outcome() {  # $1: alternation, e.g. 'PASSED|SKIPPED'
    sed -nE "s/^.*\] ($1) ([^ ].*[^ ])[[:space:]]*$/\2/p" "$_out" | sort -u
  }

  # sel: what to RE-RUN. Summary first, so a completed run's file is byte-for-byte
  # what it always was; the verbose stream only as a fallback, which is where a
  # torn-down run gets a usable file instead of an empty one.
  grep '^FAILED' "$_out" | sed 's/^FAILED //' | sed 's/].*/]/' > "${outdir}/sel${pass}.txt"
  if [ ! -s "${outdir}/sel${pass}.txt" ]; then
    _ids_by_outcome 'FAILED|ERROR' > "${outdir}/sel${pass}.txt"
    if [ -s "${outdir}/sel${pass}.txt" ]; then
      echo "run-test.sh: no short summary (session torn down?); recovered" \
           "$(wc -l < "${outdir}/sel${pass}.txt") failure(s) from the verbose stream"
    fi
  fi

  # skip: what NOT to re-run. Feed it back as PARTIAL_INFO_DIR/skip<N>.txt and the
  # next pass covers the failures AND everything the torn-down session never
  # dispatched -- which an include list cannot do, since a test that never ran
  # appears in no outcome line at all.
  #
  # SKIPPED joins PASSED because a skip here is a deterministic property of the
  # parameter set, not a result that could differ next time; re-running them is
  # tens of thousands of instant no-ops for no information.
  _ids_by_outcome 'PASSED|SKIPPED' > "${outdir}/skip${pass}.txt"
  echo "run-test.sh: skip${pass}.txt has $(wc -l < "${outdir}/skip${pass}.txt") settled" \
       "test(s); PARTIAL_INFO_DIR=${outdir} bash .ci/run-test.sh <next> ${test_level} ${backend} resumes"
  if [ -n "${RECORD_ADIFFS_TO:-}" ]; then
    SCRIPT_DIR_ABS="$(cd "${SCRIPT_DIR}" && pwd)"
    bash "${SCRIPT_DIR_ABS}/../.tune/bin/append_oom_to_adiffs.sh" "${outdir}/${fnprefix}${pass}.out" >> "${RECORD_ADIFFS_TO}"
  fi
)
