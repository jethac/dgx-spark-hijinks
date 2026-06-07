#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/pull_container_with_evidence.sh IMAGE RUN_ID

Pull a container image and capture enough evidence to distinguish registry,
platform, Docker daemon, and import/extraction failures.

Environment:
  RESULTS_DIR=results          artifact output directory
  PLATFORM=linux/arm64         docker platform and skopeo override target
  PULL_TIMEOUT=0               seconds; 0 means no timeout wrapper
  USE_SKOPEO=0                 set to 1 to try skopeo OCI copy/import if Docker pull fails
  INSTALL_SKOPEO=0             set to 1 to install skopeo with apt-get if missing
  OCI_DIR=/var/tmp/RUN_ID_oci  local OCI layout path for skopeo fallback
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

IMAGE=$1
RUN_ID=$2
RESULTS_DIR=${RESULTS_DIR:-results}
PLATFORM=${PLATFORM:-linux/arm64}
PULL_TIMEOUT=${PULL_TIMEOUT:-0}
USE_SKOPEO=${USE_SKOPEO:-0}
INSTALL_SKOPEO=${INSTALL_SKOPEO:-0}
OCI_DIR=${OCI_DIR:-/var/tmp/${RUN_ID}_oci}

mkdir -p "${RESULTS_DIR}"

SUMMARY="${RESULTS_DIR}/${RUN_ID}_image_pull_summary.md"
MANIFEST_JSON="${RESULTS_DIR}/${RUN_ID}_manifest.json"
MANIFEST_VERBOSE="${RESULTS_DIR}/${RUN_ID}_manifest_verbose.txt"
DOCKER_PULL_LOG="${RESULTS_DIR}/${RUN_ID}_docker_pull.log"
DOCKER_INSPECT_JSON="${RESULTS_DIR}/${RUN_ID}_image_inspect.json"
DOCKER_DF_BEFORE="${RESULTS_DIR}/${RUN_ID}_docker_system_df_before.txt"
DOCKER_DF_AFTER="${RESULTS_DIR}/${RUN_ID}_docker_system_df_after.txt"
FS_BEFORE="${RESULTS_DIR}/${RUN_ID}_filesystem_before.txt"
FS_AFTER="${RESULTS_DIR}/${RUN_ID}_filesystem_after.txt"
DAEMON_LOG="${RESULTS_DIR}/${RUN_ID}_docker_daemon_recent.log"
SKOPEO_COPY_LOG="${RESULTS_DIR}/${RUN_ID}_skopeo_copy.log"
SKOPEO_IMPORT_LOG="${RESULTS_DIR}/${RUN_ID}_skopeo_import.log"
SKOPEO_INSTALL_LOG="${RESULTS_DIR}/${RUN_ID}_skopeo_install.log"

started_iso=$(date -Is)
started_epoch=$(date +%s)

run_capture() {
  local outfile=$1
  shift
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } >"${outfile}" 2>&1
}

run_append() {
  local outfile=$1
  shift
  {
    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } >>"${outfile}" 2>&1
}

platform_os=${PLATFORM%%/*}
platform_arch=${PLATFORM##*/}

run_capture "${DOCKER_DF_BEFORE}" docker system df || true
run_capture "${FS_BEFORE}" df -h /var/lib/docker /var/tmp /tmp || true
run_capture "${MANIFEST_VERBOSE}" docker manifest inspect --verbose "${IMAGE}" || true
run_capture "${MANIFEST_JSON}" docker manifest inspect "${IMAGE}" || true

pull_rc=0
if [[ "${PULL_TIMEOUT}" == "0" ]]; then
  run_capture "${DOCKER_PULL_LOG}" docker pull --platform "${PLATFORM}" "${IMAGE}" || pull_rc=$?
else
  run_capture "${DOCKER_PULL_LOG}" timeout "${PULL_TIMEOUT}" docker pull --platform "${PLATFORM}" "${IMAGE}" || pull_rc=$?
fi

inspect_rc=0
run_capture "${DOCKER_INSPECT_JSON}" docker image inspect "${IMAGE}" || inspect_rc=$?

skopeo_copy_rc=
skopeo_import_rc=
if [[ "${inspect_rc}" != "0" && "${USE_SKOPEO}" == "1" ]]; then
  if ! command -v skopeo >/dev/null 2>&1; then
    if [[ "${INSTALL_SKOPEO}" == "1" ]]; then
      if [[ "$(id -u)" == "0" ]]; then
        run_capture "${SKOPEO_INSTALL_LOG}" apt-get update || true
        run_append "${SKOPEO_INSTALL_LOG}" apt-get install -y skopeo || true
      elif command -v sudo >/dev/null 2>&1; then
        run_capture "${SKOPEO_INSTALL_LOG}" sudo apt-get update || true
        run_append "${SKOPEO_INSTALL_LOG}" sudo apt-get install -y skopeo || true
      else
        printf 'skopeo missing and sudo unavailable\n' >"${SKOPEO_INSTALL_LOG}"
      fi
    else
      printf 'skopeo missing; set INSTALL_SKOPEO=1 to install it\n' >"${SKOPEO_INSTALL_LOG}"
    fi
  fi

  if command -v skopeo >/dev/null 2>&1; then
    skopeo_copy_rc=0
    run_capture "${SKOPEO_COPY_LOG}" \
      skopeo copy \
      --override-os "${platform_os}" \
      --override-arch "${platform_arch}" \
      "docker://${IMAGE}" \
      "oci:${OCI_DIR}:image" || skopeo_copy_rc=$?

    if [[ "${skopeo_copy_rc}" == "0" ]]; then
      skopeo_import_rc=0
      run_capture "${SKOPEO_IMPORT_LOG}" \
        skopeo copy \
        "oci:${OCI_DIR}:image" \
        "docker-daemon:${IMAGE}" || skopeo_import_rc=$?
      run_capture "${DOCKER_INSPECT_JSON}" docker image inspect "${IMAGE}" || inspect_rc=$?
    fi
  fi
fi

run_capture "${DOCKER_DF_AFTER}" docker system df || true
run_capture "${FS_AFTER}" df -h /var/lib/docker /var/tmp /tmp || true
run_capture "${DAEMON_LOG}" journalctl -u docker --since "@${started_epoch}" --no-pager || true

finished_iso=$(date -Is)

{
  echo "# Container Image Pull Evidence"
  echo
  echo "Started: ${started_iso}"
  echo "Finished: ${finished_iso}"
  echo
  echo "Image: \`${IMAGE}\`"
  echo "Platform: \`${PLATFORM}\`"
  echo "Docker pull return code: \`${pull_rc}\`"
  echo "Docker image inspect return code: \`${inspect_rc}\`"
  echo "Skopeo enabled: \`${USE_SKOPEO}\`"
  if [[ -n "${skopeo_copy_rc}" ]]; then
    echo "Skopeo OCI copy return code: \`${skopeo_copy_rc}\`"
  fi
  if [[ -n "${skopeo_import_rc}" ]]; then
    echo "Skopeo docker-daemon import return code: \`${skopeo_import_rc}\`"
  fi
  echo
  echo "Artifacts:"
  echo
  echo "- \`${MANIFEST_JSON}\`"
  echo "- \`${MANIFEST_VERBOSE}\`"
  echo "- \`${DOCKER_PULL_LOG}\`"
  echo "- \`${DOCKER_INSPECT_JSON}\`"
  echo "- \`${DOCKER_DF_BEFORE}\`"
  echo "- \`${DOCKER_DF_AFTER}\`"
  echo "- \`${FS_BEFORE}\`"
  echo "- \`${FS_AFTER}\`"
  echo "- \`${DAEMON_LOG}\`"
  if [[ "${USE_SKOPEO}" == "1" ]]; then
    echo "- \`${SKOPEO_INSTALL_LOG}\`"
    echo "- \`${SKOPEO_COPY_LOG}\`"
    echo "- \`${SKOPEO_IMPORT_LOG}\`"
  fi
} >"${SUMMARY}"

if [[ "${inspect_rc}" == "0" ]]; then
  exit 0
fi

if [[ "${pull_rc}" == "0" ]]; then
  exit 1
fi

exit "${pull_rc}"
