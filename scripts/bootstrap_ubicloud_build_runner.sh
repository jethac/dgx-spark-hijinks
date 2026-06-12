#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a persistent Ubicloud Ubuntu VM as a GitHub Actions build runner.
# This intentionally installs CUDA toolkit/nvcc only; no GPU is required.

RUNNER_USER=${RUNNER_USER:-actions}
RUNNER_BASE=${RUNNER_BASE:-/opt/actions-runner}
RUNNER_WORK_BASE=${RUNNER_WORK_BASE:-/opt/actions-work}
CCACHE_DIR=${CCACHE_DIR:-/opt/build-cache/ccache}
CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-100G}
CUDA_MAJOR_MINOR=${CUDA_MAJOR_MINOR:-13-0}
INSTALL_CUDA=${INSTALL_CUDA:-1}
INSTALL_DOCKER=${INSTALL_DOCKER:-1}
SWAP_SIZE=${SWAP_SIZE:-64G}

if [[ -z "${GITHUB_REPO:-}" ]]; then
  echo "GITHUB_REPO is required, for example jethac/vllm" >&2
  exit 2
fi

if [[ -z "${GITHUB_RUNNER_TOKEN:-}" ]]; then
  echo "GITHUB_RUNNER_TOKEN is required; get one with:" >&2
  echo "  gh api -X POST repos/${GITHUB_REPO}/actions/runners/registration-token --jq .token" >&2
  exit 2
fi

if [[ "$(id -u)" -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

arch=$(uname -m)
case "${arch}" in
  x86_64)
    runner_arch=x64
    cuda_repo_arch=x86_64
    default_labels="ubicloud-persistent-build-x64,cuda-toolkit-13,ccache,docker"
    ;;
  aarch64|arm64)
    runner_arch=arm64
    cuda_repo_arch=sbsa
    default_labels="ubicloud-persistent-build-arm64,cuda-toolkit-13,ccache,docker"
    ;;
  *)
    echo "Unsupported architecture: ${arch}" >&2
    exit 2
    ;;
esac

repo_slug=${GITHUB_REPO//\//-}
runner_name=${RUNNER_NAME:-"ubicloud-${repo_slug}-${runner_arch}-$(hostname)"}
runner_labels=${RUNNER_LABELS:-${default_labels}}
runner_dir=${RUNNER_DIR:-"${RUNNER_BASE}/${repo_slug}-${runner_arch}"}
runner_work=${RUNNER_WORK:-"${RUNNER_WORK_BASE}/${repo_slug}-${runner_arch}"}

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates ccache clang cmake curl file git jq less \
  ninja-build pkg-config python3 python3-dev python3-pip python3-venv \
  rsync software-properties-common unzip wget zip zstd

if [[ "${INSTALL_CUDA}" == "1" ]]; then
  distro=ubuntu2404
  keyring_deb=/tmp/cuda-keyring_1.1-1_all.deb
  curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${cuda_repo_arch}/cuda-keyring_1.1-1_all.deb" \
    -o "${keyring_deb}"
  dpkg -i "${keyring_deb}"
  apt-get update
  apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_MAJOR_MINOR}"
fi

if [[ "${INSTALL_DOCKER}" == "1" ]]; then
  apt-get update
  apt-get install -y --no-install-recommends docker.io docker-buildx
  systemctl enable docker >/dev/null 2>&1 || true
  systemctl start docker || service docker start || true
fi

if [[ "${SWAP_SIZE}" != "0" && "${SWAP_SIZE}" != "0G" ]]; then
  swap_file=/swapfile
  if ! swapon --show=NAME | grep -qx "${swap_file}"; then
    if [[ ! -f "${swap_file}" ]]; then
      fallocate -l "${SWAP_SIZE}" "${swap_file}"
      chmod 600 "${swap_file}"
      mkswap "${swap_file}"
    fi
    swapon "${swap_file}"
  fi
  if ! grep -qE "^${swap_file}[[:space:]]" /etc/fstab; then
    echo "${swap_file} none swap sw 0 0" >>/etc/fstab
  fi
fi

if ! id -u "${RUNNER_USER}" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash "${RUNNER_USER}"
fi

if [[ "${INSTALL_DOCKER}" == "1" ]]; then
  usermod -aG docker "${RUNNER_USER}"
fi

mkdir -p "${RUNNER_BASE}" "${RUNNER_WORK_BASE}" "${CCACHE_DIR}" "${runner_dir}" "${runner_work}"
chown -R "${RUNNER_USER}:${RUNNER_USER}" "${RUNNER_BASE}" "${RUNNER_WORK_BASE}" "${CCACHE_DIR}"

cat >/etc/profile.d/hijinks-build-runner.sh <<EOF
export CCACHE_DIR=${CCACHE_DIR}
export CCACHE_MAXSIZE=${CCACHE_MAXSIZE}
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:\$PATH
EOF

sudo -u "${RUNNER_USER}" ccache --set-config "max_size=${CCACHE_MAXSIZE}"
sudo -u "${RUNNER_USER}" env CCACHE_DIR="${CCACHE_DIR}" ccache --set-config "max_size=${CCACHE_MAXSIZE}"

if [[ "${INSTALL_DOCKER}" == "1" ]]; then
  docker version
  docker buildx version
  sudo -u "${RUNNER_USER}" docker version
  sudo -u "${RUNNER_USER}" docker buildx version
fi

latest_tag=$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | jq -r .tag_name)
runner_version=${latest_tag#v}
runner_tgz="actions-runner-linux-${runner_arch}-${runner_version}.tar.gz"
runner_url="https://github.com/actions/runner/releases/download/${latest_tag}/${runner_tgz}"

if [[ ! -x "${runner_dir}/config.sh" ]]; then
  tmp_tgz="/tmp/${runner_tgz}"
  curl -fsSL "${runner_url}" -o "${tmp_tgz}"
  tar -xzf "${tmp_tgz}" -C "${runner_dir}"
  chown -R "${RUNNER_USER}:${RUNNER_USER}" "${runner_dir}"
fi

pushd "${runner_dir}" >/dev/null
if [[ -f .runner ]]; then
  ./svc.sh stop || true
  ./svc.sh uninstall || true
  sudo -u "${RUNNER_USER}" ./config.sh remove --token "${GITHUB_RUNNER_TOKEN}" || true
fi

sudo -u "${RUNNER_USER}" ./config.sh \
  --url "https://github.com/${GITHUB_REPO}" \
  --token "${GITHUB_RUNNER_TOKEN}" \
  --name "${runner_name}" \
  --labels "${runner_labels}" \
  --work "${runner_work}" \
  --unattended \
  --replace

./svc.sh install "${RUNNER_USER}"
service_name=$(systemctl list-unit-files "actions.runner.*.${runner_name}.service" --no-legend | awk '{print $1}' | head -n1)
if [[ -n "${service_name}" ]]; then
  mkdir -p "/etc/systemd/system/${service_name}.d"
  cat >"/etc/systemd/system/${service_name}.d/10-hijinks-build-env.conf" <<EOF
[Service]
Environment=CUDA_HOME=/usr/local/cuda
Environment=CCACHE_DIR=${CCACHE_DIR}
Environment=CCACHE_MAXSIZE=${CCACHE_MAXSIZE}
Environment=PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin
EOF
  systemctl daemon-reload
fi
./svc.sh start
popd >/dev/null

echo "Registered ${runner_name} for ${GITHUB_REPO}"
echo "Labels: self-hosted,Linux,${runner_arch},${runner_labels}"
echo "ccache: ${CCACHE_DIR} (${CCACHE_MAXSIZE})"
if [[ "${INSTALL_DOCKER}" == "1" ]]; then
  echo "docker: $(docker --version)"
  echo "docker buildx: $(docker buildx version)"
fi
swapon --show || true
