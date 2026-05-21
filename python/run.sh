# run.sh
#!/usr/bin/env bash
set -euo pipefail

# 사용법: ./run.sh <script> <name> [-- python추가옵션...]
# script: main_w_sim, main_w_sim_1, main_w_sim_three 등
# name: 실험 이름 (결과 폴더명에 사용)
#
# 예시:
#   ./run.sh main_w_sim test1          # result/main_w_sim/test1 에 저장
#   ./run.sh main_w_sim_1 test1        # result/main_w_sim_1/test1 에 저장
#   ./run.sh main_w_sim_three test1    # result/main_w_sim_three/test1 에 저장

script=${1:?Usage: $(basename "$0") <script> <name> [-- extra args]}
name=${2:?Usage: $(basename "$0") <script> <name> [-- extra args]}

# 스크립트 기준 경로(어디서 실행해도 안전)
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# 스크립트 파일 존재 확인
if [[ ! -f "$script_dir/${script}.py" ]]; then
    echo "Error: script '$script_dir/${script}.py' not found"
    echo "Available scripts:"
    ls -1 "$script_dir"/main_w_sim*.py 2>/dev/null | xargs -n1 basename | sed 's/\.py$//'
    exit 1
fi

ts=$(date +%Y%m%d-%H%M%S)
# 결과를 script 이름 하위 폴더에 저장
out_dir="$script_dir/../result/$script/$name"
mkdir -p "$out_dir"

log="$out_dir/log_${ts}.txt"
echo ">> script: $script"
echo ">> name: $name"
echo ">> output dir: $out_dir"
echo ">> log: $log"

# Set OSQP log path
export OSQP_LOG_PATH="$out_dir/osqp_admm_iterations_${ts}.txt"
echo ">> OSQP log: $OSQP_LOG_PATH"

# Set initialGuess log path
export INITIAL_GUESS_LOG_PATH="$out_dir/initialGuess_calls.txt"
echo ">> initialGuess log: $INITIAL_GUESS_LOG_PATH"

# Set detailed fallback/projection diagnostics path
export FALLBACK_DIAGNOSTICS_LOG_PATH="$out_dir/fallback_diagnostics.csv"
echo ">> fallback diagnostics: $FALLBACK_DIAGNOSTICS_LOG_PATH"

# Route generated ASIC/MLP testcase logs under this run directory.
export ASIC_TESTCASE_BASE_DIR="$out_dir/asic_testcases"
echo ">> ASIC testcase base: $ASIC_TESTCASE_BASE_DIR"

# python에 추가 인자 전달: "${@:3}" (3번째 인자부터)
python3 "$script_dir/${script}.py" --name "$name" "${@:3}" |& tee "$log"
status=${PIPESTATUS[0]}   # tee 파이프에서도 python 종료코드 보존
ln -sfn "$log" "$out_dir/latest.log"  # 최근 로그 symlink 갱신
exit $status
