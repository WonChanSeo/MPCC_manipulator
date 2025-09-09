# run.sh
#!/usr/bin/env bash
set -euo pipefail

# 사용법: ./run.sh <name> [-- python추가옵션...]
name=${1:?Usage: $(basename "$0") <name> [-- extra args]}

ts=$(date +%Y%m%d-%H%M%S)
# 스크립트 기준 경로(어디서 실행해도 안전)
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
out_dir="$script_dir/../result/$name"
mkdir -p "$out_dir"

log="$out_dir/log_${ts}.txt"
echo ">> log: $log"

# python에 추가 인자 전달: "${@:2}"
python3 "$script_dir/main_w_sim.py" --name "$name" "${@:2}" |& tee "$log"
status=${PIPESTATUS[0]}   # tee 파이프에서도 python 종료코드 보존
ln -sfn "$log" "$out_dir/latest.log"  # 최근 로그 symlink 갱신
exit $status
