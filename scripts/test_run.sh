#!/usr/bin/env bash

export LD_LIBRARY_PATH="$(dirname "$0")/../lib:${LD_LIBRARY_PATH}"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <test_executable> [args...]" >&2
    echo
    echo "Available executables:" >&2
    (
        cd ../build/test/ || exit
        ls test*
    ) >&2
    exit 1
    exit 1
fi

# 첫 번째 인자를 실행할 테스트 실행파일 이름으로 사용
EXEC_NAME="$1"
shift

EXEC_PATH="../build/test/${EXEC_NAME}"

# 실행파일 존재 및 실행권한 확인
if [ ! -x "${EXEC_PATH}" ]; then
    echo "Error: '${EXEC_PATH}' not found or not executable" >&2
    exit 1
fi

# 남은 인자는 그대로 전달
exec "${EXEC_PATH}" "$@"
