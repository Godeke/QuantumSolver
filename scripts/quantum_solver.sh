#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/quantum_solver.sh <command> [args...]

Commands:
  solver [solver args]   Run the CLI (arguments passed to quantum_solver.cli).
  test [pytest args]     Run pytest inside the project virtual environment.
  setup                  Create the venv and install dependencies, then exit.
  help                   Show this message.

Examples:
  scripts/quantum_solver.sh setup
  scripts/quantum_solver.sh test
  scripts/quantum_solver.sh solver --config examples/bell_state.json
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv"

find_python() {
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
    elif command -v python >/dev/null 2>&1; then
        echo "python"
    else
        echo "No python interpreter found. Install Python 3.10+ and retry." >&2
        exit 1
    fi
}

create_venv_if_missing() {
    if [[ -x "${VENV_PATH}/bin/python" ]]; then
        return
    fi
    local python_bin
    python_bin="$(find_python)"
    "${python_bin}" -m venv "${VENV_PATH}"
    if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
        echo "Failed to create virtual environment at ${VENV_PATH}." >&2
        exit 1
    fi
    echo "Created virtual environment at ${VENV_PATH}."
}

activate_venv() {
    if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
        echo "Virtual environment missing at ${VENV_PATH}. Run 'scripts/quantum_solver.sh setup' first." >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
}

install_requirements() {
    create_venv_if_missing
    activate_venv
    python -m pip install --upgrade pip
    if [[ -f "${REPO_ROOT}/requirements-dev.txt" ]]; then
        python -m pip install -r "${REPO_ROOT}/requirements-dev.txt"
    fi
}

COMMAND="${1:-help}"
shift || true

case "${COMMAND}" in
    solver)
        activate_venv
        export PYTHONPATH="${REPO_ROOT}/src"
        python -m quantum_solver.cli "$@"
        ;;
    test)
        activate_venv
        export PYTHONPATH="${REPO_ROOT}/src"
        pytest "$@"
        ;;
    setup)
        install_requirements
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: ${COMMAND}" >&2
        usage >&2
        exit 1
        ;;
esac
