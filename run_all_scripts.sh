#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILED_DIR="$SCRIPT_DIR/failed"

mkdir -p "$FAILED_DIR"
rm -f "$FAILED_DIR"/*.txt

pass=0
fail=0

while IFS= read -r script; do
    # Skip __init__.py and simulator utility files
    basename=$(basename "$script")
    if [[ "$basename" == "__init__.py" ]]; then
        continue
    fi
    relpath="${script#$SCRIPT_DIR/}"
    logname="${relpath//\//__}"
    logname="${logname%.py}.txt"

    output=$(PYAUTOFIT_TEST_MODE=1 python "$script" 2>&1)
    exitcode=$?

    if [[ $exitcode -ne 0 ]]; then
        echo "FAILED: $relpath"
        {
            echo "Exit code: $exitcode"
            echo "Script: $relpath"
            echo "---"
            echo "$output"
        } > "$FAILED_DIR/$logname"
        ((fail++))
    else
        echo "PASSED: $relpath"
        ((pass++))
    fi
done < <(find "$SCRIPT_DIR/scripts" -name "*.py" | sort)

echo ""
echo "Results: $pass passed, $fail failed"
if [[ $fail -gt 0 ]]; then
    echo "Failure logs written to: $FAILED_DIR/"
    exit 1
fi
