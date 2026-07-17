#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-"/data3/chenzixuan/llava_all_image_video/downloads/media"}
RESOLVE_LOCK=${RESOLVE_LOCK:-"/tmp/videollava_hf_resolve.lock"}
REPO_URL="https://huggingface.co/datasets/LanguageBind/Video-LLaVA/resolve/main"

FILES=(
  "llava_image_tune_2.zip.001:41943040000:0ed2c933ac47b5d49bebb9517746a667c667ed67b15c2d05d00212ea16bbc893"
  "llava_image_tune_2.zip.002:30468547028:9686f200a3e5cbe76a5434b123351a850e70d7481af694ed1303afaf439947eb"
  "videochatgpt_tune_2.zip.001:41943040000:8816eb9a2cae5253a12d05df21ff85c76ec31f35686956833052b0b5d2e68320"
  "videochatgpt_tune_2.zip.002:41943040000:f4050dfdcdf086fa31a753151157b16f64d9034b58b84452a686edde6bf06db8"
  "videochatgpt_tune_2.zip.003:41943040000:a9b02e0621e7ea81faffeacc6f314185b5d9665cdfb679a1780197b8b820c9a7"
  "videochatgpt_tune_2.zip.004:41943040000:52bdb835f1df50177080a6c4e0dfc5c94609ed935267ee1a856001d164fa5db1"
  "videochatgpt_tune_2.zip.005:4103350672:bc5f2f4cb21cf0c02381810a6c1835033d350b69ed35f214bcd42b01f10c5e48"
)

mkdir -p "$OUTPUT_DIR"

resolve_url() {
  local name=$1
  local offset=$2
  local headers location

  exec 9>"$RESOLVE_LOCK"
  flock 9
  while true; do
    if headers=$(curl --silent --show-error --connect-timeout 30 --max-time 60 \
      --header "Range: bytes=${offset}-" --dump-header - --output /dev/null \
      "$REPO_URL/$name?download=true"); then
      location=$(printf '%s\n' "$headers" | tr -d '\r' | sed -n 's/^[Ll]ocation: //p' | tail -1)
      if [[ -n "$location" ]]; then
        printf '%s' "$location"
        exec 9>&-
        return 0
      fi
    fi
    echo "Retrying signed URL for $name at byte $offset" >&2
    sleep 5
  done
}

download_one() {
  local spec=$1
  local name expected sha256 output partial current signed actual_sha
  IFS=: read -r name expected sha256 <<< "$spec"
  output="$OUTPUT_DIR/$name"
  partial="$output.part"

  if [[ -f "$output" ]]; then
    current=$(stat -c '%s' "$output")
    if [[ "$current" == "$expected" ]]; then
      echo "$name already has the expected size."
      return 0
    fi
    echo "Unexpected completed-file size for $output: $current (expected $expected)" >&2
    return 1
  fi

  touch "$partial"
  while true; do
    current=$(stat -c '%s' "$partial")
    if (( current == expected )); then
      break
    fi
    if (( current > expected )); then
      echo "Partial file is larger than expected: $partial ($current > $expected)" >&2
      return 1
    fi

    signed=$(resolve_url "$name" "$current")
    echo "Downloading $name from byte $current" >&2
    if ! curl --noproxy "us.aws.cdn.hf.co" --fail --silent --show-error \
      --connect-timeout 30 --header "Range: bytes=${current}-" "$signed" >> "$partial"; then
      echo "CDN connection interrupted for $name; preserving the new checkpoint." >&2
      sleep 2
    fi
  done

  actual_sha=$(sha256sum "$partial" | cut -d' ' -f1)
  if [[ "$actual_sha" != "$sha256" ]]; then
    echo "SHA256 mismatch for $partial: $actual_sha (expected $sha256)" >&2
    return 1
  fi
  mv "$partial" "$output"
  echo "Verified $output"
}

pids=()
for spec in "${FILES[@]}"; do
  download_one "$spec" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
