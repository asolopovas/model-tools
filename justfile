# Cross-platform recipes.
# just uses sh on Unix and cmd.exe on Windows; filesystem work is delegated to Python helpers.
set shell := ["sh", "-cu"]
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"]

uv := env_var_or_default("UV", "uv")
python_version := env_var_or_default("PYTHON_VERSION", "3.12")

# Workspace layout.
input_dir := env_var_or_default("INPUT_DIR", "input/whisper-large-v3-turbo")
hf_model_dir := env_var_or_default("HF_MODEL_DIR", "input/hf")
output_dir := env_var_or_default("OUTPUT_DIR", "output")
reports_dir := env_var_or_default("REPORTS_DIR", "output/reports")
eais_workspace := env_var_or_default("EAIS_WORKSPACE", "output/eais/whisper-large-v3-turbo")
eais_device := env_var_or_default("EAIS_DEVICE", "Gen-8")

# Whisper large-v3 / large-v3-turbo encoder, 30-second window.
model := env_var_or_default("MODEL", "output/onnx/encoder_model_fp16.onnx")
static_model := env_var_or_default("STATIC_MODEL", "output/static/encoder_model_fp16_static.onnx")
compact_ln_static_model := env_var_or_default("COMPACT_LN_STATIC_MODEL", "output/static/encoder_model_fp16_static_compact_ln.onnx")
compact_ln_sim_model := env_var_or_default("COMPACT_LN_SIM_MODEL", "output/simplified/encoder_model_fp16_static_compact_ln_sim.onnx")
sim_model := env_var_or_default("SIM_MODEL", "output/simplified/encoder_model_fp16_static_sim.onnx")
input_name := env_var_or_default("INPUT_NAME", "input_features")
input_shape := env_var_or_default("INPUT_SHAPE", "1,128,3000")

_default:
    @just --list

# Show available tasks.
help:
    @just --list

# Create the standard workspace folders.
setup:
    {{uv}} run --python {{python_version}} python tools/workspace.py setup

# Check that documentation exists and mentions current just recipes.
docs-check:
    {{uv}} run --python {{python_version}} python tools/check_docs.py

# Search Hugging Face models.
hf-search query="whisper" limit="10":
    {{uv}} run --python {{python_version}} --with huggingface-hub python tools/hf_models.py search "{{query}}" --limit {{limit}}

# Short alias: search Hugging Face models.
search query="whisper" limit="10":
    just hf-search "{{query}}" "{{limit}}"

# Show Hugging Face model metadata.
hf-info repo revision="":
    {{uv}} run --python {{python_version}} --with huggingface-hub python tools/hf_models.py info "{{repo}}" {{if revision == "" { "" } else { "--revision " + revision }}}

# List files in a Hugging Face model repo without downloading.
hf-files repo limit="80" revision="":
    {{uv}} run --python {{python_version}} --with huggingface-hub python tools/hf_models.py files "{{repo}}" --limit {{limit}} {{if revision == "" { "" } else { "--revision " + revision }}}

# Short alias: list Hugging Face repo files without downloading.
files repo limit="80" revision="":
    just hf-files "{{repo}}" "{{limit}}" "{{revision}}"

# Download a clean local Hugging Face model folder under input/hf/<repo>.
hf-download repo preset="conversion" revision="":
    {{uv}} run --python {{python_version}} --with huggingface-hub python tools/hf_models.py download "{{repo}}" --preset "{{preset}}" --output-dir "{{hf_model_dir}}" {{if revision == "" { "" } else { "--revision " + revision }}}

# Short alias: download a Hugging Face model with progress and a source manifest.
download repo preset="conversion" revision="":
    just hf-download "{{repo}}" "{{preset}}" "{{revision}}"

# Download only tokenizer/config/model-card style files; useful before pulling weights.
hf-download-minimal repo revision="":
    just hf-download "{{repo}}" minimal "{{revision}}"

# List local Hugging Face model folders downloaded by this workspace.
hf-local:
    {{uv}} run --python {{python_version}} --with huggingface-hub python tools/hf_models.py local --output-dir "{{hf_model_dir}}"

# Short alias: list local Hugging Face model folders.
local:
    just hf-local

# Check whether Hugging Face auth is configured. For gated models, set HF_TOKEN or run `hf auth login`.
hf-auth:
    {{uv}} run --python {{python_version}} --with huggingface-hub python tools/hf_models.py auth

# Short alias: check Hugging Face auth status.
auth:
    just hf-auth

# Network smoke test for Hugging Face search/file listing. Does not download weights.
hf-smoke:
    just hf-search whisper 1
    just hf-files openai/whisper-tiny 20

# Inspect an ONNX model's inputs, outputs, dynamic dimensions, and op types.
inspect model=model:
    {{uv}} run --python {{python_version}} --with onnx python tools/inspect_onnx.py "{{model}}"

# Inspect the generated static ONNX model.
inspect-static static_model=static_model:
    {{uv}} run --python {{python_version}} --with onnx python tools/inspect_onnx.py "{{static_model}}"

# Create a Samsung/Exynos-friendly static-shape encoder ONNX.
static model=model static_model=static_model input_name=input_name input_shape=input_shape:
    {{uv}} run --python {{python_version}} python tools/workspace.py require-file "{{model}}"
    {{uv}} run --python {{python_version}} python tools/workspace.py ensure-parent "{{static_model}}"
    {{uv}} run --python {{python_version}} --with onnx --with onnxruntime python -m onnxruntime.tools.make_dynamic_shape_fixed --input_name "{{input_name}}" --input_shape "{{input_shape}}" "{{model}}" "{{static_model}}"
    {{uv}} run --python {{python_version}} python tools/workspace.py ls-size "{{static_model}}"

# Run ONNX checker on a model.
check model=model:
    {{uv}} run --python {{python_version}} --with onnx python -c "import onnx, sys; m=onnx.load(sys.argv[1], load_external_data=False); onnx.checker.check_model(m); print('ONNX checker OK:', sys.argv[1])" "{{model}}"

# Run ONNX checker on the static model.
check-static static_model=static_model:
    just check "{{static_model}}"

# Fail if graph inputs/outputs/value_info still contain symbolic or unknown dimensions.
verify-static static_model=static_model:
    {{uv}} run --python {{python_version}} --with onnx python tools/verify_static_onnx.py "{{static_model}}"

# Run Samsung SDK Service documented preflight checks on an ONNX upload candidate.
verify-samsung model=compact_ln_sim_model:
    {{uv}} run --python {{python_version}} --with onnx python tools/verify_samsung_onnx.py "{{model}}" --input-name "{{input_name}}" --input-shape "{{input_shape}}"

# Build and verify the static model.
prepare: setup static check-static verify-static

# Build the single Samsung SDK Service upload candidate currently under test.
prepare-samsung: setup static
    {{uv}} run --python {{python_version}} python tools/workspace.py ensure-parent "{{compact_ln_static_model}}"
    {{uv}} run --python {{python_version}} --with onnx --with numpy python tools/onnx_compact_fp32_layernorm.py "{{static_model}}" "{{compact_ln_static_model}}" --check
    {{uv}} run --python {{python_version}} python tools/workspace.py ensure-parent "{{compact_ln_sim_model}}"
    {{uv}} run --python {{python_version}} --with onnx --with onnxruntime --with onnxsim python -m onnxsim "{{compact_ln_static_model}}" "{{compact_ln_sim_model}}" --skip-optimization fuse_qkv
    {{uv}} run --python {{python_version}} --with onnx --with numpy python tools/onnx_fold_whisper_q_scale.py "{{compact_ln_sim_model}}" "{{compact_ln_sim_model}}" --check
    just check "{{compact_ln_sim_model}}"
    just verify-static "{{compact_ln_sim_model}}"
    just verify-samsung "{{compact_ln_sim_model}}"
    just upload-list "{{compact_ln_sim_model}}"

# Optional generic simplification helper.
simplify static_model=static_model sim_model=sim_model:
    {{uv}} run --python {{python_version}} python tools/workspace.py require-file "{{static_model}}"
    {{uv}} run --python {{python_version}} python tools/workspace.py ensure-parent "{{sim_model}}"
    {{uv}} run --python {{python_version}} --with onnx --with onnxruntime --with onnxsim python -m onnxsim "{{static_model}}" "{{sim_model}}"
    {{uv}} run --python {{python_version}} python tools/workspace.py ls-size "{{sim_model}}"

# Hash one or more model artifacts.
hash +files:
    {{uv}} run --python {{python_version}} python tools/hash_model.py {{files}}

# Show which model file to upload/convert first.
upload-list model=compact_ln_sim_model:
    {{uv}} run --python {{python_version}} python tools/workspace.py upload-list "{{model}}" --input-dir "{{input_dir}}"

# Check whether Samsung Exynos AI Studio's `eais` CLI is available on PATH.
eais-check:
    {{uv}} run --python {{python_version}} python tools/samsung_eais_cli.py check

# Create/update a local Samsung EAIS CLI workspace with the prepared ONNX candidate.
eais-workspace workspace=eais_workspace device=eais_device model=compact_ln_sim_model:
    {{uv}} run --python {{python_version}} python tools/samsung_eais_cli.py workspace --workspace "{{workspace}}" --device "{{device}}" --model "{{model}}" --overwrite --overwrite-config

# Print a reproducible Samsung `eais` command without running it. Profiles for conversion: safe, no-quant, baseline-no-simplify.
eais-command command="conversion" workspace=eais_workspace profile="safe":
    {{uv}} run --python {{python_version}} python tools/samsung_eais_cli.py run "{{command}}" --workspace "{{workspace}}" --profile "{{profile}}"

# Run `eais init` in the local Samsung CLI workspace to let Samsung create native templates.
eais-init workspace=eais_workspace:
    {{uv}} run --python {{python_version}} python tools/samsung_eais_cli.py run init --workspace "{{workspace}}" --execute

# Run `eais generation` in the local Samsung CLI workspace.
eais-generation workspace=eais_workspace:
    {{uv}} run --python {{python_version}} python tools/samsung_eais_cli.py run generation --workspace "{{workspace}}" --execute

# Run `eais conversion` in the local Samsung CLI workspace. Profiles: safe, no-quant, baseline-no-simplify.
eais-conversion workspace=eais_workspace profile="safe":
    {{uv}} run --python {{python_version}} python tools/samsung_eais_cli.py run conversion --workspace "{{workspace}}" --profile "{{profile}}" --execute

# Run `eais compile` in the local Samsung CLI workspace.
eais-compile workspace=eais_workspace profile="safe":
    {{uv}} run --python {{python_version}} python tools/samsung_eais_cli.py run compile --workspace "{{workspace}}" --profile "{{profile}}" --execute

# Run `eais profiling` in the local Samsung CLI workspace.
eais-profiling workspace=eais_workspace:
    {{uv}} run --python {{python_version}} python tools/samsung_eais_cli.py run profiling --workspace "{{workspace}}" --execute

# Non-destructive test of the current real model artifacts.
test-current: inspect check inspect-static check-static verify-static verify-samsung upload-list

# Smoke-test the conversion recipes on a tiny generated ONNX model, including simplify and clean-generated.
test-smoke: setup
    {{uv}} run --python {{python_version}} --with onnx python tools/create_tiny_onnx.py "{{reports_dir}}/tiny_dynamic.onnx"
    just inspect "{{reports_dir}}/tiny_dynamic.onnx"
    just check "{{reports_dir}}/tiny_dynamic.onnx"
    just static "{{reports_dir}}/tiny_dynamic.onnx" "{{reports_dir}}/tiny_static.onnx" x 1,3
    just verify-static "{{reports_dir}}/tiny_static.onnx"
    just simplify "{{reports_dir}}/tiny_static.onnx" "{{reports_dir}}/tiny_static_sim.onnx"
    just clean-generated "{{reports_dir}}/tiny_static.onnx" "{{reports_dir}}/tiny_static_sim.onnx"
    {{uv}} run --python {{python_version}} python tools/workspace.py remove "{{reports_dir}}/tiny_dynamic.onnx"

# Run local conversion checks, docs checks, and HF network checks. Does not delete real model outputs.
test-all: docs-check test-current test-smoke hf-smoke

# Delete generated static/simplified ONNX files. Pass explicit paths for test files to avoid touching real outputs.
clean-generated static_model=static_model sim_model=sim_model:
    {{uv}} run --python {{python_version}} python tools/workspace.py remove "{{static_model}}" "{{sim_model}}"
