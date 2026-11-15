IMAGE ?= shiyunqi/open3dbench-v2.0:release
DOCKER ?= docker
JOBS ?= 8
BUILD_NETWORK ?= default
BUILD_HTTP_PROXY ?=
BUILD_HTTPS_PROXY ?=
WHEEL_CACHE ?=
WHEEL_PYTHON_IMAGE ?= python:3.8-slim

PROXY_BUILD_ARGS = $(if $(BUILD_HTTP_PROXY),--build-arg http_proxy=$(BUILD_HTTP_PROXY)) \
	$(if $(BUILD_HTTPS_PROXY),--build-arg https_proxy=$(BUILD_HTTPS_PROXY))

.PHONY: image image-refresh image-toolchain wheel-assets release-assets shell check-image openroad check-openroad dreamplace flow-test verify run release-matrix release-cts-smoke release-ablation release-ledger

wheel-assets:
	DOCKER="$(DOCKER)" \
		OPEN3DBENCH_WHEEL_CACHE="$(WHEEL_CACHE)" \
		WHEEL_PYTHON_IMAGE="$(WHEEL_PYTHON_IMAGE)" \
		WHEEL_NETWORK="$(BUILD_NETWORK)" \
		BUILD_HTTP_PROXY="$(BUILD_HTTP_PROXY)" \
		BUILD_HTTPS_PROXY="$(BUILD_HTTPS_PROXY)" \
		bash docker/prepare_wheels.sh

release-assets:
	bash docker/stage_release_assets.sh --force

image: wheel-assets release-assets
	$(DOCKER) build \
		--network=$(BUILD_NETWORK) \
		$(PROXY_BUILD_ARGS) \
		--build-arg OPENROAD_JOBS=$(JOBS) \
		--build-arg YOSYS_JOBS=$(JOBS) \
		--build-arg SV_ELAB_JOBS=$(JOBS) \
		--build-arg DREAMPLACE_JOBS=$(JOBS) \
		--build-arg OPEN3DBENCH_REVISION=$$(git rev-parse HEAD) \
		--tag $(IMAGE) \
		--file docker/Dockerfile .

# Refresh the source-only DREAMPlace layer without rebuilding the pinned binary
# toolchain.  Use `image` whenever a compiled dependency changes.
image-refresh:
	$(DOCKER) build --pull=false \
		--build-arg BASE_IMAGE=$(IMAGE) \
		--build-arg OPEN3DBENCH_REVISION=$$(git rev-parse HEAD) \
		--tag $(IMAGE) \
		--file docker/Dockerfile.refresh .

image-toolchain: wheel-assets
	$(DOCKER) build \
		--network=$(BUILD_NETWORK) \
		$(PROXY_BUILD_ARGS) \
		--build-arg OPENROAD_JOBS=$(JOBS) \
		--build-arg YOSYS_JOBS=$(JOBS) \
		--build-arg SV_ELAB_JOBS=$(JOBS) \
		--build-arg DREAMPLACE_JOBS=$(JOBS) \
		--build-arg OPEN3DBENCH_REVISION=$$(git rev-parse HEAD) \
		--target open3dbench-toolchain \
		--tag $(IMAGE)-toolchain \
		--file docker/Dockerfile .

shell:
	$(DOCKER) run --rm -it --gpus all \
		--user $$(id -u):$$(id -g) \
		--env HOME=/tmp \
		--volume $(CURDIR):/workspace \
		--workdir /workspace \
		$(IMAGE) bash

check-image:
	$(DOCKER) run --rm --gpus all \
		--volume $(CURDIR):/workspace --workdir /workspace \
		$(IMAGE) bash -lc '\
		set -e; \
		python3 -c "import torch, torch_geometric; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"; \
		python3 -c "import mtkahypar; print(mtkahypar.__file__)"; \
		python3 -c "import dreamplace.configure, dreamplace.Placer; import dreamplace.ops.place_io.place_io_cpp; import dreamplace.ops.electric_potential.electric_potential_cuda"; \
		command -v hmetis-api; \
		command -v hotspot; \
		test -f "$$OPEN3DBENCH_ASSET_ROOT/asset-manifest.json"; \
		(cd "$$OPEN3DBENCH_ASSET_ROOT" && sha256sum --check SHA256SUMS); \
		nvcc --version | tail -1; \
		test -f /opt/yosys/share/yosys/techmap.v; \
		yosys -V; \
		yosys -Q -p "plugin -i slang; help read_slang" >/dev/null; \
		yosys -Q -p "read_verilog OpenROAD-3D/flow/designs/src/gcd/gcd.v; hierarchy -top gcd; synth -top gcd"; \
		openroad -version; \
		sta -version; \
		test "$$OPEN3DBENCH_STA_CONTRACT_VERSION" = "parasitic-cutpoint-v1"; \
		test -r /opt/open3dbench-contracts/check_sta_contract.tcl; \
		openroad -no_init -exit /opt/open3dbench-contracts/check_sta_contract.tcl; \
		sta -no_init -exit /opt/open3dbench-contracts/check_sta_contract.tcl; \
		! ldd "$$(command -v hotspot)" | grep -q "not found"'
	$(DOCKER) run --rm \
		--user $$(id -u):$$(id -g) \
		$(IMAGE) bash -lc '\
		test -r "$$OPEN3DBENCH_HOTSPOT_ROOT/examples/thermal/test.config"; \
		test -r "$$OPEN3DBENCH_HOTSPOT_ROOT/examples/thermal/test.materials"; \
		test -x "$$OPEN3DBENCH_HOTSPOT_ROOT/hotspot"'

openroad: image
	@echo "OpenROAD-3D/OpenSTA-3D are installed in $(IMAGE); Docker build cache was reused where possible."

check-openroad:
	$(DOCKER) run --rm \
		$(IMAGE) bash -lc '\
		set -e; \
		test "$$(command -v openroad)" = /usr/local/bin/openroad; \
		test "$$(command -v sta)" = /usr/local/bin/sta; \
		openroad -version; \
		sta -version; \
		openroad -no_init -exit /opt/open3dbench-contracts/check_sta_contract.tcl; \
		sta -no_init -exit /opt/open3dbench-contracts/check_sta_contract.tcl'

dreamplace: image
	@echo "DREAMPlace is installed in $(IMAGE); Docker build cache was reused where possible."

flow-test:
	python3 -m unittest discover -s tests -v

verify: flow-test
	IMAGE=$(IMAGE) bash Place-3D/scripts/verify.sh

run:
	test -n "$(CONFIG)"
	IMAGE=$(IMAGE) GUROBI_LICENSE_FILE="$(GUROBI_LICENSE_FILE)" \
		bash scripts/run_release.sh "$(CONFIG)" $(RUN_ARGS)

release-matrix:
	$(DOCKER) run --rm --gpus all \
		--user $$(id -u):$$(id -g) \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-e HOME=/tmp \
		-e OPEN3DBENCH_IMAGE=$(IMAGE) \
		-e OPEN3DBENCH_IMAGE_ID=$$($(DOCKER) image inspect --format '{{.Id}}' $(IMAGE)) \
		-v $(CURDIR):/workspace -w /workspace \
		$(IMAGE) python3 run_partition_placement_matrix.py \
		--output /workspace/user_runs/release-matrix-$$(date +%Y%m%d-%H%M%S) \
		--stop pdn

release-cts-smoke:
	$(DOCKER) run --rm --gpus all \
		--user $$(id -u):$$(id -g) \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-e HOME=/tmp \
		-e OPEN3DBENCH_IMAGE=$(IMAGE) \
		-e OPEN3DBENCH_IMAGE_ID=$$($(DOCKER) image inspect --format '{{.Id}}' $(IMAGE)) \
		-v $(CURDIR):/workspace -w /workspace \
		$(IMAGE) python3 -m open3dbench.release_smoke \
		--output /workspace/user_runs/release-cts-$$(date +%Y%m%d-%H%M%S) \
		--require-clean

release-ablation:
	$(DOCKER) run --rm --gpus all \
		--user $$(id -u):$$(id -g) \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-e HOME=/tmp \
		-e OPEN3DBENCH_IMAGE=$(IMAGE) \
		-e OPEN3DBENCH_IMAGE_ID=$$($(DOCKER) image inspect --format '{{.Id}}' $(IMAGE)) \
		-v $(CURDIR):/workspace -w /workspace \
		$(IMAGE) python3 -m open3dbench.release_ablation \
		--output /workspace/user_runs/release-ablation-$$(date +%Y%m%d-%H%M%S) \
		$(ABLATION_ARGS)

release-ledger:
	@test -n "$(RESULT_ROOT)" || (echo "RESULT_ROOT is required" >&2; exit 2)
	$(DOCKER) run --rm \
		--user $$(id -u):$$(id -g) \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-e HOME=/tmp \
		-v $(CURDIR):/workspace -w /workspace \
		$(IMAGE) python3 -m open3dbench.release_ledger \
		--root /workspace/$(RESULT_ROOT)
