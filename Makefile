PROJECT := tiktoken

.PHONY: default
default: editable_install

.PHONY: install_rust
install_rust:
	which cargo >/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.62

.PHONY: clean
clean:
	cargo clean
	pip uninstall -y $(PROJECT)
	find . | grep -E '__pycache__|\.pyc' | xargs rm -rf
	find . | grep -E '\.so' | xargs rm -rf
	rm -rf dist/ build/
	rm -rf $(PROJECT).egg-info/

.PHONY: format
format:
	@ which black >/dev/null || python3 -m pip install black
	@ which isort >/dev/null || python3 -m pip install isort
	cargo fmt -- --config group_imports=StdExternalCrate
	black --line-length 100 --skip-magic-trailing-comma --quiet .
	isort --line-length 100 --profile black --quiet .


.PHONY: format_check
format_check:
	@ which black >/dev/null || python3 -m pip install black
	@ which isort >/dev/null || python3 -m pip install isort
	cargo fmt --check -- --config group_imports=StdExternalCrate
	black --check --line-length 100 --skip-magic-trailing-comma --quiet .
	isort --check --line-length 100 --profile black --quiet .

.PHONY: lint
lint:
	cargo clippy --all -- -D warnings
	@ which flake8 >/dev/null || python3 -m pip install flake8==5 flake8-bugbear==22.9.11
	flake8 --ignore=E203,E501,W503,E731 --per-file-ignores="$(PROJECT)/__init__.py:F401 setup.py:E402" --exclude=build .

.PHONY: editable_install
editable_install:
	@ if [ -f $(PROJECT).egg-info ]; then \
		pip install --disable-pip-version-check --progress-bar=off setuptools wheel setuptools-rust ; \
		pip install --disable-pip-version-check --no-build-isolation -e . ; \
	else \
		pip install --disable-pip-version-check --no-deps --no-build-isolation --ignore-installed -e . ; \
	fi
