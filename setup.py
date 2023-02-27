from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="tiktoken",
    rust_extensions=[
        RustExtension(
            "tiktoken._tiktoken",
            binding=Binding.PyO3,
            path="python/Cargo.toml",
            # Between our use of editable installs and wanting to use Rust for performance sensitive
            # code, it makes sense to just always use --release
            debug=False,
        )
    ],
    include_package_data=True,
    package_data={ "tiktoken": ["py.typed", "registry.json", "model_to_encoding.json"] },
    packages=["tiktoken"],
    zip_safe=False,
)
