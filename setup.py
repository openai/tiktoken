from setuptools import setup
from setuptools_rust import Binding, RustExtension

public = True

if public:
    version = "0.1.1"

setup(
    name="tiktoken",
    version=version,
    rust_extensions=[
        RustExtension(
            "tiktoken._tiktoken",
            binding=Binding.PyO3,
            # Between our use of editable installs and wanting to use Rust for performance sensitive
            # code, it makes sense to just always use --release
            debug=False,
        )
    ],
    packages=["tiktoken", "tiktoken_ext"],
    zip_safe=False,
)
