# Changelog

This is the changelog for the open source version of tiktoken.

## [v0.9.0]
- Support for `o1` and `o3` models
- Better error messages when loading invalid vocabulary files
- Support for encoding to numpy arrays
- Delayed imports when not strictly necessary

## [v0.8.0]

- Support for `o1-` and `chatgpt-4o-` models
- Build wheels for Python 3.13
- Add possessive quantifiers to limit backtracking in regular expressions, thanks to @l0rinc!
- Provide a better error message and type for invalid token decode
- Permit tuples in type hints
- Better error message for passing invalid input to `get_encoding`
- Better error messages during plugin loading
- Add a `__version__` attribute
- Update versions of `pyo3`, `regex`, `fancy-regex`
- Drop support for Python 3.8

## [v0.7.0]

- Support for `gpt-4o`
- Performance improvements

## [v0.6.0]

- Optimise regular expressions for a 20% performance improvement, thanks to @paplorinc!
- Add `text-embedding-3-*` models to `encoding_for_model`
- Check content hash for downloaded files
- Allow pickling `Encoding` objects. Registered `Encoding` will be pickled by reference
- Workaround PyO3 bug for frozenset conversion

Thank you to @paplorinc, @mdwelsh, @Praneet460!

## [v0.5.2]

- Build wheels for Python 3.12
- Update version of PyO3 to allow multiple imports
- Avoid permission errors when using default cache logic

## [v0.5.1]

- Add `encoding_name_for_model`, undo some renames to variables that are implementation details

## [v0.5.0]

- Add `tiktoken._educational` submodule to better document how byte pair encoding works
- Ensure `encoding_for_model` knows about several new models
- Add `decode_with_offets`
- Better error for failures with the plugin mechanism
- Make more tests public
- Update versions of dependencies

## [v0.4.0]

- Add `decode_batch` and `decode_bytes_batch`
- Improve error messages and handling

## [v0.3.3]

- `tiktoken` will now make a best effort attempt to replace surrogate pairs with the corresponding
  Unicode character and will replace lone surrogates with the Unicode replacement character.

## [v0.3.2]

- Add encoding for GPT-4

## [v0.3.1]

- Build aarch64 wheels
- Make `blobfile` an optional dependency

Thank you to @messense for the environment variable that makes cargo not OOM under emulation!

## [v0.3.0]

- Improve performance by 5-20%; thank you to @nistath!
- Add `gpt-3.5-turbo` models to `encoding_for_model`
- Add prefix matching to `encoding_for_model` to better support future model versions
- Fix a bug in the README instructions on extending tiktoken
- Update the set of available encodings
- Add packaging metadata

## [v0.2.0]

- Add `tiktoken.encoding_for_model` to get the encoding for a specific model
- Improve portability of caching logic

Thank you to @fritzo, @arvid220u, @khanhvu207, @henriktorget for various small corrections

## [v0.1.2]

- Avoid use of `blobfile` for public files
- Add support for Python 3.8
- Add py.typed
- Improve the public tests

## [v0.1.1]

- Initial release
