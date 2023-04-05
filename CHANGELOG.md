# Changelog

This is the changelog for the open source version of tiktoken.

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
- Add ``tiktoken.encoding_for_model`` to get the encoding for a specific model
- Improve portability of caching logic

Thank you to @fritzo, @arvid220u, @khanhvu207, @henriktorget for various small corrections

## [v0.1.2]
- Avoid use of `blobfile` for public files
- Add support for Python 3.8
- Add py.typed
- Improve the public tests

## [v0.1.1]
- Initial release
