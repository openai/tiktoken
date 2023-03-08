# Changelog

This is the changelog for the open source version of tiktoken.

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

## [v0.1.2]
- Avoid use of `blobfile` for public files
- Add support for Python 3.8
- Add py.typed
- Improve the public tests

## [v0.1.1]
- Initial release
