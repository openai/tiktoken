import pytest
from tiktoken.core import (
    Encoding,
)
from tiktoken.model import (
    encoding_for_model,
    encoding_name_for_model,
)


def test_encoding_name_for_model_prefix_and_unknown():
    """
    Test encoding_name_for_model for three scenarios:
    1. Direct mapping: The model name exactly exists in MODEL_TO_ENCODING.
    2. Prefix mapping: The model name isn't a direct key but starts with a known prefix.
    3. Unrecognized model: The model name doesn't match any mapping, causing a KeyError.
    """
    direct_model = "gpt-4"
    direct_encoding = encoding_name_for_model(direct_model)
    assert (
        direct_encoding == "cl100k_base"
    ), f"Expected direct mapping for {direct_model} to be 'cl100k_base', got {direct_encoding}"
    prefix_model = "gpt-3.5-turbo-FAKE"
    prefix_encoding = encoding_name_for_model(prefix_model)
    assert (
        prefix_encoding == "cl100k_base"
    ), f"Expected prefix mapping for {prefix_model} to be 'cl100k_base', got {prefix_encoding}"
    with pytest.raises(KeyError) as exc_info:
        encoding_name_for_model("nonexistent-model")
    assert "nonexistent-model" in str(exc_info.value)


def test_encoding_for_model_returns_encoding_instance():
    """
    Test that encoding_for_model returns an instance of Encoding for a valid model name.

    This verifies that the higher-level helper function properly utilizes the underlying
    get_encoding function to return a valid Encoding instance.
    """
    model_name = "gpt2"
    encoding = encoding_for_model(model_name)
    assert isinstance(
        encoding, Encoding
    ), f"Expected encoding for {model_name} to be an instance of Encoding, got {type(encoding)}"


def test_encoding_name_for_model_empty_string():
    """
    Test that encoding_name_for_model raises a KeyError when passed an empty model name.
    The test validates that the error message includes the phrase "Could not automatically map"
    to indicate that no mapping was found for the given (empty) model name.
    """
    with pytest.raises(KeyError) as exc_info:
        encoding_name_for_model("")
    error_message = str(exc_info.value)
    assert "Could not automatically map" in error_message


def test_encoding_for_model_invalid_model_raises_key_error():
    """
    Test that encoding_for_model raises a KeyError when provided with an unrecognized model name.

    This ensures that the wrapper function properly propagates errors from
    encoding_name_for_model when no encoding mapping exists.
    """
    invalid_model_name = "unknown-model-123"
    with pytest.raises(KeyError) as exc_info:
        encoding_for_model(invalid_model_name)
    assert "unknown-model-123" in str(exc_info.value)
