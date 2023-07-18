// This check is new and seems buggy (possibly with PyO3 interaction)
#![allow(clippy::borrow_deref_ref)]

use std::collections::HashSet;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::PyResult;
use pyo3::types::{PyBytes, PyList, PyTuple};
use rustc_hash::FxHashMap as HashMap;

use tiktoken::core::{byte_pair_encode, CoreBPE};

#[pyclass]
pub struct PyCoreBPE {
    pub core_bpe: CoreBPE,
}


#[pymethods]
impl PyCoreBPE {
    #[new]
    fn new(
        encoder: HashMap<Vec<u8>, usize>,
        special_tokens_encoder: HashMap<String, usize>,
        pattern: &str,
    ) -> PyResult<Self> {
        println!("encoder: {:?}", encoder);
        CoreBPE::new(encoder, special_tokens_encoder, pattern)
            .map(|core_bpe| PyCoreBPE { core_bpe })
            .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.to_string()))
    }

    // ====================
    // Encoding
    // ====================

    fn encode_ordinary(&self, py: Python, text: &str) -> Vec<usize> {
        py.allow_threads(|| self.core_bpe._encode_ordinary_native(text))
    }

    fn encode(&self, py: Python, text: &str, allowed_special: HashSet<&str>) -> Vec<usize> {
        py.allow_threads(|| self.core_bpe._encode_native(text, &allowed_special).0)
    }

    fn _encode_bytes(&self, py: Python, bytes: &[u8]) -> Vec<usize> {
        py.allow_threads(|| self.core_bpe._encode_bytes(bytes))
    }

    fn encode_with_unstable(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<&str>,
    ) -> Py<PyTuple> {
        let (tokens, completions) =
            py.allow_threads(|| self.core_bpe._encode_unstable_native(text, &allowed_special));
        let py_completions =
            PyList::new(py, completions.iter().map(|seq| PyList::new(py, &seq[..])));
        (tokens, py_completions).into_py(py)
    }

    fn encode_single_token(&self, piece: &[u8]) -> PyResult<usize> {
        if let Some(token) = self.core_bpe.encoder.get(piece).copied() {
            return Ok(token);
        }
        if let Ok(piece_str) = std::str::from_utf8(piece) {
            if let Some(token) = self.core_bpe.special_tokens_encoder.get(piece_str).copied() {
                return Ok(token);
            }
        }
        Err(PyErr::new::<exceptions::PyKeyError, _>(piece.to_owned()))
    }

    fn encode_single_piece(&self, piece: &[u8]) -> Vec<usize> {
        if let Some(token) = self.core_bpe.encoder.get(piece) {
            return vec![*token];
        }
        byte_pair_encode(piece, &self.core_bpe.encoder)
    }

    // ====================
    // Decoding
    // ====================

    fn decode_bytes(&self, py: Python, tokens: Vec<usize>) -> Py<PyBytes> {
        let bytes = py.allow_threads(|| self.core_bpe._decode_native(&tokens));
        PyBytes::new(py, &bytes).into()
    }

    fn decode_single_token_bytes(&self, py: Python, token: usize) -> PyResult<Py<PyBytes>> {
        if let Some(bytes) = self.core_bpe.decoder.get(&token) {
            return Ok(PyBytes::new(py, bytes).into());
        }
        if let Some(bytes) = self.core_bpe.special_tokens_decoder.get(&token) {
            return Ok(PyBytes::new(py, bytes).into());
        }
        Err(PyErr::new::<exceptions::PyKeyError, _>(token.to_string()))
    }

    // ====================
    // Miscellaneous
    // ====================

    fn token_byte_values(&self, py: Python) -> Vec<Py<PyBytes>> {
        self.core_bpe.sorted_token_bytes
            .iter()
            .map(|x| PyBytes::new(py, x).into())
            .collect()
    }
}

#[pymodule]
pub fn _tiktoken(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCoreBPE>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap as HashMap;

    use tiktoken::core::byte_pair_split;

    #[test]
    fn very_simple_test() {
        let mut ranks = HashMap::default();
        ranks.insert(b"ab".to_vec(), 1);
        ranks.insert(b"cd".to_vec(), 2);

        let res = byte_pair_split(b"abcd", &ranks);
        assert_eq!(res, vec![b"ab", b"cd"]);
    }
}
