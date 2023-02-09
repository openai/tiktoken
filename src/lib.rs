// This check is new and seems buggy (possibly with PyO3 interaction)
#![allow(clippy::borrow_deref_ref)]

use std::collections::HashSet;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};
use pyo3::PyResult;
use rustc_hash::FxHashMap as HashMap;

mod util;
mod core;
mod load;
mod openai_public;

#[macro_use]
extern crate lazy_static;

#[pyclass]
struct CoreBPE {
    native: core::CoreBPENative,
}

#[pymethods]
impl CoreBPE {
    #[new]
    fn new(
        encoder: HashMap<Vec<u8>, usize>,
        special_tokens_encoder: HashMap<String, usize>,
        pattern: &str,
    ) -> PyResult<Self> {
        let native = core::CoreBPENative::new(encoder, special_tokens_encoder, pattern)
            .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.to_string()))?;
        Ok(CoreBPE { native })
    }

    // ====================
    // Encoding
    // ====================

    fn encode_ordinary(&self, py: Python, text: &str) -> Vec<usize> {
        py.allow_threads(|| self.native._encode_ordinary_native(text))
    }

    fn encode(&self, py: Python, text: &str, allowed_special: HashSet<&str>) -> Vec<usize> {
        py.allow_threads(|| self.native._encode_native(text, &allowed_special, None).0)
    }

    fn _encode_bytes(&self, py: Python, bytes: &[u8]) -> Vec<usize> {
        py.allow_threads(|| {
            self.native._encode_bytes(bytes)
        })
    }

    fn encode_with_unstable(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<&str>,
    ) -> Py<PyTuple> {
        let (tokens, completions) =
            py.allow_threads(|| self.native._encode_unstable_native(text, &allowed_special));
        let py_completions =
            PyList::new(py, completions.iter().map(|seq| PyList::new(py, &seq[..])));
        (tokens, py_completions).into_py(py)
    }

    fn encode_single_token(&self, piece: &[u8]) -> PyResult<usize> {
        self.native.encode_single_token(piece).map_err(|e| PyErr::new::<exceptions::PyKeyError, _>(e))
    }

    // ====================
    // Decoding
    // ====================

    fn decode_bytes(&self, py: Python, tokens: Vec<usize>) -> Py<PyBytes> {
        let bytes = py.allow_threads(|| self.native._decode_native(&tokens));
        PyBytes::new(py, &bytes).into()
    }

    fn decode_single_token_bytes(&self, py: Python, token: usize) -> PyResult<Py<PyBytes>> {
        self.native.decode_single_token_bytes(token).map(|bytes| PyBytes::new(py, &bytes).into())
            .map_err(|e| PyErr::new::<exceptions::PyKeyError, _>(e))
    }

    // ====================
    // Miscellaneous
    // ====================

    fn token_byte_values(&self, py: Python) -> Vec<Py<PyBytes>> {
        self.native.token_byte_values()
            .iter()
            .map(|x| PyBytes::new(py, x).into())
            .collect()
    }
}

// pub fn py_data_gym_to_mergable_bpe_ranks(py: Python, vocab_bpe_file: &str, encoder_json_file: &str) -> PyResult<HashMap<PyBytes, usize>> {
#[pyfunction]
pub fn py_data_gym_to_mergable_bpe_ranks(py: Python, vocab_bpe_file: &str, encoder_json_file: &str) -> PyResult<HashMap<Vec<u8>, usize>> {
    let ranks = load::data_gym_to_mergeable_bpe_ranks(vocab_bpe_file, encoder_json_file)
        .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.to_string()))?;

    Ok(ranks)
    // Ok(ranks
    //     .iter()
    //     .map(|(k, v)| (PyBytes::new(py, k).into(), *v))
    //     .collect::<HashMap<PyBytes, usize>>())
}

#[pymodule]
fn _tiktoken(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CoreBPE>()?;
    m.add_function(wrap_pyfunction!(crate::py_data_gym_to_mergable_bpe_ranks, m)?)?;
    Ok(())
}

use jni::JNIEnv;
// These objects are what you should use as arguments to your native
// function. They carry extra lifetime information to prevent them escaping
// this context and getting used after being GC'd.
use jni::objects::{JClass, JString};

// This is just a pointer. We'll be returning it from our function. We
// can't return one of the objects with lifetime information because the
// lifetime checker won't let us.
use jni::sys::jstring;

// pub extern "system" fn Java_tiktoken_Encoding_encode(env: JNIEnv,
//                                              class: JClass,
//                                              input: JString)
//                                              -> jstring {
//     // First, we have to get the string out of Java. Check out the `strings`
//     // module for more info on how this works.
//     let input: String =
//         env.get_string(input).expect("Couldn't get java string!").into();

//     // Then we have to create a new Java string to return. Again, more info
//     // in the `strings` module.
//     let output = env.new_string(format!("Hello, {}!", input))
//         .expect("Couldn't create java string!");

//     // Finally, extract the raw pointer to return.
//     output.into_inner()
// }