use std::collections::HashSet;
use std::sync::MutexGuard;

use _tiktoken_core::openai_public::EncodingLazy;
use jni::JNIEnv;
// These objects are what you should use as arguments to your native
// function. They carry extra lifetime information to prevent them escaping
// this context and getting used after being GC'd.
use jni::objects::{JObject, JString};

// This is just a pointer. We'll be returning it from our function. We
// can't return one of the objects with lifetime information because the
// lifetime checker won't let us.
use jni::sys::{jarray, jlong};

use _tiktoken_core::{self, CoreBPENative};

#[no_mangle]
pub extern "system" fn Java_tiktoken_Encoding_init(env: JNIEnv, obj: JObject, model_name: JString) {
    // First, we have to get the string out of Java. Check out the `strings`
    // module for more info on how this works.
    let model_name: String = env
        .get_string(model_name)
        .expect("Unable to get Java model name")
        .into();

    let encoding_name = _tiktoken_core::openai_public::MODEL_TO_ENCODING
        .get(&model_name)
        .expect("Unable to find model");

    // TODO: this is actually mergable_ranks (lazy)
    let mut encoding = _tiktoken_core::openai_public::REGISTRY
        .get(encoding_name)
        .expect("Unable to find encoding");

    // TODO: initialize the CoreBPE object

    // TODO: this should be CoreBPE

    let bpe_native = CoreBPENative::new(
        encoding.get().unwrap(),
        encoding.special_tokens.clone(),
        &encoding.pat_str,
    )
    .unwrap();

    unsafe {
        env.set_rust_field(obj, "handle", bpe_native).unwrap();
    }

    // env.set_field(obj, "handle", "J", jni::objects::JValue::Long(encoding_ptr)).expect("Unable to store handle");
}

#[no_mangle]
pub extern "system" fn Java_tiktoken_Encoding_destroy(env: JNIEnv, obj: JObject) {
    unsafe {
        let _: CoreBPENative = env.take_rust_field(obj, "handle").unwrap();
    }
}

#[no_mangle]
pub extern "system" fn Java_tiktoken_Encoding_encode(
    env: JNIEnv,
    obj: JObject,
    text: JString,
    allowedSpecialTokens: jarray,
    maxTokenLength: jlong,
) -> jarray {
    let encoding: MutexGuard<CoreBPENative> = unsafe { env.get_rust_field(obj, "handle").unwrap() };

    let enc = encoding;
    let input: String = env
        .get_string(text)
        .expect("Couldn't get java string!")
        .into();

    let len = env.get_array_length(allowedSpecialTokens).unwrap();
    let mut strings: Vec<String> = Vec::with_capacity(len as usize);
    for i in 0..len {
        let element: JObject = env
            .get_object_array_element(allowedSpecialTokens, i)
            .unwrap();
        let current: String = env.get_string(element.into()).unwrap().into();
        strings.push(current);
    }

    let v2: HashSet<&str> = strings.iter().map(|s| &**s).collect();

    let (tokens, _, _) = enc._encode_native(&input, &v2, Some(maxTokenLength as usize));

    let mut output = env
        .new_long_array(tokens.len().try_into().unwrap())
        .unwrap();

    let array_of_u64 = tokens.iter().map(|x| *x as i64).collect::<Vec<i64>>();
    env.set_long_array_region(output, 0, array_of_u64.as_slice())
        .unwrap();

    output
}
