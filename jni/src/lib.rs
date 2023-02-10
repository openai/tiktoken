
use jni::JNIEnv;
// These objects are what you should use as arguments to your native
// function. They carry extra lifetime information to prevent them escaping
// this context and getting used after being GC'd.
use jni::objects::{JClass, JObject, JString};

// This is just a pointer. We'll be returning it from our function. We
// can't return one of the objects with lifetime information because the
// lifetime checker won't let us.
use jni::sys::{jstring, jlong};

pub extern "system" fn Java_tiktoken_Encoding_init(
    env: JNIEnv,
    obj: JObject,
    model_name: JString
) {
    use openai_public::{REGISTRY, MODEL_TO_ENCODING};

    // First, we have to get the string out of Java. Check out the `strings`
    // module for more info on how this works.
    let model_name: String =
        env.get_string(model_name).expect("Unable to get Java model name").into();

    let encoding_name = openai_public::MODEL_TO_ENCODING.get(&model_name).expect("Unable to find model");

    let encoding = openai_public::REGISTRY.get(encoding_name).expect("Unable to find encoding");

    let encoding_ptr = Box::into_raw(Box::new(encoding)) as jlong;

    env.set_field(obj, "handle", "J", jni::objects::JValue::Long(encoding_ptr)).expect("Unable to store handle");
}

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