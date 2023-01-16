use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

// tiktoken_ext is a namespace package
// submodules inside tiktoken_ext will be inspected for ENCODING_CONSTRUCTORS attributes
// - we use namespace package pattern so `pkgutil.iter_modules` is fast
// - it's a separate top-level package because namespace subpackages of non-namespace
//   packages don't quite do what you want with editable installs
use tiktoken_ext::{Encoding, EncodingConstructor};

// A global lock to protect the encoding cache and constructors
lazy_static! {
    static ref LOCK: Mutex<()> = Mutex::new(());
}

// A global cache of encoding instances
lazy_static! {
    static ref ENCODINGS: Mutex<HashMap<String, Encoding>> = Mutex::new(HashMap::new());
}

// A global map of encoding constructors
lazy_static! {
    static ref ENCODING_CONSTRUCTORS: Mutex<Option<HashMap<String, EncodingConstructor>>> =
        Mutex::new(None);
}

// Find and load the encoding constructors from the tiktoken_ext submodules
fn find_constructors() {
    // Acquire the lock and check if the constructors are already loaded
    let mut constructors = ENCODING_CONSTRUCTORS.lock().unwrap();
    if constructors.is_some() {
        return;
    }

    // Create an empty map to store the constructors
    let mut map = HashMap::new();

    // Iterate over the tiktoken_ext submodules and look for the ENCODING_CONSTRUCTORS attribute
    for mod_name in tiktoken_ext::list_modules() {
        let mod_ = tiktoken_ext::import_module(&mod_name).unwrap();
        let mod_constructors = mod_.get_attribute("ENCODING_CONSTRUCTORS").expect(&format!(
            "tiktoken plugin {} does not define ENCODING_CONSTRUCTORS",
            mod_name
        ));

        // Iterate over the constructors and insert them into the map
        for (enc_name, constructor) in mod_constructors.iter() {
            let enc_name = enc_name.to_string();
            let constructor = constructor.to_constructor().unwrap();
            if map.contains_key(&enc_name) {
                panic!(
                    "Duplicate encoding name {} in tiktoken plugin {}",
                    enc_name, mod_name
                );
            }
            map.insert(enc_name, constructor);
        }
    }

    // Replace the constructors with the loaded map
    *constructors = Some(map);
}

// Get an encoding instance by name, creating it if necessary
pub fn get_encoding(encoding_name: &str) -> MutexGuard<'static, Encoding> {
    // Check if the encoding is already cached
    if let Some(enc) = ENCODINGS.lock().unwrap().get(encoding_name) {
        return MutexGuard::map(ENCODINGS.lock().unwrap(), |_| enc);
    }

    // Acquire the lock and check again
    let _guard = LOCK.lock().unwrap();
    if let Some(enc) = ENCODINGS.lock().unwrap().get(encoding_name) {
        return MutexGuard::map(ENCODINGS.lock().unwrap(), |_| enc);
    }

    // Ensure the constructors are loaded
    if ENCODING_CONSTRUCTORS.lock().unwrap().is_none() {
        find_constructors();
    }

    // Get the constructor for the encoding name
    let constructor = ENCODING_CONSTRUCTORS
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .get(encoding_name)
        .expect(&format!("Unknown encoding {}", encoding_name));

    // Create the encoding instance and insert it into the cache
    let enc = constructor();
    ENCODINGS
        .lock()
        .unwrap()
        .insert(encoding_name.to_string(), enc.clone());

    // Return the encoding instance
    MutexGuard::map(ENCODINGS.lock().unwrap(), |_| &enc)
}

// List the available encoding names
pub fn list_encoding_names() -> Vec<String> {
    // Ensure the constructors are loaded
    if ENCODING_CONSTRUCTORS.lock().unwrap().is_none() {
        find_constructors();
    }

    // Return the keys of the constructors map
    ENCODING_CONSTRUCTORS
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}
