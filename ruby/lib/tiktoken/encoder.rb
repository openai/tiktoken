module Tiktoken
    class Encoder
        def encode(text, allowed_special=[], disallowed_special="all")
            _encode(text, allowed_special, disallowed_special)
        end

        def encode_ordinary(text)
            _encode_ordinary(text)
        end

        def decode(tokens, utf_opts={invalid: :replace, undef: :replace})
            _bytes = _decode(tokens)
            _bytes.pack('C*').encode('UTF-8', **utf_opts)
        end
    end
end