package tiktoken;

import java.util.Set;

public class Encoding
{
    static {
        System.loadLibrary("_tiktoken_jni");
    }

    // initialized by init
    private long handle;

    private native void init(String modelName);

    public native long[] encode(String text, Set<String> allowedSpecialTokens, long maxTokenLength);

    public Encoding(String modelName) {
        this.init(modelName);
    }
}
