package tiktoken;

import org.scijava.nativelib.NativeLoader;
import java.io.IOException;

public class Encoding implements AutoCloseable
{
    static {
        // TODO: unpack the library from the jar
        // System.loadLibrary("_tiktoken_jni");
        try {
            NativeLoader.loadLibrary("_tiktoken_jni");
        }
        catch(IOException e) {
            throw new RuntimeException(e);
        }
    }

    // initialized by init
    private long handle;

    private native void init(String modelName);

    private native void destroy();

    public native long[] encode(String text, String[] allowedSpecialTokens, long maxTokenLength);

    public Encoding(String modelName) {
        this.init(modelName);
    }

    public void close() throws Exception {
        destroy();
    }
}
