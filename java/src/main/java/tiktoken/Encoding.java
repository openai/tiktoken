package tiktoken;

public class Encoding implements AutoCloseable
{
    static {
        System.loadLibrary("_tiktoken_jni");
    }

    // initialized by init
    private long handle;

    private native void init(String modelName);

    public native long[] encode(String text, String[] allowedSpecialTokens, long maxTokenLength);

    private native void destroy();


    public Encoding(String modelName) {
        this.init(modelName);
    }

    public void close() throws Exception {
        destroy();
    }
}
