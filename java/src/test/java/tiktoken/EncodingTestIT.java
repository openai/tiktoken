package tiktoken;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

// run test: mvn failsafe:integration-test
public class EncodingTestIT
{
    @Test
    public void shouldAnswerWithTrue() throws Exception
    {
        Encoding encoding = new Encoding("text-davinci-001");

        long[] a = encoding.encode("test", new String[0], 0);

        encoding.close();

        assertArrayEquals(new long[] {9288}, a);
    }
}
