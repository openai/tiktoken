package tiktoken;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class EncodingTest
{
    @Test
    public void shouldAnswerWithTrue() throws Exception
    {
        Encoding encoding = new Encoding("text-davinci-001");

        long[] a = encoding.encode("test", new String[0], 0);

        encoding.close();

        assertTrue( true );
        assertArrayEquals(new long[] {9288}, a);
    }
}
