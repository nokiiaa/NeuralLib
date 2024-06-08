using System;

namespace NeuralLib.Maths
{
    public class Error
    {
        public static float MeanSquares(MArray output, MArray expected)
        {
            float result = 0;

            for (int i = 0; i < output.Values.Length; i++)
            {
                float delta = output.Values[i] - expected.Values[i];
                result += delta * delta;
            }

            return result / 2;
        }

        public static float CrossEntropy(MArray output, MArray expected)
        {
            float E = 0;

            for (int i = 0; i < output.Values.Length; i++)
                E -= (float)Math.Log(output.Values[i]) * expected.Values[i];

            return E;
        }
    }
}
