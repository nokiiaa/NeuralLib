using System;

namespace NeuralLib.Maths
{
    public class GaussRandom
    {
        public static void RandomizeArray(MArray arr, float mean, float deviation, Random rand = null)
        {
            rand = rand ?? new Random();

            for (int i = 0; i < arr.Values.Length; i++)
            {
                float u1 = 1 - (float)rand.NextDouble();
                float u2 = 1 - (float)rand.NextDouble();
                float randStdNormal = (float)Math.Sqrt(-2 * Math.Log(u1)) * (float)Math.Sin(2 * Math.PI * u2);
                float randNormal = mean + deviation * randStdNormal;
                arr.Values[i] = (float)randNormal;
            }
        }
    }
}