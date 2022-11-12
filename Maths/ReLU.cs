using System;

namespace NeuralLib.Maths
{
    public class ReLU
    {
        public class Leaky
        {
            public static Func<float, float> Function => x => x <= 0 ? .01f * x : x;

            public static Func<float, float> Derivative => x => x <= 0 ? .01f : 1;
        }

        public static Func<float, float> Function => x => x <= 0 ? 0 : x;

        public static Func<float, float> Derivative => x => x <= 0 ? 0 : 1;
    }
}
