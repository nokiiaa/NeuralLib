using System;

namespace NeuralLib.Maths
{
    public class Sigmoid
    {
        public static Func<float, float> Function => x => 1f / (1 + (float)Math.Exp(-x));

        public static Func<float, float> Derivative => x => Function(x) * (1 - Function(x));
    }
}
