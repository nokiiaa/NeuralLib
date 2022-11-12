using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralLib.Maths
{
    public class Activations
    {
        public enum Type
        {
            ReLU,
            LeakyReLU,
            Sigmoid
        }

        public static Func<float, float>[] Table = new Func<float, float>[]
        {
            ReLU.Function,
            ReLU.Leaky.Function,
            Sigmoid.Function
        };

        public static Func<float, float>[] TablePrime = new Func<float, float>[]
        {
            ReLU.Derivative,
            ReLU.Leaky.Derivative,
            Sigmoid.Function
        };

        public static Func<float, float> Get(Type t) => Table[(int)t];

        public static Func<float, float> GetPrime(Type t) => TablePrime[(int)t];
    }
}
