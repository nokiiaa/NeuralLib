using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralLib.Maths
{
    public class Softmax
    {
        public static MArray Apply(MArray to)
        {
            MArray arr = new MArray(to.Dimensions);

            float sumExp = 0;

            for (int i = 0; i < to.Values.Length; i++)
                sumExp += (float)Math.Exp(to.Values[i]);

            sumExp = 1f / sumExp;

            for (int i = 0; i < to.Values.Length; i++)
                arr.Values[i] = (float)Math.Exp(to.Values[i]) * sumExp;

            return arr;
        }
    }
}
