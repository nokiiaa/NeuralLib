using Newtonsoft.Json;
using System;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace NeuralLib.Maths
{
    public class MArray
    {
        const int ElementsPerVector = 8;

        public int[] Dimensions { get; set; }
        public float[] Values { get; set; }

        public MArray() { }

        public MArray(float[] values, params int[] dimensions)
        {
            Dimensions = dimensions;
            Values = (float[])values.Clone();
        }

        public MArray View(params int[] dimensions)
        {
            MArray arr = new MArray(dimensions);
            arr.Values = Values;
            return arr;
        }

        public override string ToString()
        {
            if (Dimensions.Length == 1)
                return $"[ {(string.Join(" ", Values))} ]";
            else if (Dimensions.Length == 2)
            {
                string ret = "";
                for (int y = 0; y < Dimensions[1]; y++)
                {
                    ret += "[ ";

                    for (int x = 0; x < Dimensions[0]; x++)
                    {
                        ret += Values[y * Dimensions[0] + x];
                        ret += " ";
                    }

                    ret += "]";

                    if (y != Dimensions[1] - 1)
                        ret += "\n";
                }
                return ret;
            }

            throw new NotImplementedException();
        }

        public float this[int x]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Values[x];

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => Values[x] = value;
        }

        public float this[int x, int y]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Values[y * Dimensions[0] + x];

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => Values[y * Dimensions[0] + x] = value;
        }

        public float this[int x, int y, int z]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Values[(z * Dimensions[1] + y) * Dimensions[0] + x];

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => Values[(z * Dimensions[1] + y) * Dimensions[0] + x] = value;
        }

        public float this[int x, int y, int z, int w]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Values[((w * Dimensions[2] + z) * Dimensions[1] + y) * Dimensions[0] + x];

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => Values[((w * Dimensions[2] + z) * Dimensions[1] + y) * Dimensions[0] + x] = value;
        }

        public float this[params int[] index]
        {
            get
            {
                if (Dimensions.Length == 1)
                    return Values[index[0]];
                if (Dimensions.Length == 2)
                    return Values[index[1] * Dimensions[0] + index[0]];

                int linIndex = index[index.Length - 1];

                for (int i = index.Length - 2; i >= 0; i--)
                    linIndex = linIndex * Dimensions[i] + index[i];

                return Values[linIndex];
            }

            set
            {
                if (Dimensions.Length == 1)
                    Values[index[0]] = value;
                else if (Dimensions.Length == 2)
                    Values[index[1] * Dimensions[0] + index[0]] = value;
                else
                {
                    int linIndex = index[0];

                    for (int i = 1; i < index.Length; i++)
                        linIndex += index[i] * Dimensions[i - 1];

                    Values[linIndex] = value;
                }
            }
        }

        public MArray(params int[] dimensions)
        {
            Dimensions = dimensions;
            int length = 1;
            foreach (int dim in dimensions) length *= dim;
            Values = new float[length];
        }

        public MArray(int a, int b) { Dimensions = new int[2] { a, b }; Values = new float[a * b]; }
        public MArray(int n) { Dimensions = new int[1] { n }; Values = new float[n]; }

        [JsonIgnore]
        public MArray T => Transpose();

        public static MArray operator +(MArray a, MArray b)
        {
            float[] values = new float[a.Values.Length];

            int len = values.Length & -ElementsPerVector;

            for (int i = 0; i < len; i += ElementsPerVector)
                (new Vector<float>(a.Values, i) + new Vector<float>(b.Values, i)).CopyTo(values, i);

            for (int i = len; i < values.Length; i++)
                values[i] = a.Values[i] + b.Values[i];

            return new MArray(values, a.Dimensions);
        }

        public static MArray operator -(MArray a, MArray b)
        {
            float[] values = new float[a.Values.Length];

            int len = values.Length & -ElementsPerVector;

            for (int i = 0; i < len; i += ElementsPerVector)
                (new Vector<float>(a.Values, i) - new Vector<float>(b.Values, i)).CopyTo(values, i);

            for (int i = len; i < values.Length; i++)
                values[i] = a.Values[i] - b.Values[i];

            return new MArray(values, a.Dimensions);
        }

        public static MArray operator *(MArray m, float b)
        {
            float[] values = new float[m.Values.Length];

            int len = values.Length & -ElementsPerVector;

            for (int i = 0; i < len; i += ElementsPerVector)
                (new Vector<float>(m.Values, i) * b).CopyTo(values, i);

            for (int i = len; i < values.Length; i++)
                values[i] = m.Values[i] * b;

            return new MArray(values, m.Dimensions);
        }

        public static MArray operator /(MArray m, float b)
        {
            float[] values = new float[m.Values.Length];

            int len = values.Length & -ElementsPerVector;

            for (int i = 0; i < len; i += ElementsPerVector)
                (new Vector<float>(m.Values, i) * (1f / b)).CopyTo(values, i);

            for (int i = len; i < values.Length; i++)
                values[i] = m.Values[i] / b;

            return new MArray(values, m.Dimensions);
        }


        public static MArray operator /(MArray a, MArray b)
        {
            if (a.Dimensions.Length == b.Dimensions.Length)
            {
                for (int i = 0; i < a.Dimensions.Length; i++)
                    if (a.Dimensions[i] != b.Dimensions[i])
                        goto exit;

                // Element-wise division

                float[] values = new float[a.Values.Length];

                int len = values.Length & -ElementsPerVector;

                for (int i = 0; i < len; i += ElementsPerVector)
                    (new Vector<float>(a.Values, i) / new Vector<float>(b.Values, i)).CopyTo(values, i);

                for (int i = len; i < values.Length; i++)
                    values[i] = a.Values[i] / b.Values[i];

                return new MArray(values, a.Dimensions);

            }
            exit:
                throw new ArgumentException("Cannot apply division");
        }

        public float Dot(MArray b)
        {
            float ret = 0;

            int len = Values.Length & -ElementsPerVector;

            for (int i = 0; i < len; i += ElementsPerVector)
                ret += Vector.Dot(new Vector<float>(Values, i), new Vector<float>(b.Values, i));

            for (int i = len; i < Values.Length; i++)
                ret += Values[i] * b.Values[i];

            return ret;

        }

        public static MArray operator *(MArray a, MArray b)
        {
            if (a.Dimensions.Length == b.Dimensions.Length)
            {
                for (int i = 0; i < a.Dimensions.Length; i++)
                    if (a.Dimensions[i] != b.Dimensions[i])
                        goto exit;

                // Element-wise multiplication

                float[] values = new float[a.Values.Length];

                int len = values.Length & -ElementsPerVector;

                for (int i = 0; i < len; i += ElementsPerVector)
                    (new Vector<float>(a.Values, i) * new Vector<float>(b.Values, i)).CopyTo(values, i);

                for (int i = len; i < values.Length; i++)
                    values[i] = a.Values[i] * b.Values[i];

                return new MArray(values, a.Dimensions);

            }
        exit:
            if (a.Dimensions.Length == 2 && b.Dimensions.Length == 2 && a.Dimensions[0] == b.Dimensions[1])
            {
                int m = a.Dimensions[0], n = b.Dimensions[0], l = a.Dimensions[1];
                MArray c = new MArray(n, l);
                MArray bT = b.T;

                for (int y = 0; y < l; y++)
                {
                    for (int x = 0; x < n; x++)
                    {
                        float D = 0;

                        int len = m & -ElementsPerVector;

                        for (int i = 0; i < len; i += ElementsPerVector)
                            D += Vector.Dot(new Vector<float>(a.Values, i + y * m), new Vector<float>(bT.Values, i + x * m));

                        for (int i = len; i < m; i++)
                            D += a.Values[i + y * m] * b.Values[x + i * n];

                        c.Values[x + y * n] = D;
                    }
                }

                return c;
            }
            else
                throw new ArgumentException("Cannot apply matrix multiplication");
        }

        public MArray Transpose()
        {
            if (Dimensions.Length > 2)
                throw new ArgumentException("Can't take the transpose of a matrix of dimension > 2");

            if (Dimensions.Length == 1)
                return new MArray(Values, 1, Dimensions[0]);
            else
            {
                var array = new MArray(Dimensions[1], Dimensions[0]);

                for (int y = 0; y < Dimensions[1]; y++)
                    for (int x = 0; x < Dimensions[0]; x++)
                        array.Values[x * Dimensions[1] + y] = Values[y * Dimensions[0] + x];

                return array;
            }
        }

        public MArray ApplyFunc(Func<float, float> func)
        {
            float[] values = new float[Values.Length];

            for (int i = 0; i < Values.Length; i++)
                values[i] = func(Values[i]);

            return new MArray(values, Dimensions);
        }

        public void UniformRandomize(float min, float max)
        {
            Random random = new Random();

            for (int i = 0; i < Values.Length; i++)
                Values[i] = min + (max - min) * (float)random.NextDouble();
        }
    }
}