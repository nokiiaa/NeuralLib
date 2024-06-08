using NeuralLib.Accel;
using NeuralLib.Maths;
using System;
using System.Runtime.CompilerServices;

namespace NeuralLib.Layers
{
    public unsafe class PoolLayer : Layer
    {
        public override LayerType Type => LayerType.Pool;

        public override int InputCount { get; set; }

        public override int OutputCount { get; set; }

        public int Channels { get; set; }

        public int KernelSize { get; set; }

        public int InputW { get; set; }

        public int InputH { get; set; }

        public int OutputW { get; set; }

        public int OutputH { get; set; }

        public override MArray Weights { get => null; set { } }

        public override MArray Bias { get => null; set { } }

        public PoolLayer() { }

        public PoolLayer(int inputW, int inputH, int channels, int kernelSize)
        {
            if (InputW % kernelSize != 0 || InputH % kernelSize != 0)
                throw new ArgumentException("Image bounds not divisible by kernel size");

            InputW = inputW;
            InputH = inputH;
            Channels = channels;
            KernelSize = kernelSize;
            InputCount = inputW * inputH * channels;
            OutputW = inputW / kernelSize;
            OutputH = inputH / kernelSize;
            OutputCount = OutputW * OutputH * channels;
        }

        public override MArray Forward(MArray input) => Linear(input);

        public override MArray Linear(MArray input)
        {
            MArray output = new MArray(1, OutputCount);

            MArray inputView = input.View(InputW, InputH, Channels);
            MArray outputView = output.View(OutputW, OutputH, Channels);

            for (int c = 0; c < Channels; c++)
            {
                for (int y = 0; y < OutputH; y++)
                {
                    for (int x = 0; x < OutputW; x++)
                    {
                        float avg = 0;

                        for (int ky = 0; ky < KernelSize && y * KernelSize + ky < InputH; ky++)
                            for (int kx = 0; kx < KernelSize && x * KernelSize + kx < InputW; kx++)
                                avg += inputView[x * KernelSize + kx, y * KernelSize + ky, c];

                        outputView[x, y, c] = avg / (KernelSize * KernelSize);
                    }
                }
            }

            return output;
        }


        public override (MArray, MArray, MArray) Backward(
            MArray prevActivations, MArray thisLinear,
            MArray partialDelta)
        {
            var oldPartialDelta = partialDelta.View(OutputW, OutputH, Channels);
            MArray newPartialDelta = new MArray(1, InputCount);
            var deltaView = newPartialDelta.View(InputW, InputH, Channels);

            for (int c = 0; c < Channels; c++)
                for (int y = 0; y < InputH; y++)
                    for (int x = 0; x < InputW; x++)
                        deltaView[x, y, c] = oldPartialDelta[x / KernelSize, y / KernelSize, c] / (KernelSize * KernelSize);

            return (newPartialDelta, null, null);
        }
    }
}