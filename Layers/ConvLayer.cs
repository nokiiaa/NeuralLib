using NeuralLib.Accel;
using NeuralLib.Maths;
using System;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;

namespace NeuralLib.Layers
{
    public unsafe class ConvLayer : Layer
    {
        public ConvLayer() { }

        public ConvLayer(
            int inputW, int inputH, int inputChannels,
            int kernelW, int kernelH, int filters, int padding,
            Activations.Type activation,
            bool randomInit, float initScale)
        {
            InputW = inputW;
            InputH = inputH;
            InputChannels = inputChannels;
            KernelW = kernelW;
            KernelH = kernelH;
            FilterCount = filters;
            Padding = padding;
            InputCount = inputW * inputH * inputChannels;
            OutputW = padding * 2 + inputW - kernelW + 1;
            OutputH = padding * 2 + inputH - kernelH + 1;
            OutputCount = OutputW * OutputH * filters;
            Weights = new MArray(kernelW, kernelH, inputChannels, filters);
            Bias = new MArray(1, OutputCount);
            Activation = activation;

            if (randomInit)
            {
                float w = (float)Math.Sqrt(6d / (kernelW * kernelH));
                Weights.UniformRandomize(-w, w);
                //GaussRandom.RandomizeArray(Weights, 0, 1);
                Weights *= initScale;
            }
        }

        public override LayerType Type => LayerType.Conv;

        public override int InputCount { get; set; }

        public override int OutputCount { get; set; }

        public int InputW { get; set; }

        public int InputH { get; set; }

        public int InputChannels { get; set; }

        public int KernelW { get; set; }

        public int KernelH { get; set; }

        public int OutputW { get; set; }

        public int OutputH { get; set; }

        public int FilterCount { get; set; }

        public int Padding { get; set; }

        public override MArray Weights { get; set; }

        public override MArray Bias { get; set; }

        /// <summary>
        /// The activation function.
        /// </summary>
        public Activations.Type Activation { get; set; }

        public override MArray Forward(MArray input) => Linear(input).ApplyFunc(Activations.Get(Activation));

        private MArray PaddedImage(MArray input, int padding)
        {
            MArray inputView = input.View(InputW, InputH, InputChannels);
            MArray output = new MArray(InputW + padding * 2, InputH + padding * 2, InputChannels);

            for (int c = 0; c < InputChannels; c++)
                for (int y = 0; y < InputH; y++)
                    for (int x = 0; x < InputW; x++)
                        output[x + padding, y + padding, c] = inputView[x, y, c];

            return output;
        }

        public override MArray Linear(MArray input)
        {
            MArray paddedInput = PaddedImage(input, Padding);
            MArray output = new MArray(1, OutputCount);
            MArray view3d = output.View(OutputW, OutputH, FilterCount);
            MArray weightsView = Weights.View(KernelW, KernelH, InputChannels, FilterCount);
            MArray biasView = Bias.View(OutputW, OutputH, FilterCount);

            MArray dotA = new MArray(KernelW * KernelH * InputChannels);
            MArray dotB = new MArray(KernelW * KernelH * InputChannels);

            for (int y = 0; y < OutputH; y++)
            {
                for (int x = 0; x < OutputW; x++)
                {
                    for (int d = 0; d < FilterCount; d++)
                    {
                        int i = 0;

                        for (int ky = 0; ky < KernelH && y + ky < paddedInput.Dimensions[1]; ky++)
                        {
                            for (int kx = 0; kx < KernelW && x + kx < paddedInput.Dimensions[0]; kx++)
                            {
                                for (int kc = 0; kc < InputChannels; kc++, i++)
                                {
                                    dotA.Values[i] = weightsView[kx, ky, kc, d];
                                    dotB.Values[i] = paddedInput[x + kx, y + ky, kc];
                                }
                            }
                        }

                        view3d[x, y, d] = biasView[x, y, d] + dotA.Dot(dotB);
                    }
                }
            }

            return output;
        }

        public override (MArray, MArray, MArray) Backward(
            MArray prevActivations, MArray thisLinear,
            MArray partialDelta)
        {
            MArray prevActivationsView = PaddedImage(prevActivations, Padding);

            // Compute delta for this layer
            MArray gBias = partialDelta * thisLinear.ApplyFunc(Activations.GetPrime(Activation));

            // Compute weight gradient
            MArray gWeights = new MArray(Weights.Dimensions);

            MArray gWeightsRef = gWeights;

            MArray deltaView = gBias.View(OutputW, OutputH, FilterCount);

            MArray dotA = new MArray(OutputW * OutputH);
            MArray dotB = new MArray(OutputW * OutputH);

            int i;

            for (int ky = 0; ky < KernelH; ky++)
                for (int kx = 0; kx < KernelW; kx++)
                    for (int c = 0; c < InputChannels; c++)
                        for (int d = 0; d < FilterCount; d++)
                        {
                            i = 0;

                            for (int y = 0; y < OutputH && y + ky < prevActivationsView.Dimensions[1]; y++)
                            {
                                for (int x = 0; x < OutputW && x + kx < prevActivationsView.Dimensions[0]; x++, i++)
                                {
                                    dotA.Values[i] = deltaView[x, y, d];
                                    dotB.Values[i] = prevActivationsView[x + kx, y + ky, c];
                                }
                            }

                            gWeightsRef[kx, ky, c, d] = dotA.Dot(dotB);
                        }

            // Compute new partial delta
            MArray newPartialDelta = new MArray(1, InputW * InputH * InputChannels);

            MArray partialDeltaView = newPartialDelta.View(InputW, InputH, InputChannels);

            dotA = new MArray(KernelW * KernelH * FilterCount);
            dotB = new MArray(KernelW * KernelH * FilterCount);

            for (int y = Padding; y < InputH + Padding; y++)
            {
                for (int x = Padding; x < InputW + Padding; x++)
                {
                    for (int c = 0; c < InputChannels; c++)
                    {
                        i = 0;

                        for (int n = 0; n < KernelH && n <= y && y - n < OutputH; n++)
                        {
                            for (int m = 0; m < KernelW && m <= x && x - m < OutputW; m++)
                            {
                                for (int d = 0; d < FilterCount; d++, i++)
                                {
                                    dotA[i] = deltaView[x - m, y - n, d];
                                    dotB[i] = Weights[m, n, c, d];
                                }
                            }
                        }

                        partialDeltaView[x - Padding, y - Padding, c] = dotA.Dot(dotB);
                    }
                }
            }

            return (newPartialDelta, gWeights, gBias);
        }
    }
}