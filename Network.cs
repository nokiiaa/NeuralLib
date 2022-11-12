using NeuralLib.Accel;
using NeuralLib.Layers;
using NeuralLib.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralLib
{

    public unsafe class Network
    {
        public List<Layer> Layers { get; set; } = new List<Layer>();

        public MArray Forward(MArray input)
        {
            MArray ffw = input;

            foreach (Layer layer in Layers)
                ffw = layer.Forward(ffw);

            return ffw;
        }

        public void UploadToGpu()
        {
            if (GpuAccel.Cuda.Available)
            {
                foreach (Layer layer in Layers)
                {
                    switch (layer)
                    {
                        case DenseLayer dense:
                            GpuAccel.Cuda.AddDenseLayer(dense.InputCount, dense.OutputCount,
                                dense.Weights.Values, dense.Bias.Values, (int)dense.Activation);
                            break;

                        case ConvLayer conv:
                            GpuAccel.Cuda.AddConvLayer(
                                conv.InputW, conv.InputH, conv.InputChannels,
                                conv.KernelW, conv.KernelH, conv.Padding, conv.FilterCount,
                                conv.Weights.Values, conv.Bias.Values, (int)conv.Activation);
                            break;

                        case PoolLayer pool:
                            GpuAccel.Cuda.AddPoolLayer(pool.InputW, pool.InputH, pool.InputCount, pool.KernelSize);
                            break;
                    }
                }
            }
        }

        public void DownloadFromGpu()
        {
            if (GpuAccel.Cuda.Available)
            {
                float*[] weightsArray = new float*[Layers.Count];
                float*[] biasArray = new float*[Layers.Count];

                for (int i = 0; i < Layers.Count; i++)
                {
                    Layer layer = Layers[i];

                    if (layer.Weights != null)
                        fixed (float* ptr = layer.Weights.Values)
                            weightsArray[i] = ptr;
                    else
                        weightsArray[i] = (float*)0;

                    if (layer.Bias != null)
                        fixed (float* ptr = layer.Bias.Values)
                            biasArray[i] = ptr;
                    else
                        biasArray[i] = (float*)0;
                }

                GpuAccel.Cuda.UploadParamsToCpu(weightsArray, biasArray);
            }
        }

        public void AdamSGD(int times, float rate, int batchSize, (MArray, MArray)[] trainingExamples)
        {
            Console.WriteLine($"CUDA Available: {GpuAccel.Cuda.Available}");

            if (GpuAccel.Cuda.Available)
            {
                UploadToGpu();
                GpuAccel.Cuda.PrepareForTraining(batchSize);
            }

            Random random = new Random();

            int t = 0;

            var weightGradientsM = new List<MArray>();
            var biasGradientsM = new List<MArray>();
            var weightGradientsV = new List<MArray>();
            var biasGradientsV = new List<MArray>();

            float b1 = .9f, b2 = .999f;
            float epsilon = 10e-4f;

            foreach (Layer layer in Layers)
            {
                weightGradientsM.Add(layer.Weights == null ? null : new MArray(layer.Weights.Dimensions));
                biasGradientsM.Add(layer.Bias == null ? null : new MArray(layer.Bias.Dimensions));

                weightGradientsV.Add(layer.Weights == null ? null : new MArray(layer.Weights.Dimensions));
                biasGradientsV.Add(layer.Bias == null ? null : new MArray(layer.Bias.Dimensions));
            }

            var weightGradients = new List<MArray>();
            var biasGradients = new List<MArray>();

            foreach (Layer layer in Layers)
            {
                if (layer.Weights == null)
                    weightGradients.Add(null);
                else
                    weightGradients.Add(new MArray(layer.Weights.Dimensions));

                if (layer.Bias == null)
                    biasGradients.Add(null);
                else
                    biasGradients.Add(new MArray(layer.Bias.Dimensions));
            }

            int exInLength = trainingExamples[0].Item1.Values.Length;
            int exOutLength = trainingExamples[0].Item2.Values.Length;

            float[] exampleInData = new float[exInLength * batchSize];
            float[] exampleOutData = new float[exOutLength * batchSize];

            for (int i = 0; i < times; i++)
            {
                int batches = trainingExamples.Length / batchSize;

                // Shuffle training examples
                for (int k = trainingExamples.Length - 1; 1 <= k; k--)
                {
                    int j = random.Next(0, k + 1);

                    var tmp = trainingExamples[j];
                    trainingExamples[j] = trainingExamples[k];
                    trainingExamples[k] = tmp;
                }

                for (int j = 0; j < batches; j++, t++)
                {
                    if (GpuAccel.Cuda.Available)
                    {
                        int a = 0, b = 0;

                        for (int k = j * batchSize; k < (j + 1) * batchSize; k++)
                        {
                            for (int g = 0; g < exInLength; g++)
                                exampleInData[a++] = trainingExamples[k].Item1.Values[g];

                            for (int g = 0; g < exOutLength; g++)
                                exampleOutData[b++] = trainingExamples[k].Item2.Values[g];
                        }

                        GpuAccel.Cuda.TrainOnBatch(exampleInData, exInLength, exampleOutData, exOutLength,
                            out float avgLoss, out int correct, rate, b1, b2, t);

                        Console.WriteLine($"Epoch {i}, batch {j}, average loss = {avgLoss}, accuracy = {correct / (float)batchSize * 100}%;");
                    }
                    else
                    {
                        object lck = new object();

                        List<List<MArray>> linears = new List<List<MArray>>();
                        List<List<MArray>> activations = new List<List<MArray>>();
                        List<List<MArray>> weights = new List<List<MArray>>();
                        List<MArray> partialDelta = new List<MArray>();
                        List<MArray> currentGWeights = new List<MArray>();
                        List<MArray> currentGBias = new List<MArray>();

                        float[] losses = new float[batchSize];
                        MArray[] tempOutputs = new MArray[batchSize];
                        int[] correctIndices = new int[batchSize];

                        void ProcessBatch(int k)
                        {
                            int index = j * batchSize + k;

                            if (index < trainingExamples.Length)
                            {
                                (MArray, MArray) example = trainingExamples[index];

                                linears[k] = new List<MArray>();
                                activations[k] = new List<MArray>();
                                weights[k] = new List<MArray>();
                                currentGWeights[k] = null;
                                currentGBias[k] = null;

                                MArray ffw = example.Item1;

                                activations[k].Add(ffw);

                                foreach (Layer layer in Layers)
                                {
                                    linears[k].Add(layer.Linear(ffw));
                                    activations[k].Add(ffw = layer.Forward(ffw));
                                    weights[k].Add(layer.Weights);
                                }

                                int correctIndex = correctIndices[k] = example.Item2.Values.ToList().IndexOf(1f);

                                tempOutputs[k] = activations[k].Last();

                                var softmax = Softmax.Apply(activations[k].Last());

                                partialDelta[k] = softmax - example.Item2;

                                losses[k] = Error.CrossEntropy(softmax, example.Item2);

                                (partialDelta[k], currentGWeights[k], currentGBias[k]) = Layers.Last().Backward(
                                    activations[k][activations[k].Count - 2],
                                    linears[k][linears[k].Count - 1],
                                    partialDelta[k]);

                                if (currentGWeights[k] != null)
                                    lock (lck)
                                        weightGradients[Layers.Count - 1] += currentGWeights[k] / batchSize;

                                if (currentGBias[k] != null)
                                    lock (lck)
                                        biasGradients[Layers.Count - 1] += currentGBias[k] / batchSize;

                                for (int l = Layers.Count - 2; l >= 0; l--)
                                {
                                    (partialDelta[k], currentGWeights[k], currentGBias[k]) = Layers[l].Backward(
                                        activations[k][l],
                                        linears[k][l],
                                        partialDelta[k]);

                                    if (currentGWeights[k] != null)
                                        lock (lck)
                                            weightGradients[l] += currentGWeights[k] / batchSize;

                                    if (currentGBias[k] != null)
                                        lock (lck)
                                            biasGradients[l] += currentGBias[k] / batchSize;
                                }
                            }
                        }

                        for (int a = 0; a < weightGradients.Count; a++)
                        {
                            MArray prms = weightGradients[a];

                            if (prms != null)
                                for (int b = 0; b < prms.Values.Length; b++)
                                    prms.Values[b] = 0;
                        }

                        for (int a = 0; a < biasGradients.Count; a++)
                        {
                            MArray prms = biasGradients[a];

                            if (prms != null)
                                for (int b = 0; b < prms.Values.Length; b++)
                                    prms.Values[b] = 0;
                        }

                        for (int b = 0; b < batchSize; b++)
                        {
                            linears.Add(null);
                            activations.Add(null);
                            weights.Add(null);
                            partialDelta.Add(null);
                            currentGWeights.Add(null);
                            currentGBias.Add(null);
                        }

                        Parallel.For(0, batchSize, ProcessBatch);

                        float avgLoss = 0;
                        int correct = 0;

                        foreach (float l in losses)
                            avgLoss += l / batchSize;

                        for (int k = 0; k < batchSize; k++)
                        {
                            MArray arr = tempOutputs[k];
                            if (arr.Values.ToList().IndexOf(arr.Values.Max()) == correctIndices[k])
                                correct++;
                        }

                        float f1 = 1f / (1 - (float)Math.Pow(b1, t));
                        float f2 = 1f / (1 - (float)Math.Pow(b2, t));

                        for (int k = 0; k < Layers.Count; k++)
                        {
                            if (Layers[k].Weights != null)
                            {
                                weightGradientsM[k] = weightGradientsM[k] * b1 + weightGradients[k] * (1 - b1);
                                weightGradientsV[k] = weightGradientsV[k] * b2 + weightGradients[k].ApplyFunc(x => x * x) * (1 - b2);

                                Layers[k].Weights -=
                                    weightGradientsM[k] / f1
                                    / (weightGradientsV[k] / f2).ApplyFunc(x => (float)Math.Sqrt(x) + epsilon) * rate;
                            }

                            if (Layers[k].Bias != null)
                            {
                                biasGradientsM[k] = biasGradientsM[k] * b1 + biasGradients[k] * (1 - b1);
                                biasGradientsV[k] = biasGradientsV[k] * b2 + biasGradients[k].ApplyFunc(x => x * x) * (1 - b2);

                                Layers[k].Bias -=
                                    biasGradientsM[k] / f1
                                    / (biasGradientsV[k] / f2).ApplyFunc(x => (float)Math.Sqrt(x) + epsilon) * rate;
                            }
                        }

                        Console.WriteLine($"Epoch {i}, batch {j}, average loss = {avgLoss}, accuracy = {correct / (float)batchSize * 100}%;");
                    }

                }
            }

            if (GpuAccel.Cuda.Available)
            {
                DownloadFromGpu();
                GpuAccel.Cuda.CleanUpAfterTraining();
            }
        }
    }
}