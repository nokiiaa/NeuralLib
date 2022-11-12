using System.Runtime.InteropServices;

namespace NeuralLib.Accel
{
    public static unsafe class GpuAccel
    {
        public static class Cuda
        {
            public static bool Available { get; private set; } = false;

            static Cuda()
            {
                try
                {
                    Available = Initialize();
                }
                catch
                {

                }
            }

            [DllImport("CudaCore", EntryPoint = "CudaCoreInitialize")]
            public static extern bool Initialize();

            [DllImport("CudaCore", EntryPoint = "CudaCoreAddDenseLayer")]
            public static extern void AddDenseLayer(
                int input_length, int output_length,
                float[] init_weights, float[] init_bias,
                int activation);

            [DllImport("CudaCore", EntryPoint = "CudaCoreAddConvLayer")]
            public static extern void AddConvLayer(
                int iw, int ih, int ic,
                int kw, int kh, int padding,
                int kd, float[] init_kernel, float[] init_bias,
                int activation);

            [DllImport("CudaCore", EntryPoint = "CudaCoreAddPoolLayer")]
            public static extern void AddPoolLayer(int iw, int ih, int chan, int kernel_size);

            [DllImport("CudaCore", EntryPoint = "CudaCorePrepareForTraining")]
            public static extern void PrepareForTraining(int batch_size);

            [DllImport("CudaCore", EntryPoint = "CudaCoreCleanUpAfterTraining")]
            public static extern void CleanUpAfterTraining();

            [DllImport("CudaCore", EntryPoint = "CudaCoreUploadParamsToCpu")]
            public static extern void UploadParamsToCpu(
                float*[] weightsArray, float*[] biasArray);

            [DllImport("CudaCore", EntryPoint = "CudaCoreTrainOnBatch")]
            public static extern void TrainOnBatch(
                float[] example_in_data, int one_example_in_length,
                float[] example_out_data, int one_example_out_length,
                out float out_avg_loss, out int out_correct_count,
                float learning_rate, float b1, float b2,
                int t);
        }
    }
}