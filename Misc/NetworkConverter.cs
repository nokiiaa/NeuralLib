using NeuralLib.Layers;
using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace NeuralLib.Misc
{
    public class NetworkConverter : JsonCreationConverter<Layer>
    {
        protected override Layer Create(Type objectType, JObject jObject)
        {
            return ((LayerType)jObject["Type"].Value<int>()) switch
            {
                LayerType.Conv => new ConvLayer(),
                LayerType.Dense => new DenseLayer(),
                LayerType.Pool => new PoolLayer(),
                _ => null
            };
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
            => throw new NotImplementedException();
    }
}