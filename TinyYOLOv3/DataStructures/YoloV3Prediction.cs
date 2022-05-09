using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetYOLOv3ConsoleApp.DataStructures
{
    public class YoloV3Prediction
    {
        /// <summary>
        /// ((52 x 52) + (26 x 26) + 13 x 13)) x 3 = 10,647.
        /// </summary>
        public const int YoloV3BboxPredictionCount = 10_647;

        /// <summary>
        /// Boxes
        /// <para>Size is [1 x 'n_candidates' x 4]</para>
        /// </summary>
        [VectorType(1, YoloV3BboxPredictionCount, 4)]
        //[ColumnName("yolonms_layer_1/ExpandDims_1:0")] 
        [ColumnName("yolonms_layer_1")]
        public float[] Boxes { get; set; }

        /// <summary>
        /// Scores
        /// <para>Size is [1 x 80 x 'n_candidates']</para>
        /// </summary>
        [VectorType(1, 80, YoloV3BboxPredictionCount)]
        //[ColumnName("yolonms_layer_1/ExpandDims_3:0")]
        [ColumnName("yolonms_layer_1:1")]
        public float[] Scores { get; set; }

        /// <summary>
        /// Concat
        /// <para>Size is ['nbox' x 3]</para>
        /// </summary>
        [VectorType(0, 3)]
        //[ColumnName("yolonms_layer_1/concat_2:0")]
        [ColumnName("yolonms_layer_1:2")]
        public int[] Concat { get; set; }
    }
}
