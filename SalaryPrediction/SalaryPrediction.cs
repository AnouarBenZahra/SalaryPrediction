﻿using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SalaryPrediction
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary { get; set; }
    }
}
