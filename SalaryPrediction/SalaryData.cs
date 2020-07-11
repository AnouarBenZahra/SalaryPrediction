using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SalaryPrediction
{
    public class SalaryData
    {
        [LoadColumn(0)]
        public float YearsExperience { get; set; }
        [LoadColumn(1)]
        public float Salary { get; set; }
    }
}
 