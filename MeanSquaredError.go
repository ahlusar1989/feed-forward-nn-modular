package go_nn_activation_func_experiment

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

// MeanSquaredError implements Mean Squared Error calculations for ErrorCriterion interface
type MeanSquaredError struct {
}

var _ ErrorCriterion = new(MeanSquaredError)

func (*MeanSquaredError) Derivative(prediction, actual *mat64.Dense) *mat64.Dense {
	mat := &mat64.Dense{}
	mat.Sub(prediction, actual)
	return mat
}

func (*MeanSquaredError) Compute(prediction, actual *mat64.Dense) float64 {
	rows, _ := prediction.Dims()
	error := 0.0

	for i := 0; i < rows; i++ {
		diff := prediction.At(i, 0) - actual.At(i, 0)
		error += math.Pow(diff, 2)
	}
	return error / float64(rows)
}
