package go_nn_activation_func_experiment

import "github.com/gonum/matrix/mat64"

// Criterion calculates error in predicted value for a given sample
type ErrorCriterion interface {
	// Apply calculates the error given the predicted (first) and
	// actual (second) labels
	Compute(*mat64.Dense, *mat64.Dense) float64
	// Derivative calculates the derivative of the error function (Apply)
	// given the predicted (first) and actual (second) labels
	Derivative(*mat64.Dense, *mat64.Dense) *mat64.Dense
}
