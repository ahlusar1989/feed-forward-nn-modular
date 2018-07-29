package go_nn_activation_func_experiment

import "math"

type ReLU struct{}

var _ ActivationFunc = new(ReLU)

func (rectifier *ReLU) Compute(r, c int, val float64) float64 {
	return math.Log(1 + math.Exp(val))
}

func (rectifier *ReLU) Derivative(r, c int, val float64) float64 {
	return 1.0 / (1.0 + math.Exp(-val))
}
