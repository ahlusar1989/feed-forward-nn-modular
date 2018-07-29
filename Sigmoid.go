package go_nn_activation_func_experiment

import "math"

type Sigmoid struct {}

var _ ActivationFunc = new(Sigmoid)

func(sigmoidFunc *Sigmoid) Compute(r, c int, val float64) float64{
	return 1 / (1 + math.Exp(-val))
}

func(sigmoidFunc *Sigmoid) Derivative(r, c int, val float64) float64{
	phi := sigmoidFunc.Compute(r, c, val)
	return phi * (1 - phi)
}