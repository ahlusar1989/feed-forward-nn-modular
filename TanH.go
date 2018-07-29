package go_nn_activation_func_experiment

import "math"

type TanH struct {}

var _ ActivationFunc = new(TanH)

func(sigmoidFunc *TanH) Compute(r, c int, val float64) float64{
	return (1 - math.Exp(-2*val)) / (1 + math.Exp(-2*val))
}

func(sigmoidFunc *TanH) Derivative(r, c int, val float64) float64{
	return 1 - (math.Pow((math.Exp(2*val)-1)/(math.Exp(2*val)+1), 2))

}