package go_nn_activation_func_experiment

//Linear Activation Function
type LinearActivationFunc struct {
}

var _ ActivationFunc = new(LinearActivationFunc)

// Apply calculates activation for layer given the previous layer value
func (*LinearActivationFunc) Compute(r, c int, value float64) float64 {
	return value
}

func (*LinearActivationFunc) Derivative(r, c int, value float64) float64 {
	return 1
}
