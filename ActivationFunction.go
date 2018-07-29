package go_nn_activation_func_experiment

// ActivationFunc represents the activation function for a given layer
type ActivationFunc interface {
	// Compute calculates the activation of a given layer given
	// previous layer activations
	Compute(int, int, float64) float64
	// Derivative is the calculation used to update weights during backpropagation
	Derivative(int, int, float64) float64
}
