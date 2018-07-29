package go_nn_activation_func_experiment

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math/rand"
	"time"
)

// NeuralNetwork struct represents the neural network
type NeuralNetwork struct {
	ActivationFunctions []ActivationFunc
	Activations         []*mat64.Dense
	Weights             []*mat64.Dense
	Errors              []*mat64.Dense
	//Topology specifies number of hidden layers and nodes in each, as well as
	// the size of samples and labels (first and last values, respectively).
	// The expectation is that each layer specified has another activation function
	Topology     []int
	Layers       int
	LearningRate float64
	Iterations   int
	Loss         ErrorCriterion
	currentErr   float64
	// logging output during training
	Verbose bool
}

func NewNetwork(learnRate float64, iterations int, topology []int,
	activations []ActivationFunc) *NeuralNetwork {
	newNet := &NeuralNetwork{
		LearningRate:        learnRate,
		Iterations:          iterations,
		ActivationFunctions: make([]ActivationFunc, len(topology)),
		Activations:         make([]*mat64.Dense, len(topology)),
		Errors:              make([]*mat64.Dense, len(topology)),
		Weights:             make([]*mat64.Dense, len(topology)-1),
		Topology:            topology,
		Layers:              len(topology),
		Loss:                new(MeanSquaredError),
		Verbose:             true,
	}
	newNet.initializeActivations(topology)
	newNet.initializeErrors(topology)
	newNet.initializeWeights(topology)
	newNet.initializeActivators(activations)

	return newNet
}

func (network *NeuralNetwork) logging(a ...interface{}) {
	if network.Verbose {
		fmt.Println(a...)
	}
}

// Predict returns the predicted value of the provided sample
// Dimensions must match those from provided topology
// Only use after training the network

func (network *NeuralNetwork) Predict(sample []float64) *mat64.Dense {
	network.ForwardPass(sample)
	return network.Activations[network.Layers-1]
}

// Learn trains the network using the provided dataset
// Samples must have number of features and labels as specified by topology
// when constructing the network
func (network *NeuralNetwork) Learn(dataset [][][]float64) {
	network.logging("Starting to learn...")

	for i := 0; i < network.Iterations; i++ {
		network.logging("Iteration ", i+1)
		network.currentErr = 0
		for _, sample := range dataset {
			network.ForwardPass(sample[0])
			network.BackPropagate(sample[1])
		}
		network.logging("Error after %v", network.currentErr/float64(len(dataset)))
	}
}

func (network *NeuralNetwork) initializeActivations(topology []int) {
	for i, nodes := range topology {
		network.Activations[i] = mat64.NewDense(nodes, 1, nil)
	}
}

func (network *NeuralNetwork) initializeErrors(topology []int) {
	for i := 0; i < network.Layers; i++ {
		network.Errors[i] = mat64.NewDense(topology[i], 1, nil)
	}
}

func (network *NeuralNetwork) initializeWeights(topology []int) {
	for i := 0; i < network.Layers; i++ {
		network.Weights[i] = randomMatrix(topology[i+1], topology[i])
	}
}

func (n *NeuralNetwork) initializeActivators(acts []ActivationFunc) {
	acts = append([]ActivationFunc{new(LinearActivationFunc)},
		append(acts, new(LinearActivationFunc))...)

	for i := 0; i < len(acts); i++ {
		n.ActivationFunctions[i] = acts[i]
	}
}

// Forward calculates activations at each layer for given sample
func (network *NeuralNetwork) ForwardPass(sample []float64) {
	network.Activations[0].SetCol(0, sample)

	for layer := 0; layer < len(network.Weights); layer++ {
		network.Activations[layer+1].Mul(network.Weights[layer], network.Activations[layer])
		network.Activations[layer+1].Apply(
			network.ActivationFunctions[layer+1].Compute,
			network.Activations[layer+1],
		)
	}
}

func (network *NeuralNetwork) BackPropagate(label []float64) {
	network.calculateErrors(label)
	network.updateWeights()
}

func (network *NeuralNetwork) calculateErrors(label []float64) {
	actualValues := mat64.NewDense(len(label), 1, label)
	network.Errors[network.Layers-1] = network.Loss.Derivative(
		network.Activations[network.Layers-1], actualValues,
	)
	network.currentErr += network.Loss.Compute(network.Activations[network.Layers-1], actualValues)
	for i := network.Layers - 2; i >= 0; i-- {
		network.calculateErrorForCurrentLayer(i)
	}
}

func (network *NeuralNetwork) updateWeights() {
	for i := 0; i < network.Layers-1; i++ {
		mat := &mat64.Dense{}
		mat.Mul(network.Errors[i+1], network.Activations[i].T())
		mat.Scale(network.LearningRate, mat)
		network.Weights[i].Sub(network.Weights[i], mat)
	}
}

func (network *NeuralNetwork) calculateErrorForCurrentLayer(currentLayer int) {
	network.Errors[currentLayer].Mul(network.Weights[currentLayer].T(),
		network.Errors[currentLayer+1],
	)
	network.Errors[currentLayer].MulElem(network.Errors[currentLayer],
		network.Activations[currentLayer],
	)

	outputMat := &mat64.Dense{}
	outputMat.Apply(network.ActivationFunctions[currentLayer].Derivative,
		network.Activations[currentLayer],
	)

	network.Errors[currentLayer].MulElem(outputMat, network.Errors[currentLayer])
}

// Randomly generate weights
// : TODO move to utils package
func randomMatrix(rows, columns int) *mat64.Dense {
	rand.Seed(time.Now().UnixNano())
	data := make([]float64, rows*columns)
	for i := range data {
		data[i] = rand.Float64() / 2.0
	}
	return mat64.NewDense(rows, columns, data)
}
