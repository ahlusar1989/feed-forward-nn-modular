package go_nn_activation_func_experiment_test

import (
	"log"
	"go-nn-activation-func-experiment"
)

var (
	a = character(
		".#####." +
			"#.....#" +
			"#.....#" +
			"#######" +
			"#.....#" +
			"#.....#" +
			"#.....#")
	b = character(
		"######." +
			"#.....#" +
			"#.....#" +
			"######." +
			"#.....#" +
			"#.....#" +
			"######.")
	c = character(
		"#######" +
			"#......" +
			"#......" +
			"#......" +
			"#......" +
			"#......" +
			"#######")
)

func ExampleLearn() {
	m := go_nn_activation_func_experiment.NewNetwork(0.3, 10000, []int{49, 3, 1},
		[]go_nn_activation_func_experiment.ActivationFunc{new(go_nn_activation_func_experiment.Sigmoid)},
	)
	m.Learn([][][]float64{
		{c, []float64{.5}},
		{b, []float64{.3}},
		{a, []float64{.1}},
	})

	result := m.Predict(
		character(
			"#######" +
				"#......" +
				"#......" +
				"#......" +
				"#......" +
				"#......" +
				"#######"))
	log.Println(result)
}

func character(chars string) []float64 {
	flt := make([]float64, len(chars))
	for i := 0; i < len(chars); i++ {
		if chars[i] == '#' {
			flt[i] = 1.0
		} else { // if '.'
			flt[i] = 0.0
		}
	}
	return flt
}