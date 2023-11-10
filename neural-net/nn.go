package main

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/kr/pretty"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Parameters for network structure
type networkConf struct {
	numberOfInputNodes  int     // Number of input nodes
	numberOfOutputNodes int     // Number of outputs nodes
	numberOfHiddenNodes int     // Number of hidden nodes
	numberOfEpochs      int     // Number of iterations to train
	learningRate        float64 // Learning rate helps the network learning converge faster or slower
}

// network structure
type network struct {
	config        networkConf // Config struct
	hiddenWeights *mat.Dense  // Matrix of hiddenWights
	hiddenBiases  *mat.Dense  // Matrix of hiddenBiases
	outputWeights *mat.Dense  // Matrix of outputWeights
	outputBiases  *mat.Dense  // Matrix of outputBiases
}

var (
	testInputs *mat.Dense
	testLabels *mat.Dense
)

func main() {

	// // // // // //
	// TERMINAL INPUT

	// Read in from the terminal
	reader := bufio.NewReader(os.Stdin)

	// Ask for number of hidden nodes
	fmt.Print("Number of Hidden Nodes: ")
	input, err := reader.ReadString('\n') // Get the input
	if err != nil {
		log.Fatal(err)
	}
	input = strings.TrimSpace(input) // Remove the '\n' delimiter
	num, err := strconv.Atoi(input)  // Check to see if input is an int
	if err != nil {
		log.Fatal(err)
	}
	numberOfHiddenNodes := num

	// Ask for bias
	fmt.Print("Bias: ")
	input, err = reader.ReadString('\n') // Get the input
	if err != nil {
		log.Fatal(err)
	}
	input = strings.TrimSpace(input)               // Remove the '\n' delimiter
	numFloat, err := strconv.ParseFloat(input, 64) // Check to see if input is a float
	if err != nil {
		log.Fatal(err)
	}
	bias := numFloat

	// // Ask for file name
	// fmt.Print("File Name (\"none\" if none): ")
	// input, err = reader.ReadString('\n') // Get the input
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// input = strings.TrimSpace(input)
	// fileName := input

	fmt.Println()

	// // //

	// Generate some training & testing data
	// generateData("trainingData.csv", 100)
	// generateData("testingData.csv", 100)

	// Load the training matrices from a file
	inputs, labels := load("trainingData.csv", 7)

	// Create new network with config
	network := network{config: networkConf{
		numberOfInputNodes:  4,
		numberOfOutputNodes: 3,
		numberOfHiddenNodes: numberOfHiddenNodes,
		numberOfEpochs:      100,
		learningRate:        0.1,
	}}

	// Train the neural network
	err = network.train(inputs, labels, bias)
	if err != nil {
		log.Fatal(err)
	}

	// Load the testing matrices
	testInputs, testLabels = load("trainingData.csv", 7)

	// Create outputs after training
	outputs, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate the accuracy
	var hit int                          // For calculating accuracy
	numberOfOutputs, _ := outputs.Dims() // Gets the dimension of the outputs AKA number of outputs

	for i := 0; i < numberOfOutputs; i++ { // Iterate through number of outputs

		labelRow := mat.Row(nil, i, testLabels) // Get the labels from the test data
		var prediction int                      // For calculating accuracy
		for ix, label := range labelRow {       // Get labels == 1.0 and assign them to prediction
			if label == 1.0 {
				prediction = ix
				break
			}
		}

		// Total up the values
		if outputs.At(i, prediction) == floats.Max(mat.Row(nil, i, outputs)) {
			hit++
		}
	}

	// Calculates accuracy: hit / total
	accuracy := float64(hit) / float64(numberOfOutputs)

	// Print some stuff
	fmt.Printf("hiddenWeights: % v\n", mat.Formatted(network.hiddenWeights, mat.Prefix("               ")))
	fmt.Printf("\nhiddenBiases: % v\n", mat.Formatted(network.hiddenBiases, mat.Prefix("          ")))
	fmt.Printf("\noutputWeights: % v\n", mat.Formatted(network.outputWeights, mat.Prefix("               ")))
	fmt.Printf("\noutputBiases: % v\n", mat.Formatted(network.outputBiases, mat.Prefix("        ")))
	fmt.Printf("\noutputs: % v\n", mat.Formatted(outputs, mat.Prefix("         ")))
	fmt.Println("\nFinal accuracy:", accuracy)
}

// train trains a neural network using backpropagation.
func (network *network) train(inputs *mat.Dense, labels *mat.Dense, bias float64) error {

	// Randomization for wights & biases
	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)

	// For biases
	hiddenBiasesRaw := make([]float64, network.config.numberOfHiddenNodes)
	for i := range hiddenBiasesRaw {
		hiddenBiasesRaw[i] = bias
	}
	outputBiasesRaw := make([]float64, network.config.numberOfOutputNodes)
	for i := range outputBiasesRaw {
		outputBiasesRaw[i] = bias
	}

	// The hidden & output weights/biases (initialized to nil)
	hiddenWeights := mat.NewDense(network.config.numberOfInputNodes, network.config.numberOfHiddenNodes, nil)
	hiddenBiases := mat.NewDense(1, network.config.numberOfHiddenNodes, hiddenBiasesRaw)
	outputWeights := mat.NewDense(network.config.numberOfHiddenNodes, network.config.numberOfOutputNodes, nil)
	outputBiases := mat.NewDense(1, network.config.numberOfOutputNodes, outputBiasesRaw)

	// Slices for the hidden & output weights/biases (initialized to 0)
	hiddenWeightsRaw := hiddenWeights.RawMatrix().Data
	outputWeightsRaw := outputWeights.RawMatrix().Data

	// Assign the hidden/output weights/biases slices to randomized values
	for _, param := range [][]float64{hiddenWeightsRaw, outputWeightsRaw} {
		for i := range param {
			param[i] = r1.Float64()
		}
	}

	// Create output matrix
	output := new(mat.Dense)

	// Backwards propagation for adjusting weights/biases
	if err := network.propagate(inputs, labels, hiddenWeights, hiddenBiases, outputWeights, outputBiases, output); err != nil {
		return err
	}

	// Assign the slices to the neural network
	network.hiddenWeights = hiddenWeights
	network.hiddenBiases = hiddenBiases
	network.outputWeights = outputWeights
	network.outputBiases = outputBiases

	return nil
}

// propagate handles the backwards propagation for adjusting the weights and biases
func (network *network) propagate(inputs, labels, hiddenWeights, hiddenBiases, outputWeights, outputBiases, output *mat.Dense) error {

	// Loop through the number of epochs
	for i := 0; i < network.config.numberOfEpochs; i++ {

		// // // // // // // //
		// Forward propagation

		// Hidden layer inputs
		hiddenLayerInput := new(mat.Dense)                       // Create new hiddenLayerInput matrix
		hiddenLayerInput.Mul(inputs, hiddenWeights)              // Multiply matrices inputs and hiddenWeights
		addhiddenBiases := func(_, col int, v float64) float64 { // Adds the hidden biases
			return v + hiddenBiases.At(0, col)
		}
		hiddenLayerInput.Apply(addhiddenBiases, hiddenLayerInput) // Applies the addition to each element in hiddenLayerInput

		// Hidden layer activations
		hiddenLayerActivations := new(mat.Dense)            // Create new hiddenLayerActivations matrix
		applySigmoid := func(_, _ int, v float64) float64 { // Use the sigmoid function
			return sigmoid(v)
		}
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput) // Apply the sigmoid function to each element in hidddenLayerInput

		// Output layer inputs
		outputLayerInput := new(mat.Dense)                          // Create new outputLayerInput matrix
		outputLayerInput.Mul(hiddenLayerActivations, outputWeights) // Multiply matrices hiddenLayerActivations and outputWeights
		addoutputBiases := func(_, col int, v float64) float64 {    // Adds the output biases
			return v + outputBiases.At(0, col)
		}
		outputLayerInput.Apply(addoutputBiases, outputLayerInput) // Applies the addition to each element in outputLayerInput
		output.Apply(applySigmoid, outputLayerInput)              // Applies te sigmoid function to the outputLayerInput matrix, output of forward propagation

		// // // // // // // //
		// Backward propagation

		// Calculate the difference of values within the network
		networkError := new(mat.Dense)   // Create the networkError matrix
		networkError.Sub(labels, output) // Subtract outputs from labels and place them in networkError
		// networkErrorAbs := func(_, _ int, v float64) float64 {
		// 	return abs(v)
		// }
		// networkError.Apply(networkErrorAbs, networkError)
		pretty.Println("NETWORK ERROR", networkError)
		// break

		slopeOutputLayer := new(mat.Dense)                       // Create new slopeOutputLayer matrix
		applySigmoidPrime := func(_, _ int, v float64) float64 { // Uses the sigmoidPrime function
			return sigmoidPrime(v)
		}
		slopeOutputLayer.Apply(applySigmoidPrime, output) // Applies the sigmoidPrime function to each element of output

		slopeHiddenLayer := new(mat.Dense)                                // Create the slopeHiddenLayer matrix
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations) // Apply sigmoidPrime function to each element of hiddenLayerActivations

		outputDifference := new(mat.Dense)                       // Create new outputDifference matrix
		outputDifference.MulElem(networkError, slopeOutputLayer) // Multiply each element of networkError and slopeOutputLayer and place them in outputDifference

		errorAtHiddenLayer := new(mat.Dense)                        // Create new errorAtHiddenLayer matrix
		errorAtHiddenLayer.Mul(outputDifference, outputWeights.T()) // Multiply the outputDifference matrix and the transpose of the outputWeights matrix into errorAtHiddenLayer

		hiddenLayerDifference := new(mat.Dense)                             // Create hiddenLayerDifference
		hiddenLayerDifference.MulElem(errorAtHiddenLayer, slopeHiddenLayer) // Multiply each element of errorAtHiddenLayer and slopeHiddenLayer and place them in hiddenLayerDifference

		// // // // // // // //
		// Adjust the weights

		// Adjust output weights
		outputWeightsAdj := new(mat.Dense)                                    // Create new outputWeightsAdj matrix
		outputWeightsAdj.Mul(hiddenLayerActivations.T(), outputDifference)    // Multiply the transpose of the hiddenLayerActivations matrix and outputDifference matrices and place them in outputWeightsAdj
		outputWeightsAdj.Scale(network.config.learningRate, outputWeightsAdj) // Scale the matrix using a learning rate
		outputWeights.Add(outputWeights, outputWeightsAdj)                    // Add the weights

		// Adjust hidden weights
		hiddenWeightsAdj := new(mat.Dense)                                    // Create new hiddenWeightsAdj matrix
		hiddenWeightsAdj.Mul(inputs.T(), hiddenLayerDifference)               // Multiply the transpose of inputs an hiddenlayerDifference and place them in hiddenWeightsAdj
		hiddenWeightsAdj.Scale(network.config.learningRate, hiddenWeightsAdj) // Scale the hidden weights using a learning rate
		hiddenWeights.Add(hiddenWeights, hiddenWeightsAdj)                    // Add hidden weights

	}

	return nil
}

// sumAlongAxis sums a matrix along a particular dimension while preserving the other dimension
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims() // Get the dimensions of m

	var output *mat.Dense // This is the return value

	switch axis {
	case 0: // 0th axis - columns
		data := make([]float64, numCols) // Create a new data slice with the number of columns
		for i := 0; i < numCols; i++ {   // Loop through each column
			col := mat.Col(nil, i, m) // Copies the specified column from the matrix
			data[i] = floats.Sum(col) // Add the values
		}
		output = mat.NewDense(1, numCols, data)
	case 1: // 1st axis - rows
		data := make([]float64, numRows) // Create a new data slice with the number of rows
		for i := 0; i < numRows; i++ {   // Loop through each row
			row := mat.Row(nil, i, m) // Copies the specified row from the matrix
			data[i] = floats.Sum(row) // Add the values
		}
		output = mat.NewDense(numRows, 1, data) // Assign the return value
	default:
		return nil, errors.New("invalid axis: must be 0 or 1") // Error
	}

	return output, nil
}

// predict makes an output prediction
func (network *network) predict(x *mat.Dense) (*mat.Dense, error) {

	// Checks for nil values
	if network.hiddenWeights == nil || network.outputWeights == nil { // For weights
		return nil, errors.New("the weights are empty")
	}
	if network.hiddenBiases == nil || network.outputBiases == nil { // For biases
		return nil, errors.New("the biases are empty")
	}

	output := new(mat.Dense) // Create a new output matrix

	// // // // /// // // //
	// Forward propagation

	// Hidden layer inputs
	hiddenLayerInput := new(mat.Dense)             // Create new hiddenLayerInput matrix
	hiddenLayerInput.Mul(x, network.hiddenWeights) // Multiply x and the network;s hidden weights together

	// Hidden Layer biases
	addhiddenBiases := func(_, col int, v float64) float64 { // Function for adding hidden biases
		return v + network.hiddenBiases.At(0, col)
	}
	hiddenLayerInput.Apply(addhiddenBiases, hiddenLayerInput) // Applies the function

	// Hidden layer activations
	hiddenLayerActivations := new(mat.Dense)            // Create new hiddenLayerActiviations matrix
	applySigmoid := func(_, _ int, v float64) float64 { // Sigmoid function
		return sigmoid(v)
	}
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput) // Apply the function

	// Output layer input
	outputLayerInput := new(mat.Dense)                                  // Create new outputLayerInput matrix
	outputLayerInput.Mul(hiddenLayerActivations, network.outputWeights) // Multiply the hidden layer activations by the output weights

	// Output biases
	addoutputBiases := func(_, col int, v float64) float64 { // Function for adding output biases
		return v + network.outputBiases.At(0, col)
	}
	outputLayerInput.Apply(addoutputBiases, outputLayerInput) // Apply the function
	output.Apply(applySigmoid, outputLayerInput)              // Apply the sigmoid function for each output layer input

	//

	return output, nil
}

// sigmoid is the sigmoid function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime is the derivative of the sigmoid function
func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func abs(x float64) float64 {
	return math.Abs(x)
}

func calcAccuracy(testInputs, testLabels *mat.Dense) {

}

// load loads a file and palces each file in 2 matrices
func load(fileName string, fields int) (*mat.Dense, *mat.Dense) {

	// Open file
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close() // CLose when funcion exits

	reader := csv.NewReader(f)      // Create new reader
	reader.FieldsPerRecord = fields // Assign number of fields

	// Read in the data
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	inputsData := make([]float64, 4*len(rawCSVData)) // gets the first 4 values
	labelsData := make([]float64, 3*len(rawCSVData)) // get the last 3 values

	// For tracking the current index of matrix values.
	var inputsIndex int
	var labelsIndex int

	// Read in the rows
	for _, record := range rawCSVData {

		// Read in the columns
		for i, val := range record {

			parsedVal, err := strconv.ParseFloat(val, 64) // Convert value to a float
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	// Assigning the values to the created matrices
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)

	return inputs, labels
}

// generateData generates random data and saves it to a file using a file name
func generateData(fileName string, num int) {

	// Create new file
	f, err := os.Create(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close() // Close when function finishes

	// Variables for each indiviual piece of data
	var (
		data string // The combined data as one long string
		i1   string
		i2   string
		i3   string
		i4   string
		l1   string
		l2   string
		l3   string
	)

	// s1 := rand.NewSource(time.Now().UnixNano()) // Set randomization based on clock time
	// r1 := rand.New(s1)                          // Create a new rand

	// Min and max values for the random number generator
	min := 0.0
	max := 1.0

	// Generate the random data
	for i := 0; i < num; i++ {
		i1 = strconv.FormatFloat(min+rand.Float64()*(max-min), 'f', 6, 64)
		i2 = strconv.FormatFloat(min+rand.Float64()*(max-min), 'f', 6, 64)
		i3 = strconv.FormatFloat(min+rand.Float64()*(max-min), 'f', 6, 64)
		i4 = strconv.FormatFloat(min+rand.Float64()*(max-min), 'f', 6, 64)
		l1 = strconv.Itoa(rand.Intn(2))
		l2 = strconv.Itoa(rand.Intn(2))
		l3 = strconv.Itoa(rand.Intn(2))
		data += i1 + "," + i2 + "," + i3 + "," + i4 + "," + l1 + "," + l2 + "," + l3 + "\n" // Place the data in a giant string
	}

	// Write the data
	_, err = f.WriteString(data)
	if err != nil {
		log.Fatal(err)
	}

}
