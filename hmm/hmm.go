package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/kr/pretty"
)

// Default values
var (
	numberOfStates       = 5   // Number of states
	numberOfEmissions    = 5   // Number of emissions
	numberOfObservations = 5   // Number of observations
	min                  = 0.0 // Min value for random number generator
	max                  = 1.0 // Max value for random number generator
)

func main() {

	// // // // // //
	// TERMINAL INPUT

	// Read in from the terminal
	reader := bufio.NewReader(os.Stdin)

	// Ask for number of states
	fmt.Print("Number of states: ")
	input, err := reader.ReadString('\n') // Get the input
	if err != nil {
		log.Fatal(err)
	}
	input = strings.TrimSpace(input) // Remove the '\n' delimiter
	num, err := strconv.Atoi(input)  // Check to see if input is a float
	if err != nil {
		log.Fatal(err)
	}
	numberOfStates = num

	// Ask for number of emissions
	fmt.Print("Number of emissions: ")
	input, err = reader.ReadString('\n') // Get the input
	if err != nil {
		log.Fatal(err)
	}
	input = strings.TrimSpace(input) // Remove the '\n' delimiter
	num, err = strconv.Atoi(input)   // Check to see if input is a float
	if err != nil {
		log.Fatal(err)
	}
	numberOfEmissions = num

	// Ask for number of observations
	fmt.Print("Number of observations: ")
	input, err = reader.ReadString('\n') // Get the input
	if err != nil {
		log.Fatal(err)
	}
	input = strings.TrimSpace(input) // Remove the '\n' delimiter
	num, err = strconv.Atoi(input)   // Check to see if input is a float
	if err != nil {
		log.Fatal(err)
	}
	numberOfObservations = num

	fmt.Println()

	// // //

	// Create the tables
	transitions := make([][]float64, numberOfStates)
	for i := range transitions {
		transitions[i] = make([]float64, numberOfStates)
	}
	emissions := make([][]float64, numberOfStates)
	for i := range transitions {
		emissions[i] = make([]float64, numberOfEmissions)
	}
	probabilities := make([]float64, numberOfStates) // initial probabilities

	// Create the observations
	observations := make([]int, numberOfObservations)
	// Create the path
	mostProbablePath := make([]int, numberOfObservations)

	for i := range observations {
		// Ask for number of observations
		fmt.Print("Observation ", i, ": ")
		input, err = reader.ReadString('\n') // Get the input
		if err != nil {
			log.Fatal(err)
		}
		input = strings.TrimSpace(input) // Remove the '\n' delimiter
		num, err = strconv.Atoi(input)   // Check to see if input is a float
		if err != nil {
			log.Fatal(err)
		}
		observations[i] = num
	}

	// Populate the tables
	transitions = initTransitions(transitions)
	emissions = initEmissions(emissions)
	probabilities = initProbabilities(probabilities)
	mostProbablePath = findMostProbablePath(probabilities, observations, transitions, emissions)

	// Print the tables
	pretty.Println("Transition Matrix:", transitions)
	pretty.Println("Emission Matrix:", emissions)
	pretty.Println("Initial Probabilities:", probabilities)
	pretty.Println("Observations:", observations)
	pretty.Println("Most Probable Path:", mostProbablePath)

}

// initTransitions initializes the transitions with random values
func initTransitions(transitions [][]float64) [][]float64 {
	// Set each value with a random number
	for i := range transitions {
		total := 0.0
		for j := range transitions {
			r := min + rand.Float64()*(max-min) // Get a random number from min to max
			total += r                          // Add that random number to total
			transitions[i][j] = r               // Assign that random number to transitions
		}
		// Normalize the values so they all sum up to 1
		for j := range transitions {
			transitions[i][j] /= total
		}
	}
	return transitions
}

// initEmissions initializes the emmissions with random values
func initEmissions(emissions [][]float64) [][]float64 {
	// Set each value with a random number
	for i := range emissions {
		total := 0.0
		for j := range emissions[i] {
			r := min + rand.Float64()*(max-min) // Get a random number from min to max
			// r := float64(rand.Intn(2))
			total += r          // Add that random number to the total
			emissions[i][j] = r // Assign that random number to emissions
		}
		// // Normalize the values so they all sum up to 1
		// for j := range emissions {
		// 	emissions[i][j] /= total
		// }
	}
	return emissions
}

// initProbabilities initalizes the probablities with random values
func initProbabilities(probabilities []float64) []float64 {
	total := 0.0
	// Set each value with a random number
	for i := range probabilities {
		r := min + rand.Float64()*(max-min) // Get a random number from min to max
		total += r                          // Add that random number to the total
		probabilities[i] = r                // Assign that random number to emissins
	}
	// Normalize the values so they all sum up to 1
	for i, prob := range probabilities {
		prob /= total
		probabilities[i] = prob
	}
	return probabilities
}

// findMostProbablePath finds most probable path using Viterbi algorithm
func findMostProbablePath(probabilities []float64, observations []int, transitions [][]float64, emissions [][]float64) []int {

	// probabilities of the most likely path so far
	probs := make([][]float64, numberOfStates)
	for i := range probs {
		probs[i] = make([]float64, numberOfObservations)
	}
	// previous probabilities of the most likely path so far
	prevs := make([][]float64, numberOfStates)
	for i := range prevs {
		prevs[i] = make([]float64, numberOfObservations)
	}

	for i := 0; i < numberOfStates; i++ { // For each state
		probs[i][0] = probabilities[i] * emissions[i][observations[0]] // Calculates probability & place it in probs: p_i * e_(i,o_0)
		prevs[i][0] = 0                                                // Set previous to 0 since we dont have a previous yet
	}

	for j := 1; j < numberOfObservations; j++ { // For each observation
		for i := 0; i < numberOfStates; i++ { // for each state
			probs[i][j] = probs[0][j-1] * transitions[0][i] * emissions[i][observations[j]] // Calculates probability & place it in probs: p_(0,j-1) * t_(0,i) * e_(i,o_0)
			prevs[i][j] = 0                                                                 // Set previous to 0 since we dont have a previous yet

			for k := 1; k < numberOfStates; k++ {
				prob := probs[k][j-1] * transitions[k][i] * emissions[i][observations[j]] // Calculates probablity & place it in probls:  p_(k,j-1) * t_(k,i) * e_(i,o_j)
				if probs[i][j] < prob {                                                   // If the probability is greater than the assigned probability
					probs[i][j] = prob       // Assign the probabilty
					prevs[i][j] = float64(k) // Take the previous index
				}
			}

		}
	}

	// Figure out the sequence of max probabilities
	maxProb := make([]int, numberOfObservations)
	last := numberOfObservations - 1 // Last prob index
	maxProb[last] = 0                // Assign last index to 0
	max := probs[0][last]            // Assign the max
	for k := 1; k < numberOfStates; k++ {
		if max < probs[k][last] { // If there a value greater than max, assign it to max
			maxProb[last] = k // Also assign k index to the last index of maxProb
			max = probs[k][last]
		}
	}

	// Assign the maxProb values from last to first
	for i := last; i >= 1; i-- {
		maxProb[i-1] = int(prevs[maxProb[i]][i])
	}

	return maxProb
}

// func load(fileName string, fields int) {

// 	// Open file
// 	f, err := os.Open(fileName)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	defer f.Close() // CLose when funcion exits

// 	reader := csv.NewReader(f)      // Create new reader
// 	reader.FieldsPerRecord = fields // Assign number of fields

// 	// Read in the data
// 	rawCSVData, err := reader.ReadAll()
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	inputsData := make([]float64, fields*len(rawCSVData)) // gets the first 4 values
// 	labelsData := make([]float64, 3*len(rawCSVData))      // get the last 3 values

// 	// For tracking the current index of matrix values.
// 	var inputsIndex int
// 	var labelsIndex int

// 	// Read in the rows
// 	for _, record := range rawCSVData {

// 		// Read in the columns
// 		for i, val := range record {

// 			parsedVal, err := strconv.ParseFloat(val, 64) // Convert value to a float
// 			if err != nil {
// 				log.Fatal(err)
// 			}

// 			// Add to the labelsData if relevant.
// 			if i == 4 || i == 5 || i == 6 {
// 				labelsData[labelsIndex] = parsedVal
// 				labelsIndex++
// 				continue
// 			}

// 			// Add the float value to the slice of floats.
// 			inputsData[inputsIndex] = parsedVal
// 			inputsIndex++
// 		}
// 	}
// 	// Assigning the values to the created matrices
// 	// inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
// 	// labels := mat.NewDense(len(rawCSVData), 3, labelsData)
// }
