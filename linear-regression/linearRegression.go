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
)

// Default values
var (
	numberOfCoordinates = 5    // Number of coordinaes to be used
	min                 = 0.0  // Smalles value for a point
	max                 = 10.0 // Largest value for a point
)

// Point structure
type Point struct {
	x float64
	y float64
}

func main() {

	// // // // // //
	// TERMINAL INPUT

	// Read in from the terminal
	reader := bufio.NewReader(os.Stdin)

	// Ask for number of coordinates
	fmt.Print("Number of Coordinates: ")
	input, err := reader.ReadString('\n') // Get the input
	if err != nil {
		log.Fatal(err)
	}
	input = strings.TrimSpace(input) // Remove the '\n' delimiter
	num, err := strconv.Atoi(input)  // Check to see if input is an int
	if err != nil {
		log.Fatal(err)
	}
	numberOfCoordinates = num

	// Ask for file name
	fmt.Print("File Name (\"none\" if none): ")
	input, err = reader.ReadString('\n') // Get the input
	if err != nil {
		log.Fatal(err)
	}
	input = strings.TrimSpace(input)
	fileName := input

	fmt.Println()

	// // //

	// Create some stuff
	coords := make([]Point, numberOfCoordinates)              // These hold the coordinates (x,y)
	meanDiff := make([]Point, numberOfCoordinates)            // These hold the difference of the mean (x-x_mean, y-y_mean)
	meanDiffSquared := make([]Point, numberOfCoordinates)     // These hold the square of the difference of the mean ( (x-x_mean)^2, (y-y_mean)^2 )
	xyMeanDiffProduct := make([]float64, numberOfCoordinates) // These hold the product of the difference of the mean of x and y ( (x-x_mean)(y-y_mean) )
	if fileName == "none" || fileName == "" {
		populateCoords(coords) // Populate the coords with randomized values
	} else {
		coords = load(fileName, 2, coords)
	}

	// Get the xMean and yMean
	xMean, yMean := mean(coords)

	// Populate the meanDiff values
	for i := range meanDiff {
		meanDiff[i].x = coords[i].x - xMean
		meanDiff[i].y = coords[i].y - yMean
	}
	// Populate the meanDiffSquares values
	for i := range meanDiffSquared {
		meanDiffSquared[i].x = math.Pow(meanDiff[i].x, 2)
		meanDiffSquared[i].y = math.Pow(meanDiff[i].y, 2)
	}
	// Pupulate the xyMeanDiffProduct
	for i := range xyMeanDiffProduct {
		xyMeanDiffProduct[i] = meanDiff[i].x * meanDiff[i].y
	}

	// Get the x values from meanDiffSquares
	slice, err := getSlice(meanDiffSquared, 'x')
	if err != nil {
		log.Fatal(err)
	}

	// Create xMeanDiffSquaredSum and xyMeanDiffProductSum
	xMeanDiffSquaredSum := sum(slice)              // The sum of the x values of meanDiffSquared values
	xyMeanDiffProductSum := sum(xyMeanDiffProduct) // The sum of the xyMeanDiffProduct values

	slope := xyMeanDiffProductSum / xMeanDiffSquaredSum // Calculate slope
	yIntercept := yMean - (slope * xMean)               // Calculate y-intercept

	// Print all the things
	fmt.Println("Coords:", coords)
	fmt.Println("Means:", xMean, yMean)
	fmt.Println("meanDiff:", meanDiff)
	fmt.Println("meanDiffSquared:", meanDiffSquared)
	fmt.Println("xyMeanDiffProduct:", xyMeanDiffProduct)
	fmt.Println()
	fmt.Println("Equation: y =", slope, "* x +", yIntercept)

	// Read in from the terminal
	reader = bufio.NewReader(os.Stdin)
	fmt.Print("\nWould you like to predict a point? (y/N): ")
	input, err = reader.ReadString('\n') // Get the input
	if err != nil {
		log.Fatal(err)
	}

	input = strings.TrimSpace(input) // Remove the '\n' delimiter

	if input == "" || input == "N" || input == "n" { // Exit if no
		return
	} else if input == "Y" || input == "y" { // Continue if yes

		for { // Loop until quit
			fmt.Print("\nEnter independent variable ('q' to quit): ")
			input, err := reader.ReadString('\n') // Get te input
			if err != nil {
				log.Fatal(err)
			}

			input = strings.TrimSpace(input) // Remove the '\n' delimiter

			// exits the loop which quits the program
			if input == "q" || input == "Q" {
				break
			}

			num, err := strconv.ParseFloat(input, 64) // Check to see if input is a float
			if err != nil {
				log.Fatal(err)
			}

			dependentVar := slope*num + yIntercept           // Calculate the y value
			fmt.Println("Dependent variable:", dependentVar) // Prints the y value

		}
	}

}

// populateCoords populates coordinates with randomized values
func populateCoords(coords []Point) []Point {
	s1 := rand.NewSource(time.Now().UnixNano()) // Set randomization based on clock time
	r1 := rand.New(s1)                          // Create a new rand

	for i := range coords { // For each point
		coords[i].x = min + r1.Float64()*(max-min) // Randomize x value
		coords[i].y = min + r1.Float64()*(max-min) // Randomize y value
	}

	return coords
}

// sum adds a series of values together
func sum(values []float64) float64 {
	sum := 0.0 // Temporary valu to keep track of the sum

	// Sum it up
	for _, value := range values {
		sum += value
	}

	return sum
}

// pointsSum adds the x values and y values of a series of points
func pointsSum(coords []Point) (float64, float64) {
	// Temporary variables to keep track of the sum
	xSum := 0.0
	ySum := 0.0

	// Sum it up
	for _, point := range coords {
		xSum += point.x
		ySum += point.y
	}

	return xSum, ySum
}

// mean calculates the x and y mean from a series of points
func mean(coords []Point) (float64, float64) {
	xSum, ySum := pointsSum(coords)
	return xSum / float64(len(coords)), ySum / float64(len(coords))
}

// getSlice gets the series of numbers for the x or y values from a series of points
func getSlice(points []Point, axis rune) ([]float64, error) {
	slice := make([]float64, len(points)) // Temporary slice

	if axis == 'x' { // If x axis is chosen
		for i, point := range points { // Get each value from the x-axis and place them in the temporary slice
			slice[i] = point.x
		}
	} else if axis == 'y' { // If y axis is chosen
		for i, point := range points { // Get each value from the y-axis and place them in the temporary slice
			slice[i] = point.y
		}
	} else { // In case the programmer messed up
		return slice, errors.New("axis must be 'x' or 'y'")
	}

	return slice, nil
}

func load(fileName string, fields int, coords []Point) []Point {

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

	// Read in the lines
	for j, record := range rawCSVData {

		// Read in the columns
		for i, val := range record {

			parsedVal, err := strconv.ParseFloat(val, 64) // Convert value to a float
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i == 0 {
				coords[j].x = parsedVal
				continue
			} else if i == 1 {
				coords[j].y = parsedVal
				continue
			}
		}
	}

	return coords
}
