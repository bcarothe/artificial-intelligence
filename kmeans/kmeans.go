// ================================================================================
//
// kmeans.go
// Provides a set of clusters based on a set of a randomized points
//
// Usage:
// go run kmeans.go [numberOfPoints] [numberOfClusters] [iterations] [rangeMin] [rangeMax] [threshold]
//
// ================================================================================

package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

// Point stuct
type Point struct {
	x float64
	y float64
}

// Cluster struct
type Cluster struct {
	centroid Point
	points   []Point
}

// Default values
var (
	numberOfPoints   = 5     // Number of points to use
	numberOfClusters = 3     // Number of clusters desired
	iteratoins       = 5     // Number of iterations before stopping
	rangeMin         = 0     // Minimum value for a point using random number generator
	rangeMax         = 10    // Maximum value for a point using random number generator
	threshold        float64 // The threshold for points to be considered within a cluster
)

func main() {

	// // // // // //
	// TERMINAL INPUT

	// Read in from the terminal
	reader := bufio.NewReader(os.Stdin)

	// Ask for number of points
	fmt.Print("Number of Points: ")
	input, err := reader.ReadString('\n') // Get the input
	if err != nil {
		log.Fatal(err)
	}
	input = strings.TrimSpace(input) // Remove the '\n' delimiter
	num, err := strconv.Atoi(input)  // Check to see if input is an int
	if err != nil {
		log.Fatal(err)
	}
	numberOfPoints = num

	// Ask for number of clusters
	fmt.Print("Number of Clusters: ")
	input, err = reader.ReadString('\n') // Get the input
	if err != nil {
		log.Fatal(err)
	}
	input = strings.TrimSpace(input) // Remove the '\n' delimiter
	num, err = strconv.Atoi(input)   // Check to see if input is an int
	if err != nil {
		log.Fatal(err)
	}
	numberOfClusters = num

	// Ask for iterations
	fmt.Print("Number of Iterations: ")
	input, err = reader.ReadString('\n') // Get the input
	if err != nil {
		log.Fatal(err)
	}
	input = strings.TrimSpace(input) // Remove the '\n' delimiter
	num, err = strconv.Atoi(input)   // Check to see if input is an int
	if err != nil {
		log.Fatal(err)
	}
	iteratoins = num

	// Ask for threshold
	fmt.Print("Number of Threshold: ")
	input, err = reader.ReadString('\n') // Get the input
	if err != nil {
		log.Fatal(err)
	}
	input = strings.TrimSpace(input)               // Remove the '\n' delimiter
	numFloat, err := strconv.ParseFloat(input, 64) // Check to see if input is a float
	if err != nil {
		log.Fatal(err)
	}
	threshold = numFloat

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

	// Create points & clusters slices with a defined size
	points := make([]Point, numberOfPoints)
	clusters := make([]Cluster, numberOfClusters)

	// Initialize or load points & clusters with data
	clusters = initClusters(clusters, rangeMin, rangeMax)

	if fileName == "none" || fileName == "" {
		points = initPoints(points, rangeMin, rangeMax)
	} else {
		points = load(fileName, 2, points)
	}

	// Print some stuff
	fmt.Println("Points:", points)
	fmt.Println("Initial Clusters:", clusters)
	fmt.Println()

	// Loop through iterations
	for i := iteratoins; i > 0; i-- {

		previousClusters := clusters

		// fmt.Println("Iteratoins remain:", i) // Print iterations

		clusters = updateClusters(clusters, points)
		// fmt.Println("Clusters:", clusters)

		clusters = updateCentroid(clusters)

		changedCentroids := 0
		for i, cluster := range clusters {
			if cluster.centroid == previousClusters[i].centroid {
				changedCentroids++
			}
			fmt.Println("New cluster centroid:", cluster.centroid) // Print the new centroids
		}
		if changedCentroids == numberOfClusters {
			fmt.Println("\nEnding iterations sooner due to unchanged centroids")
			fmt.Println()
			break
		}

		fmt.Println()
	}

	// fmt.Println("Final clusters:", clusters)
	fmt.Printf("%+v\n", clusters)

	outliers := findOutliers(clusters, points)
	fmt.Println("Outliers:", outliers)

}

// initPoints sets a defined number of points randomly based on min and max values
func initPoints(points []Point, min int, max int) []Point {

	s1 := rand.NewSource(time.Now().UnixNano()) // Set randomization based on clock time
	r1 := rand.New(s1)                          // Create a new rand

	// Set the x and y coordinate for each point
	for i, point := range points {
		point.x = float64(r1.Intn(max+1-min) + min) // Assign a random value for x
		point.y = float64(r1.Intn(max+1-min) + min) // Assign a random value for y
		points[i] = point                           // Assign the point to the slice of points
	}

	return points
}

// initClusters sets a defined number of clusters randomly based on the min and max values (so that the clusters aren't too far from the points)
func initClusters(clusters []Cluster, min int, max int) []Cluster {

	s1 := rand.NewSource(time.Now().UnixNano()) // Set randomization based on clock time
	r1 := rand.New(s1)                          // Create a new rand

	// Set the x and y coordinate for each cluster point
	for i, cluster := range clusters {
		cluster.centroid.x = float64(r1.Intn(max+1-min) + min) // Assign a random value for x
		cluster.centroid.y = float64(r1.Intn(max+1-min) + min) // Assign a random value for y
		clusters[i].centroid = cluster.centroid                // Assign the cluster point to the centroid of the slice of clustes
	}

	return clusters
}

// updateCluster finds and assigns the points to each cluster
func updateClusters(clusters []Cluster, points []Point) []Cluster {

	distances := make([]float64, numberOfClusters) // Ceate a distances slice
	newClusters := make([]Cluster, numberOfClusters)

	for i, cluster := range newClusters {
		newClusters[i].centroid = cluster.centroid
	}

	// For each point in points find the distance between the cluster centroids
	for _, point := range points {
		for i, cluster := range clusters {
			// fmt.Println("i value", i)

			distances[i] = findDistance(cluster.centroid, point) // Get the distance for each cluster centroid for the specified point and assign it in distances
			// fmt.Println("centroid:", cluster.centroid, "- point:", point, "- distance:", distances[i]) // Print some stuff

			// If all distances to each cluster centroid has been found then continue
			if i == numberOfClusters-1 {
				index := findLeastDistanceIndex(distances) // Assign a temporary index to the index containing the shortest distance
				// fmt.Println("least distance index:", findLeastDistanceIndex(distances)) // Print some more stuff

				if index != -1 { // If no shorter index has been found (aka no outliers)
					newClusters[index].points = append(newClusters[index].points, point) // Assign that point to the cluster containing the closer centroid

				}
			}
		}
	}

	return newClusters
}

// updateCentroid finds the new centroid of a cluster based on the points
func updateCentroid(clusters []Cluster) []Cluster {

	for i, cluster := range clusters { // For each cluster

		if len(cluster.points) == 0 {
			break
		}

		// Initialize a couple temporary variables
		xSum := 0.0
		ySum := 0.0

		// For each point in the cluster, add up the x and y values
		for _, point := range cluster.points {
			xSum += point.x
			ySum += point.y
		}

		// Get the average of the x and y values and reassign them
		if len(cluster.points) != 0 {
			cluster.centroid.x = xSum / float64(len(cluster.points))
			cluster.centroid.y = ySum / float64(len(cluster.points))
			clusters[i].centroid = cluster.centroid
		}

	}

	return clusters
}

// findDistance calculates the Euclidean distance between 2 points
func findDistance(p1 Point, p2 Point) float64 {
	return math.Sqrt((math.Pow(float64(p1.x)-float64(p2.x), 2.0)) + (math.Pow(float64(p1.y)-float64(p2.y), 2.0)))
}

// findLeastDistanceIndex finds the smallest number in a slice and returns its index
func findLeastDistanceIndex(distances []float64) int {

	var leastValue float64 // Temporary value
	var leastIndex = -1    // Set the leastIndex to -1 just in case it cant find a distance within the threshold

	for i, distance := range distances { // Go through each distance

		if i == 0 || distance < leastValue {

			if threshold >= 0 && distance <= threshold { // This applies if there is a threshold value
				leastValue = distance
				leastIndex = i
			} else if threshold == 0 { // This applies for a non threshold value
				leastValue = distance
				leastIndex = i
			}
		}
	}

	return leastIndex
}

func findOutliers(clusters []Cluster, points []Point) []Point {

	outliers := make([]Point, 0)

	for _, point := range points {

		isFound := false

		for _, cluster := range clusters {

			for _, cPoint := range cluster.points {
				if cPoint == point {
					isFound = true
					break
				}
			}

			if isFound {
				break
			}

		}

		if !isFound {
			// fmt.Println("Outlier found:", point)
			outliers = append(outliers, point)
		}

	}

	return outliers

}

func load(fileName string, fields int, points []Point) []Point {

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
				points[j].x = parsedVal
				continue
			} else if i == 1 {
				points[j].y = parsedVal
				continue
			}
		}
	}

	return points
}
