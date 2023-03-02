// Generated with SOL v0.4.2.1
#include "libFlexMLP.h"
#include <stdlib.h>
int main(int argc, char** argv) {
	// Inputs
	double* F6_Output = malloc(sizeof(double) * 6);

	// Outputs
	double* F15_Output = malloc(sizeof(double) * 6);

	// Launch
	forward(0, F6_Output, F15_Output);

	// Wait for Results
	
	// Free Inputs
	free(F6_Output);

	// Free Outputs
	free(F15_Output);

	return 0;
}
