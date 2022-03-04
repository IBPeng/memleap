#include "XSbench_header.cuh"

#ifdef CUPTI_PROFILING
#include "cupti_um.hpp"
#endif

int main( int argc, char* argv[] )
{
#ifdef CUPTI_PROFILING
  uvm_profiling_init();
#endif
	// =====================================================================
	// Initialization & Command Line Read-In
	// =====================================================================
	int version = 19;
	int mype = 0;
	double omp_start, omp_end;
	int nprocs = 1;
	unsigned long long verification;

	// Process CLI Fields -- store in "Inputs" structure
	Inputs in = read_CLI( argc, argv );

	// Print-out of Input Summary
	if( mype == 0 )
		print_inputs( in, nprocs, version );

	// =====================================================================
	// Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
	// This is not reflective of a real Monte Carlo simulation workload,
	// therefore, do not profile this region!
	// =====================================================================
	
	SimulationData SD;

	// If read from file mode is selected, skip initialization and load
	// all simulation data structures from file instead
	if( in.binary_mode == READ )
		SD = binary_read(in);
	else
		SD = grid_init_do_not_profile( in, mype );

	// If writing from file mode is selected, write all simulation data
	// structures to file
	if( in.binary_mode == WRITE && mype == 0 )
		binary_write(in, SD);

	// Move data to GPU
	SimulationData GSD = move_simulation_data_to_device( in, mype, SD );

#ifdef CUPTI_PROFILING
	um_dataobj_map.emplace_back( GSD.nuclide_grid, GSD.length_nuclide_grid * sizeof(NuclideGridPoint), "nuclide_grid");
	um_dataobj_map.emplace_back( GSD.unionized_energy_array, GSD.length_unionized_energy_array * sizeof(double), "unionized_energy_array");
	um_dataobj_map.emplace_back( GSD.index_grid, GSD.length_index_grid * sizeof(int), "index_grid");
	um_dataobj_map.emplace_back( GSD.mats, GSD.length_mats * sizeof(int), "mats");
	um_dataobj_map.emplace_back( GSD.concs, GSD.length_concs * sizeof(double), "concs");
	um_dataobj_map.emplace_back( GSD.num_nucs, GSD.length_num_nucs * sizeof(int), "num_nucs");
#endif

	// =====================================================================
	// Cross Section (XS) Parallel Lookup Simulation
	// This is the section that should be profiled, as it reflects a 
	// realistic continuous energy Monte Carlo macroscopic cross section
	// lookup kernel.
	// =====================================================================
	if( mype == 0 )
	{
		printf("\n");
		border_print();
		center_print("SIMULATION", 79);
		border_print();
	}

#ifdef CUPTI_PROFILING
  uvm_profiling_start();
#endif
	// Start Simulation Timer
	omp_start = get_time();

	// Run simulation
	if( in.simulation_method == EVENT_BASED )
	{
		if( in.kernel_id == 0 )
			verification = run_event_based_simulation_baseline(in, GSD, mype);
		else if( in.kernel_id == 1 )
			verification = run_event_based_simulation_optimization_1(in, GSD, mype);
		else if( in.kernel_id == 2 )
			verification = run_event_based_simulation_optimization_2(in, GSD, mype);
		else if( in.kernel_id == 3 )
			verification = run_event_based_simulation_optimization_3(in, GSD, mype);
		else if( in.kernel_id == 4 )
			verification = run_event_based_simulation_optimization_4(in, GSD, mype);
		else if( in.kernel_id == 5 )
			verification = run_event_based_simulation_optimization_5(in, GSD, mype);
		else if( in.kernel_id == 6 )
			verification = run_event_based_simulation_optimization_6(in, GSD, mype);
		else
		{
			printf("Error: No kernel ID %d found!\n", in.kernel_id);
			exit(1);
		}
	}
	else
	{
		printf("History-based simulation not implemented in CUDA code. Instead,\nuse the event-based method with \"-m event\" argument.\n");
		exit(1);
	}

#ifdef CUPTI_PROFILING
  uvm_profiling_stop();
#endif

	if( mype == 0)	
	{	
		printf("\n" );
		printf("Simulation complete.\n" );
	}

	// End Simulation Timer
	omp_end = get_time();

	// Final Hash Step
	verification = verification % 999983;

	// Print / Save Results and Exit
	int is_invalid_result = print_results( in, mype, omp_end-omp_start, nprocs, verification );

	return 0;
	return is_invalid_result;
}
